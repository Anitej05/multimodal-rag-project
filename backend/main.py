# backend/main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Tuple
import torch
import faiss
import os
import warnings
import requests
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import pypdf
import docx
import pandas as pd
from pydantic import BaseModel
from faster_whisper import WhisperModel
import subprocess
import sys
import atexit
import numpy as np
import easyocr

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
SMOLVLM_ANSWER_API_URL = "http://127.0.0.1:5002/answer"

os.makedirs(UPLOAD_DIR, exist_ok=True)

print(f"Backend: Using device '{DEVICE}'.")

# --- Initialize FastAPI App ---
app = FastAPI(title="Multimodal RAG Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Models ---
print("Backend: Initializing EasyOCR reader...")
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

print("Backend: Loading text embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

print("Backend: Loading image embedding model (clip-ViT-B-32)...")
clip_model = SentenceTransformer('clip-ViT-B-32', device=DEVICE)

print("Backend: Loading GGUF Qwen orchestrator...")
qwen_model_id = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
model_path = hf_hub_download(repo_id=qwen_model_id, filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
qwen_model = Llama(model_path=model_path, n_gpu_layers=-1, verbose=True, n_ctx=2048)

print("Backend: Loading Whisper model...")
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
whisper_model = WhisperModel("base", device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
print("Backend: All models loaded.")


# --- Vector Stores (FAISS) ---
print("Backend: Initializing vector stores...")
# Text store
# Directly specify the known dimension for the model.
text_embedding_dim = 384 
text_vector_store = faiss.IndexFlatL2(text_embedding_dim)
text_metadata: List[dict] = []  # {"text": str, "source_path": str}

# Image store
# The get_sentence_embedding_dimension() method returns None for CLIP models,
# so we specify the known dimension (512) directly.
image_embedding_dim = 512
image_vector_store = faiss.IndexFlatL2(image_embedding_dim)
image_metadata: List[dict] = [] # {"source_path": str}

print("Backend: In-memory Vector DBs initialized.")
# --- Utilities ---
def iterative_chunking(text: str, chunk_size: int = 400) -> List[str]:
    chunks: List[str] = []
    while len(text) > chunk_size:
        split_point = -1
        if "\n\n" in text[:chunk_size]:
            split_point = text.rfind("\n\n", 0, chunk_size)
        if split_point == -1 and ". " in text[:chunk_size]:
            split_point = text.rfind(". ", 0, chunk_size)
        if split_point == -1 and " " in text[:chunk_size]:
            split_point = text.rfind(" ", 0, chunk_size)
        if split_point == -1 or split_point == 0:
            split_point = chunk_size
        chunks.append(text[:split_point])
        text = text[split_point:]
    if text:
        chunks.append(text)
    return chunks

def retrieve_multimodal_context(query: str) -> Tuple[Optional[dict], Optional[str]]:
    # 1. Search text
    text_top_k = 1
    text_score = -1.0
    text_result = None
    if text_vector_store.ntotal > 0:
        query_text_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        distances, indices = text_vector_store.search(query_text_embedding, k=min(text_top_k, text_vector_store.ntotal))
        text_score = 1.0 / (1.0 + distances[0][0]) # Normalize distance to similarity
        text_result = text_metadata[indices[0][0]]

    # 2. Search images
    image_top_k = 1
    image_score = -1.0
    image_result = None
    if image_vector_store.ntotal > 0:
        query_image_embedding = clip_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        distances, indices = image_vector_store.search(query_image_embedding, k=min(image_top_k, image_vector_store.ntotal))
        image_score = 1.0 / (1.0 + distances[0][0]) # Normalize distance to similarity
        image_result = image_metadata[indices[0][0]]

    print(f"Text search score: {text_score:.4f}, Image search score: {image_score:.4f}")

    # 3. Compare and return the best result
    if text_score == -1 and image_score == -1:
        return None, None

    if image_score > text_score:
        print(f"Winner: Image ({os.path.basename(image_result['source_path'])})")
        return image_result, "image"
    else:
        print(f"Winner: Text from {os.path.basename(text_result['source_path'])}")
        return text_result, "text"

# --- API Schemas ---
class ChatRequest(BaseModel):
    query: str


# --- API Endpoints ---
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    global text_vector_store, text_metadata, image_vector_store, image_metadata
    if not files:
        return {"status": "error", "message": "Please upload files first."}

    # Reset databases
    text_vector_store.reset()
    text_metadata = []
    image_vector_store.reset()
    image_metadata = []

    all_text_chunks: List[dict] = []
    all_image_embeddings: List[np.ndarray] = []
    all_image_metadata: List[dict] = []
    processed_files = 0

    for file_obj in files:
        file_name = file_obj.filename
        saved_path = os.path.join(UPLOAD_DIR, file_name)
        with open(saved_path, "wb") as f_out:
            f_out.write(await file_obj.read())

        try:
            text = ""
            lower = saved_path.lower()

            if lower.endswith((".png", ".jpg", ".jpeg")):
                # --- Image Processing: OCR + CLIP ---
                img = Image.open(saved_path).convert("RGB")
                
                # 1. OCR
                try:
                    # The 'paragraph=True' option groups nearby text into paragraphs.
                    ocr_result = ocr_reader.readtext(saved_path, detail=0, paragraph=True)
                    if ocr_result:
                        ocr_text = "\n".join(ocr_result)
                        print(f"Extracted OCR text from {file_name}")
                        for chunk in iterative_chunking(ocr_text):
                            all_text_chunks.append({"text": chunk, "source_path": saved_path})
                except Exception as ocr_error:
                    print(f"Could not perform OCR on {file_name} with EasyOCR: {ocr_error}")

                # 2. CLIP Embedding
                img_embedding = clip_model.encode(img, convert_to_tensor=True)
                all_image_embeddings.append(img_embedding.cpu().numpy())
                all_image_metadata.append({"source_path": saved_path})
                
                processed_files += 1
                continue # Move to next file after image processing

            elif lower.endswith((".mp3", ".wav", ".m4a")):
                segments, info = whisper_model.transcribe(saved_path, beam_size=5)
                text = "".join(seg.text for seg in segments)
            elif lower.endswith('.pdf'):
                reader = pypdf.PdfReader(saved_path)
                text = "\n".join([page.extract_text() for page in reader.pages])
            elif lower.endswith('.docx'):
                doc = docx.Document(saved_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            elif lower.endswith('.csv'):
                df = pd.read_csv(saved_path)
                text = df.to_string()
            elif lower.endswith('.txt'):
                with open(saved_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                continue

            if text:
                for chunk in iterative_chunking(text):
                    all_text_chunks.append({"text": chunk, "source_path": saved_path})
                processed_files += 1
        except Exception as e:
            return {"status": "error", "message": f"Error processing {file_name}: {e}"}

    # Batch embed and add to stores
    if all_text_chunks:
        texts = [c['text'] for c in all_text_chunks]
        text_embeddings = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        text_vector_store.add(text_embeddings.cpu().numpy())
        text_metadata.extend(all_text_chunks)
        print(f"Added {len(all_text_chunks)} text chunks to the vector store.")

    if all_image_embeddings:
        image_embeddings_np = np.vstack(all_image_embeddings)
        image_vector_store.add(image_embeddings_np)
        image_metadata.extend(all_image_metadata)
        print(f"Added {len(all_image_embeddings)} image embeddings to the vector store.")

    if not all_text_chunks and not all_image_embeddings:
        return {"status": "error", "message": "No valid content extracted."}

    return {
        "status": "ok",
        "message": f"Processed {processed_files} files.",
        "text_chunks": len(text_metadata),
        "images": len(image_metadata)
    }

@app.post("/chat")
async def chat(payload: ChatRequest):
    context, context_type = retrieve_multimodal_context(payload.query)

    if context is None:
        return {"answer": "Knowledge base is empty. Please ingest files first."}

    try:
        source_path = context["source_path"]
        is_image_source = source_path.lower().endswith((".png", ".jpg", ".jpeg"))

        # IMAGE PIPELINE (direct image match or OCR text match)
        if is_image_source:
            print("Activating Image Pipeline...")
            # Step 1: Use SmolVLM as a tool to analyze the image
            print(f"  [Tool Call] Analyzing {os.path.basename(source_path)} with SmolVLM...")
            with open(source_path, 'rb') as f:
                files = {'file': (os.path.basename(source_path), f, 'image/jpeg')}
                data = {'query': payload.query}
                res = requests.post(SMOLVLM_ANSWER_API_URL, files=files, data=data)
                res.raise_for_status()
                smolvlm_answer = res.json().get("answer", "Sorry, I couldn't analyze the image.")
            print(f"  [Tool Result] SmolVLM says: \"{smolvlm_answer[:100]}...\"")

            # Step 2: Feed result back to Qwen for final answer
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. You will be given a user's question and the output from an image analysis tool. Use the tool's output to formulate a final, comprehensive answer to the user's question."},
                {"role": "user", "content": f"USER'S QUESTION: {payload.query}\n\nIMAGE ANALYSIS TOOL OUTPUT:\n{smolvlm_answer}"}
            ]
            response = qwen_model.create_chat_completion(messages=messages)
            final_answer = response['choices'][0]['message']['content'].strip()
            
            return {
                "answer": final_answer,
                "source": os.path.basename(source_path)
            }

        # TEXT PIPELINE (standard docs)
        else:
            print("Activating Text Pipeline...")
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Answer ONLY from the provided context."},
                {"role": "user", "content": f"CONTEXT from {os.path.basename(source_path)}\n{context['text']}\n\nUSER'S QUESTION: {payload.query}"}
            ]
            response = qwen_model.create_chat_completion(messages=messages)
            final_answer = response['choices'][0]['message']['content'].strip()
            return {
                "answer": final_answer,
                "source": os.path.basename(source_path)
            }

    except requests.exceptions.ConnectionError as e:
        return {"answer": f"CONNECTION ERROR: Could not reach multimodal service: {e}"}
    except Exception as e:
        return {"answer": f"Unexpected error: {e}"}

@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, "_temp_chat_audio")
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 1. Transcribe audio to text
        segments, info = whisper_model.transcribe(temp_path, beam_size=5)
        transcribed_query = "".join(seg.text for seg in segments).strip()
        
        if not transcribed_query:
            return {"answer": "Could not understand the audio. Please try again."}
            
        print(f'Transcribed audio query: "{transcribed_query}"')
        
        # 2. Create a request payload for the existing chat endpoint
        chat_payload = ChatRequest(query=transcribed_query)
        
        # 3. Call the existing chat logic and return its result
        return await chat(chat_payload)

    except Exception as e:
        return {"answer": f"An error occurred while processing the audio chat: {e}"}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post('/transcribe')
async def transcribe(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, "_temp_audio_input")
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        segments, info = whisper_model.transcribe(temp_path, beam_size=5)
        transcription = "".join(seg.text for seg in segments)
        return {"transcription": transcription}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post('/reset')
async def reset():
    global text_vector_store, text_metadata, image_vector_store, image_metadata
    text_vector_store.reset()
    text_metadata = []
    image_vector_store.reset()
    image_metadata = []
    return {"status": "ok", "message": "All knowledge bases cleared."}


if __name__ == '__main__':
    import uvicorn
    processes = []
    services_dir = os.path.join(os.path.dirname(__file__), "services")
    service_files = [f for f in os.listdir(services_dir) if f.endswith("_service.py")]

    for service_file in service_files:
        service_path = os.path.join(services_dir, service_file)
        print(f"Starting service: {service_file}")
        process = subprocess.Popen([sys.executable, service_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(process)

    def cleanup():
        print("Terminating background services...")
        for process in processes:
            process.terminate()
            process.wait()
        print("Background services terminated.")

    atexit.register(cleanup)

    uvicorn.run(app, host='0.0.0.0', port=8000)