from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional, Tuple
import torch
import faiss
import os
import warnings
import requests
import json
import io
import soundfile as sf
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
from PIL import Image

# Try to import kokoro, but don't fail if it's not available
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

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
text_embedding_dim = 384
text_vector_store = faiss.IndexFlatL2(text_embedding_dim)
text_metadata: List[dict] = []

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

def detect_visual_query_qwen(query: str) -> Tuple[bool, float]:
    """
    Use Qwen to intelligently detect if a query is visual in nature.
    Returns (is_visual, confidence_score)
    """
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes queries to determine if they would benefit from visual/image analysis. Respond with ONLY a JSON object in this format: {\"is_visual\": true/false, \"confidence\": 0.0-1.0, \"reasoning\": \"brief explanation\"}"},
            {"role": "user", "content": f"Analyze this query and determine if it would benefit from image analysis: '{query}'"}
        ]
        response = qwen_model.create_chat_completion(messages=messages)
        result = response['choices'][0]['message']['content'].strip()

        # Try to parse JSON response
        import json
        try:
            parsed = json.loads(result)
            return parsed.get("is_visual", False), parsed.get("confidence", 0.0)
        except json.JSONDecodeError:
            # Fallback: simple keyword detection if JSON parsing fails
            visual_keywords = ['image', 'picture', 'photo', 'visual', 'show me', 'describe', 'looks like', 'see', 'view', 'diagram', 'chart', 'graph']
            query_lower = query.lower()
            matches = [kw for kw in visual_keywords if kw in query_lower]
            confidence = min(0.8, len(matches) * 0.2)
            return len(matches) > 0, confidence

    except Exception as e:
        print(f"Error in visual query detection: {e}")
        # Fallback to simple keyword detection
        visual_keywords = ['image', 'picture', 'photo', 'visual', 'show me', 'describe', 'looks like', 'see', 'view']
        query_lower = query.lower()
        matches = [kw for kw in visual_keywords if kw in query_lower]
        return len(matches) > 0, min(0.7, len(matches) * 0.2)

def retrieve_context(query: str, top_k: int = 3) -> List[dict]:
    """
    Simple but effective retrieval using FAISS similarity search.
    Enhanced with intelligent visual query detection using Qwen.
    """
    if text_vector_store.ntotal == 0:
        return []

    # Use Qwen to detect visual queries intelligently
    is_visual, visual_confidence = detect_visual_query_qwen(query)
    print(f"Query analysis: visual={is_visual}, confidence={visual_confidence:.2f}")

    # Get basic similarity results
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    distances, indices = text_vector_store.search(query_embedding.astype(np.float32), min(top_k, text_vector_store.ntotal))

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(text_metadata):
            source_path = text_metadata[idx]["source_path"]
            distance = distances[0][i]

            # Convert distance to similarity score (lower distance = higher similarity)
            similarity = 1.0 / (1.0 + distance)

            # Boost visual results if this is detected as a visual query
            if is_visual and source_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                similarity *= (1.0 + visual_confidence)
                print(f"Boosted image result {os.path.basename(source_path)} by {visual_confidence:.2f}x")

            results.append({
                "source": source_path,
                "text": text_metadata[idx]["text"],
                "score": similarity,
                "type": "image" if source_path.lower().endswith(('.png', '.jpg', '.jpeg')) else "text"
            })

    # Sort by final score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# --- API Schemas ---
class ChatRequest(BaseModel):
    query: str

# --- API Endpoints ---
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    global text_vector_store, text_metadata
    if not files:
        return {"status": "error", "message": "Please upload files first."}

    # Reset databases
    text_vector_store.reset()
    text_metadata = []

    all_text_chunks: List[dict] = []
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
                # --- Image Processing: Generate description with smolvlm ---
                try:
                    print(f"Generating description for {file_name}...")
                    with open(saved_path, 'rb') as img_file:
                        image_files = {'file': img_file}
                        res = requests.post("http://127.0.0.1:5002/describe", files=image_files)

                    res.raise_for_status()
                    response_json = res.json()
                    description = response_json.get("description", "")

                    if description:
                        print(f"Generated description for {file_name}: {len(description)} chars")
                        for chunk in iterative_chunking(description):
                            all_text_chunks.append({"text": chunk, "source_path": saved_path})
                    else:
                        print(f"Warning: Empty description for {file_name}")

                except Exception as e:
                    print(f"Error generating description for {file_name}: {e}")

                processed_files += 1
                continue

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

    # Batch embed and add to store
    if all_text_chunks:
        texts = [c['text'] for c in all_text_chunks]
        text_embeddings = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True).cpu().numpy()
        text_vector_store.add(text_embeddings)
        text_metadata.extend(all_text_chunks)
        print(f"Added {len(all_text_chunks)} chunks from {processed_files} files.")

    if not all_text_chunks:
        return {"status": "error", "message": "No valid content extracted."}

    return {
        "status": "ok",
        "message": f"Processed {processed_files} files.",
        "chunks": len(text_metadata)
    }

@app.post("/chat")
async def chat(payload: ChatRequest):
    contexts = retrieve_context(payload.query)

    if not contexts:
        return {"answer": "Knowledge base is empty. Please ingest files first."}

    try:
        # Check if we have image sources
        image_contexts = [ctx for ctx in contexts if ctx['source'].lower().endswith(('.png', '.jpg', '.jpeg'))]

        if image_contexts:
            # Use multimodal pipeline with SmolVLM
            best_image_context = image_contexts[0]
            source_path = best_image_context["source"]

            print(f"Using multimodal pipeline with {os.path.basename(source_path)}")

            with open(source_path, 'rb') as f:
                files = {'file': f}
                data = {'query': payload.query}
                res = requests.post(SMOLVLM_ANSWER_API_URL, files=files, data=data)

            res.raise_for_status()
            response_json = res.json()
            smolvlm_answer = response_json.get("answer", "Sorry, I couldn't analyze the image.")

            # Create response with citations - fix source mapping
            unique_sources = []
            for ctx in contexts:
                source_path = ctx["source"]
                # Ensure we get the actual filename, not just the path
                clean_filename = os.path.basename(source_path)
                if clean_filename not in unique_sources:
                    unique_sources.append(clean_filename)

            sources_html = "\n".join([
                f'<div class="source-item"><span class="source-key">[{i+1}]</span><span class="source-name">{src}</span></div>'
                for i, src in enumerate(unique_sources)
            ])

            messages = [
                {"role": "system", "content": """
                You are a helpful AI assistant with access to multimodal content. Use the image analysis and context to provide a comprehensive answer. Format your entire response as HTML. Include inline citations in the format [1], [2], etc. at the end of sentences where the information came from the context. Place the citation number before the period at the end of the sentence.

                After your main response, include a separator: --%Sources%--

                After the separator, provide a "Sources" section using this exact HTML structure:
                <div class="sources-section">
                  <h3>Sources</h3>
                  <div class="source-item">
                    <span class="source-key">[1]</span>
                    <span class="source-name">filename.jpg</span>
                  </div>
                  <div class="source-item">
                    <span class="source-key">[2]</span>
                    <span class="source-name">document.pdf</span>
                  </div>
                </div>

                Your entire response should follow this structure:
                MAIN RESPONSE CONTENT WITH [1] STYLE CITATIONS
                --%Sources%--
                <div class="sources-section">
                  <h3>Sources</h3>
                  <div class="source-item">
                    <span class="source-key">[1]</span>
                    <span class="source-name">source_name</span>
                  </div>
                </div>

                Use proper HTML formatting for all content - paragraphs, lists, bold, italics, etc. Do not output raw markdown, only HTML.
                The citation numbers in the sources section should have brackets around them.
                """},
                {"role": "user", "content": f"USER'S QUESTION: {payload.query}\n\nIMAGE ANALYSIS: {smolvlm_answer}\n\nRELEVANT CONTEXT:\n{' '.join([ctx['text'] for ctx in contexts[:2]])}\n\n--%Sources%--\n<div class=\"sources-section\">\n<h3>Sources</h3>\n{sources_html}\n</div>"}
            ]
            response = qwen_model.create_chat_completion(messages=messages)
            final_answer = response['choices'][0]['message']['content'].strip()

            return {"answer": final_answer}

        else:
            # Text-only pipeline
            print("Using text-only pipeline")
            context_str = "\n".join([ctx['text'] for ctx in contexts])

            # Fix source mapping for text-only pipeline
            unique_sources = []
            for ctx in contexts:
                source_path = ctx["source"]
                clean_filename = os.path.basename(source_path)
                if clean_filename not in unique_sources:
                    unique_sources.append(clean_filename)

            sources_html = "\n".join([
                f'<div class="source-item"><span class="source-key">[{i+1}]</span><span class="source-name">{src}</span></div>'
                for i, src in enumerate(unique_sources)
            ])

            messages = [
                {"role": "system", "content": """
                You are a helpful AI assistant. Answer ONLY from the provided context. Format your entire response as HTML. Include inline citations in the format [1], [2], etc. at the end of sentences where the information came from the context. Place the citation number before the period at the end of the sentence.

                After your main response, include a separator: --%Sources%--

                After the separator, provide a "Sources" section using this exact HTML structure:
                <div class="sources-section">
                  <h3>Sources</h3>
                  <div class="source-item">
                    <span class="source-key">[1]</span>
                    <span class="source-name">document.pdf</span>
                  </div>
                  <div class="source-item">
                    <span class="source-key">[2]</span>
                    <span class="source-name">document.docx</span>
                  </div>
                </div>

                Your entire response should follow this structure:
                MAIN RESPONSE CONTENT WITH [1] STYLE CITATIONS
                --%Sources%--
                <div class="sources-section">
                  <h3>Sources</h3>
                  <div class="source-item">
                    <span class="source-key">[1]</span>
                    <span class="source-name">source_name</span>
                  </div>
                </div>

                Use proper HTML formatting for all content - paragraphs, lists, bold, italics, etc. Do not output raw markdown, only HTML.
                The citation numbers in the sources section should have brackets around them.
                """},
                {"role": "user", "content": f"CONTEXT from multiple sources:\n{context_str}\n\nUSER'S QUESTION: {payload.query}\n\n--%Sources%--\n<div class=\"sources-section\">\n<h3>Sources</h3>\n{sources_html}\n</div>"}
            ]
            response = qwen_model.create_chat_completion(messages=messages)
            final_answer = response['choices'][0]['message']['content'].strip()

            return {"answer": final_answer}

    except requests.exceptions.ConnectionError:
        return {"answer": "CONNECTION ERROR: Could not reach multimodal service"}
    except Exception as e:
        return {"answer": f"Error: {e}"}

@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, "_temp_chat_audio")
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        segments, info = whisper_model.transcribe(temp_path, beam_size=5)
        transcribed_query = "".join(seg.text for seg in segments).strip()

        if not transcribed_query:
            return {"answer": "Could not understand the audio. Please try again."}

        chat_payload = ChatRequest(query=transcribed_query)
        return await chat(chat_payload)

    except Exception as e:
        return {"answer": f"Error processing audio: {e}"}
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

@app.post('/generate_audio')
async def generate_audio(text: str = Form(...), voice: str = Form(default='af_heart')):
    if not KOKORO_AVAILABLE:
        return {"error": "Kokoro TTS is not available"}

    try:
        pipeline = KPipeline(lang_code='a')
        generator = pipeline(text, voice=voice)
        audio_data = None

        for i, (gs, ps, audio) in enumerate(generator):
            audio_data = audio
            break

        if audio_data is None:
            return {"error": "Failed to generate audio"}

        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 24000, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()

        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav",
                               headers={"Content-Disposition": "inline"})
    except Exception as e:
        return {"error": str(e)}

@app.post('/reset')
async def reset():
    global text_vector_store, text_metadata
    text_vector_store.reset()
    text_metadata = []
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
