# app.py

import gradio as gr
import torch
import faiss
import numpy as np
import pypdf
import os
import requests
import warnings
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import docx
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_DIR = "uploads"
WHISPER_API_URL = "http://127.0.0.1:5001/transcribe"
SMOLVLM_DESCRIBE_API_URL = "http://127.0.0.1:5002/describe"
SMOLVLM_ANSWER_API_URL = "http://127.0.0.1:5002/answer"

print(f"Main App: Using device '{DEVICE}'.")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Load Models (Orchestrator and Embedding) ---
print("Main App: Loading SentenceTransformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

print("Main App: Loading GGUF Qwen orchestrator...")
qwen_model_id = "enacimie/Qwen3-0.6B-Q4_K_M-GGUF"
model_path = hf_hub_download(repo_id=qwen_model_id, filename="qwen3-0.6b-q4_k_m.gguf")
qwen_model = Llama(model_path=model_path, n_gpu_layers=-1, verbose=True, n_ctx=2048)
print("Main App: All local models loaded.")

# --- Setup Vector Database (FAISS) ---
embedding_dim = embedding_model.get_sentence_embedding_dimension()
vector_store = faiss.IndexFlatL2(embedding_dim)
chunk_metadata = [] # Stores {"text": chunk, "source_path": file_path}
print("Main App: In-memory Vector DB initialized.")

# --- Text Chunking ---
def iterative_chunking(text, chunk_size=400):
    chunks = []
    while len(text) > chunk_size:
        split_point = -1
        # Try to split by paragraph
        if "\n\n" in text[:chunk_size]:
            split_point = text.rfind("\n\n", 0, chunk_size)
        # If no paragraph, try to split by sentence
        if split_point == -1 and ". " in text[:chunk_size]:
            split_point = text.rfind(". ", 0, chunk_size)
        # If no sentence, split by word
        if split_point == -1 and " " in text[:chunk_size]:
            split_point = text.rfind(" ", 0, chunk_size)
        # If no space, force split
        if split_point == -1 or split_point == 0:
            split_point = chunk_size

        chunks.append(text[:split_point])
        text = text[split_point:]
    
    if text:
        chunks.append(text)
        
    return chunks

# --- Ingestion Pipeline ---
def process_and_embed_files(file_objects):
    global vector_store, chunk_metadata
    if not file_objects:
        return "⚠️ Please upload files first."
        
    # Clear previous data
    vector_store.reset()
    chunk_metadata = []
    all_chunks = []
    processed_files = 0

    for file_obj in file_objects:
        file_path = file_obj.name
        file_name = os.path.basename(file_path)
        print(f"Main App: Processing file: {file_name}")
        
        # Save the uploaded file to the UPLOAD_DIR
        saved_path = os.path.join(UPLOAD_DIR, file_name)
        with open(saved_path, "wb") as f_out, open(file_path, "rb") as f_in:
            f_out.write(f_in.read())

        try:
            text = ""
            if saved_path.lower().endswith((".png", ".jpg", ".jpeg")):
                with open(saved_path, 'rb') as f:
                    files = {'file': (file_name, f, 'image/jpeg')}
                    response = requests.post(SMOLVLM_DESCRIBE_API_URL, files=files)
                    response.raise_for_status()
                    text = response.json().get("description", "")
            
            elif saved_path.lower().endswith((".mp3", ".wav", ".m4a")):
                with open(saved_path, 'rb') as f:
                    files = {'file': (file_name, f, 'audio/mpeg')}
                    response = requests.post(WHISPER_API_URL, files=files)
                    response.raise_for_status()
                    text = response.json().get("transcription", "")

            elif saved_path.lower().endswith('.pdf'):
                reader = pypdf.PdfReader(saved_path)
                text = "\n".join([page.extract_text() for page in reader.pages])

            elif saved_path.lower().endswith('.docx'):
                doc = docx.Document(saved_path)
                text = "\n".join([para.text for para in doc.paragraphs])

            elif saved_path.lower().endswith('.csv'):
                df = pd.read_csv(saved_path)
                text = df.to_string()

            elif saved_path.lower().endswith('.txt'):
                with open(saved_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                print(f"Unsupported file type: {file_name}")
                continue

            if text:
                chunks = iterative_chunking(text)
                for chunk in chunks:
                    all_chunks.append({"text": chunk, "source_path": saved_path})
                processed_files += 1

        except Exception as e:
            print(f"Main App: Error processing {file_name}: {e}")
            return f"Error processing {file_name}: {e}"
            
    if all_chunks:
        texts_to_embed = [item['text'] for item in all_chunks]
        embeddings = embedding_model.encode(texts_to_embed, convert_to_tensor=True, show_progress_bar=True)
        vector_store.add(embeddings.cpu().numpy())
        chunk_metadata.extend(all_chunks)
        return f"✅ Successfully processed {processed_files} files. Knowledge base has {vector_store.ntotal} chunks. Ready for questions!"
    else:
        return "⚠️ No valid content could be extracted from the files."

# --- RAG Query Logic ---
def retrieve_context(query, top_k=1):
    if vector_store.ntotal == 0:
        return None, None
        
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    distances, indices = vector_store.search(query_embedding, k=min(top_k, vector_store.ntotal))
    
    retrieved_chunk_meta = chunk_metadata[indices[0][0]]
    source_path = retrieved_chunk_meta["source_path"]
    
    if source_path.lower().endswith((".png", ".jpg", ".jpeg")):
        print(f"Main App: Best context is IMAGE: '{os.path.basename(source_path)}'")
        return source_path, "image"
    else:
        print(f"Main App: Best context is TEXT from '{os.path.basename(source_path)}'")
        return retrieved_chunk_meta, "text"

def chat_interface(user_query, chat_history):
    context, context_type = retrieve_context(user_query)

    if context_type is None:
        response = "I don't have any information in my knowledge base. Please upload and process files first."
        chat_history.append((user_query, response))
        return "", chat_history

    final_response = ""
    try:
        if context_type == "text":
            print("Main App: Routing to Qwen2 for text-based answer...")
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Answer the user's question based ONLY on the provided context."},
                {"role": "user", "content": f"CONTEXT from {os.path.basename(context['source_path'])}:\n{context['text']}\n\nUSER'S QUESTION: {user_query}"}
            ]
            response = qwen_model.create_chat_completion(messages=messages)
            raw_response = response['choices'][0]['message']['content']
            if "</think>" in raw_response:
                final_response = raw_response.split("</think>")[-1].strip()
            else:
                final_response = raw_response.strip()
            final_response += f"\n\n**Source:**\n- {os.path.basename(context['source_path'])}"

        elif context_type == "image":
            print("Main App: Routing to SmolVLM for image-based answer...")
            image_path = context
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                data = {'query': user_query}
                response = requests.post(SMOLVLM_ANSWER_API_URL, files=files, data=data)
                response.raise_for_status()
                final_response = response.json().get("answer", "Sorry, I couldn't analyze the image.")
                final_response += f"\n\n**Source:**\n- {os.path.basename(image_path)}"

    except requests.exceptions.ConnectionError as e:
        final_response = f"CONNECTION ERROR: Could not connect to a backend service. Please ensure all services are running. Details: {e}"
    except Exception as e:
        final_response = f"An unexpected error occurred: {e}"

    chat_history.append((user_query, final_response))
    return "", chat_history

def reset_state():
    return [], "", None, ""

# --- Build Gradio App ---
with gr.Blocks(theme=gr.themes.Soft(), title="Multi-Modal RAG System") as demo:
    gr.Markdown("# 🤖 Full Multi-Modal RAG System")
    gr.Markdown("Upload images, audio, PDFs, or text files. Click 'Process Files' to build the knowledge base, then ask questions.")

    with gr.Row():
        with gr.Column(scale=1):
            file_uploader = gr.Files(label="Upload Files (Images, Audio, PDF, TXT, DOCX, CSV)", file_count="multiple")
            btn_process = gr.Button("⚙️ Process Files", variant="primary")
            process_status = gr.Textbox(label="Processing Status", interactive=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=600, bubble_full_width=False, avatar_images=("user.png", "bot.png"))
            with gr.Row():
                text_input = gr.Textbox(placeholder="Ask a question about your files...", label="Your Question", scale=4)
                submit_btn = gr.Button("➢", variant="primary", scale=1)
                reset_btn = gr.Button("🗑️", variant="secondary", scale=1)


    # --- Event Handlers ---
    btn_process.click(
        fn=process_and_embed_files,
        inputs=file_uploader,
        outputs=process_status
    )

    submit_btn.click(
        fn=chat_interface,
        inputs=[text_input, chatbot],
        outputs=[text_input, chatbot]
    )

    text_input.submit(
        fn=chat_interface,
        inputs=[text_input, chatbot],
        outputs=[text_input, chatbot]
    )

    reset_btn.click(
        fn=reset_state,
        inputs=[],
        outputs=[chatbot, text_input, file_uploader, process_status]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)