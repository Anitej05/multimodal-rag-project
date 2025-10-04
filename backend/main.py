#!/usr/bin/env python3
"""
New main.py using LM Studio API instead of local models
Combines functionality from main.py and main_hybrid.py
Preserves all original functionality except LLM inference method
"""

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
import re
import subprocess
import sys
import atexit
import numpy as np
import easyocr
from PIL import Image
from sentence_transformers import SentenceTransformer, CrossEncoder
import pypdf
import docx
import pandas as pd
from pydantic import BaseModel

# Try to import kokoro, but don't fail if it's not available
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    KPipeline = None  # type: ignore

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
WHISPER_SERVICE_URL = "http://127.0.0.1:5001/transcribe"
LM_STUDIO_URL = "http://localhost:1234"

os.makedirs(UPLOAD_DIR, exist_ok=True)

print(f"Backend: Using device '{DEVICE}'.")
print(f"Backend: LM Studio endpoint: {LM_STUDIO_URL}")
print(f"Backend: Whisper service endpoint: {WHISPER_SERVICE_URL}")

# --- Initialize FastAPI App ---
app = FastAPI(title="Multimodal RAG Backend with LM Studio")
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

print("Backend: Loading Cross-encoder for re-ranking...")
# Try multiple cross-encoder models - fall back if first fails
try:
    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
except Exception as e:
    print(f"Failed to load primary cross-encoder: {e}")
    try:
        cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', max_length=512)
        print("Loaded fallback TinyBERT cross-encoder")
    except Exception as e2:
        print(f"Failed to load fallback cross-encoder: {e2}")
        cross_encoder_model = None

print("Backend: All models loaded.")

# --- Vector Stores (FAISS) ---
print("Backend: Initializing vector stores...")
# Text store
text_embedding_dim = 384
nlist = 100  # Number of clusters
quantizer = faiss.IndexFlatL2(text_embedding_dim)
text_vector_store = faiss.IndexIVFFlat(quantizer, text_embedding_dim, nlist)
text_metadata: List[dict] = []  # {"text": str, "source_path": str}

# Image store - REMOVED as we're using text embeddings for images now
# All content (text and image descriptions) will be stored in the text vector store

print("Backend: In-memory Vector DBs initialized.")

# --- LM Studio API Helper Functions ---
def call_lm_studio_text(messages, model="qwen3-1.7b", max_tokens=1000, temperature=0.7):
    """Call LM Studio API for text generation"""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        result = response.json()
        content = result['choices'][0]['message']['content']

        # Strip thinking tags from Qwen response
        if '<think>' in content and '</think>' in content:
            # Extract content between thinking tags and after them
            think_pattern = r'<think>.*?</think>'
            clean_content = re.sub(think_pattern, '', content, flags=re.DOTALL)
            clean_content = clean_content.strip()
            return clean_content

        return content

    except Exception as e:
        print(f"LM Studio text API error: {e}")
        raise

def call_lm_studio_vision(image_path, query, model="smolvlm2-500m-video-instruct"):
    """Call LM Studio API for vision tasks"""
    try:
        # Encode image to base64
        with open(image_path, 'rb') as img_file:
            import base64
            image_b64 = base64.b64encode(img_file.read()).decode('utf-8')

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7
        }

        response = requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        result = response.json()
        content = result['choices'][0]['message']['content']
        return content

    except Exception as e:
        print(f"LM Studio vision API error: {e}")
        raise

# --- Helper function for safe FAISS training ---
def safe_train_index(index, embeddings, min_points_ratio=0.8):
    """
    Safely train a FAISS index, handling cases where there aren't enough training points.

    Args:
        index: The FAISS index to train
        embeddings: Training embeddings
        min_points_ratio: Minimum ratio of points to nlist for training

    Returns:
        bool: True if training was successful, False otherwise
    """
    if not hasattr(index, 'is_trained') or index.is_trained:
        return True

    n_samples, n_features = embeddings.shape
    min_required = max(1, int(index.nlist * min_points_ratio))

    if n_samples < min_required:
        print(f"Warning: Not enough training points ({n_samples}) for nlist={index.nlist}. Need at least {min_required}.")
        print("Consider using fewer clusters or providing more training data.")
        return False

    try:
        index.train(embeddings)
        print(f"Successfully trained FAISS index with {n_samples} points and {index.nlist} clusters.")
        return True
    except RuntimeError as e:
        print(f"FAISS training failed: {e}")
        print("Falling back to IndexFlatL2 for text storage.")
        return False

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

def normalize_score(raw_score: float, modality: str) -> float:
    """Normalize scores across different embedding spaces and models"""
    if modality == "text_faiss":
        # MiniLM embeddings - raw scores are already similarities, but FAISS returns distances
        # Convert FAISS L2 distance to similarity (higher = better)
        return max(0.0, min(1.0, 1.0 / (1.0 + raw_score)))
    elif modality == "cross_encoder":
        # Cross-encoder returns similarity scores, but can be negative
        # Normalize to 0-1 range, treating negative scores as very low confidence
        if raw_score < 0:
            return max(0.0, (raw_score + 1.0) / 2.0)  # Maps [-1,0] to [0, 0.5]
        else:
            return min(1.0, raw_score / 2.0)  # Maps [0,2] to [0, 1] (typical cross-encoder range)
    return raw_score

def calculate_final_score(candidate: dict) -> float:
    """Calculate final combined score from multiple sources"""
    text_score = candidate.get("text_score", 0)
    cross_encoder_score = candidate.get("cross_encoder_score", 0)

    # Combine text score with cross-encoder score if available
    if text_score > 0 and cross_encoder_score > 0:
        # For items with both base text score and cross-encoder score, combine with weighted average
        combined_score = (text_score * 0.6) + (cross_encoder_score * 0.4)
        return min(combined_score, 1.0)
    elif text_score > 0:
        # For items with only text score (from embedding similarity)
        return min(text_score, 1.0)
    elif cross_encoder_score > 0:
        # For items with only cross-encoder score
        return min(cross_encoder_score, 1.0)
    else:
        # If no scores are available, return 0
        return 0.0

def detect_visual_query_qwen(query: str) -> Tuple[bool, float]:
    """
    Use Qwen via LM Studio to intelligently detect if a query is visual in nature.
    Returns (is_visual, confidence_score)
    """
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes queries to determine if they would benefit from visual/image analysis. Respond with ONLY a JSON object in this format: {\"is_visual\": true/false, \"confidence\": 0.0-1.0, \"reasoning\": \"brief explanation\"}"},
            {"role": "user", "content": f"Analyze this query and determine if it would benefit from image analysis: '{query}'"}
        ]

        response_content = call_lm_studio_text(messages, model="qwen3-1.7b", max_tokens=100, temperature=0.1)

        # Try to parse JSON response
        try:
            parsed = json.loads(response_content)
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

def retrieve_multimodal_context(query: str, top_k: int = 5) -> List[dict]:
    """
    Unified multimodal retrieval that searches across all content types simultaneously.
    Returns the most relevant content regardless of modality (text, image, audio).
    All content (text documents and image descriptions) is now in the same text embedding space.
    """
    print(f"\n=== DEBUG: retrieve_multimodal_context called with query: {query!r} ===")

    # Query-specific weighting for better multimodal balance
    # Enhanced visual keywords to better detect when user is asking about images
    visual_keywords = [
        'image', 'picture', 'photo', 'visual', 'show me', 'describe', 'mountain', 'diagram', 'chart',
        'graph', 'photo of', 'image of', 'picture of', 'visualize', 'look like', 'what is this',
        'what does', 'appear', 'looks like', 'depicts', 'illustrates', 'scene', 'object in',
        'see in', 'view', 'screenshot', 'snapshot', 'portrait', 'landscape', 'object'
    ]

    # Calculate visual query confidence based on keyword matches
    query_lower = query.lower()
    visual_matches = [keyword for keyword in visual_keywords if keyword in query_lower]
    is_visual_query = len(visual_matches) > 0

    # Increase the visual boost based on how many visual keywords were detected
    if is_visual_query:
        visual_boost = 1.5 + (len(visual_matches) * 0.2)  # Additional boost for each keyword match
        visual_boost = min(visual_boost, 3.0)  # Cap the boost to prevent overwhelming text results
    else:
        visual_boost = 1.0

    print(f"DEBUG: Visual query detected: {is_visual_query}, matched keywords: {visual_matches}, applying {visual_boost}x boost to images")

    all_candidates = []

    # 1. SEARCH TEXT CONTENT (includes text from documents and image descriptions)
    print(f"DEBUG: Text vector store size: {text_vector_store.ntotal}")
    if text_vector_store.ntotal > 0:
        query_text_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        distances, indices = text_vector_store.search(query_text_embedding, k=min(top_k * 2, text_vector_store.ntotal))

        print(f"DEBUG: Text search found {len(indices[0])} candidates")
        for i, idx in enumerate(indices[0]):
            if idx < len(text_metadata):
                source_path = text_metadata[idx]["source_path"]
                raw_distance = distances[0][i]  # This is the raw distance from FAISS
                print(f"DEBUG: Text candidate {i}: {os.path.basename(source_path)} (distance: {raw_distance:.3f})")

                # Determine if this is a text from an image (description or OCR)
                if source_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    candidate_type = "image_description"  # Now using smolvlm descriptions instead of OCR
                else:
                    candidate_type = "text"

                all_candidates.append({
                    "source": source_path,
                    "text": text_metadata[idx]["text"],
                    "text_score": normalize_score(raw_distance, "text_faiss"),  # Pass raw distance to normalize function
                    "type": candidate_type,
                    "modality": "text"
                })

    # 3. RE-RANK USING CROSS-ENCODER (for text content)
    text_candidates = [c for c in all_candidates if c.get("text", "").strip()]
    print(f"DEBUG: Found {len(text_candidates)} text candidates for cross-encoder")

    if text_candidates and cross_encoder_model is not None:
        try:
            pairs = [[query, cand["text"]] for cand in text_candidates]
            cross_encoder_scores = cross_encoder_model.predict(pairs)

            for i, candidate in enumerate(text_candidates):
                candidate["cross_encoder_score"] = normalize_score(cross_encoder_scores[i], "cross_encoder")
                print(f"DEBUG: Cross-encoder score for {os.path.basename(candidate['source'])}: {cross_encoder_scores[i]:.3f}")
        except Exception as e:
            print(f"DEBUG: Cross-encoder failed: {e}")
    else:
        print("DEBUG: Skipping cross-encoder (no model or no text candidates)")

    # 4. CALCULATE FINAL SCORES
    for candidate in all_candidates:
        candidate["final_score"] = calculate_final_score(candidate)

        # Apply query-specific boost for visual queries
        if is_visual_query and candidate.get("text_score", 0) > 0:
            candidate["final_score"] *= visual_boost
            print(f"DEBUG: Applied {visual_boost}x visual boost to {os.path.basename(candidate['source'])}")

        print(f"DEBUG: Final score for {os.path.basename(candidate['source'])}: {candidate['final_score']:.3f}")

    # 5. SORT BY FINAL SCORE (unified ranking across all modalities)
    all_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # 6. RETURN TOP RESULTS
    top_results = all_candidates[:3]
    print(f"DEBUG: Top 3 results:")
    for i, result in enumerate(top_results, 1):
        print(f"  {i}. {os.path.basename(result['source'])} (score: {result['final_score']:.3f}, type: {result['type']})")

    print("=== END DEBUG: retrieve_multimodal_context ===\n")
    return top_results

# --- API Schemas ---
class ChatRequest(BaseModel):
    query: str

# --- API Endpoints ---
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...), reset_db: str = Form(default="true")):
    global text_vector_store, text_metadata
    if not files:
        return {"status": "error", "message": "Please upload files first."}

    if reset_db.lower() == "true":
        # Reset the vector stores
        if hasattr(text_vector_store, 'reset'):
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
                # --- Image Processing: Generate description with LM Studio vision model ---
                img = Image.open(saved_path).convert("RGB")

                # Use LM Studio vision model to generate detailed description
                try:
                    print(f"DEBUG: Calling LM Studio vision model for {file_name}")
                    description = call_lm_studio_vision(
                        saved_path,
                        "Describe this image in detail, focusing on all visible elements, objects, people, text, and context. Be comprehensive but concise."
                    )

                    print(f"DEBUG: LM Studio vision description for {file_name}: {description}")
                    print(f"DEBUG: Description length: {len(description)} characters")

                    if description and description.strip():
                        print(f"SUCCESS: Generated description for {file_name}")
                        for i, chunk in enumerate(iterative_chunking(description)):
                            print(f"DEBUG: Chunk {i+1} for {file_name}: {chunk[:100]}...")
                            all_text_chunks.append({"text": chunk, "source_path": saved_path})
                    else:
                        print(f"WARNING: Empty description for {file_name}")

                except Exception as desc_error:
                    print(f"ERROR: Could not generate description for {file_name} with LM Studio: {desc_error}")
                    print(f"DEBUG: Exception type: {type(desc_error).__name__}")
                    import traceback
                    print(f"DEBUG: Full traceback: {traceback.format_exc()}")

                processed_files += 1
                continue

            elif lower.endswith((".mp3", ".wav", ".m4a")):
                # Use whisper service for audio transcription
                try:
                    with open(saved_path, 'rb') as audio_file:
                        filename = os.path.basename(saved_path) or "audio.wav"
                        files = {'file': (filename, audio_file, 'audio/wav')}
                        response = requests.post(WHISPER_SERVICE_URL, files=files)

                    if response.status_code == 200:
                        result = response.json()
                        text = result.get('transcription', '')
                        print(f"Transcribed audio {file_name}: {len(text)} characters")
                    else:
                        print(f"Whisper service error for {file_name}: {response.status_code}")
                        text = ""

                except Exception as e:
                    print(f"Error transcribing audio {file_name}: {e}")
                    text = ""

            elif lower.endswith('.pdf'):
                print(f"DEBUG: Processing PDF file: {file_name}")
                try:
                    reader = pypdf.PdfReader(saved_path)
                    text = "\n".join([page.extract_text() for page in reader.pages])
                    print(f"DEBUG: Extracted {len(text)} characters from PDF {file_name}")
                except Exception as e:
                    print(f"DEBUG: Error reading PDF {file_name}: {e}")
                    text = ""

            elif lower.endswith('.docx'):
                print(f"DEBUG: Processing DOCX file: {file_name}")
                try:
                    doc = docx.Document(saved_path)
                    text = "\n".join([para.text for para in doc.paragraphs])
                    print(f"DEBUG: Extracted {len(text)} characters from DOCX {file_name}")
                except Exception as e:
                    print(f"DEBUG: Error reading DOCX {file_name}: {e}")
                    text = ""

            elif lower.endswith('.csv'):
                print(f"DEBUG: Processing CSV file: {file_name}")
                try:
                    df = pd.read_csv(saved_path)
                    text = df.to_string()
                    print(f"DEBUG: Extracted {len(text)} characters from CSV {file_name}")
                except Exception as e:
                    print(f"DEBUG: Error reading CSV {file_name}: {e}")
                    text = ""

            elif lower.endswith('.txt'):
                print(f"DEBUG: Processing TXT file: {file_name}")
                try:
                    with open(saved_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    print(f"DEBUG: Extracted {len(text)} characters from TXT {file_name}")
                except Exception as e:
                    print(f"DEBUG: Error reading TXT {file_name}: {e}")
                    text = ""
            else:
                print(f"DEBUG: Skipping unsupported file type: {file_name}")
                continue

            if text:
                print(f"DEBUG: Creating chunks for {file_name}")
                chunks = iterative_chunking(text)
                print(f"DEBUG: Created {len(chunks)} chunks for {file_name}")
                for i, chunk in enumerate(chunks):
                    print(f"DEBUG: Chunk {i+1} for {file_name}: {chunk[:100]}...")
                    all_text_chunks.append({"text": chunk, "source_path": saved_path})
                processed_files += 1
                print(f"DEBUG: Successfully processed document: {file_name}")
            else:
                print(f"DEBUG: No text content extracted from {file_name}")
        except Exception as e:
            return {"status": "error", "message": f"Error processing {file_name}: {e}"}

    # Batch embed and add to the text store
    if all_text_chunks:
        texts = [c['text'] for c in all_text_chunks]
        text_embeddings = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True).cpu().numpy()

        # Train the FAISS index if it's not trained (with safe training)
        if not text_vector_store.is_trained:
            training_success = safe_train_index(text_vector_store, text_embeddings)
            if not training_success:
                # Fallback: Create a new IndexFlatL2 if training fails
                print("Creating fallback IndexFlatL2 for text storage...")
                text_vector_store = faiss.IndexFlatL2(text_embedding_dim)
                # Re-add any existing embeddings if we're not resetting
                if text_metadata:
                    print("Warning: Switching to flat index - existing embeddings will be lost. Please re-ingest files.")

        text_vector_store.add(text_embeddings)
        text_metadata.extend(all_text_chunks)
        print(f"Added {len(all_text_chunks)} text chunks to the vector store.")

    if not all_text_chunks:
        return {"status": "error", "message": "No valid content extracted."}

    return {
        "status": "ok",
        "message": f"Processed {processed_files} files.",
        "text_chunks": len(text_metadata)
    }

@app.post("/chat")
async def chat(payload: ChatRequest):
    contexts = retrieve_multimodal_context(payload.query)

    if not contexts:
        return {"answer": "Knowledge base is empty. Please ingest files first."}

    try:
        # NEW: Unified Multimodal Pipeline Selection
        # Find the best image match (if any) from the unified search results
        image_contexts = [ctx for ctx in contexts if ctx['source'].lower().endswith(('.png', '.jpg', '.jpeg'))]

        if image_contexts:
            # Use the highest-scored image for analysis
            best_image_context = image_contexts[0]
            print(f"Activating Multimodal Pipeline with image: {os.path.basename(best_image_context['source'])}")

            source_path = best_image_context["source"]
            print(f"DEBUG: Calling LM Studio vision model for visual QA with query: {payload.query}")

            # Get vision analysis using LM Studio
            vision_answer = call_lm_studio_vision(source_path, payload.query)
            print(f"DEBUG: LM Studio vision answer: {vision_answer}")
            print(f"DEBUG: Answer length: {len(vision_answer)} characters")

            # Enhanced context: combine text content with image analysis
            text_contexts = [ctx for ctx in contexts if ctx.get('text', '').strip()]
            if text_contexts:
                additional_context = "\n".join([ctx['text'] for ctx in text_contexts[:2]])  # Include top 2 text contexts
                combined_context = f"{additional_context}\n\nIMAGE ANALYSIS: {vision_answer}"
            else:
                combined_context = vision_answer

            # Create properly formatted sources with clean names and correct citation mapping
            unique_sources = list(set([ctx["source"] for ctx in contexts]))
            sources_list = []

            for i, source_path in enumerate(unique_sources, 1):
                # Clean the filename (remove extension and path)
                clean_name = os.path.splitext(os.path.basename(source_path))[0]
                sources_list.append(f'<div class="source-item"><span class="source-key">[{i}]</span><span class="source-name">{clean_name}</span></div>')

            sources_str = "\n".join(sources_list)

            messages = [
                {"role": "system", "content": f"""
                You are a helpful AI assistant with access to multimodal content. Answer from the provided context, which may include both text and image analysis. Format your entire response as HTML. Include inline citations in the format [1], [2], etc. at the end of sentences where the information came from the context. Place the citation number before the period at the end of the sentence.

                After your main response, include a separator: --%Sources%--

                After the separator, provide a "Sources" section using this exact HTML structure:
                <div class="sources-section">
                  <h3>Sources</h3>
                  {sources_str}
                </div>

                Your entire response should follow this structure:
                MAIN RESPONSE CONTENT WITH [1] STYLE CITATIONS
                --%Sources%--
                <div class="sources-section">
                  <h3>Sources</h3>
                  <div class="source-item">
                    <span class="source-key">1</span>
                    <span class="source-name">source_name</span>
                  </div>
                </div>

                CRITICAL: Use ONLY the actual filenames provided in the sources section. Do NOT make up or invent source names like "image_analysis.txt" or "caption_inference.txt". Use the real filenames exactly as provided.

                Use proper HTML formatting for all content - paragraphs, lists, bold, italics, etc. Do not output raw markdown, only HTML.
                The citation numbers in the sources section should NOT have brackets around them.
                """},
                {"role": "user", "content": f"USER'S QUESTION: {payload.query}\n\nRELEVANT CONTEXT:\n{combined_context}"}
            ]

            # Use LM Studio for final response generation
            final_answer = call_lm_studio_text(messages, model="qwen3-1.7b", max_tokens=1500, temperature=0.7)

            return {
                "answer": final_answer
            }

        else:
            # Text-only pipeline (no relevant images found)
            print("Activating Text Pipeline...")
            text_contexts = [ctx for ctx in contexts if ctx.get('text', '').strip()]

            if not text_contexts:
                return {"answer": "No relevant text content found. Please ingest some documents first."}

            context_str = "\n".join([ctx['text'] for ctx in text_contexts])

            # Create properly formatted sources with clean names and correct citation mapping
            unique_sources = list(set([ctx["source"] for ctx in contexts]))
            sources_list = []

            for i, source_path in enumerate(unique_sources, 1):
                # Clean the filename (remove extension and path)
                clean_name = os.path.splitext(os.path.basename(source_path))[0]
                sources_list.append(f'<div class="source-item"><p class="source-name">{clean_name}</p></div>')

            sources_str = "\n".join(sources_list)

            # Debug: Print the generated sources for troubleshooting
            print(f"DEBUG: Generated sources HTML: {sources_str}")

            messages = [
                {"role": "system", "content": f"""
                You are a helpful AI assistant. Answer ONLY from the provided context. Format your entire response as HTML. Include inline citations in the format [1], [2], etc. at the end of sentences where the information came from the context. Place the citation number before the period at the end of the sentence.

                After your main response, include a separator: --%Sources%--

                After the separator, provide a "Sources" section using this exact HTML structure:
                <div class="sources-section">
                  <h3>Sources</h3>
                  <div class="source-item">
                    <span class="source-key">[1]</span>
                    <span class="source-name">document1.docx</span>
                  </div>
                  <div class="source-item">
                    <span class="source-key">[2]</span>
                    <span class="source-name">document2.pdf</span>
                  </div>
                </div>

                Your entire response should follow this structure:
                MAIN RESPONSE CONTENT WITH [1] STYLE CITATIONS
                --%Sources%--
                <div class="sources-section">
                  <h3>Sources</h3>
                  <div class="source-item">
                    <span class="source-key">1</span>
                    <span class="source-name">source_name</span>
                  </div>
                </div>

                Use proper HTML formatting for all content - paragraphs, lists, bold, italics, etc. Do not output raw markdown, only HTML.
                The citation numbers in the sources section should NOT have brackets around them.
                """},
                {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nUSER'S QUESTION: {payload.query}\n\n--%Sources%--\n<div class=\"sources-section\">\n<h3>Sources</h3>\n{sources_str}\n</div>"}
            ]

            # Use LM Studio for final response generation
            final_answer = call_lm_studio_text(messages, model="qwen3-1.7b", max_tokens=1500, temperature=0.7)

            return {
                "answer": final_answer
            }

    except Exception as e:
        return {"answer": f"Unexpected error: {e}"}

@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, "_temp_chat_audio")
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # Use whisper service for transcription
        try:
            with open(temp_path, 'rb') as audio_file:
                files = {'file': ('audio.wav', audio_file, 'audio/wav')}
                response = requests.post(WHISPER_SERVICE_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                transcribed_query = result.get('transcription', '').strip()
            else:
                return {"answer": f"Whisper service error: {response.status_code}"}

        except Exception as e:
            return {"answer": f"Error transcribing audio: {e}"}

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

        # Use whisper service for transcription
        try:
            with open(temp_path, 'rb') as audio_file:
                files = {'file': audio_file}
                response = requests.post(WHISPER_SERVICE_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                transcription = result.get('transcription', '')
                return {"transcription": transcription}
            else:
                return {"error": f"Whisper service error: {response.status_code}"}

        except Exception as e:
            return {"error": f"Error in transcription: {e}"}

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post('/generate_audio')
async def generate_audio(text: str = Form(...), voice: str = Form(default='af_heart')):
    if not KOKORO_AVAILABLE:
        return {"error": "Kokoro TTS is not available. Please install kokoro."}

    try:
        # Initialize the pipeline
        if not KOKORO_AVAILABLE:
            return {"error": "Kokoro TTS is not available. Please install kokoro."}
        pipeline = KPipeline(lang_code='a')
        # Generate audio from text
        generator = pipeline(text, voice=voice)
        audio_data = None
        sample_rate = 24000  # Kokoro's default sample rate

        # Get the first chunk of audio
        for i, (gs, ps, audio) in enumerate(generator):
            audio_data = audio
            break  # Only take the first chunk for now

        if audio_data is None:
            return {"error": "Failed to generate audio"}

        # Write audio to an in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()

        # Return the audio file with a more compatible content type
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg", headers={"Content-Disposition": "inline", "Content-Length": str(len(audio_bytes))})
    except Exception as e:
        return {"error": str(e)}

@app.post('/reset')
def reset():
    global text_vector_store, text_metadata
    if hasattr(text_vector_store, 'reset'):
        text_vector_store.reset()
    text_metadata = []
    return {"status": "ok", "message": "All knowledge bases cleared."}

if __name__ == '__main__':
    import uvicorn

    # Start whisper service as background process
    whisper_service_path = os.path.join(os.path.dirname(__file__), "whisper_service.py")
    print(f"Starting whisper service: {whisper_service_path}")

    whisper_process = subprocess.Popen(
        [sys.executable, whisper_service_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    def cleanup():
        print("Terminating whisper service...")
        whisper_process.terminate()
        whisper_process.wait()
        print("Whisper service terminated.")

    atexit.register(cleanup)

    print("Starting main server on port 8000...")
    uvicorn.run(app, host='0.0.0.0', port=8000)
