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
text_vector_store = None  # Will be initialized dynamically
text_metadata: List[dict] = []  # {"text": str, "source_path": str}

# Image store - REMOVED as we're using text embeddings for images now
# All content (text and image descriptions) will be stored in the text vector store

print("Backend: In-memory Vector DBs initialized.")

# --- LM Studio API Helper Functions ---
def call_lm_studio_text(messages, model="qwen/qwen3-4b", max_tokens=1000, temperature=0.7):
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

        # Validate response structure
        if 'choices' not in result or len(result['choices']) == 0:
            print(f"LM Studio API returned invalid response structure: {result}")
            return ""

        content = result['choices'][0]['message']['content']

        # Validate content
        if not content or not content.strip():
            print("LM Studio API returned empty content")
            return ""

        # Strip thinking tags from Qwen response
        if '<think>' in content and '</think>' in content:
            # Extract content between thinking tags and after them
            think_pattern = r'<think>.*?</think>'
            clean_content = re.sub(think_pattern, '', content, flags=re.DOTALL)
            clean_content = clean_content.strip()
            return clean_content

        return content.strip()

    except requests.exceptions.RequestException as e:
        print(f"LM Studio API request error: {e}")
        return ""
    except (KeyError, ValueError, TypeError) as e:
        print(f"LM Studio API response parsing error: {e}")
        return ""
    except Exception as e:
        print(f"LM Studio text API unexpected error: {e}")
        return ""

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

# --- FAISS Helper Functions ---
def calculate_optimal_nlist(n_samples: int) -> int:
    """
    Calculate optimal nlist based on number of samples using FAISS best practices.

    Args:
        n_samples: Number of data points

    Returns:
        int: Optimal nlist value
    """
    if n_samples < 10:
        return 1  # Minimal clustering for very small datasets
    elif n_samples < 40:
        return 1  # Still use minimal clustering
    else:
        # Use 4 * sqrt(N) heuristic, capped at 256
        return min(int(4 * (n_samples ** 0.5)), 256)

def choose_index_type(n_samples: int, embedding_dim: int):
    """
    Choose the best FAISS index type based on dataset size.

    Args:
        n_samples: Number of data points
        embedding_dim: Dimension of embeddings

    Returns:
        tuple: (index, index_type) where index_type is 'flat' or 'ivf'
    """
    if n_samples < 39:
        # Small dataset - use brute force (flat) index to avoid FAISS warnings
        print(f"Using IndexFlatL2 for small dataset ({n_samples} samples)")
        return faiss.IndexFlatL2(embedding_dim), 'flat'
    else:
        # Larger dataset - use IVF with calculated nlist
        nlist = calculate_optimal_nlist(n_samples)
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        print(f"Using IndexIVFFlat with nlist={nlist} for dataset ({n_samples} samples)")
        return index, 'ivf'

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

        response_content = call_lm_studio_text(messages, model="qwen/qwen3-4b", max_tokens=100, temperature=0.1)

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

def parse_simple_format(response_content: str) -> dict:
    """
    Parse simple key-value format responses like 'modality: text, boost: 1.2'
    """
    if not response_content or not response_content.strip():
        return {}

    response_content = response_content.strip()
    
    # Pattern to match modality: value, boost: value format
    modality_pattern = r'modality:\s*(\w+)[,\s]*boost:\s*([\d.]+)'
    match = re.search(modality_pattern, response_content, re.IGNORECASE)
    
    if match:
        modality = match.group(1).lower()
        boost = float(match.group(2))
        return {modality: boost}
    
    # If modality-boost pattern doesn't match, try to extract any modality/boost values separately
    modality_match = re.search(r'modality:\s*(\w+)', response_content, re.IGNORECASE)
    boost_match = re.search(r'boost:\s*([\d.]+)', response_content, re.IGNORECASE)
    
    if modality_match and boost_match:
        modality = modality_match.group(1).lower()
        boost = float(boost_match.group(1))
        return {modality: boost}
    
    # If we can't find the format, try to extract just the modality if mentioned
    if 'image' in response_content.lower():
        return {"image": 1.0}
    elif 'audio' in response_content.lower():
        return {"audio": 1.0}
    else:
        return {"text": 1.0}  # Default to text if nothing else matches

def parse_sources_format(response_content: str) -> dict:
    """
    Parse simple format for sources like 'best_sources: [file1.pdf, file2.pdf]'
    or 'best_sources: file1.pdf, file2.pdf'
    """
    if not response_content or not response_content.strip():
        return {}

    response_content = response_content.strip()
    
    # Pattern to match best_sources: [file1, file2] format
    sources_pattern = r'best_sources:\s*\[([^\]]+)\]'
    match = re.search(sources_pattern, response_content, re.IGNORECASE)
    
    if match:
        sources_str = match.group(1)
        # Split by comma and clean up the filenames
        sources = [s.strip().strip('"\'') for s in sources_str.split(',')]
        sources = [s for s in sources if s]  # Remove empty strings
        return {"best_sources": sources}
    
    # Alternative pattern for best_sources: file1, file2 format (without brackets)
    alt_pattern = r'best_sources:\s*([^\n\r]+)'
    match = re.search(alt_pattern, response_content, re.IGNORECASE)
    
    if match:
        sources_str = match.group(1)
        # Split by comma and clean up the filenames
        sources = [s.strip().strip('"\'') for s in sources_str.split(',')]
        sources = [s for s in sources if s]  # Remove empty strings
        return {"best_sources": sources}
    
    # If we can't find the specific format, try to extract any filenames we can find
    # Look for common file extensions in the response
    filename_pattern = r'[\w\-\s]+\.(pdf|docx|jpg|jpeg|png|txt|csv|wav|mp3|m4a)'
    found_files = re.findall(filename_pattern, response_content, re.IGNORECASE)
    if found_files:
        # We only found the extensions, we need to get the full names
        # This is a simpler fallback
        pass
    
    return {}  # Return empty dict if no pattern matches

def parse_json_with_retry(response_content: str, max_retries: int = 3) -> dict:
    """
    Try to parse JSON response with multiple retry attempts.

    Args:
        response_content: The response content to parse
        max_retries: Maximum number of retry attempts

    Returns:
        dict: Parsed JSON object or empty dict if all retries fail
    """
    if not response_content or not response_content.strip():
        return {}

    response_content = response_content.strip()

    for attempt in range(max_retries):
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                # Could add a small delay here if needed
                continue
            else:
                print(f"All {max_retries} JSON parsing attempts failed")
                # Try to parse in simple format as fallback
                return parse_simple_format(response_content)

    return {}

def detect_modality(query: str, document_names: List[str]) -> dict:
    """
    Detect modality to boost based on user query and available documents.
    """
    messages = [
        {
            "role": "system",
            "content": """You are an intelligent assistant that determines which modality to boost for a search. Your options are "text", "image", or "audio". Respond ONLY with the format "modality: [modality_name], boost: [number]"

Examples:
modality: text, boost: 1.2
modality: image, boost: 1.5
modality: audio, boost: 1.3

Response format: modality: [modality_name], boost: [number]
- Use ONLY the format above, no additional text
- Choose from: text, image, or audio
- Boost value should be between 1.0 and 2.0
- Respond with EXACTLY ONE line in the specified format"""
        },
        {
            "role": "user",
            "content": f'Given the user query "{query}" and the following documents in the knowledge base: {document_names}, which modality should be boosted?'
        }
    ]

    max_retries = 3

    for attempt in range(max_retries):
        try:
            print(f"Modality detection attempt {attempt + 1}/{max_retries}")
            response_content = call_lm_studio_text(messages, model="qwen/qwen3-4b", max_tokens=100, temperature=0.1)

            # Check if response is empty or None
            if not response_content or not response_content.strip():
                print(f"LM Studio returned empty response for modality detection (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    return {"text": 1.0}
                continue

            # Parse the simple format response
            parsed = parse_simple_format(response_content)

            if parsed:
                # Get the first key-value pair from the parsed result
                for modality, boost in parsed.items():
                    print(f"Successfully detected modality: {modality} with boost {boost}")
                    return {modality: boost}
            else:
                print(f"Failed to parse response for modality detection (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    print("All modality detection attempts failed, using fallback")
                    break

        except Exception as e:
            print(f"Error in modality detection (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                break

    # Fallback to keyword-based detection after all retries failed
    print("Using keyword-based fallback for modality detection")
    query_lower = query.lower()
    if any(ext in query_lower for ext in ['image', 'picture', 'photo', 'visual', 'show me', 'describe', 'looks like', 'see', 'view', 'diagram', 'chart', 'graph']):
        return {"image": 1.5}
    elif any(ext in query_lower for ext in ['audio', 'sound', 'listen']):
        return {"audio": 1.5}
    else:
        return {"text": 1.2} # Default to boosting text slightly

def retrieve_multimodal_context(query: str, top_k: int = 5) -> List[dict]:
    """
    Unified multimodal retrieval that searches across all content types simultaneously.
    Returns the most relevant content regardless of modality (text, image, audio).
    All content (text documents and image descriptions) is now in the same text embedding space.
    """
    print(f"\n=== DEBUG: retrieve_multimodal_context called with query: {query!r} ===")

    # 1. DETECT MODALITY
    document_names = [os.path.basename(meta['source_path']) for meta in text_metadata]
    modality_boosts = detect_modality(query, document_names)
    print(f"DEBUG: Modality boosts: {modality_boosts}")

    all_candidates = []

    # 2. SEARCH TEXT CONTENT (includes text from documents and image descriptions)
    if text_vector_store is not None:
        print(f"DEBUG: Text vector store size: {text_vector_store.ntotal}")
        if text_vector_store.ntotal > 0:
            query_text_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
            distances, indices = text_vector_store.search(query_text_embedding, k=min(top_k * 2, text_vector_store.ntotal))

            print(f"DEBUG: Text search found {len(indices[0])} candidates")
            for i, idx in enumerate(indices[0]):
                if idx < len(text_metadata):
                    source_path = text_metadata[idx]["source_path"]
                    raw_distance = distances[0][i]
                    score = normalize_score(raw_distance, "text_faiss")

                    modality = "text"
                    if source_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        modality = "image"
                    elif source_path.lower().endswith(('.mp3', '.wav', '.m4a')):
                        modality = "audio"

                    boost = modality_boosts.get(modality, 1.0)
                    boosted_score = score * boost

                    all_candidates.append({
                        "source": source_path,
                        "text": text_metadata[idx]["text"],
                        "text_score": boosted_score,
                        "type": "text" if modality != "image" else "image_description",
                        "modality": modality
                    })
        else:
            print("DEBUG: Text vector store is empty")
    else:
        print("DEBUG: Text vector store is not initialized")

    # 3. RE-RANK USING CROSS-ENCODER (for text content)
    text_candidates = [c for c in all_candidates if c.get("text", "").strip()]
    if text_candidates and cross_encoder_model is not None:
        try:
            pairs = [[query, cand["text"]] for cand in text_candidates]
            cross_encoder_scores = cross_encoder_model.predict(pairs)
            for i, candidate in enumerate(text_candidates):
                candidate["cross_encoder_score"] = normalize_score(cross_encoder_scores[i], "cross_encoder")
        except Exception as e:
            print(f"DEBUG: Cross-encoder failed: {e}")

    # 4. CALCULATE FINAL SCORES
    for candidate in all_candidates:
        candidate["final_score"] = calculate_final_score(candidate)

    # 5. SORT BY FINAL SCORE
    all_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # 6. RETURN TOP RESULTS
    top_results = all_candidates[:5]
    print(f"DEBUG: Top 5 results:")
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
        if text_vector_store is not None and hasattr(text_vector_store, 'reset'):
            text_vector_store.reset()
        text_metadata = []

    all_text_chunks: List[dict] = []
    processed_files = 0

    for file_obj in files:
        file_name = file_obj.filename or f"temp_file_{id(file_obj)}"
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

        # Initialize or update the vector store if needed
        if text_vector_store is None or (reset_db.lower() == "true" and len(text_metadata) == 0):
            # Choose appropriate index type based on number of chunks
            text_vector_store, _ = choose_index_type(len(all_text_chunks), text_embedding_dim)

        # Train the FAISS index if it's not trained and it's an IVF index
        if hasattr(text_vector_store, 'is_trained') and not text_vector_store.is_trained:
            training_success = safe_train_index(text_vector_store, text_embeddings)
            if not training_success:
                # Fallback: Create a new IndexFlatL2 if training fails
                print("Creating fallback IndexFlatL2 for text storage...")
                old_store = text_vector_store
                text_vector_store = faiss.IndexFlatL2(text_embedding_dim)
                # Re-add any existing embeddings if we're not resetting
                if text_metadata and hasattr(old_store, 'ntotal') and old_store.ntotal > 0:
                    print("Warning: Switching to flat index - existing embeddings will be lost. Please re-ingest files.")

        # Add embeddings to the vector store
        if text_vector_store is not None:
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
        # Skip the judge step and directly use all contexts
        final_contexts = contexts

        # GENERATE RESPONSE with all contexts
        context_str = "\n".join([ctx['text'] for ctx in final_contexts])
        unique_sources = list(set([ctx["source"] for ctx in final_contexts]))
        
        # Create a consistent mapping of source paths to citation numbers
        source_to_citation = {}
        citation_counter = 1
        for ctx in final_contexts:
            source_path = ctx["source"]
            if source_path not in source_to_citation:
                source_to_citation[source_path] = citation_counter
                citation_counter += 1
        
        # Create sources list with consistent numbering
        sources_list = []
        added_sources = set()
        for source_path in unique_sources:
            if source_path not in added_sources:
                citation_number = source_to_citation[source_path]
                clean_name = os.path.splitext(os.path.basename(source_path))[0]
                sources_list.append(f'<div class="source-item"><p class="source-name">{clean_name}</p></div>')
                added_sources.add(source_path)
        sources_str = "\n".join(sources_list)

        messages = [
            {
                "role": "system",
                "content": f'''
                You are a Multimodal RAG agent. Generate responses in VALID HTML format only.

**STRICT REQUIREMENTS:**
1. Respond with COMPLETE HTML document structure
2. Use proper HTML tags: <p>, <ul>, <li>, <strong>, <em>, etc.
3. Include citations as [1], [2], etc. within the HTML content
4. End with --%Sources%-- separator
5. Follow with proper sources section in HTML format

**SOURCE SELECTION CRITERIA:**
- ONLY include sources that directly contribute to answering the user's query
- If a source is not relevant to the user's question, DO NOT include it in your response
- It is NOT mandatory to use all provided sources - only use what is necessary
- The sources provided are candidates detected by algorithms, but you must judge their relevance

**CITATION CONSISTENCY REQUIREMENT:**
- The citation numbers [1], [2], etc. in the content must correspond to the same numbered sources in the sources section below
- Source [1] in content must match source [1] in the sources section, source [2] must match, etc.
- The same document must always use the same citation number throughout the response

**RESPONSE STRUCTURE:**
```html
<p>Your main content here with citations [1] and proper HTML formatting.</p>
<p>Multiple paragraphs are allowed.</p>
<ul>
    <li>List items with citations [2]</li>
    <li>More list items</li>
</ul>
<p><strong>Bold text</strong> and <em>italic text</em> are allowed.</p>
--%Sources%--
<div class="sources-section">
    <h3>Sources</h3>
    <div class="source-item">
        <span class="source-key">1</span>
        <span class="source-name">source_name_1</span>
    </div>
    <div class="source-item">
        <span class="source-key">2</span>
        <span class="source-name">source_name_2</span>
    </div>
</div>
```

**CRITICAL:**
- NO plain text outside HTML tags
- NO malformed citations like "citation.1"
- NO text after the sources section
- ONLY include sources you actually cited in your response
- Use EXACTLY the source names provided
- Ensure citation numbers match source numbers (e.g., [1] in content matches [1] in sources)
- Ensure valid HTML structure
- The same document must always use the same citation number throughout the response

**EXAMPLES OF INCORRECT RESPONSES TO AVOID:**
❌ "This is a list item with a citation.1"
❌ "Sources 1 output"
❌ Plain text without HTML tags
❌ Including irrelevant sources that don't contribute to the answer

**EXAMPLES OF CORRECT RESPONSES:**
✅ "<p>This is properly formatted HTML content [1].</p><p>More content here.</p>--%Sources%--<div class=\"sources-section\"><h3>Sources</h3><div class=\"source-item\"><span class=\"source-key\">1</span><span class=\"source-name\">document.pdf</span></div></div>"
'''
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context_str}\n\nUSER'S QUESTION: {payload.query}\n\n--%Sources%--\n<div class=\"sources-section\">\n<h3>Sources</h3>\n{sources_str}\n</div>"
            }
        ]

        final_answer = call_lm_studio_text(messages, model="qwen/qwen3-4b", max_tokens=1500, temperature=0.7)

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
                files = {'file': audio_file}
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
        if not KOKORO_AVAILABLE or KPipeline is None:
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
    if text_vector_store is not None and hasattr(text_vector_store, 'reset'):
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
