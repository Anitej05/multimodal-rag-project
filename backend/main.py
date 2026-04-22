#!/usr/bin/env python3
"""
New main.py using LM Studio API instead of local models
Combines functionality from main.py and main_hybrid.py
Preserves all original functionality except LLM inference method
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, RedirectResponse
import mammoth
from typing import List
import threading
import torch
import faiss
import os
import base64
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
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WHISPER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "faster-whisper-base")
LM_STUDIO_URL = "http://localhost:1234"

os.makedirs(UPLOAD_DIR, exist_ok=True)

print(f"Backend: Using device '{DEVICE}'.")
print(f"Backend: Local models directory: {MODELS_DIR}")
print(f"Backend: LM Studio endpoint: {LM_STUDIO_URL}")
print(f"Backend: Whisper model path: {WHISPER_MODEL_PATH}")

# --- Initialize FastAPI App ---
app = FastAPI(title="Multimodal RAG Backend with LM Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Manager (GPU VRAM Mode-Switching) ---
EMB_MODEL_PATH = os.path.join(MODELS_DIR, "qwen3-vl-embedding-2b")
CE_MODEL_PATH = os.path.join(MODELS_DIR, "cross-encoder-minilm")
WHISPER_COMPUTE = "float16" if DEVICE == "cuda" else "int8"

class ModelManager:
    """
    Manages GPU VRAM by dynamically loading/unloading models.
    Modes:
      - "rag": embedding, cross-encoder, whisper loaded on GPU
      - "digitize": all RAG models offloaded, GPU free for PaddleOCR
    """
    def __init__(self):
        self.mode = "rag"
        self.lock = threading.Lock()
        self.embedding_model = None
        self.cross_encoder_model = None
        self.whisper_model = None
        self._load_rag_models()

    def _load_rag_models(self):
        """Load all RAG models onto GPU."""
        print("ModelManager: Loading RAG models on GPU...")

        print(f"  Loading embedding model from {EMB_MODEL_PATH}...")
        self.embedding_model = SentenceTransformer(
            EMB_MODEL_PATH,
            device=DEVICE,
            model_kwargs={"torch_dtype": "bfloat16"},
        )
        print(f"  Embedding model loaded. Dim: {self.embedding_model.get_sentence_embedding_dimension()}")

        print(f"  Loading cross-encoder from {CE_MODEL_PATH}...")
        try:
            self.cross_encoder_model = CrossEncoder(CE_MODEL_PATH, max_length=512, device=DEVICE)
            print(f"  Cross-encoder loaded on {DEVICE}.")
        except Exception as e:
            print(f"  Failed to load cross-encoder: {e}")
            self.cross_encoder_model = None

        print(f"  Loading Whisper from {WHISPER_MODEL_PATH}...")
        try:
            from faster_whisper import WhisperModel
            self.whisper_model = WhisperModel(WHISPER_MODEL_PATH, device=DEVICE, compute_type=WHISPER_COMPUTE)
            print(f"  Whisper loaded on {DEVICE}.")
        except Exception as e:
            print(f"  Whisper failed to load: {e}")
            self.whisper_model = None

        self.mode = "rag"
        print("ModelManager: All RAG models loaded. Mode = rag")

    def _unload_rag_models(self):
        """Unload all RAG models from GPU to free VRAM."""
        print("ModelManager: Unloading RAG models from GPU...")
        self.embedding_model = None
        self.cross_encoder_model = None
        self.whisper_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()
        self.mode = "digitize"
        print("ModelManager: RAG models unloaded. VRAM freed. Mode = digitize")

    def switch_to_digitize(self):
        """Switch to digitize mode — free GPU for OCR."""
        with self.lock:
            if self.mode == "digitize":
                return {"mode": "digitize", "status": "already_active"}
            self._unload_rag_models()
            return {"mode": "digitize", "status": "ready"}

    def switch_to_rag(self):
        """Switch to RAG mode — reload models on GPU."""
        with self.lock:
            if self.mode == "rag":
                return {"mode": "rag", "status": "already_active"}
            self._load_rag_models()
            return {"mode": "rag", "status": "ready"}

    def get_status(self):
        return {
            "mode": self.mode,
            "embedding_loaded": self.embedding_model is not None,
            "cross_encoder_loaded": self.cross_encoder_model is not None,
            "whisper_loaded": self.whisper_model is not None,
            "gpu_available": torch.cuda.is_available(),
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1) if torch.cuda.is_available() else 0,
        }

# Initialize the model manager (loads RAG models on startup)
model_mgr = ModelManager()

# Convenience accessors (so existing code doesn't break)
# These are properties that read from model_mgr
def _get_embedding_model():
    return model_mgr.embedding_model

def _get_cross_encoder():
    return model_mgr.cross_encoder_model

def _get_whisper():
    return model_mgr.whisper_model

print("Backend: All models loaded via ModelManager.")

# --- Vector Stores (FAISS) ---
print("Backend: Initializing vector stores...")
# Text store (Qwen3-VL-Embedding-2B produces 2048-dim embeddings)
text_embedding_dim = 2048
text_vector_store = None  # Will be initialized dynamically
text_metadata: List[dict] = []  # {"text": str, "source_path": str}

# Image store - REMOVED as we're using text embeddings for images now
# All content (text and image descriptions) will be stored in the text vector store

# --- Knowledge Graph (LLM-extracted entities & relationships) ---
knowledge_graph_data = {"nodes": [], "edges": [], "clusters": 0}
kg_lock = threading.Lock()

# --- Ingestion Status (for async background processing) ---
ingest_status = {
    "is_running": False,
    "total_files": 0,
    "processed_files": 0,
    "current_file": "",
    "phase": "idle",  # idle, embedding, extracting_kg, done, error
    "message": "",
    "kg_entities_found": 0,
}
ingest_lock = threading.Lock()

print("Backend: In-memory Vector DBs initialized.")

# --- LM Studio API Helper Functions ---
# Model configuration - Qwen3.5-4B is natively multimodal (text + images)
LM_STUDIO_TEXT_MODEL = "qwen3.5-4b"

def call_lm_studio_text(messages, model=None, max_tokens=1000, temperature=0.7):
    """Call LM Studio API for text generation"""
    if model is None:
        model = LM_STUDIO_TEXT_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
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

def stream_lm_studio_text(messages, model=None, max_tokens=1500, temperature=0.7):
    """Stream text generation from LM Studio API, yielding SSE-formatted chunks."""
    if model is None:
        model = LM_STUDIO_TEXT_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    try:
        response = requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        response.raise_for_status()

        inside_think = False
        think_buffer = ""

        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode('utf-8')
            if not decoded.startswith('data: '):
                continue
            data_str = decoded[6:]  # Remove 'data: ' prefix
            if data_str.strip() == '[DONE]':
                break  # Don't yield [DONE] here - let the caller control termination

            try:
                chunk = json.loads(data_str)
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                content = delta.get('content', '')

                if not content:
                    continue

                # Strip <think>...</think> tags on the fly
                if '<think>' in content:
                    inside_think = True
                    think_buffer = ""
                    # Check if </think> is also in the same chunk
                    if '</think>' in content:
                        inside_think = False
                        content = content.split('</think>', 1)[-1]
                        if not content:
                            continue
                    else:
                        continue

                if inside_think:
                    if '</think>' in content:
                        inside_think = False
                        content = content.split('</think>', 1)[-1]
                        if not content:
                            continue
                    else:
                        continue

                # Send the token to the client
                sse_data = json.dumps({"token": content})
                yield f"data: {sse_data}\n\n"

            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    except Exception as e:
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n"
        yield f"data: [DONE]\n\n"

# NOTE: call_lm_studio_vision() has been removed.
# Image understanding is now handled natively by Qwen3-VL-Embedding-2B
# which embeds images directly into the same vector space as text.

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


def retrieve_multimodal_context(query: str, top_k: int = 5) -> List[dict]:
    """
    Unified multimodal retrieval: vector search + cross-encoder re-ranking.
    All content (text, images, audio transcriptions) lives in the same embedding space.
    """
    print(f"\n=== DEBUG: retrieve_multimodal_context called with query: {query!r} ===")

    all_candidates = []

    # 1. VECTOR SEARCH across all content
    if text_vector_store is not None:
        print(f"DEBUG: Vector store size: {text_vector_store.ntotal}")
        if text_vector_store.ntotal > 0:
            query_embedding = model_mgr.embedding_model.encode(query, convert_to_tensor=True).float().cpu().numpy()
            query_embedding = query_embedding.reshape(1, -1)
            distances, indices = text_vector_store.search(query_embedding, k=min(top_k * 2, text_vector_store.ntotal))

            print(f"DEBUG: Search found {len(indices[0])} candidates")
            for i, idx in enumerate(indices[0]):
                if idx < len(text_metadata):
                    source_path = text_metadata[idx]["source_path"]
                    raw_distance = distances[0][i]
                    score = normalize_score(raw_distance, "text_faiss")

                    all_candidates.append({
                        "source": source_path,
                        "text": text_metadata[idx]["text"],
                        "text_score": score,
                        "type": "image" if text_metadata[idx].get("is_image") else "text",
                    })
        else:
            print("DEBUG: Vector store is empty")
    else:
        print("DEBUG: Vector store is not initialized")

    # 2. RE-RANK with cross-encoder (only for text candidates, not raw images)
    text_candidates = [c for c in all_candidates if c["type"] == "text" and c.get("text", "").strip()]
    if text_candidates and model_mgr.cross_encoder_model is not None:
        try:
            pairs = [[query, cand["text"]] for cand in text_candidates]
            cross_encoder_scores = model_mgr.cross_encoder_model.predict(pairs)
            for i, candidate in enumerate(text_candidates):
                candidate["cross_encoder_score"] = normalize_score(cross_encoder_scores[i], "cross_encoder")
        except Exception as e:
            print(f"DEBUG: Cross-encoder failed: {e}")

    # 3. CALCULATE FINAL SCORES
    for candidate in all_candidates:
        candidate["final_score"] = calculate_final_score(candidate)

    # 4. SORT AND RETURN TOP RESULTS
    all_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = all_candidates[:top_k]

    print(f"DEBUG: Top {len(top_results)} results:")
    for i, result in enumerate(top_results, 1):
        print(f"  {i}. {os.path.basename(result['source'])} (score: {result['final_score']:.3f}, type: {result['type']})")

    print("=== END DEBUG: retrieve_multimodal_context ===\n")
    return top_results

# --- Multimodal Message Builder ---
def build_chat_messages(query: str, contexts: list, system_prompt: str) -> list:
    """
    Build LM Studio chat messages with multimodal support.
    If retrieved contexts include images, they are sent as base64 to Qwen3.5-4B
    which natively understands both text and images.
    """
    # Separate text and image contexts
    text_parts = []
    image_parts = []
    for ctx in contexts:
        if ctx.get("type") == "image":
            source_path = ctx["source"]
            if os.path.exists(source_path):
                image_parts.append(source_path)
        else:
            text_parts.append(ctx["text"])

    context_str = "\n".join(text_parts)

    # Build the user message content
    user_content = []

    # Add images first so the model can see them
    for img_path in image_parts:
        try:
            with open(img_path, 'rb') as img_file:
                image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
            ext = os.path.splitext(img_path)[1].lower()
            mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png'}.get(ext.lstrip('.'), 'image/jpeg')
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{image_b64}"}
            })
            print(f"DEBUG: Attached image {os.path.basename(img_path)} to LLM message")
        except Exception as e:
            print(f"DEBUG: Failed to attach image {img_path}: {e}")

    # Add the text prompt
    prompt_text = f"CONTEXT:\n{context_str}\n\nUSER'S QUESTION: {query}" if context_str else f"USER'S QUESTION: {query}"
    user_content.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    return messages

# --- Knowledge Graph: LLM-based Entity & Relationship Extraction ---
def extract_entities_from_chunk(chunk_text: str, source_file: str) -> dict:
    """
    Use Qwen3.5-4B via LM Studio to extract entities and relationships from a text chunk.
    Returns {entities: [{name, type}], relationships: [{source, target, relation}]}
    """
    if not chunk_text or len(chunk_text.strip()) < 20:
        return {"entities": [], "relationships": []}

    messages = [
        {
            "role": "system",
            "content": """You are an entity extraction engine. Extract named entities and relationships from the given text.

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{"entities":[{"name":"Entity Name","type":"person|organization|technology|concept|location|event"}],"relationships":[{"source":"Entity1","target":"Entity2","relation":"relation_description"}]}

Rules:
- Extract 3-8 important entities per chunk
- Entity types: person, organization, technology, concept, location, event
- Only extract clear, meaningful relationships
- Keep entity names concise (1-4 words)
- If no clear entities exist, return {"entities":[],"relationships":[]}"""
        },
        {
            "role": "user",
            "content": f"Extract entities and relationships from this text:\n\n{chunk_text[:800]}"
        }
    ]

    try:
        response = call_lm_studio_text(messages, max_tokens=500, temperature=0.1)
        if not response or not response.strip():
            return {"entities": [], "relationships": []}

        # Try to extract JSON from the response
        response = response.strip()

        # Handle markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            response = "\n".join(json_lines)

        # Find JSON object in response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            parsed = json.loads(json_str)

            # Validate structure
            entities = parsed.get("entities", [])
            relationships = parsed.get("relationships", [])

            # Filter out malformed entries
            valid_entities = []
            for e in entities:
                if isinstance(e, dict) and "name" in e and "type" in e:
                    e["name"] = str(e["name"]).strip()[:50]
                    e["type"] = str(e["type"]).lower().strip()
                    if e["type"] not in {"person", "organization", "technology", "concept", "location", "event"}:
                        e["type"] = "concept"
                    if len(e["name"]) > 1:
                        valid_entities.append(e)

            valid_rels = []
            entity_names = {e["name"].lower() for e in valid_entities}
            for r in relationships:
                if isinstance(r, dict) and "source" in r and "target" in r and "relation" in r:
                    if r["source"].lower() in entity_names and r["target"].lower() in entity_names:
                        valid_rels.append(r)

            return {"entities": valid_entities, "relationships": valid_rels}

        return {"entities": [], "relationships": []}

    except Exception as e:
        print(f"DEBUG: Entity extraction failed for chunk from {source_file}: {e}")
        return {"entities": [], "relationships": []}


def build_knowledge_graph_from_metadata():
    """
    Build a rich knowledge graph using multi-layer NLP analysis (NO LLM calls):
      1. Named entity extraction (regex-based: proper nouns, dates, emails, URLs)
      2. Keyphrase extraction (bigram/trigram TF-IDF)
      3. Document similarity from existing FAISS embeddings
      4. Sentence-level co-occurrence analysis
      5. Entity type classification via pattern matching
    """
    global knowledge_graph_data
    import math
    from collections import Counter, defaultdict

    STOP_WORDS = {
        'the','a','an','is','are','was','were','be','been','being','have','has',
        'had','do','does','did','will','would','could','should','may','might',
        'shall','can','need','to','of','in','for','on','with','at','by','from',
        'as','into','through','during','before','after','above','below','between',
        'out','off','over','under','again','further','then','once','and','but',
        'or','nor','not','so','if','when','what','which','who','whom','this',
        'that','these','those','me','my','myself','our','ours','you','your','he',
        'she','it','its','they','them','their','each','all','both','few','more',
        'most','other','some','such','only','own','same','than','too','very',
        'just','because','here','there','where','how','also','about','up','down',
        'any','every','while','until','like','using','make','made','many','much',
        'well','even','still','however','therefore','thus','hence','since',
        'although','though','whereas','whether','within','without','upon','among',
        'along','across','behind','beyond','toward','towards','onto','into',
        'content','information','based','include','including','provide','following',
        'example','note','type','name','number','first','second','third','last',
        'next','new','old','good','best','high','low','long','short','large',
        'small','different','important','available','possible','specific','general',
        'common','know','think','want','come','take','give','tell','work','call',
        'look','find','help','show','part','place','case','point','group','need',
        'turn','start','might','world','area','image','file','document','chunk',
        'text','data','page','used','must','also','said','says','get','got',
        'one','two','three','four','five','use','way','see','now','may','will',
    }

    nodes = []
    edges = []
    file_nodes = {}
    entity_registry = {}     # key -> {id, label, type, sources, count, category}
    edge_set = set()         # prevent duplicate edges

    def add_edge(src_id, tgt_id, weight, edge_type, label=""):
        key = f"{src_id}|{tgt_id}|{edge_type}"
        if key not in edge_set:
            edge_set.add(key)
            edges.append({"source": src_id, "target": tgt_id, "weight": weight,
                          "type": edge_type, "label": label})

    def register_entity(name, category, source_path):
        key = name.lower().strip()
        if len(key) < 3 or key in STOP_WORDS:
            return None
        if key not in entity_registry:
            entity_registry[key] = {
                "id": f"entity-{len(entity_registry)}",
                "label": name.strip(),
                "category": category,
                "sources": set(),
                "count": 0
            }
        entity_registry[key]["sources"].add(source_path)
        entity_registry[key]["count"] += 1
        return entity_registry[key]["id"]

    # ── Hub node ──
    nodes.append({
        "id": "hub-knowledge-base", "label": "Knowledge Base", "type": "hub",
        "size": 55, "description": f"Central repository with {len(text_metadata)} indexed items"
    })

    # ── Group metadata by source file ──
    source_chunks = {}
    for meta in text_metadata:
        src = meta.get("source_path", "unknown")
        source_chunks.setdefault(src, []).append(meta)

    num_docs = max(len(source_chunks), 1)

    # ══════════════════════════════════════════
    # LAYER 1: Named Entity Extraction (regex)
    # ══════════════════════════════════════════
    # Patterns for different entity types
    date_pattern = re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b', re.IGNORECASE)
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
    number_pattern = re.compile(r'\b\d+(?:\.\d+)?(?:\s*(?:%|percent|dollars?|USD|EUR|GBP|INR|kg|km|miles?|hours?|years?|months?|days?|MB|GB|TB))\b', re.IGNORECASE)
    proper_noun_pattern = re.compile(r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
    acronym_pattern = re.compile(r'\b[A-Z]{2,6}\b')
    tech_pattern = re.compile(r'\b(?:Python|JavaScript|TypeScript|React|Angular|Vue|Node\.?js|Docker|Kubernetes|AWS|Azure|GCP|API|REST|GraphQL|SQL|NoSQL|MongoDB|PostgreSQL|Redis|TensorFlow|PyTorch|CUDA|GPU|CPU|RAM|HTTP|HTTPS|JSON|XML|CSV|HTML|CSS|ML|AI|NLP|LLM|RAG|BERT|GPT|Transformer|FAISS|ONNX)\b', re.IGNORECASE)

    file_texts = {}  # source_path -> full text
    file_sentences = {}  # source_path -> list of sentences

    for source_path, metas in source_chunks.items():
        text_metas = [m for m in metas if not m.get("is_image")]
        full_text = " ".join(m.get("text", "") for m in text_metas)
        file_texts[source_path] = full_text

        # Split into sentences for co-occurrence
        sentences = re.split(r'[.!?]+', full_text)
        file_sentences[source_path] = [s.strip() for s in sentences if len(s.strip()) > 15]

        # Extract proper nouns (multi-word capitalized phrases)
        for match in proper_noun_pattern.finditer(full_text):
            name = match.group()
            if len(name) > 4 and name.lower() not in STOP_WORDS:
                register_entity(name, "person_or_org", source_path)

        # Extract dates
        for match in date_pattern.finditer(full_text):
            register_entity(match.group(), "date", source_path)

        # Extract emails
        for match in email_pattern.finditer(full_text):
            register_entity(match.group(), "email", source_path)

        # Extract URLs
        for match in url_pattern.finditer(full_text):
            url = match.group()[:50]  # truncate long URLs
            register_entity(url, "url", source_path)

        # Extract metrics/numbers with units
        for match in number_pattern.finditer(full_text):
            register_entity(match.group().strip(), "metric", source_path)

        # Extract technology terms
        for match in tech_pattern.finditer(full_text):
            register_entity(match.group(), "technology", source_path)

        # Extract acronyms (but filter common ones)
        common_acronyms = {'THE','AND','FOR','BUT','NOT','YOU','ALL','CAN','HAD','HER','WAS','ONE','OUR','OUT','HAS','HIS','HOW','ITS','LET','MAY','OLD','SEE','WAY','WHO','BOY','DID','GET','HIM','HIT','HOT','MAN','OIL','SIT','TOP'}
        for match in acronym_pattern.finditer(full_text):
            acr = match.group()
            if acr not in common_acronyms and len(acr) >= 2:
                register_entity(acr, "acronym", source_path)

    # ══════════════════════════════════════════
    # LAYER 2: Keyphrase Extraction (bigram TF-IDF)
    # ══════════════════════════════════════════
    file_bigrams = {}
    doc_bigram_freq = Counter()

    for source_path, full_text in file_texts.items():
        words = re.findall(r'[a-z]+', full_text.lower())
        filtered = [w for w in words if w not in STOP_WORDS and len(w) > 3]

        # Bigrams
        bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered)-1)]
        bigram_count = Counter(bigrams)
        # Filter rare bigrams
        bigram_count = {k: v for k, v in bigram_count.items() if v >= 2}
        file_bigrams[source_path] = bigram_count
        for bg in bigram_count:
            doc_bigram_freq[bg] += 1

    # TF-IDF for bigrams -> top 5 keyphrases per file
    for source_path, bigram_count in file_bigrams.items():
        total = sum(bigram_count.values())
        if total == 0:
            continue
        scored = {}
        for bg, count in bigram_count.items():
            tf = count / total
            idf = math.log((num_docs + 1) / (doc_bigram_freq[bg] + 1)) + 1
            scored[bg] = tf * idf
        top_phrases = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:5]
        for phrase, score in top_phrases:
            register_entity(phrase.title(), "keyphrase", source_path)

    # Also do single-word TF-IDF for top concepts
    file_word_counts = {}
    doc_word_freq = Counter()
    for source_path, full_text in file_texts.items():
        words = re.findall(r'[a-z]+', full_text.lower())
        filtered = [w for w in words if w not in STOP_WORDS and len(w) > 4]
        wc = Counter(filtered)
        file_word_counts[source_path] = wc
        for w in wc:
            doc_word_freq[w] += 1

    for source_path, wc in file_word_counts.items():
        total = sum(wc.values())
        if total == 0:
            continue
        scored = {}
        for word, count in wc.items():
            tf = count / total
            idf = math.log((num_docs + 1) / (doc_word_freq[word] + 1)) + 1
            scored[word] = tf * idf
        top_words = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:6]
        for word, score in top_words:
            register_entity(word.capitalize(), "concept", source_path)

    # ══════════════════════════════════════════
    # LAYER 3: Create File Nodes
    # ══════════════════════════════════════════
    file_idx = 0
    for source_path, metas in source_chunks.items():
        fname = os.path.basename(source_path)
        ext = os.path.splitext(fname)[1].lower()
        base_name = os.path.splitext(fname)[0]

        node_type = "document"
        if ext in ['.png','.jpg','.jpeg','.gif','.webp','.bmp']:
            node_type = "image"
        elif ext in ['.mp3','.wav','.m4a','.ogg','.flac']:
            node_type = "audio"

        file_size = os.path.getsize(source_path) if os.path.exists(source_path) else 0
        file_node_id = f"file-{file_idx}"
        file_nodes[source_path] = file_node_id

        nodes.append({
            "id": file_node_id, "label": base_name, "type": node_type,
            "size": max(28, min(45, 22 + int(math.log(max(file_size, 1)) * 1.5))),
            "description": f"{fname} ({len(metas)} chunks, {file_size/1024:.1f} KB)",
            "status": "indexed", "fileType": ext[1:].upper() if ext else ""
        })
        add_edge("hub-knowledge-base", file_node_id, 0.9, "contains")
        file_idx += 1

    # ══════════════════════════════════════════
    # LAYER 4: Create Entity Nodes + Edges
    # ══════════════════════════════════════════
    category_to_type = {
        "person_or_org": "entity", "date": "entity", "email": "entity",
        "url": "entity", "metric": "entity", "technology": "entity",
        "acronym": "entity", "keyphrase": "concept", "concept": "concept"
    }
    category_icons = {
        "person_or_org": "👤", "date": "📅", "email": "📧", "url": "🔗",
        "metric": "📊", "technology": "⚙️", "acronym": "🏷️",
        "keyphrase": "💡", "concept": "🧠"
    }

    for key, info in entity_registry.items():
        node_type = category_to_type.get(info["category"], "concept")
        icon = category_icons.get(info["category"], "")
        nodes.append({
            "id": info["id"],
            "label": info["label"],
            "type": node_type,
            "size": min(35, 12 + info["count"] * 3),
            "description": f"{icon} {info['category'].replace('_', ' ').title()}: {info['label']} (in {len(info['sources'])} file(s))",
            "entityType": info["category"]
        })
        # Link entity to source files
        for src_path in info["sources"]:
            if src_path in file_nodes:
                add_edge(file_nodes[src_path], info["id"],
                         min(1.0, 0.3 + info["count"] * 0.1), "has_entity")

    # ══════════════════════════════════════════
    # LAYER 5: Co-occurrence Edges (entities in same sentence)
    # ══════════════════════════════════════════
    for source_path, sentences in file_sentences.items():
        for sentence in sentences[:30]:  # limit to 30 sentences per file
            sent_lower = sentence.lower()
            found_entities = []
            for key, info in entity_registry.items():
                if key in sent_lower:
                    found_entities.append(info["id"])
            # Connect all pairs found in the same sentence
            for i in range(len(found_entities)):
                for j in range(i+1, min(len(found_entities), i+4)):
                    add_edge(found_entities[i], found_entities[j], 0.5,
                             "related_to", "co-occurs")

    # ══════════════════════════════════════════
    # LAYER 6: Document Similarity (from embeddings)
    # ══════════════════════════════════════════
    if text_vector_store is not None and text_vector_store.ntotal > 0:
        file_paths = list(file_nodes.keys())
        # Get average embedding per file
        file_avg_embeddings = {}
        for source_path in file_paths:
            indices_for_file = [i for i, m in enumerate(text_metadata)
                                if m.get("source_path") == source_path and i < text_vector_store.ntotal]
            if indices_for_file:
                try:
                    vecs = np.array([text_vector_store.reconstruct(int(idx)) for idx in indices_for_file[:10]])
                    avg_vec = vecs.mean(axis=0)
                    norm = np.linalg.norm(avg_vec)
                    if norm > 0:
                        file_avg_embeddings[source_path] = avg_vec / norm
                except Exception:
                    pass

        # Compute cosine similarity between file pairs
        paths_with_emb = list(file_avg_embeddings.keys())
        for i in range(len(paths_with_emb)):
            for j in range(i+1, len(paths_with_emb)):
                sim = float(np.dot(file_avg_embeddings[paths_with_emb[i]],
                                   file_avg_embeddings[paths_with_emb[j]]))
                if sim > 0.5:  # Only connect similar files
                    add_edge(file_nodes[paths_with_emb[i]],
                             file_nodes[paths_with_emb[j]],
                             sim, "related_to",
                             f"similar ({sim:.0%})")

    # ── Finalize ──
    types = set(n["type"] for n in nodes)
    entity_count = len(entity_registry)
    categories = Counter(info["category"] for info in entity_registry.values())

    with kg_lock:
        knowledge_graph_data = {
            "nodes": nodes,
            "edges": edges,
            "clusters": len(types),
            "stats": {
                "entities": entity_count,
                "categories": dict(categories),
                "files": len(file_nodes)
            }
        }

    with ingest_lock:
        ingest_status["kg_entities_found"] = entity_count

    print(f"Knowledge graph built: {len(nodes)} nodes, {len(edges)} edges, "
          f"{entity_count} entities across {len(categories)} categories")


# --- API Schemas ---
class ChatRequest(BaseModel):
    query: str

# --- API Endpoints ---

@app.post("/save-file")
async def save_file(file: UploadFile = File(...)):
    """Save a file to the uploads directory (no ingestion/embedding).
    Used by Digitize → Ingest to RAG when models may be unloaded."""
    try:
        file_name = file.filename or f"digitized_{id(file)}.txt"
        saved_path = os.path.join(UPLOAD_DIR, file_name)
        contents = await file.read()
        with open(saved_path, "wb") as f_out:
            f_out.write(contents)
        return {
            "status": "ok",
            "filename": file_name,
            "path": saved_path,
            "size": len(contents),
            "message": f"File '{file_name}' saved. Switch to Chat tab and click 'Index Knowledge Base' to ingest."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...), reset_db: str = Form(default="true")):
    global text_vector_store, text_metadata, knowledge_graph_data

    if not files:
        return {"status": "error", "message": "Please upload files first."}

    # Check if already running
    with ingest_lock:
        if ingest_status["is_running"]:
            return {"status": "error", "message": "Ingestion already in progress. Please wait."}

    if reset_db.lower() == "true":
        if text_vector_store is not None and hasattr(text_vector_store, 'reset'):
            text_vector_store.reset()
        text_metadata = []
        with kg_lock:
            knowledge_graph_data = {"nodes": [], "edges": [], "clusters": 0}

    # Save all files to disk first (must happen in the async context)
    saved_files = []
    for file_obj in files:
        file_name = file_obj.filename or f"temp_file_{id(file_obj)}"
        saved_path = os.path.join(UPLOAD_DIR, file_name)
        with open(saved_path, "wb") as f_out:
            f_out.write(await file_obj.read())
        saved_files.append({"name": file_name, "path": saved_path})

    # Update status
    with ingest_lock:
        ingest_status["is_running"] = True
        ingest_status["total_files"] = len(saved_files)
        ingest_status["processed_files"] = 0
        ingest_status["current_file"] = ""
        ingest_status["phase"] = "embedding"
        ingest_status["message"] = "Starting ingestion..."
        ingest_status["kg_entities_found"] = 0

    # Launch background thread for embedding + KG extraction
    def background_ingest(saved_files_list, should_reset):
        global text_vector_store, text_metadata
        try:
            all_text_chunks = []
            processed = 0

            for file_info in saved_files_list:
                file_name = file_info["name"]
                saved_path = file_info["path"]

                with ingest_lock:
                    ingest_status["current_file"] = file_name
                    ingest_status["message"] = f"Processing {file_name}..."

                try:
                    text = ""
                    lower = saved_path.lower()

                    if lower.endswith((".png", ".jpg", ".jpeg")):
                        # Direct multimodal embedding via Qwen3-VL-Embedding-2B
                        try:
                            print(f"DEBUG: Embedding image directly: {file_name}")
                            img = Image.open(saved_path).convert("RGB")
                            img_embedding = model_mgr.embedding_model.encode(img, convert_to_tensor=True).float().cpu().numpy()
                            img_embedding = img_embedding.reshape(1, -1)

                            if text_vector_store is None:
                                text_vector_store, _ = choose_index_type(1, text_embedding_dim)
                                if hasattr(text_vector_store, 'is_trained') and not text_vector_store.is_trained:
                                    text_vector_store = faiss.IndexFlatL2(text_embedding_dim)

                            text_vector_store.add(img_embedding)
                            text_metadata.append({
                                "text": f"[Image: {file_name}]",
                                "source_path": saved_path,
                                "is_image": True
                            })
                            processed += 1
                        except Exception as img_error:
                            print(f"ERROR: Could not embed image {file_name}: {img_error}")
                        continue

                    elif lower.endswith((".mp3", ".wav", ".m4a")):
                        try:
                            if model_mgr.whisper_model is not None:
                                segments, _ = model_mgr.whisper_model.transcribe(saved_path, beam_size=5)
                                text = "".join(seg.text for seg in segments)
                            else:
                                text = ""
                                print(f"Whisper model not loaded, skipping audio: {file_name}")
                        except Exception as e:
                            print(f"Error transcribing audio {file_name}: {e}")
                            text = ""

                    elif lower.endswith('.pdf'):
                        try:
                            reader = pypdf.PdfReader(saved_path)
                            text = "\n".join([page.extract_text() for page in reader.pages])
                        except Exception as e:
                            print(f"Error reading PDF {file_name}: {e}")
                            text = ""

                    elif lower.endswith('.docx'):
                        try:
                            doc = docx.Document(saved_path)
                            text = "\n".join([para.text for para in doc.paragraphs])
                        except Exception as e:
                            print(f"Error reading DOCX {file_name}: {e}")
                            text = ""

                    elif lower.endswith('.csv'):
                        try:
                            df = pd.read_csv(saved_path)
                            text = df.to_string()
                        except Exception as e:
                            print(f"Error reading CSV {file_name}: {e}")
                            text = ""

                    elif lower.endswith('.txt'):
                        try:
                            with open(saved_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                        except Exception as e:
                            print(f"Error reading TXT {file_name}: {e}")
                            text = ""
                    else:
                        continue

                    if text:
                        chunks = iterative_chunking(text)
                        for chunk in chunks:
                            all_text_chunks.append({"text": chunk, "source_path": saved_path})
                        processed += 1

                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

                with ingest_lock:
                    ingest_status["processed_files"] = processed

            # Batch embed text chunks
            if all_text_chunks:
                with ingest_lock:
                    ingest_status["phase"] = "embedding"
                    ingest_status["message"] = f"Embedding {len(all_text_chunks)} text chunks..."

                texts = [c['text'] for c in all_text_chunks]
                text_embeddings = model_mgr.embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True).float().cpu().numpy()

                if text_vector_store is None or (should_reset and len(text_metadata) == 0):
                    text_vector_store, _ = choose_index_type(len(all_text_chunks), text_embedding_dim)

                if hasattr(text_vector_store, 'is_trained') and not text_vector_store.is_trained:
                    training_success = safe_train_index(text_vector_store, text_embeddings)
                    if not training_success:
                        text_vector_store = faiss.IndexFlatL2(text_embedding_dim)

                if text_vector_store is not None:
                    text_vector_store.add(text_embeddings)
                    text_metadata.extend(all_text_chunks)
                    print(f"Added {len(all_text_chunks)} text chunks to the vector store.")

            # Build knowledge graph (LLM entity extraction)
            with ingest_lock:
                ingest_status["phase"] = "extracting_kg"
                ingest_status["message"] = "Building knowledge graph..."

            build_knowledge_graph_from_metadata()

            # Done
            with ingest_lock:
                ingest_status["phase"] = "done"
                ingest_status["message"] = f"Done! {processed} files processed, {len(text_metadata)} chunks indexed."
                ingest_status["is_running"] = False

        except Exception as e:
            print(f"Background ingestion error: {e}")
            import traceback
            traceback.print_exc()
            with ingest_lock:
                ingest_status["phase"] = "error"
                ingest_status["message"] = f"Error: {e}"
                ingest_status["is_running"] = False

    thread = threading.Thread(
        target=background_ingest,
        args=(saved_files, reset_db.lower() == "true"),
        daemon=True
    )
    thread.start()

    return {
        "status": "ok",
        "message": f"Ingestion started for {len(saved_files)} files. Processing in background.",
        "text_chunks": len(text_metadata)
    }

@app.get("/ingest-status")
def get_ingest_status():
    """Return current ingestion progress for frontend polling."""
    with ingest_lock:
        return {**ingest_status}

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
                actual_filename = os.path.basename(source_path)
                sources_list.append(f'<div class="source-item" data-filename="{actual_filename}"><p class="source-name">{clean_name}</p></div>')
                added_sources.add(source_path)
        sources_str = "\n".join(sources_list)

        messages = build_chat_messages(
            query=payload.query,
            contexts=final_contexts,
            system_prompt=f'''
                You are a Multimodal RAG agent. Generate responses in VALID HTML format only.

**STRICT REQUIREMENTS:**
1. Respond with COMPLETE HTML document structure
2. Use proper HTML tags: <p>, <ul>, <li>, <strong>, <em>, etc.
3. Include citations as [1], [2], etc. within the HTML content
4. End with --%Sources%-- separator
5. Follow with proper sources section in HTML format
6. If images are provided, describe and reference them in your response

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
        )

        # Append sources info as a separate text block in the user message
        # (the images + query are already in the message from build_chat_messages)
        messages[-1]["content"].append({
            "type": "text",
            "text": f"\n\n--%Sources%--\n<div class=\"sources-section\">\n<h3>Sources</h3>\n{sources_str}\n</div>"
        })

        final_answer = call_lm_studio_text(messages, max_tokens=1500, temperature=0.7)

        return {
            "answer": final_answer
        }

    except Exception as e:
        return {"answer": f"Unexpected error: {e}"}

@app.post("/chat-stream")
async def chat_stream(payload: ChatRequest):
    """Streaming chat endpoint - returns Server-Sent Events with token-by-token response."""
    contexts = retrieve_multimodal_context(payload.query)

    if not contexts:
        async def empty_stream():
            yield f"data: {json.dumps({'token': 'Knowledge base is empty. Please ingest files first.'})}\n\n"
            yield f"data: [DONE]\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })

    try:
        final_contexts = contexts
        context_str = "\n".join([ctx['text'] for ctx in final_contexts])
        unique_sources = list(set([ctx["source"] for ctx in final_contexts]))

        # Create citation mapping
        source_to_citation = {}
        citation_counter = 1
        for ctx in final_contexts:
            source_path = ctx["source"]
            if source_path not in source_to_citation:
                source_to_citation[source_path] = citation_counter
                citation_counter += 1

        # Build sources HTML
        sources_list = []
        added_sources = set()
        for source_path in unique_sources:
            if source_path not in added_sources:
                citation_number = source_to_citation[source_path]
                clean_name = os.path.splitext(os.path.basename(source_path))[0]
                actual_filename = os.path.basename(source_path)
                sources_list.append(f'<div class="source-item" data-filename="{actual_filename}"><p class="source-name">{clean_name}</p></div>')
                added_sources.add(source_path)
        sources_str = "\n".join(sources_list)

        # Build the sources metadata to send at the end
        sources_metadata = []
        added_meta = set()
        for source_path in unique_sources:
            if source_path not in added_meta:
                citation_number = source_to_citation[source_path]
                clean_name = os.path.splitext(os.path.basename(source_path))[0]
                actual_filename = os.path.basename(source_path)
                sources_metadata.append({
                    "key": citation_number,
                    "name": clean_name,
                    "filename": actual_filename
                })
                added_meta.add(source_path)

        messages = build_chat_messages(
            query=payload.query,
            contexts=final_contexts,
            system_prompt=f'''
                You are a Multimodal RAG agent. Generate responses in VALID HTML format only.

**STRICT REQUIREMENTS:**
1. Respond with COMPLETE HTML document structure
2. Use proper HTML tags: <p>, <ul>, <li>, <strong>, <em>, etc.
3. Include citations as [1], [2], etc. within the HTML content
4. Do NOT include the sources section - it will be added automatically
5. If images are provided, describe and reference them in your response

**SOURCE SELECTION CRITERIA:**
- ONLY include sources that directly contribute to answering the user's query
- Use citation numbers [1], [2], etc. to reference sources

**RESPONSE FORMAT:**
- Output ONLY the HTML content with citations
- Do NOT include --%Sources%-- or any sources section
- Keep it concise and well-formatted

**EXAMPLE:**
<p>This is properly formatted HTML content [1].</p>
<p>More content here with another citation [2].</p>
<ul>
    <li>List items work too [1]</li>
</ul>
'''
        )

        def generate():
            # Accumulate the full response to check which sources were actually cited
            accumulated_text = []

            # Stream the LLM tokens
            for chunk in stream_lm_studio_text(messages, max_tokens=1500, temperature=0.7):
                # Filter out any stray [DONE] from inner stream
                if '[DONE]' in chunk:
                    continue
                yield chunk

                # Extract token text from the SSE chunk for accumulation
                try:
                    if chunk.startswith('data: '):
                        token_data = json.loads(chunk[6:].strip())
                        if 'token' in token_data:
                            accumulated_text.append(token_data['token'])
                except (json.JSONDecodeError, KeyError):
                    pass

            # After streaming is done, check which citations were actually used
            full_response = ''.join(accumulated_text)
            cited_numbers = set(re.findall(r'\[(\d+)\]', full_response))

            # Only send sources that were actually cited by the LLM
            filtered_sources = [
                src for src in sources_metadata
                if str(src['key']) in cited_numbers
            ]

            # If no citations found but we have sources, send all (fallback)
            if not filtered_sources and sources_metadata:
                filtered_sources = sources_metadata

            sources_event = json.dumps({"sources": filtered_sources})
            yield f"data: {sources_event}\n\n"
            yield f"data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    except Exception as e:
        async def error_stream():
            yield f"data: {json.dumps({'token': f'Unexpected error: {e}'})}\n\n"
            yield f"data: [DONE]\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, "_temp_chat_audio")
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # Use in-process Whisper model
        if model_mgr.whisper_model is None:
            return {"answer": "Whisper model not loaded. Switch to RAG mode first."}
        try:
            segments, _ = model_mgr.whisper_model.transcribe(temp_path, beam_size=5)
            transcribed_query = "".join(seg.text for seg in segments).strip()
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

        # Use in-process Whisper model
        if model_mgr.whisper_model is None:
            return {"error": "Whisper model not loaded. Switch to RAG mode first."}
        try:
            segments, _ = model_mgr.whisper_model.transcribe(temp_path, beam_size=5)
            transcription = "".join(seg.text for seg in segments)
            return {"transcription": transcription}
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
        kokoro_path = os.path.join(MODELS_DIR, 'kokoro-82m')
        pipeline = KPipeline(lang_code='a', repo_id=kokoro_path, device=DEVICE)
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

@app.get('/files')
def list_files():
    """List all files in the uploads directory"""
    files = []
    if os.path.exists(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.isfile(fpath) and not fname.startswith('_temp'):
                ext = os.path.splitext(fname)[1].lower()
                file_type = 'document'
                if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                    file_type = 'image'
                elif ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac']:
                    file_type = 'audio'
                elif ext in ['.mp4', '.webm', '.avi', '.mov']:
                    file_type = 'video'
                files.append({
                    "name": fname,
                    "size": os.path.getsize(fpath),
                    "type": file_type
                })
    return {"files": files}

@app.get('/files/{filename}')
def get_file(filename: str):
    """Serve an uploaded file for inline preview in the browser"""
    # Sanitize the filename to prevent path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File '{safe_filename}' not found")

    # Determine media type based on extension
    ext = os.path.splitext(safe_filename)[1].lower()
    media_types = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.txt': 'text/plain',
        '.csv': 'text/csv',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.m4a': 'audio/mp4',
        '.ogg': 'audio/ogg',
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
    }
    media_type = media_types.get(ext, 'application/octet-stream')

    # Use inline content-disposition so the browser displays the file
    # instead of downloading it. Do NOT pass filename= to FileResponse
    # because that sets Content-Disposition: attachment which forces download.
    response = FileResponse(
        file_path,
        media_type=media_type,
    )
    # Set inline disposition explicitly
    response.headers["Content-Disposition"] = f'inline; filename="{safe_filename}"'
    return response

@app.get('/files/{filename}/preview')
def preview_file(filename: str):
    """Preview a file in the browser. Converts DOCX/DOC to HTML for in-browser viewing."""
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File '{safe_filename}' not found")

    ext = os.path.splitext(safe_filename)[1].lower()

    # For DOCX files, convert to PDF using MS Word for exact rendering
    if ext in ['.docx', '.doc']:
        try:
            import subprocess, sys

            # Create a PDF path next to the original file
            pdf_filename = os.path.splitext(safe_filename)[0] + '_preview.pdf'
            pdf_path = os.path.join(UPLOAD_DIR, pdf_filename)
            abs_file_path = os.path.abspath(file_path)
            abs_pdf_path = os.path.abspath(pdf_path)

            # Check if a cached PDF already exists and is newer than the DOCX
            needs_conversion = True
            if os.path.exists(abs_pdf_path):
                pdf_mtime = os.path.getmtime(abs_pdf_path)
                docx_mtime = os.path.getmtime(abs_file_path)
                if pdf_mtime > docx_mtime:
                    needs_conversion = False
                    print(f"Using cached PDF for {safe_filename}")

            if needs_conversion:
                print(f"Converting {safe_filename} to PDF using MS Word...")
                # Run in a subprocess because docx2pdf uses COM automation
                # which doesn't work in uvicorn's async thread context
                conv_script = (
                    f'from docx2pdf import convert\n'
                    f'try:\n'
                    f'    convert(r"{abs_file_path}", r"{abs_pdf_path}")\n'
                    f'except AttributeError:\n'
                    f'    pass  # Known docx2pdf bug at Word.Application.Quit\n'
                )
                result = subprocess.run(
                    [sys.executable, '-c', conv_script],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode != 0:
                    print(f"docx2pdf subprocess error: {result.stderr}")

            # Serve the PDF if it was created
            if os.path.exists(abs_pdf_path) and os.path.getsize(abs_pdf_path) > 0:
                response = FileResponse(
                    abs_pdf_path,
                    media_type='application/pdf',
                )
                response.headers["Content-Disposition"] = f'inline; filename="{pdf_filename}"'
                return response
            else:
                raise Exception("PDF conversion produced no output file")

        except Exception as e:
            print(f"docx2pdf conversion failed for {safe_filename}: {e}")
            print("Falling back to mammoth HTML conversion...")

            # Fallback: use mammoth for HTML conversion
            try:
                with open(file_path, 'rb') as docx_file:
                    result = mammoth.convert_to_html(docx_file)
                    html_content = result.value

                full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_filename} — Document Preview</title>

    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Inter', system-ui, sans-serif; background: linear-gradient(180deg, #071028, #0f1724); color: #e2e8f0; line-height: 1.7; padding: 0; min-height: 100vh; }}
        .doc-header {{ position: sticky; top: 0; z-index: 10; background: rgba(7,16,40,0.92); backdrop-filter: blur(16px); border-bottom: 1px solid rgba(255,255,255,0.06); padding: 16px 32px; display: flex; align-items: center; gap: 12px; }}
        .doc-icon {{ width: 36px; height: 36px; border-radius: 8px; background: linear-gradient(135deg, #6366f1, #8b5cf6); display: grid; place-items: center; font-size: 16px; }}
        .doc-title {{ font-size: 15px; font-weight: 700; color: #f1f5f9; }}
        .doc-badge {{ font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 6px; background: rgba(245,158,11,0.15); color: #fcd34d; text-transform: uppercase; }}
        .doc-content {{ max-width: 820px; margin: 40px auto; padding: 48px 56px; background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }}
        .doc-content h1 {{ font-size: 28px; font-weight: 800; color: #f8fafc; margin: 28px 0 16px; }}
        .doc-content h2 {{ font-size: 22px; font-weight: 700; color: #e2e8f0; margin: 24px 0 12px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px; }}
        .doc-content h3 {{ font-size: 18px; font-weight: 700; color: #cbd5e1; margin: 20px 0 10px; }}
        .doc-content p {{ margin: 0 0 14px; font-size: 15px; color: #cbd5e1; }}
        .doc-content ul, .doc-content ol {{ margin: 12px 0 16px; padding-left: 24px; }}
        .doc-content li {{ margin-bottom: 6px; font-size: 15px; color: #cbd5e1; }}
        .doc-content strong {{ color: #f1f5f9; }}
        .doc-content em {{ color: #a5b4fc; }}
        .doc-content table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }}
        .doc-content th {{ background: rgba(99,102,241,0.1); padding: 10px 14px; text-align: left; font-weight: 700; color: #e2e8f0; border: 1px solid rgba(255,255,255,0.08); }}
        .doc-content td {{ padding: 10px 14px; border: 1px solid rgba(255,255,255,0.06); color: #cbd5e1; }}
        .doc-content img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 16px 0; }}
    </style>
</head>
<body>
    <div class="doc-header"><div class="doc-icon">📄</div><span class="doc-title">{safe_filename}</span><span class="doc-badge">HTML Fallback</span></div>
    <div class="doc-content">{html_content}</div>
</body>
</html>"""
                return HTMLResponse(content=full_html)
            except Exception as fallback_err:
                raise HTTPException(status_code=500, detail=f"Failed to preview document: {str(e)}. Fallback also failed: {str(fallback_err)}")

    # For TXT files, render as HTML with styling
    elif ext == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text_content = f.read()

            # Escape HTML and convert line breaks
            import html
            escaped = html.escape(text_content)
            html_lines = escaped.replace('\n', '<br>')

            full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_filename} — Text Preview</title>

    <style>
        body {{ font-family: 'JetBrains Mono', monospace; background: linear-gradient(180deg, #071028, #0f1724); color: #cbd5e1; padding: 0; min-height: 100vh; margin: 0; }}
        .doc-header {{ position: sticky; top: 0; z-index: 10; background: rgba(7,16,40,0.92); backdrop-filter: blur(16px); border-bottom: 1px solid rgba(255,255,255,0.06); padding: 16px 32px; display: flex; align-items: center; gap: 12px; }}
        .doc-icon {{ width: 36px; height: 36px; border-radius: 8px; background: linear-gradient(135deg, #10b981, #34d399); display: grid; place-items: center; font-size: 16px; }}
        .doc-title {{ font-family: 'Inter', sans-serif; font-size: 15px; font-weight: 700; color: #f1f5f9; }}
        .doc-badge {{ font-family: 'Inter', sans-serif; font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 6px; background: rgba(16,185,129,0.15); color: #6ee7b7; text-transform: uppercase; }}
        .doc-content {{ max-width: 900px; margin: 40px auto; padding: 36px 44px; background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; font-size: 14px; line-height: 1.8; white-space: pre-wrap; word-wrap: break-word; }}
    </style>
</head>
<body>
    <div class="doc-header"><div class="doc-icon">📝</div><span class="doc-title">{safe_filename}</span><span class="doc-badge">TXT</span></div>
    <div class="doc-content">{html_lines}</div>
</body>
</html>"""
            return HTMLResponse(content=full_html)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to preview text file: {str(e)}")

    # For CSV files, render as a styled HTML table
    elif ext == '.csv':
        try:
            df = pd.read_csv(file_path)
            table_html = df.to_html(index=False, classes='csv-table', border=0)

            full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_filename} — CSV Preview</title>

    <style>
        body {{ font-family: 'Inter', sans-serif; background: linear-gradient(180deg, #071028, #0f1724); color: #e2e8f0; padding: 0; min-height: 100vh; margin: 0; }}
        .doc-header {{ position: sticky; top: 0; z-index: 10; background: rgba(7,16,40,0.92); backdrop-filter: blur(16px); border-bottom: 1px solid rgba(255,255,255,0.06); padding: 16px 32px; display: flex; align-items: center; gap: 12px; }}
        .doc-icon {{ width: 36px; height: 36px; border-radius: 8px; background: linear-gradient(135deg, #f59e0b, #fbbf24); display: grid; place-items: center; font-size: 16px; }}
        .doc-title {{ font-size: 15px; font-weight: 700; color: #f1f5f9; }}
        .doc-badge {{ font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 6px; background: rgba(245,158,11,0.15); color: #fcd34d; text-transform: uppercase; }}
        .doc-content {{ max-width: 1100px; margin: 40px auto; padding: 32px; background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; overflow-x: auto; }}
        .csv-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .csv-table th {{ background: rgba(99,102,241,0.12); padding: 10px 14px; text-align: left; font-weight: 700; color: #e2e8f0; border: 1px solid rgba(255,255,255,0.08); position: sticky; top: 0; }}
        .csv-table td {{ padding: 8px 14px; border: 1px solid rgba(255,255,255,0.05); color: #cbd5e1; }}
        .csv-table tr:nth-child(even) td {{ background: rgba(255,255,255,0.015); }}
        .csv-table tr:hover td {{ background: rgba(99,102,241,0.06); }}
    </style>
</head>
<body>
    <div class="doc-header"><div class="doc-icon">📊</div><span class="doc-title">{safe_filename}</span><span class="doc-badge">CSV</span></div>
    <div class="doc-content">{table_html}</div>
</body>
</html>"""
            return HTMLResponse(content=full_html)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to preview CSV file: {str(e)}")

    # For all other file types (images, PDFs, audio), redirect to the direct file endpoint
    else:
        return RedirectResponse(url=f"/files/{safe_filename}")

@app.post('/reset')
def reset():
    global text_vector_store, text_metadata, knowledge_graph_data
    if text_vector_store is not None and hasattr(text_vector_store, 'reset'):
        text_vector_store.reset()
    text_metadata = []
    with kg_lock:
        knowledge_graph_data = {"nodes": [], "edges": [], "clusters": 0}
    with ingest_lock:
        ingest_status["is_running"] = False
        ingest_status["phase"] = "idle"
        ingest_status["message"] = ""
        ingest_status["kg_entities_found"] = 0
    return {"status": "ok", "message": "All knowledge bases and knowledge graph cleared."}

@app.get('/knowledge-graph')
def knowledge_graph():
    """Return the LLM-extracted knowledge graph built during ingestion."""
    with kg_lock:
        return knowledge_graph_data

# --- GPU Mode Switching (RAG ↔ Digitize) ---
@app.post('/mode/digitize')
def switch_to_digitize():
    """Unload RAG models from GPU to free VRAM for PaddleOCR."""
    result = model_mgr.switch_to_digitize()
    return result

@app.post('/mode/rag')
def switch_to_rag():
    """Reload RAG models onto GPU."""
    result = model_mgr.switch_to_rag()
    return result

@app.get('/mode/status')
def mode_status():
    """Get current mode and model status."""
    return model_mgr.get_status()

# --- PaddleOCR Proxy ---
PADDLE_OCR_URL = "http://localhost:8010"

@app.post('/ocr/upload')
async def ocr_upload(file: UploadFile = File(...)):
    """
    Proxy endpoint: forwards an uploaded file to the PaddleOCR PPStructure service.
    Returns structured blocks, annotated image, and DOCX download info.
    """
    try:
        file_bytes = await file.read()
        filename = file.filename or "upload.png"

        # Forward to PPStructure endpoint for full layout analysis
        ocr_response = requests.post(
            f"{PADDLE_OCR_URL}/ocr/structure",
            files={"file": (filename, file_bytes, file.content_type or "image/png")},
            timeout=600
        )

        if ocr_response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"OCR service returned {ocr_response.status_code}")

        return ocr_response.json()

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="PaddleOCR service is not running. Please start it on port 8010."
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="OCR processing timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.get('/ocr/download/{filename}')
def ocr_download(filename: str):
    """Proxy DOCX download from the OCR service."""
    try:
        r = requests.get(f"{PADDLE_OCR_URL}/ocr/download/{filename}", timeout=30, stream=True)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail="File not found")

        from starlette.responses import Response
        return Response(
            content=r.content,
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="OCR service not reachable")


@app.get('/ocr/health')
def ocr_health():
    """Check if the PaddleOCR microservice is reachable."""
    try:
        r = requests.get(f"{PADDLE_OCR_URL}/health", timeout=5)
        return {"status": "ok", "ocr_service": r.json()}
    except Exception:
        return {"status": "unavailable", "message": "PaddleOCR service not reachable on port 8010"}

if __name__ == '__main__':
    import uvicorn
    print("Starting main server on port 8000...")
    uvicorn.run(app, host='0.0.0.0', port=8000)
