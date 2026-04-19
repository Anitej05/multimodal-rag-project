"""
Save all ML models to backend/models/ for fully offline operation.

Expected directory structure after running:
  backend/models/
    qwen3-vl-embedding-2b/     <- Qwen/Qwen3-VL-Embedding-2B
    cross-encoder-minilm/      <- cross-encoder/ms-marco-MiniLM-L-6-v2
    faster-whisper-base/       <- Systran/faster-whisper-base
    kokoro-82m/                <- hexgrad/Kokoro-82M

Usage (run inside multimodal-rag conda env):
  python save_models_locally.py
"""
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Qwen3-VL-Embedding-2B ──
emb_dir = os.path.join(MODELS_DIR, "qwen3-vl-embedding-2b")
if not os.path.exists(os.path.join(emb_dir, "config.json")):
    print("[1/4] Saving Qwen3-VL-Embedding-2B...")
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('Qwen/Qwen3-VL-Embedding-2B', model_kwargs={"torch_dtype": "bfloat16"})
    m.save(emb_dir)
    del m
    print(f"  Done -> {emb_dir}")
else:
    print("[1/4] Qwen3-VL-Embedding-2B already exists, skipping.")

# ── 2. CrossEncoder ──
ce_dir = os.path.join(MODELS_DIR, "cross-encoder-minilm")
if not os.path.exists(os.path.join(ce_dir, "config.json")):
    print("[2/4] Saving CrossEncoder ms-marco-MiniLM-L-6-v2...")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    ce.save(ce_dir)
    del ce
    print(f"  Done -> {ce_dir}")
else:
    print("[2/4] CrossEncoder already exists, skipping.")

# ── 3. Faster-Whisper ──
whisper_dir = os.path.join(MODELS_DIR, "faster-whisper-base")
if not os.path.exists(os.path.join(whisper_dir, "model.bin")):
    print("[3/4] Saving faster-whisper-base...")
    from huggingface_hub import snapshot_download
    snapshot_download("Systran/faster-whisper-base", local_dir=whisper_dir)
    print(f"  Done -> {whisper_dir}")
else:
    print("[3/4] faster-whisper-base already exists, skipping.")

# ── 4. Kokoro TTS ──
kokoro_dir = os.path.join(MODELS_DIR, "kokoro-82m")
if not os.path.exists(kokoro_dir) or len(os.listdir(kokoro_dir)) == 0:
    print("[4/4] Saving Kokoro-82M...")
    from huggingface_hub import snapshot_download
    snapshot_download("hexgrad/Kokoro-82M", local_dir=kokoro_dir)
    print(f"  Done -> {kokoro_dir}")
else:
    print("[4/4] Kokoro-82M already exists, skipping.")

print("\nAll models saved to backend/models/")
