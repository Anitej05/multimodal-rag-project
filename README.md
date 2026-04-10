# 🧠 Multimodal RAG Assistant

> A full-stack **Retrieval-Augmented Generation (RAG)** system that ingests **text, images, and audio** documents, indexes them into a FAISS vector database, and answers user queries via a streaming AI chat interface — powered by **LM Studio** for local LLM inference, with an interactive **Knowledge Graph** visualization.

---

## 📑 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Data Flow Pipeline](#-data-flow-pipeline)
- [Tech Stack](#-tech-stack)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [RAG Pipeline — How It Works](#-rag-pipeline--how-it-works)
- [Knowledge Graph](#-knowledge-graph)
- [Setup & Installation](#-setup--installation)
- [API Reference](#-api-reference)

---

## 🏗 Architecture Overview

```
+------------------------------------------------------------------------+
|                         FRONTEND (React 18)                            |
|                                                                        |
|   +--------------+   +----------------+   +--------------------------+ |
|   | Knowledge    |   | AI Chat        |   | Knowledge Graph          | |
|   | Base Panel   |   | Interface      |   | Visualization            | |
|   |              |   |                |   |                          | |
|   | - Upload     |   | - Streaming    |   | - Cytoscape.js           | |
|   | - Index      |   | - Citations    |   | - Force-Directed Layout  | |
|   | - Preview    |   | - Voice I/O    |   | - Search & Filter        | |
|   | - Stats      |   | - Read Aloud   |   | - Interactive Nodes      | |
|   +--------------+   +----------------+   +--------------------------+ |
|                                                                        |
|                        api.js (Fetch + SSE)                            |
+-----------------------------------+------------------------------------+
                                    |  HTTP / SSE
+-----------------------------------v------------------------------------+
|                      BACKEND (FastAPI - Python)                        |
|                                                                        |
|   +----------------------------------------------------------------+   |
|   |                       API Endpoints                            |   |
|   |  /ingest  /chat-stream  /chat  /knowledge-graph               |   |
|   |  /transcribe  /generate_audio  /files  /reset                 |   |
|   +-------------------------------+--------------------------------+   |
|                                   |                                    |
|   +-------------------------------v--------------------------------+   |
|   |                     Core RAG Pipeline                          |   |
|   |                                                                |   |
|   |  +--------------+  +------------+  +----------+  +----------+ |   |
|   |  | Sentence     |  | FAISS      |  | Cross-   |  | LM Studio| |   |
|   |  | Transformer  |  | Vector DB  |  | Encoder  |  | (LLM)    | |   |
|   |  | MiniLM-L6-v2 |  | L2 / IVF   |  | Re-Rank  |  | Qwen3-4B | |   |
|   |  +--------------+  +------------+  +----------+  +----------+ |   |
|   +----------------------------------------------------------------+   |
|                                                                        |
|   +--------------+   +--------------+   +----------------------------+ |
|   | Whisper      |   | EasyOCR      |   | Kokoro TTS                 | |
|   | (ASR, 5001)  |   | (Image OCR)  |   | (Text-to-Speech)           | |
|   +--------------+   +--------------+   +----------------------------+ |
+------------------------------------------------------------------------+
                                    |  HTTP
+-----------------------------------v------------------------------------+
|                     LM STUDIO (Local LLM Server)                       |
|                                                                        |
|    Text Model  : qwen/qwen3-4b              (Port 1234)               |
|    Vision Model: smolvlm2-500m-video-instruct                         |
|    API         : OpenAI-compatible /v1/chat/completions               |
+------------------------------------------------------------------------+
```

---

## 🔄 Data Flow Pipeline

### 1. Document Ingestion

```
User uploads files --> FileUpload.jsx --> POST /ingest
                                              |
                    +-------------------------+
                    |   File Type Router      |
                    |                         |
                    |   .pdf  --> pypdf       |
                    |   .docx --> python-docx |
                    |   .txt  --> UTF-8 read  |
                    |   .csv  --> pandas      |
                    |   .png  --> Vision LLM  |
                    |   .mp3  --> Whisper ASR |
                    +------------+------------+
                                 |
                                 v
                    Chunking (400-char, natural breaks)
                                 |
                                 v
                    SentenceTransformer --> 384-dim embeddings
                                 |
                                 v
                    FAISS Vector Store (FlatL2 / IVFFlat)
```

### 2. Query & Retrieval (RAG)

```
User asks a question --> Chat.jsx --> POST /chat-stream (SSE)
                                            |
                Step 1:  Modality Detection  |  LLM classifies query --> boost factor
                Step 2:  Vector Search       |  FAISS top-k x2 candidates
                Step 3:  Cross-Encoder       |  Re-rank with ms-marco-MiniLM
                Step 4:  Score Fusion        |  60% text + 40% cross-encoder
                Step 5:  Stream Response     |  Top-5 contexts --> LLM --> SSE tokens
                                            |
                                            v
                    Chat.jsx renders tokens via requestAnimationFrame
                    --> Citations [1],[2] link to source file previews
```

### 3. Voice Interaction

```
Record  --> MediaRecorder --> POST /transcribe --> Whisper --> text --> RAG pipeline
Read    --> AI response   --> POST /generate_audio --> Kokoro TTS --> browser playback
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18, Vanilla CSS | UI with dark/light theme support |
| **Graph Viz** | Cytoscape.js + cose-bilkent | Interactive knowledge graph |
| **Backend** | FastAPI + Uvicorn | REST API + SSE streaming server |
| **LLM** | LM Studio (Qwen3-4B) | Text generation & streaming |
| **VLM** | LM Studio (SmolVLM2-500M) | Image understanding |
| **Embeddings** | SentenceTransformer (MiniLM-L6-v2) | 384-dim text embeddings |
| **Re-ranking** | CrossEncoder (ms-marco-MiniLM) | Search result re-ranking |
| **Vector DB** | FAISS (FlatL2 / IVFFlat) | Similarity search index |
| **ASR** | Faster-Whisper (base) | Speech-to-text |
| **TTS** | Kokoro TTS | Text-to-speech |
| **OCR** | EasyOCR | Image text extraction |
| **Doc Parsing** | pypdf, python-docx, pandas, mammoth | File content extraction |

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Multimodal Ingestion** | Supports PDF, DOCX, TXT, CSV, PNG/JPG, MP3/WAV — each processed by a specialized extractor |
| **Hybrid Retrieval** | Two-stage retrieval: FAISS vector search → Cross-Encoder re-ranking with modality-aware boosting |
| **Real-time Streaming** | Token-by-token SSE streaming with `requestAnimationFrame` DOM updates for smooth rendering |
| **Source Citations** | Inline clickable citations `[1]`, `[2]` linking directly to in-browser file previews |
| **Voice I/O** | Record audio queries via Whisper ASR; read AI responses aloud via Kokoro TTS |
| **Knowledge Graph** | Obsidian-style interactive graph showing documents, chunks, and extracted concepts |
| **In-Browser Previews** | Smart file preview — DOCX→PDF conversion, styled CSV tables, formatted text views |
| **Fully Local** | All inference runs locally via LM Studio; no cloud API keys required |

---

## 📁 Project Structure

```
multimodal-rag-project/
├── backend/
│   ├── main.py                 # FastAPI server — ingestion, RAG, streaming, graph
│   ├── whisper_service.py      # Whisper ASR microservice (port 5001)
│   ├── requirements.txt        # Python dependencies
│   └── uploads/                # Uploaded files
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # App shell — routing, theme, global state
│   │   ├── components/         # React UI components (Chat, KnowledgeBase, Graph, etc.)
│   │   ├── services/           # API service layer (fetch + SSE)
│   │   ├── styles/             # CSS design system (dark/light themes)
│   │   └── utils/              # Utility & helper functions
│   └── package.json
│
└── README.md
```

---

## 🔍 RAG Pipeline — How It Works

The Retrieval-Augmented Generation pipeline is the core intelligence of the system. Here's a detailed breakdown:

### Ingestion Phase

1. **File Upload** — User drops files into the drag-and-drop zone; files are sent as `FormData` to the backend.
2. **Content Extraction** — Each file type is processed by a specialized extractor:
   - **PDF**: `pypdf.PdfReader` extracts page text
   - **DOCX**: `python-docx` extracts paragraphs
   - **TXT/CSV**: Direct read / pandas parsing
   - **Images**: Sent to LM Studio's Vision model for text description
   - **Audio**: Forwarded to the Whisper microservice for transcription
3. **Chunking** — Extracted text is split into ~400-character chunks at natural break points (paragraphs → sentences → words → hard cut).
4. **Embedding** — Each chunk is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`.
5. **Indexing** — Vectors are stored in FAISS; the index type is chosen dynamically:
   - < 39 chunks → `IndexFlatL2` (brute-force, exact search)
   - ≥ 39 chunks → `IndexIVFFlat` (clustered, approximate search with `nlist = 4√N`, capped at 256)

### Retrieval Phase

1. **Modality Detection** — The LLM analyzes the user's query and determines which document modality (text, image, audio) to boost (1.0×–2.0×).
2. **Vector Search** — The query is embedded and searched against FAISS for `top_k × 2` candidates. L2 distances are normalized to similarity scores and multiplied by the modality boost.
3. **Cross-Encoder Re-ranking** — Each `(query, candidate_text)` pair is scored by `ms-marco-MiniLM-L-6-v2`. The final score combines:
   - **60%** vector similarity score
   - **40%** cross-encoder relevance score
4. **Top-5 Selection** — The highest-scoring 5 chunks are selected, along with their source file paths.

### Generation Phase

1. **Prompt Construction** — A system prompt is built with the top-5 context chunks and citation instructions.
2. **SSE Streaming** — The LLM generates a response token-by-token, streamed as Server-Sent Events. `<think>` tags are stripped on-the-fly.
3. **Citation Mapping** — After generation, the backend identifies which citations `[1]`, `[2]`, etc. were actually used and sends only the referenced sources.
4. **Frontend Rendering** — Tokens are accumulated in a ref (not React state) and pushed to the DOM via `requestAnimationFrame` for 60fps rendering. On completion, the full message is committed to React state.

---

## 🕸 Knowledge Graph

The Knowledge Graph provides an **interactive, Obsidian-inspired visualization** of all ingested documents and their relationships.

### Node Types

| Type | Color | Description |
|------|-------|-------------|
| **Hub** | 🔵 Blue | Central "Knowledge Base" node |
| **Document** | 🟣 Indigo | PDF, DOCX, TXT, CSV files |
| **Image** | 🩷 Pink | PNG, JPG image files |
| **Audio** | 🟢 Green | MP3, WAV audio files |
| **Chunk** | ⚪ Gray | Text chunks (up to 5 per file) |
| **Concept** | 🟡 Gold | Keywords extracted from content |

### Edge Types

| Type | From → To | Meaning |
|------|-----------|---------|
| **contains** | Hub → File | File belongs to the knowledge base |
| **chunk_of** | File → Chunk | Chunk was extracted from this file |
| **mentions** | File → Concept | File content mentions this concept |
| **related** | Concept → Concept | Concepts co-occur across files |

### Graph Features

- **4 Layout Options**: Force-directed (cose-bilkent), Concentric, Circle, Grid
- **Interactive**: Node hover highlights neighborhood; click opens detail panel
- **Search**: Fuzzy search across node labels and descriptions
- **Zoom Controls**: Zoom in/out, fit-to-screen
- **Stats Bar**: Live node, edge, and cluster counts
- **Concept Extraction**: Regex-based keyword extraction → top 5 concepts per file → inter-concept edges for shared sources

---

## ⚙ Setup & Installation

### Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10+ with pip |
| **Node.js** | 18+ with npm |
| **LM Studio** | Running on `localhost:1234` with `qwen/qwen3-4b` (text) and `smolvlm2-500m-video-instruct` (vision) loaded |
| **Optional** | NVIDIA GPU for CUDA acceleration; MS Word for DOCX→PDF preview |

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
# → Backend on http://localhost:8000
# → Whisper service auto-starts on port 5001
```

### Frontend

```bash
cd frontend
npm install
npm start
# → Frontend on http://localhost:3000
```

### Optional: Kokoro TTS

```bash
pip install kokoro>=0.9.2
# If not installed, Read Aloud is disabled but the app functions normally.
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Upload and index files into the knowledge base |
| `/chat-stream` | POST | Streaming RAG chat via Server-Sent Events |
| `/chat` | POST | One-shot (non-streaming) RAG chat |
| `/chat-audio` | POST | Upload audio → transcribe → RAG query |
| `/transcribe` | POST | Audio-to-text transcription |
| `/generate_audio` | POST | Text-to-speech via Kokoro TTS |
| `/knowledge-graph` | GET | Graph data (nodes, edges, clusters) for visualization |
| `/files` | GET | List all uploaded files |
| `/files/{filename}` | GET | Serve file inline |
| `/files/{filename}/preview` | GET | Smart file preview (DOCX→PDF, CSV→table, etc.) |
| `/reset` | POST | Clear all vector stores and metadata |

### Key Request/Response Examples

**POST /ingest** — `multipart/form-data` with `files` field
```json
{ "status": "ok", "message": "Processed 3 files.", "text_chunks": 42 }
```

**POST /chat-stream** — `application/json` with `{ "query": "..." }`
```
data: {"token": "Based on the document"}
data: {"token": " [1], the answer is..."}
data: {"sources": [{"key": 1, "name": "paper", "filename": "paper.pdf"}]}
data: [DONE]
```

**GET /knowledge-graph**
```json
{
  "nodes": [
    {"id": "hub", "label": "Knowledge Base", "type": "hub", "size": 55},
    {"id": "file-0", "label": "paper.pdf", "type": "document", "size": 35}
  ],
  "edges": [
    {"source": "hub", "target": "file-0", "weight": 0.9, "type": "contains"}
  ],
  "clusters": 5
}
```

---

## 📜 License

This project is developed for educational and research purposes.
