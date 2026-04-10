# 🧠 Multimodal RAG Assistant

> A full-stack **Retrieval-Augmented Generation (RAG)** system that ingests **text, images, and audio** documents, indexes them into a FAISS vector database, and answers user queries via a streaming AI chat interface — powered by **LM Studio** for local LLM inference, with an interactive **Knowledge Graph** visualization of all ingested content.

---

## 📑 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Complete Data Flow](#complete-data-flow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Backend — Deep Dive](#backend--deep-dive)
  - [main.py — The Core Engine](#mainpy--the-core-engine)
  - [whisper_service.py — Audio Transcription](#whisper_servicepy--audio-transcription)
- [Frontend — Deep Dive](#frontend--deep-dive)
  - [App.jsx — Application Shell](#appjsx--application-shell)
  - [Chat.jsx — AI Chat Interface](#chatjsx--ai-chat-interface)
  - [KnowledgeBase.jsx — File Management Panel](#knowledgebasejsx--file-management-panel)
  - [FileUpload.jsx — Drag-and-Drop Upload](#fileuploadjsx--drag-and-drop-upload)
  - [FileList.jsx — Uploaded File Browser](#filelistjsx--uploaded-file-browser)
  - [StatsCard.jsx — Knowledge Base Stats](#statscardj​sx--knowledge-base-stats)
  - [KnowledgeGraph.jsx — Interactive Graph Visualization](#knowledgegraphjsx--interactive-graph-visualization)
  - [Header.jsx — App Header](#headerjsx--app-header)
  - [Toast.jsx — Notification System](#toastjsx--notification-system)
  - [ConfirmDialog.jsx — Confirmation Modal](#confirmdialogj​sx--confirmation-modal)
  - [api.js — API Service Layer](#apijs--api-service-layer)
  - [helpers.js — Utility Functions](#helpersjs--utility-functions)
- [Knowledge Graph — In Detail](#knowledge-graph--in-detail)
- [File Preview System — In Detail](#file-preview-system--in-detail)
- [Setup & Installation](#setup--installation)
- [API Reference](#api-reference)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FRONTEND (React 18)                             │
│  ┌───────────┐  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │  Header   │  │ KnowledgeBase│  │    Chat.jsx    │  │KnowledgeGraph  │  │
│  │           │  │ ┌──────────┐ │  │  • Streaming   │  │ • Cytoscape.js │  │
│  │  App.jsx  │  │ │FileUpload│ │  │  • Voice I/O   │  │ • Force Layout │  │
│  │  (Shell)  │  │ │FileList  │ │  │  • Citations   │  │ • Search/Zoom  │  │
│  │           │  │ │StatsCard │ │  │  • File Links  │  │ • Detail Panel │  │
│  └───────────┘  └──────────────┘  └────────────────┘  └────────────────┘  │
│                              ↕ api.js (fetch)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                               ↕ HTTP REST
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BACKEND (FastAPI · Python)                           │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │  /ingest   │  │   /chat      │  │ /chat-stream │  │/knowledge-graph│   │
│  │  (Upload & │  │  (One-shot)  │  │  (SSE Stream)│  │  (Graph Data)  │   │
│  │  Vectorize)│  │              │  │              │  │                │   │
│  └──────┬─────┘  └──────┬───────┘  └──────┬───────┘  └────────┬───────┘   │
│         │               │                 │                    │           │
│  ┌──────▼───────────────▼─────────────────▼────────────────────▼────────┐  │
│  │                        Core Pipeline                                │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ ┌─────────────┐ │  │
│  │  │ SentenceTr. │  │  FAISS       │  │ Cross-     │ │ LM Studio   │ │  │
│  │  │ (Embeddings)│  │  (Vector DB) │  │ Encoder    │ │ (LLM + VLM) │ │  │
│  │  │ MiniLM-L6-v2│  │  L2 / IVF   │  │ Re-ranker  │ │ Qwen3-4B    │ │  │
│  │  └─────────────┘  └──────────────┘  └────────────┘ └─────────────┘ │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │ Whisper Service │  │   EasyOCR    │  │  Kokoro TTS (Text-to-Speech) │  │
│  │ (Port 5001)     │  │  (Image OCR) │  │  (Audio Generation)          │  │
│  └─────────────────┘  └──────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                               ↕ HTTP
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LM STUDIO (Local LLM Server)                           │
│    • Text Model:  qwen/qwen3-4b  (Port 1234)                              │
│    • Vision Model: smolvlm2-500m-video-instruct                            │
│    • OpenAI-compatible /v1/chat/completions API                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow

### 1️⃣ Document Ingestion Flow

```
User drops files in UI
        │
        ▼
FileUpload.jsx ──(FormData)──► POST /ingest
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Backend: File Type Router                       │
│                                                  │
│  .pdf  ──► pypdf.PdfReader → extract text        │
│  .docx ──► python-docx     → extract paragraphs │
│  .txt  ──► UTF-8 file read → raw text            │
│  .csv  ──► pandas.read_csv → df.to_string()      │
│  .png/.jpg ──► LM Studio Vision API              │
│              (smolvlm2-500m-video-instruct)       │
│              → generates text description         │
│  .mp3/.wav ──► Whisper Service (port 5001)        │
│              → audio transcription text            │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │  iterative_chunking()   │
          │  Split text → 400-char  │
          │  chunks at natural      │
          │  break points (¶, ., ␣) │
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │  SentenceTransformer    │
          │  all-MiniLM-L6-v2      │
          │  → 384-dim embeddings  │
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │  FAISS Vector Store     │
          │  • <39 chunks: FlatL2   │
          │  • ≥39 chunks: IVFFlat  │
          │    (nlist = 4√N, ≤256)  │
          │  + text_metadata[]      │
          └─────────────────────────┘
```

### 2️⃣ Query & Retrieval Flow (RAG Pipeline)

```
User types question in Chat
        │
        ▼
Chat.jsx ──(SSE)──► POST /chat-stream
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Step 1: MODALITY DETECTION                            │
│  ┌──────────────────────────────┐                      │
│  │ detect_modality()            │                      │
│  │ Asks LM Studio: "Which       │                      │
│  │ modality to boost: text,     │                      │
│  │ image, or audio?"            │                      │
│  │ → Returns: {"text": 1.2}     │                      │
│  └──────────────┬───────────────┘                      │
│                 │                                       │
│  Step 2: VECTOR SEARCH                                 │
│  ┌──────────────▼───────────────┐                      │
│  │ Encode query → 384-dim vec   │                      │
│  │ FAISS.search(top_k * 2)      │                      │
│  │ Normalize L2 → similarity    │                      │
│  │ Apply modality boost         │                      │
│  └──────────────┬───────────────┘                      │
│                 │                                       │
│  Step 3: RE-RANK with CROSS-ENCODER                    │
│  ┌──────────────▼───────────────┐                      │
│  │ ms-marco-MiniLM-L-6-v2      │                      │
│  │ Score(query, candidate)      │                      │
│  │ Combined: 60% text + 40%    │                      │
│  │ cross-encoder score          │                      │
│  └──────────────┬───────────────┘                      │
│                 │                                       │
│  Step 4: TOP-5 RESULTS                                 │
│  ┌──────────────▼───────────────┐                      │
│  │ Sort by final_score          │                      │
│  │ Return top 5 contexts        │                      │
│  │ with source file paths       │                      │
│  └──────────────┬───────────────┘                      │
│                 │                                       │
│  Step 5: STREAM LLM RESPONSE                           │
│  ┌──────────────▼───────────────┐                      │
│  │ Build prompt with context    │                      │
│  │ + citation instructions      │                      │
│  │ Stream via LM Studio SSE     │                      │
│  │ Strip <think> tags on-the-fly│                      │
│  │ Emit token-by-token          │                      │
│  │ Append sources at end        │                      │
│  └──────────────────────────────┘                      │
└────────────────────────────────────────────────────────┘
        │
        ▼  SSE: data: {"token": "..."} → data: {"sources": [...]} → data: [DONE]
Chat.jsx receives tokens via EventSource
  → requestAnimationFrame DOM updates
  → Final message committed to React state
  → Source citations [1], [2] become clickable
  → Clicking a source opens /files/{name}/preview
```

### 3️⃣ Voice Interaction Flow

```
User clicks 🎙️ → MediaRecorder → WebM blob
        │
        ▼
POST /transcribe → Whisper Service → transcription text
        │
        ▼
Text inserted into chat input → User reviews → Send
        │
        ▼
(Follows same RAG pipeline as text queries)

──── Read Aloud ────
User clicks 🔊 → last AI message text
        │
        ▼
POST /generate_audio → Kokoro TTS → WAV audio blob
        │
        ▼
Audio plays in browser via HTML5 Audio API
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 | UI framework |
| **Graph Viz** | Cytoscape.js + cose-bilkent | Interactive knowledge graph |
| **Styling** | Vanilla CSS (dark/light themes) | UI design system |
| **Backend** | FastAPI + Uvicorn | REST API server |
| **LLM** | LM Studio (Qwen3-4B) | Text generation & streaming |
| **VLM** | LM Studio (SmolVLM2-500M) | Image description |
| **Embeddings** | SentenceTransformer (MiniLM-L6-v2) | 384-dim text embeddings |
| **Re-ranking** | CrossEncoder (ms-marco-MiniLM-L-6-v2) | Candidate re-ranking |
| **Vector DB** | FAISS (FlatL2 / IVFFlat) | Similarity search |
| **ASR** | Faster-Whisper (base) | Speech-to-text |
| **TTS** | Kokoro TTS | Text-to-speech |
| **OCR** | EasyOCR | Image text extraction |
| **Doc Parsing** | pypdf, python-docx, pandas, mammoth | File content extraction |
| **Doc Preview** | docx2pdf + mammoth (fallback) | In-browser DOCX viewing |

---

## Project Structure

```
multimodal-rag-project/
├── backend/
│   ├── main.py                 # Core FastAPI server (1677 lines)
│   ├── whisper_service.py      # Standalone Whisper transcription microservice
│   ├── requirements.txt        # Python dependencies
│   └── uploads/                # Uploaded files stored here
│
├── frontend/
│   ├── package.json            # React app config & dependencies
│   ├── public/                 # Static assets
│   ├── src/
│   │   ├── index.js            # React entry point
│   │   ├── App.jsx             # Application shell & routing
│   │   ├── components/
│   │   │   ├── Chat.jsx        # AI chat with streaming & voice
│   │   │   ├── KnowledgeBase.jsx  # File management panel
│   │   │   ├── KnowledgeGraph.jsx # Interactive graph visualization
│   │   │   ├── FileUpload.jsx  # Drag-drop file upload
│   │   │   ├── FileList.jsx    # Uploaded file browser
│   │   │   ├── StatsCard.jsx   # Upload statistics display
│   │   │   ├── Header.jsx      # App header/branding
│   │   │   ├── Toast.jsx       # Toast notification system
│   │   │   └── ConfirmDialog.jsx  # Confirmation modal
│   │   ├── services/
│   │   │   └── api.js          # Backend API service layer
│   │   ├── styles/
│   │   │   ├── globals.css     # Global design system & themes
│   │   │   └── knowledgeGraph.css # Knowledge graph styles
│   │   └── utils/
│   │       ├── helpers.js      # Utility functions
│   │       └── messageParser.test.js  # Unit tests
│   └── build/                  # Production build output
│
├── README.md                   # This file
└── test-parsing.js             # Parsing test utilities
```

---

## Backend — Deep Dive

### `main.py` — The Core Engine

> **File**: [`backend/main.py`](backend/main.py) · **1,677 lines** · The monolithic backend handling all AI, indexing, and serving logic.

#### Initialization (Lines 1–100)

On startup, the backend initializes all ML models and infrastructure:

| Component | Model / Config | Purpose |
|-----------|---------------|---------|
| `ocr_reader` | EasyOCR (`en`, GPU-aware) | Extract text from images |
| `embedding_model` | `all-MiniLM-L6-v2` | Encode text → 384-dim vectors |
| `cross_encoder_model` | `ms-marco-MiniLM-L-6-v2` (with TinyBERT fallback) | Re-rank search candidates |
| `text_vector_store` | FAISS index (dynamic FlatL2/IVFFlat) | Store & search embeddings |
| `text_metadata` | Python list of dicts `{text, source_path}` | Chunk ↔ source mapping |

#### LM Studio API Integration (Lines 101–287)

Three LM Studio wrapper functions:

| Function | Purpose | Model | Max Tokens |
|----------|---------|-------|------------|
| `call_lm_studio_text()` | Synchronous text generation | `qwen/qwen3-4b` | 1000 |
| `stream_lm_studio_text()` | SSE streaming generation (yields `data: {token}`) | `qwen/qwen3-4b` | 1500 |
| `call_lm_studio_vision()` | Image description via base64 encoding | `smolvlm2-500m-video-instruct` | 300 |

All functions strip Qwen's `<think>...</think>` tags from responses.

#### FAISS Vector Store Management (Lines 288–361)

- **`calculate_optimal_nlist(n)`** — Uses `4√N` heuristic capped at 256 clusters
- **`choose_index_type(n, dim)`** — `< 39 samples → IndexFlatL2` (brute force), `≥ 39 → IndexIVFFlat`
- **`safe_train_index(index, embeddings)`** — Trains IVF index with automatic fallback to FlatL2

#### Text Chunking & Scoring (Lines 362–415)

- **`iterative_chunking(text, 400)`** — Splits text at natural boundaries: `\n\n` → `. ` → ` ` → hard cut at 400 chars
- **`normalize_score(raw, modality)`** — Converts FAISS L2 distances to 0–1 similarity scores
- **`calculate_final_score(candidate)`** — Combines text score (60%) + cross-encoder score (40%)

#### Intelligent Query Routing (Lines 416–627)

- **`detect_visual_query_qwen(query)`** — Uses LLM to classify if a query needs image analysis
- **`detect_modality(query, docs)`** — LLM-based modality boost detection (text/image/audio, boost 1.0–2.0) with keyword fallback
- **`parse_simple_format()` / `parse_sources_format()` / `parse_json_with_retry()`** — Robust response parsing with multiple fallbacks

#### Multimodal Retrieval Pipeline (Lines 628–703)

`retrieve_multimodal_context(query, top_k=5)`:
1. **Detect modality** — ask LLM which modality to boost
2. **FAISS search** — encode query, search `top_k * 2` candidates
3. **Apply modality boost** — multiply scores by boost factor
4. **Cross-encoder re-rank** — score `(query, candidate_text)` pairs
5. **Combine scores** — `0.6 × text_score + 0.4 × cross_encoder_score`
6. **Return top 5** — sorted by final combined score

#### API Endpoints (Lines 705–1677)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `POST /ingest` | Upload files & index into FAISS | File ingestion |
| `POST /chat` | One-shot RAG query (non-streaming) | Chat (sync) |
| `POST /chat-stream` | Streaming RAG with SSE token-by-token | Chat (stream) |
| `POST /chat-audio` | Upload audio → transcribe → RAG query | Voice chat |
| `POST /transcribe` | Audio → text transcription only | Transcription |
| `POST /generate_audio` | Text → WAV audio via Kokoro TTS | Read aloud |
| `GET /files` | List uploaded files with types/sizes | File listing |
| `GET /files/{filename}` | Serve file inline (PDF, image, etc.) | File serving |
| `GET /files/{filename}/preview` | Smart preview (DOCX→PDF, TXT→HTML, CSV→HTML table) | File preview |
| `POST /reset` | Clear vector store & metadata | Reset KB |
| `GET /knowledge-graph` | Build & return graph data (nodes, edges, clusters) | Graph data |

#### Streaming Chat Architecture (Lines 999–1142)

The `/chat-stream` endpoint:
1. Retrieves multimodal context (top 5 chunks)
2. Builds citation mapping `{source_path → citation_number}`
3. Constructs system prompt requiring HTML + citation format
4. Streams tokens via `stream_lm_studio_text()` → yields SSE chunks
5. Accumulates full response to detect which citations `[1]`, `[2]` were used
6. Sends final `sources` event with only cited sources
7. Sends `[DONE]` signal

---

### `whisper_service.py` — Audio Transcription

> **File**: [`backend/whisper_service.py`](backend/whisper_service.py) · **53 lines** · Standalone FastAPI microservice on port 5001.

- Uses **Faster-Whisper** (`base` model) for speech-to-text
- Auto-selects CUDA/CPU and float16/int8 compute type
- Single endpoint: `POST /transcribe` — accepts audio file, returns `{transcription: "..."}`
- Launched automatically by `main.py` as a subprocess on startup

---

## Frontend — Deep Dive

### `App.jsx` — Application Shell

> **File**: [`frontend/src/App.jsx`](frontend/src/App.jsx) · **156 lines**

The root component managing:
- **View switching** between `'chat'` and `'graph'` tabs via `activeView` state
- **Theme toggling** (dark/light mode with localStorage persistence)
- **Global state**: `uploadedFiles`, `messages`, `toast`
- **Message management**: `addMessage()`, `updateLastMessage()`, `clearMessages()`

Layout: Both views share the `KnowledgeBase` panel on the left; the right panel switches between `Chat` and `KnowledgeGraph`.

---

### `Chat.jsx` — AI Chat Interface

> **File**: [`frontend/src/components/Chat.jsx`](frontend/src/components/Chat.jsx) · **501 lines**

The core chat component implementing:

#### Streaming Architecture
- Uses `api.chatTextStream()` with SSE callbacks: `onToken`, `onSources`, `onDone`, `onError`
- **Performance optimization**: Tokens accumulate in `streamContentRef` (a ref, not state) and update the DOM via `requestAnimationFrame` — avoids React re-renders during streaming
- A dedicated `streamBubbleRef` div is used for live rendering; final content is committed to React state only on `onDone`

#### Citation System
- Backend sends `[1]`, `[2]` etc. in the HTML response
- `processedContentText` replaces `[N]` with styled `<span class="citation">` elements
- Source items include `data-filename` attributes for file linking

#### File Preview Integration
- `findActualFilename(cleanName)` — fuzzy-matches source names to actual uploaded filenames
- `handleSourceClick()` — event delegation on the chat container; clicking a source opens `api.getFilePreviewUrl(filename)` in a new tab

#### Voice I/O
- **Record** 🎙️: `MediaRecorder` → WebM blob → `POST /transcribe` → text inserted into input
- **Read Aloud** 🔊: Last AI message → `POST /generate_audio` → HTML5 `Audio` playback with pause/resume

---

### `KnowledgeBase.jsx` — File Management Panel

> **File**: [`frontend/src/components/KnowledgeBase.jsx`](frontend/src/components/KnowledgeBase.jsx) · **163 lines**

Two-tab panel:
- **Upload tab**: `FileUpload` + `StatsCard` + `Index Knowledge Base` button + `Reset` button
- **Files tab**: `FileList` showing all uploaded files

Key actions:
- `handleIndex()` — Calls `api.ingestFiles()` → updates file statuses to `'indexed'`
- `handleReset()` — Shows `ConfirmDialog` → calls `api.resetKB()` → clears state

---

### `FileUpload.jsx` — Drag-and-Drop Upload

> **File**: [`frontend/src/components/FileUpload.jsx`](frontend/src/components/FileUpload.jsx) · **113 lines**

- Drag-and-drop zone with click-to-browse fallback
- Creates file objects with `{id, file, name, type, size, status, previewUrl}`
- Image files get `URL.createObjectURL()` previews
- Supported formats: PDF, TXT, DOCX, CSV, PNG, JPG, MP3, WAV, M4A

---

### `FileList.jsx` — Uploaded File Browser

> **File**: [`frontend/src/components/FileList.jsx`](frontend/src/components/FileList.jsx) · **46 lines**

- Renders each uploaded file with icon, name, size, file type, and indexing status
- Status shown as `pending` or `indexed` badges
- Image files show thumbnail previews

---

### `StatsCard.jsx` — Knowledge Base Stats

> **File**: [`frontend/src/components/StatsCard.jsx`](frontend/src/components/StatsCard.jsx) · **27 lines**

Displays three stat cards:
- **Total Files** — count of all uploads
- **Indexed** — count of files that have been indexed
- **Total Size** — sum of all file sizes (formatted as KB/MB/GB)

---

### `KnowledgeGraph.jsx` — Interactive Graph Visualization

> **File**: [`frontend/src/components/KnowledgeGraph.jsx`](frontend/src/components/KnowledgeGraph.jsx) · **874 lines**

*See dedicated [Knowledge Graph](#knowledge-graph--in-detail) section below for complete documentation.*

---

### `Header.jsx` — App Header

> **File**: [`frontend/src/components/Header.jsx`](frontend/src/components/Header.jsx) · **16 lines**

Simple branding header: `🧠 Multimodal RAG Assistant`

---

### `Toast.jsx` — Notification System

> **File**: [`frontend/src/components/Toast.jsx`](frontend/src/components/Toast.jsx) · **13 lines**

Renders toast notifications with dynamic CSS classes (`toast-info`, `toast-success`, `toast-error`). Auto-dismisses after 3 seconds (managed by `App.jsx`).

---

### `ConfirmDialog.jsx` — Confirmation Modal

> **File**: [`frontend/src/components/ConfirmDialog.jsx`](frontend/src/components/ConfirmDialog.jsx) · **23 lines**

Used by `KnowledgeBase` (reset) and `Chat` (clear chat). Overlay modal with OK/Cancel buttons.

---

### `api.js` — API Service Layer

> **File**: [`frontend/src/services/api.js`](frontend/src/services/api.js) · **142 lines**

Central API client pointing to `http://127.0.0.1:8000`:

| Method | Endpoint | Returns |
|--------|----------|---------|
| `ingestFiles(files)` | `POST /ingest` | `{status, message, text_chunks}` |
| `chatText(query)` | `POST /chat` | `{answer}` |
| `chatTextStream(query, callbacks)` | `POST /chat-stream` | SSE stream |
| `chatAudio(file)` | `POST /chat-audio` | `{answer}` |
| `transcribeAudio(file)` | `POST /transcribe` | `{transcription}` |
| `generateAudio(text, voice)` | `POST /generate_audio` | Audio blob |
| `resetKB()` | `POST /reset` | `{status, message}` |
| `getFileUrl(filename)` | — | Direct file URL |
| `getFilePreviewUrl(filename)` | — | Pre​view URL (smart routing) |
| `listFiles()` | `GET /files` | `{files: [...]}` |

#### SSE Streaming Implementation
`chatTextStream()` uses the Fetch API with `response.body.getReader()` to process Server-Sent Events manually:
- Reads chunks from the stream, decodes UTF-8, splits by newline
- Parses `data: {...}` lines as JSON
- Routes to callbacks: `onToken`, `onSources`, `onDone`, `onError`
- Handles `data: [DONE]` termination signal

---

### `helpers.js` — Utility Functions

> **File**: [`frontend/src/utils/helpers.js`](frontend/src/utils/helpers.js) · **31 lines**

| Function | Purpose |
|----------|---------|
| `formatBytes(bytes)` | Human-readable file sizes (B, KB, MB, GB) |
| `getFileIcon(type)` | MIME type → emoji icon (🖼️📄📝📊🎵📎) |
| `escapeHtml(str)` | XSS-safe HTML escaping |
| `formatTime()` | Current time as HH:MM |
| `generateId()` | Unique ID (timestamp + random) |

---

## Knowledge Graph — In Detail

The Knowledge Graph provides an **interactive, Obsidian-inspired visualization** of all ingested documents, their text chunks, and extracted concepts.

### Backend: `GET /knowledge-graph` (Lines 1505–1652)

The endpoint dynamically builds a graph from `text_metadata`:

#### Node Types

| Type | Shape | Color | Description |
|------|-------|-------|-------------|
| `hub` | Circle (55px) | 🔵 `#2563eb` | Central "Knowledge Base" node |
| `document` | Circle (25–45px) | 🟣 `#6366f1` | PDF, DOCX, TXT, CSV files |
| `image` | Circle (25–45px) | 🩷 `#f472b6` | PNG, JPG, JPEG files |
| `audio` | Circle (25–45px) | 🟢 `#34d399` | MP3, WAV, M4A files |
| `chunk` | Rounded rect (12px) | ⚪ `#94a3b8` | Text chunks (max 5 per file) |
| `concept` | Diamond (14–28px) | 🟡 `#fbbf24` | Keywords extracted from content |

#### Edge Types

| Type | Style | From → To | Meaning |
|------|-------|-----------|---------|
| `contains` | Solid, indigo | Hub → File | File is in the KB |
| `chunk_of` | Default | File → Chunk | Chunk belongs to file |
| `mentions` | Dashed, gold | File → Concept | File mentions this concept |
| `related` | Default, green | Concept → Concept | Concepts share source files |

#### Concept Extraction
1. Regex extracts all 4+ letter words from all chunks of each file
2. Filters against a 100+ word stop list
3. Top 5 concepts per file become concept nodes
4. Inter-concept edges created when concepts share source files

### Frontend: `KnowledgeGraph.jsx` (874 lines)

Built on **Cytoscape.js** with the **cose-bilkent** force-directed layout plugin.

#### Features

| Feature | Implementation |
|---------|---------------|
| **Force-directed layout** | cose-bilkent with gravity, repulsion, edge elasticity |
| **4 layout options** | Force Directed, Concentric, Circle, Grid |
| **Node hover** | Highlights neighborhood, fades everything else |
| **Node click** | Opens detail panel with metadata |
| **Search** | Fuzzy search on node labels/descriptions |
| **Zoom controls** | Zoom in, zoom out, fit-to-screen |
| **stats bar** | Node count, edge count, cluster count |
| **Legend** | Color-coded node type legend |
| **Particles** | 20 animated background particles |
| **Tooltip** | Shows node type badge, label, description on hover |
| **Detail panel** | Slide-in panel showing description, file type, status, connections, degree |

#### Cytoscape Style System
- Nodes have `shadow-blur` glow effects matching their type color
- `transition-property` enables smooth CSS transitions on hover/select
- Edges use bezier curves with width proportional to weight
- `.faded` class reduces opacity to 0.15 for focus effects
- `.search-match` class adds gold border highlight

#### Layout Configurations

| Layout | Key parameters |
|--------|---------------|
| **cose-bilkent** | nodeRepulsion: 8500, idealEdgeLength: 120, gravity: 0.25, 2500 iterations |
| **concentric** | Hub → center (priority 10), files → ring 2 (7), concepts → ring 3 (4), chunks → outer (1) |
| **circle** | Even spacing, avoidOverlap, spacingFactor: 1.5 |
| **grid** | Auto rows, condensed, avoidOverlap |

---

## File Preview System — In Detail

The file preview system enables **in-browser viewing** of all uploaded file types directly from source citations.

### How It Works

```
User clicks source citation [1] in chat
        │
        ▼
Chat.jsx: handleSourceClick()
  → finds actual filename via data-filename attribute
  → opens api.getFilePreviewUrl(filename) in new tab
        │
        ▼
GET /files/{filename}/preview
        │
        ├── .docx/.doc ──► docx2pdf (MS Word COM automation in subprocess)
        │                      ├── Success → serve as PDF inline
        │                      └── Failure → mammoth HTML conversion fallback
        │                           → styled dark-theme HTML page
        │
        ├── .txt ──► Read UTF-8 → styled HTML with JetBrains Mono font
        │
        ├── .csv ──► pandas.read_csv → HTML table with sticky headers
        │
        └── .pdf/.png/.jpg/.mp3 ──► Redirect to /files/{filename}
                                     → FileResponse with inline Content-Disposition
```

### Preview Rendering

| File Type | Preview Method | Styling |
|-----------|---------------|---------|
| **PDF** | Browser native viewer | N/A |
| **DOCX/DOC** | docx2pdf → PDF, or mammoth → HTML | Dark theme, Inter font, styled tables |
| **TXT** | Raw text → HTML `<br>` conversion | JetBrains Mono, dark theme, monospace |
| **CSV** | pandas → HTML table | Sticky headers, hover rows, zebra striping |
| **Images** | Browser native `<img>` | Inline display |
| **Audio** | Browser native `<audio>` | Inline player |

---

## Setup & Installation

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **LM Studio** running with:
  - Text model: `qwen/qwen3-4b` loaded
  - Vision model: `smolvlm2-500m-video-instruct` loaded
  - Server running on `http://localhost:1234`
- **(Optional)** MS Word for DOCX→PDF preview conversion
- **(Optional)** NVIDIA GPU for CUDA-accelerated inference

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the backend (also launches Whisper service on port 5001)
python main.py
```

The backend starts on `http://localhost:8000`.

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start development server
npm start
```

The frontend starts on `http://localhost:3000`.

### Optional: Kokoro TTS

```bash
pip install kokoro>=0.9.2
```

If not installed, the "Read Aloud" feature will be disabled but the app will still function.

---

## API Reference

### `POST /ingest`
Upload and index files into the knowledge base.

**Request:** `multipart/form-data`
- `files` (required): One or more files
- `reset_db` (optional): `"true"` to clear existing data first (default: `"true"`)

**Response:**
```json
{
  "status": "ok",
  "message": "Processed 3 files.",
  "text_chunks": 42
}
```

---

### `POST /chat-stream`
Streaming RAG chat via Server-Sent Events.

**Request:** `application/json`
```json
{ "query": "What does the document say about X?" }
```

**Response:** `text/event-stream`
```
data: {"token": "<p>Based on"}
data: {"token": " the document"}
data: {"token": " [1]...</p>"}
data: {"sources": [{"key": 1, "name": "research_paper", "filename": "research_paper.pdf"}]}
data: [DONE]
```

---

### `POST /chat`
One-shot (non-streaming) RAG chat.

**Request:** `application/json`
```json
{ "query": "What does the document say about X?" }
```

**Response:**
```json
{
  "answer": "<p>HTML formatted response with [1] citations...</p>--%Sources%--<div>...</div>"
}
```

---

### `GET /knowledge-graph`
Returns graph data for visualization.

**Response:**
```json
{
  "nodes": [
    {"id": "hub-knowledge-base", "label": "Knowledge Base", "type": "hub", "size": 55, "description": "..."},
    {"id": "file-0", "label": "research_paper", "type": "document", "size": 35, "description": "..."}
  ],
  "edges": [
    {"source": "hub-knowledge-base", "target": "file-0", "weight": 0.9, "type": "contains"}
  ],
  "clusters": 5
}
```

---

### `GET /files/{filename}/preview`
Smart file preview with format-specific rendering.

**Response varies by file type:**
- `.pdf`, `.png`, `.jpg` → `FileResponse` (inline)
- `.docx` → `FileResponse` (converted PDF) or `HTMLResponse` (mammoth fallback)
- `.txt` → `HTMLResponse` (styled monospace)
- `.csv` → `HTMLResponse` (styled table)

---

### `POST /transcribe`
Audio-to-text transcription.

**Request:** `multipart/form-data`
- `file`: Audio file (WAV, MP3, WebM, M4A)

**Response:**
```json
{ "transcription": "Hello, this is a test recording." }
```

---

### `POST /generate_audio`
Text-to-speech via Kokoro TTS.

**Request:** `multipart/form-data`
- `text` (required): Text to synthesize
- `voice` (optional): Voice ID (default: `af_heart`)

**Response:** `audio/mpeg` WAV binary stream

---

### `POST /reset`
Clear all vector stores and metadata.

**Response:**
```json
{ "status": "ok", "message": "All knowledge bases cleared." }
```

---

### `GET /files`
List all uploaded files.

**Response:**
```json
{
  "files": [
    {"name": "research_paper.pdf", "size": 204800, "type": "document"},
    {"name": "diagram.png", "size": 51200, "type": "image"}
  ]
}
```

---

## License

This project is developed for educational and research purposes.
