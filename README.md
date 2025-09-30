# Multimodal RAG Engine

This project is a sophisticated, multimodal Retrieval-Augmented Generation (RAG) engine. It provides a backend service that can ingest various types of documents, including text files, PDFs, audio, and images, and answer questions based on the ingested knowledge.

## Features

- **Multimodal Knowledge Base:** Ingest and understand information from a wide range of file types:
    - Documents: `.pdf`, `.docx`, `.txt`, `.csv`
    - Audio: `.mp3`, `.wav`, `.m4a` (transcribed using Whisper)
    - Images: `.png`, `.jpg`, `.jpeg` (analyzed for both visual content and text)
- **Advanced Image Analysis:** Utilizes a dual-approach for images:
    - **OCR:** Extracts text from images using EasyOCR.
    - **Visual Embedding:** Creates a vector representation of the image's visual content using CLIP.
- **Intelligent Chat Orchestration:** Features a powerful chat interface that:
    - Searches across all content types (text and images) to find the most relevant information.
    - Uses a large language model (`Qwen`) as an orchestrator.
    - Intelligently routes tasks to a specialized multimodal model (`SmolVLM`) when visual analysis is required.
- **Voice-Enabled Queries:** Supports asking questions via audio input directly to the chat endpoint.

## Architecture

The backend is built with **FastAPI** and consists of two main services that are launched automatically:

1.  **Main Backend (`main.py`):** Handles ingestion, retrieval, and orchestration.
2.  **Multimodal Service (`services/smolvlm_service.py`):** A dedicated service for answering questions about images.

The RAG pipeline uses two separate in-memory **FAISS** vector stores:
- A **Text Vector Store** for embeddings of documents, audio transcriptions, and OCR text.
- An **Image Vector Store** for visual embeddings of images.

### Models Used

- **Text Embedding:** `all-MiniLM-L6-v2`
- **Image Embedding:** `clip-ViT-B-32`
- **Orchestrator LLM:** `Qwen3-4B-Instruct-2507-GGUF`
- **Vision-Language Model:** `SmolVLM2-2.2B-Instruct-GGUF`
- **Audio Transcription:** `faster-whisper (base)`
- **OCR:** `EasyOCR`

## Setup and Installation

1.  **Prerequisites:**
    - Python 3.9+
    - `pip`

2.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd multimodal-rag-project
    ```

3.  **Install Python dependencies:**
    Navigate to the `backend` directory and install the required packages.
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
    *(Note: The first time you run the application, EasyOCR and Hugging Face Transformers will download their required models. This is a one-time setup.)*
    
    


## Running the Project

The application is designed to be simple to run. All necessary services are started automatically.

From the `backend` directory, run:
```bash
python main.py
```
This single command will:
- Start the multimodal `smolvlm_service` in the background.
- Start the main FastAPI server on `http://127.0.0.1:8000`.

The server is now ready to accept API requests.

## API Endpoints

The following endpoints are available:

#### `POST /ingest`
Uploads and processes a list of files to build the knowledge base. This clears any previously ingested data.

- **Body:** `multipart/form-data` with a `files` field containing one or more files.
- **Example (`curl`):**
  ```bash
  curl -X POST -F "files=@document1.pdf" -F "files=@image1.png" http://127.0.0.1:8000/ingest
  ```

#### `POST /chat`
Asks a question to the RAG engine using a text query.

- **Body:** JSON payload with a `query` field.
  ```json
  {
    "query": "What is the main topic of the document?"
  }
  ```
- **Example (`curl`):**
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"query": "What is written in the image?"}' http://127.0.0.1:8000/chat
  ```

#### `POST /chat-audio`
Asks a question to the RAG engine using an audio file.

- **Body:** `multipart/form-data` with a `file` field containing a single audio file.
- **Example (`curl`):**
  ```bash
  curl -X POST -F "file=@my_question.mp3" http://127.0.0.1:8000/chat-audio
  ```

#### `POST /transcribe`
Transcribes a given audio file and returns the text.

- **Body:** `multipart/form-data` with a `file` field containing a single audio file.
- **Example (`curl`):**
  ```bash
  curl -X POST -F "file=@meeting_notes.wav" http://127.0.0.1:8000/transcribe
  ```

#### `POST /reset`
Clears all knowledge bases (both text and images).

- **Example (`curl`):**
  ```bash
  curl -X POST http://127.0.0.1:8000/reset
  ```
