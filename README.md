# Full Multi-Modal RAG System Project

This project refactors a Jupyter Notebook into a robust, multi-process Python application. It uses a Retrieval-Augmented Generation (RAG) architecture to answer questions about a knowledge base of uploaded files, including images, audio, PDFs, and text.

## Architecture

The system is composed of three main parts:

1.  **Whisper Service**: A dedicated Flask server that handles audio transcription. It exposes an API endpoint to convert audio files to text.
2.  **SmolVLM Service**: A dedicated Flask server that handles all image-related tasks. It provides endpoints for generating detailed image descriptions (for ingestion) and answering questions about an image (for querying).
3.  **Main Gradio App**: The central orchestrator and user interface. It handles file uploads, text processing (PDFs, TXT), vector database management (FAISS), and communication with the backend services. It uses a lightweight LLM (Qwen2) for text-based answers and routes image-related queries to the SmolVLM service.


## How to Run

### 1. Prerequisites

-   Python 3.8+
-   A virtual environment (recommended)

### 2. Setup

First, clone the project and create a directory for file uploads.

```bash
git clone <your-repo-url>
cd multimodal-rag-project
mkdir uploads
```
Next, create a virtual environment and install the required dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
### 3. Run the Services
You need to run each component in a separate terminal.

**Terminal 1: Start the Whisper Service**

```bash
python services/whisper_service.py
```
You should see output indicating the Flask server is running on port 5001.

**Terminal 2: Start the SmolVLM Service**

```bash
python services/smolvlm_service.py
```
This will take some time to load the model. You should see output indicating the Flask server is running on port 5002.

**Terminal 3: Start the Main Gradio App**

```bash
python app.py
```
This will provide a public or local URL for the Gradio interface. Open it in your browser to use the application.

### 4. How to Use
Open the Gradio URL in your browser.

Upload any combination of supported files (.png, .jpg, .mp3, .wav, .pdf, .txt).

Click the "⚙️ Process Files" button and wait for the confirmation message.

Ask questions about the content of your uploaded files in the chatbox.
