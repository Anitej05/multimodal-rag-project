# 🤖 Full Multi-Modal RAG System

This project is a sophisticated, multi-process Python application that implements a Retrieval-Augmented Generation (RAG) system capable of handling a wide variety of file types. You can upload images, audio, PDFs, Word documents, CSVs, and text files to build a knowledge base, and then ask questions about the content.

## Features

*   **Multi-Modal:**  Handles various file formats, including images (`.png`, `.jpg`), audio (`.mp3`, `.wav`), documents (`.pdf`, `.docx`, `.txt`), and data files (`.csv`).
*   **RAG Architecture:**  Utilizes a RAG pipeline to retrieve relevant context from the knowledge base and generate informed answers.
*   **Web Interface:**  A user-friendly web interface built with Gradio for easy file uploads and interaction.
*   **Microservice-based:** The system is broken down into three distinct services for better organization and scalability:
    *   A Whisper service for audio transcription.
    *   A SmolVLM service for image understanding.
    *   A central Gradio app for orchestration.
*   **Efficient Vector Search:**  Employs FAISS for fast and efficient similarity search in the vector database.

## Project Structure

```
multimodal-rag-project/
├── app.py                  # The main Gradio application and orchestrator
├── requirements.txt        # Python dependencies
├── services/
│   ├── smolvlm_service.py  # Flask service for image-related tasks
│   └── whisper_service.py  # Flask service for audio transcription
├── uploads/                # Directory for uploaded files
├── .git/
├── .gradio/
├── bot.png
└── user.png
```

## Models Used

This project leverages several open-source models:

*   **`all-MiniLM-L6-v2`**: A `SentenceTransformer` model for creating text embeddings.
*   **`Qwen3-0.6B-Q4_K_M-GGUF`**: A quantized version of the Qwen2 model, used as the orchestrator for text-based question answering.
*   **`SmolVLM2-500M-Video-Instruct-GGUF`**: A vision language model for describing images and answering questions about them.
*   **`faster-whisper`**: A reimplementation of OpenAI's Whisper model for fast and accurate audio transcription.

## How to Run

### 1. Prerequisites

*   Python 3.8+
*   A virtual environment (recommended)

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

1.  Open the Gradio URL in your browser.
2.  Upload any combination of supported files.
3.  Click the "**⚙️ Process Files**" button and wait for the confirmation message.
4.  Ask questions about the content of your uploaded files in the chatbox.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.