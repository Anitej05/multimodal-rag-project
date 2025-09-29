# services/smolvlm_service.py

from flask import Flask, request, jsonify
from PIL import Image
import os
import warnings
import base64
from io import BytesIO
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from huggingface_hub import hf_hub_download

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
MODEL_REPO = "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF"
MODEL_FILE = "SmolVLM2-500M-Video-Instruct-Q8_0.gguf"
MMPROJ_FILE = "mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf"

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model ---
print("SmolVLM Service: Downloading and loading SmolVLM GGUF model...")
try:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    mmproj_path = hf_hub_download(repo_id=MODEL_REPO, filename=MMPROJ_FILE)

    chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=2048, # Context window
        n_gpu_layers=-1, # Offload all layers to GPU
        verbose=True,
    )
    print("SmolVLM Service: Model loaded successfully.")
except Exception as e:
    print(f"SmolVLM Service: Error loading model: {e}")
    llm = None

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- API Endpoints ---
@app.route('/describe', methods=['POST'])
def describe_image():
    if not llm:
        return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    try:
        image = Image.open(file.stream).convert("RGB")
        image_b64 = image_to_base64(image)

        print(f"SmolVLM Service: Generating description for '{file.filename}'...")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": "Describe this image in detail."}
            ]}
        ]

        response = llm.create_chat_completion(messages=messages)
        description = response['choices'][0]['message']['content']

        print("SmolVLM Service: Description successful.")
        return jsonify({"description": description})
    except Exception as e:
        print(f"SmolVLM Service: Error during description: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/answer', methods=['POST'])
def answer_question():
    if not llm:
        return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    query = request.form.get('query', '')
    if not query:
        return jsonify({"error": "Query is missing"}), 400
        
    try:
        image = Image.open(file.stream).convert("RGB")
        image_b64 = image_to_base64(image)

        print(f"SmolVLM Service: Answering query '{query}' for image '{file.filename}'...")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": query}
            ]}
        ]

        response = llm.create_chat_completion(messages=messages)
        answer = response['choices'][0]['message']['content']

        print("SmolVLM Service: Answer successful.")
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"SmolVLM Service: Error during answering: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use port 5002 for the SmolVLM service
    app.run(host='0.0.0.0', port=5002)
