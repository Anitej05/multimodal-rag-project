# backend/services/smolvlm_service.py

from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
import warnings
import base64
from io import BytesIO
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from huggingface_hub import hf_hub_download
import uvicorn

warnings.filterwarnings("ignore")

MODEL_REPO = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF"
MODEL_FILE = "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf"

app = FastAPI()

print("SmolVLM Service: Downloading and loading SmolVLM GGUF model...")
try:
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    mmproj_path = hf_hub_download(repo_id=MODEL_REPO, filename=MMPROJ_FILE)

    chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=True,
    )
    print("SmolVLM Service: Model loaded successfully.")
except Exception as e:
    print(f"SmolVLM Service: Error loading model: {e}")
    llm = None


def image_to_base64_optimized(image):
    # Resize image to reduce token usage (from ~50K to ~5K tokens)
    max_size = 512
    if image.width > max_size or image.height > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    buffered = BytesIO()
    # Use lower quality to further reduce size
    image.save(buffered, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


@app.post('/describe')
async def describe_image(file: UploadFile = File(...)):
    if not llm:
        return {"error": "Model not loaded"}
    try:
        image = Image.open(file.file).convert("RGB")
        image_b64 = image_to_base64_optimized(image)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": "Describe this image concisely in 2-3 sentences. Focus on key elements and avoid unnecessary details."}
            ]}
        ]
        response = llm.create_chat_completion(messages=messages)
        description = response['choices'][0]['message']['content']
        return {"description": description}
    except Exception as e:
        return {"error": str(e)}


@app.post('/answer')
async def answer_question(query: str = Form(...), file: UploadFile = File(...)):
    if not llm:
        return {"error": "Model not loaded"}
    try:
        image = Image.open(file.file).convert("RGB")
        image_b64 = image_to_base64_optimized(image)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": query}
            ]}
        ]
        response = llm.create_chat_completion(messages=messages)
        answer = response['choices'][0]['message']['content']
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5002)
