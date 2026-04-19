# backend/whisper_service.py — loads Whisper from local models/ directory

import torch
from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel
import os
import warnings
import uvicorn

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
WHISPER_MODEL_PATH = os.path.join(MODELS_DIR, "faster-whisper-base")

print(f"Whisper Service: Using device '{DEVICE}' with compute type '{COMPUTE_TYPE}'.")
print(f"Whisper Service: Loading model from {WHISPER_MODEL_PATH}")

app = FastAPI()

try:
    audio_transcriber = WhisperModel(WHISPER_MODEL_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Whisper Service: Model loaded successfully from local path.")
except Exception as e:
    print(f"Whisper Service: Error loading model: {e}")
    audio_transcriber = None


@app.post('/transcribe')
async def transcribe_audio(file: UploadFile = File(...)):
    if not audio_transcriber:
        return {"error": "Model not loaded"}

    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_audio_file.wav")
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        segments, info = audio_transcriber.transcribe(temp_path, beam_size=5)
        transcription = "".join(segment.text for segment in segments)
        return {"transcription": transcription}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5001)
