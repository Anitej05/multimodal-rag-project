# services/whisper_service.py

import torch
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os
import warnings
import librosa
import soundfile as sf

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
print(f"Whisper Service: Using device '{DEVICE}' with compute type '{COMPUTE_TYPE}'.")

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load Model ---
print("Whisper Service: Loading Whisper model...")
try:
    model_size = "base"
    audio_transcriber = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Whisper Service: Model loaded successfully.")
except Exception as e:
    print(f"Whisper Service: Error loading model: {e}")
    audio_transcriber = None

# --- API Endpoint ---
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if not audio_transcriber:
        return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save file temporarily
        temp_path = "temp_audio_file.wav"
        file.save(temp_path)

        print(f"Whisper Service: Transcribing '{file.filename}'...")
        segments, info = audio_transcriber.transcribe(temp_path, beam_size=5)
        
        transcription = "".join(segment.text for segment in segments)
        
        # Clean up the temp file
        os.remove(temp_path)
        
        print(f"Whisper Service: Transcription successful.")
        return jsonify({"transcription": transcription})

    except Exception as e:
        print(f"Whisper Service: Error during transcription: {e}")
        # Clean up if error occurs
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use port 5001 for the whisper service
    app.run(host='0.0.0.0', port=5001)
