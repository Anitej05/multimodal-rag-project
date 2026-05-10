@echo off
echo ============================================================
echo  Starting Multimodal RAG Backend + Pix2Text OCR Service
echo ============================================================
echo.

REM --- Start OCR service in pix2text-ocr conda env ---
echo [1/2] Starting Pix2Text OCR service on port 8011...
start "Pix2Text OCR Service" /MIN conda run -n pix2text-ocr --no-banner python "%~dp0ocr_service.py"
timeout /t 5 /nobreak >nul

REM --- Start main backend in multimodal-rag conda env ---
echo [2/2] Starting main backend on port 8000...
cd /d "%~dp0"
python main.py
