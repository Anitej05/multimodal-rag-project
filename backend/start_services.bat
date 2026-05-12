@echo off
echo ============================================================
echo  Starting Multimodal RAG Backend + PaddleOCR Service
echo ============================================================
echo.

REM --- Start OCR service in ocr conda env ---
echo [1/2] Starting PaddleOCR service on port 8011...
start "PaddleOCR Service" /MIN conda run -n ocr --no-banner python "%~dp0ocr_service.py"
timeout /t 5 /nobreak >nul

REM --- Start main backend in multimodal-rag conda env ---
echo [2/2] Starting main backend on port 8000...
cd /d "%~dp0"
python main.py
