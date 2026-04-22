# -*- coding: utf-8 -*-

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import ocr

app = FastAPI(
    title="Paddle OCR API",
    description="PaddleOCR FastAPI Service for Document Digitization"
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(ocr.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "paddle-ocr"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
