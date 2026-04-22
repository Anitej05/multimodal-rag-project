# -*- coding: utf-8 -*-

import os
import tempfile
import traceback
import shutil
import base64
import cv2
import numpy as np

from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from paddleocr import PaddleOCR, PPStructure, draw_structure_result, save_structure_res
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray
from models.OCRModel import Base64PostModel
from models.RestfulModel import RestfulModel
from PIL import Image
import requests

OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "en")
USE_GPU = os.environ.get("USE_GPU", "true").lower() == "true"
OUTPUT_DIR = "/app/ocr_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

router = APIRouter(prefix="/ocr", tags=["OCR"])

# --- Initialize PaddleOCR (basic OCR) ---
print(f"Initializing PaddleOCR with lang={OCR_LANGUAGE}, gpu={USE_GPU}...")
ocr = PaddleOCR(use_angle_cls=True, lang=OCR_LANGUAGE, use_gpu=USE_GPU)
print("PaddleOCR initialized successfully.")

# --- Initialize PPStructure (layout analysis + recovery) ---
print(f"Initializing PPStructure with recovery=True, lang={OCR_LANGUAGE}, gpu={USE_GPU}...")
structure_engine = PPStructure(
    recovery=True,
    lang=OCR_LANGUAGE,
    use_gpu=USE_GPU,
    show_log=False,
)
print("PPStructure initialized successfully.")


def normalize_result(result):
    """
    Normalize PaddleOCR output to a consistent format:
    [ [ [coords, (text, confidence)], ... ], ... ]
    Works with both v2 and v3 APIs.
    """
    if result is None:
        return [[]]

    normalized = []
    for page in result:
        if page is None:
            normalized.append([])
            continue
        page_items = []
        for item in page:
            if isinstance(item, dict):
                # v3 format
                text = item.get("rec_text", item.get("text", ""))
                score = item.get("rec_score", item.get("score", 0.0))
                coords = item.get("dt_polys", item.get("poly", []))
                page_items.append([coords, [str(text), float(score)]])
            elif isinstance(item, (list, tuple)):
                # v2 format: [coords, (text, confidence)]
                page_items.append(item)
        normalized.append(page_items)
    return normalized


def run_ocr(img_input):
    """Run OCR and return normalized results."""
    try:
        result = ocr.ocr(img_input, cls=True)
        return normalize_result(result)
    except Exception as e:
        print(f"OCR engine error: {e}")
        traceback.print_exc()
        raise


def draw_ocr_boxes(image_bytes_or_path, ocr_results):
    """Draw bounding boxes on the image and return as base64 PNG."""
    try:
        if isinstance(image_bytes_or_path, str):
            img = cv2.imread(image_bytes_or_path)
        elif isinstance(image_bytes_or_path, bytes):
            nparr = np.frombuffer(image_bytes_or_path, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = image_bytes_or_path

        if img is None:
            return None

        for page in ocr_results:
            if not isinstance(page, list):
                continue
            for item in page:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                coords = item[0]
                text_info = item[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    confidence = float(text_info[1])
                else:
                    confidence = 0.0

                # Color based on confidence
                if confidence >= 0.9:
                    color = (0, 200, 0)   # Green
                elif confidence >= 0.7:
                    color = (0, 180, 255)  # Orange
                else:
                    color = (0, 0, 255)    # Red

                if coords and len(coords) >= 4:
                    pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], True, color, 2)

        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error drawing boxes: {e}")
        traceback.print_exc()
        return None


def run_structure_analysis(img_path_or_bytes, filename="document"):
    """
    Run PPStructure for layout analysis + recovery.
    Returns structured result, annotated image, and path to recovered DOCX.
    """
    try:
        # Read image
        if isinstance(img_path_or_bytes, str):
            img = cv2.imread(img_path_or_bytes)
            img_pil = Image.open(img_path_or_bytes).convert('RGB')
        else:
            nparr = np.frombuffer(img_path_or_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if img is None:
            raise ValueError("Failed to read image")

        # Run PPStructure
        result = structure_engine(img)

        # Build structured output
        structured_blocks = []
        full_text_parts = []
        for block in result:
            block_type = block.get('type', 'unknown')
            bbox = block.get('bbox', [])
            res = block.get('res', None)

            block_data = {
                'type': block_type,
                'bbox': bbox,
            }

            if block_type == 'table' and isinstance(res, dict):
                block_data['html'] = res.get('html', '')
                block_data['text'] = '[Table]'
                full_text_parts.append('[Table]')
            elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                # v3 format (paddleocr 2.9.x): list of {'text': ..., 'confidence': ..., 'text_region': ...}
                texts = []
                for item in res:
                    t = item.get('text', '')
                    if t:
                        texts.append(str(t))
                block_data['text'] = ' '.join(texts)
                full_text_parts.append(' '.join(texts))
            elif isinstance(res, (list, tuple)) and len(res) == 2:
                # v2 format: (boxes, [(text, conf), ...])
                boxes, text_results = res
                texts = []
                if isinstance(text_results, list):
                    for text_conf in text_results:
                        if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                            texts.append(str(text_conf[0]))
                        elif isinstance(text_conf, dict):
                            texts.append(str(text_conf.get('text', '')))
                        else:
                            texts.append(str(text_conf))
                block_data['text'] = ' '.join(texts)
                full_text_parts.append(' '.join(texts))
            else:
                block_data['text'] = ''

            structured_blocks.append(block_data)

        # Draw annotated image with colored bounding boxes
        try:
            annotated_img = img.copy()
            type_colors = {
                'text': (0, 200, 0),       # Green
                'title': (180, 105, 255),   # Purple
                'table': (255, 180, 0),     # Blue-ish
                'figure': (0, 140, 255),    # Orange
                'list': (180, 105, 255),    # Pink
                'header': (0, 200, 255),    # Yellow
                'footer': (160, 160, 160),  # Gray
            }
            for block in result:
                bbox = block.get('bbox', [])
                btype = block.get('type', 'text').lower()
                color = type_colors.get(btype, (0, 200, 0))
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    # Draw type label
                    label_size, _ = cv2.getTextSize(btype, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 6), (x1 + label_size[0] + 4, y1), color, -1)
                    cv2.putText(annotated_img, btype, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            _, buffer = cv2.imencode('.png', annotated_img)
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Warning: annotation drawing failed: {e}")
            traceback.print_exc()
            annotated_b64 = None

        # Layout recovery to DOCX
        docx_path = None
        try:
            from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx
            h, w, _ = img.shape
            basename = os.path.splitext(filename)[0]
            save_dir = os.path.join(OUTPUT_DIR, basename)
            os.makedirs(save_dir, exist_ok=True)
            # Save structure results first (creates cropped images needed by recovery)
            save_structure_res(result, save_dir, basename)
            res = sorted_layout_boxes(result, w)
            convert_info_docx(img, res, save_dir, basename)
            expected_docx = os.path.join(save_dir, f"{basename}.docx")
            if os.path.exists(expected_docx):
                docx_path = expected_docx
                print(f"DOCX saved: {docx_path}")
            else:
                # Check for _ocr.docx variant
                alt_docx = os.path.join(save_dir, f"{basename}_ocr.docx")
                if os.path.exists(alt_docx):
                    docx_path = alt_docx
                    print(f"DOCX saved: {docx_path}")
        except Exception as e:
            print(f"Warning: Layout recovery failed: {e}")
            traceback.print_exc()

        return {
            'blocks': structured_blocks,
            'full_text': '\n'.join(full_text_parts),
            'block_count': len(structured_blocks),
            'annotated_image': annotated_b64,
            'docx_path': docx_path,
        }
    except Exception as e:
        print(f"Structure analysis error: {e}")
        traceback.print_exc()
        raise


@router.get('/predict-by-path', summary="Recognize local image")
def predict_by_path(image_path: str):
    result = run_ocr(image_path)
    return RestfulModel(resultcode=200, message="Success", data=result)


@router.post('/predict-by-base64', summary="Recognize Base64 data")
def predict_by_base64(base64model: Base64PostModel):
    img = base64_to_ndarray(base64model.base64_str)
    result = run_ocr(img)
    return RestfulModel(resultcode=200, message="Success", data=result)


@router.post('/predict-by-file', summary="Recognize uploaded file")
async def predict_by_file(file: UploadFile):
    filename = (file.filename or "upload").lower()
    file_bytes = file.file.read()

    if filename.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff", ".webp")):
        # Image file
        img = bytes_to_ndarray(file_bytes)
        result = run_ocr(img)
    elif filename.endswith(".pdf"):
        # PDF file — save to temp, PaddleOCR handles multi-page natively
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            result = run_ocr(tmp_path)
        finally:
            os.remove(tmp_path)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supported formats: .jpg, .png, .jpeg, .bmp, .tiff, .webp, .pdf"
        )

    return RestfulModel(resultcode=200, message=file.filename or "upload", data=result)


@router.post('/structure', summary="Structure analysis with layout recovery & annotation")
async def structure_analysis(file: UploadFile):
    """
    Run PPStructure: layout analysis + OCR + recovery to DOCX.
    Returns structured blocks, annotated image (base64), and DOCX download path.
    """
    filename = (file.filename or "document").strip()
    file_bytes = file.file.read()

    if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff", ".webp")):
        result = run_structure_analysis(file_bytes, filename)
    elif filename.lower().endswith(".pdf"):
        # For PDFs, convert first page to image using PyMuPDF
        import fitz
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            doc = fitz.open(tmp_path)
            all_blocks = []
            all_text_parts = []
            annotated_images = []
            docx_path = None

            for page_idx, page in enumerate(doc):
                # Render page at 300 DPI
                mat = fitz.Matrix(300 / 72, 300 / 72)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                page_name = f"{os.path.splitext(filename)[0]}_page{page_idx + 1}"

                page_result = run_structure_analysis(img_bytes, page_name)
                all_blocks.extend(page_result['blocks'])
                all_text_parts.append(page_result['full_text'])
                if page_result.get('annotated_image'):
                    annotated_images.append(page_result['annotated_image'])
                if page_result.get('docx_path') and docx_path is None:
                    docx_path = page_result['docx_path']


            total_pages = len(doc)
            doc.close()
            result = {
                'blocks': all_blocks,
                'full_text': '\n\n--- Page Break ---\n\n'.join(all_text_parts),
                'block_count': len(all_blocks),
                'annotated_image': annotated_images[0] if annotated_images else None,
                'annotated_pages': annotated_images,
                'docx_path': docx_path,
                'page_count': total_pages,
            }
        finally:
            os.remove(tmp_path)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supported formats: .jpg, .png, .jpeg, .bmp, .tiff, .webp, .pdf"
        )

    # Build response
    response = {
        'status': 'ok',
        'filename': filename,
        'blocks': result['blocks'],
        'full_text': result['full_text'],
        'block_count': result['block_count'],
        'annotated_image': result.get('annotated_image'),
        'annotated_pages': result.get('annotated_pages', []),
        'has_docx': result.get('docx_path') is not None,
        'docx_filename': os.path.basename(result['docx_path']) if result.get('docx_path') else None,
    }
    return JSONResponse(content=response)


@router.get('/download/{filename}', summary="Download recovered DOCX")
def download_docx(filename: str):
    """Download a recovered DOCX file."""
    # Search for the file in output directory
    for root, dirs, files in os.walk(OUTPUT_DIR):
        if filename in files:
            filepath = os.path.join(root, filename)
            return FileResponse(
                filepath,
                media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                filename=filename,
            )
    raise HTTPException(status_code=404, detail=f"File {filename} not found")


@router.get('/predict-by-url', summary="Recognize image URL")
async def predict_by_url(imageUrl: str):
    response = requests.get(imageUrl, timeout=30)
    image_bytes = response.content

    if not (image_bytes[:3] == b"\xff\xd8\xff" or image_bytes[:8] == b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL must point to a JPG or PNG image"
        )

    img = bytes_to_ndarray(image_bytes)
    result = run_ocr(img)
    return RestfulModel(resultcode=200, message="Success", data=result)
