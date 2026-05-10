#!/usr/bin/env python3
"""
Standalone PaddleOCR microservice with PP-StructureV3.
Runs in its own conda env (ocr) to avoid dependency conflicts.
Main backend proxies requests to this service on port 8011.

Uses the official PPStructureV3 API from PaddleOCR 3.x:
  pipeline = PPStructureV3()
  output = pipeline.predict(input_file)
  for res in output:
      md_info = res.markdown   # {"markdown_texts": ..., "markdown_images": {...}}
  markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)
"""

import os, io, sys, base64, tempfile, traceback, gc, time, re, json
from pathlib import Path

# Disable oneDNN — PaddlePaddle 3.3.1 has a bug with ArrayAttribute conversion
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ.setdefault("MKLDNN_DISABLE", "1")
from typing import List, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import uvicorn

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "ocr")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_pipeline = None          # PPStructureV3 pipeline
_model_loaded = False
SERVICE_PORT = int(os.environ.get("OCR_SERVICE_PORT", "8011"))

app = FastAPI(title="PaddleOCR PPStructureV3 Service")


@app.on_event("startup")
def startup():
    global _pipeline, _model_loaded
    print(f"[OCR-Service] PaddleOCR PPStructureV3 service starting on port {SERVICE_PORT}")
    sys.stdout.flush()
    print("[OCR-Service] Auto-loading PPStructureV3 pipeline...")
    sys.stdout.flush()
    try:
        from paddleocr import PPStructureV3
        _pipeline = PPStructureV3()
        _model_loaded = True
        print("[OCR-Service] PPStructureV3 pipeline loaded and ready")
        sys.stdout.flush()
    except Exception as e:
        print(f"[OCR-Service] Failed to auto-load PPStructureV3: {e}")
        traceback.print_exc()


@app.post("/ocr/load")
def api_load():
    """No-op — pipeline auto-loads on startup. Kept for backward compat."""
    global _pipeline, _model_loaded
    if _pipeline is not None:
        return {"status": "already_loaded", "engine": "PPStructureV3"}
    # Fallback: try loading if startup failed
    try:
        from paddleocr import PPStructureV3
        _pipeline = PPStructureV3()
        _model_loaded = True
        return {"status": "loaded", "engine": "PPStructureV3"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/ocr/unload")
def api_unload():
    """No-op — pipeline stays loaded on CPU. Kept for backward compat."""
    return {"status": "no_op", "engine": "PPStructureV3"}


@app.get("/ocr/health")
def api_health():
    loaded = _pipeline is not None
    return {"status": "ok" if loaded else "standby", "engine": "PPStructureV3", "ready": loaded}


def pdf_to_images(pdf_bytes, dpi=200):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    imgs = []
    for page in doc:
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        imgs.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()
    return imgs


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _safe_get(obj, key, default=None):
    """Try bracket access, dict.get, then attribute access."""
    try:
        return obj[key]
    except (TypeError, KeyError, IndexError):
        pass
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _draw_labeled_annotations(res_obj) -> Optional[Image.Image]:
    """Draw annotations using official PaddleX color palette but with label names
    instead of order numbers. Mirrors the official _to_img() style exactly.
    """
    try:
        # Use official color palette from PaddleX
        from paddlex.inference.pipelines.layout_parsing.utils import get_show_color

        # Get the preprocessed image (same as official annotator uses)
        doc_pre = _safe_get(res_obj, "doc_preprocessor_res", None)
        if doc_pre is None:
            return None
        output_img = _safe_get(doc_pre, "output_img", None)
        if output_img is None:
            return None
        image = Image.fromarray(output_img[:, :, ::-1]).copy()
        draw = ImageDraw.Draw(image, "RGBA")

        # Font setup (same as official)
        font_size = int(0.018 * int(image.width)) + 2
        font = None
        try:
            from paddlex.inference.pipelines.layout_parsing.utils import PINGFANG_FONT
            font = ImageFont.truetype(PINGFANG_FONT.path, font_size, encoding="utf-8")
        except Exception:
            for fn in ["arial.ttf", "DejaVuSans.ttf", "Tahoma.ttf"]:
                try:
                    font = ImageFont.truetype(fn, font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        parsing_result = _safe_get(res_obj, "parsing_res_list", [])
        print(f"[OCR-Service] Annotation: found {len(parsing_result)} blocks")
        sys.stdout.flush()

        if not parsing_result:
            return image

        for block in parsing_result:
            bbox = _safe_get(block, "bbox", None)
            label = str(_safe_get(block, "label", "unknown"))
            if bbox is None:
                continue

            # Use official PaddleX color for this label
            fill_color = get_show_color(label, False)
            draw.rectangle(bbox, fill=fill_color)

            # Draw label name (e.g. "text", "figure", "table") instead of order number
            label_text = label
            text_position = (bbox[2] + 2, bbox[1] - font_size // 2)
            if int(image.width) - bbox[2] < font_size * len(label_text) * 0.6:
                text_position = (int(bbox[2] - font_size * len(label_text) * 0.6), bbox[1] - font_size // 2)
            draw.text(text_position, label_text, font=font, fill="red")

        print(f"[OCR-Service] Annotation: drew {len(parsing_result)} labeled blocks")
        sys.stdout.flush()
        return image
    except Exception as e:
        print(f"[OCR-Service] Label annotation failed: {e}")
        traceback.print_exc()
    return None


def md_to_pdf(md_text, output_path):
    try:
        import markdown as md_lib
        from xhtml2pdf import pisa
        html = md_lib.markdown(md_text, extensions=["tables","fenced_code","codehilite","toc"])
        css = '<style>body{font-family:Helvetica,Arial,sans-serif;font-size:11pt;line-height:1.6;margin:2cm}h1{font-size:18pt;border-bottom:2px solid #6366f1}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background:#f5f5f5}code{background:#f4f4f4;padding:2px 4px}pre{background:#f4f4f4;padding:12px}img{max-width:100%}</style>'
        with open(output_path, "wb") as f: pisa.CreatePDF(css + html, dest=f)
    except Exception as e:
        print(f"[OCR-Service] PDF gen failed: {e}")
        with open(output_path, "w", encoding="utf-8") as f: f.write(md_text)


def md_to_blocks(md: str) -> List[Dict]:
    """Parse markdown into block list for frontend display."""
    blocks, lines = [], md.splitlines()
    i, in_code, code_lines, in_table, table_lines = 0, False, [], False, []
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if in_code:
                blocks.append({"type": "code", "bbox": [], "text": "\n".join(code_lines)}); code_lines = []; in_code = False
            else:
                if table_lines: blocks.append({"type": "table", "bbox": [], "text": "\n".join(table_lines)}); table_lines = []; in_table = False
                in_code = True
            i += 1; continue
        if in_code: code_lines.append(line); i += 1; continue
        if "|" in line and line.strip().startswith("|"):
            if not in_table: in_table = True
            table_lines.append(line); i += 1; continue
        else:
            if in_table: blocks.append({"type": "table", "bbox": [], "text": "\n".join(table_lines)}); table_lines = []; in_table = False
        if line.startswith("#"):
            lvl = len(line) - len(line.lstrip("#"))
            blocks.append({"type": "title" if lvl == 1 else f"heading_{min(lvl,6)}", "bbox": [], "text": line.lstrip("#").strip()})
        elif line.strip().startswith("!["):
            blocks.append({"type": "figure", "bbox": [], "text": line.strip()})
        elif line.strip().startswith("$$"):
            eq = [line]; i += 1
            while i < len(lines) and not lines[i].strip().endswith("$$"): eq.append(lines[i]); i += 1
            if i < len(lines): eq.append(lines[i])
            blocks.append({"type": "formula", "bbox": [], "text": "\n".join(eq)})
        elif line.strip() in ["---","***","___"]:
            blocks.append({"type": "separator", "bbox": [], "text": ""})
        elif line.strip():
            blocks.append({"type": "text", "bbox": [], "text": line.strip()})
        i += 1
    if table_lines: blocks.append({"type": "table", "bbox": [], "text": "\n".join(table_lines)})
    if code_lines: blocks.append({"type": "code", "bbox": [], "text": "\n".join(code_lines)})
    return blocks


@app.post("/ocr/process")
async def api_process(file: UploadFile = File(...)):
    """
    Process a document using the official PPStructureV3 API.

    Follows the pattern from the official documentation:
      pipeline = PPStructureV3()
      output = pipeline.predict(input_file)
      for res in output:
          md_info = res.markdown
      markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)
    """
    try:
        file_bytes = await file.read()
        filename = file.filename or "upload.png"
        suffix = os.path.splitext(filename)[1].lower() or ".png"
        if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
            raise ValueError(f"Unsupported format '{suffix}'")

        if _pipeline is None:
            raise RuntimeError("PPStructureV3 failed to auto-load — check service logs")

        import uuid, hashlib, json as _json

        # --- Cache check: hash file content to avoid re-processing ---
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        cache_dir = os.path.join(OUTPUT_DIR, "_cache", file_hash)
        cache_result_path = os.path.join(cache_dir, "result.json")
        cache_md_path = os.path.join(cache_dir, "output.md")
        cache_ann_dir = os.path.join(cache_dir, "annotations")

        if os.path.isfile(cache_result_path) and os.path.isfile(cache_md_path):
            print(f"[OCR-Service] Cache HIT for {filename} (hash={file_hash})")
            sys.stdout.flush()
            with open(cache_md_path, "r", encoding="utf-8") as f:
                full_md = f.read()
            with open(cache_result_path, "r", encoding="utf-8") as f:
                cached = _json.load(f)
            ann_imgs = []
            if os.path.isdir(cache_ann_dir):
                for fn in sorted(os.listdir(cache_ann_dir)):
                    if fn.endswith(".png"):
                        with open(os.path.join(cache_ann_dir, fn), "rb") as f:
                            ann_imgs.append(base64.b64encode(f.read()).decode("utf-8"))
            blocks = md_to_blocks(full_md)
            valid_ann_imgs = [a for a in ann_imgs if a is not None]
            return {
                "markdown": full_md,
                "blocks": blocks,
                "annotated_images": valid_ann_imgs,
                "annotated_image": valid_ann_imgs[0] if valid_ann_imgs else None,
                "figure_images": [],
                "page_count": cached.get("page_count", 1),
                "markdown_file": f"{cached.get('job_id', file_hash)}/{os.path.splitext(filename)[0]}.md",
                "pdf_file": f"{cached.get('job_id', file_hash)}/{os.path.splitext(filename)[0]}.pdf",
                "block_count": len(blocks),
                "cached": True,
            }

        print(f"[OCR-Service] Cache MISS for {filename} (hash={file_hash})")
        sys.stdout.flush()

        job_id = uuid.uuid4().hex[:8]
        out_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(out_dir, exist_ok=True)

        # Save uploaded file to disk so PPStructureV3.predict can read it
        input_path = os.path.join(out_dir, f"input{suffix}")
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        t0 = time.time()
        print(f"[OCR-Service] Running PPStructureV3 on {filename}...")
        sys.stdout.flush()

        # For PDFs: convert pages to images first so PPStructureV3 does
        # actual OCR on rendered pixels instead of reading the (often corrupted)
        # embedded text layer. For images, pass directly.
        if suffix == ".pdf" and PYMUPDF_AVAILABLE:
            pdf_imgs = pdf_to_images(file_bytes, dpi=200)
            page_paths = []
            for pi, pimg in enumerate(pdf_imgs):
                ppath = os.path.join(out_dir, f"page_{pi}.png")
                pimg.save(ppath)
                page_paths.append(ppath)
            print(f"[OCR-Service] PDF split into {len(page_paths)} page images")
            sys.stdout.flush()
        else:
            page_paths = [input_path]

        markdown_list = []
        markdown_images_all = {}  # {path: PIL_image}
        ann_imgs = []

        for page_idx, page_path in enumerate(page_paths):
            print(f"[OCR-Service] Processing page {page_idx+1}/{len(page_paths)}...")
            sys.stdout.flush()

            # --- Official PPStructureV3 API ---
            output = list(_pipeline.predict(input=page_path))

            for res in output:
                # Get markdown from the official API
                md_info = res.markdown
                markdown_list.append(md_info)

                # Collect figure images
                md_images = md_info.get("markdown_images", {})
                if md_images:
                    markdown_images_all.update(md_images)

                # Draw labeled annotations (label names instead of order numbers)
                try:
                    ann_img = _draw_labeled_annotations(res)
                    if ann_img is not None:
                        ann_imgs.append(_pil_to_b64(ann_img))
                    else:
                        ann_imgs.append(None)
                except Exception as e:
                    print(f"[OCR-Service] Annotation failed for page {page_idx}: {e}")
                    ann_imgs.append(None)

        # Concatenate multi-page markdown using official API
        if markdown_list:
            full_md = _pipeline.concatenate_markdown_pages(markdown_list)
        else:
            full_md = ""

        # If concatenate_markdown_pages returns a dict, extract the text
        if isinstance(full_md, dict):
            full_md = full_md.get("markdown_texts", str(full_md))

        t1 = time.time()
        print(f"[OCR-Service] PPStructureV3 total: {t1-t0:.2f}s, pages={page_idx}, md_len={len(full_md)}")
        sys.stdout.flush()

        # Save markdown file (will be written after image embedding below)

        # Save figure images to disk AND build base64 replacements for markdown
        img_b64_map = {}  # {relative_path: base64_data_url}
        for img_path, img_data in markdown_images_all.items():
            try:
                save_path = os.path.join(out_dir, img_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if hasattr(img_data, 'save'):
                    img_data.save(save_path)
                    # Convert to base64 for embedding in markdown
                    buf = io.BytesIO()
                    img_data.save(buf, format="JPEG")
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    img_b64_map[img_path] = f"data:image/jpeg;base64,{b64}"
                elif isinstance(img_data, bytes):
                    with open(save_path, "wb") as f:
                        f.write(img_data)
                    b64 = base64.b64encode(img_data).decode("utf-8")
                    img_b64_map[img_path] = f"data:image/jpeg;base64,{b64}"
            except Exception as e:
                print(f"[OCR-Service] Failed to save figure {img_path}: {e}")

        # Replace relative image paths in markdown with base64 data URLs
        for rel_path, data_url in img_b64_map.items():
            # Handle both forward and backslash paths
            full_md = full_md.replace(f'src="{rel_path}"', f'src="{data_url}"')
            full_md = full_md.replace(f"src='{rel_path}'", f'src="{data_url}"')
            # Also try with backslashes
            win_path = rel_path.replace("/", "\\")
            full_md = full_md.replace(f'src="{win_path}"', f'src="{data_url}"')

        # Save markdown file (with embedded base64 images)
        base = os.path.splitext(filename)[0]
        md_path = os.path.join(out_dir, f"{base}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_md)

        # Generate PDF
        pdf_path = os.path.join(out_dir, f"{base}.pdf")
        md_to_pdf(full_md, pdf_path)

        # Parse blocks for frontend
        blocks = md_to_blocks(full_md)

        # Filter out None annotations
        valid_ann_imgs = [a for a in ann_imgs if a is not None]

        # --- Write cache for future requests ---
        try:
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(cache_ann_dir, exist_ok=True)
            # Cache markdown
            with open(cache_md_path, "w", encoding="utf-8") as f:
                f.write(full_md)
            # Cache annotations as PNG files
            for ai, ann_b64 in enumerate(valid_ann_imgs):
                ann_bytes = base64.b64decode(ann_b64)
                with open(os.path.join(cache_ann_dir, f"page_{ai:03d}.png"), "wb") as f:
                    f.write(ann_bytes)
            # Cache metadata
            with open(cache_result_path, "w", encoding="utf-8") as f:
                _json.dump({"job_id": job_id, "page_count": len(page_paths), "filename": filename}, f)
            print(f"[OCR-Service] Cached result for hash={file_hash}")
            sys.stdout.flush()
        except Exception as e:
            print(f"[OCR-Service] Cache write failed: {e}")

        return {
            "markdown": full_md,
            "blocks": blocks,
            "annotated_images": valid_ann_imgs,
            "annotated_image": valid_ann_imgs[0] if valid_ann_imgs else None,
            "figure_images": [],
            "page_count": len(page_paths),
            "markdown_file": f"{job_id}/{base}.md",
            "pdf_file": f"{job_id}/{base}.pdf",
            "block_count": len(blocks),
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ocr/download/{filepath:path}")
def api_download(filepath: str):
    full = os.path.join(OUTPUT_DIR, filepath)
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="File not found")
    mt = "application/pdf" if filepath.endswith(".pdf") else "text/markdown" if filepath.endswith(".md") else "application/octet-stream"
    return FileResponse(path=full, filename=os.path.basename(filepath), media_type=mt)


@app.get("/ocr/image/{filepath:path}")
def api_image(filepath: str):
    """Serve figure images from the output directory."""
    full = os.path.join(OUTPUT_DIR, filepath)
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="Image not found")
    ext = os.path.splitext(filepath)[1].lower()
    mt = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")
    return FileResponse(path=full, media_type=mt)


if __name__ == "__main__":
    print(f"Starting PaddleOCR PPStructureV3 service on port {SERVICE_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
