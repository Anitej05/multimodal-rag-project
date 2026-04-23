# -*- coding: utf-8 -*-
"""
PP-StructureV3 Document Digitization Router
Uses PaddleX pipeline for state-of-the-art layout analysis,
OCR, table recognition, formula recognition, and Markdown output.
"""

import os
import io
import sys
import base64
import tempfile
import traceback
import shutil

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse

router = APIRouter(prefix="/ocr")

# ── Output directory ──
OUTPUT_DIR = "/app/ocr_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Global pipeline instance ──
_pipeline = None


def get_pipeline():
    """Lazy-init the PP-StructureV3 pipeline (only once)."""
    global _pipeline
    if _pipeline is None:
        print("Initializing PP-StructureV3 pipeline (this may download models on first run)...")
        sys.stdout.flush()
        from paddlex import create_pipeline
        _pipeline = create_pipeline(pipeline="PP-StructureV3")
        print("PP-StructureV3 pipeline initialized successfully.")
        sys.stdout.flush()
    return _pipeline


# ── Warm up on import ──
try:
    get_pipeline()
except Exception as e:
    print(f"Warning: PP-StructureV3 init deferred — will retry on first request. Error: {e}")
    sys.stdout.flush()


# ════════════════════════════════════════════════════
#  Helper: Convert a PIL Image to base64 PNG string
# ════════════════════════════════════════════════════
def pil_to_base64(pil_img):
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ════════════════════════════════════════════════════
#  Helper: Extract annotated image from result
# ════════════════════════════════════════════════════
def get_annotated_image(res, viz_dir):
    """
    Try multiple methods to get the annotated/visualization image:
    1. res.img dict → look for layout visualization
    2. save_to_img → read from disk
    Returns base64 string or None.
    """
    # Method 1: Try res.img attribute
    try:
        img_data = res.img
        if isinstance(img_data, dict):
            # Try common keys
            for key in ["layout", "ocr", "table", "res"]:
                if key in img_data and img_data[key] is not None:
                    pil_img = img_data[key]
                    if isinstance(pil_img, Image.Image):
                        return pil_to_base64(pil_img)
            # Try first available image
            for key, val in img_data.items():
                if isinstance(val, Image.Image):
                    return pil_to_base64(val)
        elif isinstance(img_data, Image.Image):
            return pil_to_base64(img_data)
        elif isinstance(img_data, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
            return pil_to_base64(pil_img)
    except Exception as e:
        print(f"  res.img failed: {e}")

    # Method 2: Try save_to_img
    try:
        page_viz_dir = os.path.join(viz_dir, "viz_page")
        os.makedirs(page_viz_dir, exist_ok=True)
        res.save_to_img(save_path=page_viz_dir)
        # Find any saved image
        for root, dirs, files in os.walk(page_viz_dir):
            for f in sorted(files):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(root, f)
                    with open(img_path, "rb") as fh:
                        b64 = base64.b64encode(fh.read()).decode("utf-8")
                    shutil.rmtree(page_viz_dir, ignore_errors=True)
                    return b64
        shutil.rmtree(page_viz_dir, ignore_errors=True)
    except Exception as e:
        print(f"  save_to_img failed: {e}")

    return None


# ════════════════════════════════════════════════════
#  Helper: Extract structured blocks from JSON result
# ════════════════════════════════════════════════════
def extract_blocks(res):
    """Parse the JSON result into structured blocks with type, bbox, text."""
    blocks = []
    try:
        json_data = res.json
        if not isinstance(json_data, dict):
            return blocks

        # PP-StructureV3 JSON has different possible structures
        # Try to get layout elements
        parsing_result = json_data.get("parsing_result", [])
        if isinstance(parsing_result, list):
            for elem in parsing_result:
                if isinstance(elem, dict):
                    blocks.append({
                        "type": elem.get("label", elem.get("type", "text")),
                        "bbox": elem.get("bbox", []),
                        "text": elem.get("text", elem.get("content", "")),
                    })

        # If no parsing_result, try layout_det_result
        if not blocks:
            layout_result = json_data.get("layout_det_result", json_data.get("layout_result", {}))
            if isinstance(layout_result, dict):
                det_boxes = layout_result.get("boxes", [])
                for box_info in det_boxes:
                    if isinstance(box_info, dict):
                        blocks.append({
                            "type": box_info.get("label", "text"),
                            "bbox": box_info.get("coordinate", box_info.get("bbox", [])),
                            "text": box_info.get("text", ""),
                        })
    except Exception as e:
        print(f"  Block extraction failed: {e}")
        traceback.print_exc()

    return blocks


# ════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════

@router.get("/health")
def health():
    """Health check endpoint."""
    engine_ready = _pipeline is not None
    return {
        "status": "ok",
        "engine": "PP-StructureV3 (PaddleX)",
        "ready": engine_ready
    }


@router.post("/structure")
async def structure_analysis(file: UploadFile = File(...)):
    """
    Run PP-StructureV3 on an uploaded image or PDF.
    Returns: markdown text, annotated images, structured blocks.
    """
    pipe = get_pipeline()
    filename = file.filename or "upload.png"
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Save uploaded file to temp path
    suffix = os.path.splitext(filename)[1].lower() or ".png"
    valid_suffixes = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    if suffix not in valid_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Supported: {', '.join(valid_suffixes)}"
        )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    viz_dir = tempfile.mkdtemp()

    try:
        print(f"Processing: {filename} ({len(file_bytes)} bytes)")
        sys.stdout.flush()

        # ── Run PP-StructureV3 pipeline ──
        output = pipe.predict(
            input=tmp_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        all_markdown_parts = []
        all_blocks = []
        annotated_pages = []
        page_count = 0

        for res in output:
            page_count += 1
            print(f"  Page {page_count}: processing...")
            sys.stdout.flush()

            # ── 1. Extract Markdown (with embedded images) ──
            md_text = ""
            try:
                md_data = res.markdown
                if isinstance(md_data, dict):
                    md_text = md_data.get("markdown_texts", "")
                    # Embed images as base64 data URIs so they render everywhere
                    md_images = md_data.get("markdown_images", {})
                    if md_images and isinstance(md_images, dict):
                        for img_name, pil_img in md_images.items():
                            try:
                                if isinstance(pil_img, Image.Image):
                                    buf = io.BytesIO()
                                    pil_img.save(buf, format="PNG")
                                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                                    data_uri = f"data:image/png;base64,{b64}"
                                    # Replace markdown image references
                                    md_text = md_text.replace(f"]({img_name})", f"]({data_uri})")
                                    md_text = md_text.replace(f'src="{img_name}"', f'src="{data_uri}"')
                                    print(f"    Embedded image: {img_name}")
                            except Exception as img_err:
                                print(f"    Image embed failed for {img_name}: {img_err}")
                elif isinstance(md_data, str):
                    md_text = md_data
                else:
                    md_text = str(md_data) if md_data else ""
            except Exception as e:
                print(f"  Markdown extraction failed: {e}")
                traceback.print_exc()

            all_markdown_parts.append(md_text)

            # ── 2. Extract annotated visualization ──
            ann_b64 = get_annotated_image(res, viz_dir)
            if ann_b64:
                annotated_pages.append(ann_b64)

            # ── 3. Extract structured blocks ──
            page_blocks = extract_blocks(res)
            all_blocks.extend(page_blocks)

            print(f"  Page {page_count}: done (md={len(md_text)} chars, blocks={len(page_blocks)}, viz={'yes' if ann_b64 else 'no'})")
            sys.stdout.flush()

        # ── Save Markdown file for download ──
        basename = os.path.splitext(filename)[0]
        save_dir = os.path.join(OUTPUT_DIR, basename)
        os.makedirs(save_dir, exist_ok=True)

        combined_markdown = "\n\n---\n\n".join(all_markdown_parts) if len(all_markdown_parts) > 1 else (all_markdown_parts[0] if all_markdown_parts else "")

        md_filename = f"{basename}.md"
        md_path = os.path.join(save_dir, md_filename)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(combined_markdown)
        print(f"Markdown saved: {md_path}")
        sys.stdout.flush()

        # ── Also generate a styled PDF from the markdown ──
        pdf_filename = f"{basename}_structured.pdf"
        pdf_path = os.path.join(save_dir, pdf_filename)
        try:
            _generate_pdf_from_markdown(combined_markdown, pdf_path)
            has_pdf = True
            print(f"PDF saved: {pdf_path}")
        except Exception as pdf_err:
            has_pdf = False
            print(f"Warning: PDF generation failed: {pdf_err}")
            traceback.print_exc()
        sys.stdout.flush()

        return {
            "status": "ok",
            "filename": filename,
            "markdown": combined_markdown,
            "full_text": combined_markdown,
            "annotated_image": annotated_pages[0] if annotated_pages else None,
            "annotated_pages": annotated_pages,
            "block_count": len(all_blocks) or page_count,
            "blocks": all_blocks,
            "page_count": page_count,
            "has_markdown": True,
            "has_pdf": has_pdf,
            "markdown_filename": md_filename,
            "pdf_filename": pdf_filename if has_pdf else None,
        }

    except Exception as e:
        print(f"ERROR processing {filename}: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        # Cleanup temp files
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        shutil.rmtree(viz_dir, ignore_errors=True)


# ════════════════════════════════════════════════════
#  Markdown → PDF generation
# ════════════════════════════════════════════════════

_PDF_CSS = """
@page {
    size: A4;
    margin: 2cm;
}
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a1a;
}
h1 { font-size: 22pt; margin-top: 0.5em; margin-bottom: 0.3em; color: #1a1a2e; border-bottom: 2px solid #4a4e69; padding-bottom: 4pt; }
h2 { font-size: 17pt; margin-top: 0.5em; color: #22223b; }
h3 { font-size: 14pt; margin-top: 0.4em; color: #4a4e69; }
h4, h5, h6 { font-size: 12pt; margin-top: 0.3em; }
p { margin: 0.3em 0; }
table { border-collapse: collapse; width: 100%; margin: 0.5em 0; }
th, td { border: 1px solid #9a8c98; padding: 6pt 8pt; text-align: left; font-size: 10pt; }
th { background-color: #f2e9e4; font-weight: bold; }
code { font-family: monospace; background: #f0f0f0; padding: 1pt 3pt; font-size: 10pt; }
pre { background: #f5f5f5; padding: 8pt; border-radius: 4pt; overflow: auto; font-size: 9pt; }
blockquote { border-left: 3pt solid #9a8c98; margin-left: 0; padding-left: 10pt; color: #555; }
hr { border: none; border-top: 1px solid #ccc; margin: 1em 0; }
img { max-width: 100%; }
"""


def _generate_pdf_from_markdown(md_text, output_path):
    """Convert markdown text to a styled PDF file."""
    import markdown as md_lib
    from xhtml2pdf import pisa

    # Convert Markdown → HTML
    html_body = md_lib.markdown(
        md_text,
        extensions=["tables", "fenced_code", "codehilite", "toc", "nl2br"]
    )

    # Wrap in a full HTML document with styling
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{_PDF_CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    with open(output_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

    if pisa_status.err:
        raise RuntimeError(f"PDF generation returned {pisa_status.err} error(s)")


@router.get("/download/{filename}")
def download_file(filename: str):
    """Download a generated file (markdown, pdf, etc)."""
    for root, dirs, files in os.walk(OUTPUT_DIR):
        if filename in files:
            file_path = os.path.join(root, filename)
            if filename.endswith(".md"):
                media_type = "text/markdown"
            elif filename.endswith(".pdf"):
                media_type = "application/pdf"
            else:
                media_type = "application/octet-stream"
            return FileResponse(
                path=file_path,
                filename=filename,
                media_type=media_type
            )

    raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
