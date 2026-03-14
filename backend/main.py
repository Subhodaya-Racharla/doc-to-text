import io
import time

import fitz  # PyMuPDF
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="doc-to-text API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://frontend-tau-rose-13.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
OCR_TEXT_THRESHOLD = 50  # characters; below this a page is treated as scanned


def file_extension(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot != -1 else ""


def ocr_page(page: fitz.Page) -> str:
    """Render a PDF page to a PIL Image and run Tesseract OCR on it."""
    pixmap = page.get_pixmap(dpi=300)
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    return pytesseract.image_to_string(image)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    if file_extension(file.filename) not in PDF_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()

    try:
        doc = fitz.open(stream=contents, filetype="pdf")
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Could not open file. It may be corrupted or not a valid PDF.",
        )

    pages = []
    start = time.perf_counter()
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text()

            if len(text.strip()) < OCR_TEXT_THRESHOLD:
                try:
                    text = ocr_page(page)
                    method = "ocr"
                except pytesseract.TesseractNotFoundError:
                    raise HTTPException(
                        status_code=500,
                        detail="Tesseract is not installed or not found in PATH. Please install it on the server.",
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"OCR failed on page {i + 1}: {str(e)}")
            else:
                method = "direct"

            pages.append({
                "page_number": i + 1,
                "text": text,
                "method": method,
                "character_count": len(text),
            })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading page: {str(e)}")
    finally:
        doc.close()

    extraction_time = round(time.perf_counter() - start, 2)

    return {
        "filename": file.filename,
        "total_pages": len(pages),
        "total_characters": sum(p["character_count"] for p in pages),
        "extraction_time_seconds": extraction_time,
        "pages": pages,
    }


@app.post("/extract-text-image")
async def extract_text_image(file: UploadFile = File(...)):
    if file_extension(file.filename) not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Accepted formats: PNG, JPG, JPEG, TIFF, BMP.",
        )

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  # confirm it's a valid image
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Could not open image. The file may be corrupted or is not a valid image.",
        )

    try:
        # Re-open after verify() — verify() exhausts the stream
        image = Image.open(io.BytesIO(contents))
        text = pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Tesseract is not installed or not found in PATH. Please install it on the server.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

    return {
        "filename": file.filename,
        "text": text,
    }
