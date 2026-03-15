import io
import re
import time
from typing import List

import fitz  # PyMuPDF
import openpyxl
import pytesseract
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


def extract_full_text(contents: bytes) -> str:
    """Extract all text from a PDF, using OCR for scanned pages."""
    doc = fitz.open(stream=contents, filetype="pdf")
    all_text = []
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text()
            if len(text.strip()) < OCR_TEXT_THRESHOLD:
                try:
                    text = ocr_page(page)
                except Exception:
                    pass
            all_text.append(text)
    finally:
        doc.close()
    return "\n".join(all_text)


def extract_field_value(text: str, field: str) -> str:
    """Try multiple patterns to find a field value in text."""
    escaped = re.escape(field)
    patterns = [
        # "Field Name: value" or "Field Name : value"
        rf"(?i){escaped}\s*:\s*(.+?)(?:\n|$)",
        # "Field Name - value"
        rf"(?i){escaped}\s*-\s*(.+?)(?:\n|$)",
        # "Field Name  value" (multiple spaces)
        rf"(?i){escaped}\s{{2,}}(.+?)(?:\n|$)",
        # "Field Name\nvalue" (field on one line, value on next)
        rf"(?i){escaped}\s*\n\s*(.+?)(?:\n|$)",
        # "Field Name = value"
        rf"(?i){escaped}\s*=\s*(.+?)(?:\n|$)",
        # Tabular: "Field Name\tvalue"
        rf"(?i){escaped}\t+(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return ""


async def _extract_rows(files: List[UploadFile], fields: str) -> tuple[list[str], list[dict]]:
    """Shared logic: parse fields JSON, extract text from each file, return (field_list, rows)."""
    import json

    try:
        field_list = json.loads(fields)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid fields format. Send a JSON array of strings.")

    if not field_list or not isinstance(field_list, list):
        raise HTTPException(status_code=400, detail="Provide at least one field name.")

    rows = []
    for file in files:
        ext = file_extension(file.filename)
        contents = await file.read()

        if ext in PDF_EXTENSIONS:
            text = extract_full_text(contents)
        elif ext in IMAGE_EXTENSIONS:
            try:
                image = Image.open(io.BytesIO(contents))
                text = pytesseract.image_to_string(image)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"OCR failed for {file.filename}: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file: {file.filename}")

        row = {"Filename": file.filename}
        for field in field_list:
            row[field] = extract_field_value(text, field)
        rows.append(row)

    return field_list, rows


@app.post("/extract-fields")
async def extract_fields(
    files: List[UploadFile] = File(...),
    fields: str = Form(...),
):
    """Extract user-defined fields from multiple PDFs and return JSON preview."""
    field_list, rows = await _extract_rows(files, fields)
    return {"fields": field_list, "rows": rows}


@app.post("/extract-fields-xlsx")
async def extract_fields_xlsx(
    files: List[UploadFile] = File(...),
    fields: str = Form(...),
):
    """Extract user-defined fields and return an Excel spreadsheet."""
    from openpyxl.styles import Alignment, Font, PatternFill

    field_list, rows = await _extract_rows(files, fields)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Extracted Data"

    headers = ["Filename"] + field_list
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="2563EB", end_color="2563EB", fill_type="solid")

    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for row_idx, row_data in enumerate(rows, 2):
        for col_idx, header in enumerate(headers, 1):
            ws.cell(row=row_idx, column=col_idx, value=row_data.get(header, ""))

    for col_idx, header in enumerate(headers, 1):
        max_len = len(header)
        for row_idx in range(2, len(rows) + 2):
            val = str(ws.cell(row=row_idx, column=col_idx).value or "")
            if len(val) > max_len:
                max_len = len(val)
        ws.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = min(max_len + 4, 50)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=extracted_data.xlsx"},
    )
