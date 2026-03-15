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


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation (.:,), collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[.,:;!?()'\"-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Common abbreviation / alias expansions for fuzzy matching
_ALIASES: dict[str, list[str]] = {
    "dob": ["date of birth", "d o b", "birth date", "birthdate"],
    "date of birth": ["dob", "d o b", "birth date"],
    "name": ["full name", "applicant name", "candidate name"],
    "license no": ["licence no", "license number", "licence number",
                    "dl no", "dl number", "driving license no",
                    "driving licence no"],
    "present address": ["current address", "address", "residential address"],
    "phone": ["phone no", "phone number", "mobile", "mobile no", "contact no"],
    "email": ["email id", "email address", "e-mail"],
}


def _field_variants(field: str) -> list[str]:
    """Generate normalized variants of a user field for fuzzy matching."""
    norm = _normalize(field)
    variants = {norm}

    # Add known aliases
    for key, aliases in _ALIASES.items():
        nk = _normalize(key)
        if norm == nk or norm in [_normalize(a) for a in aliases]:
            variants.add(nk)
            for a in aliases:
                variants.add(_normalize(a))

    # Generate word-boundary-flexible patterns:
    # "License No" should match "LICENCE NO", "License No.", "License No :"
    result = []
    for v in variants:
        # Build a regex that matches each word with optional trailing punctuation
        words = v.split()
        # Allow single-char OCR variations: c↔e, i↔l, etc. via char-class on each word
        parts = []
        for w in words:
            # Exact word match (case-insensitive), allow trailing dots/colons
            parts.append(re.escape(w) + r"[.,:;]?")
        pattern = r"\s+".join(parts)
        result.append(pattern)

    return result


def _find_field_position(text: str, field: str) -> re.Match | None:
    """Find where a field label appears in text using fuzzy matching."""
    variants = _field_variants(field)

    for variant_pattern in variants:
        # Match the field label followed by a separator
        pattern = rf"(?im)({variant_pattern})\s*[:=\-—\t]\s*"
        m = re.search(pattern, text)
        if m:
            return m

    # Fallback: field label at start of line without explicit separator
    for variant_pattern in variants:
        pattern = rf"(?im)^[ \t]*({variant_pattern})\s{{2,}}"
        m = re.search(pattern, text)
        if m:
            return m

    # Last resort: field label followed by newline (value on next line)
    for variant_pattern in variants:
        pattern = rf"(?im)({variant_pattern})\s*$"
        m = re.search(pattern, text)
        if m:
            return m

    return None


def _looks_like_field_label(line: str) -> bool:
    """Heuristic: does this line look like the start of a new field?"""
    stripped = line.strip()
    if not stripped:
        return False
    # A line that contains a colon/equals with text before it is likely a new field
    if re.match(r"^[A-Za-z][A-Za-z\s.]{1,40}\s*[:=]\s*\S", stripped):
        return True
    # ALL-CAPS label followed by separator
    if re.match(r"^[A-Z][A-Z\s.]{1,40}\s*[:=\-]\s*", stripped):
        return True
    return False


def extract_field_value(text: str, field: str) -> str:
    """Extract a field value with normalization, fuzzy matching, and multi-line support."""
    match = _find_field_position(text, field)
    if not match:
        return ""

    # Get text after the matched field label
    after = text[match.end():]

    # Capture the first line of the value
    lines = after.split("\n")
    if not lines:
        return ""

    first_line = lines[0].strip()

    # If first line is empty, the value might be on the next line(s)
    start_idx = 0
    if not first_line and len(lines) > 1:
        start_idx = 1
        first_line = lines[1].strip()

    # Collect multi-line value: keep reading lines until we hit an empty line,
    # a new field label, or end of text
    value_parts = []
    for i in range(start_idx, len(lines)):
        line = lines[i]
        stripped = line.strip()

        # Stop at empty line (but only after we have some content)
        if not stripped and value_parts:
            break

        # Stop if this line looks like a new field label (but not the first value line)
        if value_parts and _looks_like_field_label(stripped):
            break

        if stripped:
            value_parts.append(stripped)

        # For single-line fields (value on same line as label), just take the first line
        if start_idx == 0 and value_parts:
            # Check if next line could be a continuation (no separator = likely continuation)
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if not next_stripped or _looks_like_field_label(next_stripped):
                    break
                # If next line is just data (no colon/separator), it's a continuation
                if not re.search(r"[:=]", next_stripped):
                    continue
                else:
                    break
            else:
                break

    value = " ".join(value_parts).strip()
    # Clean up any trailing separators or garbage
    value = re.sub(r"\s*[:=\-]\s*$", "", value)
    return value


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
