"""Upload parsing for documents, spreadsheets, and images."""

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from docx import Document
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FILES_PER_REQUEST = 5
MAX_EXTRACTED_CHARS = 2_000_000

ALLOWED_EXTENSIONS: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ParseError(ValueError):
    """The upload could not be read as the declared file type."""


@dataclass(frozen=True)
class ParsedUpload:
    name: str
    mime: str
    size: int
    extracted_text: str
    image_b64: str | None
    pages: list[tuple[int | None, str]]


def extension_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def safe_filename(filename: str | None) -> str:
    raw = Path(filename or "").name.strip()
    if not raw or raw in {".", ".."}:
        raise ParseError("A file name is required")
    return raw[:255]


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _cap(text: str) -> str:
    if len(text) <= MAX_EXTRACTED_CHARS:
        return text
    return text[:MAX_EXTRACTED_CHARS] + "\n\n[truncated]"


def _pdf_pages(data: bytes) -> list[tuple[int | None, str]]:
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise ParseError("Could not read the PDF") from exc
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ParseError("Could not read the encrypted PDF") from exc
    pages: list[tuple[int | None, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.info("pdf page extract failed page=%s error=%s", index, exc.__class__.__name__)
            text = ""
        cleaned = text.strip()
        if cleaned:
            pages.append((index, cleaned))
    return pages


def _docx_text(data: bytes) -> str:
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        raise ParseError("Could not read the Word document") from exc
    parts: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _csv_text(data: bytes) -> str:
    raw = decode_text(data)
    try:
        frame = pd.read_csv(io.StringIO(raw))
    except Exception as exc:
        raise ParseError("Could not read the CSV file") from exc
    return frame.to_csv(index=False)


def _xlsx_text(data: bytes) -> str:
    try:
        workbook = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    except Exception as exc:
        raise ParseError("Could not read the Excel file") from exc
    sections: list[str] = []
    for sheet in workbook.sheet_names:
        frame = workbook.parse(sheet)
        body = frame.to_csv(index=False)
        sections.append(f"--- sheet: {sheet} ---\n{body}--- end sheet ---")
    if not sections:
        raise ParseError("The Excel file has no sheets")
    return "\n".join(sections)


def _image_b64(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ParseError("Could not read the image") from exc
    return base64.b64encode(data).decode("ascii")


def parse_upload(filename: str, data: bytes) -> ParsedUpload:
    """Validate an allowed upload and extract text or image bytes."""
    name = safe_filename(filename)
    extension = extension_of(name)
    mime = ALLOWED_EXTENSIONS.get(extension)
    if mime is None:
        raise ParseError(
            "Unsupported file type. Upload pdf, docx, txt, md, csv, xlsx, jpg, jpeg, png, or webp."
        )
    if len(data) == 0:
        raise ParseError(f"{name} is empty")
    if len(data) > MAX_FILE_BYTES:
        raise ParseError(f"{name} exceeds the 20MB limit")

    image_b64: str | None = None
    pages: list[tuple[int | None, str]] = []
    if extension == ".pdf":
        pages = _pdf_pages(data)
        extracted = "\n\n".join(text for _, text in pages)
    elif extension == ".docx":
        extracted = _docx_text(data)
        if extracted.strip():
            pages = [(None, extracted.strip())]
    elif extension in {".txt", ".md"}:
        extracted = decode_text(data)
        if extracted.strip():
            pages = [(None, extracted.strip())]
    elif extension == ".csv":
        extracted = _csv_text(data)
        if extracted.strip():
            pages = [(None, extracted.strip())]
    elif extension == ".xlsx":
        extracted = _xlsx_text(data)
        if extracted.strip():
            pages = [(None, extracted.strip())]
    elif extension in IMAGE_EXTENSIONS:
        image_b64 = _image_b64(data)
        extracted = ""
    else:
        raise ParseError("Unsupported file type")

    return ParsedUpload(
        name=name,
        mime=mime,
        size=len(data),
        extracted_text=_cap(extracted),
        image_b64=image_b64,
        pages=[(page, _cap(text)) for page, text in pages],
    )
