
# ============================================================
# PROJECT 01 — AI DOCUMENT INTELLIGENCE
# STEP 09 — PRODUCTION READINESS
# PART 03 — DOCUMENT VALIDATION & SAFE LOADING
#
# FILE: app/ingestion/document_loader.py
#
# PURPOSE:
# 1. Validate supported document extensions.
# 2. Reject empty and invalid files.
# 3. Validate PDF and DOCX file structure.
# 4. Extract text and document metadata.
# 5. Handle corrupted documents safely.
# ============================================================

from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pymupdf
from docx import Document


# ============================================================
# STEP 1 — Supported document formats
# ============================================================

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ============================================================
# STEP 2 — Validate document file
# ============================================================

def validate_document(path: Path) -> None:
    """
    Validate file existence, size and basic format.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            "The uploaded file is empty."
        )

    # Validate PDF signature
    if extension == ".pdf":

        with path.open("rb") as file:
            signature = file.read(5)

        if signature != b"%PDF-":
            raise ValueError(
                "Invalid PDF file format."
            )

    # Validate DOCX container and required document part
    elif extension == ".docx":

        try:

            with ZipFile(path, "r") as archive:

                if archive.testzip() is not None:
                    raise ValueError(
                        "The DOCX file is corrupted."
                    )

                if "word/document.xml" not in archive.namelist():
                    raise ValueError(
                        "Invalid DOCX file structure."
                    )

                if "[Content_Types].xml" not in archive.namelist():
                    raise ValueError(
                        "Invalid DOCX file structure."
                    )

        except (BadZipFile, OSError) as error:
            raise ValueError(
                "Invalid or corrupted DOCX file."
            ) from error


# ============================================================
# STEP 3 — Load document
# ============================================================

def load_document(file_path: str) -> list[dict]:
    """
    Extract text and metadata from a supported document.
    """

    path = Path(file_path)

    # Validate before reading
    validate_document(path)

    extension = path.suffix.lower()

    results = []

    # ========================================================
    # STEP 3.1 — Read PDF
    # ========================================================

    if extension == ".pdf":

        try:

            with pymupdf.open(path) as pdf:

                for page_number, page in enumerate(
                    pdf,
                    start=1,
                ):

                    text = page.get_text("text").strip()

                    if text:

                        results.append({
                            "text": text,
                            "source": path.name,
                            "page": page_number,
                        })

        except (pymupdf.FileDataError, ValueError) as error:

            raise ValueError(
                "The PDF file is invalid or corrupted."
            ) from error

    # ========================================================
    # STEP 3.2 — Read DOCX
    # ========================================================

    elif extension == ".docx":

        try:

            document = Document(str(path))

            paragraphs = [
                paragraph.text.strip()
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ]

            text = "\n".join(paragraphs)

            if text:

                results.append({
                    "text": text,
                    "source": path.name,
                    "page": None,
                })

        except (BadZipFile, ValueError, KeyError) as error:

            raise ValueError(
                "The DOCX file is invalid or corrupted."
            ) from error

    # ========================================================
    # STEP 3.3 — Read TXT
    # ========================================================

    elif extension == ".txt":

        try:

            text = path.read_text(
                encoding="utf-8-sig"
            ).strip()

        except UnicodeDecodeError as error:

            raise ValueError(
                "TXT files must use UTF-8 encoding."
            ) from error

        if text:

            results.append({
                "text": text,
                "source": path.name,
                "page": None,
            })

    return results
