
# ============================================================
# PROJECT 01 — AI DOCUMENT INTELLIGENCE
# STEP 09 — PRODUCTION READINESS
# PART 03 — DOCUMENT SECURITY TESTS
#
# FILE: tests/test_document_loader.py
#
# PURPOSE:
# 1. Test valid TXT, DOCX and PDF loading.
# 2. Verify document metadata.
# 3. Reject empty files.
# 4. Reject fake or corrupted PDF files.
# 5. Reject invalid DOCX files.
# 6. Reject invalid TXT encoding.
#
# NOTE:
# All test files are created inside tmp_path.
# Real project documents are not modified.
# ============================================================

from zipfile import ZipFile

import pymupdf
import pytest

from docx import Document

from app.ingestion.document_loader import load_document


# ============================================================
# TEST 01 — Load TXT document
# ============================================================

def test_txt_document_loading(tmp_path):

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "Employees receive 21 days of annual leave.",
        encoding="utf-8",
    )

    documents = load_document(str(file_path))

    assert len(documents) >= 1

    assert "21 days" in documents[0]["text"]

    assert documents[0]["source"] == "sample.txt"


# ============================================================
# TEST 02 — Load DOCX document
# ============================================================

def test_docx_document_loading(tmp_path):

    file_path = tmp_path / "sample.docx"

    document = Document()

    document.add_paragraph(
        "The project manager is Sarah Johnson."
    )

    document.save(str(file_path))

    documents = load_document(str(file_path))

    assert len(documents) >= 1

    assert "Sarah Johnson" in documents[0]["text"]

    assert documents[0]["source"] == "sample.docx"


# ============================================================
# TEST 03 — Load PDF document
# ============================================================

def test_pdf_document_loading(tmp_path):

    file_path = tmp_path / "sample.pdf"

    pdf = pymupdf.open()

    page = pdf.new_page()

    page.insert_text(
        (72, 72),
        "The project budget is 250000 USD.",
    )

    pdf.save(str(file_path))

    pdf.close()

    documents = load_document(str(file_path))

    assert len(documents) >= 1

    assert "250000 USD" in documents[0]["text"]

    assert documents[0]["source"] == "sample.pdf"


# ============================================================
# TEST 04 — Verify metadata structure
# ============================================================

def test_document_metadata(tmp_path):

    file_path = tmp_path / "metadata.txt"

    file_path.write_text(
        "Metadata validation test.",
        encoding="utf-8",
    )

    documents = load_document(str(file_path))

    for document in documents:

        assert "text" in document

        assert "source" in document

        assert "page" in document


# ============================================================
# TEST 05 — Reject empty document
# ============================================================

def test_reject_empty_document(tmp_path):

    file_path = tmp_path / "empty.txt"

    file_path.write_bytes(b"")

    with pytest.raises(
        ValueError,
        match="The uploaded file is empty",
    ):
        load_document(str(file_path))


# ============================================================
# TEST 06 — Reject fake PDF signature
# ============================================================

def test_reject_fake_pdf(tmp_path):

    file_path = tmp_path / "fake.pdf"

    file_path.write_bytes(
        b"This is not a real PDF document."
    )

    with pytest.raises(
        ValueError,
        match="Invalid PDF file format",
    ):
        load_document(str(file_path))


# ============================================================
# TEST 07 — Reject corrupted PDF content
# ============================================================

def test_reject_corrupted_pdf(tmp_path):

    file_path = tmp_path / "corrupted.pdf"

    file_path.write_bytes(
        b"%PDF-1.7\n"
        b"This is corrupted PDF content."
    )

    with pytest.raises(
        ValueError,
        match="The PDF file is invalid or corrupted",
    ):
        load_document(str(file_path))


# ============================================================
# TEST 08 — Reject invalid DOCX container
# ============================================================

def test_reject_invalid_docx(tmp_path):

    file_path = tmp_path / "invalid.docx"

    file_path.write_bytes(
        b"This is not a ZIP archive."
    )

    with pytest.raises(
        ValueError,
        match="Invalid or corrupted DOCX file",
    ):
        load_document(str(file_path))


# ============================================================
# TEST 09 — Reject incomplete DOCX structure
# ============================================================

def test_reject_incomplete_docx(tmp_path):

    file_path = tmp_path / "incomplete.docx"

    with ZipFile(file_path, "w") as archive:

        archive.writestr(
            "random.txt",
            "This ZIP file is not a DOCX document.",
        )

    with pytest.raises(
        ValueError,
        match="Invalid DOCX file structure",
    ):
        load_document(str(file_path))


# ============================================================
# TEST 10 — Reject invalid TXT encoding
# ============================================================

def test_reject_invalid_txt_encoding(tmp_path):

    file_path = tmp_path / "invalid.txt"

    file_path.write_bytes(
        b"\xff\xfe\xfa"
    )

    with pytest.raises(
        ValueError,
        match="TXT files must use UTF-8 encoding",
    ):
        load_document(str(file_path))
