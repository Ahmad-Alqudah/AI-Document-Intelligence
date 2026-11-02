
# ============================================================
# PROJECT 01 — AI DOCUMENT INTELLIGENCE
# STEP 08 — AUTOMATED TESTING
# PART 01 — CHUNKING TESTS
#
# FILE: tests/test_chunking.py
#
# PURPOSE:
# 1. Test document splitting.
# 2. Verify chunk size.
# 3. Verify metadata preservation.
# 4. Verify unique chunk IDs.
# ============================================================

from app.rag.chunking import split_documents


# ============================================================
# TEST 01 — Split a document
# ============================================================

def test_document_chunking():

    documents = [
        {
            "text": "Artificial intelligence is useful. " * 30,
            "source": "ai_document.txt",
            "page": None,
        }
    ]

    chunks = split_documents(
        documents,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1


# ============================================================
# TEST 02 — Validate chunk size
# ============================================================

def test_chunk_size():

    documents = [
        {
            "text": "Machine learning improves predictions. " * 30,
            "source": "ml_document.txt",
            "page": None,
        }
    ]

    chunks = split_documents(
        documents,
        chunk_size=100,
        chunk_overlap=20,
    )

    for chunk in chunks:
        assert len(chunk["text"]) <= 100


# ============================================================
# TEST 03 — Preserve source metadata
# ============================================================

def test_chunk_metadata():

    documents = [
        {
            "text": "Document processing and chunking test. " * 20,
            "source": "company_policy.txt",
            "page": 2,
        }
    ]

    chunks = split_documents(
        documents,
        chunk_size=100,
        chunk_overlap=20,
    )

    for chunk in chunks:

        assert chunk["source"] == "company_policy.txt"

        assert chunk["page"] == 2


# ============================================================
# TEST 04 — Validate unique chunk IDs
# ============================================================

def test_unique_chunk_ids():

    documents = [
        {
            "text": "Vector database testing. " * 40,
            "source": "vector_test.txt",
            "page": None,
        }
    ]

    chunks = split_documents(
        documents,
        chunk_size=100,
        chunk_overlap=20,
    )

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))
