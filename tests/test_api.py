
# ============================================================
# PROJECT 01 — AI DOCUMENT INTELLIGENCE
# STEP 09 — PRODUCTION READINESS
# PART 04 — FASTAPI AUTOMATED TESTS
#
# FILE: tests/test_api.py
#
# PURPOSE:
# 1. Test root and health endpoints.
# 2. Test valid RAG questions.
# 3. Test Pydantic input validation.
# 4. Test safe Ollama failure handling.
# 5. Test document upload.
# 6. Test unsupported file extensions.
#
# IMPORTANT:
# Mock services are used to avoid running Ollama,
# loading real embeddings, or modifying the real index.
# ============================================================

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import app.main as main_module


# ============================================================
# STEP 1 — Mock RAG Pipeline
# ============================================================

class FakeRAGPipeline:

    def __init__(self, vector_store):

        self.vector_store = vector_store
        self.should_fail = False

    def answer_question(self, question, top_k=3):

        if self.should_fail:
            raise RuntimeError("Ollama unavailable")

        return {
            "question": question,
            "answer": "21 days",
            "sources": [
                {
                    "source": "employee_handbook.txt",
                    "page": None,
                    "chunk_id": 0,
                    "score": 0.95,
                }
            ],
        }


# ============================================================
# STEP 2 — Mock FAISS Vector Store
# ============================================================

class FakeVectorStore:

    def __init__(self, embedding_service):

        self.embedding_service = embedding_service
        self.chunks = []

        self.index = MagicMock()
        self.index.ntotal = 0

    def load(self, folder_path):

        return None

    def add_documents(self, chunks):

        self.chunks = list(chunks)
        self.index.ntotal = len(self.chunks)

    def save(self, folder_path):

        return None


# ============================================================
# STEP 3 — Create isolated API test client
# ============================================================

@pytest.fixture
def client(monkeypatch, tmp_path):

    # Mock embedding service
    monkeypatch.setattr(
        main_module,
        "EmbeddingService",
        lambda: object(),
    )

    # Mock vector store
    monkeypatch.setattr(
        main_module,
        "VectorStore",
        FakeVectorStore,
    )

    # Mock RAG pipeline
    monkeypatch.setattr(
        main_module,
        "RAGPipeline",
        lambda vector_store, model_name: FakeRAGPipeline(
            vector_store
        ),
    )

    # Use temporary upload directory
    monkeypatch.setattr(
        main_module,
        "UPLOAD_DIRECTORY",
        tmp_path / "uploads",
    )

    # Use temporary FAISS index path
    monkeypatch.setattr(
        main_module,
        "FAISS_INDEX_PATH",
        tmp_path / "faiss_index",
    )

    # Start FastAPI lifespan
    with TestClient(main_module.app) as test_client:

        yield test_client


# ============================================================
# TEST 01 — Root endpoint
# ============================================================

def test_root_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["status"] == "running"


# ============================================================
# TEST 02 — Health endpoint
# ============================================================

def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


# ============================================================
# TEST 03 — Valid RAG question
# ============================================================

def test_ask_valid_question(client):

    response = client.post(
        "/ask",
        json={
            "question": "How many annual leave days?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["answer"] == "21 days"

    assert result["sources"][0]["source"] == (
        "employee_handbook.txt"
    )


# ============================================================
# TEST 04 — Invalid top_k
# ============================================================

def test_ask_invalid_top_k(client):

    response = client.post(
        "/ask",
        json={
            "question": "What is the leave policy?",
            "top_k": 50,
        },
    )

    assert response.status_code == 422


# ============================================================
# TEST 05 — Empty question
# ============================================================

def test_ask_empty_question(client):

    response = client.post(
        "/ask",
        json={
            "question": "",
            "top_k": 3,
        },
    )

    assert response.status_code == 422


# ============================================================
# TEST 06 — Safe Ollama failure handling
# ============================================================

def test_ask_ollama_failure(client):

    pipeline = main_module.app.state.rag_pipeline

    # Simulate Ollama service failure
    pipeline.should_fail = True

    response = client.post(
        "/ask",
        json={
            "question": "What is the annual leave policy?",
            "top_k": 3,
        },
    )

    # Service unavailable
    assert response.status_code == 503

    # API returns a safe public error message
    assert response.json()["detail"] == (
        "The question-answering service is unavailable."
    )

    # Internal error details must not be exposed
    assert "Ollama unavailable" not in (
        response.json()["detail"]
    )


# ============================================================
# TEST 07 — Valid TXT upload
# ============================================================

def test_upload_valid_txt(client):

    response = client.post(
        "/upload",
        files={
            "file": (
                "test_policy.txt",
                b"Employees receive 21 days of annual leave.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["filename"] == "test_policy.txt"

    assert result["status"] == "indexed"

    assert result["chunks_added"] >= 1

    assert result["total_indexed_chunks"] >= 1


# ============================================================
# TEST 08 — Unsupported file extension
# ============================================================

def test_upload_invalid_extension(client):

    response = client.post(
        "/upload",
        files={
            "file": (
                "malicious.exe",
                b"Invalid executable content",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400

    assert "Only PDF, DOCX and TXT" in (
        response.json()["detail"]
    )
