
# ============================================================
# PROJECT 01 — AI DOCUMENT INTELLIGENCE
# STEP 09 — PRODUCTION READINESS
# PART 02 — FAISS PERSISTENCE & RECOVERY TESTS
#
# FILE: tests/test_vector_store.py
#
# PURPOSE:
# 1. Test FAISS index creation.
# 2. Test semantic similarity search.
# 3. Test saving and reloading.
# 4. Test top_k behavior.
# 5. Verify old index survives a write failure.
# 6. Verify backup restoration after activation failure.
# 7. Prevent overwriting an existing backup.
#
# NOTE:
# All persistence tests use temporary directories.
# ============================================================

from pathlib import Path

import faiss
import pytest

from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore


# ============================================================
# STEP 1 — Reusable test data
# ============================================================

@pytest.fixture
def sample_chunks():

    return [
        {
            "text": (
                "Employees receive 21 days of "
                "paid annual leave per year."
            ),
            "source": "employee_handbook.txt",
            "page": None,
            "chunk_id": 0,
            "chunk_number": 1,
        },
        {
            "text": (
                "The company provides technical "
                "support from Sunday to Thursday."
            ),
            "source": "company_policy.txt",
            "page": None,
            "chunk_id": 1,
            "chunk_number": 1,
        },
        {
            "text": (
                "All employees must use multi-factor "
                "authentication for company accounts."
            ),
            "source": "security_policy.txt",
            "page": None,
            "chunk_id": 2,
            "chunk_number": 1,
        },
    ]


# ============================================================
# STEP 2 — Embedding service fixture
# ============================================================

@pytest.fixture(scope="session")
def embedding_service():

    return EmbeddingService()


# ============================================================
# STEP 3 — Vector store fixture
# ============================================================

@pytest.fixture
def vector_store(embedding_service, sample_chunks):

    store = VectorStore(embedding_service)

    store.add_documents(sample_chunks)

    return store


# ============================================================
# TEST 01 — FAISS index creation
# ============================================================

def test_faiss_index_creation(vector_store):

    assert vector_store.index is not None

    assert vector_store.index.ntotal == 3

    assert len(vector_store.chunks) == 3


# ============================================================
# TEST 02 — Semantic similarity search
# ============================================================

def test_semantic_search(vector_store):

    results = vector_store.search(
        query="How many annual leave days do employees get?",
        top_k=1,
    )

    assert len(results) == 1

    assert results[0]["source"] == "employee_handbook.txt"

    assert "21 days" in results[0]["text"]


# ============================================================
# TEST 03 — Save and reload FAISS index
# ============================================================

def test_faiss_save_and_reload(
    vector_store,
    embedding_service,
    tmp_path,
):

    index_folder = tmp_path / "test_faiss_index"

    vector_store.save(str(index_folder))

    assert (index_folder / "index.faiss").exists()

    assert (index_folder / "chunks.json").exists()

    reloaded_store = VectorStore(embedding_service)

    reloaded_store.load(str(index_folder))

    assert reloaded_store.index.ntotal == 3

    assert len(reloaded_store.chunks) == 3

    results = reloaded_store.search(
        query="How many days of annual leave are provided?",
        top_k=1,
    )

    assert len(results) == 1

    assert results[0]["source"] == "employee_handbook.txt"


# ============================================================
# TEST 04 — Validate top_k
# ============================================================

def test_top_k_results(vector_store):

    results = vector_store.search(
        query="What are the company policies?",
        top_k=2,
    )

    assert len(results) == 2

    assert all(
        "text" in result
        and "source" in result
        and "score" in result
        for result in results
    )


# ============================================================
# TEST 05 — Preserve old index on write failure
# ============================================================

def test_save_preserves_old_index_on_write_failure(
    vector_store,
    embedding_service,
    tmp_path,
    monkeypatch,
):

    index_folder = tmp_path / "test_faiss_index"

    # Save a valid original index
    vector_store.save(str(index_folder))

    original_index_bytes = (
        index_folder / "index.faiss"
    ).read_bytes()

    original_chunks_bytes = (
        index_folder / "chunks.json"
    ).read_bytes()

    # Simulate failure while writing new FAISS data
    def fail_write_index(*args, **kwargs):
        raise OSError("Simulated FAISS write failure")

    monkeypatch.setattr(
        faiss,
        "write_index",
        fail_write_index,
    )

    with pytest.raises(
        OSError,
        match="Simulated FAISS write failure",
    ):
        vector_store.save(str(index_folder))

    # Original files must remain unchanged
    assert (
        index_folder / "index.faiss"
    ).read_bytes() == original_index_bytes

    assert (
        index_folder / "chunks.json"
    ).read_bytes() == original_chunks_bytes

    # No unfinished temporary directory remains
    assert not list(
        tmp_path.glob("test_faiss_index.tmp-*")
    )

    assert not (
        tmp_path / "test_faiss_index.backup"
    ).exists()


# ============================================================
# TEST 06 — Restore backup on activation failure
# ============================================================

def test_save_restores_backup_on_activation_failure(
    vector_store,
    tmp_path,
    monkeypatch,
):

    index_folder = tmp_path / "test_faiss_index"

    vector_store.save(str(index_folder))

    original_index_bytes = (
        index_folder / "index.faiss"
    ).read_bytes()

    original_chunks_bytes = (
        index_folder / "chunks.json"
    ).read_bytes()

    original_rename = Path.rename

    # Fail only when activating the new temporary folder
    def fail_activation(self, target):

        if (
            self.name.startswith("test_faiss_index.tmp-")
            and Path(target) == index_folder
        ):
            raise OSError("Simulated activation failure")

        return original_rename(self, target)

    monkeypatch.setattr(
        Path,
        "rename",
        fail_activation,
    )

    with pytest.raises(
        OSError,
        match="Simulated activation failure",
    ):
        vector_store.save(str(index_folder))

    # The previous index must be restored
    assert index_folder.exists()

    assert (
        index_folder / "index.faiss"
    ).read_bytes() == original_index_bytes

    assert (
        index_folder / "chunks.json"
    ).read_bytes() == original_chunks_bytes

    assert not (
        tmp_path / "test_faiss_index.backup"
    ).exists()

    assert not list(
        tmp_path.glob("test_faiss_index.tmp-*")
    )


# ============================================================
# TEST 07 — Do not overwrite existing backup
# ============================================================

def test_save_does_not_overwrite_existing_backup(
    vector_store,
    tmp_path,
):

    index_folder = tmp_path / "test_faiss_index"

    backup_folder = tmp_path / "test_faiss_index.backup"

    vector_store.save(str(index_folder))

    # Simulate an existing recovery backup
    backup_folder.mkdir()

    marker_file = backup_folder / "recovery_marker.txt"

    marker_file.write_text(
        "Keep this backup",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="A FAISS backup already exists",
    ):
        vector_store.save(str(index_folder))

    # Backup must remain untouched
    assert marker_file.read_text(
        encoding="utf-8",
    ) == "Keep this backup"

    # Current index must still exist
    assert (index_folder / "index.faiss").exists()

    assert (index_folder / "chunks.json").exists()

    assert not list(
        tmp_path.glob("test_faiss_index.tmp-*")
    )
