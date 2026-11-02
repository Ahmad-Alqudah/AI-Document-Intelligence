
# ============================================================
# PROJECT 01 — AI DOCUMENT INTELLIGENCE
# STEP 09 — PRODUCTION READINESS
# PART 04 — FASTAPI UPLOAD SECURITY
#
# FILE: app/main.py
#
# PURPOSE:
# 1. Configure and initialize the RAG application.
# 2. Validate uploaded documents and file sizes.
# 3. Return appropriate HTTP error codes.
# 4. Clean up failed uploads.
# 5. Preserve the existing FAISS index on failure.
# 6. Support first-time startup without an existing FAISS index.
# 7. Keep existing API endpoints and response formats.
# ============================================================

import logging
import os
import threading

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.ingestion.document_loader import load_document
from app.rag.chunking import split_documents
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore
from app.rag.pipeline import RAGPipeline


# ============================================================
# STEP 1 — Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "faiss_index"

UPLOAD_DIRECTORY = PROJECT_ROOT / "data" / "uploads"

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:0.5b",
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

DEFAULT_TOP_K = int(
    os.getenv("DEFAULT_TOP_K", "3")
)

MAX_UPLOAD_SIZE_MB = int(
    os.getenv("MAX_UPLOAD_SIZE_MB", "10")
)

MAX_FILE_SIZE = MAX_UPLOAD_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

logger = logging.getLogger(__name__)


# ============================================================
# STEP 2 — Request / Response Models
# ============================================================

class AskRequest(BaseModel):

    question: Annotated[
        str,
        Field(min_length=1, max_length=2000),
    ]

    top_k: Annotated[
        int,
        Field(ge=1, le=10),
    ] = DEFAULT_TOP_K


class SourceResponse(BaseModel):

    source: str
    page: int | None
    chunk_id: str | int
    score: float


class AskResponse(BaseModel):

    question: str
    answer: str
    sources: list[SourceResponse]


class UploadResponse(BaseModel):

    filename: str
    status: str
    chunks_added: int
    total_indexed_chunks: int


# ============================================================
# STEP 3 — Initialize RAG Components
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Loading AI Document Intelligence components."
    )

    UPLOAD_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_service = EmbeddingService()

    vector_store = VectorStore(
        embedding_service
    )

    # --------------------------------------------------------
    # 3.1 — Check whether a persisted FAISS index exists
    # --------------------------------------------------------

    index_file = (
        FAISS_INDEX_PATH / "index.faiss"
    )

    chunks_file = (
        FAISS_INDEX_PATH / "chunks.json"
    )

    index_exists = index_file.is_file()
    chunks_exist = chunks_file.is_file()

    # --------------------------------------------------------
    # 3.2 — Load existing index
    # --------------------------------------------------------

    if index_exists and chunks_exist:

        logger.info(
            "Existing FAISS index found. Loading index."
        )

        vector_store.load(
            str(FAISS_INDEX_PATH)
        )

        logger.info(
            "FAISS index loaded successfully. Total chunks: %s",
            len(vector_store.chunks),
        )

    # --------------------------------------------------------
    # 3.3 — First-time startup without an index
    # --------------------------------------------------------

    elif not index_exists and not chunks_exist:

        logger.info(
            "No existing FAISS index found. "
            "Starting with an empty vector store."
        )

    # --------------------------------------------------------
    # 3.4 — Detect incomplete index files
    # --------------------------------------------------------

    else:

        raise RuntimeError(
            "Incomplete FAISS index detected. "
            "Both index.faiss and chunks.json "
            "must exist together."
        )

    # --------------------------------------------------------
    # 3.5 — Initialize RAG Pipeline
    # --------------------------------------------------------

    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        model_name=OLLAMA_MODEL,
    )

    app.state.embedding_service = embedding_service

    app.state.vector_store = vector_store

    app.state.rag_pipeline = rag_pipeline

    app.state.index_lock = threading.RLock()

    logger.info(
        "RAG Pipeline initialized successfully."
    )

    yield

    app.state.rag_pipeline = None

    logger.info(
        "AI Document Intelligence API stopped."
    )


# ============================================================
# STEP 4 — FastAPI Application
# ============================================================

app = FastAPI(
    title="AI Document Intelligence API",
    description=(
        "Local document question-answering system "
        "using RAG, FAISS and Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# STEP 5 — Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "application": "AI Document Intelligence",
        "version": "1.0.0",
        "status": "running",
    }


# ============================================================
# STEP 6 — Health Endpoint
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "AI Document Intelligence API",
    }


# ============================================================
# STEP 7 — Question Answering Endpoint
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask_question(request: AskRequest):

    try:

        with app.state.index_lock:

            result = app.state.rag_pipeline.answer_question(
                question=request.question,
                top_k=request.top_k,
            )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:

        logger.warning(
            "RAG service unavailable: %s",
            error,
        )

        raise HTTPException(
            status_code=503,
            detail="The question-answering service is unavailable.",
        ) from error

    except Exception as error:

        logger.exception(
            "Unexpected question-answering failure."
        )

        raise HTTPException(
            status_code=500,
            detail="Question answering failed.",
        ) from error


# ============================================================
# STEP 8 — Document Upload Endpoint
# ============================================================

@app.post(
    "/upload",
    response_model=UploadResponse,
)
def upload_document(
    file: UploadFile = File(...),
):

    original_filename = Path(
        (file.filename or "").replace("\\", "/")
    ).name

    extension = Path(
        original_filename
    ).suffix.lower()

    file_path = None

    try:

        # ----------------------------------------------------
        # 8.1 — Validate file extension
        # ----------------------------------------------------

        if extension not in ALLOWED_EXTENSIONS:

            raise HTTPException(
                status_code=400,
                detail="Only PDF, DOCX and TXT files are supported.",
            )

        # ----------------------------------------------------
        # 8.2 — Create a safe server-side filename
        # ----------------------------------------------------

        stored_filename = (
            f"{uuid4().hex}{extension}"
        )

        UPLOAD_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path = (
            UPLOAD_DIRECTORY / stored_filename
        )

        # ----------------------------------------------------
        # 8.3 — Save file with a strict size limit
        # ----------------------------------------------------

        total_size = 0

        with file_path.open("wb") as destination:

            while True:

                data = file.file.read(
                    1024 * 1024
                )

                if not data:
                    break

                total_size += len(data)

                if total_size > MAX_FILE_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "File size exceeds the "
                            f"{MAX_UPLOAD_SIZE_MB} MB limit."
                        ),
                    )

                destination.write(data)

        # ----------------------------------------------------
        # 8.4 — Validate and extract document content
        # ----------------------------------------------------

        try:

            documents = load_document(
                str(file_path)
            )

        except ValueError as error:

            raise HTTPException(
                status_code=400,
                detail=str(error),
            ) from error

        if not documents:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded document contains "
                    "no readable text."
                ),
            )

        # ----------------------------------------------------
        # 8.5 — Preserve the original filename in metadata
        # ----------------------------------------------------

        for document in documents:

            document["source"] = original_filename

        # ----------------------------------------------------
        # 8.6 — Split extracted text into chunks
        # ----------------------------------------------------

        new_chunks = split_documents(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        if not new_chunks:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded document contains "
                    "no readable text."
                ),
            )

        # ----------------------------------------------------
        # 8.7 — Build and save an updated FAISS index
        # ----------------------------------------------------

        with app.state.index_lock:

            current_store = app.state.vector_store

            existing_chunks = [
                chunk.copy()
                for chunk in current_store.chunks
            ]

            for chunk in new_chunks:

                chunk["chunk_id"] = len(
                    existing_chunks
                )

                existing_chunks.append(
                    chunk
                )

            # Build a separate index before replacing the old one

            updated_store = VectorStore(
                app.state.embedding_service
            )

            updated_store.add_documents(
                existing_chunks
            )

            # Save first; only publish the new index on success

            updated_store.save(
                str(FAISS_INDEX_PATH)
            )

            app.state.vector_store = updated_store

            app.state.rag_pipeline.vector_store = (
                updated_store
            )

            total_chunks = (
                updated_store.index.ntotal
            )

        return UploadResponse(
            filename=original_filename,
            status="indexed",
            chunks_added=len(new_chunks),
            total_indexed_chunks=total_chunks,
        )

    # --------------------------------------------------------
    # 8.8 — Preserve expected HTTP errors
    # --------------------------------------------------------

    except HTTPException:

        raise

    # --------------------------------------------------------
    # 8.9 — Hide unexpected internal errors
    # --------------------------------------------------------

    except Exception as error:

        logger.exception(
            "Unexpected document upload or indexing failure."
        )

        raise HTTPException(
            status_code=500,
            detail="Document processing or indexing failed.",
        ) from error

    # --------------------------------------------------------
    # 8.10 — Always clean up the temporary upload
    # --------------------------------------------------------

    finally:

        if file_path is not None:

            try:

                file_path.unlink(
                    missing_ok=True
                )

            except OSError:

                logger.warning(
                    "Could not remove temporary upload: %s",
                    file_path,
                )

        file.file.close()
