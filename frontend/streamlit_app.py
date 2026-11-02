
# ============================================================
# PROJECT 01 — AI DOCUMENT INTELLIGENCE
# STEP 09 — PRODUCTION READINESS
# PART 01 — STREAMLIT ENVIRONMENT CONFIGURATION
#
# FILE: frontend/streamlit_app.py
#
# PURPOSE:
# 1. Load frontend configuration from .env.
# 2. Connect Streamlit to FastAPI.
# 3. Validate and upload PDF, DOCX and TXT documents.
# 4. Ask questions through the RAG API.
# 5. Display generated answers and retrieved sources.
# ============================================================

import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# STEP 1 — Load environment configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

API_BASE_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

DEFAULT_TOP_K = int(
    os.getenv("DEFAULT_TOP_K", "3")
)

MAX_UPLOAD_SIZE_MB = int(
    os.getenv("MAX_UPLOAD_SIZE_MB", "10")
)

MAX_FILE_SIZE = MAX_UPLOAD_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = ["pdf", "docx", "txt"]


# ============================================================
# STEP 2 — Streamlit application configuration
# ============================================================

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# STEP 3 — API communication functions
# ============================================================

def check_backend_health():

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )

        response.raise_for_status()

        return response.json().get("status") == "healthy"

    except (requests.RequestException, ValueError):
        return False


def upload_document(uploaded_file):

    uploaded_file.seek(0)

    response = requests.post(
        f"{API_BASE_URL}/upload",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file,
                uploaded_file.type
                or "application/octet-stream",
            )
        },
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


def ask_question(question, top_k):

    response = requests.post(
        f"{API_BASE_URL}/ask",
        json={
            "question": question,
            "top_k": top_k,
        },
        timeout=180,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# STEP 4 — Application header
# ============================================================

st.title("📄 AI Document Intelligence")

st.caption(
    "Ask questions about your documents using "
    "local RAG, FAISS and Ollama."
)


# ============================================================
# STEP 5 — Backend health status
# ============================================================

backend_available = check_backend_health()

if backend_available:

    st.success("FastAPI Backend: Connected")

else:

    st.error(
        "FastAPI Backend: Disconnected. "
        "Start the backend before using this application."
    )

    st.stop()


# ============================================================
# STEP 6 — Sidebar configuration
# ============================================================

with st.sidebar:

    st.header("Settings")

    top_k = st.slider(
        "Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
    )

    st.divider()

    st.write("**Backend:** FastAPI")
    st.write("**Vector Database:** FAISS")
    st.write("**LLM:** Qwen2.5 via Ollama")


# ============================================================
# STEP 7 — Document upload section
# ============================================================

st.subheader("Upload Documents")

uploaded_file = st.file_uploader(
    "Choose a PDF, DOCX or TXT document",
    type=ALLOWED_EXTENSIONS,
    accept_multiple_files=False,
)

st.caption(
    f"Maximum supported file size: {MAX_UPLOAD_SIZE_MB} MB"
)

if uploaded_file is not None:

    file_size = uploaded_file.size

    st.write(f"Selected file: **{uploaded_file.name}**")

    if file_size > MAX_FILE_SIZE:

        st.error(
            f"File exceeds the {MAX_UPLOAD_SIZE_MB} MB limit."
        )

    elif st.button(
        "Upload and Index Document",
        type="primary",
    ):

        with st.spinner(
            "Processing and indexing document..."
        ):

            try:

                result = upload_document(uploaded_file)

                st.success(
                    f"Document indexed: {result['filename']}"
                )

                st.write(
                    "Chunks Added:",
                    result["chunks_added"],
                )

                st.write(
                    "Total Indexed Chunks:",
                    result["total_indexed_chunks"],
                )

            except requests.HTTPError as error:

                response = error.response

                if response is not None:

                    try:
                        detail = response.json().get(
                            "detail",
                            "Upload failed.",
                        )

                    except ValueError:
                        detail = "Upload failed."

                else:
                    detail = "Upload failed."

                st.error(f"Upload Error: {detail}")

            except (requests.RequestException, ValueError):

                st.error(
                    "Could not upload the document. "
                    "Please check the backend connection."
                )


# ============================================================
# STEP 8 — Question answering section
# ============================================================

st.divider()

st.subheader("Ask Your Documents")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "How many annual leave days do employees receive?"
    ),
    height=100,
    max_chars=2000,
)

if st.button(
    "Generate Answer",
    type="primary",
    disabled=not question.strip(),
):

    with st.spinner(
        "Searching documents and generating answer..."
    ):

        try:

            result = ask_question(
                question=question,
                top_k=top_k,
            )

            st.subheader("Answer")

            st.write(result["answer"])

            st.subheader("Retrieved Sources")

            sources = result.get("sources", [])

            if not sources:

                st.info("No sources were retrieved.")

            for index, source in enumerate(
                sources,
                start=1,
            ):

                with st.expander(
                    f"Source {index}: {source['source']}"
                ):

                    st.write(
                        "**File:**",
                        source["source"],
                    )

                    st.write(
                        "**Page:**",
                        source["page"]
                        if source["page"] is not None
                        else "N/A",
                    )

                    st.write(
                        "**Chunk ID:**",
                        source["chunk_id"],
                    )

                    st.write(
                        "**Similarity Score:**",
                        source["score"],
                    )

        except requests.HTTPError as error:

            response = error.response

            if response is not None:

                try:
                    detail = response.json().get(
                        "detail",
                        "Question answering failed.",
                    )

                except ValueError:
                    detail = "Question answering failed."

            else:
                detail = "Question answering failed."

            st.error(f"API Error: {detail}")

        except (requests.RequestException, ValueError):

            st.error(
                "Could not generate an answer. "
                "Please check FastAPI and Ollama."
            )


# ============================================================
# STEP 9 — Footer
# ============================================================

st.divider()

st.caption(
    "AI Document Intelligence | "
    "FastAPI + Streamlit + FAISS + Ollama"
)
