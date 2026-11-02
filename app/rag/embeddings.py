
# ============================================================
# PROJECT 01 — AI DOCUMENT INTELLIGENCE
# STEP 09 — PRODUCTION READINESS
# PART 01 — EMBEDDING ENVIRONMENT CONFIGURATION
#
# FILE: app/rag/embeddings.py
#
# PURPOSE:
# 1. Load embedding model configuration from .env.
# 2. Initialize SentenceTransformer.
# 3. Generate normalized document embeddings.
# 4. Generate normalized query embeddings.
# ============================================================

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# ============================================================
# STEP 1 — Load environment configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# ============================================================
# STEP 2 — Embedding Service
# ============================================================

class EmbeddingService:

    def __init__(self, model_name: str = MODEL_NAME):

        self.model = SentenceTransformer(model_name)

    # ========================================================
    # STEP 3 — Document embeddings
    # ========================================================

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """
        Convert texts into normalized embeddings.
        """

        if not texts:
            raise ValueError(
                "The texts list cannot be empty."
            )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings

    # ========================================================
    # STEP 4 — Query embedding
    # ========================================================

    def encode_query(self, query: str) -> np.ndarray:
        """
        Convert a question into a normalized embedding.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding[0]
