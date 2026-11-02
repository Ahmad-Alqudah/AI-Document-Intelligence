
# ============================================================
# PROJECT 01 — AI DOCUMENT INTELLIGENCE
# STEP 09 — PRODUCTION READINESS
# PART 02 — SAFE FAISS INDEX PERSISTENCE
#
# FILE: app/rag/vector_store.py
#
# PURPOSE:
# 1. Build a FAISS index from document chunks.
# 2. Perform semantic similarity search.
# 3. Save index and metadata safely.
# 4. Validate persisted index files.
# 5. Restore the previous index if replacement fails.
# ============================================================

import json
import shutil

from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np

from app.rag.embeddings import EmbeddingService


# ============================================================
# STEP 1 — Vector Store
# ============================================================

class VectorStore:

    def __init__(
        self,
        embedding_service: EmbeddingService,
    ):
        self.embedding_service = embedding_service
        self.index = None
        self.chunks = []

    # ========================================================
    # STEP 2 — Build FAISS index
    # ========================================================

    def add_documents(
        self,
        chunks: list[dict],
    ) -> None:
        """
        Create a FAISS index from document chunks.
        """

        if not chunks:
            raise ValueError(
                "Chunks list cannot be empty."
            )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedding_service.encode_texts(
            texts
        )

        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)

        index.add(embeddings)

        self.index = index
        self.chunks = chunks.copy()

    # ========================================================
    # STEP 3 — Semantic search
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Retrieve the most similar document chunks.
        """

        if self.index is None:
            raise RuntimeError(
                "Vector index has not been created."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        query_embedding = (
            self.embedding_service.encode_query(query)
        )

        query_embedding = np.ascontiguousarray(
            query_embedding.reshape(1, -1),
            dtype=np.float32,
        )

        k = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):

            if index_id < 0:
                continue

            chunk = self.chunks[
                int(index_id)
            ].copy()

            chunk["score"] = float(score)

            results.append(chunk)

        return results

    # ========================================================
    # STEP 4 — Validate persisted index
    # ========================================================

    def _validate_saved_index(
        self,
        folder: Path,
    ) -> None:
        """
        Check FAISS and metadata consistency.
        """

        index_path = folder / "index.faiss"
        chunks_path = folder / "chunks.json"

        index = faiss.read_index(
            str(index_path)
        )

        with chunks_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            chunks = json.load(file)

        if not isinstance(chunks, list):
            raise ValueError(
                "Chunk metadata must be a list."
            )

        if index.ntotal != len(chunks):
            raise ValueError(
                "FAISS index and chunk metadata "
                "are inconsistent."
            )

        expected_dimension = (
            self.embedding_service.model
            .get_embedding_dimension()
        )

        if index.d != expected_dimension:
            raise ValueError(
                "FAISS index dimension does not "
                "match the embedding model."
            )

    # ========================================================
    # STEP 5 — Save index with backup
    # ========================================================

    def save(
        self,
        folder_path: str,
    ) -> None:
        """
        Save new files before replacing the old index.
        """

        if self.index is None:
            raise RuntimeError(
                "No index available to save."
            )

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                "FAISS index and chunks are inconsistent."
            )

        folder = Path(folder_path)

        folder.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        unique_id = uuid4().hex

        temporary_folder = folder.with_name(
            f"{folder.name}.tmp-{unique_id}"
        )

        backup_folder = folder.with_name(
            f"{folder.name}.backup"
        )

        try:

            # Write new index into a temporary directory
            temporary_folder.mkdir(
                parents=True,
                exist_ok=False,
            )

            faiss.write_index(
                self.index,
                str(
                    temporary_folder / "index.faiss"
                ),
            )

            with (
                temporary_folder / "chunks.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    self.chunks,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            # Validate the new files before activation
            self._validate_saved_index(
                temporary_folder
            )

            # Do not overwrite an existing recovery backup
            if backup_folder.exists():
                raise RuntimeError(
                    "A FAISS backup already exists. "
                    "Review it before saving again."
                )

            # Preserve the previous index
            if folder.exists():
                folder.rename(
                    backup_folder
                )

            # Activate the new index
            try:

                temporary_folder.rename(
                    folder
                )

            except Exception:

                # Restore previous index if activation fails
                if backup_folder.exists():
                    backup_folder.rename(
                        folder
                    )

                raise

            # Remove backup only after successful activation
            if backup_folder.exists():
                shutil.rmtree(
                    backup_folder
                )

        finally:

            # Remove unfinished temporary files
            if temporary_folder.exists():
                shutil.rmtree(
                    temporary_folder
                )

    # ========================================================
    # STEP 6 — Load saved index
    # ========================================================

    def load(
        self,
        folder_path: str,
    ) -> None:
        """
        Load and validate a persisted FAISS index.
        """

        folder = Path(folder_path)

        # Validate before changing in-memory state
        self._validate_saved_index(
            folder
        )

        index = faiss.read_index(
            str(folder / "index.faiss")
        )

        with (
            folder / "chunks.json"
        ).open(
            "r",
            encoding="utf-8",
        ) as file:

            chunks = json.load(file)

        self.index = index
        self.chunks = chunks
