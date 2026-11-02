
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Split extracted documents into smaller chunks
    while preserving source metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = []

    for document in documents:
        text = document["text"]

        text_chunks = splitter.split_text(text)

        for chunk_number, chunk_text in enumerate(
            text_chunks,
            start=1,
        ):
            chunks.append({
                "text": chunk_text,
                "source": document["source"],
                "page": document["page"],
                "chunk_id": len(chunks) + 1,
                "chunk_number": chunk_number,
            })

    return chunks
