import math
from dataclasses import dataclass

from app.embeddings.model import EmbeddingModel
from app.ingestion.models import DocumentChunk


@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions")

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class SemanticSearch:
    """Search document chunks using embedding similarity."""

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.chunks: list[DocumentChunk] = []
        self.embeddings: list[list[float]] = []

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks and generate their embeddings."""

        self.chunks.extend(chunks)

        texts = [chunk.text for chunk in chunks]

        if texts:
            embeddings = self.embedding_model.embed_texts(texts)
            self.embeddings.extend(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Return the most similar chunks, optionally filtering weak matches."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if min_score is not None and not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1.0 and 1.0")

        if not self.chunks:
            return []

        query_embedding = self.embedding_model.embed_text(query)

        results = []

        for chunk, embedding in zip(
            self.chunks,
            self.embeddings,
        ):
            score = cosine_similarity(
                query_embedding,
                embedding,
            )

            if min_score is not None and score < min_score:
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]