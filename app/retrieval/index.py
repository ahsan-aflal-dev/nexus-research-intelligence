
from pathlib import Path
import json

import numpy as np

from app.embeddings.model import EmbeddingModel
from app.ingestion.models import DocumentChunk
from app.retrieval.search import SearchResult, cosine_similarity


class VectorIndex:
    """Persistent vector index for NEXUS document chunks."""

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embed and append document chunks to the index."""

        if not chunks:
            return

        vectors = np.asarray(
            self.embedding_model.embed_texts(
                [chunk.text for chunk in chunks]
            ),
            dtype=np.float32,
        )

        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError("Embedding count does not match chunk count.")

        if self.embeddings is not None:
            if vectors.shape[1] != self.embeddings.shape[1]:
                raise ValueError("Embedding dimensions do not match.")

        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])

        self.chunks.extend(chunks)

    def remove_source(self, source: str | Path) -> int:
        """Remove all chunks whose source matches the supplied path."""

        source_key = str(source)
        keep_positions = [
            position
            for position, chunk in enumerate(self.chunks)
            if chunk.source != source_key
        ]

        removed_count = len(self.chunks) - len(keep_positions)

        if removed_count == 0:
            return 0

        self.chunks = [self.chunks[position] for position in keep_positions]

        if self.embeddings is not None:
            if self.chunks:
                self.embeddings = self.embeddings[keep_positions]
            else:
                dimension = self.embeddings.shape[1]
                self.embeddings = np.empty(
                    (0, dimension),
                    dtype=np.float32,
                )

        return removed_count

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Search stored vectors and optionally filter weak matches."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if min_score is not None and not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1.0 and 1.0")

        if not self.chunks or self.embeddings is None:
            return []

        query_embedding = self.embedding_model.embed_text(query)
        results = []

        for chunk, embedding in zip(self.chunks, self.embeddings):
            score = cosine_similarity(
                query_embedding,
                embedding.tolist(),
            )

            if min_score is not None and score < min_score:
                continue

            results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def save(self, directory: str | Path) -> None:
        """Persist vectors and chunk metadata, including an empty index."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.embeddings is None:
            dimension = getattr(self.embedding_model, "dimension", None)

            if not isinstance(dimension, int) or dimension <= 0:
                raise ValueError(
                    "Cannot save an empty index without a valid "
                    "embedding model dimension."
                )

            embeddings = np.empty((0, dimension), dtype=np.float32)
        else:
            embeddings = self.embeddings

        if len(self.chunks) != len(embeddings):
            raise ValueError(
                "Index is inconsistent: chunk and embedding counts differ."
            )

        np.save(directory / "embeddings.npy", embeddings)

        metadata = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "page_number": chunk.page_number,
            }
            for chunk in self.chunks
        ]

        with (directory / "metadata.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        embedding_model: EmbeddingModel,
    ) -> "VectorIndex":
        """Load a previously saved vector index."""

        directory = Path(directory)
        embeddings_path = directory / "embeddings.npy"
        metadata_path = directory / "metadata.json"

        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Vector index not found: {embeddings_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Vector metadata not found: {metadata_path}"
            )

        embeddings = np.load(embeddings_path, allow_pickle=False)

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        if embeddings.ndim != 2:
            raise ValueError("Invalid index: embeddings must be a 2D array.")

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Invalid index: embedding and metadata counts differ."
            )

        dimension = getattr(embedding_model, "dimension", None)
        if dimension is not None and embeddings.shape[1] != dimension:
            raise ValueError(
                "Embedding dimension does not match the configured model."
            )

        index = cls(embedding_model)
        index.embeddings = embeddings.astype(np.float32, copy=False)
        index.chunks = [
            DocumentChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                source=item["source"],
                page_number=item["page_number"],
            )
            for item in metadata
        ]

        return index
