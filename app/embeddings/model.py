from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Wrapper around the NEXUS embedding model."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """Convert one piece of text into an embedding vector."""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple pieces of text into embedding vectors."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return the size of the embedding vector."""
        return self.model.get_embedding_dimension()