
import logging
from pathlib import Path

from app.embeddings.model import EmbeddingModel
from app.generation.context import build_prompt
from app.generation.llm import HermesLLM
from app.generation.validation import validate_answer
from app.retrieval.index import VectorIndex
from app.retrieval.search import SearchResult


logger = logging.getLogger(__name__)

DEFAULT_MIN_SCORE = 0.30


def answer_question(
    question: str,
    index_directory: str | Path = "data/index",
    top_k: int = 5,
    llm: HermesLLM | None = None,
    min_score: float | None = DEFAULT_MIN_SCORE,
) -> tuple[str, list[SearchResult]]:
    """Generate an answer using evidence that meets the relevance threshold."""

    if not question.strip():
        raise ValueError("Question must not be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if min_score is not None and not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between -1.0 and 1.0")

    embedding_model = EmbeddingModel()

    index = VectorIndex.load(
        directory=Path(index_directory),
        embedding_model=embedding_model,
    )

    results = index.search(
        query=question,
        top_k=top_k,
        min_score=min_score,
    )

    if not results:
        return (
            "I couldn't find sufficiently relevant evidence "
            "in the indexed documents.",
            [],
        )

    prompt = build_prompt(
        question=question,
        results=results,
    )

    generator = llm if llm is not None else HermesLLM()
    answer = generator.generate(prompt)

    for warning in validate_answer(answer, results):
        logger.warning("NEXUS answer validation: %s", warning)

    # Preserve the original answer. Warnings are logged rather than
    # silently altering model output or changing the public return type.
    return answer, results
