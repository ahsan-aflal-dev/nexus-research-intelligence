from unittest.mock import Mock

import pytest

from app.ingestion.models import DocumentChunk
from app.retrieval.index import VectorIndex


@pytest.fixture
def evaluation_index():
    """Create a predictable index for testing retrieval behavior."""

    model = Mock()
    model.embed_text.return_value = [1.0, 0.0]
    model.embed_texts.return_value = [
        [0.95, 0.05],
        [0.0, 1.0],
        [0.8, 0.6],
    ]

    index = VectorIndex(model)

    index.add_chunks(
        [
            DocumentChunk(
                chunk_id="research:p1:c1",
                text="NEXUS ingests documents and provides evidence-based answers.",
                source="research.pdf",
                page_number=1,
            ),
            DocumentChunk(
                chunk_id="research:p2:c1",
                text="Python is a programming language.",
                source="research.pdf",
                page_number=2,
            ),
            DocumentChunk(
                chunk_id="research:p3:c1",
                text="NEXUS retrieves relevant information from documents.",
                source="research.pdf",
                page_number=3,
            ),
        ]
    )

    return index


def test_relevant_evidence_ranks_first(evaluation_index):
    results = evaluation_index.search(
        query="What is the purpose of NEXUS?",
        top_k=3,
    )

    assert results
    assert results[0].chunk.chunk_id == "research:p1:c1"


def test_retrieval_returns_results_in_score_order(evaluation_index):
    results = evaluation_index.search(
        query="What is the purpose of NEXUS?",
        top_k=3,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)


def test_similarity_threshold_filters_weak_matches(evaluation_index):
    results = evaluation_index.search(
        query="What is the purpose of NEXUS?",
        top_k=3,
        min_score=0.81,
    )

    assert all(result.score >= 0.81 for result in results)
    assert len(results) == 1


def test_top_k_limits_number_of_results(evaluation_index):
    results = evaluation_index.search(
        query="What is the purpose of NEXUS?",
        top_k=1,
    )

    assert len(results) == 1