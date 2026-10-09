
import json

import pytest

from evaluation.run_real_evaluation import (
    evaluate,
    load_dataset,
    recall_at_k,
    reciprocal_rank,
)


def make_result(chunk_id, score=0.9):
    """Create a minimal search result for evaluation tests."""
    class Chunk:
        def __init__(self):
            self.chunk_id = chunk_id
            self.source = "research.pdf"
            self.page_number = 1

    class Result:
        def __init__(self):
            self.chunk = Chunk()
            self.score = score

    return Result()


def test_reciprocal_rank_returns_one_for_first_result():
    results = [
        make_result("chunk-a"),
        make_result("chunk-b"),
    ]

    assert reciprocal_rank(results, {"chunk-a"}) == 1.0


def test_reciprocal_rank_rewards_higher_ranking():
    results = [
        make_result("chunk-a"),
        make_result("chunk-b"),
        make_result("chunk-c"),
    ]

    assert reciprocal_rank(results, {"chunk-b"}) == 0.5
    assert reciprocal_rank(results, {"chunk-c"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_when_not_found():
    results = [make_result("chunk-a")]

    assert reciprocal_rank(results, {"chunk-z"}) == 0.0


def test_recall_at_k_counts_relevant_results_in_top_k():
    results = [
        make_result("chunk-a"),
        make_result("chunk-b"),
        make_result("chunk-c"),
    ]

    relevant = {"chunk-b", "chunk-c"}

    assert recall_at_k(results, relevant, 1) == 0.0
    assert recall_at_k(results, relevant, 2) == 0.5
    assert recall_at_k(results, relevant, 3) == 1.0


def test_recall_at_k_returns_zero_when_no_relevant_ids():
    results = [make_result("chunk-a")]

    assert recall_at_k(results, set(), 5) == 0.0


def test_load_dataset_rejects_empty_question_list(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps({"questions": []}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must contain questions",
    ):
        load_dataset(dataset_path)


def test_load_dataset_rejects_duplicate_question_ids(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "question": "Question one?",
                        "relevant_chunk_ids": ["chunk-a"],
                    },
                    {
                        "id": "q1",
                        "question": "Question two?",
                        "relevant_chunk_ids": ["chunk-b"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate question ID"):
        load_dataset(dataset_path)


def test_load_dataset_rejects_missing_relevance_labels(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "question": "Question one?",
                        "relevant_chunk_ids": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="needs relevant_chunk_ids"):
        load_dataset(dataset_path)


def test_evaluate_calculates_expected_metrics():
    class FakeIndex:
        chunks = [
            make_result("chunk-a").chunk,
            make_result("chunk-b").chunk,
            make_result("chunk-c").chunk,
        ]

        def search(self, query, top_k, min_score):
            ranking = {
                "question one": [
                    make_result("chunk-a"),
                    make_result("chunk-b"),
                    make_result("chunk-c"),
                ],
                "question two": [
                    make_result("chunk-b"),
                    make_result("chunk-a"),
                    make_result("chunk-c"),
                ],
            }
            return ranking[query][:top_k]

    questions = [
        {
            "id": "q1",
            "question": "question one",
            "relevant_chunk_ids": ["chunk-a"],
        },
        {
            "id": "q2",
            "question": "question two",
            "relevant_chunk_ids": ["chunk-c"],
        },
    ]

    metrics, results = evaluate(
        index=FakeIndex(),
        questions=questions,
        top_k=3,
    )

    assert metrics["question_count"] == 2
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr"] == pytest.approx(2 / 3)
    assert metrics["questions_with_missing_labels"] == 0
    assert metrics["questions_with_no_relevant_result"] == 0
    assert results[0]["first_relevant_rank"] == 1
    assert results[1]["first_relevant_rank"] == 3


def test_evaluate_reports_relevance_labels_missing_from_index():
    class FakeIndex:
        chunks = [make_result("chunk-a").chunk]

        def search(self, query, top_k, min_score):
            return [make_result("chunk-a")]

    questions = [
        {
            "id": "q1",
            "question": "question one",
            "relevant_chunk_ids": ["chunk-missing"],
        }
    ]

    metrics, results = evaluate(
        index=FakeIndex(),
        questions=questions,
        top_k=5,
    )

    assert metrics["questions_with_missing_labels"] == 1
    assert metrics["questions_with_no_relevant_result"] == 1
    assert results[0]["missing_relevant_chunk_ids"] == ["chunk-missing"]
