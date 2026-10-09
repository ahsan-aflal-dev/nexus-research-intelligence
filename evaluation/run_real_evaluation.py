import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.embeddings.model import EmbeddingModel
from app.retrieval.index import VectorIndex


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "dataset.json"
DEFAULT_REPORT = PROJECT_ROOT / "evaluation" / "reports" / "latest.json"


def reciprocal_rank(results, relevant_ids):
    """Return the reciprocal rank of the first relevant result."""

    for rank, result in enumerate(results, start=1):
        if result.chunk.chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def recall_at_k(results, relevant_ids, k):
    """Return the fraction of relevant chunks retrieved in the top k."""

    if not relevant_ids:
        return 0.0

    retrieved_ids = {
        result.chunk.chunk_id for result in results[:k]
    }

    return len(retrieved_ids & relevant_ids) / len(relevant_ids)


def load_dataset(path):
    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset.get("questions"):
        raise ValueError("Evaluation dataset must contain questions.")

    seen_ids = set()

    for item in dataset["questions"]:
        if not item.get("id") or not item.get("question", "").strip():
            raise ValueError("Each question needs an id and question text.")

        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate question ID: {item['id']}")

        seen_ids.add(item["id"])

        if not item.get("relevant_chunk_ids"):
            raise ValueError(
                f"Question {item['id']} needs relevant_chunk_ids."
            )

    return dataset


def evaluate(index, questions, top_k):
    results_by_question = []
    recall_1_values = []
    recall_3_values = []
    recall_5_values = []
    reciprocal_ranks = []

    available_chunk_ids = {
        chunk.chunk_id for chunk in index.chunks
    }

    for item in questions:
        relevant_ids = set(item["relevant_chunk_ids"])
        missing_ids = relevant_ids - available_chunk_ids

        search_results = index.search(
            query=item["question"],
            top_k=top_k,
            min_score=None,
        )

        ranked_results = []

        for rank, result in enumerate(search_results, start=1):
            ranked_results.append(
                {
                    "rank": rank,
                    "chunk_id": result.chunk.chunk_id,
                    "source": result.chunk.source,
                    "page_number": result.chunk.page_number,
                    "similarity": round(float(result.score), 6),
                    "is_relevant": result.chunk.chunk_id in relevant_ids,
                }
            )

        rr = reciprocal_rank(search_results, relevant_ids)
        r1 = recall_at_k(search_results, relevant_ids, 1)
        r3 = recall_at_k(search_results, relevant_ids, 3)
        r5 = recall_at_k(search_results, relevant_ids, 5)

        recall_1_values.append(r1)
        recall_3_values.append(r3)
        recall_5_values.append(r5)
        reciprocal_ranks.append(rr)

        results_by_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "relevant_chunk_ids": sorted(relevant_ids),
                "missing_relevant_chunk_ids": sorted(missing_ids),
                "recall_at_1": round(r1, 6),
                "recall_at_3": round(r3, 6),
                "recall_at_5": round(r5, 6),
                "reciprocal_rank": round(rr, 6),
                "first_relevant_rank": (
                    next(
                        (
                            result["rank"]
                            for result in ranked_results
                            if result["is_relevant"]
                        ),
                        None,
                    )
                ),
                "retrieved_results": ranked_results,
            }
        )

    count = len(questions)

    metrics = {
        "question_count": count,
        "recall_at_1": round(sum(recall_1_values) / count, 6),
        "recall_at_3": round(sum(recall_3_values) / count, 6),
        "recall_at_5": round(sum(recall_5_values) / count, 6),
        "mrr": round(sum(reciprocal_ranks) / count, 6),
        "questions_with_missing_labels": sum(
            bool(result["missing_relevant_chunk_ids"])
            for result in results_by_question
        ),
        "questions_with_no_relevant_result": sum(
            result["reciprocal_rank"] == 0
            for result in results_by_question
        ),
    }

    return metrics, results_by_question


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NEXUS semantic retrieval quality."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the evaluation dataset JSON.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Where to save the JSON evaluation report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to inspect per question.",
    )
    args = parser.parse_args()

    if args.top_k < 5:
        parser.error("--top-k must be at least 5 to calculate Recall@5.")

    dataset = load_dataset(args.dataset)

    index_directory = Path(dataset["index_directory"])
    if not index_directory.is_absolute():
        index_directory = PROJECT_ROOT / index_directory

    model = EmbeddingModel()
    index = VectorIndex.load(index_directory, model)

    metrics, question_results = evaluate(
        index=index,
        questions=dataset["questions"],
        top_k=args.top_k,
    )

    report = {
        "dataset_name": dataset.get("dataset_name", "Unnamed dataset"),
        "dataset_version": dataset.get("version", "unspecified"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_directory": str(index_directory),
        "indexed_chunk_count": len(index.chunks),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": args.top_k,
        "metrics": metrics,
        "questions": question_results,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)

    with args.report.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nNEXUS RETRIEVAL EVALUATION")
    print("=" * 48)
    print(f"Dataset:       {report['dataset_name']}")
    print(f"Questions:     {metrics['question_count']}")
    print(f"Indexed chunks:{len(index.chunks):>5}")
    print(f"Recall@1:      {metrics['recall_at_1']:.3f}")
    print(f"Recall@3:      {metrics['recall_at_3']:.3f}")
    print(f"Recall@5:      {metrics['recall_at_5']:.3f}")
    print(f"MRR:           {metrics['mrr']:.3f}")
    print(
        "Missing labels:"
        f" {metrics['questions_with_missing_labels']}"
    )
    print(
        "No relevant result:"
        f" {metrics['questions_with_no_relevant_result']}"
    )

    print("\nPER-QUESTION RESULTS")
    print("-" * 48)

    for result in question_results:
        rank = result["first_relevant_rank"]
        rank_text = str(rank) if rank is not None else "NOT FOUND"

        print(f"\n[{result['id']}] {result['question']}")
        print(f"First relevant rank: {rank_text}")

        for retrieved in result["retrieved_results"]:
            marker = "*" if retrieved["is_relevant"] else " "
            print(
                f" {marker} {retrieved['rank']}. "
                f"{retrieved['chunk_id']} "
                f"(score={retrieved['similarity']:.4f})"
            )

        if result["missing_relevant_chunk_ids"]:
            print("WARNING: Expected chunk ID not found in index.")

    print(f"\nJSON report saved to: {args.report}")


if __name__ == "__main__":
    main()