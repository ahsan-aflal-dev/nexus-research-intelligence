import argparse
from pathlib import Path

from app.embeddings.model import EmbeddingModel
from app.generation.rag import answer_question
from app.ingestion.manager import (
    ingest_document,
    list_documents,
    rebuild_index,
    remove_document,
)
from app.retrieval.index import VectorIndex


# Keep the legacy data/index directory untouched during validation.
DEFAULT_INDEX_DIRECTORY = Path("data/managed_index")


def ingest(file_path: str, index_directory: Path) -> None:
    result = ingest_document(file_path, index_directory)

    print(result["message"])
    print(f"Status: {result['status']}")
    print(f"Source: {result['source_path']}")
    print(f"Chunks: {result.get('chunk_count', 0)}")


def query(
    question: str,
    index_directory: Path,
    top_k: int,
    sources_only: bool = False,
) -> None:
    if sources_only:
        model = EmbeddingModel()
        index = VectorIndex.load(index_directory, model)
        results = index.search(query=question, top_k=top_k)
        answer = None
    else:
        answer, results = answer_question(
            question=question,
            index_directory=index_directory,
            top_k=top_k,
        )

    print(f"\nQuestion: {question}\n")
    print("=" * 80)

    if answer is not None:
        print("\nNEXUS ANSWER\n")
        print(answer)
        print("\n" + "=" * 80)

    if not results:
        print("\nNo relevant evidence found in the index.")
        return

    print("\nRETRIEVED SOURCES\n")

    for number, result in enumerate(results, start=1):
        chunk = result.chunk

        print(f"\n[S{number}] {chunk.source}")
        print(f"Page: {chunk.page_number}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Similarity: {result.score:.4f}")
        print(f"\n{chunk.text}")
        print("-" * 80)


def show_documents(index_directory: Path) -> None:
    documents = list_documents(index_directory)

    if not documents:
        print("No registered documents.")
        return

    print(f"\nIndexed documents: {len(documents)}\n")

    for number, document in enumerate(documents, start=1):
        print(f"{number}. {document['file_name']}")
        print(f"   Path: {document['source_path']}")
        print(f"   Chunks: {document['chunk_count']}")
        print(f"   Indexed: {document['ingested_at_utc']}")


def remove(file_path: str, index_directory: Path) -> None:
    result = remove_document(file_path, index_directory)
    print(result["message"])
    print(f"Source: {result['source_path']}")


def rebuild(index_directory: Path) -> None:
    result = rebuild_index(index_directory)
    print(result["message"])
    print(f"Documents: {result['document_count']}")
    print(f"Chunks: {result['chunk_count']}")


def add_index_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX_DIRECTORY,
        help="Directory containing the vector index.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="NEXUS Research & Knowledge Intelligence System",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Add a PDF to the persistent knowledge index.",
    )
    ingest_parser.add_argument("file", help="Path to the PDF.")
    add_index_argument(ingest_parser)

    query_parser = subparsers.add_parser(
        "query",
        help="Ask questions about indexed documents.",
    )
    query_parser.add_argument("question", help="Question to answer.")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of evidence chunks to retrieve.",
    )
    query_parser.add_argument(
        "--sources-only",
        action="store_true",
        help="Retrieve evidence without generating an answer.",
    )
    add_index_argument(query_parser)

    list_parser = subparsers.add_parser(
        "list-documents",
        help="List registered documents.",
    )
    add_index_argument(list_parser)

    remove_parser = subparsers.add_parser(
        "remove-document",
        help="Remove a document and its chunks.",
    )
    remove_parser.add_argument("file", help="Path to the PDF to remove.")
    add_index_argument(remove_parser)

    rebuild_parser = subparsers.add_parser(
        "rebuild-index",
        help="Rebuild the index from registered source PDFs.",
    )
    add_index_argument(rebuild_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.file, args.index)

    elif args.command == "query":
        query(
            question=args.question,
            index_directory=args.index,
            top_k=args.top_k,
            sources_only=args.sources_only,
        )

    elif args.command == "list-documents":
        show_documents(args.index)

    elif args.command == "remove-document":
        remove(args.file, args.index)

    elif args.command == "rebuild-index":
        rebuild(args.index)


if __name__ == "__main__":
    main()