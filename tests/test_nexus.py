from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_document, split_into_sentences
from app.ingestion.models import Document, DocumentPage
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import (
    chunk_document,
    split_into_paragraphs,
    split_into_sentences,
)
from app.embeddings.model import EmbeddingModel
from app.retrieval.search import (
    SemanticSearch,
    cosine_similarity,
)
from app.pipeline import search_document
from app.ingestion.indexer import index_document
from app.retrieval.index import VectorIndex
from app.cli import build_parser
from unittest.mock import Mock

from app.generation.context import build_context, build_prompt
from app.generation.rag import answer_question
from app.retrieval.search import SearchResult


def test_pdf_loader_rejects_missing_file():
    missing_file = Path("data/raw/does_not_exist.pdf")

    with pytest.raises(FileNotFoundError):
        load_pdf(missing_file)


def test_pdf_loader_extracts_text():
    pdf_file = Path("data/raw/test_document.pdf")

    document = load_pdf(pdf_file)

    assert len(document.pages) == 1
    assert document.pages[0].page_number == 1
    assert "NEXUS Test Document" in document.pages[0].text
    assert "personal research and knowledge intelligence system" in document.pages[0].text
    assert "PDF text extraction" in document.pages[0].text


def test_pdf_loader_preserves_source():
    pdf_file = Path("data/raw/test_document.pdf")

    document = load_pdf(pdf_file)

    assert document.source == str(pdf_file)


def test_sentence_splitter():
    text = "First sentence. Second sentence! Third sentence?"

    sentences = split_into_sentences(text)

    assert sentences == [
        "First sentence.",
        "Second sentence!",
        "Third sentence?",
    ]

def test_paragraph_splitter():
    text = (
        "First paragraph sentence one. "
        "First paragraph sentence two.\n\n"
        "Second paragraph sentence one. "
        "Second paragraph sentence two."
    )

    paragraphs = split_into_paragraphs(text)

    assert paragraphs == [
        "First paragraph sentence one. First paragraph sentence two.",
        "Second paragraph sentence one. Second paragraph sentence two.",
    ]


def test_paragraph_splitter_ignores_extra_blank_lines():
    text = (
        "First paragraph.\n\n\n"
        "Second paragraph.\n\n"
        "Third paragraph."
    )

    paragraphs = split_into_paragraphs(text)

    assert paragraphs == [
        "First paragraph.",
        "Second paragraph.",
        "Third paragraph.",
    ]

    
def test_chunker_keeps_sentences_together():
    document = Document(
        source="test.pdf",
        pages=[
            DocumentPage(
                page_number=1,
                text=(
                    "This is the first sentence. "
                    "This is the second sentence. "
                    "This is the third sentence."
                ),
            )
        ],
    )

    chunks = chunk_document(
        document,
        chunk_size=50,
        overlap=1,
    )

    assert len(chunks) == 3

    assert chunks[0].text == "This is the first sentence."
    assert chunks[1].text == (
        "This is the first sentence. This is the second sentence."
    )
    assert chunks[2].text == (
        "This is the second sentence. This is the third sentence."
    )


def test_chunker_generates_chunk_ids():
    document = Document(
        source="research.pdf",
        pages=[
            DocumentPage(
                page_number=3,
                text="This is the first sentence. This is the second sentence.",
            )
        ],
    )

    chunks = chunk_document(
        document,
        chunk_size=100,
        overlap=1,
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "research.pdf:p3:c1"


def test_chunker_preserves_metadata():
    document = Document(
        source="research.pdf",
        pages=[
            DocumentPage(
                page_number=7,
                text="This is some research text.",
            )
        ],
    )

    chunks = chunk_document(
        document,
        chunk_size=100,
        overlap=1,
    )

    assert len(chunks) == 1
    assert chunks[0].source == "research.pdf"
    assert chunks[0].page_number == 7


def test_chunker_rejects_invalid_settings():
    document = Document(
        source="test.pdf",
        pages=[
            DocumentPage(
                page_number=1,
                text="Some text.",
            )
        ],
    )

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=0)

    with pytest.raises(ValueError):
        chunk_document(document, chunk_size=10, overlap=-1)

def test_embedding_model_dimension():
    model = EmbeddingModel()

    assert model.dimension == 384


def test_embedding_model_generates_vector():
    model = EmbeddingModel()

    embedding = model.embed_text(
        "NEXUS is a research and knowledge intelligence system."
    )

    assert len(embedding) == 384
    assert all(isinstance(value, float) for value in embedding)


def test_embedding_model_generates_multiple_vectors():
    model = EmbeddingModel()

    embeddings = model.embed_texts(
        [
            "Machine learning is a field of artificial intelligence.",
            "Python is a programming language.",
        ]
    )

    assert len(embeddings) == 2
    assert all(len(embedding) == 384 for embedding in embeddings)


def test_cosine_similarity_identical_vectors():
    vector = [1.0, 2.0, 3.0]

    score = cosine_similarity(vector, vector)

    assert score == pytest.approx(1.0)


def test_cosine_similarity_rejects_different_dimensions():
    with pytest.raises(ValueError):
        cosine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )


def test_cosine_similarity_handles_zero_vector():
    score = cosine_similarity(
        [0.0, 0.0],
        [1.0, 2.0],
    )

    assert score == 0.0


def test_semantic_search_returns_relevant_chunks():
    document = Document(
        source="research.pdf",
        pages=[
            DocumentPage(
                page_number=1,
                text=(
                    "Transformers use attention mechanisms to process "
                    "relationships between tokens. "
                    "The model achieved high accuracy on the dataset."
                    "\n\n"
                    "Python is a programming language used for software "
                    "development."
                ),
            )
        ],
    )

    chunks = chunk_document(
        document,
        chunk_size=150,
        overlap=0,
    )

    model = EmbeddingModel()
    search = SemanticSearch(model)

    search.add_chunks(chunks)

    results = search.search(
        "How accurate was the model?",
        top_k=1,
    )

    assert len(results) == 1
    assert "accuracy" in results[0].chunk.text.lower()


def test_end_to_end_document_search():
    pdf_file = Path("data/raw/test_document.pdf")

    results = search_document(
        pdf_file,
        "What is the purpose of NEXUS?",
        top_k=1,
    )

    assert len(results) == 1

    result = results[0]

    assert result.score > 0
    assert "research and knowledge intelligence system" in (
        result.chunk.text.lower()
    )

    assert result.chunk.source == str(pdf_file)
    assert result.chunk.page_number == 1


def test_vector_index_can_be_saved_and_loaded(tmp_path):
    pdf_file = Path("data/raw/test_document.pdf")
    index_directory = tmp_path / "index"

    index_document(
        pdf_file,
        index_directory,
    )

    assert (index_directory / "embeddings.npy").exists()
    assert (index_directory / "metadata.json").exists()

    model = EmbeddingModel()

    loaded_index = VectorIndex.load(
        index_directory,
        model,
    )

    results = loaded_index.search(
        "What is the purpose of NEXUS?",
        top_k=1,
    )

    assert len(results) == 1
    assert "research and knowledge intelligence system" in (
        results[0].chunk.text.lower()
    )


def test_cli_ingest_command():
    parser = build_parser()

    args = parser.parse_args(
        ["ingest", "data/raw/test_document.pdf"]
    )

    assert args.command == "ingest"
    assert args.file == "data/raw/test_document.pdf"


def test_cli_query_command():
    parser = build_parser()

    args = parser.parse_args(
        [
            "query",
            "What is the purpose of NEXUS?",
            "--top-k",
            "3",
        ]
    )

    assert args.command == "query"
    assert args.question == "What is the purpose of NEXUS?"
    assert args.top_k == 3


def test_build_context_includes_source_metadata():
    chunk = Mock()
    chunk.source = "research.pdf"
    chunk.page_number = 3
    chunk.chunk_id = "research.pdf:p3:c1"
    chunk.text = "NEXUS retrieves relevant evidence."

    result = SearchResult(
        chunk=chunk,
        score=0.85,
    )

    context = build_context([result])

    assert "research.pdf" in context
    assert "Page: 3" in context
    assert "research.pdf:p3:c1" in context
    assert "NEXUS retrieves relevant evidence." in context
    assert "[S1]" in context


def test_build_prompt_includes_question_and_evidence():
    chunk = Mock()
    chunk.source = "research.pdf"
    chunk.page_number = 2
    chunk.chunk_id = "research.pdf:p2:c1"
    chunk.text = "NEXUS answers questions using retrieved documents."

    result = SearchResult(
        chunk=chunk,
        score=0.90,
    )

    prompt = build_prompt(
        question="How does NEXUS answer questions?",
        results=[result],
    )

    assert "How does NEXUS answer questions?" in prompt
    assert "NEXUS answers questions using retrieved documents." in prompt
    assert "[S1]" in prompt
    assert "Do not invent facts" in prompt
    assert "Separate words correctly" in prompt
    assert "silently proofread" in prompt
    assert "Do not remove, fabricate, or alter source citations" in prompt


def test_build_prompt_rejects_empty_question():
    with pytest.raises(ValueError):
        build_prompt(
            question="   ",
            results=[],
        )


def test_answer_question_rejects_empty_question():
    llm = Mock()

    with pytest.raises(ValueError):
        answer_question(
            question="   ",
            llm=llm,
        )

    llm.generate.assert_not_called()


def test_rag_pipeline_returns_answer_and_sources(monkeypatch):
    from app.generation import rag

    mock_chunk = Mock()
    mock_chunk.source = "research.pdf"
    mock_chunk.page_number = 1
    mock_chunk.chunk_id = "research.pdf:p1:c1"
    mock_chunk.text = "NEXUS provides evidence-based answers."

    mock_result = SearchResult(
        chunk=mock_chunk,
        score=0.95,
    )

    mock_index = Mock()
    mock_index.search.return_value = [mock_result]

    mock_embedding_model = Mock()
    mock_llm = Mock()
    mock_llm.generate.return_value = (
        "NEXUS provides evidence-based answers [S1]."
    )

    monkeypatch.setattr(
        rag,
        "EmbeddingModel",
        lambda: mock_embedding_model,
    )
    monkeypatch.setattr(
        rag.VectorIndex,
        "load",
        lambda **kwargs: mock_index,
    )

    answer, sources = answer_question(
        question="What does NEXUS provide?",
        llm=mock_llm,
    )

    assert answer == "NEXUS provides evidence-based answers [S1]."
    assert sources == [mock_result]

    mock_index.search.assert_called_once_with(
    query="What does NEXUS provide?",
    top_k=5,
    min_score=0.30,
    )
    mock_llm.generate.assert_called_once()
    assert "NEXUS provides evidence-based answers." in (
        mock_llm.generate.call_args.args[0]
    )


def test_build_context_reports_missing_evidence():
    context = build_context([])

    assert context == "No relevant evidence was retrieved."


def test_rag_does_not_call_llm_without_results(monkeypatch):
    from app.generation import rag

    mock_index = Mock()
    mock_index.search.return_value = []

    mock_llm = Mock()

    monkeypatch.setattr(
        rag,
        "EmbeddingModel",
        lambda: Mock(),
    )
    monkeypatch.setattr(
        rag.VectorIndex,
        "load",
        lambda **kwargs: mock_index,
    )

    answer, sources = answer_question(
        question="What is the population of Mars?",
        llm=mock_llm,
    )

    assert answer == (
    "I couldn't find sufficiently relevant evidence "
    "in the indexed documents."
    )
    assert sources == []
    mock_llm.generate.assert_not_called()

def test_semantic_search_filters_results_below_min_score():
    model = Mock()

    model.embed_text.return_value = [1.0, 0.0]
    model.embed_texts.return_value = [
        [0.9, 0.1],
        [0.0, 1.0],
    ]

    search = SemanticSearch(model)

    chunks = [
        Mock(text="Relevant evidence."),
        Mock(text="Unrelated evidence."),
    ]

    search.add_chunks(chunks)

    results = search.search(
        query="Find relevant evidence.",
        top_k=5,
        min_score=0.8,
    )

    assert len(results) == 1
    assert results[0].chunk.text == "Relevant evidence."
    assert results[0].score >= 0.8


def test_semantic_search_rejects_invalid_min_score():
    search = SemanticSearch(Mock())

    with pytest.raises(ValueError):
        search.search(
            query="Test",
            min_score=1.5,
        )


def test_semantic_search_without_threshold_keeps_existing_behavior():
    model = Mock()

    model.embed_text.return_value = [1.0, 0.0]
    model.embed_texts.return_value = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    search = SemanticSearch(model)

    chunks = [
        Mock(text="First chunk."),
        Mock(text="Second chunk."),
    ]

    search.add_chunks(chunks)

    results = search.search(
        query="Test",
        top_k=5,
    )

    assert len(results) == 2

def test_vector_index_filters_weak_matches():
    model = Mock()
    model.embed_text.return_value = [1.0, 0.0]
    model.embed_texts.return_value = [
        [0.9, 0.1],
        [0.0, 1.0],
    ]

    index = VectorIndex(model)

    strong_chunk = Mock()
    strong_chunk.text = "Relevant evidence."

    weak_chunk = Mock()
    weak_chunk.text = "Weak evidence."

    index.add_chunks([strong_chunk, weak_chunk])

    results = index.search(
        query="Find relevant evidence.",
        top_k=5,
        min_score=0.8,
    )

    assert len(results) == 1
    assert results[0].chunk.text == "Relevant evidence."
    assert results[0].score >= 0.8
