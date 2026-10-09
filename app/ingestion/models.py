from dataclasses import dataclass


@dataclass
class DocumentPage:
    page_number: int
    text: str


@dataclass
class Document:
    source: str
    pages: list[DocumentPage]


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    page_number: int