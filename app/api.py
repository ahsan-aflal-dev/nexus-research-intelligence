import json
import re
import subprocess
import sys
from uuid import uuid4

from fastapi import File, UploadFile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.generation.rag import answer_question
from app.ingestion.manager import (
    ingest_document,
    list_documents,
    rebuild_index,
    remove_document,
)


DEFAULT_INDEX_DIRECTORY = Path("data/managed_index")
UPLOAD_DIRECTORY = Path("data/uploads")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_REPORT = PROJECT_ROOT / "evaluation" / "reports" / "latest.json"
EVALUATION_TIMEOUT_SECONDS = 300

app = FastAPI(
    title="NEXUS API",
    description=(
        "Research and Knowledge Intelligence API for "
        "document ingestion, semantic retrieval, and evidence-based answers."
    ),
    version="0.1.0",
)


class IngestRequest(BaseModel):
    file_path: str = Field(
        min_length=1,
        description="Path to a local PDF file.",
    )


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to answer using indexed evidence.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of evidence chunks to retrieve.",
    )


class RemoveDocumentRequest(BaseModel):
    file_path: str = Field(
        min_length=1,
        description="Path of the indexed PDF to remove.",
    )


@app.get("/")
def root() -> dict:
    return {
        "name": "NEXUS API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "service": "NEXUS API",
    }


@app.get("/documents")
def get_documents() -> dict:
    try:
        documents = list_documents(DEFAULT_INDEX_DIRECTORY)
        return {
            "count": len(documents),
            "documents": documents,
        }
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/documents")
def create_document(request: IngestRequest) -> dict:
    try:
        result = ingest_document(
            file_path=request.file_path,
            index_directory=DEFAULT_INDEX_DIRECTORY,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post("/query")
def query_knowledge_base(request: QueryRequest) -> dict:
    try:
        answer, results = answer_question(
            question=request.question,
            index_directory=DEFAULT_INDEX_DIRECTORY,
            top_k=request.top_k,
        )

        sources = [
            {
                "citation": f"S{number}",
                "source": result.chunk.source,
                "page_number": result.chunk.page_number,
                "chunk_id": result.chunk.chunk_id,
                "similarity": result.score,
                "text": result.chunk.text,
            }
            for number, result in enumerate(results, start=1)
        ]

        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "source_count": len(sources),
        }
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The knowledge index is not available. "
                "Ingest a PDF before querying."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/index/rebuild")
def rebuild_knowledge_index() -> dict:
    try:
        return rebuild_index(DEFAULT_INDEX_DIRECTORY)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Save and index a PDF uploaded through the dashboard."""

    original_name = Path(file.filename or "").name

    if not original_name or not original_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    safe_stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        Path(original_name).stem,
    ).strip("._") or "document"

    stored_path = UPLOAD_DIRECTORY / f"{safe_stem}_{uuid4().hex[:12]}.pdf"
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    size = 0

    try:
        with stored_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 25 MB upload limit.",
                    )

                destination.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        # Check the PDF signature before passing the file to ingestion.
        with stored_path.open("rb") as uploaded:
            if uploaded.read(5) != b"%PDF-":
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded file does not have a valid PDF signature.",
                )

        result = ingest_document(
            file_path=stored_path,
            index_directory=DEFAULT_INDEX_DIRECTORY,
        )

        return {
            **result,
            "filename": original_name,
        }

    except HTTPException:
        stored_path.unlink(missing_ok=True)
        raise
    except (ValueError, OSError) as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400 if isinstance(exc, ValueError) else 500,
            detail=str(exc),
        ) from exc
    finally:
        await file.close()

@app.get("/evaluation/latest")
def get_latest_evaluation() -> dict:
    """Return the most recently saved retrieval evaluation report."""
    if not EVALUATION_REPORT.is_file():
        return {
            "available": False,
            "message": "No evaluation report exists yet. Run an evaluation first.",
        }

    try:
        with EVALUATION_REPORT.open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="The saved evaluation report could not be read.",
        ) from exc

    return {"available": True, "report": report}


@app.post("/evaluation/run")
def run_evaluation() -> dict:
    """Run the project's fixed retrieval evaluator and return its saved report."""
    command = [
        sys.executable,
        "-m",
        "evaluation.run_real_evaluation",
    ]

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=EVALUATION_TIMEOUT_SECONDS,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail="Evaluation exceeded the 300-second time limit.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not start the evaluation process.",
        ) from exc

    if result.returncode != 0:
        # Keep detailed process output in the server log, not the HTTP response.
        print("NEXUS evaluation failed:", result.stderr[-4000:])
        raise HTTPException(
            status_code=500,
            detail="Evaluation failed. Check the backend terminal for details.",
        )

    if not EVALUATION_REPORT.is_file():
        raise HTTPException(
            status_code=500,
            detail="Evaluation completed but its report was not found.",
        )

    try:
        with EVALUATION_REPORT.open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Evaluation ran, but the saved report could not be read.",
        ) from exc

    return {
        "status": "completed",
        "message": "Retrieval evaluation completed successfully.",
        "report": report,
        "output": result.stdout[-6000:],
    }


@app.delete("/documents")
def delete_document(request: RemoveDocumentRequest) -> dict:
    source_path = Path(request.file_path).expanduser().resolve()
    upload_root = UPLOAD_DIRECTORY.resolve()

    # Only physically delete PDFs managed by NEXUS uploads.
    try:
        source_path.relative_to(upload_root)
        is_managed_upload = True
    except ValueError:
        is_managed_upload = False

    try:
        result = remove_document(
            file_path=source_path,
            index_directory=DEFAULT_INDEX_DIRECTORY,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    if is_managed_upload and source_path.is_file():
        try:
            source_path.unlink()
            result["file_deleted"] = True
            result["message"] = (
                "Document removed from the index and uploaded PDF deleted."
            )
        except OSError:
            result["file_deleted"] = False
            result["message"] = (
                "Document removed from the index, but the PDF file "
                "could not be deleted. Remove it manually if necessary."
            )
    else:
        result["file_deleted"] = False
        result["message"] = (
            "Document removed from the index. The original file was preserved."
        )

    return result