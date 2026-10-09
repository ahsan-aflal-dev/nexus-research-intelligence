from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_upload_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_rejects_invalid_pdf_signature():
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.pdf", b"not a pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert "signature" in response.json()["detail"]