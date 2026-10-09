import json

from fastapi.testclient import TestClient

import app.api as api_module
from app.api import app

client = TestClient(app)


def test_latest_evaluation_returns_unavailable_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api_module,
        "EVALUATION_REPORT",
        tmp_path / "missing.json",
    )

    response = client.get("/evaluation/latest")

    assert response.status_code == 200
    assert response.json()["available"] is False


def test_latest_evaluation_returns_saved_report(monkeypatch, tmp_path):
    report_path = tmp_path / "latest.json"
    report_path.write_text(
        json.dumps({
            "dataset_name": "Test dataset",
            "indexed_chunk_count": 5,
            "metrics": {"recall_at_1": 0.75, "mrr": 0.9375},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "EVALUATION_REPORT", report_path)

    response = client.get("/evaluation/latest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["report"]["metrics"]["recall_at_1"] == 0.75


def test_run_evaluation_returns_saved_report(monkeypatch, tmp_path):
    report_path = tmp_path / "latest.json"
    report_path.write_text(
        json.dumps({
            "dataset_name": "Test dataset",
            "metrics": {"recall_at_1": 0.75},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "EVALUATION_REPORT", report_path)

    class FakeProcess:
        returncode = 0
        stdout = "Evaluation completed"
        stderr = ""

    def fake_run(command, **kwargs):
        assert command[1:3] == ["-m", "evaluation.run_real_evaluation"]
        assert kwargs["shell"] is False
        return FakeProcess()

    monkeypatch.setattr(api_module.subprocess, "run", fake_run)

    response = client.post("/evaluation/run")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["report"]["metrics"]["recall_at_1"] == 0.75


def test_run_evaluation_reports_failure(monkeypatch, tmp_path):
    report_path = tmp_path / "latest.json"
    monkeypatch.setattr(api_module, "EVALUATION_REPORT", report_path)

    class FakeProcess:
        returncode = 1
        stdout = ""
        stderr = "simulated failure"

    monkeypatch.setattr(
        api_module.subprocess,
        "run",
        lambda *args, **kwargs: FakeProcess(),
    )

    response = client.post("/evaluation/run")

    assert response.status_code == 500
    assert "Evaluation failed" in response.json()["detail"]
