
import subprocess
from types import SimpleNamespace

import pytest

from app.generation.llm import HermesLLM
from app.generation.validation import validate_answer


def test_hermes_command_uses_nexus_toolset_and_reasoning():
    llm = HermesLLM(
        executable="hermes-test",
        toolsets="clarify",
        reasoning="none",
    )

    command = llm._build_command("Test prompt")

    assert command == [
        "hermes-test",
        "--toolsets", "clarify",
        "--reasoning", "none",
        "-z", "Test prompt",
    ]


def test_all_toolsets_omits_toolset_override():
    llm = HermesLLM(
        toolsets="all",
        reasoning="none",
    )

    command = llm._build_command("Test prompt")

    assert "--toolsets" not in command
    assert command[-2:] == ["-z", "Test prompt"]


def test_environment_configures_hermes(monkeypatch):
    monkeypatch.setenv("NEXUS_HERMES_TIMEOUT", "45")
    monkeypatch.setenv("NEXUS_HERMES_TOOLSETS", "clarify,web")
    monkeypatch.setenv("NEXUS_HERMES_REASONING", "low")

    llm = HermesLLM()

    assert llm.timeout == 45
    assert llm.toolsets == "clarify,web"
    assert llm.reasoning == "low"


def test_invalid_timeout_environment_value(monkeypatch):
    monkeypatch.setenv("NEXUS_HERMES_TIMEOUT", "not-a-number")

    with pytest.raises(ValueError, match="NEXUS_HERMES_TIMEOUT"):
        HermesLLM()


def test_invalid_reasoning_is_rejected():
    with pytest.raises(ValueError, match="NEXUS_HERMES_REASONING"):
        HermesLLM(reasoning="super-fast")


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="Prompt must not be empty"):
        HermesLLM().generate("   ")


def test_generate_returns_stdout(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="  Generated answer  \n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    answer = HermesLLM().generate("Test prompt")

    assert answer == "Generated answer"


def test_generate_timeout_is_reported(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=2)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="did not respond"):
        HermesLLM(timeout=2).generate("Test prompt")


def test_generate_missing_executable_is_reported(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="was not found"):
        HermesLLM().generate("Test prompt")


def test_generate_nonzero_exit_is_reported(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="model failed",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="model failed"):
        HermesLLM().generate("Test prompt")


def test_generate_empty_output_is_rejected(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="   ",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="empty response"):
        HermesLLM().generate("Test prompt")


def test_validation_accepts_known_citation():
    warnings = validate_answer(
        "Retrieval returns relevant evidence [S1].",
        results=[object()],
    )

    assert warnings == []


def test_validation_reports_unknown_citation():
    warnings = validate_answer(
        "This claim uses an unknown source [S3].",
        results=[object(), object()],
    )

    assert any("Unknown source citation" in item for item in warnings)


def test_validation_reports_missing_citations():
    warnings = validate_answer(
        "This answer has no citation.",
        results=[object()],
    )

    assert any("no source citations" in item for item in warnings)


def test_validation_reports_missing_space_after_punctuation():
    warnings = validate_answer(
        "First sentence.Second sentence [S1].",
        results=[object()],
    )

    assert any("missing space" in item for item in warnings)


def test_validation_does_not_modify_answer():
    answer = "A relevantitems example [S1]."

    validate_answer(answer, results=[object()])

    assert answer == "A relevantitems example [S1]."


def test_validation_detects_passagecontains():
    warnings = validate_answer(
        "Each passagecontains enough context [S1].",
        results=[object()],
    )

    assert any(
        "passagecontains" in warning
        for warning in warnings
    )


def test_validation_detects_relevantitems():
    warnings = validate_answer(
        "The system returns relevantitems [S1].",
        results=[object()],
    )

    assert any(
        "relevantitems" in warning
        for warning in warnings
    )


def test_validation_does_not_flag_normal_database_word():
    warnings = validate_answer(
        "The database contains relevant information [S1].",
        results=[object()],
    )

    assert not any(
        "accidentally joined words" in warning
        for warning in warnings
    )
