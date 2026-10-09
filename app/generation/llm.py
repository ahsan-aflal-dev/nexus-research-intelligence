
import os
import subprocess


def _positive_int_from_env(name: str, default: int) -> int:
    """Read a positive integer configuration value from the environment."""
    value = os.getenv(name)

    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc

    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")

    return parsed


class HermesLLM:
    """Generate text through the locally configured Hermes agent."""

    def __init__(
        self,
        timeout: int | None = None,
        executable: str = "hermes",
        toolsets: str | None = None,
        reasoning: str | None = None,
    ) -> None:
        self.timeout = (
            timeout
            if timeout is not None
            else _positive_int_from_env("NEXUS_HERMES_TIMEOUT", 180)
        )

        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        self.executable = executable

        configured_toolsets = (
            toolsets
            if toolsets is not None
            else os.getenv("NEXUS_HERMES_TOOLSETS", "clarify")
        )
        self.toolsets = configured_toolsets.strip()

        configured_reasoning = (
            reasoning
            if reasoning is not None
            else os.getenv("NEXUS_HERMES_REASONING", "none")
        )
        self.reasoning = configured_reasoning.strip().lower()

        valid_reasoning = {
            "none", "minimal", "low", "medium",
            "high", "xhigh", "max", "ultra",
        }

        if self.reasoning not in valid_reasoning:
            raise ValueError(
                "NEXUS_HERMES_REASONING must be one of: "
                + ", ".join(sorted(valid_reasoning))
            )

    def _build_command(self, prompt: str) -> list[str]:
        """Build a Hermes command using NEXUS-specific settings."""
        command = [self.executable]

        # "all" leaves Hermes's normal tool configuration unchanged.
        if self.toolsets and self.toolsets.lower() != "all":
            command.extend(["--toolsets", self.toolsets])

        command.extend(["--reasoning", self.reasoning, "-z", prompt])
        return command

    def generate(self, prompt: str) -> str:
        """Send a prompt to Hermes and return its response."""

        if not prompt.strip():
            raise ValueError("Prompt must not be empty")

        command = self._build_command(prompt)

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
                env=os.environ.copy(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Hermes executable was not found. "
                "Check that Hermes is installed and available on PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Hermes did not respond within {self.timeout} seconds."
            ) from exc

        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Hermes generation failed: {details}")

        answer = result.stdout.strip()

        if not answer:
            raise RuntimeError("Hermes returned an empty response")

        return answer
