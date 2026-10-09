import re

from app.retrieval.search import SearchResult


CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
PUNCTUATION_SPACING_PATTERN = re.compile(r"[.!?,;:][A-Za-z]")

# Known joined-word errors observed in NEXUS answers.
# This targeted approach avoids flagging normal words such as "database".
KNOWN_JOINED_WORD_ERRORS = {
    "passagecontains",
    "relevantitems",
    "systemcaptures",
    "canlead",
    "itinvolves",
    "andcoherence",
}


def _find_possible_joined_words(text: str) -> list[str]:
    """Find known joined-word errors without flagging ordinary words."""
    findings: list[str] = []

    for match in re.finditer(r"\b[A-Za-z]{4,}\b", text):
        word = match.group()

        if word.lower() in KNOWN_JOINED_WORD_ERRORS:
            findings.append(word)

    return findings


def validate_answer(
    answer: str,
    results: list[SearchResult],
) -> list[str]:
    """
    Report potential answer-quality problems without modifying the answer.

    These checks are conservative and cannot guarantee factual accuracy
    or detect every grammar and formatting problem.
    """
    warnings: list[str] = []

    allowed_labels = {
        f"S{number}"
        for number in range(1, len(results) + 1)
    }

    used_labels = set(CITATION_PATTERN.findall(answer))

    unknown_labels = sorted(used_labels - allowed_labels)

    if unknown_labels:
        labels = ", ".join(
            f"[{label}]" for label in unknown_labels
        )
        warnings.append(
            f"Unknown source citation(s): {labels}"
        )

    if results and not used_labels:
        warnings.append(
            "The answer contains no source citations."
        )

    if PUNCTUATION_SPACING_PATTERN.search(answer):
        warnings.append(
            "Possible missing space after punctuation; review the answer."
        )

    joined_words = _find_possible_joined_words(answer)

    if joined_words:
        examples = ", ".join(
            sorted(set(joined_words), key=str.lower)
        )
        warnings.append(
            "Possible accidentally joined words: "
            f"{examples}. Review the answer."
        )

    return warnings