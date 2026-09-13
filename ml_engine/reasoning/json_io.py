"""Extracting a JSON object from a model response.

Models sometimes wrap JSON in a markdown fence or surround it with a sentence. That is
recoverable; anything else is not, and is discarded rather than guessed at.
"""

import json
import re
from typing import Any

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class ReasoningOutputError(RuntimeError):
    """Raised when the model's output cannot be validated against the schema."""


def parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise ReasoningOutputError(
                "The model did not return JSON. The response was discarded."
            ) from None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ReasoningOutputError(
                f"The model returned malformed JSON: {exc}. The response was discarded."
            ) from exc
    if not isinstance(parsed, dict):
        raise ReasoningOutputError("The model returned JSON that is not an object.")
    return parsed
