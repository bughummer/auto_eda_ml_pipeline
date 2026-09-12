"""Reasoning model clients.

The analyzer depends on this narrow protocol, so the reasoning layer is testable without
AWS and the Bedrock specifics stay in one adapter.
"""

import json
from typing import Any, Protocol


class ReasoningUnavailableError(RuntimeError):
    """Raised when the reasoning model cannot be reached or refuses the request."""


class ReasoningClient(Protocol):
    model_id: str

    def invoke(self, system_prompt: str, user_prompt: str) -> str: ...


class BedrockReasoningClient:
    """Amazon Bedrock adapter. Deterministic settings: temperature 0, fixed token budget."""

    def __init__(
        self,
        client: Any,
        *,
        model_id: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self.model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
        }
        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read())
        except Exception as exc:
            raise ReasoningUnavailableError(
                f"The reasoning model could not be invoked: {type(exc).__name__}: {exc}"
            ) from exc
        return _extract_text(payload)


def _extract_text(payload: dict[str, Any]) -> str:
    """Read the text out of a Bedrock response, tolerating the small shape differences."""
    content = payload.get("content")
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    for key in ("completion", "outputText", "generated_text"):
        if isinstance(payload.get(key), str):
            return payload[key]
    results = payload.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return str(results[0].get("outputText", ""))
    raise ReasoningUnavailableError("The reasoning model returned an unrecognized response shape.")
