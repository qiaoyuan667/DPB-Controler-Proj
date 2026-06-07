from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    model: str
    base_url: str
    api_key_env: str
    stream: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1200


class OpenAICompatibleChatClient:
    """Small wrapper for Swiss AI and other OpenAI-compatible chat endpoints."""

    def __init__(self, config: OpenAICompatibleConfig):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleChatClient requires the optional 'openai' "
                "package. Install it with: pip install -r requirements-api.txt"
            ) from exc

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key environment variable: {config.api_key_env}"
            )

        self.config = config
        self.client = openai.Client(
            api_key=api_key,
            base_url=config.base_url.rstrip("/"),
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stream: bool | None = None,
        seed: int | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "max_tokens": max_tokens
            if max_tokens is not None
            else self.config.max_tokens,
            "temperature": temperature
            if temperature is not None
            else self.config.temperature,
            "top_p": top_p if top_p is not None else self.config.top_p,
            "stream": self.config.stream if stream is None else stream,
        }
        if seed is not None:
            kwargs["seed"] = seed

        response = self.client.chat.completions.create(**kwargs)
        if kwargs["stream"]:
            return self._collect_stream(response)

        content = response.choices[0].message.content
        return "" if content is None else str(content)

    @staticmethod
    def _collect_stream(response: Any) -> str:
        chunks: list[str] = []
        for chunk in response:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                chunks.append(str(content))
        return "".join(chunks)

