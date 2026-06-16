"""LLM provider abstraction — Ollama and OpenAI-compatible back-ends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx
import ollama

from src.config import settings
from src.models import (
    FeedbackRequest,
    FeedbackResponse,
    MappingRequest,
    MappingResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract provider interface
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Base contract for all LLM back-ends."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat-completion request and return the assistant's text content."""

    @abstractmethod
    async def generate_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Run the multi-pass pipeline and return a FeedbackResponse."""

    @abstractmethod
    async def suggest_mapping(self, request: MappingRequest) -> MappingResponse:
        """Analyse CSV headers and propose a mapping to the target schema."""

    @abstractmethod
    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        """Generate an embedding vector for the given text."""


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Uses the `ollama` Python package to communicate with a local Ollama server."""

    def __init__(self) -> None:
        self._model = settings.ollama_model

    # -- chat ----------------------------------------------------------------
    async def chat(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        chat_options: dict[str, Any] = {
            "num_ctx": settings.num_ctx,
            "num_predict": settings.num_predict,
            "temperature": options.get("temperature", 0.0) if options else 0.0,
        }
        if options:
            chat_options.update({k: v for k, v in options.items() if k not in chat_options})

        kwargs: dict[str, Any] = {
            "model": options.get("model", self._model) if options else self._model,
            "messages": messages,
            "options": chat_options,
        }
        if schema is not None:
            kwargs["format"] = schema

        logger.debug("Ollama chat call: model=%s, schema=%s, options=%s", kwargs["model"], schema is not None, chat_options)
        response = ollama.chat(**kwargs)  # type: ignore[arg-type]
        content: str = response.message.content  # type: ignore[union-attr]
        return content

    # -- feedback pipeline ---------------------------------------------------
    async def generate_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        from src.pipeline import run_feedback_pipeline
        model_override = request.model or self._model
        return await run_feedback_pipeline(request, self.chat, model_override)

    # -- mapping suggestion --------------------------------------------------
    async def suggest_mapping(self, request: MappingRequest) -> MappingResponse:
        from src.pipeline import run_mapping_suggestion
        return await run_mapping_suggestion(request, self.chat)

    # -- embeddings ----------------------------------------------------------
    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        emb_model = model or settings.embedding_model
        response = ollama.embeddings(model=emb_model, prompt=text)
        return response["embedding"]


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------

class OpenAICompatibleProvider(LLMProvider):
    """Calls any OpenAI-compatible REST API using httpx."""

    def __init__(self) -> None:
        self._base_url = settings.openai_base_url.rstrip("/")
        self._api_key = settings.openai_api_key
        self._model = settings.openai_model or settings.ollama_model

    async def _post(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": options.get("model", self._model) if options else self._model,
            "messages": messages,
            "temperature": options.get("temperature", 0.0) if options else 0.0,
        }
        if schema is not None:
            # Some providers use `response_format` with JSON schema; Ollama uses `format`
            payload["response_format"] = {"type": "json_schema", "json_schema": schema}

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        logger.debug("OpenAI-compatible call: url=%s, model=%s", f"{self._base_url}/chat/completions", payload["model"])
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            resp = await client.post(f"{self._base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        # Extract content from standard OpenAI response shape
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned from OpenAI-compatible API")
        content: str = choices[0]["message"]["content"]
        return content

    async def chat(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        return await self._post(messages, schema, options)

    async def generate_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        from src.pipeline import run_feedback_pipeline
        model_override = request.model or self._model
        return await run_feedback_pipeline(request, self.chat, model_override)

    async def suggest_mapping(self, request: MappingRequest) -> MappingResponse:
        from src.pipeline import run_mapping_suggestion
        return await run_mapping_suggestion(request, self.chat)

    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        emb_model = model or self._model
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"input": text, "model": emb_model}
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        return data["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider() -> LLMProvider:
    """Return the configured LLM provider instance."""
    provider_name = settings.llm_provider.lower()
    if provider_name == "ollama":
        return OllamaProvider()
    elif provider_name == "openai":
        return OpenAICompatibleProvider()
    raise ValueError(f"Unknown LLM provider: {provider_name}")
