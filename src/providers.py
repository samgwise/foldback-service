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

# Cache of discovered character limits per embedding model.
# Populated on first embed call via binary-search probing.
_EMBEDDING_LIMITS: dict[str, int] = {}
_DEFAULT_EMBEDDING_LIMIT = 8000  # Fallback if probing fails


async def _discover_embedding_limit(
    model: str,
    probe_fn,
) -> int:
    """Binary-search probe to discover the max character count for an embedding model.

    The `probe_fn(text)` is an async callable that returns an embedding vector on
    success, or raises an exception when the text is too long. This function finds
    the largest character count that succeeds and caches it in `_EMBEDDING_LIMITS`.

    Search range: 256 chars (safe lower bound) to 65536 chars (generous upper bound).
    Uses 12 iterations of binary search — enough precision at low overhead.
    """
    if model in _EMBEDDING_LIMITS:
        return _EMBEDDING_LIMITS[model]

    lo, hi = 256, 65536
    best = _DEFAULT_EMBEDDING_LIMIT

    # Seed that's representative of typical English text
    seed = "a "

    logger.info("Probing embedding limit for model '%s' (range %d–%d chars)", model, lo, hi)

    for _ in range(12):
        if lo >= hi:
            break
        mid = (lo + hi) // 2
        test_text = seed * mid
        try:
            await probe_fn(test_text)
            best = mid
            lo = mid + 1
        except Exception as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ["context", "too long", "exceed", "maximum", "limit", "input length"]):
                hi = mid - 1
            else:
                logger.warning("Embedding probe failed for non-length reason: %s", e)
                best = _DEFAULT_EMBEDDING_LIMIT
                break

    _EMBEDDING_LIMITS[model] = best
    logger.info("Discovered embedding limit for '%s': %d characters", model, best)
    return best


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
        self._embedding_model = settings.embedding_model
        self._embedding_limit: int | None = None

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

    async def _ensure_embedding_limit(self) -> None:
        """Discover and cache the embedding model's character limit on first use."""
        if self._embedding_limit is not None:
            return
        emb_model = self._embedding_model

        async def probe(text: str) -> list[float]:
            response = ollama.embeddings(model=emb_model, prompt=text)
            return response["embedding"]

        self._embedding_limit = await _discover_embedding_limit(emb_model, probe)

    # -- embeddings ----------------------------------------------------------
    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        emb_model = model or self._embedding_model

        # Auto-discover limit on first embed call
        await self._ensure_embedding_limit()
        max_chars = self._embedding_limit or _DEFAULT_EMBEDDING_LIMIT

        if len(text) > max_chars:
            logger.info("Truncating embedding text from %d to %d characters to fit model context limit", len(text), max_chars)
            text = text[:max_chars]
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
        self._embedding_model = settings.openai_model or settings.ollama_model
        self._embedding_limit: int | None = None

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

    async def _do_embedding(self, text: str, model: str) -> list[float]:
        """Low-level embedding call used by both probing and normal embeds."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"input": text, "model": model}
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        return data["data"][0]["embedding"]

    async def _ensure_embedding_limit(self) -> None:
        """Discover and cache the embedding model's character limit on first use."""
        if self._embedding_limit is not None:
            return
        emb_model = self._embedding_model
        self._embedding_limit = await _discover_embedding_limit(
            emb_model,
            lambda t: self._do_embedding(t, emb_model),
        )

    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        emb_model = model or self._embedding_model

        # Auto-discover limit on first embed call
        await self._ensure_embedding_limit()
        max_chars = self._embedding_limit or _DEFAULT_EMBEDDING_LIMIT

        if len(text) > max_chars:
            logger.info("Truncating embedding text from %d to %d characters to fit model context limit", len(text), max_chars)
            text = text[:max_chars]
        return await self._do_embedding(text, emb_model)


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
