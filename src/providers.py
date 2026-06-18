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
_MIN_EMBEDDING_LIMIT = 4096     # Never go below this — representative English text fits easily


def _discover_embedding_limit(model: str) -> int:
    """Discover embedding limit by querying the model's architecture info via Ollama.

    Calls `ollama.show` to retrieve the hardcoded context_length from the
    model's GGUF metadata, then converts tokens to a conservative character estimate.
    """
    if model in _EMBEDDING_LIMITS:
        return _EMBEDDING_LIMITS[model]

    # Known limits for common models (characters) used as fallback.
    KNOWN_LIMITS = {
        "nomic-embed-text": 24000,
        "nomic-embed-text:latest": 24000,
        "mxbai-embed-large": 1500,
        "all-minilm": 750,
    }
    
    try:
        # Try to get context length from Ollama model info
        response = ollama.show(model=model)
        model_info = response.get("model_info", {})
        
        context_tokens = 0
        for key, val in model_info.items():
            if key.endswith("context_length"):
                context_tokens = val
                break
        
        if context_tokens > 0:
            # Conservative estimate: 2.5 chars per token for English text
            char_limit = int(context_tokens * 2.5)
            logger.info("Retrieved embedding limit for '%s': %d tokens -> %d chars", model, context_tokens, char_limit)
            _EMBEDDING_LIMITS[model] = char_limit
            return char_limit
            
    except Exception as e:
        logger.warning("Could not fetch model info from Ollama for '%s': %s — using default", model, e)

    # Fallback
    limit = KNOWN_LIMITS.get(model, _DEFAULT_EMBEDDING_LIMIT)
    logger.info("Using fallback embedding limit for '%s': %d characters", model, limit)
    _EMBEDDING_LIMITS[model] = limit
    return limit


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

    def _ensure_embedding_limit(self) -> None:
        """Discover and cache the embedding model's character limit on first use."""
        if self._embedding_limit is not None:
            return
        self._embedding_limit = _discover_embedding_limit(self._embedding_model)

    # -- embeddings ----------------------------------------------------------
    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        emb_model = model or self._embedding_model

        # Auto-discover limit on first embed call
        self._ensure_embedding_limit()
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

    def _ensure_embedding_limit(self) -> None:
        """Discover and cache the embedding model's character limit on first use."""
        if self._embedding_limit is not None:
            return
        self._embedding_limit = _discover_embedding_limit(self._embedding_model)

    async def embed_text(self, text: str, model: str | None = None) -> list[float]:
        emb_model = model or self._embedding_model

        # Auto-discover limit on first embed call
        self._ensure_embedding_limit()
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
