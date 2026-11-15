"""
llm_engine.py — LLM engine wrapping Ollama via LangChain for EduBot.

Provides streaming and non-streaming generation, model info retrieval,
system-prompt + history composition, and robust error handling.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config

logger = logging.getLogger(__name__)


class OllamaNotRunningError(RuntimeError):
    """Raised when the Ollama server is unreachable."""


class ModelNotFoundError(ValueError):
    """Raised when the requested model is not available in Ollama."""


class LLMEngine:
    """High-level interface to Ollama models via LangChain.

    Usage::

        engine = LLMEngine()
        # Non-streaming
        text = await engine.generate("Hello!", model_name="llama3.1:8b")
        # Streaming
        async for chunk in engine.generate_stream("Hello!", model_name="llama3.1:8b"):
            print(chunk, end="")
    """

    def __init__(self, base_url: str = config.OLLAMA_BASE_URL):
        self._base_url = base_url
        self._model_cache: dict[str, ChatOllama] = {}
        self._verified_models: set[str] = set()
        logger.info("LLMEngine initialised (base_url=%s)", base_url)

    # ── Model management ─────────────────────

    def _get_model(self, model_name: str) -> ChatOllama:
        """Return (and cache) a ``ChatOllama`` instance for *model_name*."""
        if model_name not in self._model_cache:
            self._model_cache[model_name] = ChatOllama(
                model=model_name,
                base_url=self._base_url,
                temperature=0.7,
                num_predict=1024,
            )
            logger.info("Created ChatOllama instance for '%s'", model_name)
        return self._model_cache[model_name]

    # ── Message composition ──────────────────

    @staticmethod
    def _build_messages(
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> list[SystemMessage | HumanMessage | AIMessage]:
        """Compose a LangChain message list from components.

        Args:
            prompt:        The current user message.
            system_prompt: Optional system instruction.
            history:       Previous turns as ``[{"role": ..., "content": ...}]``.

        Returns:
            Ordered list of LangChain message objects.
        """
        messages: list[SystemMessage | HumanMessage | AIMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=prompt))
        return messages

    # ── Generation: non-streaming ────────────

    async def generate(
        self,
        prompt: str,
        model_name: str | None = None,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        stream: bool = False,
    ) -> str:
        """Generate a complete response (non-streaming).

        Args:
            prompt:        User message.
            model_name:    Ollama model tag. Defaults to ``config.DEFAULT_CHAT_MODEL``.
            system_prompt: System instruction prepended to the conversation.
            history:       Conversation history turns.
            stream:        Ignored; kept for API symmetry.

        Returns:
            The full assistant response text.

        Raises:
            OllamaNotRunningError: If the Ollama server cannot be reached.
            ModelNotFoundError:    If the model tag is not pulled in Ollama.
        """
        model_name = model_name or config.DEFAULT_CHAT_MODEL
        await self._ensure_available(model_name)

        model = self._get_model(model_name)
        messages = self._build_messages(prompt, system_prompt, history)

        try:
            response = await model.ainvoke(messages)
            return response.content  # type: ignore[return-value]
        except Exception as exc:
            logger.error("Generation failed (%s): %s", model_name, exc)
            raise

    # ── Generation: streaming ────────────────

    async def generate_stream(
        self,
        prompt: str,
        model_name: str | None = None,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield response chunks as they arrive from Ollama.

        Args:
            prompt:        User message.
            model_name:    Ollama model tag. Defaults to ``config.DEFAULT_CHAT_MODEL``.
            system_prompt: System instruction.
            history:       Conversation history turns.

        Yields:
            String chunks of the assistant response.

        Raises:
            OllamaNotRunningError: If the Ollama server cannot be reached.
            ModelNotFoundError:    If the model tag is not pulled in Ollama.
        """
        model_name = model_name or config.DEFAULT_CHAT_MODEL
        await self._ensure_available(model_name)

        model = self._get_model(model_name)
        messages = self._build_messages(prompt, system_prompt, history)

        try:
            async for chunk in model.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as exc:
            logger.error("Streaming generation failed (%s): %s", model_name, exc)
            yield f"\n\n[Error: {exc}]"

    # ── Model introspection ──────────────────

    async def list_models(self) -> list[dict]:
        """Query Ollama for all locally available models.

        Returns:
            List of dicts with model metadata (name, size, etc.).
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return data.get("models", [])
        except httpx.ConnectError:
            raise OllamaNotRunningError(
                f"Cannot connect to Ollama at {self._base_url}. Is it running?"
            )
        except Exception as exc:
            logger.error("Failed to list models: %s", exc)
            return []

    async def get_model_info(self, model_name: str) -> dict:
        """Retrieve detailed information about a specific model.

        Uses Ollama's ``/api/show`` endpoint.

        Args:
            model_name: The model tag (e.g. ``'llama3.1:8b'``).

        Returns:
            Dict with keys like ``parameters``, ``template``, ``modelfile``, etc.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{self._base_url}/api/show",
                    json={"name": model_name},
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ModelNotFoundError(
                    f"Model '{model_name}' not found. Pull it with: ollama pull {model_name}"
                )
            raise
        except httpx.ConnectError:
            raise OllamaNotRunningError(
                f"Cannot connect to Ollama at {self._base_url}. Is it running?"
            )

    async def is_ollama_running(self) -> bool:
        """Return ``True`` if the Ollama server responds to a health check."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def get_available_model_names(self) -> list[str]:
        """Return a sorted list of locally-available model tags."""
        models = await self.list_models()
        return sorted(m.get("name", m.get("model", "")) for m in models)

    # ── Internal helpers ─────────────────────

    async def _ensure_available(self, model_name: str) -> None:
        """Verify that Ollama is running and the model is pulled."""
        if model_name in self._verified_models:
            return
        if not await self.is_ollama_running():
            raise OllamaNotRunningError(
                f"Ollama is not running at {self._base_url}. "
                "Start it with: ollama serve"
            )
        available = await self.get_available_model_names()
        # Accept exact or base-name match
        matched = any(model_name == a or model_name.split(":")[0] == a.split(":")[0] for a in available)
        if not matched:
            raise ModelNotFoundError(
                f"Model '{model_name}' is not available. "
                f"Available: {available}. Pull it with: ollama pull {model_name}"
            )
        self._verified_models.add(model_name)

    def build_full_prompt_string(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Return a human-readable representation of the full prompt chain.

        Useful for explainability / debugging.

        Args:
            prompt:        User message.
            system_prompt: System instruction.
            history:       Conversation history turns.

        Returns:
            Formatted multi-line string.
        """
        parts: list[str] = []
        if system_prompt:
            parts.append(f"[SYSTEM]\n{system_prompt}\n")
        if history:
            parts.append("[CONVERSATION HISTORY]")
            for msg in history:
                label = "User" if msg["role"] == "user" else "Assistant"
                parts.append(f"  {label}: {msg['content']}")
            parts.append("")
        parts.append(f"[USER QUERY]\n{prompt}")
        return "\n".join(parts)
