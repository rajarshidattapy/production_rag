"""Shared LLM provider client (OpenAI / Anthropic) with retry handling.

Centralizes the provider dispatch + retry logic that was previously
duplicated across ``Generator``, ``FaithfulnessScorer``, and
``AnswerRelevanceScorer``.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from src.utils.retry import async_retry_with_backoff, retry_with_backoff

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _openai_api_key() -> str:
    """Get the key from the process environment or the application's .env file."""
    if key := os.getenv("OPENAI_API_KEY"):
        return key

    # Settings reads .env itself.  Import lazily to keep this client usable in
    # small scripts that do not otherwise need the application configuration.
    from src.config import settings

    return settings.openai_api_key or ""


def _openrouter_mode() -> str:
    """Read the routing override from the environment or application's .env."""
    if mode := os.getenv("RAG_OPENROUTER_MODE"):
        return mode

    from src.config import settings

    return settings.openrouter_mode


def uses_openrouter(model: str) -> bool:
    """Return whether an OpenAI-compatible model name targets OpenRouter.

    OpenRouter model IDs are provider-qualified (for example,
    ``nvidia/nemotron-3-ultra-550b-a55b:free`` or ``openai/gpt-4o``), while
    native OpenAI model IDs are not.  An OpenRouter key is also unambiguous and
    lets callers use an OpenRouter alias that does not contain a slash.

    ``RAG_OPENROUTER_MODE`` may be set to ``true`` or ``false`` to override
    automatic detection; its default is ``auto``.
    """
    mode = _openrouter_mode().strip().lower()
    if mode in {"true", "1", "yes", "on"}:
        return True
    if mode in {"false", "0", "no", "off"}:
        return False

    api_key = _openai_api_key()
    return api_key.startswith("sk-or-") or "/" in model.strip()


class LLMClient:
    """Thin wrapper around the OpenAI and Anthropic chat/messages APIs.

    Args:
        provider: "openai" or "anthropic".
        model: Model name passed to the provider API.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        provider: Literal["openai", "anthropic"] = "openai",
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.timeout = timeout

    def openai_client_kwargs(self) -> dict[str, Any]:
        """Connection options for the OpenAI SDK.

        The application reads ``OPENAI_API_KEY`` in both cases. Only the base
        URL changes for OpenRouter, so users can switch providers by changing
        ``RAG_LLM_MODEL`` without renaming their key variable.
        """
        options: dict[str, Any] = {}
        if api_key := _openai_api_key():
            options["api_key"] = api_key
        # An explicit OPENAI_BASE_URL always wins, so a proxy or local endpoint can
        # never be silently bypassed by the OpenRouter auto-detection.
        if (
            self.provider == "openai"
            and not os.getenv("OPENAI_BASE_URL")
            and uses_openrouter(self.model)
        ):
            options["base_url"] = OPENROUTER_BASE_URL
        return options

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a completion for ``prompt``, optionally with a system prompt.

        Args:
            prompt: The user message content.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            The model's text response (empty string if none).

        Raises:
            ImportError: If the required provider SDK is not installed.
            ValueError: If ``provider`` is not supported.
        """
        if self.provider == "openai":
            return self._call_openai(prompt, system, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, system, max_tokens)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    async def complete_async(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a completion for ``prompt`` asynchronously, optionally with a system prompt.

        Args:
            prompt: The user message content.
            system: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            The model's text response (empty string if none).

        Raises:
            ImportError: If the required provider SDK is not installed.
            ValueError: If ``provider`` is not supported.
        """
        if self.provider == "openai":
            return await self._call_openai_async(prompt, system, temperature, max_tokens)
        elif self.provider == "anthropic":
            return await self._call_anthropic_async(prompt, system, max_tokens)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    def _call_openai(
        self, prompt: str, system: str | None, temperature: float, max_tokens: int
    ) -> str:
        try:
            from openai import OpenAI
            from openai.types.chat import ChatCompletionMessageParam
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai") from None

        client = OpenAI(timeout=self.timeout, max_retries=0, **self.openai_client_kwargs())
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def _api_call():
            return client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        import time

        from src.utils.usage import request_usage

        start_time = time.perf_counter()
        response = retry_with_backoff(_api_call)
        latency = time.perf_counter() - start_time

        if getattr(response, "usage", None):
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
            completion_tokens = getattr(response.usage, "completion_tokens", 0)
            tracker = request_usage.get()
            if tracker is not None:
                tracker.add_call(prompt_tokens, completion_tokens, latency)

        return response.choices[0].message.content or ""

    def _call_anthropic(self, prompt: str, system: str | None, max_tokens: int) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            ) from None

        client = anthropic.Anthropic(timeout=self.timeout, max_retries=0)

        def _api_call():
            if system:
                return client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    system=system,
                )
            else:
                return client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )

        import time

        from src.utils.usage import request_usage

        start_time = time.perf_counter()
        response = retry_with_backoff(_api_call)
        latency = time.perf_counter() - start_time

        if getattr(response, "usage", None):
            prompt_tokens = getattr(response.usage, "input_tokens", 0)
            completion_tokens = getattr(response.usage, "output_tokens", 0)
            tracker = request_usage.get()
            if tracker is not None:
                tracker.add_call(prompt_tokens, completion_tokens, latency)

        return response.content[0].text if response.content else ""

    async def _call_openai_async(
        self, prompt: str, system: str | None, temperature: float, max_tokens: int
    ) -> str:
        try:
            from openai import AsyncOpenAI
            from openai.types.chat import ChatCompletionMessageParam
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai") from None

        client = AsyncOpenAI(timeout=self.timeout, max_retries=0, **self.openai_client_kwargs())
        messages: list[ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async def _api_call():
            return await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        import time

        from src.utils.usage import request_usage

        start_time = time.perf_counter()
        response = await async_retry_with_backoff(_api_call)
        latency = time.perf_counter() - start_time

        if getattr(response, "usage", None):
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
            completion_tokens = getattr(response.usage, "completion_tokens", 0)
            tracker = request_usage.get()
            if tracker is not None:
                tracker.add_call(prompt_tokens, completion_tokens, latency)

        return response.choices[0].message.content or ""

    async def _call_anthropic_async(self, prompt: str, system: str | None, max_tokens: int) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            ) from None

        client = anthropic.AsyncAnthropic(timeout=self.timeout, max_retries=0)

        async def _api_call():
            if system:
                return await client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    system=system,
                )
            else:
                return await client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )

        import time

        from src.utils.usage import request_usage

        start_time = time.perf_counter()
        response = await async_retry_with_backoff(_api_call)
        latency = time.perf_counter() - start_time

        if getattr(response, "usage", None):
            prompt_tokens = getattr(response.usage, "input_tokens", 0)
            completion_tokens = getattr(response.usage, "output_tokens", 0)
            tracker = request_usage.get()
            if tracker is not None:
                tracker.add_call(prompt_tokens, completion_tokens, latency)

        return response.content[0].text if response.content else ""
