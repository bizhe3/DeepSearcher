"""LLM client wrappers for Anthropic and OpenAI-compatible APIs."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Literal, Tuple

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Thin async chat client that adapts OpenAI-style messages to Anthropic."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        """Initialize an AsyncAnthropic client with API key and default model."""
        try:
            import anthropic
        except ImportError as error:
            raise ImportError("anthropic package is required to use AnthropicClient.") from error

        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._rate_limit_error = anthropic.RateLimitError
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def chat(
        self,
        messages: List[dict],
        response_format: Literal["text", "json"] = "text",
    ) -> str:
        """Send chat messages and return text, retrying once on rate limits."""
        anthropic_messages, system_prompt = self._prepare_messages(messages, response_format)

        request_payload: Dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": 1024,
        }
        if system_prompt:
            request_payload["system"] = system_prompt

        for attempt in range(2):
            try:
                response = self._client.messages.create(**request_payload)
                if inspect.isawaitable(response):
                    response = await response
                usage = getattr(response, "usage", None)
                if usage is None:
                    usage = type("Usage", (), {"input_tokens": 0, "output_tokens": 0})()
                self.total_input_tokens += int(usage.input_tokens or 0)
                self.total_output_tokens += int(usage.output_tokens or 0)
                logger.debug(
                    f"tokens input={usage.input_tokens} output={usage.output_tokens} model={self.model}"
                )
                return self._response_to_text(response)
            except Exception as error:
                if isinstance(error, self._rate_limit_error) and attempt == 0:
                    await asyncio.sleep(5)
                    continue
                raise

        raise RuntimeError("Unexpected retry loop termination.")

    @staticmethod
    def _prepare_messages(
        messages: List[dict],
        response_format: Literal["text", "json"],
    ) -> Tuple[List[Dict[str, str]], str]:
        """Convert OpenAI-style messages into Anthropic input fields."""
        system_parts: List[str] = []
        anthropic_messages: List[Dict[str, str]] = []

        for message in messages:
            if not isinstance(message, dict):
                continue

            role = str(message.get("role", "user"))
            content = AnthropicClient._to_text(message.get("content", ""))
            if role == "system":
                if content:
                    system_parts.append(content)
                continue

            mapped_role = role if role in {"user", "assistant"} else "user"
            anthropic_messages.append({"role": mapped_role, "content": content})

        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": ""})

        if response_format == "json":
            system_parts.append("Reply in valid JSON only.")

        system_prompt = "\n\n".join(part for part in system_parts if part).strip()
        return anthropic_messages, system_prompt

    @staticmethod
    def _response_to_text(response: Any) -> str:
        """Extract text blocks from an Anthropic messages.create response."""
        content = getattr(response, "content", None)
        if content is None and isinstance(response, dict):
            content = response.get("content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                text = None
                if isinstance(block, dict):
                    text = block.get("text")
                else:
                    text = getattr(block, "text", None)

                if isinstance(text, str):
                    parts.append(text)

            return "".join(parts).strip()

        return "" if content is None else str(content).strip()

    @staticmethod
    def _to_text(content: Any) -> str:
        """Normalize message content to plain text for Anthropic requests."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
            return "".join(parts)

        return "" if content is None else str(content)


class OpenAICompatibleClient:
    """Async chat client for OpenAI-compatible APIs (DeepSeek, etc.)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        """Initialize an AsyncOpenAI client pointed at the given base URL."""
        try:
            from openai import AsyncOpenAI
        except ImportError as error:
            raise ImportError("openai package is required to use OpenAICompatibleClient.") from error

        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat(
        self,
        messages: List[dict],
        response_format: Literal["text", "json"] = "text",
    ) -> str:
        """Send chat messages and return assistant text content."""
        prepared = self._prepare_messages(messages, response_format)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=prepared,
            max_tokens=4096,
        )
        if inspect.isawaitable(response):
            response = await response
        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        if not content.strip():
            content = getattr(message, "reasoning_content", None) or ""
        return content

    @staticmethod
    def _prepare_messages(
        messages: List[dict],
        response_format: Literal["text", "json"],
    ) -> List[Dict[str, str]]:
        """Pass messages through, appending JSON instruction when needed."""
        result: List[Dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            result.append({"role": role, "content": content})

        if response_format == "json" and result:
            result[0] = dict(result[0])
            result[0]["content"] = result[0]["content"] + "\n\nReply in valid JSON only."

        return result


