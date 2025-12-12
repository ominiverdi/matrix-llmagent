"""llama.cpp client using OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
from typing import Any

from . import BaseAPIClient

logger = logging.getLogger(__name__)

try:
    # Lazy import so the package is optional until installed
    from openai import AsyncOpenAI as _AsyncOpenAI
except Exception:  # pragma: no cover - handled at runtime
    _AsyncOpenAI = None  # type: ignore


class LlamaCppClient(BaseAPIClient):
    """llama.cpp API client using OpenAI-compatible API.

    This client connects to a llama.cpp server running with the --api-key flag
    or without authentication. The server provides an OpenAI-compatible API.

    Example llama.cpp server command:
        ./llama-server -m model.gguf --port 8080
    """

    def __init__(self, config: dict[str, Any]):
        providers = config.get("providers", {}) if isinstance(config, dict) else {}
        cfg = providers.get("llamacpp", {})
        super().__init__(cfg)

        if _AsyncOpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed. Run 'uv sync' to install dependencies."
            )

        # llama.cpp server configuration
        base_url = self.config.get("base_url", "http://localhost:8080/v1")
        api_key = self.config.get("key", "not-needed")  # llama.cpp doesn't require auth by default

        self._client = _AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        logger.info(f"LlamaCpp client initialized with base_url: {base_url}")

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert internal tool schema to OpenAI Chat Completion function tools.

        Note: llama.cpp has limited tool support. Function calling may not work
        with all models. Check llama.cpp documentation for compatible models.
        """
        converted = []
        if not tools:
            return converted
        for tool in tools:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
            )
        return converted

    async def call_raw(
        self,
        context: list[dict],
        system_prompt: str,
        model: str,
        tools: list | None = None,
        tool_choice: list | None = None,
        reasoning_effort: str = "minimal",
        modalities: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Call llama.cpp API with context and system prompt.

        Args:
            context: List of message dictionaries with 'role' and 'content'
            system_prompt: System prompt to prepend
            model: Model name (can be anything, llama.cpp uses loaded model)
            tools: Optional list of tool definitions
            tool_choice: Tool choice strategy (ignored for llama.cpp)
            reasoning_effort: Reasoning effort level (ignored for llama.cpp)
            modalities: Modalities to use (ignored for llama.cpp)
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with API response in OpenAI format
        """
        # Build messages array with system prompt
        messages = [{"role": "system", "content": system_prompt}] + context

        # Prepare request parameters
        request_params: dict[str, Any] = {
            "model": model,  # llama.cpp ignores this, uses loaded model
            "messages": messages,
        }

        # Add max_tokens if specified
        if max_tokens is None:
            max_tokens = self.config.get("max_tokens", 2048)
        if max_tokens:
            request_params["max_tokens"] = max_tokens

        # Add tools if provided
        # Note: Tool support in llama.cpp is limited and model-dependent
        if tools:
            converted_tools = self._convert_tools(tools)
            if converted_tools:
                request_params["tools"] = converted_tools
                logger.debug(f"Adding {len(converted_tools)} tools to llama.cpp request")

        try:
            logger.debug(f"Calling llama.cpp API with {len(messages)} messages")
            response = await self._client.chat.completions.create(**request_params)

            # Convert to dict for consistent handling
            result = response.model_dump()
            logger.debug("llama.cpp API call successful")
            return result

        except Exception as e:
            logger.error(f"llama.cpp API error: {e}")
            return {"error": str(e)}

    def _extract_raw_text(self, response: dict) -> str:
        """Extract text content from llama.cpp response (OpenAI format)."""
        if "choices" not in response or not response["choices"]:
            return ""

        choice = response["choices"][0]
        message = choice.get("message", {})
        return message.get("content", "")

    def has_tool_calls(self, response: dict) -> bool:
        """Check if response contains tool calls."""
        if "choices" not in response or not response["choices"]:
            return False

        choice = response["choices"][0]
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])

        return bool(tool_calls)

    def extract_tool_calls(self, response: dict) -> list[dict] | None:
        """Extract tool calls from llama.cpp response.

        Returns:
            List of dicts with 'id', 'name', 'input' keys or None
        """
        if not self.has_tool_calls(response):
            return None

        choice = response["choices"][0]
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])

        result = []
        for tc in tool_calls:
            function = tc.get("function", {})

            # Parse arguments from JSON string to dict
            arguments_str = function.get("arguments", "{}")
            try:
                arguments_dict = (
                    json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool arguments: {arguments_str}")
                arguments_dict = {}

            result.append(
                {
                    "id": tc.get("id", ""),
                    "name": function.get("name", ""),
                    "input": arguments_dict,
                }
            )

        return result

    def format_assistant_message(self, response: dict) -> dict:
        """Format assistant response for conversation history (OpenAI format)."""
        if "choices" not in response or not response["choices"]:
            return {"role": "assistant", "content": ""}

        choice = response["choices"][0]
        message = choice.get("message", {})

        # Return the message as-is for OpenAI compatibility
        return {
            "role": "assistant",
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls"),
        }

    def format_tool_results(self, tool_results: list[dict]) -> list[dict]:
        """Format tool results for next API call (OpenAI format).

        Args:
            tool_results: List of dicts with 'tool_use_id', 'type', 'content'
                         (Anthropic format from actor.py)

        Returns:
            List of tool message dicts
        """
        formatted = []
        for result in tool_results:
            # Handle Anthropic-style tool_use_id (from actor.py)
            tool_id = result.get("tool_use_id", result.get("tool_call_id", ""))

            # Extract content - could be string or complex structure
            content = result.get("content", "")
            if isinstance(content, list):
                # Anthropic content blocks - convert to string
                content_str = ""
                for block in content:
                    if block.get("type") == "text":
                        content_str += block.get("text", "")
                    elif block.get("type") == "image":
                        content_str += "[Image]"
            else:
                content_str = str(content)

            formatted.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": content_str,
                }
            )
        return formatted
