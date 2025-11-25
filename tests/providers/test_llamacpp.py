"""Tests for LlamaCpp provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLlamaCppClient:
    """Test LlamaCpp client."""

    @pytest.mark.asyncio
    async def test_llamacpp_client_initialization(self):
        """Test LlamaCpp client can be initialized."""
        with patch("matrix_llmagent.providers.llamacpp._AsyncOpenAI") as MockOpenAI:
            MockOpenAI.return_value = MagicMock()

            from matrix_llmagent.providers.llamacpp import LlamaCppClient

            config = {
                "providers": {
                    "llamacpp": {
                        "base_url": "http://localhost:8080/v1",
                        "key": "not-needed",
                        "max_tokens": 2048,
                    }
                }
            }

            client = LlamaCppClient(config)
            assert client is not None
            assert client.config["base_url"] == "http://localhost:8080/v1"

    @pytest.mark.asyncio
    async def test_llamacpp_call_raw(self):
        """Test LlamaCpp API call."""
        with patch("matrix_llmagent.providers.llamacpp._AsyncOpenAI") as MockOpenAI:
            # Mock the OpenAI client
            mock_client_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.model_dump.return_value = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello from llama.cpp!",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
            mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_client_instance

            from matrix_llmagent.providers.llamacpp import LlamaCppClient

            config = {
                "providers": {
                    "llamacpp": {
                        "base_url": "http://localhost:8080/v1",
                        "key": "not-needed",
                        "max_tokens": 2048,
                    }
                }
            }

            client = LlamaCppClient(config)
            context = [{"role": "user", "content": "Hello"}]
            response = await client.call_raw(
                context=context,
                system_prompt="You are a helpful assistant",
                model="llama-3.1-8b-instruct",
            )

            assert "choices" in response
            assert response["choices"][0]["message"]["content"] == "Hello from llama.cpp!"

    @pytest.mark.asyncio
    async def test_llamacpp_extract_text(self):
        """Test extracting text from llama.cpp response."""
        with patch("matrix_llmagent.providers.llamacpp._AsyncOpenAI"):
            from matrix_llmagent.providers.llamacpp import LlamaCppClient

            config = {
                "providers": {
                    "llamacpp": {
                        "base_url": "http://localhost:8080/v1",
                        "key": "not-needed",
                    }
                }
            }

            client = LlamaCppClient(config)
            response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Test response",
                        }
                    }
                ]
            }

            text = client._extract_raw_text(response)
            assert text == "Test response"

    @pytest.mark.asyncio
    async def test_llamacpp_has_tool_calls(self):
        """Test detecting tool calls in llama.cpp response."""
        with patch("matrix_llmagent.providers.llamacpp._AsyncOpenAI"):
            from matrix_llmagent.providers.llamacpp import LlamaCppClient

            config = {
                "providers": {
                    "llamacpp": {
                        "base_url": "http://localhost:8080/v1",
                        "key": "not-needed",
                    }
                }
            }

            client = LlamaCppClient(config)

            # Response without tool calls
            response_no_tools = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "No tools here",
                        }
                    }
                ]
            }
            assert not client.has_tool_calls(response_no_tools)

            # Response with tool calls
            response_with_tools = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query": "test"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
            assert client.has_tool_calls(response_with_tools)
