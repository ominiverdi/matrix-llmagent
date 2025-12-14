"""Tests for LlamaCpp provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from matrix_llmagent.providers.llamacpp import _normalize_messages


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

    @pytest.mark.asyncio
    async def test_llamacpp_multi_provider_initialization(self):
        """Test LlamaCpp client can be initialized with different provider keys."""
        with patch("matrix_llmagent.providers.llamacpp._AsyncOpenAI") as MockOpenAI:
            MockOpenAI.return_value = MagicMock()

            from matrix_llmagent.providers.llamacpp import LlamaCppClient

            config = {
                "providers": {
                    "llamacpp": {
                        "base_url": "http://localhost:8080/v1",
                        "key": "not-needed",
                        "max_tokens": 2048,
                    },
                    "llamacpp2": {
                        "base_url": "http://localhost:8081/v1",
                        "key": "not-needed",
                        "max_tokens": 2048,
                    },
                    "llamacpp3": {
                        "base_url": "http://localhost:8082/v1",
                        "key": "not-needed",
                        "max_tokens": 2048,
                    },
                }
            }

            # Test default provider
            client1 = LlamaCppClient(config)
            assert client1.config["base_url"] == "http://localhost:8080/v1"
            assert client1._provider_key == "llamacpp"

            # Test llamacpp2 provider
            client2 = LlamaCppClient(config, provider_key="llamacpp2")
            assert client2.config["base_url"] == "http://localhost:8081/v1"
            assert client2._provider_key == "llamacpp2"

            # Test llamacpp3 provider
            client3 = LlamaCppClient(config, provider_key="llamacpp3")
            assert client3.config["base_url"] == "http://localhost:8082/v1"
            assert client3._provider_key == "llamacpp3"

    @pytest.mark.asyncio
    async def test_llamacpp_connection_error_friendly_message(self):
        """Test that connection errors return friendly error message."""
        with patch("matrix_llmagent.providers.llamacpp._AsyncOpenAI") as MockOpenAI:
            mock_client_instance = MagicMock()
            # Simulate connection refused error
            mock_client_instance.chat.completions.create = AsyncMock(
                side_effect=Exception("Connection refused")
            )
            MockOpenAI.return_value = mock_client_instance

            from matrix_llmagent.providers.llamacpp import LlamaCppClient

            config = {
                "providers": {
                    "llamacpp2": {
                        "base_url": "http://localhost:8081/v1",
                        "key": "not-needed",
                    }
                }
            }

            client = LlamaCppClient(config, provider_key="llamacpp2")
            context = [{"role": "user", "content": "Hello"}]
            response = await client.call_raw(
                context=context,
                system_prompt="You are a helpful assistant",
                model="test-model",
            )

            assert "error" in response
            assert "Model slot 2 is not active" in response["error"]

    @pytest.mark.asyncio
    async def test_llamacpp_default_slot_connection_error(self):
        """Test that default slot connection error shows 'default'."""
        with patch("matrix_llmagent.providers.llamacpp._AsyncOpenAI") as MockOpenAI:
            mock_client_instance = MagicMock()
            mock_client_instance.chat.completions.create = AsyncMock(
                side_effect=Exception("ConnectError: connection failed")
            )
            MockOpenAI.return_value = mock_client_instance

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
            response = await client.call_raw(
                context=[{"role": "user", "content": "Hello"}],
                system_prompt="Test",
                model="test-model",
            )

            assert "error" in response
            assert "Model slot default is not active" in response["error"]


class TestNormalizeMessages:
    """Test message normalization for strict alternation."""

    def test_normalize_merges_consecutive_user_messages(self):
        """Test that consecutive user messages are merged."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First question"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Answer"},
        ]
        result = _normalize_messages(messages)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert "First question" in result[1]["content"]
        assert "Second question" in result[1]["content"]
        assert result[2]["role"] == "assistant"

    def test_normalize_preserves_tool_calls(self):
        """Test that assistant messages with tool_calls are not merged."""
        messages = [
            {"role": "user", "content": "Search for X"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "function": {"name": "search"}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": "Results"},
            {"role": "assistant", "content": "Here are the results"},
        ]
        result = _normalize_messages(messages)

        assert len(result) == 4  # All preserved
        assert result[1].get("tool_calls") is not None

    def test_normalize_preserves_tool_messages(self):
        """Test that tool messages are preserved as-is."""
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "Result 1"},
            {"role": "tool", "tool_call_id": "2", "content": "Result 2"},
        ]
        result = _normalize_messages(messages)

        # Tool messages should not be merged even if consecutive
        assert len(result) == 3

    def test_normalize_handles_empty_messages(self):
        """Test that empty message list is handled."""
        assert _normalize_messages([]) == []

    def test_normalize_single_message(self):
        """Test that single message passes through."""
        messages = [{"role": "user", "content": "Hello"}]
        result = _normalize_messages(messages)
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_normalize_already_alternating(self):
        """Test that already alternating messages are unchanged."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Assistant"},
            {"role": "user", "content": "User again"},
        ]
        result = _normalize_messages(messages)
        assert len(result) == 4
