"""Tests for main application functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from matrix_llmagent.main import MatrixLLMAgent, cli_message


class MockAPIClient:
    """Mock API client with all required methods."""

    def __init__(self, response_text: str = "Mock response"):
        self.response_text = response_text

    def extract_text_from_response(self, r):
        return self.response_text

    def has_tool_calls(self, response):
        return False

    def extract_tool_calls(self, response):
        return None

    def format_assistant_message(self, response):
        return {"role": "assistant", "content": self.response_text}

    def format_tool_results(self, tool_results):
        return {"role": "user", "content": "Tool results"}


class TestMatrixLLMAgent:
    """Test main agent functionality."""

    def test_load_config(self, temp_config_file, api_type):
        """Test configuration loading."""
        agent = MatrixLLMAgent(temp_config_file)
        assert agent.config is not None
        assert "providers" in agent.config  # Provider sections exist
        assert "matrix" in agent.config
        assert "homeserver" in agent.config["matrix"]
        assert "command" in agent.config["matrix"]


@pytest.mark.skip(reason="CLI mode disabled until Matrix monitor is implemented (Phase 4)")
class TestCLIMode:
    """Test CLI mode functionality."""

    @pytest.mark.asyncio
    async def test_cli_message_sarcastic_message(self, temp_config_file):
        """Test CLI mode with sarcastic message."""
        with patch("builtins.print") as mock_print:
            # Mock the ChatHistory import in cli_message
            with patch("matrix_llmagent.main.ChatHistory") as mock_history_class:
                mock_history = AsyncMock()
                mock_history.add_message = AsyncMock()
                mock_history.get_context.return_value = [
                    {"role": "user", "content": "!S tell me a joke"}
                ]
                # Add new chronicling methods
                mock_history.count_recent_unchronicled = AsyncMock(return_value=0)
                mock_history.get_recent_unchronicled = AsyncMock(return_value=[])
                mock_history.mark_chronicled = AsyncMock()
                mock_history_class.return_value = mock_history

                # Create a real agent
                from matrix_llmagent.main import MatrixLLMAgent

                agent = MatrixLLMAgent(temp_config_file)

                async def fake_call_raw_with_model(*args, **kwargs):
                    resp = {"output_text": "Sarcastic response"}

                    return resp, MockAPIClient("Sarcastic response"), None

                # Patch the agent creation in cli_message and model router
                with patch("matrix_llmagent.main.MatrixLLMAgent", return_value=agent):
                    with patch(
                        "matrix_llmagent.agentic_actor.actor.ModelRouter.call_raw_with_model",
                        new=AsyncMock(side_effect=fake_call_raw_with_model),
                    ):
                        await cli_message("!S tell me a joke", temp_config_file)

                        # Verify output
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        assert any(
                            "Simulating message: !S tell me a joke" in call for call in print_calls
                        )
                        assert any("Sarcastic response" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_cli_message_perplexity_message(self, temp_config_file):
        """Test CLI mode with Perplexity message."""
        with patch("builtins.print") as mock_print:
            with patch(
                "matrix_llmagent.rooms.irc.monitor.PerplexityClient"
            ) as mock_perplexity_class:
                with patch("matrix_llmagent.main.ChatHistory") as mock_history_class:
                    # Mock history to return only the current message
                    mock_history = AsyncMock()
                    mock_history.add_message = AsyncMock()
                    mock_history.get_context.return_value = [
                        {"role": "user", "content": "!p what is the weather?"}
                    ]
                    # Add new chronicling methods
                    mock_history.count_recent_unchronicled = AsyncMock(return_value=0)
                    mock_history.get_recent_unchronicled = AsyncMock(return_value=[])
                    mock_history.mark_chronicled = AsyncMock()
                    mock_history_class.return_value = mock_history

                    mock_perplexity = AsyncMock()
                    mock_perplexity.call_perplexity = AsyncMock(return_value="Weather is sunny")
                    mock_perplexity_class.return_value = mock_perplexity

                    # Create a real agent
                    from matrix_llmagent.main import MatrixLLMAgent

                    agent = MatrixLLMAgent(temp_config_file)

                    # Patch the agent creation in cli_message
                    with patch("matrix_llmagent.main.MatrixLLMAgent", return_value=agent):
                        await cli_message("!p what is the weather?", temp_config_file)

                        # Verify Perplexity was called with the actual message in context
                        mock_perplexity.call_perplexity.assert_called_once()
                        call_args = mock_perplexity.call_perplexity.call_args
                        context = call_args[0][0]  # First argument is context

                        # Verify the user message is in the context - should be the last message
                        assert len(context) >= 1
                        assert context[-1]["role"] == "user"
                        assert "!p what is the weather?" in context[-1]["content"]

                        # Verify output
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        assert any(
                            "Simulating message: !p what is the weather?" in call
                            for call in print_calls
                        )
                        assert any("Weather is sunny" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_cli_message_agent_message(self, temp_config_file):
        """Test CLI mode with agent message."""
        with patch("builtins.print") as mock_print:
            with patch("matrix_llmagent.main.AgenticLLMActor") as mock_agent_class:
                with patch("matrix_llmagent.main.ChatHistory") as mock_history_class:
                    # Mock history to return only the current message
                    mock_history = AsyncMock()
                    mock_history.add_message = AsyncMock()
                    mock_history.get_context.return_value = [
                        {"role": "user", "content": "!s search for Python news"}
                    ]
                    # Add new chronicling methods
                    mock_history.count_recent_unchronicled = AsyncMock(return_value=0)
                    mock_history.get_recent_unchronicled = AsyncMock(return_value=[])
                    mock_history.mark_chronicled = AsyncMock()
                    mock_history_class.return_value = mock_history

                    mock_agent = AsyncMock()
                    mock_agent.run_agent = AsyncMock(return_value="Agent response")
                    mock_agent_class.return_value = mock_agent

                    # Create a real agent
                    from matrix_llmagent.main import MatrixLLMAgent

                    agent = MatrixLLMAgent(temp_config_file)

                    # Patch the agent creation in cli_message
                    with patch("matrix_llmagent.main.MatrixLLMAgent", return_value=agent):
                        await cli_message("!s search for Python news", temp_config_file)

                        # Verify agent was called with context only
                        mock_agent.run_agent.assert_called_once()
                        call_args = mock_agent.run_agent.call_args
                        assert len(call_args[0]) == 1  # Only context parameter
                        context = call_args[0][0]
                        assert isinstance(context, list)  # Should be context list
                        # Verify the user message is the last in context
                        assert "!s search for Python news" in context[-1]["content"]

                        # Verify output
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        assert any(
                            "Simulating message: !s search for Python news" in call
                            for call in print_calls
                        )
                        assert any("Agent response" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_cli_message_message_content_validation(self, temp_config_file):
        """Test that CLI mode passes actual message content, not placeholder text."""
        with patch("builtins.print"):
            with patch("matrix_llmagent.main.AgenticLLMActor") as mock_agent_class:
                with patch("matrix_llmagent.main.ChatHistory") as mock_history_class:
                    # Mock history to return only the current message
                    mock_history = AsyncMock()
                    mock_history.add_message = AsyncMock()
                    mock_history.get_context.return_value = [
                        {"role": "user", "content": "!s specific test message"}
                    ]
                    # Add new chronicling methods
                    mock_history.count_recent_unchronicled = AsyncMock(return_value=0)
                    mock_history.get_recent_unchronicled = AsyncMock(return_value=[])
                    mock_history.mark_chronicled = AsyncMock()
                    mock_history_class.return_value = mock_history

                    mock_agent = AsyncMock()
                    mock_agent.run_agent = AsyncMock(return_value="Agent response")
                    mock_agent_class.return_value = mock_agent

                    # Create a real agent
                    from matrix_llmagent.main import MatrixLLMAgent

                    agent = MatrixLLMAgent(temp_config_file)

                    # Patch the agent creation in cli_message
                    with patch("matrix_llmagent.main.MatrixLLMAgent", return_value=agent):
                        await cli_message("!s specific test message", temp_config_file)

                        # Verify agent was called once for serious mode
                        mock_agent.run_agent.assert_called_once()
                        call_args = mock_agent.run_agent.call_args
                        context = call_args[0][0]

                        # This test would catch the bug where empty context resulted in "..." placeholder
                        # The user message should be the last message in context
                        assert "!s specific test message" in context[-1]["content"]
                        assert (
                            context[-1]["content"] != "..."
                        )  # Explicitly check it's not placeholder

    @pytest.mark.asyncio
    async def test_cli_message_config_not_found(self):
        """Test CLI mode handles missing config file."""
        with patch("sys.exit") as mock_exit:
            with patch("builtins.print") as mock_print:
                await cli_message("test query", "/nonexistent/config.json")

                mock_exit.assert_called_with(1)  # Just check it was called with 1, not once
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                assert any("Config file not found" in call for call in print_calls)


class TestModeShortcutParsing:
    """Test mode shortcut parsing for model comparison slots."""

    def test_determine_mode_numbered_slots(self, temp_config_file):
        """Test that !2, !3, !4 shortcuts map to serious2, serious3, serious4 modes."""
        agent = MatrixLLMAgent(temp_config_file)
        monitor = agent.matrix_monitor

        # Test !2 -> serious2
        mode, verbose = monitor.determine_mode("!2 what is osgeo?")
        assert mode == "serious2"
        assert not verbose

        # Test !3 -> serious3
        mode, verbose = monitor.determine_mode("!3 what is osgeo?")
        assert mode == "serious3"
        assert not verbose

        # Test !4 -> serious4
        mode, verbose = monitor.determine_mode("!4 what is osgeo?")
        assert mode == "serious4"
        assert not verbose

    def test_determine_mode_standard_shortcuts(self, temp_config_file):
        """Test that standard shortcuts still work correctly."""
        agent = MatrixLLMAgent(temp_config_file)
        monitor = agent.matrix_monitor

        # Test existing shortcuts
        mode, verbose = monitor.determine_mode("!s test")
        assert mode == "serious"

        mode, verbose = monitor.determine_mode("!d test")
        assert mode == "sarcastic"

        mode, verbose = monitor.determine_mode("!u test")
        assert mode == "unsafe"

        mode, verbose = monitor.determine_mode("!a test")
        assert mode == "agent"

        mode, verbose = monitor.determine_mode("!p test")
        assert mode == "perplexity"

    def test_determine_mode_verbose_with_numbered_slot(self, temp_config_file):
        """Test verbose modifier with numbered slots."""
        agent = MatrixLLMAgent(temp_config_file)
        monitor = agent.matrix_monitor

        # Test !v !2 -> serious2 with verbose
        mode, verbose = monitor.determine_mode("!v !2 what is osgeo?")
        assert mode == "serious2"
        assert verbose

    def test_determine_mode_default_mode(self, temp_config_file):
        """Test that messages without prefix use default mode."""
        agent = MatrixLLMAgent(temp_config_file)
        monitor = agent.matrix_monitor

        mode, verbose = monitor.determine_mode("what is osgeo?")
        assert mode == "serious"  # Default mode
        assert not verbose


class TestModeInheritance:
    """Test mode configuration inheritance for numbered slots."""

    def test_serious2_inherits_system_prompt(self, temp_config_file):
        """Test that serious2 inherits system_prompt from serious when not specified."""
        agent = MatrixLLMAgent(temp_config_file)
        command_config = agent.config.get("matrix", {}).get("command", {})
        modes = command_config.get("modes", {})

        # Get base serious config
        serious_cfg = modes.get("serious", {})
        assert "system_prompt" in serious_cfg

        # Get serious2 config (may not have system_prompt)
        serious2_cfg = modes.get("serious2", {})

        # Simulate inheritance logic from matrix_monitor.py
        if "system_prompt" not in serious2_cfg:
            merged_cfg = {**serious_cfg, **serious2_cfg}
            assert "system_prompt" in merged_cfg
            assert merged_cfg["system_prompt"] == serious_cfg["system_prompt"]


class TestMultiProviderRouting:
    """Test that ModelRouter correctly routes to different llamacpp providers."""

    def test_router_creates_llamacpp2_client(self, temp_config_file):
        """Test that ModelRouter creates correct client for llamacpp2."""
        with patch("matrix_llmagent.providers.llamacpp._AsyncOpenAI") as MockOpenAI:
            MockOpenAI.return_value = MagicMock()

            agent = MatrixLLMAgent(temp_config_file)

            # Get client for llamacpp2
            client = agent.model_router.client_for("llamacpp2")

            # Verify it was created with correct provider key
            assert client._provider_key == "llamacpp2"
            assert client.config["base_url"] == "http://localhost:8081/v1"

    def test_router_creates_llamacpp3_client(self, temp_config_file):
        """Test that ModelRouter creates correct client for llamacpp3."""
        with patch("matrix_llmagent.providers.llamacpp._AsyncOpenAI") as MockOpenAI:
            MockOpenAI.return_value = MagicMock()

            agent = MatrixLLMAgent(temp_config_file)

            # Get client for llamacpp3
            client = agent.model_router.client_for("llamacpp3")

            # Verify it was created with correct provider key
            assert client._provider_key == "llamacpp3"
            assert client.config["base_url"] == "http://localhost:8082/v1"


class TestSourcesCommand:
    """Test !sources and !source N command functionality.

    These tests pre-populate the cache to test the command logic
    without requiring LLM calls.
    """

    @pytest.mark.asyncio
    async def test_sources_no_cache(self, temp_config_file):
        """Test !sources when no search has been run."""
        from matrix_llmagent.main import _capture_cli_source_command

        agent = MatrixLLMAgent(temp_config_file)
        arc = "cli#test"

        # No cache populated
        result = await _capture_cli_source_command(agent, "list", None, arc)

        assert "No sources available" in result

    @pytest.mark.asyncio
    async def test_sources_with_kb_cache(self, temp_config_file):
        """Test !sources with pre-populated KB cache."""
        from matrix_llmagent.agentic_actor.tools import KnowledgeBaseResultsCache
        from matrix_llmagent.main import _capture_cli_source_command

        agent = MatrixLLMAgent(temp_config_file)
        arc = "cli#test"

        # Pre-populate KB cache
        agent.kb_cache = KnowledgeBaseResultsCache()
        agent.kb_cache.store(
            arc,
            pages=[
                {
                    "page_title": "GeoServer",
                    "url": "https://wiki.osgeo.org/GeoServer",
                    "resume": "GeoServer is an open source server for sharing geospatial data.",
                    "keywords": "GIS, mapping",
                }
            ],
            entities=[
                {
                    "entity_name": "OSGeo Foundation",
                    "entity_type": "Organization",
                    "url": "https://wiki.osgeo.org/OSGeo",
                }
            ],
        )

        # Test !sources (list)
        result = await _capture_cli_source_command(agent, "list", None, arc)

        assert "Sources from last search (Wiki)" in result
        assert "GeoServer" in result
        assert "OSGeo Foundation" in result
        assert "!source N" in result

    @pytest.mark.asyncio
    async def test_source_detail_page(self, temp_config_file):
        """Test !source N for a page result."""
        from matrix_llmagent.agentic_actor.tools import KnowledgeBaseResultsCache
        from matrix_llmagent.main import _capture_cli_source_command

        agent = MatrixLLMAgent(temp_config_file)
        arc = "cli#test"

        # Pre-populate KB cache
        agent.kb_cache = KnowledgeBaseResultsCache()
        agent.kb_cache.store(
            arc,
            pages=[
                {
                    "page_title": "QGIS Project",
                    "url": "https://wiki.osgeo.org/QGIS",
                    "resume": "QGIS is a professional GIS application.",
                    "keywords": "desktop GIS, open source",
                }
            ],
            entities=[],
        )

        # Test !source 1 (view page)
        result = await _capture_cli_source_command(agent, "view", 1, arc)

        assert "[1] QGIS Project" in result
        assert "QGIS is a professional GIS application" in result
        assert "URL:" in result
        assert "https://wiki.osgeo.org/QGIS" in result

    @pytest.mark.asyncio
    async def test_source_detail_entity(self, temp_config_file):
        """Test !source N for an entity result."""
        from matrix_llmagent.agentic_actor.tools import KnowledgeBaseResultsCache
        from matrix_llmagent.main import _capture_cli_source_command

        agent = MatrixLLMAgent(temp_config_file)
        arc = "cli#test"

        # Pre-populate KB cache with 1 page + 1 entity
        agent.kb_cache = KnowledgeBaseResultsCache()
        agent.kb_cache.store(
            arc,
            pages=[{"page_title": "Page 1", "url": "https://example.com", "resume": "Summary"}],
            entities=[
                {
                    "entity_name": "Jeff McKenna",
                    "entity_type": "Person",
                    "url": "https://wiki.osgeo.org/JeffMcKenna",
                }
            ],
        )

        # Test !source 2 (entity is index 2 after the page)
        result = await _capture_cli_source_command(agent, "view", 2, arc)

        assert "[2] Jeff McKenna (Person)" in result
        assert "https://wiki.osgeo.org/JeffMcKenna" in result

    @pytest.mark.asyncio
    async def test_source_invalid_index(self, temp_config_file):
        """Test !source N with invalid index."""
        from matrix_llmagent.agentic_actor.tools import KnowledgeBaseResultsCache
        from matrix_llmagent.main import _capture_cli_source_command

        agent = MatrixLLMAgent(temp_config_file)
        arc = "cli#test"

        # Pre-populate KB cache with 1 item
        agent.kb_cache = KnowledgeBaseResultsCache()
        agent.kb_cache.store(
            arc,
            pages=[{"page_title": "Only Page", "url": "", "resume": ""}],
            entities=[],
        )

        # Test invalid index
        result = await _capture_cli_source_command(agent, "view", 5, arc)

        assert "Invalid source number" in result

    @pytest.mark.asyncio
    async def test_sources_library_vs_kb_priority(self, temp_config_file):
        """Test that most recent cache is used when both have results."""
        import time

        from matrix_llmagent.agentic_actor.library_tool import LibraryResultsCache
        from matrix_llmagent.agentic_actor.tools import KnowledgeBaseResultsCache
        from matrix_llmagent.main import _capture_cli_source_command

        agent = MatrixLLMAgent(temp_config_file)
        arc = "cli#test"

        # Populate KB cache first
        agent.kb_cache = KnowledgeBaseResultsCache()
        agent.kb_cache.store(
            arc,
            pages=[{"page_title": "KB Page", "url": "", "resume": "From KB"}],
            entities=[],
        )

        # Wait a bit then populate library cache
        time.sleep(0.1)

        agent.library_cache = LibraryResultsCache()
        agent.library_cache.store(
            arc,
            [
                {
                    "document_title": "Library Doc",
                    "page_number": 1,
                    "content": "From Library",
                    "document_slug": "lib-doc",
                }
            ],
        )

        # Library is more recent, should be used
        result = await _capture_cli_source_command(agent, "list", None, arc)

        assert "Sources from last search:" in result  # Library format
        assert "Library Doc" in result

    @pytest.mark.asyncio
    async def test_sources_kb_more_recent(self, temp_config_file):
        """Test that KB cache is used when it's more recent."""
        import time

        from matrix_llmagent.agentic_actor.library_tool import LibraryResultsCache
        from matrix_llmagent.agentic_actor.tools import KnowledgeBaseResultsCache
        from matrix_llmagent.main import _capture_cli_source_command

        agent = MatrixLLMAgent(temp_config_file)
        arc = "cli#test"

        # Populate library cache first
        agent.library_cache = LibraryResultsCache()
        agent.library_cache.store(
            arc,
            [{"document_title": "Old Library Doc", "page_number": 1, "content": "Old"}],
        )

        # Wait a bit then populate KB cache
        time.sleep(0.1)

        agent.kb_cache = KnowledgeBaseResultsCache()
        agent.kb_cache.store(
            arc,
            pages=[{"page_title": "New KB Page", "url": "", "resume": "New from KB"}],
            entities=[],
        )

        # KB is more recent, should be used
        result = await _capture_cli_source_command(agent, "list", None, arc)

        assert "Sources from last search (Wiki)" in result  # KB format
        assert "New KB Page" in result
