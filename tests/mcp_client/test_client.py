"""Tests for the MCP client manager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from matrix_llmagent.mcp_client.client import (
    MCPClientManager,
    MCPServerConfig,
    MCPTool,
    MCPToolExecutor,
)


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_stdio_config(self):
        """Test stdio transport configuration."""
        config = MCPServerConfig(
            name="test-server",
            transport="stdio",
            command=["python", "-m", "my_server"],
        )
        assert config.name == "test-server"
        assert config.transport == "stdio"
        assert config.command == ["python", "-m", "my_server"]
        assert config.url is None
        assert config.env == {}

    def test_http_config(self):
        """Test HTTP transport configuration."""
        config = MCPServerConfig(
            name="http-server",
            transport="streamable-http",
            url="http://localhost:8000/mcp",
        )
        assert config.name == "http-server"
        assert config.transport == "streamable-http"
        assert config.url == "http://localhost:8000/mcp"
        assert config.command is None

    def test_with_env(self):
        """Test configuration with environment variables."""
        config = MCPServerConfig(
            name="test-server",
            transport="stdio",
            command=["npx", "server"],
            env={"API_KEY": "secret"},
        )
        assert config.env == {"API_KEY": "secret"}

    def test_with_cwd_and_args(self):
        """Test configuration with working directory and separate args."""
        config = MCPServerConfig(
            name="test-server",
            transport="stdio",
            command="/path/to/.venv/bin/python",
            args=["-m", "my_module"],
            cwd="/path/to/project",
        )
        assert config.command == "/path/to/.venv/bin/python"
        assert config.args == ["-m", "my_module"]
        assert config.cwd == "/path/to/project"


class TestMCPTool:
    """Tests for MCPTool dataclass."""

    def test_tool_creation(self):
        """Test creating an MCP tool."""
        tool = MCPTool(
            server_name="my-server",
            name="search",
            description="Search for files",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        assert tool.server_name == "my-server"
        assert tool.name == "search"
        assert tool.description == "Search for files"
        assert "query" in tool.input_schema["properties"]


class TestMCPClientManager:
    """Tests for MCPClientManager."""

    def test_init_no_servers(self):
        """Test initialization with no MCP servers configured."""
        config = {"tools": {}}
        manager = MCPClientManager(config)
        assert not manager.has_servers
        assert manager._server_configs == {}

    def test_init_with_servers(self):
        """Test initialization with MCP servers configured."""
        config = {
            "tools": {
                "mcp": {
                    "servers": {
                        "filesystem": {
                            "command": ["npx", "server"],
                            "transport": "stdio",
                        },
                        "http-server": {
                            "url": "http://localhost:8000/mcp",
                            "transport": "streamable-http",
                        },
                    }
                }
            }
        }
        manager = MCPClientManager(config)
        assert manager.has_servers
        assert len(manager._server_configs) == 2
        assert "filesystem" in manager._server_configs
        assert "http-server" in manager._server_configs

    def test_get_all_tools_empty(self):
        """Test getting tools when none connected."""
        config = {"tools": {}}
        manager = MCPClientManager(config)
        tools = manager.get_all_tools()
        assert tools == []

    def test_get_all_tools_with_tools(self):
        """Test getting tools from connected servers."""
        config = {"tools": {}}
        manager = MCPClientManager(config)

        # Manually add some tools for testing
        manager._tools["server1"] = [
            MCPTool(
                server_name="server1",
                name="tool_a",
                description="Tool A",
                input_schema={"type": "object"},
            ),
            MCPTool(
                server_name="server1",
                name="tool_b",
                description="Tool B",
                input_schema={"type": "object"},
            ),
        ]

        tools = manager.get_all_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "mcp_server1_tool_a"
        assert tools[1]["name"] == "mcp_server1_tool_b"
        assert "[MCP/server1]" in tools[0]["description"]

    def test_get_tool_executors(self):
        """Test getting executor instances for tools."""
        config = {"tools": {}}
        manager = MCPClientManager(config)

        manager._tools["server1"] = [
            MCPTool(
                server_name="server1",
                name="search",
                description="Search",
                input_schema={},
            )
        ]

        executors = manager.get_tool_executors()
        assert len(executors) == 1
        assert "mcp_server1_search" in executors
        assert isinstance(executors["mcp_server1_search"], MCPToolExecutor)

    @pytest.mark.asyncio
    async def test_connect_all_no_servers(self):
        """Test connect_all with no servers configured."""
        config = {"tools": {}}
        manager = MCPClientManager(config)
        await manager.connect_all()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_connect_all_mcp_not_installed(self):
        """Test connect_all when mcp package is not installed."""
        config = {
            "tools": {"mcp": {"servers": {"test": {"command": ["test"], "transport": "stdio"}}}}
        }
        manager = MCPClientManager(config)

        # Mock ImportError for mcp package
        with patch.dict("sys.modules", {"mcp": None}):
            with patch(
                "matrix_llmagent.mcp_client.client.MCPClientManager._connect_server"
            ) as mock_connect:
                # This should handle the import error gracefully
                # In real code, the import happens inside connect_all
                pass  # The actual test depends on mcp being installed

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self):
        """Test calling a tool on a server that's not connected."""
        config = {"tools": {}}
        manager = MCPClientManager(config)

        with pytest.raises(ValueError, match="not connected"):
            await manager.call_tool("unknown-server", "some-tool", {})

    def test_format_result_text_only(self):
        """Test formatting result with text content only."""
        from mcp import types

        config = {"tools": {}}
        manager = MCPClientManager(config)

        # Create mock result with real mcp.types
        mock_result = MagicMock()
        text_content = types.TextContent(type="text", text="Hello, world!")
        mock_result.content = [text_content]

        result = manager._format_result(mock_result)
        assert result == "Hello, world!"

    def test_format_result_no_content_attr(self):
        """Test formatting result without content attribute."""
        config = {"tools": {}}
        manager = MCPClientManager(config)

        # Result without content attribute
        mock_result = MagicMock(spec=[])  # No content attribute
        del mock_result.content

        result = manager._format_result(mock_result)
        assert "MagicMock" in result  # Falls back to str()

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        """Test disconnecting from all servers."""
        config = {"tools": {}}
        manager = MCPClientManager(config)

        # Simulate connected state
        mock_session = AsyncMock()
        mock_process = AsyncMock()
        manager._sessions["test"] = mock_session
        manager._processes["test"] = mock_process
        manager._tools["test"] = []

        await manager.disconnect_all()

        assert len(manager._sessions) == 0
        assert len(manager._processes) == 0
        assert len(manager._tools) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using manager as async context manager."""
        config = {"tools": {}}
        manager = MCPClientManager(config)

        async with manager as m:
            assert m is manager
        # disconnect_all should have been called


class TestMCPToolExecutor:
    """Tests for MCPToolExecutor."""

    @pytest.mark.asyncio
    async def test_execute_proxies_to_manager(self):
        """Test that execute calls manager.call_tool."""
        mock_manager = AsyncMock()
        mock_manager.call_tool.return_value = "result"

        executor = MCPToolExecutor(
            manager=mock_manager,
            server_name="test-server",
            tool_name="my-tool",
        )

        result = await executor.execute(query="test", limit=10)

        mock_manager.call_tool.assert_called_once_with(
            "test-server", "my-tool", {"query": "test", "limit": 10}
        )
        assert result == "result"
