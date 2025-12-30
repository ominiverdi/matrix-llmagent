"""MCP client manager for connecting to MCP servers.

This module provides the MCPClientManager class for managing connections
to multiple MCP servers and exposing their tools to the agent.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    transport: str  # "stdio" or "streamable-http"
    command: list[str] | None = None  # For stdio transport
    args: list[str] | None = None  # Arguments for command (alternative to embedding in command)
    cwd: str | None = None  # Working directory for stdio transport
    url: str | None = None  # For HTTP transport
    env: dict[str, str] = field(default_factory=dict)  # Environment variables for stdio
    headers: dict[str, str] = field(default_factory=dict)  # HTTP headers for HTTP transport


@dataclass
class MCPTool:
    """A tool discovered from an MCP server."""

    server_name: str
    name: str
    description: str
    input_schema: dict


class MCPToolExecutor:
    """Executor that proxies tool calls to an MCP server.

    This wraps MCP tool calls in our executor interface so they can
    be used alongside native tools in the agent.
    """

    def __init__(self, manager: "MCPClientManager", server_name: str, tool_name: str):
        self.manager = manager
        self.server_name = server_name
        self.tool_name = tool_name

    async def execute(self, **kwargs: Any) -> str | list[dict]:
        """Execute the MCP tool with given arguments."""
        return await self.manager.call_tool(self.server_name, self.tool_name, kwargs)


class MCPClientManager:
    """Manages connections to multiple MCP servers.

    This class handles:
    - Connecting to MCP servers via stdio or HTTP transports
    - Discovering tools from connected servers
    - Proxying tool calls to the appropriate server
    - Formatting MCP results for our tool system

    Usage:
        manager = MCPClientManager(config)
        await manager.connect_all()
        tools = manager.get_all_tools()  # Returns list of tool definitions
        executors = manager.get_tool_executors()  # Returns executor dict
    """

    def __init__(self, config: dict):
        """Initialize the MCP client manager.

        Args:
            config: Full application config dict. MCP servers are read from
                   config["tools"]["mcp"]["servers"].
        """
        self.config = config
        self._server_configs: dict[str, MCPServerConfig] = {}
        self._tools: dict[str, list[MCPTool]] = {}  # server_name -> tools
        self._sessions: dict[str, Any] = {}  # server_name -> (read, write, session)
        self._processes: dict[str, Any] = {}  # server_name -> process context managers
        self._connected = False

        # Parse server configurations
        mcp_config = config.get("tools", {}).get("mcp", {})
        servers = mcp_config.get("servers", {})

        for name, server_cfg in servers.items():
            # Support both "transport" and "type" keys (type is standard MCP config format)
            transport = server_cfg.get("transport") or server_cfg.get("type", "stdio")
            self._server_configs[name] = MCPServerConfig(
                name=name,
                transport=transport,
                command=server_cfg.get("command"),
                args=server_cfg.get("args"),
                cwd=server_cfg.get("cwd"),
                url=server_cfg.get("url"),
                env=server_cfg.get("env", {}),
                headers=server_cfg.get("headers", {}),
            )

    @property
    def has_servers(self) -> bool:
        """Check if any MCP servers are configured."""
        return len(self._server_configs) > 0

    async def connect_all(self) -> None:
        """Connect to all configured MCP servers.

        Servers that fail to connect are logged but don't prevent
        other servers from connecting.
        """
        if not self.has_servers:
            logger.debug("No MCP servers configured")
            return

        try:
            from mcp import ClientSession
        except ImportError:
            logger.warning(
                "MCP package not installed. Install with: pip install mcp. "
                "MCP servers will not be available."
            )
            return

        for name, server_cfg in self._server_configs.items():
            try:
                await self._connect_server(name, server_cfg)
            except Exception as e:
                logger.error(f"Failed to connect to MCP server '{name}': {e}")

        self._connected = True
        total_tools = sum(len(tools) for tools in self._tools.values())
        logger.info(
            f"MCP client connected to {len(self._sessions)} servers, {total_tools} tools available"
        )

    async def _connect_server(self, name: str, cfg: MCPServerConfig) -> None:
        """Connect to a single MCP server and discover its tools."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        if cfg.transport == "stdio":
            if not cfg.command:
                raise ValueError(f"MCP server '{name}': stdio transport requires 'command'")

            # Support both styles:
            # 1. command: ["python", "-m", "module"] (args embedded in command)
            # 2. command: "python", args: ["-m", "module"] (separate args)
            if isinstance(cfg.command, list):
                command = cfg.command[0]
                args = cfg.command[1:] if len(cfg.command) > 1 else []
            else:
                command = cfg.command
                args = cfg.args or []

            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=cfg.env if cfg.env else None,
                cwd=cfg.cwd,
            )

            # Create the stdio client context
            stdio_ctx = stdio_client(server_params)
            read, write = await stdio_ctx.__aenter__()

            # Create session
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()

            # Store for cleanup
            self._processes[name] = stdio_ctx
            self._sessions[name] = session

            # Discover tools
            tools_response = await session.list_tools()
            self._tools[name] = [
                MCPTool(
                    server_name=name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                )
                for tool in tools_response.tools
            ]

            logger.info(f"Connected to MCP server '{name}' (stdio): {len(self._tools[name])} tools")

        elif cfg.transport in ("streamable-http", "http"):
            if not cfg.url:
                raise ValueError(f"MCP server '{name}': HTTP transport requires 'url'")

            # Create the HTTP client context, with optional headers
            import httpx

            http_client = None
            if cfg.headers:
                http_client = httpx.AsyncClient(headers=cfg.headers)

            http_ctx = streamable_http_client(cfg.url, http_client=http_client)
            read, write, _ = await http_ctx.__aenter__()

            # Create session
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()

            # Store for cleanup
            self._processes[name] = http_ctx
            self._sessions[name] = session

            # Discover tools
            tools_response = await session.list_tools()
            self._tools[name] = [
                MCPTool(
                    server_name=name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                )
                for tool in tools_response.tools
            ]

            logger.info(f"Connected to MCP server '{name}' (HTTP): {len(self._tools[name])} tools")

        else:
            raise ValueError(f"MCP server '{name}': unknown transport '{cfg.transport}'")

    def get_all_tools(self) -> list[dict]:
        """Get all tools from all connected servers in our Tool format.

        Returns tool definitions compatible with the TOOLS list format,
        with names prefixed by server name to avoid collisions.
        """
        tools = []
        for server_name, server_tools in self._tools.items():
            for mcp_tool in server_tools:
                # Create unique tool name: mcp_<server>_<tool>
                # Using underscores instead of colons for provider compatibility
                safe_server = server_name.replace("-", "_").replace(":", "_")
                safe_tool = mcp_tool.name.replace("-", "_").replace(":", "_")
                full_name = f"mcp_{safe_server}_{safe_tool}"
                tools.append(
                    {
                        "name": full_name,
                        "description": f"[MCP/{server_name}] {mcp_tool.description}",
                        "input_schema": mcp_tool.input_schema
                        or {"type": "object", "properties": {}},
                        "persist": "summary",
                        # Internal metadata for routing
                        "_mcp_server": server_name,
                        "_mcp_tool_name": mcp_tool.name,
                    }
                )
        return tools

    def get_tool_executors(self) -> dict[str, MCPToolExecutor]:
        """Get executor instances for all MCP tools.

        Returns a dict mapping tool names to MCPToolExecutor instances,
        ready to be merged into the agent's tool_executors dict.
        """
        executors = {}
        for server_name, server_tools in self._tools.items():
            for mcp_tool in server_tools:
                # Must match naming in get_all_tools()
                safe_server = server_name.replace("-", "_").replace(":", "_")
                safe_tool = mcp_tool.name.replace("-", "_").replace(":", "_")
                full_name = f"mcp_{safe_server}_{safe_tool}"
                executors[full_name] = MCPToolExecutor(
                    manager=self,
                    server_name=server_name,
                    tool_name=mcp_tool.name,
                )
        return executors

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict
    ) -> str | list[dict]:
        """Call a tool on an MCP server.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool to call.
            arguments: Tool arguments as a dict.

        Returns:
            Tool result as a string or list of content blocks.
        """
        session = self._sessions.get(server_name)
        if not session:
            raise ValueError(f"MCP server '{server_name}' not connected")

        try:
            result = await session.call_tool(tool_name, arguments)
            return self._format_result(result)
        except Exception as e:
            logger.error(f"MCP tool call failed ({server_name}:{tool_name}): {e}")
            raise

    def _format_result(self, result: Any) -> str | list[dict]:
        """Format MCP CallToolResult for our tool system.

        MCP returns content blocks similar to Anthropic's format.
        We convert them to our standard format.
        """
        from mcp import types

        if not hasattr(result, "content"):
            return str(result)

        blocks = []
        text_parts = []

        for item in result.content:
            if isinstance(item, types.TextContent):
                text_parts.append(item.text)
            elif isinstance(item, types.ImageContent):
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.mimeType,
                            "data": item.data,
                        },
                    }
                )
            elif isinstance(item, types.EmbeddedResource):
                # Handle embedded resources as text
                resource = item.resource
                if hasattr(resource, "text"):
                    text_parts.append(f"[Resource: {resource.uri}]\n{resource.text}")
                else:
                    text_parts.append(f"[Resource: {resource.uri}] (binary data)")

        # If we only have text, return as simple string
        if text_parts and not blocks:
            return "\n".join(text_parts)

        # If we have images or mixed content, return as content blocks
        if text_parts:
            blocks.insert(0, {"type": "text", "text": "\n".join(text_parts)})

        return blocks if blocks else "Tool executed successfully with no output."

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers and clean up resources.

        Note: The MCP SDK's stdio_client uses anyio task groups which can raise
        RuntimeError during cleanup when called from a different async context.
        These errors are harmless and suppressed.
        """
        import asyncio

        for name in list(self._sessions.keys()):
            try:
                session = self._sessions.pop(name, None)
                if session:
                    try:
                        await session.__aexit__(None, None, None)
                    except (RuntimeError, asyncio.CancelledError):
                        pass  # Suppress anyio cancel scope errors

                process_ctx = self._processes.pop(name, None)
                if process_ctx:
                    try:
                        await process_ctx.__aexit__(None, None, None)
                    except (RuntimeError, asyncio.CancelledError):
                        pass  # Suppress anyio cancel scope errors

                logger.debug(f"Disconnected from MCP server '{name}'")
            except Exception as e:
                # Only log unexpected errors, not cancel scope issues
                if "cancel scope" not in str(e).lower():
                    logger.warning(f"Error disconnecting from MCP server '{name}': {e}")

        self._tools.clear()
        self._connected = False
        logger.info("MCP client disconnected from all servers")

    async def __aenter__(self) -> "MCPClientManager":
        """Async context manager entry."""
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect_all()
