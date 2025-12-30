"""MCP (Model Context Protocol) client integration.

This module provides optional MCP client support for connecting to external
MCP servers and exposing their tools to the agent. MCP servers are only
connected when explicitly configured in config.json.

Example configuration:
    {
        "tools": {
            "mcp": {
                "servers": {
                    "filesystem": {
                        "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                        "transport": "stdio"
                    },
                    "my-http-server": {
                        "url": "http://localhost:8000/mcp",
                        "transport": "streamable-http"
                    }
                }
            }
        }
    }
"""

from matrix_llmagent.mcp_client.client import MCPClientManager, MCPToolExecutor

__all__ = ["MCPClientManager", "MCPToolExecutor"]
