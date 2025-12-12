"""Main application entry point for matrix-llmagent."""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from .agentic_actor import AgenticLLMActor
from .agentic_actor.tools import (
    entity_info_tool_def,
    knowledge_base_tool_def,
    relationship_search_tool_def,
)
from .chronicler.chronicle import Chronicle
from .chronicler.quests import QuestOperator
from .history import ChatHistory
from .matrix_monitor import MatrixMonitor
from .providers import ModelRouter

# Set up logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Console handler for INFO and above
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler for DEBUG and above
file_handler = logging.FileHandler("debug.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Suppress noisy third-party library messages
logging.getLogger("aiosqlite").setLevel(logging.INFO)
logging.getLogger("e2b.api").setLevel(logging.WARNING)
logging.getLogger("e2b.sandbox_sync").setLevel(logging.WARNING)
logging.getLogger("e2b.sandbox_sync.main").setLevel(logging.WARNING)
logging.getLogger("e2b_code_interpreter.code_interpreter_sync").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MatrixLLMAgent:
    """Main Matrix LLM agent application."""

    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.model_router: ModelRouter = ModelRouter(self.config)

        # TODO: Update for Matrix config structure
        # For now, use legacy config structure for compatibility
        # Get Matrix config
        room_config = self.config.get("matrix", {}).get("command", {})
        history_size = room_config.get("history_size", 30)

        self.history = ChatHistory(
            self.config.get("history", {}).get("database", {}).get("path", "chat_history.db"),
            history_size,
        )
        # Initialize chronicle
        chronicler_config = self.config.get("chronicler", {})
        chronicle_db_path = chronicler_config.get("database", {}).get("path", "chronicle.db")
        self.chronicle = Chronicle(chronicle_db_path)

        # Initialize Matrix monitor
        self.matrix_monitor = MatrixMonitor(self)

        self.quests = QuestOperator(self)

        # Build additional tools from config
        self.additional_tools: list[dict[str, Any]] = []
        kb_config = self.config.get("tools", {}).get("knowledge_base", {})
        if kb_config.get("enabled") and kb_config.get("database_url"):
            kb_name = kb_config.get("name", "Knowledge Base")
            kb_description = kb_config.get(
                "description",
                f"Search {kb_name} for information about projects, people, organizations, and events.",
            )
            predicate_hints = kb_config.get("predicate_hints", {})
            self.additional_tools.append(dict(knowledge_base_tool_def(kb_name, kb_description)))
            self.additional_tools.append(
                dict(relationship_search_tool_def(kb_name, predicate_hints))
            )
            self.additional_tools.append(dict(entity_info_tool_def(kb_name)))
            logger.info(f"Knowledge base tool enabled: {kb_name}")

    async def run_actor(
        self,
        context: list[dict[str, str]],
        *,
        mode_cfg: dict[str, Any],
        system_prompt: str,
        arc: str = "",
        progress_callback=None,
        model: str | list[str] | None = None,
        **actor_kwargs,
    ) -> str | None:
        prepended_context: list[dict[str, str]] = []
        if mode_cfg.get("include_chapter_summary", True) and arc:
            prepended_context = await self.chronicle.get_chapter_context_messages(arc)

        # Append knowledge base guidance if enabled
        kb_config = self.config.get("tools", {}).get("knowledge_base", {})
        if kb_config.get("enabled"):
            kb_name = kb_config.get("name", "Knowledge Base")
            kb_description = kb_config.get("description", "")
            system_prompt = system_prompt + (
                f"\n\nIMPORTANT: You have access to a local knowledge base ({kb_name}) with two tools:\n"
                f"1. knowledge_base: Full-text search for pages, entities, and general information\n"
                f"2. relationship_search: Query structured relationships (leadership, projects, chapters, events)\n\n"
                f"ALWAYS try the local knowledge base tools FIRST before using web_search. "
                f"{kb_description}\n\n"
                "For questions about 'who is X', 'what are the projects/chapters/events', use relationship_search "
                "with the appropriate predicate. For general information queries, use knowledge_base.\n\n"
                "When searching, use the EXACT terms from the user's question - do not assume typos or 'correct' "
                "unfamiliar terms. Only use web_search if the local knowledge base doesn't have the information."
            )

        actor = AgenticLLMActor(
            config=self.config,
            model=model or mode_cfg["model"],
            system_prompt_generator=lambda: system_prompt,
            prompt_reminder_generator=lambda: mode_cfg.get("prompt_reminder"),
            prepended_context=prepended_context,
            agent=self,
            vision_model=mode_cfg.get("vision_model"),
            allowed_tools=mode_cfg.get("allowed_tools"),
            additional_tools=self.additional_tools,
            **actor_kwargs,
        )
        response = await actor.run_agent(
            context,
            progress_callback=progress_callback,
            arc=arc,
        )

        if not response or response.strip().upper() == "NULL":
            return None
        cleaned = response.strip()
        # Strip leading prefixes from context-echoed outputs: timestamps and non-quest tags like <nick>.
        # Never strip <quest> or <quest_finished> because those carry semantics for the chronicler.
        cleaned = re.sub(
            r"^(?:\s*(?:\[?\d{1,2}:\d{2}\]?\s*)?(?:<(?!/?quest(?:_finished)?\b)[^>]+>))*\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned

    def load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path) as f:
                config = json.load(f)
                logger.debug(f"Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            logger.error(
                f"Config file {config_path} not found. "
                "Copy config.json.example to config.json and configure."
            )
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            sys.exit(1)

    async def run(self) -> None:
        """Run the main agent loop by delegating to Matrix monitor."""
        # Initialize shared resources
        await self.history.initialize()
        await self.chronicle.initialize()
        # Scan and resume any open quests for whitelisted arcs
        await self.quests.scan_and_trigger_open_quests()

        try:
            await self.matrix_monitor.run()
        finally:
            # Clean up shared resources
            await self.history.close()
            # Chronicle doesn't need explicit cleanup


def _print_cli_help(config: dict[str, Any]) -> None:
    """Print CLI help message."""
    tools_config = config.get("tools", {})
    kb_config = tools_config.get("knowledge_base", {})
    kb_name = kb_config.get("name", "Knowledge Base") if kb_config.get("enabled") else None

    print("""
Available Commands
==================

Modes:
  !s <message>  - Serious mode (default) - thoughtful responses with web tools
  !d <message>  - Sarcastic mode - witty, humorous responses
  !a <message>  - Agent mode - multi-turn research with tool chaining
  !p <message>  - Perplexity mode - web-enhanced AI responses
  !u <message>  - Unsafe mode - uncensored responses
  !v <message>  - Verbose mode - get detailed responses instead of concise ones
  !h            - Show this help message

Tools Available:
  - Web search and webpage visiting
  - Code execution (if configured)
  - Image generation (if configured)""")

    if kb_name:
        print(f"  - {kb_name} search")

    print("""
Examples:
  uv run matrix-llmagent --message "what is Python?"
  uv run matrix-llmagent --message "!v explain machine learning"
  uv run matrix-llmagent --message "!d tell me a programming joke"
  uv run matrix-llmagent --message "!a research recent AI developments"

Tips:
  - Responses are concise by default (1 sentence) - say "tell me more" for details
  - Use !v prefix when you need a comprehensive answer upfront
  - Use !a for complex research that needs multiple steps
  - Use !d when you want fun, sarcastic responses
""")


async def cli_message(message: str, config_path: str | None = None) -> None:
    """CLI mode for testing message handling including command parsing."""
    # Load configuration
    config_file = Path(config_path) if config_path else Path(__file__).parent.parent / "config.json"

    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        print("Please create config.json from config.json.example")
        sys.exit(1)

    try:
        agent = MatrixLLMAgent(str(config_file))
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Handle help command
        if message.lower().strip() in ("!h", "!help", "help"):
            _print_cli_help(agent.config)
            return

        # Parse mode and verbose flag from message prefix
        mode = "serious"  # default
        verbose = False
        clean_message = message

        # Check for verbose modifier first
        if message.startswith("!v ") or message.startswith("!V "):
            verbose = True
            clean_message = message[3:]

        # Check for mode prefix (after stripping verbose if present)
        if clean_message.startswith("!s ") or clean_message.startswith("!S "):
            mode = "serious"
            clean_message = clean_message[3:]
        elif clean_message.startswith("!d ") or clean_message.startswith("!D "):
            mode = "sarcastic"
            clean_message = clean_message[3:]
        elif clean_message.startswith("!u ") or clean_message.startswith("!U "):
            mode = "unsafe"
            clean_message = clean_message[3:]
        elif clean_message.startswith("!a ") or clean_message.startswith("!A "):
            mode = "agent"
            clean_message = clean_message[3:]
        elif clean_message.startswith("!p ") or clean_message.startswith("!P "):
            mode = "perplexity"
            clean_message = clean_message[3:]

        # Get mode configuration
        command_config = agent.config.get("matrix", {}).get("command", {})
        mode_cfg = command_config.get("modes", {}).get(mode, {})

        if not mode_cfg:
            print(f"Error: Mode '{mode}' not configured")
            sys.exit(1)

        if not mode_cfg.get("model"):
            print(f"Error: No model configured for mode '{mode}'")
            sys.exit(1)

        print(f"Mode: {mode}")
        print(f"Model: {mode_cfg.get('model')}")
        print(f"Query: {clean_message}")
        print("-" * 60)

        # Build context (just the current message for CLI)
        context = [{"role": "user", "content": clean_message}]

        # Build system prompt
        system_prompt = mode_cfg.get("system_prompt", "You are a helpful assistant.")
        system_prompt = system_prompt.replace("{mynick}", "CLI-Bot")

        # Apply verbosity modifier to system prompt
        if verbose:
            system_prompt += " Provide comprehensive, detailed responses."
        else:
            system_prompt += (
                " Keep responses to 1 sentence max. Users can say 'tell me more' for details."
            )

        # Run actor
        response = await agent.run_actor(
            context,
            mode_cfg=mode_cfg,
            system_prompt=system_prompt,
            arc="cli#test",
        )

        print("-" * 60)
        if response:
            print(response)
        else:
            print("(No response)")

    except Exception as e:
        logger.error(f"CLI error: {e}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        await agent.history.close()


async def cli_chronicler(arc: str, instructions: str, config_path: str | None = None) -> None:
    """CLI mode for Chronicler operations."""
    # Load configuration
    config_file = Path(config_path) if config_path else Path(__file__).parent.parent / "config.json"

    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        print("Please create config.json from config.json.example")
        sys.exit(1)

    print(f"🔮 Chronicler arc '{arc}': {instructions}")
    print("=" * 60)

    try:
        # Create agent instance
        agent = MatrixLLMAgent(str(config_file))
        await agent.chronicle.initialize()

        print(
            "Error: Chronicler subagent functionality has been removed. Use direct chronicle_append and chronicle_read tools instead."
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="matrix-llmagent - Matrix chatbot with AI and tools"
    )
    parser.add_argument(
        "--message", type=str, help="Run in CLI mode to simulate handling a message"
    )
    parser.add_argument(
        "--config", type=str, help="Path to config file (default: config.json in project root)"
    )
    parser.add_argument(
        "--chronicler",
        type=str,
        help="Run Chronicler subagent with instructions (NLI over Chronicle)",
    )
    parser.add_argument(
        "--arc", type=str, help="Arc name for Chronicler (required with --chronicler)"
    )

    args = parser.parse_args()

    if args.chronicler:
        if not args.arc:
            print("Error: --arc is required with --chronicler")
            sys.exit(1)
        asyncio.run(cli_chronicler(args.arc, args.chronicler, args.config))
        return

    if args.message:
        asyncio.run(cli_message(args.message, args.config))
    else:
        agent = MatrixLLMAgent()
        asyncio.run(agent.run())


if __name__ == "__main__":
    main()
