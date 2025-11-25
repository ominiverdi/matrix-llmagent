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
        try:
            # Try Matrix config first
            room_config = self.config.get("matrix", {}).get("command", {})
            history_size = room_config.get("history_size", 30)
        except (KeyError, TypeError):
            # Fallback to IRC config for backward compatibility during migration
            room_config = self.config.get("rooms", {}).get("irc", {}).get("command", {})
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

        actor = AgenticLLMActor(
            config=self.config,
            model=model or mode_cfg["model"],
            system_prompt_generator=lambda: system_prompt,
            prompt_reminder_generator=lambda: mode_cfg.get("prompt_reminder"),
            prepended_context=prepended_context,
            agent=self,
            vision_model=mode_cfg.get("vision_model"),
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
        # Strip IRC-style leading prefixes from context-echoed outputs: timestamps and non-quest tags like <nick>.
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


async def cli_message(message: str, config_path: str | None = None) -> None:
    """CLI mode for testing message handling including command parsing."""
    # Load configuration
    config_file = Path(config_path) if config_path else Path(__file__).parent.parent / "config.json"

    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        print("Please create config.json from config.json.example")
        sys.exit(1)

    print(f"🤖 Simulating message: {message}")
    print("=" * 60)
    print("⚠️  CLI message simulation not yet implemented for Matrix.")
    print("   See PLAN.md Phase 4 for Matrix monitor implementation status.")
    print("=" * 60)

    # TODO: Implement CLI message handling once Matrix monitor is ready
    # try:
    #     agent = MatrixLLMAgent(str(config_file))
    #     await agent.history.initialize()
    #     await agent.chronicle.initialize()
    #     # Simulate Matrix message handling
    # finally:
    #     await agent.history.close()


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
