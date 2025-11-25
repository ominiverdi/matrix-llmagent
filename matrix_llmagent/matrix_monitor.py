"""Matrix room monitor for handling messages and events."""

import asyncio
import logging
import re
from typing import Any

from nio import MatrixRoom, RoomMessageText

from .matrix_client import MatrixClient
from .rate_limiter import RateLimiter
from .rooms import ProactiveDebouncer

logger = logging.getLogger(__name__)


class MatrixMonitor:
    """Monitor Matrix rooms and handle messages."""

    def __init__(self, agent):
        """Initialize Matrix monitor.

        Args:
            agent: MatrixLLMAgent instance
        """
        self.agent = agent
        self.config = agent.config
        self.matrix_config = self.config.get("matrix", {})

        # Initialize Matrix client
        self.client = MatrixClient(self.config)

        # Get command configuration
        self.command_config = self.matrix_config.get("command", {})
        self.rate_limiter = RateLimiter(
            self.command_config.get("rate_limit", 30),
            self.command_config.get("rate_period", 900),
        )

        # Proactive interjecting configuration
        proactive_config = self.matrix_config.get("proactive", {})
        if proactive_config:
            self.proactive = ProactiveDebouncer(proactive_config.get("debounce_seconds", 15.0))
        else:
            self.proactive = None

        # Get our bot's user ID for mention detection
        self.bot_user_id = self.matrix_config.get("user_id", "")

        logger.info("Matrix monitor initialized")

    async def run(self) -> None:
        """Main event loop for Matrix monitor."""
        # Connect to Matrix
        await self.client.connect()

        # Set up event callbacks
        self.client.add_event_callback(self.on_room_message, RoomMessageText)

        logger.info("Matrix monitor starting sync loop")

        try:
            # Continuous sync loop
            while True:
                await self.client.sync(timeout=30000)
        except Exception as e:
            logger.error(f"Matrix monitor error: {e}")
            raise
        finally:
            await self.client.close()

    async def on_room_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        """Handle incoming room messages.

        Args:
            room: Matrix room where message was sent
            event: Room message event
        """
        # Ignore our own messages
        if event.sender == self.bot_user_id:
            return

        # Get message details
        room_id = room.room_id
        sender = event.sender
        message = event.body

        logger.debug(f"Received message in {room_id} from {sender}: {message}")

        # Check if message is addressed to us
        if self.is_addressed_to_bot(message, sender, room_id):
            await self.handle_command(room_id, sender, message)
        elif self.proactive:
            # Check for proactive interjecting
            await self.handle_proactive(room_id, sender, message)

    def is_addressed_to_bot(self, message: str, sender: str, room_id: str) -> bool:
        """Check if message is addressed to the bot.

        Args:
            message: Message text
            sender: Sender user ID
            room_id: Room ID

        Returns:
            True if message is addressed to bot
        """
        logger.debug(f"Checking if message is addressed to bot. Message: '{message}'")
        logger.debug(f"Bot user ID: {self.bot_user_id}")

        # Check for Matrix mention (e.g., "@llm-assitant:matrix.org")
        if self.bot_user_id in message:
            logger.debug("Bot user ID found in message!")
            return True

        # Check for display name mention (e.g., "llm-assistant")
        if "llm-assistant" in message.lower() or "llm-assitant" in message.lower():
            logger.debug("Bot display name found in message!")
            return True

        logger.debug("Message not addressed to bot")
        return False

    async def handle_command(self, room_id: str, sender: str, message: str) -> None:
        """Handle a command addressed to the bot.

        Args:
            room_id: Matrix room ID
            sender: Sender user ID
            message: Message text
        """
        # Rate limiting
        if not self.rate_limiter.check_limit():
            logger.info(f"Rate limit exceeded for {sender}")
            await self.client.send_message(
                room_id,
                "⏱️ Rate limit exceeded. Please wait before sending more messages.",
            )
            return

        # Remove bot mention from message
        clean_message = re.sub(rf"{re.escape(self.bot_user_id)}:?\s*", "", message).strip()

        # Parse command mode
        mode = self.determine_mode(clean_message)

        logger.info(f"Processing command from {sender} in {room_id}, mode: {mode}")

        # Get mode configuration
        mode_cfg = self.command_config.get("modes", {}).get(mode, {})
        if not mode_cfg:
            await self.client.send_message(room_id, "❌ Unknown command mode")
            return

        # Build context from chat history
        history_size = mode_cfg.get("history_size", self.command_config.get("history_size", 30))
        context = await self.agent.history.get_context("matrix", room_id, history_size)

        # Add current message to context
        context.append({"role": "user", "content": clean_message})

        # Build system prompt
        system_prompt = mode_cfg.get("system_prompt", "You are a helpful assistant.")

        # Get bot display name for system prompt
        bot_name = await self.client.get_display_name()
        if "{mynick}" in system_prompt:
            system_prompt = system_prompt.replace("{mynick}", bot_name)

        # Run actor
        try:
            response = await self.agent.run_actor(
                context,
                mode_cfg=mode_cfg,
                system_prompt=system_prompt,
                arc=f"matrix#{room_id}",
            )

            if response:
                await self.client.send_message(room_id, response)

                # Save to history
                await self.agent.history.add_message(
                    "matrix", room_id, clean_message, sender, sender
                )
                await self.agent.history.add_message(
                    "matrix", room_id, response, bot_name, bot_name, True
                )
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            await self.client.send_message(room_id, f"❌ Error: {str(e)}")

    def determine_mode(self, message: str) -> str:
        """Determine which mode to use based on message.

        Args:
            message: Message text

        Returns:
            Mode name (e.g., 'serious', 'sarcastic')
        """
        # Check for explicit mode commands
        if message.startswith("!s ") or message.startswith("!S "):
            return "serious"
        elif message.startswith("!d ") or message.startswith("!D "):
            return "sarcastic"
        elif message.startswith("!u ") or message.startswith("!U "):
            return "unsafe"
        elif message.startswith("!a ") or message.startswith("!A "):
            return "agent"
        elif message.startswith("!p ") or message.startswith("!P "):
            return "perplexity"

        # Use default or classifier
        return self.command_config.get("default_mode", "serious")

    async def handle_proactive(self, room_id: str, sender: str, message: str) -> None:
        """Handle proactive interjecting.

        Args:
            room_id: Matrix room ID
            sender: Sender user ID
            message: Message text
        """
        # TODO: Implement proactive interjecting
        # This would check if the conversation is relevant and interject
        pass

    async def send_message(self, room_id: str, message: str) -> None:
        """Send a message to a Matrix room.

        Args:
            room_id: Matrix room ID
            message: Message text
        """
        await self.client.send_message(room_id, message)

    async def get_mynick(self, server: str = "matrix") -> str:
        """Get bot's display name.

        Args:
            server: Server name (ignored for Matrix, kept for compatibility)

        Returns:
            Bot's display name
        """
        return await self.client.get_display_name()

    def build_system_prompt(self, mode: str, bot_name: str) -> str:
        """Build system prompt for a given mode.

        Args:
            mode: Mode name
            bot_name: Bot's display name

        Returns:
            System prompt string
        """
        mode_cfg = self.command_config.get("modes", {}).get(mode, {})
        system_prompt = mode_cfg.get("system_prompt", "You are a helpful assistant.")

        if "{mynick}" in system_prompt:
            system_prompt = system_prompt.replace("{mynick}", bot_name)

        return system_prompt
