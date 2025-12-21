"""Matrix room monitor for handling messages and events."""

import html
import logging
import re

from nio import MatrixRoom, RoomMessageText

from .agentic_actor.library_tool import (
    LibraryResultsCache,
    fetch_library_image,
    fetch_library_page,
    get_best_image_path,
    get_citation_tag,
    search_library_direct,
)
from .matrix_client import MatrixClient
from .rate_limiter import RateLimiter
from .rooms import ProactiveDebouncer

logger = logging.getLogger(__name__)

# Default threshold for collapsible messages (in characters)
DEFAULT_LONG_MESSAGE_THRESHOLD = 300
DEFAULT_SUMMARY_WORDS = 30
# Threshold for paste service (when available) - messages longer than this get uploaded
DEFAULT_PASTE_THRESHOLD = 2000


# TODO: Implement paste service support
# async def _upload_to_paste_service(text: str, paste_url: str) -> str | None:
#     """Upload long text to paste service and return the URL.
#
#     Args:
#         text: Text content to upload
#         paste_url: Base URL of paste service (e.g., "https://paste.example.com")
#
#     Returns:
#         URL to the paste, or None if upload failed
#
#     Example integration in handle_command():
#         paste_config = self.config.get("behavior", {}).get("paste_service", {})
#         if paste_config.get("enabled") and len(response) > paste_config.get("threshold", 2000):
#             paste_url = await _upload_to_paste_service(response, paste_config["url"])
#             if paste_url:
#                 summary = _truncate_to_words(response, 50)
#                 response = f"{summary}\n\nFull response: {paste_url}"
#
#     Config example:
#         "behavior": {
#             "paste_service": {
#                 "enabled": true,
#                 "url": "https://paste.example.com",
#                 "threshold": 2000
#             }
#         }
#     """
#     import aiohttp
#     try:
#         async with aiohttp.ClientSession() as session:
#             # Assuming 0x0.st-compatible API
#             data = aiohttp.FormData()
#             data.add_field('file', text, filename='response.txt', content_type='text/plain')
#             async with session.post(paste_url, data=data) as resp:
#                 if resp.status == 200:
#                     return (await resp.text()).strip()
#     except Exception as e:
#         logger.warning(f"Failed to upload to paste service: {e}")
#     return None


def _parse_show_command(message: str) -> list[int] | None:
    """Parse 'show 1,2,3' or 'show 1 2 3' command.

    Args:
        message: Message text

    Returns:
        List of 1-indexed result numbers, or None if not a show command.
    """
    # Match "show" followed by numbers (comma or space separated)
    match = re.match(r"^show\s+([\d,\s]+)$", message.strip(), re.IGNORECASE)
    if not match:
        return None

    # Extract all numbers from the argument
    indices = [int(x) for x in re.findall(r"\d+", match.group(1))]
    return indices if indices else None


def _parse_page_command(message: str) -> tuple[str, int | None] | None:
    """Parse page navigation commands.

    Supports:
        !next, !prev - navigate relative to current page
        !page 50, !page50 - jump to specific page

    Returns:
        Tuple of (command, page_number) where command is 'next', 'prev', or 'goto'.
        page_number is None for next/prev, or the target page for goto.
        Returns None if not a page command.
    """
    msg = message.strip().lower()

    if msg in ("!next", "!n"):
        return ("next", None)
    if msg in ("!prev", "!p", "!previous"):
        return ("prev", None)

    # !page N or !pageN
    match = re.match(r"^!page\s*(\d+)$", msg)
    if match:
        return ("goto", int(match.group(1)))

    return None


def _truncate_to_words(text: str, max_words: int = DEFAULT_SUMMARY_WORDS) -> str:
    """Truncate text to a maximum number of words.

    Args:
        text: Text to truncate
        max_words: Maximum number of words to keep

    Returns:
        Truncated text with "..." if truncated
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _wrap_long_message(
    text: str, threshold: int = DEFAULT_LONG_MESSAGE_THRESHOLD
) -> tuple[str, str | None]:
    """Wrap long messages in a collapsible <details> tag for Matrix.

    Args:
        text: Message text
        threshold: Character threshold for collapsible wrapping

    Returns:
        Tuple of (plain_text, html_or_none). If html is None, send as plain text.
    """
    if len(text) <= threshold:
        return text, None

    summary = _truncate_to_words(text)
    # Escape HTML in both summary and full content
    safe_summary = html.escape(summary)
    safe_text = html.escape(text)

    # Format with <details> tag - use <pre> to preserve formatting
    html_body = f"<details><summary>{safe_summary}</summary>\n<pre>{safe_text}</pre></details>"

    return text, html_body


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

        # Do initial sync WITHOUT event callbacks to skip historical messages
        logger.info("Matrix monitor performing initial sync (skipping historical messages)...")
        await self.client.sync(timeout=10000)
        logger.info("Initial sync complete, now listening for new messages")

        # Set up event callbacks AFTER initial sync
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

        # Check for show command first (works without addressing the bot)
        show_indices = _parse_show_command(message)
        if show_indices is not None:
            await self._handle_show_command(room_id, show_indices)
            return

        # Check for page navigation commands (works without addressing the bot)
        page_cmd = _parse_page_command(message)
        if page_cmd is not None:
            cmd, page_num = page_cmd
            await self._handle_page_command(room_id, cmd, page_num)
            return

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

        # Check for display name at the START of message (e.g., "llm-assistant: question")
        # This avoids false positives when the bot name is mentioned mid-sentence
        msg_lower = message.lower().strip()
        for name in ["llm-assistant", "llm-assitant"]:
            # Match "name:" or "name," or "name " at start of message
            if (
                msg_lower.startswith(name + ":")
                or msg_lower.startswith(name + ",")
                or msg_lower.startswith(name + " ")
            ):
                logger.debug(f"Bot display name '{name}' found at start of message!")
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

        # Remove bot mention from message (both user ID and display name)
        clean_message = re.sub(rf"{re.escape(self.bot_user_id)}:?\s*", "", message).strip()
        clean_message = re.sub(
            r"llm-assistant:?\s*", "", clean_message, flags=re.IGNORECASE
        ).strip()
        clean_message = re.sub(r"llm-assitant:?\s*", "", clean_message, flags=re.IGNORECASE).strip()

        # Handle help command
        if clean_message.lower().strip() in ("!h", "!help", "help"):
            await self._send_help(room_id)
            return

        # Handle !l library search (direct, no LLM)
        if clean_message.startswith("!l ") or clean_message.startswith("!L "):
            query = clean_message[3:].strip()
            if query:
                await self._handle_library_search(room_id, query)
            return

        # Parse command mode
        mode, verbose = self.determine_mode(clean_message)

        # Strip mode prefix from message after determining mode
        mode_prefixes = [
            r"!v\s+",
            r"!V\s+",
            r"!s\s+",
            r"!S\s+",
            r"!d\s+",
            r"!D\s+",
            r"!u\s+",
            r"!U\s+",
            r"!a\s+",
            r"!A\s+",
            r"!p\s+",
            r"!P\s+",
            r"!2\s+",
            r"!3\s+",
            r"!4\s+",
            r"!5\s+",
            r"!6\s+",
            r"!7\s+",
        ]
        for prefix in mode_prefixes:
            clean_message = re.sub(f"^{prefix}", "", clean_message)

        logger.info(f"Processing command from {sender} in {room_id}, mode: {mode}")

        # Get mode configuration with inheritance for numbered slots
        modes = self.command_config.get("modes", {})
        mode_cfg = modes.get(mode, {})

        # Inherit from base mode if this is a numbered slot (e.g., serious2 inherits from serious)
        if mode.startswith("serious") and mode != "serious":
            base_cfg = modes.get("serious", {})
            # Merge: base config first, then mode-specific overrides
            mode_cfg = {**base_cfg, **mode_cfg}

        if not mode_cfg:
            await self.client.send_message(room_id, "Unknown command mode")
            return
        if not mode_cfg.get("model"):
            await self.client.send_message(room_id, f"Mode '{mode}' is not configured")
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

        # Apply verbosity modifier to system prompt
        if verbose:
            system_prompt += " Provide comprehensive, detailed responses."
        else:
            system_prompt += (
                " Keep responses to 1 sentence max. Users can say 'tell me more' for details."
            )

        # Image callback for Matrix - upload to room
        async def matrix_image_callback(image_bytes: bytes, mimetype: str) -> None:
            await self.client.send_image(
                room_id,
                image_bytes,
                f"page.{mimetype.split('/')[-1]}",
                mimetype,
            )

        # Run actor
        try:
            response = await self.agent.run_actor(
                context,
                mode_cfg=mode_cfg,
                system_prompt=system_prompt,
                arc=f"matrix#{room_id}",
                image_callback=matrix_image_callback,
            )

            if response:
                # Add slot label prefix if configured (for model comparison)
                slot_label = mode_cfg.get("slot_label")
                if slot_label:
                    response = f"[{slot_label}] {response}"

                # Get collapsible message config
                behavior_config = self.config.get("behavior", {})
                collapsible_enabled = behavior_config.get("collapsible_messages", False)

                if collapsible_enabled:
                    threshold = behavior_config.get(
                        "max_message_length", DEFAULT_LONG_MESSAGE_THRESHOLD
                    )
                    # Wrap long messages in collapsible <details> tag
                    plain_text, html_body = _wrap_long_message(response, threshold)

                    if html_body:
                        await self.client.send_html_message(room_id, plain_text, html_body)
                    else:
                        await self.client.send_message(room_id, response)
                else:
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

    def determine_mode(self, message: str) -> tuple[str, bool]:
        """Determine which mode to use based on message.

        Args:
            message: Message text

        Returns:
            Tuple of (mode_name, verbose_flag)
        """
        # Check for verbose modifier first
        verbose = message.startswith("!v ") or message.startswith("!V ")

        # Strip verbose prefix if present for further mode detection
        check_msg = message
        if verbose:
            check_msg = message[3:]  # Remove "!v "

        # Check for explicit mode commands
        if check_msg.startswith("!s ") or check_msg.startswith("!S "):
            return "serious", verbose
        elif check_msg.startswith("!d ") or check_msg.startswith("!D "):
            return "sarcastic", verbose
        elif check_msg.startswith("!u ") or check_msg.startswith("!U "):
            return "unsafe", verbose
        elif check_msg.startswith("!a ") or check_msg.startswith("!A "):
            return "agent", verbose
        elif check_msg.startswith("!p ") or check_msg.startswith("!P "):
            return "perplexity", verbose

        # Check for numbered model slots (!2, !3, !4, etc.)
        elif check_msg.startswith("!2 "):
            return "serious2", verbose
        elif check_msg.startswith("!3 "):
            return "serious3", verbose
        elif check_msg.startswith("!4 "):
            return "serious4", verbose
        elif check_msg.startswith("!5 "):
            return "serious5", verbose
        elif check_msg.startswith("!6 "):
            return "serious6", verbose
        elif check_msg.startswith("!7 "):
            return "serious7", verbose

        # Use default or classifier
        return self.command_config.get("default_mode", "serious"), verbose

    async def _send_help(self, room_id: str) -> None:
        """Send help message with available commands."""
        # Check which tools are available
        tools_config = self.agent.config.get("tools", {})
        kb_config = tools_config.get("knowledge_base", {})
        kb_name = kb_config.get("name", "Knowledge Base") if kb_config.get("enabled") else None

        # Build model slots section dynamically from config
        modes = self.command_config.get("modes", {})
        model_slots_text = ""
        for slot_num in ["2", "3", "4", "5", "6", "7"]:
            mode_key = f"serious{slot_num}"
            mode_cfg = modes.get(mode_key, {})
            if mode_cfg:
                slot_label = mode_cfg.get("slot_label", mode_cfg.get("model", "unknown"))
                model_slots_text += f"\n- `!{slot_num} <message>` - {slot_label}"

        if not model_slots_text:
            model_slots_text = "\n- No model slots configured"

        # Get default model info
        default_mode = self.command_config.get("default_mode", "serious")
        default_cfg = modes.get(default_mode, {})
        default_model = default_cfg.get("model", "unknown")
        if isinstance(default_model, list):
            default_model = default_model[0] if default_model else "unknown"
        # Extract just the model name part after the provider prefix
        if ":" in str(default_model):
            default_model = default_model.split(":")[-1]

        help_text = f"""**Available Commands**

**Modes:**
- `!s <message>` - Serious mode (default: {default_model})
- `!d <message>` - Sarcastic mode - witty, humorous responses
- `!a <message>` - Agent mode - multi-turn research with tool chaining
- `!p <message>` - Perplexity mode - web-enhanced AI responses
- `!l <query>` - Library search - direct search without LLM
- `!u <message>` - Unsafe mode - uncensored responses
- `!v <message>` - Verbose mode - get detailed responses instead of concise ones
- `!h` - Show this help message

**Page Navigation (after viewing a document page):**
- `!next` - Next page
- `!prev` - Previous page
- `!page N` - Jump to page N

**Model Comparison Slots:**{model_slots_text}

**Tools Available:**
- Web search and webpage visiting
- Code execution (if configured)
- Image generation (if configured)"""

        if kb_name:
            help_text += f"\n- {kb_name} search"

        help_text += """

**Examples:**
```
llm-assistant: what is Python?
llm-assistant: !v explain machine learning
llm-assistant: !d tell me a programming joke
llm-assistant: !a research recent AI developments
llm-assistant: !l mercator projection
```

**Tips:**
- Responses are concise by default (1 sentence) - say "tell me more" for details
- Use `!v` prefix when you need a comprehensive answer upfront
- Use `!a` for complex research that needs multiple steps
- Use `!d` when you want fun, sarcastic responses
- Use `!l` for quick library search, then `show N` to view images
- Ask "show me page N of <document>" to browse document pages
- Use `!next`/`!prev`/`!page N` for quick page navigation"""

        await self.client.send_message(room_id, help_text)

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

    async def _handle_show_command(self, room_id: str, indices: list[int]) -> None:
        """Handle show command to display library search results.

        Args:
            room_id: Matrix room ID
            indices: List of 1-indexed result numbers to show
        """
        # Check if library is configured and cache is available
        lib_config = self.config.get("tools", {}).get("library", {})
        if not lib_config.get("enabled") or not lib_config.get("base_url"):
            await self.client.send_message(
                room_id,
                "Library search is not configured. Use library_search tool first.",
            )
            return

        # Get cached results from agent
        cache = getattr(self.agent, "library_cache", None)
        if cache is None:
            await self.client.send_message(
                room_id,
                "No library search results available. Try searching first.",
            )
            return

        # Use consistent arc format for cache key
        arc = f"matrix#{room_id}"
        results = cache.get(arc)
        if not results:
            await self.client.send_message(
                room_id,
                "No library search results available for this room. Try searching first.",
            )
            return

        base_url = lib_config["base_url"]

        for idx in indices:
            # Validate index
            if idx < 1 or idx > len(results):
                await self.client.send_message(
                    room_id,
                    f"Invalid index [{idx}]. Available: 1-{len(results)}",
                )
                continue

            result = results[idx - 1]  # Convert to 0-indexed
            tag = get_citation_tag(result)
            doc_title = result.get("document_title", "Unknown")
            page = result.get("page_number", "?")

            # Handle text chunks - show the text content
            if result.get("source_type") == "chunk":
                content = result.get("content", "")[:1000]  # Limit to 1000 chars
                label = f"[{tag}:{idx}] TEXT from {doc_title}, page {page}"
                await self.client.send_message(room_id, f"{label}\n\n{content}")
                continue

            # Handle elements - try to fetch and display image
            image_path = get_best_image_path(result)
            element_label = result.get("element_label", "Element")
            element_type = result.get("element_type", "element").upper()
            content = result.get("content", "")
            # Build caption with description if available
            caption = f"{element_type}: {element_label} from {doc_title}, page {page}"
            if content:
                desc = content[:300] + "..." if len(content) > 300 else content
                caption = f"{caption}\n\n{desc}"

            if not image_path:
                # No image available, show caption with description
                await self.client.send_message(
                    room_id,
                    f"{caption}\n\n(No image available)",
                )
                continue

            # Fetch image from library server
            try:
                image_result = await fetch_library_image(
                    base_url,
                    result.get("document_slug", ""),
                    image_path,
                )

                if image_result is None:
                    await self.client.send_message(
                        room_id,
                        f"Could not fetch image for {element_label} - library server may be unavailable.",
                    )
                    continue

                image_bytes, mimetype = image_result
                filename = image_path.split("/")[-1]

                # Send caption first, then image
                await self.client.send_message(room_id, caption)
                await self.client.send_image(
                    room_id,
                    image_bytes,
                    filename,
                    mimetype,
                )

            except Exception as e:
                logger.error(f"Error handling show command for [{tag}:{idx}]: {e}")
                await self.client.send_message(
                    room_id,
                    f"Error fetching image for [{tag}:{idx}]: {e}",
                )

    async def _handle_page_command(
        self, room_id: str, command: str, page_number: int | None
    ) -> None:
        """Handle page navigation commands (!next, !prev, !page N).

        Args:
            room_id: Matrix room ID
            command: 'next', 'prev', or 'goto'
            page_number: Target page for 'goto', None for relative navigation
        """
        lib_config = self.config.get("tools", {}).get("library", {})
        if not lib_config.get("enabled") or not lib_config.get("base_url"):
            await self.client.send_message(room_id, "Library is not configured.")
            return

        cache = getattr(self.agent, "library_cache", None)
        if cache is None:
            await self.client.send_message(
                room_id, "No document open. Use library_search with page_number first."
            )
            return

        # Get current page view
        arc = f"matrix#{room_id}"
        page_view = cache.get_page_view(arc)
        if page_view is None:
            await self.client.send_message(
                room_id, "No document open. Try: 'show me page 1 of <document name>'"
            )
            return

        # Calculate target page
        if command == "next":
            target_page = page_view.page + 1
        elif command == "prev":
            target_page = page_view.page - 1
        else:  # goto
            target_page = page_number or 1

        # Bounds check
        if target_page < 1:
            await self.client.send_message(
                room_id,
                f"Already at first page (page 1 of {page_view.total_pages}).",
            )
            return
        if target_page > page_view.total_pages:
            await self.client.send_message(
                room_id,
                f"Already at last page (page {page_view.total_pages} of {page_view.total_pages}).",
            )
            return

        # Fetch the page
        base_url = lib_config["base_url"]
        await self.client.send_message(
            room_id, f"Fetching page {target_page} of '{page_view.document_title}'..."
        )

        result = await fetch_library_page(base_url, page_view.document_slug, target_page)

        if isinstance(result, str):
            await self.client.send_message(room_id, f"Error: {result}")
            return

        # Update page view in cache
        cache.store_page_view(
            arc,
            result.document_slug,
            result.document_title,
            result.page_number,
            result.total_pages,
        )

        # Send caption first, then image
        caption = f"Page {result.page_number} of {result.total_pages}: {result.document_title}\n\n[!next/!prev to navigate, !page N to jump]"
        await self.client.send_message(room_id, caption)
        await self.client.send_image(
            room_id,
            result.image_data,
            f"page_{result.page_number}.png",
            result.image_mimetype,
        )

    async def _handle_library_search(self, room_id: str, query: str) -> None:
        """Handle !l library search command (direct, no LLM).

        Args:
            room_id: Matrix room ID
            query: Search query string
        """
        lib_config = self.config.get("tools", {}).get("library", {})
        if not lib_config.get("enabled") or not lib_config.get("base_url"):
            await self.client.send_message(
                room_id,
                "Library search is not configured.",
            )
            return

        lib_name = lib_config.get("name", "OSGeo Library")
        max_results = lib_config.get("max_results", 10)

        # Ensure cache exists on agent
        cache = getattr(self.agent, "library_cache", None)
        if cache is None:
            cache_config = lib_config.get("cache", {})
            cache = LibraryResultsCache(
                ttl_hours=cache_config.get("ttl_hours", 24),
                max_rooms=cache_config.get("max_rooms", 100),
            )
            self.agent.library_cache = cache

        # Use room_id as the arc for caching
        arc = f"matrix#{room_id}"

        logger.info(f"Direct library search in {room_id}: {query}")

        results, formatted = await search_library_direct(
            base_url=lib_config["base_url"],
            query=query,
            cache=cache,
            arc=arc,
            name=lib_name,
            limit=max_results,
        )

        await self.client.send_message(room_id, formatted)

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
