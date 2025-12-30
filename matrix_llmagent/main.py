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
from .agentic_actor.library_tool import (
    LibraryResultsCache,
    fetch_library_image,
    fetch_library_page,
    format_document_info,
    format_document_list,
    format_element_list,
    format_sources_list,
    get_best_image_path,
    get_document_info,
    get_library_context_summary,
    library_search_tool_def,
    list_documents,
    list_elements,
    search_library_direct,
)
from .agentic_actor.tools import (
    KBCachedResults,
    KnowledgeBaseResultsCache,
    WebSearchCachedResults,
    WebSearchResultsCache,
    entity_info_tool_def,
    format_kb_source_detail,
    format_kb_sources_list,
    format_web_source_detail,
    format_web_sources_list,
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

        # Add kb_cache for knowledge base sources
        self.kb_cache: KnowledgeBaseResultsCache | None = None
        if kb_config.get("enabled") and kb_config.get("database_url"):
            cache_config = kb_config.get("cache", {})
            self.kb_cache = KnowledgeBaseResultsCache(
                ttl_hours=cache_config.get("ttl_hours", 24),
                max_rooms=cache_config.get("max_rooms", 100),
            )

        # Add library_search tool if configured
        lib_config = self.config.get("tools", {}).get("library", {})
        self.library_cache: LibraryResultsCache | None = None
        if lib_config.get("enabled") and lib_config.get("base_url"):
            lib_name = lib_config.get("name", "OSGeo Library")
            lib_description = lib_config.get(
                "description",
                f"Search {lib_name} for scientific documents, figures, tables, and equations.",
            )
            self.additional_tools.append(dict(library_search_tool_def(lib_name, lib_description)))
            # Create shared cache for library results (persists across actor runs)
            cache_config = lib_config.get("cache", {})
            self.library_cache = LibraryResultsCache(
                ttl_hours=cache_config.get("ttl_hours", 24),
                max_rooms=cache_config.get("max_rooms", 100),
            )
            logger.info(f"Library search tool enabled: {lib_name}")

        # Add web_search_cache for web search sources (always created if search is enabled)
        tools_config = self.config.get("tools", {})
        search_provider = tools_config.get("search_provider")
        self.web_search_cache: WebSearchResultsCache | None = None
        if search_provider and search_provider.lower() not in ("none", "disabled", "off", "false"):
            self.web_search_cache = WebSearchResultsCache(
                ttl_hours=24,
                max_rooms=100,
            )

        # Initialize MCP client manager if configured
        from .mcp_client import MCPClientManager

        self.mcp_manager: MCPClientManager | None = None
        mcp_config = self.config.get("tools", {}).get("mcp", {})
        if mcp_config.get("servers"):
            self.mcp_manager = MCPClientManager(self.config)
            logger.info("MCP client manager initialized (will connect on startup)")

    async def run_actor(
        self,
        context: list[dict[str, str]],
        *,
        mode_cfg: dict[str, Any],
        system_prompt: str,
        arc: str = "",
        progress_callback=None,
        image_callback=None,
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
                f"\n\nYou have access to a local knowledge base ({kb_name}) with these tools:\n"
                f"1. knowledge_base: Full-text search for pages, entities, and general information\n"
                f"2. relationship_search: Query structured relationships (leadership, projects, chapters, events)\n\n"
                f"{kb_description}\n\n"
                "TOOL SELECTION RULES:\n"
                "- If user explicitly asks to 'search the web/internet', use web_search\n"
                "- If user asks about 'the wiki' or mentions the knowledge base by name, use knowledge_base\n"
                "- For general questions about OSGeo projects/people/events, prefer knowledge_base first\n"
                "- For current news, recent releases, or real-time information, use web_search\n\n"
                "For questions about 'who is X', 'what are the projects/chapters/events', use relationship_search "
                "with the appropriate predicate. For general information queries, use knowledge_base.\n\n"
                "When searching, use the EXACT terms from the user's question - do not assume typos or 'correct' "
                "unfamiliar terms."
            )

        # Append library search guidance if enabled
        lib_config = self.config.get("tools", {}).get("library", {})
        if lib_config.get("enabled"):
            lib_name = lib_config.get("name", "OSGeo Library")
            lib_description = lib_config.get("description", "")
            system_prompt = system_prompt + (
                f"\n\nYou also have access to {lib_name} via the library_search tool. "
                f"{lib_description}\n"
                "Use library_search for questions about scientific documents, map projections, equations, figures, and tables. "
                "Results include citation tags like [f:1] for figures, [t:2] for text, [eq:3] for equations, [tb:4] for tables. "
                "When elements are available, suggest the user can type `show N` to view images.\n"
                "To show a specific PAGE from a document, use library_search with the page_number parameter. "
                "Example: library_search(query='snyder', page_number=28) fetches page 28 of the Snyder document.\n\n"
                "IMPORTANT: When answering vague questions about the library (e.g., 'what's in the library?'), "
                "ALWAYS end your response by mentioning: Type `!l help` for a guided tour."
            )

        # Create system prompt generator that includes library context
        base_prompt = system_prompt

        def generate_system_prompt() -> str:
            prompt = base_prompt
            # Add library browsing context if available
            if self.library_cache is not None:
                doc_view = self.library_cache.get_document_view(arc)
                page_view = self.library_cache.get_page_view(arc)
                context_str = get_library_context_summary(doc_view, page_view)
                if context_str:
                    prompt = prompt + f"\n\n{context_str}"
            return prompt

        actor = AgenticLLMActor(
            config=self.config,
            model=model or mode_cfg["model"],
            system_prompt_generator=generate_system_prompt,
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
            image_callback=image_callback,
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

        # Connect to MCP servers if configured
        if self.mcp_manager is not None:
            await self.mcp_manager.connect_all()

        # Scan and resume any open quests for whitelisted arcs
        await self.quests.scan_and_trigger_open_quests()

        try:
            await self.matrix_monitor.run()
        finally:
            # Clean up shared resources
            if self.mcp_manager is not None:
                await self.mcp_manager.disconnect_all()
            await self.history.close()
            # Chronicle doesn't need explicit cleanup


def _print_cli_help(config: dict[str, Any]) -> None:
    """Print CLI help message based on what's actually configured."""
    tools_config = config.get("tools", {})
    command_config = config.get("matrix", {}).get("command", {})
    modes = command_config.get("modes", {})
    providers_config = config.get("providers", {})

    # Build modes section dynamically based on what's configured
    mode_lines = []

    # Serious mode (always available if configured)
    if modes.get("serious"):
        default_cfg = modes["serious"]
        default_model = default_cfg.get("model", "unknown")
        if isinstance(default_model, list):
            default_model = default_model[0] if default_model else "unknown"
        if ":" in str(default_model):
            default_model = default_model.split(":")[-1]
        mode_lines.append(f"  !s <message>  - Serious mode (default: {default_model})")

    # Sarcastic mode
    if modes.get("sarcastic"):
        mode_lines.append("  !d <message>  - Sarcastic mode - witty, humorous responses")

    # Agent mode (serious mode with tools)
    if modes.get("serious"):
        mode_lines.append("  !a <message>  - Agent mode - multi-turn research with tools")

    # Perplexity mode (only if perplexity provider is configured)
    if providers_config.get("perplexity", {}).get("key"):
        mode_lines.append("  !p <message>  - Perplexity mode - web-enhanced AI responses")

    # Library search (only if library is enabled)
    lib_config = tools_config.get("library", {})
    if lib_config.get("enabled") and lib_config.get("base_url"):
        lib_name = lib_config.get("name", "Library")
        mode_lines.append(f"  !l <query>    - {lib_name} search - direct search without LLM")

    # Unsafe mode
    if modes.get("unsafe"):
        mode_lines.append("  !u <message>  - Unsafe mode - uncensored responses")

    # Verbose is always available as a modifier
    mode_lines.append("  !v <message>  - Verbose mode - detailed responses instead of concise")
    mode_lines.append("  !h            - Show this help message")

    # Build numbered model slots section
    slot_lines = []
    for slot_num in ["2", "3", "4", "5", "6", "7"]:
        mode_key = f"serious{slot_num}"
        mode_cfg = modes.get(mode_key, {})
        if mode_cfg and mode_cfg.get("model"):
            slot_label = mode_cfg.get("slot_label", mode_cfg.get("model", "unknown"))
            slot_lines.append(f"  !{slot_num} <message>  - {slot_label}")

    # Build tools section dynamically
    tool_lines = []

    # Web search (check if any web search provider is configured)
    web_config = tools_config.get("web_search", {})
    if web_config.get("provider"):
        tool_lines.append("  - Web search and webpage visiting")

    # Knowledge base
    kb_config = tools_config.get("knowledge_base", {})
    if kb_config.get("enabled"):
        tool_lines.append("  - OSGeo Knowledge (wiki, news, relationships)")

    # Library
    if lib_config.get("enabled") and lib_config.get("base_url"):
        lib_name = lib_config.get("name", "Library")
        tool_lines.append(f"  - {lib_name} (publications, figures, tables, equations)")

    # Code execution
    if tools_config.get("code_execution", {}).get("enabled"):
        tool_lines.append("  - Code execution (Python sandbox)")

    # Image generation
    if tools_config.get("image_generation", {}).get("enabled"):
        tool_lines.append("  - Image generation")

    # Print the help
    print("\nAvailable Commands")
    print("==================")

    print("\nModes:")
    print("\n".join(mode_lines))

    if slot_lines:
        print("\nAlternate Models:")
        print("\n".join(slot_lines))

    print("\nSource Viewing:")
    print("  sources       - List sources from last search")
    print("  source N      - View source page N")

    print("\nPage Navigation:")
    print("  next / prev   - Navigate pages")
    print("  page N        - Jump to page N")

    print("\nSession:")
    print("  !clear        - Clear chat history and all caches (start fresh)")

    if tool_lines:
        print("\nTools Available:")
        print("\n".join(tool_lines))

    print("\nTips:")
    print("  - Responses are concise by default - say 'tell me more' for details")
    print("  - Use !v prefix when you need a comprehensive answer upfront")
    if lib_config.get("enabled"):
        print("  - Use !l for quick library search, then 'show N' to view images")
    print("  - Use next/prev/page N to navigate document pages")
    print()


def _print_library_help(config: dict[str, Any]) -> None:
    """Print library-specific help message."""
    lib_config = config.get("tools", {}).get("library", {})
    lib_name = lib_config.get("name", "Library")
    lib_description = lib_config.get(
        "description",
        "Search scientific documents, view figures, tables, and equations.",
    )

    print(f"""
{lib_name}
{"=" * len(lib_name)}
{lib_description}

Commands:
  !l <query>        - Search the library

View Sources (golden cord):
  sources           - List sources from last search
  source N          - View source page N

View Elements:
  show N            - View element N (figure/table/equation)

Page Navigation:
  next / prev       - Navigate pages
  page N            - Jump to page N

Examples:
  !l mercator projection
  sources
  source 2
  show 3
  next

Full guide: https://github.com/ominiverdi/matrix-llmagent/blob/main/docs/LIBRARY_TOUR.md
""")


def _render_image_with_chafa(image_data: bytes, element_type: str | None = None) -> bool:
    """Render image in terminal using chafa.

    Args:
        image_data: Raw image bytes
        element_type: Optional element type for sizing hints

    Returns:
        True if rendering succeeded, False otherwise
    """
    import shutil
    import subprocess
    import tempfile

    # Check if chafa is available
    if not shutil.which("chafa"):
        print("(Install chafa for terminal image preview: sudo apt install chafa)")
        return False

    # Get terminal size for proportional display
    term_size = shutil.get_terminal_size((120, 40))
    max_width = min(term_size.columns - 4, 100)

    # Adjust height based on element type
    if element_type == "equation":
        max_height = 12
    elif element_type == "table":
        max_height = min(term_size.lines - 8, 40)
    else:
        max_height = min(term_size.lines - 8, 35)

    size = f"{max_width}x{max_height}"

    try:
        # Write image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(image_data)
            temp_path = f.name

        # Run chafa with high-quality settings
        result = subprocess.run(
            [
                "chafa",
                "--size",
                size,
                "--symbols",
                "all",
                "-w",
                "9",
                "-c",
                "full",
                temp_path,
            ],
            capture_output=False,
        )

        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

        return result.returncode == 0

    except Exception as e:
        logger.warning(f"Failed to render image with chafa: {e}")
        return False


def _parse_show_indices(message: str) -> list[int] | None:
    """Parse 'show 1,2,3' command from message.

    Returns:
        List of 1-indexed result numbers, or None if not a show command.
    """
    match = re.match(r"^show\s+([\d,\s]+)$", message.strip(), re.IGNORECASE)
    if not match:
        return None
    indices = [int(x) for x in re.findall(r"\d+", match.group(1))]
    return indices if indices else None


async def _handle_cli_show_command(
    agent: "MatrixLLMAgent", indices: list[int], arc: str = "cli#test"
) -> None:
    """Handle show command in CLI mode - display images with chafa.

    Args:
        agent: MatrixLLMAgent instance with library_cache
        indices: List of 1-indexed result numbers to show
        arc: Arc identifier for cache lookup (default: cli#test for single-message mode)
    """
    lib_config = agent.config.get("tools", {}).get("library", {})
    if not lib_config.get("enabled") or not lib_config.get("base_url"):
        print("Library search is not configured.")
        return

    cache = agent.library_cache
    if cache is None:
        print("No library search results available. Try searching first.")
        return

    results = cache.get(arc)
    if not results:
        print("No library search results available. Try searching first.")
        return

    base_url = lib_config["base_url"]

    for idx in indices:
        # Validate index
        if idx < 1 or idx > len(results):
            print(f"Invalid index [{idx}]. Available: 1-{len(results)}")
            continue

        result = results[idx - 1]  # Convert to 0-indexed
        doc_title = result.get("document_title", "Unknown")
        page = result.get("page_number", "?")
        element_type = result.get("element_type")

        # Handle text chunks - show the text content
        if result.get("source_type") == "chunk":
            content = result.get("content", "")[:1500]
            print(f"\n#{idx}. TEXT from {doc_title}, page {page}")
            print("-" * 40)
            print(content)
            print()
            continue

        # Handle elements
        element_label = result.get("element_label", "Element")
        element_type_upper = (element_type or "element").upper()
        print(f'\n#{idx}. {element_type_upper}: "{element_label}"')
        print(f"From: {doc_title}, page {page}")

        image_path = get_best_image_path(result)
        if not image_path:
            # No image available, show content text instead
            content = result.get("content", "No content available")[:500]
            print("(No image available)")
            print(content)
            continue

        # Fetch image from library server
        try:
            image_result = await fetch_library_image(
                base_url,
                result.get("document_slug", ""),
                image_path,
            )

            if image_result is None:
                print("Could not fetch image - library server may be unavailable.")
                continue

            image_bytes, _ = image_result
            print()  # Blank line before image

            if not _render_image_with_chafa(image_bytes, element_type):
                print("(Image available but could not be rendered)")

        except Exception as e:
            logger.error(f"Error handling show command for #{idx}: {e}")
            print(f"Error fetching image: {e}")


def _parse_page_command(message: str) -> tuple[str, int | None] | None:
    """Parse page navigation commands.

    Supports (with or without ! prefix):
        next, prev - navigate relative to current page
        page 50, page50 - jump to specific page

    Returns:
        Tuple of (command, page_number) where command is 'next', 'prev', or 'goto'.
        page_number is None for next/prev, or the target page for goto.
        Returns None if not a page command.
    """
    msg = message.strip().lower()

    # Accept with or without ! prefix
    if msg in ("!next", "!n", "next", "n"):
        return ("next", None)
    if msg in ("!prev", "!p", "!previous", "prev", "previous"):
        return ("prev", None)

    # page N or pageN (with or without !)
    match = re.match(r"^!?page\s*(\d+)$", msg)
    if match:
        return ("goto", int(match.group(1)))

    return None


def _parse_source_command(message: str) -> tuple[str, int | None] | None:
    """Parse source viewing commands.

    Supports (with or without ! prefix):
        sources - list all sources from last search
        source N - view source page N

    Returns:
        Tuple of (command, index) where command is 'list' or 'view'.
        index is None for list, or the source index for view.
        Returns None if not a source command.
    """
    msg = message.strip().lower()

    # Accept with or without ! prefix
    if msg in ("!sources", "sources"):
        return ("list", None)

    # source N (with or without !)
    match = re.match(r"^!?source\s+(\d+)$", msg)
    if match:
        return ("view", int(match.group(1)))

    return None


def _parse_docs_command(message: str) -> int | None:
    """Parse 'docs' command with optional page number.

    Supports:
        docs - list documents (page 1)
        docs N - list documents page N

    Returns:
        Page number (default 1) or None if not a docs command.
    """
    msg = message.strip().lower()

    if msg == "docs":
        return 1

    match = re.match(r"^docs\s+(\d+)$", msg)
    if match:
        return int(match.group(1))

    return None


def _parse_doc_command(message: str) -> str | int | None:
    """Parse 'doc' command with slug or index.

    Supports:
        doc <slug> - select document by slug
        doc N - select document by index from last docs listing

    Returns:
        Slug string, index int, or None if not a doc command.
    """
    msg = message.strip()

    match = re.match(r"^doc\s+(\d+)$", msg.lower())
    if match:
        return int(match.group(1))

    match = re.match(r"^doc\s+(\S+)$", msg.lower())
    if match:
        return match.group(1)

    return None


def _parse_elements_command(message: str) -> tuple[str, bool] | None:
    """Parse element listing commands.

    Supports:
        figures, tables, equations - list elements (page-filtered if viewing a page)
        figures all, tables all, equations all - list all elements in document

    Returns:
        Tuple of (element_type, show_all) or None if not an elements command.
    """
    msg = message.strip().lower()

    # Check for "all" variants first
    for elem_type in ("figures", "tables", "equations"):
        if msg == f"{elem_type} all":
            # Convert plural to singular
            singular = elem_type[:-1] if elem_type != "equations" else "equation"
            return (singular, True)

    # Check for simple commands
    type_map = {
        "figures": "figure",
        "tables": "table",
        "equations": "equation",
    }
    if msg in type_map:
        return (type_map[msg], False)

    return None


async def _handle_cli_page_command(
    agent: "MatrixLLMAgent",
    command: str,
    page_number: int | None,
    arc: str,
) -> None:
    """Handle page navigation commands in CLI mode.

    Args:
        agent: MatrixLLMAgent instance with library_cache
        command: 'next', 'prev', or 'goto'
        page_number: Target page for 'goto', None for relative navigation
        arc: Arc identifier for cache lookup
    """
    lib_config = agent.config.get("tools", {}).get("library", {})
    if not lib_config.get("enabled") or not lib_config.get("base_url"):
        print("Library is not configured.")
        return

    cache = agent.library_cache
    if cache is None:
        print("No document open. Use library_search with page_number first.")
        return

    # Get current page view
    page_view = cache.get_page_view(arc)
    if page_view is None:
        print("No document open. Try: 'show me page 1 of <document name>'")
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
        print(f"Already at first page (page 1 of {page_view.total_pages}).")
        return
    if target_page > page_view.total_pages:
        print(f"Already at last page (page {page_view.total_pages} of {page_view.total_pages}).")
        return

    # Fetch the page
    base_url = lib_config["base_url"]
    print(f"Fetching page {target_page} of '{page_view.document_title}'...")

    result = await fetch_library_page(base_url, page_view.document_slug, target_page)

    if isinstance(result, str):
        print(f"Error: {result}")
        return

    # Update page view in cache
    cache.store_page_view(
        arc,
        result.document_slug,
        result.document_title,
        result.page_number,
        result.total_pages,
    )

    # Display with chafa
    print(f"\nPage {result.page_number} of {result.total_pages}: {result.document_title}")
    print("-" * 60)

    if not _render_image_with_chafa(result.image_data, None):
        print("(Page image could not be rendered)")

    print("\n[next/prev to navigate, page N to jump]")


async def _handle_cli_source_command(
    agent: "MatrixLLMAgent",
    command: str,
    index: int | None,
    arc: str,
) -> None:
    """Handle source viewing commands in CLI mode (!sources, !source N).

    The 'golden cord' - lets users see sources from last search and
    view the actual source pages. Supports library, knowledge base, and web search sources.

    Args:
        agent: MatrixLLMAgent instance with library_cache, kb_cache, and web_search_cache
        command: 'list' or 'view'
        index: Source index for 'view', None for 'list'
        arc: Arc identifier for cache lookup
    """
    # Get all caches
    lib_cache = agent.library_cache
    kb_cache = agent.kb_cache
    web_cache = agent.web_search_cache

    # Get results and timestamps from each cache
    lib_results = lib_cache.get(arc) if lib_cache else None
    kb_results = kb_cache.get(arc) if kb_cache else None
    web_results = web_cache.get(arc) if web_cache else None

    lib_timestamp = lib_cache.get_timestamp(arc) if lib_cache else 0.0
    kb_timestamp = kb_cache.get_timestamp(arc) if kb_cache else 0.0
    web_timestamp = web_cache.get_timestamp(arc) if web_cache else 0.0

    # Determine which source to use based on most recent timestamp
    source_type = None
    timestamps = []
    if lib_results:
        timestamps.append(("library", lib_timestamp))
    if kb_results:
        timestamps.append(("kb", kb_timestamp))
    if web_results:
        timestamps.append(("web", web_timestamp))

    if timestamps:
        source_type = max(timestamps, key=lambda x: x[1])[0]

    if source_type is None:
        print("No sources available. Run a search first.")
        return

    # Handle sources based on type
    if source_type == "library":
        assert lib_cache is not None
        await _handle_cli_library_source_command(agent, command, index, arc, lib_cache)
    elif source_type == "kb":
        assert kb_results is not None
        kb_name = (
            agent.config.get("tools", {}).get("knowledge_base", {}).get("name", "Knowledge Base")
        )
        _handle_cli_kb_source_command(command, index, kb_results, kb_name)
    else:  # web
        assert web_results is not None
        _handle_cli_web_source_command(command, index, web_results)


async def _handle_cli_library_source_command(
    agent: "MatrixLLMAgent",
    command: str,
    index: int | None,
    arc: str,
    cache: LibraryResultsCache,
) -> None:
    """Handle source commands for library results in CLI mode."""
    results = cache.get(arc)

    if command == "list":
        if not results:
            print("No sources available. Run a library search first.")
            return

        sources_text = format_sources_list(results)
        print(sources_text)
        return

    # command == "view" - fetch and display source page
    if not results:
        print("No sources available. Run a library search first.")
        return

    if index is None or index < 1 or index > len(results):
        print(f"Invalid source number. Use 1-{len(results)}.")
        return

    # Get the source result
    result = results[index - 1]
    doc_slug = result.get("document_slug")
    page_num = result.get("page_number")
    doc_title = result.get("document_title", "Unknown")

    if not doc_slug or not page_num:
        print("Source is missing document or page information.")
        return

    # Fetch the page
    lib_config = agent.config.get("tools", {}).get("library", {})
    base_url = lib_config.get("base_url")
    if not base_url:
        print("Library is not configured.")
        return

    print(f"Fetching p.{page_num} of '{doc_title}'...")

    page_result = await fetch_library_page(base_url, doc_slug, page_num)

    if isinstance(page_result, str):
        print(f"Error: {page_result}")
        return

    # Update page view in cache for navigation
    cache.store_page_view(
        arc,
        page_result.document_slug,
        page_result.document_title,
        page_result.page_number,
        page_result.total_pages,
    )

    # Display with chafa
    print(
        f"\nSource [{index}]: Page {page_result.page_number} of {page_result.total_pages} - {page_result.document_title}"
    )
    print("-" * 60)

    if not _render_image_with_chafa(page_result.image_data, None):
        print("(Page image could not be rendered)")

    print("\n[next/prev to navigate, sources to list all]")


def _handle_cli_kb_source_command(
    command: str,
    index: int | None,
    results: KBCachedResults,
    kb_name: str = "Knowledge Base",
) -> None:
    """Handle source commands for knowledge base results in CLI mode."""
    kb_results = results

    if command == "list":
        sources_text = format_kb_sources_list(kb_results, kb_name)
        print(sources_text)
        return

    # command == "view" - show full details for the source
    total_items = len(kb_results.pages) + len(kb_results.entities)

    if index is None or index < 1 or index > total_items:
        print(f"Invalid source number. Use 1-{total_items}.")
        return

    detail_text = format_kb_source_detail(kb_results, index)
    print(detail_text)


def _handle_cli_web_source_command(
    command: str,
    index: int | None,
    results: WebSearchCachedResults,
) -> None:
    """Handle source commands for web search results in CLI mode."""
    if command == "list":
        sources_text = format_web_sources_list(results)
        print(sources_text)
        return

    # command == "view" - show full details for the source
    total_items = len(results.results)

    if index is None or index < 1 or index > total_items:
        print(f"Invalid source number. Use 1-{total_items}.")
        return

    detail_text = format_web_source_detail(results, index)
    print(detail_text)


async def _handle_cli_docs_command(
    agent: "MatrixLLMAgent",
    page: int,
    arc: str,
) -> None:
    """Handle 'docs' command - list documents with pagination.

    Args:
        agent: MatrixLLMAgent instance with library_cache
        page: Page number to display (1-indexed)
        arc: Arc identifier for cache storage
    """
    lib_config = agent.config.get("tools", {}).get("library", {})
    if not lib_config.get("enabled") or not lib_config.get("base_url"):
        print("Library is not configured.")
        return

    cache = agent.library_cache
    if cache is None:
        print("Library cache not initialized.")
        return

    base_url = lib_config["base_url"]
    page_size = 5  # Match Rust CLI

    result = await list_documents(base_url, page=page, page_size=page_size)

    if isinstance(result, str):
        print(f"Error: {result}")
        return

    documents = result.get("documents", [])
    total_pages = result.get("total_pages", 1)

    # Store in cache for doc N selection and next/prev navigation
    cache.store_document_list(arc, page, total_pages, page_size, documents)

    # Format and display
    output = format_document_list(documents, page, total_pages)
    print(output)


async def _handle_cli_doc_command(
    agent: "MatrixLLMAgent",
    slug_or_index: str | int,
    arc: str,
) -> None:
    """Handle 'doc' command - select document and show details.

    Args:
        agent: MatrixLLMAgent instance with library_cache
        slug_or_index: Document slug string or index from last docs listing
        arc: Arc identifier for cache storage
    """
    lib_config = agent.config.get("tools", {}).get("library", {})
    if not lib_config.get("enabled") or not lib_config.get("base_url"):
        print("Library is not configured.")
        return

    cache = agent.library_cache
    if cache is None:
        print("Library cache not initialized.")
        return

    base_url = lib_config["base_url"]

    # Resolve index to slug if needed
    if isinstance(slug_or_index, int):
        doc_list = cache.get_document_list(arc)
        if doc_list is None:
            print("Run 'docs' first to see available documents.")
            return

        index = slug_or_index
        if index < 1 or index > len(doc_list.documents):
            print(f"Invalid document number. Use 1-{len(doc_list.documents)}.")
            return

        slug = doc_list.documents[index - 1].get("slug")
        if not slug:
            print("Error: Document slug not found in cache.")
            return
    else:
        slug = slug_or_index

    # Fetch document info
    result = await get_document_info(base_url, slug)

    if isinstance(result, str):
        print(f"Error: {result}")
        return

    # Store document view in cache
    cache.store_document_view(
        arc,
        document_slug=result.get("slug", slug),
        document_title=result.get("title", "Unknown"),
        total_pages=result.get("total_pages", 0),
        element_counts=result.get("element_counts", {}),
        keywords=result.get("keywords", []),
        summary=result.get("summary", ""),
    )

    # Format and display
    output = format_document_info(result)
    print(output)


async def _handle_cli_elements_command(
    agent: "MatrixLLMAgent",
    element_type: str,
    show_all: bool,
    arc: str,
) -> None:
    """Handle 'figures'/'tables'/'equations' commands - list elements.

    Args:
        agent: MatrixLLMAgent instance with library_cache
        element_type: Type to list (figure, table, equation)
        show_all: If True, show all elements; if False, filter to current page
        arc: Arc identifier for cache storage
    """
    lib_config = agent.config.get("tools", {}).get("library", {})
    if not lib_config.get("enabled") or not lib_config.get("base_url"):
        print("Library is not configured.")
        return

    cache = agent.library_cache
    if cache is None:
        print("Library cache not initialized.")
        return

    # Need a document selected
    doc_view = cache.get_document_view(arc)
    if doc_view is None:
        print("Select a document first with 'doc <slug>'.")
        return

    base_url = lib_config["base_url"]

    # Determine page filter
    page_filter = None
    if not show_all:
        page_view = cache.get_page_view(arc)
        if page_view and page_view.document_slug == doc_view.document_slug:
            page_filter = page_view.page

    # Fetch elements
    result = await list_elements(
        base_url,
        doc_view.document_slug,
        element_type=element_type,
        page=page_filter,
        limit=50,
    )

    if isinstance(result, str):
        print(f"Error: {result}")
        return

    elements = result.get("elements", [])
    total = result.get("total", len(elements))

    # Store elements in cache for show N command
    # Convert to format expected by show command (add source_type for compatibility)
    cache_elements = []
    for elem in elements:
        cache_elements.append(
            {
                "source_type": "element",
                "element_type": element_type,
                "element_label": elem.get("label", ""),
                "page_number": elem.get("page_number"),
                "content": elem.get("description", ""),
                "crop_path": elem.get("crop_path", ""),
                "rendered_path": elem.get("rendered_path", ""),
                "document_slug": doc_view.document_slug,
                "document_title": doc_view.document_title,
            }
        )
    cache.store_element_list(arc, cache_elements)

    # Also store in main cache for show command compatibility
    cache.store(arc, cache_elements)

    # Format and display
    output = format_element_list(
        elements,
        element_type,
        doc_view.document_title,
        page_filter=page_filter,
        total=total,
    )
    print(output)


async def _handle_cli_nav_command(
    agent: "MatrixLLMAgent",
    direction: str,
    arc: str,
) -> None:
    """Handle context-aware next/prev navigation.

    Navigates documents list if last action was 'docs', otherwise navigates pages.

    Args:
        agent: MatrixLLMAgent instance with library_cache
        direction: 'next' or 'prev'
        arc: Arc identifier for cache lookup
    """
    cache = agent.library_cache
    if cache is None:
        print("Nothing to navigate. Try 'docs' or view a page first.")
        return

    context = cache.get_navigation_context(arc)

    if context == "documents":
        # Navigate document list
        doc_list = cache.get_document_list(arc)
        if doc_list is None:
            print("No document list. Run 'docs' first.")
            return

        if direction == "next":
            if doc_list.page >= doc_list.total_pages:
                print(f"Already at last page (page {doc_list.page}/{doc_list.total_pages}).")
                return
            await _handle_cli_docs_command(agent, doc_list.page + 1, arc)
        else:  # prev
            if doc_list.page <= 1:
                print(f"Already at first page (page 1/{doc_list.total_pages}).")
                return
            await _handle_cli_docs_command(agent, doc_list.page - 1, arc)

    elif context == "pages":
        # Navigate pages - delegate to existing handler
        await _handle_cli_page_command(agent, direction, None, arc)

    else:
        print("Nothing to navigate. Try 'docs' or view a page first.")


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

        # Connect to MCP servers if configured
        if agent.mcp_manager is not None:
            await agent.mcp_manager.connect_all()

        # Handle help command
        if message.lower().strip() in ("!h", "!help", "help"):
            _print_cli_help(agent.config)
            return

        # Handle show command
        show_indices = _parse_show_indices(message)
        if show_indices is not None:
            await _handle_cli_show_command(agent, show_indices)
            return

        # Handle page navigation commands (!next, !prev, !page N)
        page_cmd = _parse_page_command(message)
        if page_cmd is not None:
            cmd, page_num = page_cmd
            if cmd in ("next", "prev"):
                # Use context-aware navigation
                await _handle_cli_nav_command(agent, cmd, "cli#test")
            else:
                await _handle_cli_page_command(agent, cmd, page_num, "cli#test")
            return

        # Handle source viewing commands (!sources, !source N)
        source_cmd = _parse_source_command(message)
        if source_cmd is not None:
            cmd, idx = source_cmd
            await _handle_cli_source_command(agent, cmd, idx, "cli#test")
            return

        # Handle docs command
        docs_page = _parse_docs_command(message)
        if docs_page is not None:
            await _handle_cli_docs_command(agent, docs_page, "cli#test")
            return

        # Handle doc command
        doc_arg = _parse_doc_command(message)
        if doc_arg is not None:
            await _handle_cli_doc_command(agent, doc_arg, "cli#test")
            return

        # Handle elements commands (figures, tables, equations)
        elements_cmd = _parse_elements_command(message)
        if elements_cmd is not None:
            elem_type, show_all = elements_cmd
            await _handle_cli_elements_command(agent, elem_type, show_all, "cli#test")
            return

        # Handle !l library search command (direct, no LLM)
        if message.lower() == "!l" or message.startswith("!l ") or message.startswith("!L "):
            query = message[2:].strip() if len(message) > 2 else ""
            if not query or query.lower() == "help":
                _print_library_help(agent.config)
                return

            lib_config = agent.config.get("tools", {}).get("library", {})
            if not lib_config.get("enabled") or not lib_config.get("base_url"):
                print("Library search is not configured.")
                return

            lib_name = lib_config.get("name", "OSGeo Library")
            max_results = lib_config.get("max_results", 10)

            # Ensure cache exists
            if agent.library_cache is None:
                cache_config = lib_config.get("cache", {})
                agent.library_cache = LibraryResultsCache(
                    ttl_hours=cache_config.get("ttl_hours", 24),
                    max_rooms=cache_config.get("max_rooms", 100),
                )

            print(f"Searching {lib_name} for: {query}")
            print("-" * 60)

            results, formatted = await search_library_direct(
                base_url=lib_config["base_url"],
                query=query,
                cache=agent.library_cache,
                arc="cli#test",
                name=lib_name,
                limit=max_results,
            )

            print(formatted)
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
        # Numbered model slots for comparison testing
        elif clean_message.startswith("!2 "):
            mode = "serious2"
            clean_message = clean_message[3:]
        elif clean_message.startswith("!3 "):
            mode = "serious3"
            clean_message = clean_message[3:]
        elif clean_message.startswith("!4 "):
            mode = "serious4"
            clean_message = clean_message[3:]
        elif clean_message.startswith("!5 "):
            mode = "serious5"
            clean_message = clean_message[3:]
        elif clean_message.startswith("!6 "):
            mode = "serious6"
            clean_message = clean_message[3:]
        elif clean_message.startswith("!7 "):
            mode = "serious7"
            clean_message = clean_message[3:]

        # Get mode configuration with inheritance for numbered slots
        command_config = agent.config.get("matrix", {}).get("command", {})
        modes = command_config.get("modes", {})
        mode_cfg = modes.get(mode, {})

        # Inherit from base mode if this is a numbered slot (e.g., serious2 inherits from serious)
        if mode.startswith("serious") and mode != "serious":
            base_cfg = modes.get("serious", {})
            # Merge: base config first, then mode-specific overrides
            mode_cfg = {**base_cfg, **mode_cfg}

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

        # Image callback for CLI - render with chafa
        async def cli_image_callback(image_bytes: bytes, mimetype: str) -> None:
            print()  # Blank line before image
            _render_image_with_chafa(image_bytes, None)

        # Run actor
        response = await agent.run_actor(
            context,
            mode_cfg=mode_cfg,
            system_prompt=system_prompt,
            arc="cli#test",
            image_callback=cli_image_callback,
        )

        print("-" * 60)
        if response:
            # Add slot label prefix if configured (for model comparison)
            slot_label = mode_cfg.get("slot_label")
            if slot_label:
                response = f"[{slot_label}] {response}"
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
        # Cleanup MCP connections
        if agent.mcp_manager is not None:
            await agent.mcp_manager.disconnect_all()
        await agent.history.close()


async def cli_chronicler(arc: str, instructions: str, config_path: str | None = None) -> None:
    """CLI mode for Chronicler operations."""
    # Load configuration
    config_file = Path(config_path) if config_path else Path(__file__).parent.parent / "config.json"

    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        print("Please create config.json from config.json.example")
        sys.exit(1)

    print(f"Chronicler arc '{arc}': {instructions}")
    print("=" * 60)

    try:
        # Create agent instance
        agent = MatrixLLMAgent(str(config_file))
        await agent.chronicle.initialize()

        print(
            "Error: Chronicler subagent functionality has been removed. Use direct chronicle_append and chronicle_read tools instead."
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


async def cli_interactive(config_path: str | None = None) -> None:
    """Interactive CLI mode that simulates a Matrix room.

    Maintains conversation history, supports library search with chafa image display,
    and follows the same command patterns as the Matrix client.
    """
    import readline  # noqa: F401 - imported for side effects (line editing)

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

        # Persistent arc for CLI interactive session (like a Matrix room)
        arc = "cli#interactive"
        bot_name = "cli-assistant"

        # Get command configuration
        command_config = agent.config.get("matrix", {}).get("command", {})
        default_mode = command_config.get("default_mode", "serious")

        print("matrix-llmagent interactive mode")
        print("Type 'help' for commands, 'quit' to exit")
        print("-" * 60)

        while True:
            try:
                message = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not message:
                continue

            # Handle quit
            if message.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # Handle help
            if message.lower() in ("help", "!h", "!help"):
                _print_cli_help(agent.config)
                continue

            # Handle clear command - reset history and caches
            if message.lower() in ("!clear", "!reset", "clear"):
                # Clear chat history for this arc
                await agent.history.clear_arc("cli", "interactive")
                # Clear all caches
                if agent.library_cache:
                    agent.library_cache.clear_all(arc)
                if agent.kb_cache:
                    agent.kb_cache.clear(arc)
                if agent.web_search_cache:
                    agent.web_search_cache.clear(arc)
                print("Cleared: chat history, library cache, KB cache, web search cache")
                continue

            # Handle show command
            show_indices = _parse_show_indices(message)
            if show_indices is not None:
                await _handle_cli_show_command(agent, show_indices, arc)
                continue

            # Handle page navigation commands (!next, !prev, !page N)
            page_cmd = _parse_page_command(message)
            if page_cmd is not None:
                cmd, page_num = page_cmd
                if cmd in ("next", "prev"):
                    # Use context-aware navigation
                    await _handle_cli_nav_command(agent, cmd, arc)
                else:
                    await _handle_cli_page_command(agent, cmd, page_num, arc)
                continue

            # Handle source viewing commands (!sources, !source N)
            source_cmd = _parse_source_command(message)
            if source_cmd is not None:
                cmd, idx = source_cmd
                await _handle_cli_source_command(agent, cmd, idx, arc)
                continue

            # Handle docs command
            docs_page = _parse_docs_command(message)
            if docs_page is not None:
                await _handle_cli_docs_command(agent, docs_page, arc)
                continue

            # Handle doc command
            doc_arg = _parse_doc_command(message)
            if doc_arg is not None:
                await _handle_cli_doc_command(agent, doc_arg, arc)
                continue

            # Handle elements commands (figures, tables, equations)
            elements_cmd = _parse_elements_command(message)
            if elements_cmd is not None:
                elem_type, show_all = elements_cmd
                await _handle_cli_elements_command(agent, elem_type, show_all, arc)
                continue

            # Handle !l library search (direct, no LLM)
            if message.lower() == "!l" or message.startswith("!l ") or message.startswith("!L "):
                query = message[2:].strip() if len(message) > 2 else ""
                if not query or query.lower() == "help":
                    _print_library_help(agent.config)
                    continue

                lib_config = agent.config.get("tools", {}).get("library", {})
                if not lib_config.get("enabled") or not lib_config.get("base_url"):
                    print("Library search is not configured.")
                    continue

                lib_name = lib_config.get("name", "OSGeo Library")
                max_results = lib_config.get("max_results", 10)

                # Ensure cache exists
                if agent.library_cache is None:
                    cache_config = lib_config.get("cache", {})
                    agent.library_cache = LibraryResultsCache(
                        ttl_hours=cache_config.get("ttl_hours", 24),
                        max_rooms=cache_config.get("max_rooms", 100),
                    )

                print(f"Searching {lib_name}...")

                results, formatted = await search_library_direct(
                    base_url=lib_config["base_url"],
                    query=query,
                    cache=agent.library_cache,
                    arc=arc,
                    name=lib_name,
                    limit=max_results,
                )
                print(formatted)
                continue

            # Determine mode from message (same as Matrix client)
            mode = default_mode
            verbose = False
            clean_message = message

            # Check for verbose modifier
            if message.startswith("!v ") or message.startswith("!V "):
                verbose = True
                clean_message = message[3:]

            # Check for mode prefixes
            if clean_message.startswith("!s ") or clean_message.startswith("!S "):
                mode = "serious"
                clean_message = clean_message[3:]
            elif clean_message.startswith("!d ") or clean_message.startswith("!D "):
                mode = "sarcastic"
                clean_message = clean_message[3:]
            elif clean_message.startswith("!2 "):
                mode = "serious2"
                clean_message = clean_message[3:]
            elif clean_message.startswith("!3 "):
                mode = "serious3"
                clean_message = clean_message[3:]
            elif clean_message.startswith("!4 "):
                mode = "serious4"
                clean_message = clean_message[3:]
            elif clean_message.startswith("!5 "):
                mode = "serious5"
                clean_message = clean_message[3:]
            elif clean_message.startswith("!6 "):
                mode = "serious6"
                clean_message = clean_message[3:]
            elif clean_message.startswith("!7 "):
                mode = "serious7"
                clean_message = clean_message[3:]

            # Get mode configuration with inheritance
            modes = command_config.get("modes", {})
            mode_cfg = modes.get(mode, {})

            # Inherit from base mode if this is a numbered slot
            if mode.startswith("serious") and mode != "serious":
                base_cfg = modes.get("serious", {})
                mode_cfg = {**base_cfg, **mode_cfg}

            if not mode_cfg:
                print(f"Error: Mode '{mode}' not configured")
                continue
            if not mode_cfg.get("model"):
                print(f"Error: No model configured for mode '{mode}'")
                continue

            # Build context from chat history (same as Matrix room)
            history_size = mode_cfg.get("history_size", command_config.get("history_size", 30))
            context = await agent.history.get_context("cli", "interactive", history_size)

            # Add current message to context
            context.append({"role": "user", "content": clean_message})

            # Build system prompt
            system_prompt = mode_cfg.get("system_prompt", "You are a helpful assistant.")
            system_prompt = system_prompt.replace("{mynick}", bot_name)

            # Apply verbosity modifier
            if verbose:
                system_prompt += " Provide comprehensive, detailed responses."
            else:
                system_prompt += (
                    " Keep responses to 1 sentence max. Users can say 'tell me more' for details."
                )

            # Image callback for CLI - render with chafa
            async def cli_image_callback(image_bytes: bytes, mimetype: str) -> None:
                print()  # Blank line before image
                _render_image_with_chafa(image_bytes, None)

            # Run actor
            try:
                response = await agent.run_actor(
                    context,
                    mode_cfg=mode_cfg,
                    system_prompt=system_prompt,
                    arc=arc,
                    image_callback=cli_image_callback,
                )

                if response:
                    # Add slot label if configured
                    slot_label = mode_cfg.get("slot_label")
                    if slot_label:
                        response = f"[{slot_label}] {response}"

                    print("-" * 60)
                    print(response)

                    # Save to history (same as Matrix)
                    # nick="user", mynick=bot_name so user messages get role="user"
                    await agent.history.add_message(
                        "cli", "interactive", clean_message, "user", bot_name
                    )
                    await agent.history.add_message(
                        "cli", "interactive", response, bot_name, bot_name, True
                    )
                else:
                    print("(No response)")

            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}")

    except Exception as e:
        logger.error(f"CLI interactive error: {e}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        await agent.history.close()


async def run_conversation_test(
    test_file: str,
    config_path: str | None = None,
    verbose: bool = False,
) -> bool:
    """Run a multi-turn conversation test from JSON file.

    Test file format:
    {
        "description": "Test description",
        "timeout_seconds": 300,
        "mode": "serious",
        "steps": [
            {
                "input": "User message",
                "expect_contains": ["expected", "strings"],
                "expect_not_contains": ["unwanted", "strings"]
            }
        ]
    }

    Args:
        test_file: Path to JSON test file
        config_path: Optional path to config file
        verbose: Print full responses if True

    Returns:
        True if all steps pass, False otherwise
    """
    import time as time_module

    # Load test file
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"Error: Test file not found: {test_file}")
        return False

    try:
        with open(test_path) as f:
            test_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in test file: {e}")
        return False

    description = test_data.get("description", test_path.stem)
    timeout_seconds = test_data.get("timeout_seconds", 300)
    test_mode = test_data.get("mode", "serious")
    steps = test_data.get("steps", [])

    if not steps:
        print("Error: No steps defined in test file")
        return False

    # Load configuration
    config_file = Path(config_path) if config_path else Path(__file__).parent.parent / "config.json"

    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        return False

    print(f"\n{'=' * 60}")
    print(f"Conversation Test: {description}")
    print(f"{'=' * 60}")
    print(f"Mode: {test_mode} | Steps: {len(steps)} | Timeout: {timeout_seconds}s/step")
    print()

    try:
        agent = MatrixLLMAgent(str(config_file))
        await agent.history.initialize()
        await agent.chronicle.initialize()

        # Use a unique arc for this test run
        arc = f"cli#test_{int(time_module.time())}"

        # Get mode configuration
        command_config = agent.config.get("matrix", {}).get("command", {})
        modes = command_config.get("modes", {})
        mode_cfg = modes.get(test_mode, {})

        if not mode_cfg:
            print(f"Error: Mode '{test_mode}' not configured")
            return False

        # Build system prompt
        system_prompt = mode_cfg.get("system_prompt", "You are a helpful assistant.")
        system_prompt = system_prompt.replace("{mynick}", "CLI-Bot")
        system_prompt += (
            " Keep responses to 1 sentence max. Users can say 'tell me more' for details."
        )

        # Track results
        results: list[dict[str, Any]] = []
        passed = 0
        failed = 0

        # Conversation context (maintains history across steps)
        context: list[dict[str, str]] = []

        for i, step in enumerate(steps, 1):
            step_input = step.get("input", "")
            expect_contains = step.get("expect_contains", [])
            expect_not_contains = step.get("expect_not_contains", [])

            print(
                f'[Step {i}/{len(steps)}] "{step_input[:50]}{"..." if len(step_input) > 50 else ""}"'
            )

            start_time = time_module.time()
            response = ""
            iterations = 0
            is_command = False

            try:
                # Check if this is a heuristic command (not sent to LLM)
                # Library browsing commands (no ! prefix)
                docs_page = _parse_docs_command(step_input)
                if docs_page is not None:
                    is_command = True
                    response = await _capture_cli_docs_command(agent, docs_page, arc)

                if not is_command:
                    doc_arg = _parse_doc_command(step_input)
                    if doc_arg is not None:
                        is_command = True
                        response = await _capture_cli_doc_command(agent, doc_arg, arc)

                if not is_command:
                    elements_cmd = _parse_elements_command(step_input)
                    if elements_cmd is not None:
                        is_command = True
                        elem_type, show_all = elements_cmd
                        response = await _capture_cli_elements_command(
                            agent, elem_type, show_all, arc
                        )

                if not is_command:
                    # Navigation commands: n, p, next, prev
                    nav_input = step_input.strip().lower()
                    if nav_input in ("n", "next"):
                        is_command = True
                        response = await _capture_cli_nav_command(agent, "next", arc)
                    elif nav_input in ("p", "prev"):
                        is_command = True
                        response = await _capture_cli_nav_command(agent, "prev", arc)

                if not is_command and step_input.strip().lower() == "clear":
                    # Clear command
                    is_command = True
                    response = await _capture_cli_clear_command(agent, arc)

                if not is_command:
                    # Show command (no ! prefix): show N, show N,M,...
                    show_indices = _parse_show_indices(step_input)
                    if show_indices is not None:
                        is_command = True
                        response = await _capture_cli_show_command(agent, show_indices)

                if not is_command:
                    # Page command: page N
                    page_cmd = _parse_page_command(step_input)
                    if page_cmd is not None:
                        is_command = True
                        cmd, page_num = page_cmd
                        response = await _capture_cli_page_command(agent, cmd, page_num, arc)

                # Legacy ! prefixed commands
                if not is_command and step_input.startswith("!"):
                    source_cmd = _parse_source_command(step_input)
                    if source_cmd is not None:
                        is_command = True
                        cmd, idx = source_cmd
                        response = await _capture_cli_source_command(agent, cmd, idx, arc)
                    elif step_input.lower().startswith("!l "):
                        is_command = True
                        query = step_input[3:].strip()
                        response = await _capture_cli_library_search(agent, query, arc)

                if not is_command:
                    # Regular message - send to LLM
                    context.append({"role": "user", "content": step_input})

                    response = await agent.run_actor(
                        context,
                        mode_cfg=mode_cfg,
                        system_prompt=system_prompt,
                        arc=arc,
                    )
                    response = response or ""

                    # Add response to context for conversation continuity
                    if response:
                        context.append({"role": "assistant", "content": response})

                    # Try to get iteration count from actor (if available)
                    iterations = getattr(agent, "_last_iterations", 0)

            except TimeoutError:
                response = f"TIMEOUT after {timeout_seconds}s"
            except Exception as e:
                response = f"ERROR: {e}"

            elapsed = time_module.time() - start_time

            # Check assertions
            missing = []
            unwanted = []

            for expected in expect_contains:
                if expected.lower() not in response.lower():
                    missing.append(expected)

            for not_expected in expect_not_contains:
                if not_expected.lower() in response.lower():
                    unwanted.append(not_expected)

            step_passed = len(missing) == 0 and len(unwanted) == 0

            # Record result
            result = {
                "step": i,
                "input": step_input,
                "passed": step_passed,
                "elapsed": elapsed,
                "iterations": iterations if not is_command else 0,
                "is_command": is_command,
                "response_length": len(response),
            }
            results.append(result)

            if step_passed:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            # Print result
            print(f"  Status: {status}")
            if is_command:
                print("  Type: command")
            else:
                print(f"  Iterations: {iterations}")
            print(f"  Time: {elapsed:.1f}s")

            if not step_passed:
                if missing:
                    print(f"  Missing: {missing}")
                if unwanted:
                    print(f"  Unwanted: {unwanted}")

            if verbose or not step_passed:
                # Show response preview
                preview = response[:200] + "..." if len(response) > 200 else response
                preview = preview.replace("\n", " ")
                print(f"  Response: {preview}")

            print()

        # Summary
        print(f"{'=' * 60}")
        total = passed + failed
        status_str = "PASSED" if failed == 0 else "FAILED"
        print(f"Result: {passed}/{total} {status_str}")

        # Print timing summary
        total_time = sum(r["elapsed"] for r in results)
        llm_steps = [r for r in results if not r["is_command"]]
        if llm_steps:
            avg_time = sum(r["elapsed"] for r in llm_steps) / len(llm_steps)
            total_iterations = sum(r["iterations"] for r in llm_steps)
            print(
                f"Total time: {total_time:.1f}s | Avg LLM step: {avg_time:.1f}s | Total iterations: {total_iterations}"
            )

        print(f"{'=' * 60}\n")

        return failed == 0

    except Exception as e:
        logger.error(f"Conversation test error: {e}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        await agent.history.close()


async def _capture_cli_show_command(agent: "MatrixLLMAgent", indices: list[int]) -> str:
    """Capture output from show command for testing."""
    # For now, return a simple message - show command displays images
    return f"Show command executed for indices: {indices}"


async def _capture_cli_page_command(
    agent: "MatrixLLMAgent", command: str, page_num: int | None, arc: str
) -> str:
    """Capture output from page command for testing."""
    cache = agent.library_cache
    if cache is None:
        return "No page navigation available. Run a library search first."

    page_view = cache.get_page_view(arc)
    if page_view is None:
        return "No page navigation available. View a source first."

    return f"Page command '{command}' executed. Current: p.{page_view.page}/{page_view.total_pages}"


async def _capture_cli_source_command(
    agent: "MatrixLLMAgent", command: str, index: int | None, arc: str
) -> str:
    """Capture output from source command for testing."""
    # Get all caches
    lib_cache = agent.library_cache
    kb_cache = agent.kb_cache
    web_cache = agent.web_search_cache

    # Get results and timestamps from each cache
    lib_results = lib_cache.get(arc) if lib_cache else None
    kb_results = kb_cache.get(arc) if kb_cache else None
    web_results = web_cache.get(arc) if web_cache else None

    lib_timestamp = lib_cache.get_timestamp(arc) if lib_cache else 0.0
    kb_timestamp = kb_cache.get_timestamp(arc) if kb_cache else 0.0
    web_timestamp = web_cache.get_timestamp(arc) if web_cache else 0.0

    # Determine which source to use based on most recent timestamp
    source_type = None
    timestamps = []
    if lib_results:
        timestamps.append(("library", lib_timestamp))
    if kb_results:
        timestamps.append(("kb", kb_timestamp))
    if web_results:
        timestamps.append(("web", web_timestamp))

    if timestamps:
        source_type = max(timestamps, key=lambda x: x[1])[0]

    if source_type is None:
        return "No sources available. Run a search first."

    if command == "list":
        if source_type == "library":
            assert lib_results is not None
            return format_sources_list(lib_results)
        elif source_type == "kb":
            assert kb_results is not None
            kb_name = (
                agent.config.get("tools", {})
                .get("knowledge_base", {})
                .get("name", "Knowledge Base")
            )
            return format_kb_sources_list(kb_results, kb_name)
        else:  # web
            assert web_results is not None
            return format_web_sources_list(web_results)
    else:
        # view command
        if source_type == "library":
            assert lib_results is not None
            if index is None or index < 1 or index > len(lib_results):
                return f"Invalid source number. Use 1-{len(lib_results)}."
            result = lib_results[index - 1]
            doc_title = result.get("document_title", "Unknown")
            page_num = result.get("page_number", "?")
            return f"Source [{index}]: {doc_title} p.{page_num} [image would be displayed]"
        elif source_type == "kb":
            assert kb_results is not None
            if index is None:
                return "Invalid source number."
            return format_kb_source_detail(kb_results, index)
        else:  # web
            assert web_results is not None
            if index is None:
                return "Invalid source number."
            return format_web_source_detail(web_results, index)


async def _capture_cli_library_search(agent: "MatrixLLMAgent", query: str, arc: str) -> str:
    """Capture output from library search for testing."""
    lib_config = agent.config.get("tools", {}).get("library", {})
    if not lib_config.get("enabled") or not lib_config.get("base_url"):
        return "Library search is not configured."

    lib_name = lib_config.get("name", "OSGeo Library")
    max_results = lib_config.get("max_results", 10)

    # Ensure cache exists
    if agent.library_cache is None:
        cache_config = lib_config.get("cache", {})
        agent.library_cache = LibraryResultsCache(
            ttl_hours=cache_config.get("ttl_hours", 24),
            max_rooms=cache_config.get("max_rooms", 100),
        )

    results, formatted = await search_library_direct(
        base_url=lib_config["base_url"],
        query=query,
        cache=agent.library_cache,
        arc=arc,
        name=lib_name,
        limit=max_results,
    )

    return formatted


async def _capture_cli_docs_command(agent: "MatrixLLMAgent", page: int, arc: str) -> str:
    """Capture output from docs command for testing."""
    lib_config = agent.config.get("tools", {}).get("library", {})
    if not lib_config.get("enabled") or not lib_config.get("base_url"):
        return "Library is not configured."

    cache = agent.library_cache
    if cache is None:
        return "Library cache not initialized."

    base_url = lib_config["base_url"]
    page_size = 5

    result = await list_documents(base_url, page=page, page_size=page_size)

    if isinstance(result, str):
        return f"Error: {result}"

    documents = result.get("documents", [])
    total_pages = result.get("total_pages", 1)

    cache.store_document_list(arc, page, total_pages, page_size, documents)

    return format_document_list(documents, page, total_pages)


async def _capture_cli_doc_command(
    agent: "MatrixLLMAgent", slug_or_index: str | int, arc: str
) -> str:
    """Capture output from doc command for testing."""
    lib_config = agent.config.get("tools", {}).get("library", {})
    if not lib_config.get("enabled") or not lib_config.get("base_url"):
        return "Library is not configured."

    cache = agent.library_cache
    if cache is None:
        return "Library cache not initialized."

    base_url = lib_config["base_url"]

    if isinstance(slug_or_index, int):
        doc_list = cache.get_document_list(arc)
        if doc_list is None:
            return "Run 'docs' first to see available documents."

        index = slug_or_index
        if index < 1 or index > len(doc_list.documents):
            return f"Invalid document number. Use 1-{len(doc_list.documents)}."

        slug = doc_list.documents[index - 1].get("slug")
        if not slug:
            return "Error: Document slug not found in cache."
    else:
        slug = slug_or_index

    result = await get_document_info(base_url, slug)

    if isinstance(result, str):
        return f"Error: {result}"

    cache.store_document_view(
        arc,
        document_slug=result.get("slug", slug),
        document_title=result.get("title", "Unknown"),
        total_pages=result.get("total_pages", 0),
        element_counts=result.get("element_counts", {}),
        keywords=result.get("keywords", []),
        summary=result.get("summary", ""),
    )

    return format_document_info(result)


async def _capture_cli_elements_command(
    agent: "MatrixLLMAgent", element_type: str, show_all: bool, arc: str
) -> str:
    """Capture output from figures/tables/equations command for testing."""
    lib_config = agent.config.get("tools", {}).get("library", {})
    if not lib_config.get("enabled") or not lib_config.get("base_url"):
        return "Library is not configured."

    cache = agent.library_cache
    if cache is None:
        return "Library cache not initialized."

    doc_view = cache.get_document_view(arc)
    if doc_view is None:
        return "Select a document first with 'doc <slug>'."

    base_url = lib_config["base_url"]

    page_filter = None
    if not show_all:
        page_view = cache.get_page_view(arc)
        if page_view and page_view.document_slug == doc_view.document_slug:
            page_filter = page_view.page

    result = await list_elements(
        base_url,
        doc_view.document_slug,
        element_type=element_type,
        page=page_filter,
        limit=50,
    )

    if isinstance(result, str):
        return f"Error: {result}"

    elements = result.get("elements", [])
    total = result.get("total", len(elements))

    cache_elements = []
    for elem in elements:
        cache_elements.append(
            {
                "source_type": "element",
                "element_type": element_type,
                "element_label": elem.get("label", ""),
                "page_number": elem.get("page_number"),
                "content": elem.get("description", ""),
                "crop_path": elem.get("crop_path", ""),
                "rendered_path": elem.get("rendered_path", ""),
                "document_slug": doc_view.document_slug,
                "document_title": doc_view.document_title,
            }
        )
    cache.store_element_list(arc, cache_elements)
    cache.store(arc, cache_elements)

    return format_element_list(
        elements,
        element_type,
        doc_view.document_title,
        page_filter=page_filter,
        total=total,
    )


async def _capture_cli_nav_command(agent: "MatrixLLMAgent", direction: str, arc: str) -> str:
    """Capture output from next/prev navigation for testing."""
    cache = agent.library_cache
    if cache is None:
        return "Nothing to navigate. Try 'docs' or view a page first."

    context = cache.get_navigation_context(arc)

    if context == "documents":
        doc_list = cache.get_document_list(arc)
        if doc_list is None:
            return "No document list. Run 'docs' first."

        if direction == "next":
            if doc_list.page >= doc_list.total_pages:
                return f"Already at last page (page {doc_list.page}/{doc_list.total_pages})."
            return await _capture_cli_docs_command(agent, doc_list.page + 1, arc)
        else:
            if doc_list.page <= 1:
                return f"Already at first page (page 1/{doc_list.total_pages})."
            return await _capture_cli_docs_command(agent, doc_list.page - 1, arc)

    elif context == "pages":
        return await _capture_cli_page_command(agent, direction, None, arc)

    else:
        return "Nothing to navigate. Try 'docs' or view a page first."


async def _capture_cli_clear_command(agent: "MatrixLLMAgent", arc: str) -> str:
    """Capture output from clear command for testing."""
    cache = agent.library_cache
    if cache:
        cache.clear_all(arc)
    return "Cleared library state."


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="matrix-llmagent - Matrix chatbot with AI and tools"
    )
    parser.add_argument(
        "--message", type=str, help="Run in CLI mode to simulate handling a message"
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive CLI mode (simulates a Matrix room)",
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
    parser.add_argument(
        "--test-conversation",
        type=str,
        metavar="FILE",
        help="Run a multi-turn conversation test from JSON file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output for conversation tests",
    )

    args = parser.parse_args()

    if args.chronicler:
        if not args.arc:
            print("Error: --arc is required with --chronicler")
            sys.exit(1)
        asyncio.run(cli_chronicler(args.arc, args.chronicler, args.config))
        return

    if args.test_conversation:
        success = asyncio.run(
            run_conversation_test(args.test_conversation, args.config, args.verbose)
        )
        sys.exit(0 if success else 1)

    if args.interactive:
        asyncio.run(cli_interactive(args.config))
    elif args.message:
        asyncio.run(cli_message(args.message, args.config))
    else:
        agent = MatrixLLMAgent()
        asyncio.run(agent.run())


if __name__ == "__main__":
    main()
