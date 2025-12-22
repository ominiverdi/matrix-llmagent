"""OSGeo Library search tool with result caching and image support.

This module provides semantic search over scientific documents (PDFs) stored
in the osgeo-library server. Features:
- Search text chunks and visual elements (figures, tables, equations)
- Per-room result caching for follow-up `show N` commands
- Image fetching from the library server
- Guided tour for new users
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

# --- Tour Text ---

# Path to the tour file (relative to this module)
TOUR_FILE_PATH = Path(__file__).parent.parent.parent / "docs" / "LIBRARY_TOUR.md"


def load_tour_text() -> str:
    """Load the library tour text from docs/LIBRARY_TOUR.md."""
    try:
        if TOUR_FILE_PATH.exists():
            return TOUR_FILE_PATH.read_text(encoding="utf-8")
        else:
            logger.warning(f"Tour file not found: {TOUR_FILE_PATH}")
            return "Library tour not available. Try searching with a specific query."
    except Exception as e:
        logger.error(f"Failed to load tour file: {e}")
        return "Library tour not available. Try searching with a specific query."


# --- Cache ---


@dataclass
class CachedResults:
    """Cached search results with timestamp for TTL expiration."""

    results: list[dict]
    timestamp: float
    element_footer: str | None = None  # Footer with element references for hybrid mode


@dataclass
class LastPageView:
    """Tracks the last viewed page for navigation shortcuts."""

    document_slug: str
    document_title: str
    page: int
    total_pages: int
    timestamp: float


class LibraryResultsCache:
    """Per-room cache for library search results with TTL and LRU eviction."""

    def __init__(self, ttl_hours: float = 24, max_rooms: int = 100):
        """Initialize cache.

        Args:
            ttl_hours: Time-to-live in hours for cached results.
            max_rooms: Maximum number of rooms to cache before LRU eviction.
        """
        self.ttl_seconds = ttl_hours * 3600
        self.max_rooms = max_rooms
        self._cache: dict[str, CachedResults] = {}
        self._page_views: dict[str, LastPageView] = {}  # Per-arc page navigation state

    def store(self, room_id: str, results: list[dict], element_footer: str | None = None) -> None:
        """Store search results for a room."""
        # Evict oldest if at capacity and this is a new room
        if room_id not in self._cache and len(self._cache) >= self.max_rooms:
            self._evict_oldest()

        self._cache[room_id] = CachedResults(
            results=results, timestamp=time.time(), element_footer=element_footer
        )
        logger.debug(f"Cached {len(results)} library results for room {room_id}")

    def get(self, room_id: str) -> list[dict] | None:
        """Get cached results for a room. Returns None if not found or expired."""
        entry = self._cache.get(room_id)
        if entry is None:
            return None

        if self._is_expired(entry):
            del self._cache[room_id]
            logger.debug(f"Library cache expired for room {room_id}")
            return None

        return entry.results

    def get_timestamp(self, room_id: str) -> float:
        """Get the timestamp of cached results for a room. Returns 0 if not found."""
        entry = self._cache.get(room_id)
        if entry is None or self._is_expired(entry):
            return 0.0
        return entry.timestamp

    def clear(self, room_id: str) -> None:
        """Clear cached results for a room."""
        if room_id in self._cache:
            del self._cache[room_id]

    def pop_element_footer(self, room_id: str) -> str | None:
        """Get and clear the element footer for a room (one-time use after LLM response)."""
        entry = self._cache.get(room_id)
        if entry is None:
            return None
        footer = entry.element_footer
        entry.element_footer = None  # Clear after retrieval
        return footer

    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry (LRU)."""
        if not self._cache:
            return

        oldest_room = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[oldest_room]
        logger.debug(f"Evicted oldest library cache entry for room {oldest_room}")

    def _is_expired(self, entry: CachedResults) -> bool:
        """Check if a cache entry has expired."""
        return time.time() - entry.timestamp > self.ttl_seconds

    # --- Page Navigation State ---

    def store_page_view(
        self,
        arc: str,
        document_slug: str,
        document_title: str,
        page: int,
        total_pages: int,
    ) -> None:
        """Store the current page view for navigation shortcuts."""
        self._page_views[arc] = LastPageView(
            document_slug=document_slug,
            document_title=document_title,
            page=page,
            total_pages=total_pages,
            timestamp=time.time(),
        )
        logger.debug(f"Stored page view for {arc}: {document_slug} p.{page}/{total_pages}")

    def get_page_view(self, arc: str) -> LastPageView | None:
        """Get the current page view for an arc. Returns None if not found or expired."""
        view = self._page_views.get(arc)
        if view is None:
            return None

        # Use same TTL as search results
        if time.time() - view.timestamp > self.ttl_seconds:
            del self._page_views[arc]
            logger.debug(f"Page view expired for {arc}")
            return None

        return view

    def clear_page_view(self, arc: str) -> None:
        """Clear the page view for an arc."""
        if arc in self._page_views:
            del self._page_views[arc]


# --- Citation Tags ---


def get_citation_tag(result: dict) -> str:
    """Get citation tag for a search result (f, t, eq, tb)."""
    if result.get("source_type") == "element":
        element_type = result.get("element_type", "")
        return {
            "figure": "f",
            "table": "tb",
            "equation": "eq",
            "chart": "ch",
            "diagram": "d",
        }.get(element_type, "e")
    return "t"


def format_sources_list(results: list[dict]) -> str:
    """Format cached search results as a sources list for the golden cord.

    Shows unique (document, page) combinations with content snippets,
    allowing users to view source pages with !source N.

    Args:
        results: Cached search results from library search.

    Returns:
        Formatted string with numbered sources for !source N command.
    """
    if not results:
        return "No sources available. Run a library search first."

    lines = ["Sources from last search:", ""]

    for i, result in enumerate(results, 1):
        doc_title = result.get("document_title", "Unknown")
        if len(doc_title) > 35:
            doc_title = doc_title[:32] + "..."
        page = result.get("page_number", "?")
        content = result.get("content", "").replace("\n", " ").strip()

        # Shorter snippet for sources list
        snippet = content[:80] + "..." if len(content) > 80 else content

        if result.get("source_type") == "element":
            elem_type = result.get("element_type", "element")
            label = result.get("element_label", "")
            lines.append(f"  [{i}] {doc_title} p.{page} - {elem_type}: {label}")
        else:
            lines.append(f'  [{i}] {doc_title} p.{page} - "{snippet}"')

    lines.append("")
    lines.append("View source page: !source N")

    return "\n".join(lines)


# --- Result Formatting ---


def format_results_hybrid(results: list[dict], library_name: str) -> tuple[str, str | None]:
    """Format search results with text for LLM and elements as separate footer.

    This hybrid approach gives the LLM text content to discuss naturally,
    while element references (figures, tables, equations) are extracted
    and returned separately to be appended after the LLM response.

    Args:
        results: List of search result dicts from the library API.
        library_name: Name of the library for the header.

    Returns:
        Tuple of (llm_content, element_footer).
        - llm_content: Text chunks for LLM to discuss
        - element_footer: Element references to append after LLM response, or None
    """
    if not results:
        return f"No results found in {library_name}.", None

    text_lines = [f"**{library_name}** found relevant content:"]
    element_lines = []

    for i, result in enumerate(results, 1):
        page = result.get("page_number", "?")
        doc_title = result.get("document_title", "Unknown")
        if len(doc_title) > 30:
            doc_title = doc_title[:27] + "..."
        content = result.get("content", "").replace("\n", " ").strip()

        if result.get("source_type") == "element":
            # Elements go to footer - user can request with show N
            label = result.get("element_label", "")
            elem_type = result.get("element_type", "element").upper()
            desc = content[:120] + "..." if len(content) > 120 else content
            element_lines.append(f'  show {i}: {elem_type} "{label}" (p.{page}) - {desc}')
        else:
            # Text chunks go to LLM for discussion
            desc = content[:500] + "..." if len(content) > 500 else content
            text_lines.append(f"\nFrom {doc_title}, p.{page}:")
            text_lines.append(f"  {desc}")

    # Build LLM content
    if len(text_lines) == 1:
        # No text results, only elements
        llm_content = f"**{library_name}** found {len(element_lines)} visual elements (figures, tables, equations). Use the `show N` commands below to view them."
    else:
        llm_content = "\n".join(text_lines)

    # Build element footer
    if element_lines:
        element_footer = "Available elements:\n" + "\n".join(element_lines)
    else:
        element_footer = None

    return llm_content, element_footer


def format_results(results: list[dict], library_name: str, *, compact: bool = False) -> str:
    """Format search results with rich descriptions for LLM comprehension.

    The format uses explicit result numbers (1, 2, 3...) that users can reference
    with `show N`. Element labels (e.g., "Figure 10") are kept separate to avoid
    confusion between result indices and element labels.

    Args:
        results: List of search result dicts from the library API.
        library_name: Name of the library for the header.
        compact: If True, use shorter truncation for display (120/150 chars).
                 If False, use longer content for LLM understanding (500 chars).

    Returns:
        Formatted string with numbered results, citation tags, and descriptions.
    """
    if not results:
        return f"No results found in {library_name}."

    # Truncation limits: compact for user display, longer for LLM
    element_limit = 120 if compact else 500
    text_limit = 150 if compact else 500

    lines = [f"**{library_name}** ({len(results)} results)"]
    lines.append(f"Results numbered 1-{len(results)} for `show N` command:")

    has_elements = False
    for i, result in enumerate(results, 1):
        tag = get_citation_tag(result)
        page = result.get("page_number", "?")
        doc_title = result.get("document_title", "Unknown")
        # Shorten doc title if too long
        if len(doc_title) > 30:
            doc_title = doc_title[:27] + "..."
        content = result.get("content", "").replace("\n", " ").strip()

        if result.get("source_type") == "element":
            has_elements = True
            label = result.get("element_label", "")
            elem_type = result.get("element_type", "element")
            desc = content[:element_limit] + "..." if len(content) > element_limit else content
            # Format emphasizes result number for LLM accuracy
            lines.append(f'[RESULT #{i}] {elem_type.upper()} "{label}" (p.{page}, {doc_title})')
            if desc:
                lines.append(f"    {desc}")
        else:
            desc = content[:text_limit] + "..." if len(content) > text_limit else content
            lines.append(f"[RESULT #{i}] TEXT p.{page} ({doc_title})")
            lines.append(f"    {desc}")

    if has_elements:
        lines.append("")
        lines.append(
            f"IMPORTANT: Use result numbers (1-{len(results)}) with `show N`, not element labels."
        )

    return "\n".join(lines)


# --- Executor ---


class LibrarySearchExecutor:
    """Search the OSGeo document library.

    This executor calls the osgeo-library server's /search endpoint and
    returns formatted results. Results are cached per-room to support
    the follow-up `show N` command.
    """

    def __init__(
        self,
        base_url: str,
        cache: LibraryResultsCache,
        arc: str = "",
        name: str = "OSGeo Library",
        description: str = "",
        max_results: int = 10,
        timeout: int = 30,
    ):
        """Initialize the library search executor.

        Args:
            base_url: Base URL of the osgeo-library server (e.g., http://localhost:8095).
            cache: Shared cache instance for storing results.
            arc: Arc identifier (e.g., "matrix#!roomid" or "cli#test") for cache key.
            name: Display name of the library.
            description: Description for tool definition.
            max_results: Default maximum results to return.
            timeout: HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.arc = arc
        self.name = name
        self.description = description
        self.max_results = max_results
        self.timeout = timeout

    async def execute(
        self,
        query: str | None = None,
        mode: str | None = None,
        document_slug: str | None = None,
        element_type: str | None = None,
        limit: int | None = None,
        page_number: int | None = None,
    ) -> str | list[dict]:
        """Execute a library search, fetch a page, or return the guided tour.

        Args:
            query: Search query string. Also used for document name matching when page_number is set.
            mode: Operation mode - 'search' (default) or 'tour' for guided tour.
            document_slug: Filter to specific document (optional).
            element_type: Filter by element type: figure, table, equation (optional).
            limit: Maximum number of results (optional, uses max_results default).
            page_number: If set, fetch this page instead of searching (1-indexed).

        Returns:
            Formatted string with search results, tour text, or list with image content block for pages.
        """
        # Handle tour mode
        if mode == "tour":
            logger.info(f"Returning library tour for arc {self.arc}")
            return load_tour_text()

        # Handle page request
        if page_number is not None:
            return await self._execute_page_request(query or "", document_slug, page_number)

        # For search mode, query is required
        if not query:
            return "Please provide a search query, or use mode='tour' for a guided tour."

        # Regular search
        limit = limit or self.max_results

        # Build request payload
        payload = {
            "query": query,
            "limit": limit,
            "include_chunks": True,
            "include_elements": True,
        }
        if document_slug:
            payload["document_slug"] = document_slug
        if element_type:
            payload["element_type"] = element_type
            # If filtering by element type, don't include chunks
            payload["include_chunks"] = False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/search",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Library search failed: {response.status} - {error_text}")
                        return f"Library search failed (HTTP {response.status}). The library server may be experiencing issues."

                    data = await response.json()

        except aiohttp.ClientConnectorError as e:
            logger.error(f"Cannot connect to library server at {self.base_url}: {e}")
            return f"Cannot connect to {self.name} server at {self.base_url}. Please notify the administrator."
        except TimeoutError:
            logger.error(f"Library search timed out after {self.timeout}s")
            return "Library search timed out. Please try again or notify the administrator."
        except Exception as e:
            logger.error(f"Library search error: {e}")
            return f"Library search error: {e}"

        results = data.get("results", [])

        # Use hybrid formatting: text for LLM, elements as footer
        llm_content, element_footer = format_results_hybrid(results, self.name)

        # Cache results and footer for show command
        if self.arc and results:
            self.cache.store(self.arc, results, element_footer)

        return llm_content

    async def _execute_page_request(
        self,
        query: str,
        document_slug: str | None,
        page_number: int,
    ) -> str | list[dict]:
        """Fetch a specific page from a document.

        Args:
            query: Document name query (used if document_slug not provided).
            document_slug: Direct document slug (preferred if known).
            page_number: Page number to fetch (1-indexed).

        Returns:
            List with image content block for the model, or error string.
        """
        import base64

        # Resolve document slug if not provided
        if not document_slug:
            docs = await search_documents(self.base_url, query, limit=5)
            if isinstance(docs, str):
                return docs  # Error message

            if not docs:
                return f"No documents found matching '{query}'."

            if len(docs) == 1:
                document_slug = docs[0]["slug"]
            else:
                # Multiple matches - ask user to clarify
                doc_list = "\n".join(
                    f"  - {d['slug']}: {d['title']} ({d['total_pages']} pages)" for d in docs
                )
                return (
                    f"Multiple documents match '{query}'. Please specify document_slug:\n{doc_list}"
                )

        # At this point document_slug is guaranteed to be set
        assert document_slug is not None

        # Fetch the page
        result = await fetch_library_page(self.base_url, document_slug, page_number)

        if isinstance(result, str):
            return result  # Error message

        # Store page view for navigation
        if self.arc:
            self.cache.store_page_view(
                self.arc,
                result.document_slug,
                result.document_title,
                result.page_number,
                result.total_pages,
            )

        # Return image as content block for the model (Anthropic format)
        image_b64 = base64.b64encode(result.image_data).decode("utf-8")

        return [
            {
                "type": "text",
                "text": f"Page {result.page_number} of {result.total_pages} from '{result.document_title}'. "
                f"User can say 'next page' or 'prev page' to navigate, or use !next/!prev shortcuts.",
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": result.image_mimetype,
                    "data": image_b64,
                },
            },
        ]


# --- Tool Definition ---


def library_search_tool_def(name: str, description: str) -> dict:
    """Generate library_search tool definition.

    Args:
        name: Display name of the library.
        description: Description of what the library contains.

    Returns:
        Tool definition dict for the agentic actor.
    """
    return {
        "name": "library_search",
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"Search query for {name}. Use specific terms related to the document content. Also used to find documents by name when page_number is specified. Required for 'search' mode, ignored for 'tour' mode.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["search", "tour"],
                    "description": "Mode of operation. 'search' (default): search the library. 'tour': return a guided tour explaining how to use the library.",
                },
                "document_slug": {
                    "type": "string",
                    "description": "Optional: filter results to a specific document by its slug identifier.",
                },
                "element_type": {
                    "type": "string",
                    "enum": ["figure", "table", "equation"],
                    "description": "Optional: filter results to a specific element type.",
                },
                "page_number": {
                    "type": "integer",
                    "description": "Optional: fetch a full page image instead of searching. Use with document_slug or query to identify the document. Page numbers are 1-indexed.",
                },
            },
            "required": [],
        },
        "persist": "summary",
    }


# --- Image Fetching ---


async def fetch_library_image(
    base_url: str,
    document_slug: str,
    image_path: str,
    timeout: int = 30,
) -> tuple[bytes, str] | None:
    """Fetch an image from the library server.

    Args:
        base_url: Base URL of the osgeo-library server.
        document_slug: Document identifier.
        image_path: Path to the image (e.g., "elements/p51_figure_1.png").
        timeout: HTTP request timeout in seconds.

    Returns:
        Tuple of (image_bytes, mimetype) or None if fetch failed.
    """
    url = f"{base_url.rstrip('/')}/image/{document_slug}/{image_path}"

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response,
        ):
            if response.status != 200:
                logger.warning(f"Failed to fetch library image: {response.status} - {url}")
                return None

            content_type = response.headers.get("content-type", "image/png")
            image_data = await response.read()

            logger.debug(f"Fetched library image: {url} ({len(image_data)} bytes)")
            return (image_data, content_type)

    except aiohttp.ClientConnectorError as e:
        logger.error(f"Cannot connect to library server for image: {e}")
        return None
    except TimeoutError:
        logger.error(f"Library image fetch timed out: {url}")
        return None
    except Exception as e:
        logger.error(f"Library image fetch error: {e}")
        return None


def get_best_image_path(result: dict) -> str | None:
    """Get the best image path for a search result.

    For equations, prefers rendered_path (clean LaTeX) over crop_path (raw PDF crop).

    Args:
        result: Search result dict.

    Returns:
        Image path string or None if no image available.
    """
    if result.get("source_type") != "element":
        return None

    # For equations, prefer rendered version if available
    if result.get("element_type") == "equation":
        rendered = result.get("rendered_path")
        if rendered:
            return rendered

    return result.get("crop_path") or None


# --- Document Search ---


async def search_documents(
    base_url: str,
    query: str,
    limit: int = 20,
    timeout: int = 30,
) -> list[dict] | str:
    """Search for documents by title/slug/filename.

    Args:
        base_url: Base URL of the osgeo-library server.
        query: Search query (partial match on title, slug, or filename).
        limit: Maximum number of results.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of document dicts with slug, title, source_file, total_pages.
        Returns error string if request failed.
    """
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{base_url.rstrip('/')}/documents/search",
                json={"query": query, "limit": limit},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response,
        ):
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Document search failed: {response.status} - {error_text}")
                return f"Document search failed (HTTP {response.status})."

            data = await response.json()
            return data.get("results", [])

    except aiohttp.ClientConnectorError as e:
        logger.error(f"Cannot connect to library server for document search: {e}")
        return "Cannot connect to library server."
    except TimeoutError:
        logger.error(f"Document search timed out after {timeout}s")
        return "Document search timed out."
    except Exception as e:
        logger.error(f"Document search error: {e}")
        return f"Document search error: {e}"


# --- Page Fetching ---


@dataclass
class PageResult:
    """Result from fetching a document page."""

    document_slug: str
    document_title: str
    page_number: int
    total_pages: int
    image_data: bytes
    image_mimetype: str
    image_width: int
    image_height: int


async def fetch_library_page(
    base_url: str,
    document_slug: str,
    page_number: int,
    timeout: int = 60,
) -> PageResult | str:
    """Fetch a full page image from the library server.

    Args:
        base_url: Base URL of the osgeo-library server.
        document_slug: Document identifier.
        page_number: Page number (1-indexed).
        timeout: HTTP request timeout in seconds (higher for large pages).

    Returns:
        PageResult with image data and metadata, or error string.
    """
    import base64

    url = f"{base_url.rstrip('/')}/page/{document_slug}/{page_number}"

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response,
        ):
            if response.status == 404:
                data = await response.json()
                error_msg = data.get("message", data.get("detail", "Not found"))
                return error_msg

            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Failed to fetch page: {response.status} - {error_text}")
                return f"Failed to fetch page (HTTP {response.status})."

            data = await response.json()

            # Decode base64 image data (API returns "image_base64")
            image_b64 = data.get("image_base64", data.get("image_data", ""))
            if not image_b64:
                return "Page response missing image data."

            image_bytes = base64.b64decode(image_b64)

            return PageResult(
                document_slug=data.get("document_slug", document_slug),
                document_title=data.get("document_title", "Unknown"),
                page_number=data.get("page_number", page_number),
                total_pages=data.get("total_pages", 0),
                image_data=image_bytes,
                image_mimetype=data.get("mime_type", data.get("image_mimetype", "image/png")),
                image_width=data.get("image_width", 0),
                image_height=data.get("image_height", 0),
            )

    except aiohttp.ClientConnectorError as e:
        logger.error(f"Cannot connect to library server for page: {e}")
        return "Cannot connect to library server."
    except TimeoutError:
        logger.error(f"Page fetch timed out after {timeout}s")
        return "Page fetch timed out. Try again."
    except Exception as e:
        logger.error(f"Page fetch error: {e}")
        return f"Page fetch error: {e}"


# --- Direct Search (for !l mode) ---


async def search_library_direct(
    base_url: str,
    query: str,
    cache: LibraryResultsCache | None = None,
    arc: str = "",
    name: str = "OSGeo Library",
    limit: int = 10,
    timeout: int = 30,
) -> tuple[list[dict], str]:
    """Search the library directly without LLM intermediation.

    Used by the !l command mode for fast, direct library search.

    Args:
        base_url: Base URL of the osgeo-library server.
        query: Search query string.
        cache: Optional cache instance to store results for show command.
        arc: Arc identifier for cache key.
        name: Display name of the library.
        limit: Maximum number of results.
        timeout: HTTP request timeout in seconds.

    Returns:
        Tuple of (raw results list, formatted string for display).
    """
    payload = {
        "query": query,
        "limit": limit,
        "include_chunks": True,
        "include_elements": True,
    }

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{base_url.rstrip('/')}/search",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response,
        ):
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Library search failed: {response.status} - {error_text}")
                return [], f"Library search failed (HTTP {response.status})."

            data = await response.json()

    except aiohttp.ClientConnectorError as e:
        logger.error(f"Cannot connect to library server at {base_url}: {e}")
        return [], f"Cannot connect to {name} server. Is it running?"
    except TimeoutError:
        logger.error(f"Library search timed out after {timeout}s")
        return [], "Library search timed out."
    except Exception as e:
        logger.error(f"Library search error: {e}")
        return [], f"Library search error: {e}"

    results = data.get("results", [])

    # Cache results for show command
    if cache and arc and results:
        cache.store(arc, results)

    # Use compact format for direct user display (!l command)
    formatted = format_results(results, name, compact=True)
    return results, formatted
