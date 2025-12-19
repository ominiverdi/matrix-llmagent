"""OSGeo Library search tool with result caching and image support.

This module provides semantic search over scientific documents (PDFs) stored
in the osgeo-library server. Features:
- Search text chunks and visual elements (figures, tables, equations)
- Per-room result caching for follow-up `show N` commands
- Image fetching from the library server
"""

import logging
import time
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


# --- Cache ---


@dataclass
class CachedResults:
    """Cached search results with timestamp for TTL expiration."""

    results: list[dict]
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

    def store(self, room_id: str, results: list[dict]) -> None:
        """Store search results for a room."""
        # Evict oldest if at capacity and this is a new room
        if room_id not in self._cache and len(self._cache) >= self.max_rooms:
            self._evict_oldest()

        self._cache[room_id] = CachedResults(results=results, timestamp=time.time())
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

    def clear(self, room_id: str) -> None:
        """Clear cached results for a room."""
        if room_id in self._cache:
            del self._cache[room_id]

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


# --- Result Formatting ---


def format_results(results: list[dict], library_name: str) -> str:
    """Format search results for LLM consumption.

    Args:
        results: List of search result dicts from the library API.
        library_name: Name of the library for the header.

    Returns:
        Formatted string with numbered results and citation tags.
    """
    if not results:
        return f"No results found in {library_name}."

    lines = [f"## {library_name} Results ({len(results)} found)", ""]

    has_elements = False
    for i, result in enumerate(results, 1):
        tag = get_citation_tag(result)
        score = result.get("score_pct", 0)
        page = result.get("page_number", "?")
        doc_title = result.get("document_title", "Unknown")
        content = result.get("content", "")[:200].strip()

        if result.get("source_type") == "element":
            has_elements = True
            elem_type = result.get("element_type", "element").upper()
            label = result.get("element_label", "")
            lines.append(f"[{tag}:{i}] {elem_type}: {label} ({score:.0f}%, p.{page}, {doc_title})")
        else:
            chunk_idx = result.get("chunk_index", 0)
            lines.append(
                f"[{tag}:{i}] TEXT chunk {chunk_idx} ({score:.0f}%, p.{page}, {doc_title})"
            )

        if content:
            # Indent content preview
            lines.append(f"  {content}...")
        lines.append("")

    # Add hint about show command if there are elements
    if has_elements:
        lines.append("---")
        lines.append("Use `show N` or `show 1,2,3` to view element images or text content.")

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
        query: str,
        document_slug: str | None = None,
        element_type: str | None = None,
        limit: int | None = None,
    ) -> str:
        """Execute a library search.

        Args:
            query: Search query string.
            document_slug: Filter to specific document (optional).
            element_type: Filter by element type: figure, table, equation (optional).
            limit: Maximum number of results (optional, uses max_results default).

        Returns:
            Formatted string with search results.
        """
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

        # Cache results for show command using arc as key
        if self.arc and results:
            self.cache.store(self.arc, results)

        return format_results(results, self.name)


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
                    "description": f"Search query for {name}. Use specific terms related to the document content.",
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
            },
            "required": ["query"],
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

    formatted = format_results(results, name)
    return results, formatted
