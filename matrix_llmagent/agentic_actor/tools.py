"""Tool definitions and executors for AI agent."""

import asyncio
import base64
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, TypedDict

import aiohttp
from ddgs import DDGS

from ..chronicler.tools import ChapterAppendExecutor, ChapterRenderExecutor, chronicle_tools_defs

logger = logging.getLogger(__name__)


DEFAULT_USER_AGENT = "matrix-llmagent/1.0"


async def fetch_image_b64(
    session: aiohttp.ClientSession,
    url: str,
    max_size: int,
    timeout: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
) -> tuple[str, str]:
    """
    Fetch an image from URL and return (content_type, base64_string).
    Raises ValueError if not an image, too large, or fetch fails.
    """
    async with session.head(
        url,
        timeout=aiohttp.ClientTimeout(total=timeout),
        headers={"User-Agent": user_agent},
    ) as head_response:
        content_type = head_response.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            raise ValueError(f"URL is not an image (content-type: {content_type})")

    async with session.get(
        url,
        timeout=aiohttp.ClientTimeout(total=timeout),
        headers={"User-Agent": user_agent},
        max_line_size=8190 * 2,
        max_field_size=8190 * 2,
    ) as response:
        response.raise_for_status()

        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > max_size:
            raise ValueError(
                f"Image too large ({content_length} bytes). Maximum allowed: {max_size} bytes"
            )

        image_data = await response.read()
        if len(image_data) > max_size:
            raise ValueError(
                f"Image too large ({len(image_data)} bytes). Maximum allowed: {max_size} bytes"
            )

        image_b64 = base64.b64encode(image_data).decode()
        logger.info(
            f"Downloaded image from {url}, content-type: {content_type}, size: {len(image_data)} bytes"
        )
        return (content_type, image_b64)


class Tool(TypedDict):
    """Tool definition schema."""

    name: str
    description: str
    input_schema: dict
    persist: str  # "none", "exact", "summary", or "artifact"


# Available tools for AI agent
TOOLS: list[Tool] = [
    {
        "name": "web_search",
        "description": "Search the web and return top results with titles, URLs, and descriptions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to perform. Never use \\u unicode escapes.",
                }
            },
            "required": ["query"],
        },
        "persist": "summary",
    },
    {
        "name": "visit_webpage",
        "description": "Visit the given URL and return its content as markdown text if HTML website, or picture if an image URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "The URL of the webpage to visit.",
                }
            },
            "required": ["url"],
        },
        "persist": "summary",
    },
    {
        "name": "execute_python",
        "description": "Execute Python code in a sandbox environment and return the output. The sandbox environment is persisted to follow-up calls of this tool within this thread.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute in the sandbox.",
                }
            },
            "required": ["code"],
        },
        "persist": "artifact",
    },
    {
        "name": "progress_report",
        "description": "Send a brief one-line progress update to the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "One-line progress update. Keep it super concise, but very casual and even snarky in line with your instructions and previous conversation.",
                }
            },
            "required": ["text"],
        },
        "persist": "none",
    },
    {
        "name": "final_answer",
        "description": "Provide the final answer to the user's question. This tool MUST be used when the agent is ready to give its final response.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer or response to the user's question. Start with final deliberation in <thinking>...</thinking>. Never say 'you are doing something' or 'you will do something' - at this point, you are *done*.",
                }
            },
            "required": ["answer"],
        },
        "persist": "none",
    },
    {
        "name": "make_plan",
        "description": "Consider different approaches and formulate a brief research and/or execution plan. Only use this tool if research or code execution seems necessary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "A brief research and/or execution plan to handle the user's request, outlining (a) concerns and key goals that require further actions before responding, (b) the key steps and approach how to address them.",
                }
            },
            "required": ["plan"],
        },
        "persist": "none",
    },
    {
        "name": "share_artifact",
        "description": "Share an artifact (additional command output - created script, detailed report, supporting data) with the user. The content is made available on a public link that is returned by the tool. Use this only for additional content that doesn't fit into your standard IRC message response (or when explicitly requested).",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content of the artifact to share (script, report, detailed data, etc.).",
                }
            },
            "required": ["content"],
        },
        "persist": "none",
    },
    {
        "name": "generate_image",
        "description": "Generate image(s) using {tools.image_gen.model} on OpenRouter. Optionally provide reference image URLs for editing/variations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate.",
                },
                "image_urls": {
                    "type": "array",
                    "items": {"type": "string", "format": "uri"},
                    "description": "Optional list of reference image URLs to include as input for editing or creating variations.",
                },
            },
            "required": ["prompt"],
        },
        "persist": "artifact",
    },
]

# Add chronicle tools to the main tools list
TOOLS.extend(chronicle_tools_defs())  # type: ignore


def knowledge_base_tool_def(name: str, description: str) -> Tool:
    """Generate knowledge_base tool definition with configured name and description."""
    return {
        "name": "knowledge_base",
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        f"Search query for {name}. Use the EXACT terms from the user's question. "
                        "Do NOT modify, correct, or assume what the user meant - search for what "
                        "they actually wrote first."
                    ),
                }
            },
            "required": ["query"],
        },
        "persist": "summary",
    }


def _build_predicate_description(predicate_hints: dict[str, dict] | None) -> str:
    """Build predicate description from config hints.

    Args:
        predicate_hints: Optional dict of predicate -> {description, hint, filter} from config

    Returns:
        Formatted description string for the predicate parameter
    """
    # Default generic predicates (always included as base)
    base_description = (
        "The predicate name (relationship type). Common predicates:\n"
        "- is_president_of, was_president_of (filter: object=OrgName)\n"
        "- is_member_of, is_board_member_of, is_project_of, is_chapter_of\n"
        "- founded_by, founded_in (filter: subject=OrgName)\n"
        "- located_in, happened_in (filter: subject=EntityName)"
    )

    if not predicate_hints:
        return base_description

    # Add domain-specific hints from config
    hint_lines = []
    for predicate, info in predicate_hints.items():
        if isinstance(info, dict):
            desc = info.get("description", "")
            hint = info.get("hint", "")
            if desc and hint:
                hint_lines.append(f"- {predicate}: {desc} ({hint})")
            elif desc:
                hint_lines.append(f"- {predicate}: {desc}")
            elif hint:
                hint_lines.append(f"- {predicate} ({hint})")
            else:
                hint_lines.append(f"- {predicate}")

    if hint_lines:
        return base_description + "\nDomain-specific:\n" + "\n".join(hint_lines)

    return base_description


def relationship_search_tool_def(name: str, predicate_hints: dict[str, dict] | None = None) -> Tool:
    """Generate relationship_search tool definition for querying entity relationships.

    Args:
        name: Knowledge base name for description
        predicate_hints: Optional dict of predicate hints from config
    """
    return {
        "name": "relationship_search",
        "description": (
            f"Search relationships between entities in {name}. "
            "Relationships are stored as: subject -> predicate -> object. "
            "IMPORTANT: Always filter by subject or object to get relevant results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "predicate": {
                    "type": "string",
                    "description": _build_predicate_description(predicate_hints),
                },
                "subject": {
                    "type": "string",
                    "description": (
                        "Filter by left side of relationship. "
                        "Use for: 'who founded X?' -> subject=X, predicate=founded_by"
                    ),
                },
                "object": {
                    "type": "string",
                    "description": (
                        "Filter by right side of relationship. "
                        "Use for: 'who is president of X?' -> object=X, predicate=is_president_of"
                    ),
                },
            },
            "required": ["predicate"],
        },
        "persist": "summary",
    }


def entity_info_tool_def(name: str) -> Tool:
    """Generate entity_info tool definition for exploring entities and their relationships."""
    return {
        "name": "entity_info",
        "description": (
            f"Get information about entities in {name}. "
            "Returns entity type and all relationships for matching entities. "
            "Supports batch queries: use match='prefix' with a common prefix "
            "to get all matching entities at once. "
            "Use this to explore what data exists for a topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "Entity name or pattern to search for. "
                        "For batch queries, use a common prefix."
                    ),
                },
                "match": {
                    "type": "string",
                    "enum": ["exact", "prefix", "contains"],
                    "description": (
                        "Match type: 'exact' for exact name, "
                        "'prefix' for starts-with (RECOMMENDED for batch queries), "
                        "'contains' for substring match. Default: 'contains'."
                    ),
                },
                "predicates": {
                    "type": "string",
                    "description": (
                        "Optional: comma-separated list of predicates to filter results. "
                        "Example: 'located_in,happened_in' to only show locations and years. "
                        "If not specified, returns all relationships."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max entities to return (default: 50, max: 100).",
                },
            },
            "required": ["name"],
        },
        "persist": "summary",
    }


class RateLimiter:
    """Simple rate limiter for tool calls."""

    def __init__(self, max_calls_per_second: float = 1.0):
        self.max_calls_per_second = max_calls_per_second
        self.min_interval = 1.0 / max_calls_per_second if max_calls_per_second > 0 else 0.0
        self.last_call_time = 0.0

    async def wait_if_needed(self):
        """Wait if needed to respect rate limit."""
        if self.min_interval <= 0:
            return

        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            await asyncio.sleep(wait_time)
        self.last_call_time = time.time()


class BraveSearchExecutor:
    """Async Brave Search API executor."""

    def __init__(self, api_key: str, max_results: int = 5, max_calls_per_second: float = 1.0):
        self.api_key = api_key
        self.max_results = max_results
        self.rate_limiter = RateLimiter(max_calls_per_second)

    async def execute(self, query: str) -> str:
        """Execute Brave search and return formatted results."""
        await self.rate_limiter.wait_if_needed()

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": self.max_results,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    results = data.get("web", {}).get("results", [])
                    logger.info(f"Brave searching '{query}': {len(results)} results")

                    if not results:
                        return "No search results found. Try a different query."

                    # Format results as markdown
                    formatted_results = []
                    for result in results:
                        title = result.get("title", "No title")
                        url = result.get("url", "#")
                        description = result.get("description", "No description")
                        formatted_results.append(f"[{title}]({url})\n{description}")

                    return "## Search Results\n\n" + "\n\n".join(formatted_results)

            except Exception as e:
                logger.error(f"Brave search failed: {e}")
                return f"Search failed: {e}"


class WebSearchExecutor:
    """Async ddgs web search executor."""

    def __init__(
        self, max_results: int = 10, max_calls_per_second: float = 1.0, backend: str = "auto"
    ):
        self.max_results = max_results
        self.rate_limiter = RateLimiter(max_calls_per_second)
        self.backend = backend

    async def execute(self, query: str) -> str:
        """Execute web search and return formatted results."""
        await self.rate_limiter.wait_if_needed()

        # Note: DDGS is not async, so we run it in executor
        loop = asyncio.get_event_loop()
        with DDGS() as ddgs:
            results = await loop.run_in_executor(
                None,
                lambda: list(ddgs.text(query, max_results=self.max_results, backend=self.backend)),
            )

        logger.info(f"Searching '{query}': {len(results)} results")

        if not results:
            return "No search results found. Try a different query."

        # Format results as markdown
        formatted_results = []
        for result in results:
            title = result.get("title", "No title")
            url = result.get("href", "#")
            body = result.get("body", "No description")
            formatted_results.append(f"[{title}]({url})\n{body}")

        return "## Search Results\n\n" + "\n\n".join(formatted_results)


class GoogleSearchExecutor:
    """Async Google Custom Search API executor.

    Requires a Google API key and Custom Search Engine ID (cx).
    Get credentials at: https://developers.google.com/custom-search/v1/overview
    """

    def __init__(
        self,
        api_key: str,
        cx: str,
        max_results: int = 10,
        max_calls_per_second: float = 1.0,
    ):
        self.api_key = api_key
        self.cx = cx
        self.max_results = max_results
        self.rate_limiter = RateLimiter(max_calls_per_second)

    async def execute(self, query: str) -> str:
        """Execute Google search and return formatted results."""
        await self.rate_limiter.wait_if_needed()

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
            "num": min(self.max_results, 10),  # Google API max is 10 per request
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    items = data.get("items", [])
                    logger.info(f"Google searching '{query}': {len(items)} results")

                    if not items:
                        return "No search results found. Try a different query."

                    # Format results as markdown
                    formatted_results = []
                    for item in items:
                        title = item.get("title", "No title")
                        link = item.get("link", "#")
                        snippet = item.get("snippet", "No description")
                        formatted_results.append(f"[{title}]({link})\n{snippet}")

                    return "## Search Results\n\n" + "\n\n".join(formatted_results)

            except aiohttp.ClientResponseError as e:
                logger.error(f"Google search failed: {e}")
                if e.status == 403:
                    return "Search failed: API quota exceeded or invalid credentials"
                return f"Search failed: {e}"
            except Exception as e:
                logger.error(f"Google search failed: {e}")
                return f"Search failed: {e}"


class JinaSearchExecutor:
    """Async Jina.ai search executor."""

    def __init__(
        self,
        max_results: int = 10,
        max_calls_per_second: float = 1.0,
        api_key: str | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self.max_results = max_results
        self.rate_limiter = RateLimiter(max_calls_per_second)
        self.api_key = api_key
        self.user_agent = user_agent

    async def execute(self, query: str, **kwargs) -> str:
        """Execute Jina search and return formatted results."""
        await self.rate_limiter.wait_if_needed()

        warning_prefix = ""
        if kwargs:
            logger.warning(f"JinaSearchExecutor received unsupported arguments: {kwargs}")
            warning_prefix = (
                f"Warning: The following parameters were ignored: {', '.join(kwargs.keys())}\n\n"
            )

        url = "https://s.jina.ai/?q=" + query
        headers = {
            "User-Agent": self.user_agent,
            "X-Respond-With": "no-content",
            "Accept": "text/plain",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    response.raise_for_status()
                    content = await response.text()

                    logger.info(f"Jina searching '{query}': retrieved search results")

                    if not content.strip():
                        return f"{warning_prefix}No search results found. Try a different query."

                    return f"{warning_prefix}## Search Results\n\n{content.strip()}"

            except Exception as e:
                logger.error(f"Jina search failed: {e}")
                return f"{warning_prefix}Search failed: {e}"


class LocalWebpageVisitor:
    """Local webpage visitor that converts HTML to Markdown without external services."""

    def __init__(
        self,
        max_content_length: int = 40000,
        timeout: int = 60,
        max_image_size: int = 3_500_000,
        progress_callback: Any | None = None,
        user_agent: str | None = None,
    ):
        self.max_content_length = max_content_length
        self.timeout = timeout
        self.max_image_size = max_image_size
        self.progress_callback = progress_callback
        self.user_agent = (
            user_agent
            or "Mozilla/5.0 (compatible; matrix-llmagent/1.0; +https://github.com/matrix-llmagent)"
        )

    async def execute(self, url: str) -> str | list[dict]:
        """Visit webpage and return content as markdown, or image data for images."""
        logger.info(f"Visiting {url} (local mode)")

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL. Must start with http:// or https://")

        async with aiohttp.ClientSession() as session:
            # First, check the original URL for content-type to detect images
            try:
                async with session.head(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers={"User-Agent": self.user_agent},
                ) as head_response:
                    content_type = head_response.headers.get("content-type", "").lower()

                    if content_type.startswith("image/"):
                        try:
                            content_type, image_b64 = await fetch_image_b64(
                                session, url, self.max_image_size, self.timeout
                            )
                            # Return Anthropic content blocks with image
                            return [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": content_type,
                                        "data": image_b64,
                                    },
                                }
                            ]
                        except ValueError as e:
                            return f"Error: {e}"
            except Exception as e:
                # HEAD request failed, continue with GET
                logger.debug(f"HEAD request failed for {url}: {e}")

            # Fetch HTML content
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers={"User-Agent": self.user_agent},
                ) as response:
                    response.raise_for_status()
                    html = await response.text()

                    # Extract main content using readability
                    try:
                        from markdownify import markdownify as md
                        from readability import Document

                        doc = Document(html)
                        title = doc.title()
                        clean_html = doc.summary()

                        # Convert to markdown
                        markdown_content = md(
                            clean_html, heading_style="ATX", strip=["script", "style"]
                        )

                        # Add title if available
                        if title:
                            markdown_content = f"# {title}\n\n{markdown_content}"

                        # Clean up multiple line breaks
                        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

                        # Truncate if too long
                        if len(markdown_content) > self.max_content_length:
                            truncated_content = markdown_content[: self.max_content_length - 100]
                            markdown_content = truncated_content + "\n\n..._Content truncated_..."
                            logger.warning(
                                f"{url} truncated from {len(markdown_content)} to {len(truncated_content)}"
                            )

                        return f"## Content from {url}\n\n{markdown_content}"

                    except ImportError as e:
                        logger.error(f"Missing dependencies for local webpage visitor: {e}")
                        return "Error: Local webpage visitor requires readability-lxml and markdownify. Please install dependencies."
                    except Exception as e:
                        logger.error(f"Error extracting content from {url}: {e}")
                        return f"Error extracting content: {e}"

            except aiohttp.ClientError as e:
                logger.error(f"Error fetching {url}: {e}")
                return f"Error fetching URL: {e}"


class WebpageVisitorExecutor:
    """Async webpage visitor and content extractor using Jina.ai service."""

    def __init__(
        self,
        max_content_length: int = 40000,
        timeout: int = 60,
        max_image_size: int = 3_500_000,
        progress_callback: Any | None = None,
        api_key: str | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self.max_content_length = max_content_length
        self.timeout = timeout
        self.max_image_size = max_image_size  # 5MB default limit post base64 encode
        self.progress_callback = progress_callback
        self.api_key = api_key
        self.user_agent = user_agent

    async def execute(self, url: str) -> str | list[dict]:
        """Visit webpage and return content as markdown, or image data for images."""
        logger.info(f"Visiting {url}")

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL. Must start with http:// or https://")

        async with aiohttp.ClientSession() as session:
            # First, check the original URL for content-type to detect images
            async with session.head(
                url,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"User-Agent": self.user_agent},
            ) as head_response:
                content_type = head_response.headers.get("content-type", "").lower()

                if content_type.startswith("image/"):
                    try:
                        content_type, image_b64 = await fetch_image_b64(
                            session, url, self.max_image_size, self.timeout, self.user_agent
                        )
                        # Return Anthropic content blocks with image
                        return [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": content_type,
                                    "data": image_b64,
                                },
                            }
                        ]
                    except ValueError as e:
                        return f"Error: {e}"

            # Handle text/HTML content - use jina.ai reader service
            jina_url = f"https://r.jina.ai/{url}"
            markdown_content = await self._fetch(session, jina_url)

        # Clean up multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        # Truncate if too long
        if len(markdown_content) > self.max_content_length:
            truncated_content = markdown_content[
                : self.max_content_length - 100
            ]  # Leave room for message
            markdown_content = truncated_content + "\n\n..._Content truncated_..."
            logger.warning(
                f"{url} truncated from {len(markdown_content)} to {len(truncated_content)}"
            )

        return f"## Content from {url}\n\n{markdown_content}"

    async def _fetch(self, session: aiohttp.ClientSession, jina_url: str) -> str:
        """Fetch from jina.ai with backoff on HTTP 451."""
        backoff_delays = [0, 30, 90]  # No delay, then 30s, then 90s

        for attempt, delay in enumerate(backoff_delays):
            if delay > 0:
                logger.info(f"Waiting {delay}s before retry {attempt + 1}/3 for jina.ai")
                await asyncio.sleep(delay)

            try:
                headers = {"User-Agent": self.user_agent}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                async with session.get(
                    jina_url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    content = await response.text()
                    return content.strip()

            except aiohttp.ClientResponseError as e:
                if (e.status == 451 or e.status >= 500) and attempt < len(backoff_delays) - 1:
                    # Only send error info on second failure (attempt 1) to reduce spam
                    if self.progress_callback and attempt == 1:
                        await self.progress_callback(
                            f"r.jina.ai HTTP {e.status}, retrying in a bit...", "progress"
                        )
                    continue
                raise

        raise RuntimeError("This should not be reached")


class PythonExecutorE2B:
    """Python code executor using E2B sandbox."""

    def __init__(self, api_key: str | None = None, timeout: int = 180):
        self.api_key = api_key
        self.timeout = timeout
        self.sandbox = None

    async def _ensure_sandbox(self):
        """Ensure sandbox is created and connected."""
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError:
            raise ImportError(
                "e2b-code-interpreter package not installed. Install with: pip install e2b-code-interpreter"
            ) from None

        if self.sandbox is None:
            import asyncio

            def create_sandbox():
                from typing import Any

                sandbox_args: dict[str, Any] = {"timeout": self.timeout}
                if self.api_key:
                    sandbox_args["api_key"] = self.api_key
                sandbox = Sandbox(**sandbox_args)
                return sandbox

            self.sandbox = await asyncio.to_thread(create_sandbox)
            logger.info(f"Created new E2B sandbox with ID: {self.sandbox.sandbox_id}")

    async def execute(self, code: str) -> str:
        """Execute Python code in E2B sandbox and return output."""
        try:
            await self._ensure_sandbox()
        except ImportError as e:
            return str(e)

        try:
            import asyncio

            assert self.sandbox is not None
            result = await asyncio.to_thread(self.sandbox.run_code, code)
            logger.debug(result)

            # Collect output
            output_parts = []

            # Check logs for stdout/stderr (E2B stores them in logs.stdout/stderr as lists)
            logs = getattr(result, "logs", None)
            if logs:
                stdout_list = getattr(logs, "stdout", None)
                if stdout_list:
                    stdout_text = "".join(stdout_list).strip()
                    if stdout_text:
                        output_parts.append(f"**Output:**\n```\n{stdout_text}\n```")

                stderr_list = getattr(logs, "stderr", None)
                if stderr_list:
                    stderr_text = "".join(stderr_list).strip()
                    if stderr_text:
                        output_parts.append(f"**Errors:**\n```\n{stderr_text}\n```")

            # Check for text result (final evaluation result) - only if no stdout to avoid duplicates
            text = getattr(result, "text", None)
            if text and text.strip() and not (logs and getattr(logs, "stdout", None)):
                output_parts.append(f"**Result:**\n```\n{text.strip()}\n```")

            # Check for rich results (plots, images, etc.)
            results_list = getattr(result, "results", None)
            if results_list:
                for res in results_list:
                    result_text = getattr(res, "text", None)
                    if result_text and result_text.strip():
                        output_parts.append(f"**Result:**\n```\n{result_text.strip()}\n```")
                    # Check for images/plots
                    if hasattr(res, "png") and getattr(res, "png", None):
                        output_parts.append("**Result:** Generated plot/image (PNG data available)")
                    if hasattr(res, "jpeg") and getattr(res, "jpeg", None):
                        output_parts.append(
                            "**Result:** Generated plot/image (JPEG data available)"
                        )

            if not output_parts:
                output_parts.append("Code executed successfully with no output.")

            logger.info(
                f"Executed Python code in E2B sandbox: {code[:512]}... -> {output_parts[:512]}"
            )

            return "\n\n".join(output_parts)

        except Exception as e:
            logger.error(f"E2B sandbox execution failed: {e}")
            # If sandbox connection is broken, reset it for next call
            if "sandbox" in str(e).lower() or "connection" in str(e).lower():
                self.sandbox = None
            return f"Error executing code: {e}"

    async def cleanup(self):
        """Clean up sandbox resources."""
        if self.sandbox:
            try:
                import asyncio

                sandbox_id = self.sandbox.sandbox_id
                await asyncio.to_thread(self.sandbox.kill)
                logger.info(f"Cleaned up E2B sandbox with ID: {sandbox_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up E2B sandbox: {e}")
            finally:
                self.sandbox = None


class ProgressReportExecutor:
    """Executor that sends progress updates via a provided callback."""

    def __init__(
        self,
        send_callback: Any | None = None,
        min_interval_seconds: int = 15,
    ):
        self.send_callback = send_callback
        self.min_interval_seconds = min_interval_seconds
        self._last_sent: float | None = None

    async def execute(self, text: str) -> str:
        # Sanitize to single line and trim
        clean = re.sub(r"\s+", " ", text or "").strip()
        logger.info(f"progress_report: {text}")
        if not clean:
            return "OK"

        # No-op if no callback (e.g., proactive mode)
        if not self.send_callback:
            return "OK"

        now = time.time()
        if self._last_sent is not None and (now - self._last_sent) < self.min_interval_seconds:
            return "OK"

        # Send update
        try:
            await self.send_callback(clean, "progress")
            self._last_sent = now
        except Exception as e:
            logger.warning(f"progress_report send failed: {e}")
        return "OK"


class FinalAnswerExecutor:
    """Executor for providing final answers."""

    async def execute(self, answer: str) -> str:
        """Return the final answer."""
        logger.info(f"Final answer provided: {answer[:100]}...")
        return answer


class MakePlanExecutor:
    """Executor for making plans (no-op that confirms receipt)."""

    async def execute(self, plan: str) -> str:
        """Confirm plan receipt."""
        logger.info(f"Plan formulated: {plan[:200]}...")
        return "OK, follow this plan"


class ArtifactStore:
    """Shared artifact storage for files and URLs."""

    def __init__(self, artifacts_path: str | None = None, artifacts_url: str | None = None):
        self.artifacts_path = Path(artifacts_path).expanduser() if artifacts_path else None
        self.artifacts_url = artifacts_url.rstrip("/") if artifacts_url else None

    @classmethod
    def from_config(cls, config: dict) -> "ArtifactStore":
        """Create store from configuration."""
        artifacts_config = config.get("tools", {}).get("artifacts", {})
        return cls(
            artifacts_path=artifacts_config.get("path"),
            artifacts_url=artifacts_config.get("url"),
        )

    def _ensure_configured(self) -> str | None:
        """Check if store is configured, return error message if not."""
        if not self.artifacts_path or not self.artifacts_url:
            return "Error: artifacts.path and artifacts.url must be configured"
        return None

    def write_text(self, content: str, suffix: str = ".txt") -> str:
        """Write text content to artifact file, return URL."""
        if err := self._ensure_configured():
            return err

        assert self.artifacts_path is not None
        assert self.artifacts_url is not None

        try:
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create artifacts directory: {e}")
            return f"Error: Failed to create artifacts directory: {e}"

        file_id = uuid.uuid4().hex
        filepath = self.artifacts_path / f"{file_id}{suffix}"

        try:
            filepath.write_text(content, encoding="utf-8")
            logger.info(f"Created artifact file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to write artifact file: {e}")
            return f"Error: Failed to create artifact file: {e}"

        return f"{self.artifacts_url}/{file_id}{suffix}"

    def write_bytes(self, data: bytes, suffix: str) -> str:
        """Write binary data to artifact file, return URL."""
        if err := self._ensure_configured():
            return err

        assert self.artifacts_path is not None
        assert self.artifacts_url is not None

        try:
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create artifacts directory: {e}")
            return f"Error: Failed to create artifacts directory: {e}"

        file_id = uuid.uuid4().hex
        filepath = self.artifacts_path / f"{file_id}{suffix}"

        try:
            filepath.write_bytes(data)
            logger.info(f"Created artifact file: {filepath}")
        except Exception as e:
            logger.error(f"Failed to write artifact file: {e}")
            return f"Error: Failed to create artifact file: {e}"

        return f"{self.artifacts_url}/{file_id}{suffix}"


class ShareArtifactExecutor:
    """Executor for sharing artifacts via files and links."""

    def __init__(self, store: ArtifactStore):
        self.store = store

    @classmethod
    def from_config(cls, config: dict) -> "ShareArtifactExecutor":
        """Create executor from configuration."""
        return cls(ArtifactStore.from_config(config))

    async def execute(self, content: str) -> str:
        """Share an artifact by creating a file and providing a link."""
        url = self.store.write_text(content, ".txt")
        if url.startswith("Error:"):
            return url
        return f"Artifact shared: {url}"


class ImageGenExecutor:
    """Executor for generating images via OpenRouter."""

    def __init__(
        self,
        router: Any,
        config: dict,
        max_image_size: int = 5 * 1024 * 1024,
        timeout: int = 30,
    ):
        from ..providers import parse_model_spec

        self.router = router
        self.config = config
        self.max_image_size = max_image_size
        self.timeout = timeout
        self.store = ArtifactStore.from_config(config)

        tools_config = config.get("tools", {}).get("image_gen", {})
        self.model = tools_config.get("model", "openrouter:google/gemini-2.5-flash-preview-image")

        spec = parse_model_spec(self.model)
        if spec.provider != "openrouter":
            raise ValueError(f"image_gen.model must use openrouter provider, got: {spec.provider}")

    @classmethod
    def from_config(cls, config: dict, router: Any) -> "ImageGenExecutor":
        """Create executor from configuration."""
        return cls(router=router, config=config)

    async def execute(self, prompt: str, image_urls: list[str] | None = None) -> str | list[dict]:
        """Generate image(s) using OpenRouter and store as artifacts."""

        # Build message content with text and optional images
        content: str | list[dict]
        logger.info(f"Generating image with prompt: {prompt}")
        if image_urls:
            content_blocks: list[dict] = [{"type": "text", "text": prompt}]
            async with aiohttp.ClientSession() as session:
                for url in image_urls:
                    try:
                        ct, b64 = await fetch_image_b64(
                            session, url, self.max_image_size, self.timeout
                        )
                        content_blocks.append(
                            {"type": "image_url", "image_url": {"url": f"data:{ct};base64,{b64}"}}
                        )
                        logger.info(f"Including additional image as input: {url}")
                    except ValueError as e:
                        logger.warning(f"Failed to fetch reference image {url}: {e}")
                        return f"Error: Failed to fetch reference image {url}: {e}"
            content = content_blocks
        else:
            content = prompt

        context = [{"role": "user", "content": content}]

        try:
            response, _, _ = await self.router.call_raw_with_model(
                model_str=self.model,
                context=context,
                system_prompt="",
                modalities=["image", "text"],
            )
        except Exception as e:
            logger.error(f"OpenRouter image generation failed: {e}")
            return f"Error: Image generation failed: {e}"

        if "error" in response:
            return f"Error: {response['error']}"

        # Extract images from response
        choices = response.get("choices", [])
        if not choices:
            logger.warning(f"No choices in response: {response}")
            return "Error: Model returned no output"

        message = choices[0].get("message", {})
        images = message.get("images", [])

        if not images:
            logger.warning(f"No images in message: {message}")
            return "Error: No images generated by model"

        artifact_urls = []
        image_blocks = []

        for img in images:
            img_url = None
            if isinstance(img, dict):
                img_url = img.get("image_url", {}).get("url") or img.get("url")
            elif isinstance(img, str):
                img_url = img

            if not img_url or not img_url.startswith("data:"):
                logger.warning(f"Invalid image data: {img}")
                continue

            # Parse data URL: data:image/png;base64,<data>
            try:
                parts = img_url.split(",", 1)
                if len(parts) != 2:
                    continue
                header, b64_data = parts
                mime_type = header.split(";")[0].replace("data:", "")
                img_bytes = base64.b64decode(b64_data)
            except Exception as e:
                logger.error(f"Failed to parse image data URL: {e}")
                continue

            ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
            suffix = ext_map.get(mime_type, ".png")

            url = self.store.write_bytes(img_bytes, suffix)
            if url.startswith("Error:"):
                return url

            # Add slop watermark using ImageMagick
            if self.store.artifacts_path:
                file_id = url.split("/")[-1].rsplit(".", 1)[0]
                filepath = self.store.artifacts_path / f"{file_id}{suffix}"
                try:
                    import subprocess

                    subprocess.run(
                        [
                            "convert",
                            str(filepath),
                            "-gravity",
                            "SouthEast",
                            "-pointsize",
                            "20",
                            "-fill",
                            "rgba(255,255,255,0.6)",
                            "-stroke",
                            "rgba(0,0,0,0.8)",
                            "-strokewidth",
                            "1",
                            "-annotate",
                            "+10+10",
                            "🍌slop",
                            str(filepath),
                        ],
                        check=True,
                        capture_output=True,
                    )
                except Exception as e:
                    logger.warning(f"Failed to add watermark to {filepath}: {e}")

            artifact_urls.append(url)

            # Add image block (reuse b64_data already parsed)
            image_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": b64_data,
                    },
                }
            )

        if not artifact_urls:
            return "Error: No images could be extracted from response"

        # Return Anthropic content blocks: text (URLs) + images
        content_blocks = [
            {
                "type": "text",
                "text": "\n".join(f"Generated image: {url}" for url in artifact_urls),
            }
        ] + image_blocks

        return content_blocks


class KnowledgeBaseSearchExecutor:
    """Search a PostgreSQL knowledge base with semantic extractions.

    Expected schema:
    - page_extensions: page_title, resume, keywords, url (with tsvector indexes)
    - entities: entity_name, entity_type, url (optional)
    - entity_relationships: subject_id, predicate, object_id
    """

    def __init__(
        self,
        database_url: str,
        name: str = "Knowledge Base",
        max_results: int = 5,
        max_entities: int = 10,
    ):
        self.database_url = database_url
        self.name = name
        self.max_results = max_results
        self.max_entities = max_entities
        self._pool: Any | None = None

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        return self._pool

    def _generate_query_variants(self, query: str) -> list[str]:
        """Generate query variants to improve full-text search matching.

        Returns a list of query variants to try, in order of preference:
        1. Original query
        2. Split on number+word boundaries (e.g., 'Project4Server' -> 'Project4 Server')
        3. Split on CamelCase (e.g., 'GeoServer' -> 'Geo Server')
        """
        import re

        variants = [query]

        # Variant: split number + word (2+ letters)
        # e.g., "Test4Server" -> "Test4 Server", but "Test4W" stays intact
        split_num = re.sub(r"([0-9])([a-zA-Z]{2,})", r"\1 \2", query)
        if split_num != query and split_num not in variants:
            variants.append(split_num)

        # Variant: split CamelCase (lowercase followed by uppercase)
        # e.g., "GeoServer" -> "Geo Server"
        split_camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", query)
        if split_camel != query and split_camel not in variants:
            variants.append(split_camel)

        # Variant: combine both transformations
        split_both = re.sub(r"([a-z])([A-Z])", r"\1 \2", split_num)
        if split_both != query and split_both not in variants:
            variants.append(split_both)

        return variants

    async def execute(self, query: str) -> str:
        """Search the knowledge base and return formatted results."""
        try:
            pool = await self._get_pool()
        except Exception as e:
            logger.error(f"Failed to connect to knowledge base: {e}")
            return f"Error: Failed to connect to {self.name}: {e}"

        results_parts = []
        pages = []
        entities = []

        # Generate query variants and try them in order until we get results
        query_variants = self._generate_query_variants(query)
        effective_query = query

        async with pool.acquire() as conn:  # type: ignore[union-attr]
            # 1. Full-text search on page_extensions - try variants until results found
            for variant in query_variants:
                pages = await conn.fetch(
                    """
                    SELECT page_title, url, resume, keywords,
                           ts_rank(resume_tsv, websearch_to_tsquery('english', $1)) +
                           ts_rank(keywords_tsv, websearch_to_tsquery('english', $1)) * 2 +
                           ts_rank(page_title_tsv, websearch_to_tsquery('english', $1)) * 3 AS rank
                    FROM page_extensions
                    WHERE resume_tsv @@ websearch_to_tsquery('english', $1)
                       OR keywords_tsv @@ websearch_to_tsquery('english', $1)
                       OR page_title_tsv @@ websearch_to_tsquery('english', $1)
                    ORDER BY rank DESC
                    LIMIT $2
                    """,
                    variant,
                    self.max_results,
                )
                if pages:
                    effective_query = variant
                    if variant != query:
                        logger.info(
                            f"Knowledge base query variant matched: '{query}' -> '{variant}'"
                        )
                    break

            if pages:
                results_parts.append(f"## Pages from {self.name}\n")
                for page in pages:
                    title = page["page_title"]
                    url = page["url"]
                    resume = page["resume"]
                    # Truncate resume for display
                    if len(resume) > 300:
                        resume = resume[:300] + "..."
                    results_parts.append(f"### [{title}]({url})\n{resume}\n")

            # 2. Entity search using trigram similarity (fuzzy matching)
            entities = await conn.fetch(
                """
                SELECT entity_name, entity_type, url,
                       similarity(entity_name, $1) AS sim
                FROM entities
                WHERE entity_name % $1
                   OR entity_name ILIKE '%' || $1 || '%'
                ORDER BY sim DESC, entity_name
                LIMIT $2
                """,
                query,
                self.max_entities,
            )

            # 3. Fallback: if no pages found but entities matched, search for top entity
            if not pages and entities:
                top_entity = entities[0]["entity_name"]
                if top_entity.lower() != query.lower():
                    logger.info(f"Knowledge base fallback: searching for entity '{top_entity}'")
                    pages = await conn.fetch(
                        """
                        SELECT page_title, url, resume, keywords,
                               ts_rank(resume_tsv, websearch_to_tsquery('english', $1)) +
                               ts_rank(keywords_tsv, websearch_to_tsquery('english', $1)) * 2 +
                               ts_rank(page_title_tsv, websearch_to_tsquery('english', $1)) * 3 AS rank
                        FROM page_extensions
                        WHERE resume_tsv @@ websearch_to_tsquery('english', $1)
                           OR keywords_tsv @@ websearch_to_tsquery('english', $1)
                           OR page_title_tsv @@ websearch_to_tsquery('english', $1)
                        ORDER BY rank DESC
                        LIMIT $2
                        """,
                        top_entity,
                        self.max_results,
                    )
                    if pages:
                        effective_query = top_entity
                        results_parts.append(f"## Pages from {self.name}\n")
                        for page in pages:
                            title = page["page_title"]
                            url = page["url"]
                            resume = page["resume"]
                            if len(resume) > 300:
                                resume = resume[:300] + "..."
                            results_parts.append(f"### [{title}]({url})\n{resume}\n")

            # 4. Chunk-based fallback: if few/no pages found, search raw content
            if len(pages) < 2:
                chunk_results = await conn.fetch(
                    """
                    SELECT p.title, p.url, LEFT(pc.chunk_text, 400) as excerpt,
                           ts_rank(pc.tsv, websearch_to_tsquery('english', $1)) as rank
                    FROM page_chunks pc
                    JOIN pages p ON pc.page_id = p.id
                    WHERE pc.tsv @@ websearch_to_tsquery('english', $1)
                    ORDER BY rank DESC
                    LIMIT $2
                    """,
                    effective_query,
                    self.max_results,
                )
                if chunk_results:
                    # Filter out pages we already have from page_extensions
                    existing_urls = {p["url"] for p in pages} if pages else set()
                    new_chunks = [c for c in chunk_results if c["url"] not in existing_urls]
                    if new_chunks:
                        logger.info(
                            f"Knowledge base chunk fallback: {len(new_chunks)} additional pages"
                        )
                        results_parts.append(f"\n## Additional content from {self.name}\n")
                        for chunk in new_chunks[:3]:  # Limit to 3 extra chunks
                            title = chunk["title"]
                            url = chunk["url"]
                            excerpt = chunk["excerpt"]
                            if len(excerpt) > 300:
                                excerpt = excerpt[:300] + "..."
                            results_parts.append(f"### [{title}]({url})\n{excerpt}\n")

            if entities:
                results_parts.append("\n## Entities\n")
                for ent in entities:
                    ent_name = ent["entity_name"]
                    ent_type = ent["entity_type"]
                    ent_url = ent["url"]
                    if ent_url:
                        results_parts.append(
                            f"- **{ent_name}** ({ent_type}) - [{ent_url}]({ent_url})"
                        )
                    else:
                        results_parts.append(f"- **{ent_name}** ({ent_type})")

            # 5. Get relationships for matched entities
            if entities:
                entity_names = [e["entity_name"] for e in entities[:5]]
                relationships = await conn.fetch(
                    """
                    SELECT s.entity_name AS subject, r.predicate, o.entity_name AS object
                    FROM entity_relationships r
                    JOIN entities s ON r.subject_id = s.id
                    JOIN entities o ON r.object_id = o.id
                    WHERE s.entity_name = ANY($1) OR o.entity_name = ANY($1)
                    LIMIT 15
                    """,
                    entity_names,
                )

                if relationships:
                    results_parts.append("\n## Relationships\n")
                    for rel in relationships:
                        results_parts.append(
                            f"- {rel['subject']} **{rel['predicate']}** {rel['object']}"
                        )

        if not results_parts:
            return f"No results found in {self.name} for: {query}"

        logger.info(
            f"Knowledge base search '{effective_query}': {len(pages)} pages, {len(entities)} entities"
        )
        return "\n".join(results_parts)

    async def cleanup(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


class RelationshipSearchExecutor:
    """Search entity relationships in a PostgreSQL knowledge base.

    This tool allows the model to query relationships by predicate type,
    with optional subject/object filters. It provides a flexible way to
    answer questions about organizational structure, leadership, projects, etc.
    """

    def __init__(self, database_url: str, name: str = "Knowledge Base"):
        self.database_url = database_url
        self.name = name
        self._pool: Any | None = None

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        return self._pool

    async def execute(
        self, predicate: str, subject: str | None = None, object: str | None = None
    ) -> str:
        """Search relationships by predicate with optional subject/object filters."""
        try:
            pool = await self._get_pool()
        except Exception as e:
            logger.error(f"Failed to connect to knowledge base: {e}")
            return f"Error: Failed to connect to {self.name}: {e}"

        async with pool.acquire() as conn:
            # Build query based on filters
            if subject and object:
                rows = await conn.fetch(
                    """
                    SELECT s.entity_name AS subject, r.predicate, o.entity_name AS object
                    FROM entity_relationships r
                    JOIN entities s ON r.subject_id = s.id
                    JOIN entities o ON r.object_id = o.id
                    WHERE r.predicate = $1
                      AND s.entity_name ILIKE $2
                      AND o.entity_name ILIKE $3
                    ORDER BY s.entity_name
                    LIMIT 50
                    """,
                    predicate,
                    f"%{subject}%",
                    f"%{object}%",
                )
            elif subject:
                rows = await conn.fetch(
                    """
                    SELECT s.entity_name AS subject, r.predicate, o.entity_name AS object
                    FROM entity_relationships r
                    JOIN entities s ON r.subject_id = s.id
                    JOIN entities o ON r.object_id = o.id
                    WHERE r.predicate = $1
                      AND s.entity_name ILIKE $2
                    ORDER BY s.entity_name
                    LIMIT 50
                    """,
                    predicate,
                    f"%{subject}%",
                )
            elif object:
                rows = await conn.fetch(
                    """
                    SELECT s.entity_name AS subject, r.predicate, o.entity_name AS object
                    FROM entity_relationships r
                    JOIN entities s ON r.subject_id = s.id
                    JOIN entities o ON r.object_id = o.id
                    WHERE r.predicate = $1
                      AND o.entity_name ILIKE $2
                    ORDER BY s.entity_name
                    LIMIT 50
                    """,
                    predicate,
                    f"%{object}%",
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT s.entity_name AS subject, r.predicate, o.entity_name AS object
                    FROM entity_relationships r
                    JOIN entities s ON r.subject_id = s.id
                    JOIN entities o ON r.object_id = o.id
                    WHERE r.predicate = $1
                    ORDER BY s.entity_name
                    LIMIT 50
                    """,
                    predicate,
                )

        if not rows:
            filters = []
            if subject:
                filters.append(f"subject='{subject}'")
            if object:
                filters.append(f"object='{object}'")
            filter_str = f" with filters: {', '.join(filters)}" if filters else ""
            return f"No relationships found for predicate '{predicate}'{filter_str}"

        # Format results
        results = [f"## Relationships: {predicate}"]
        for row in rows:
            results.append(f"- {row['subject']} -> {row['object']}")

        # Add hint for getting more details if multiple results
        if len(rows) > 3:
            # Find common prefix of subjects for hint
            subjects = [row["subject"] for row in rows]
            common_prefix = subjects[0]
            for s in subjects[1:]:
                while common_prefix and not s.startswith(common_prefix):
                    common_prefix = common_prefix[:-1]
            if len(common_prefix) >= 5:
                results.append(
                    f"\nTip: Use entity_info with name='{common_prefix.rstrip()}', "
                    "match='prefix', predicates='located_in,happened_in' to get details for all."
                )

        logger.info(f"Relationship search '{predicate}': {len(rows)} results")
        return "\n".join(results)

    async def cleanup(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


class EntityInfoExecutor:
    """Get information about entities and their relationships.

    This tool allows exploring entities by name pattern, returning
    all relevant relationships in one query. Useful for batch queries
    like listing all events with their locations.
    """

    def __init__(self, database_url: str, name: str = "Knowledge Base"):
        self.database_url = database_url
        self.name = name
        self._pool: Any | None = None

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        return self._pool

    async def execute(
        self,
        name: str,
        match: str = "contains",
        predicates: str | None = None,
        limit: int = 50,
    ) -> str:
        """Get entity info with relationships."""
        try:
            pool = await self._get_pool()
        except Exception as e:
            logger.error(f"Failed to connect to knowledge base: {e}")
            return f"Error: Failed to connect to {self.name}: {e}"

        # Validate and cap limit
        limit = min(max(1, limit), 100)

        # Build name pattern based on match type
        if match == "exact":
            name_pattern = name
            name_condition = "s.entity_name = $1"
        elif match == "prefix":
            name_pattern = f"{name}%"
            name_condition = "s.entity_name ILIKE $1"
        else:  # contains
            name_pattern = f"%{name}%"
            name_condition = "s.entity_name ILIKE $1"

        # Parse predicates filter
        predicate_filter = ""
        if predicates:
            pred_list = [p.strip() for p in predicates.split(",") if p.strip()]
            if pred_list:
                placeholders = ", ".join(f"${i + 2}" for i in range(len(pred_list)))
                predicate_filter = f"AND r.predicate IN ({placeholders})"

        async with pool.acquire() as conn:
            # Build name condition
            if match == "exact":
                entity_name_condition = "e.entity_name = $1"
            else:
                entity_name_condition = "e.entity_name ILIKE $1"

            # Parse predicates filter
            pred_list = []
            if predicates:
                pred_list = [p.strip() for p in predicates.split(",") if p.strip()]

            if pred_list:
                # When predicates specified, query entities that HAVE those predicates
                placeholders = ", ".join(f"${i + 2}" for i in range(len(pred_list)))
                query = f"""
                    SELECT 
                        e.entity_name AS entity,
                        e.entity_type,
                        r.predicate,
                        o.entity_name AS related_to
                    FROM entities e
                    JOIN entity_relationships r ON e.id = r.subject_id
                    JOIN entities o ON r.object_id = o.id
                    WHERE {entity_name_condition}
                      AND r.predicate IN ({placeholders})
                    ORDER BY e.entity_name, r.predicate, o.entity_name
                    LIMIT ${len(pred_list) + 2}
                """
                rows = await conn.fetch(query, name_pattern, *pred_list, limit * 10)
            else:
                # No predicate filter - get entities first, then all relationships
                entity_query = f"""
                    SELECT DISTINCT id, entity_name, entity_type
                    FROM entities e
                    WHERE {entity_name_condition.replace("e.", "")}
                    ORDER BY entity_name
                    LIMIT $2
                """
                entity_rows = await conn.fetch(entity_query, name_pattern, limit)

                if not entity_rows:
                    return f"No entities found matching '{name}' (match={match})"

                entity_ids = [row["id"] for row in entity_rows]

                rel_query = """
                    SELECT 
                        s.entity_name AS entity,
                        s.entity_type,
                        r.predicate,
                        o.entity_name AS related_to
                    FROM entities s
                    JOIN entity_relationships r ON s.id = r.subject_id
                    JOIN entities o ON r.object_id = o.id
                    WHERE s.id = ANY($1)
                    ORDER BY s.entity_name, r.predicate, o.entity_name
                """
                rows = await conn.fetch(rel_query, entity_ids)

        if not rows:
            return f"No entities found matching '{name}' with specified predicates"

        # Group results by entity
        entities: dict[str, dict[str, Any]] = {}
        for row in rows:
            entity_name = row["entity"]
            if entity_name not in entities:
                entities[entity_name] = {
                    "type": row["entity_type"],
                    "relationships": {},
                }
            pred = row["predicate"]
            if pred not in entities[entity_name]["relationships"]:
                entities[entity_name]["relationships"][pred] = []
            entities[entity_name]["relationships"][pred].append(row["related_to"])

        # Format output - use compact list format for multiple entities
        if len(entities) > 1:
            # Compact list format: "- EntityName: pred1=val1, pred2=val2"
            results = [f"## {len(entities)} entities matching '{name}'"]
            for entity_name, info in entities.items():
                parts = []
                for pred, values in info["relationships"].items():
                    if len(values) == 1:
                        parts.append(f"{values[0]}")
                    else:
                        parts.append(f"{pred}: {', '.join(values[:3])}")
                if parts:
                    results.append(f"- {entity_name}: {', '.join(parts)}")
                else:
                    results.append(f"- {entity_name}")
        else:
            # Detailed format for single entity
            results = [f"## Entities matching '{name}' ({len(entities)} found)"]
            for entity_name, info in entities.items():
                results.append(f"\n### {entity_name} ({info['type']})")
                for pred, values in info["relationships"].items():
                    if len(values) > 5:
                        display_values = values[:5] + [f"... and {len(values) - 5} more"]
                    else:
                        display_values = values
                    results.append(f"- {pred}: {', '.join(display_values)}")

        logger.info(
            f"Entity info '{name}' (match={match}): {len(entities)} entities, {len(rows)} relationships"
        )
        return "\n".join(results)

    async def cleanup(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


def create_tool_executors(
    config: dict | None = None,
    *,
    progress_callback: Any | None = None,
    agent: Any,
    arc: str,
    router: Any = None,
) -> dict[str, Any]:
    """Create tool executors with configuration."""
    # Tool configs
    tools = config.get("tools", {}) if config else {}

    # E2B config
    e2b_config = tools.get("e2b", {})
    e2b_api_key = e2b_config.get("api_key")

    # Jina config
    jina_config = tools.get("jina", {})
    jina_api_key = jina_config.get("api_key")

    # Search provider config
    tools_config = config.get("tools", {}) if config else {}
    search_provider = tools_config.get("search_provider")  # None if not configured
    user_agent = tools_config.get("user_agent") or DEFAULT_USER_AGENT

    # Create appropriate search executor based on provider
    # Set to None if disabled (empty string, "none", or "disabled")
    search_executor = None
    if search_provider and search_provider.lower() not in ("none", "disabled", "off", "false"):
        if search_provider == "jina":
            search_executor = JinaSearchExecutor(api_key=jina_api_key, user_agent=user_agent)
        elif search_provider == "brave":
            brave_config = tools.get("brave", {})
            brave_api_key = brave_config.get("api_key")
            if not brave_api_key:
                logger.warning("Brave search configured but no API key found, falling back to ddgs")
                search_executor = WebSearchExecutor(backend="brave")
            else:
                search_executor = BraveSearchExecutor(api_key=brave_api_key)
        elif search_provider == "google":
            google_config = tools.get("google", {})
            google_api_key = google_config.get("api_key")
            google_cx = google_config.get("cx")
            if not google_api_key or not google_cx:
                logger.warning(
                    "Google search configured but missing api_key or cx, falling back to ddgs"
                )
                search_executor = WebSearchExecutor(backend="auto")
            else:
                search_executor = GoogleSearchExecutor(api_key=google_api_key, cx=google_cx)
        else:
            if "jina" in search_provider:
                raise ValueError(
                    f"Jina search provider must be exclusive. Found: '{search_provider}'. "
                    "Use exactly 'jina' for jina search (recommended provider, but API key required)."
                )
            search_executor = WebSearchExecutor(backend=search_provider)
    else:
        logger.info("Web search disabled (search_provider not configured or set to disabled)")

    # Webpage visitor config
    webpage_visitor_type = tools_config.get("webpage_visitor", "local")

    if webpage_visitor_type == "local":
        webpage_visitor = LocalWebpageVisitor(
            progress_callback=progress_callback, user_agent=user_agent
        )
        logger.info("Using local webpage visitor (no external API calls)")
    else:
        # Default to Jina
        webpage_visitor = WebpageVisitorExecutor(
            progress_callback=progress_callback, api_key=jina_api_key, user_agent=user_agent
        )
        logger.info("Using Jina.ai webpage visitor")

    # Progress executor settings
    behavior = (config or {}).get("behavior", {})
    progress_cfg = behavior.get("progress", {}) if behavior else {}
    min_interval = int(progress_cfg.get("min_interval_seconds", 15))

    executors = {
        "visit_webpage": webpage_visitor,
        "progress_report": ProgressReportExecutor(
            send_callback=progress_callback, min_interval_seconds=min_interval
        ),
        "final_answer": FinalAnswerExecutor(),
        "make_plan": MakePlanExecutor(),
        "share_artifact": ShareArtifactExecutor.from_config(config or {}),
        "chronicle_append": ChapterAppendExecutor(agent=agent, arc=arc),
        "chronicle_read": ChapterRenderExecutor(chronicle=agent.chronicle, arc=arc),
    }

    # Add web_search only if a search provider is configured
    if search_executor:
        executors["web_search"] = search_executor
        logger.info(f"Web search enabled (provider: {search_provider})")

    # Add execute_python only if E2B API key is configured
    if e2b_api_key:
        executors["execute_python"] = PythonExecutorE2B(api_key=e2b_api_key)
        logger.info("Python execution enabled (E2B sandbox)")

    # Add generate_image only if router is available and openrouter is configured
    openrouter_config = (config or {}).get("providers", {}).get("openrouter", {})
    if router and openrouter_config.get("key"):
        executors["generate_image"] = ImageGenExecutor.from_config(config or {}, router)

    # Add knowledge_base if configured
    kb_config = tools.get("knowledge_base", {})
    if kb_config.get("enabled") and kb_config.get("database_url"):
        executors["knowledge_base"] = KnowledgeBaseSearchExecutor(
            database_url=kb_config["database_url"],
            name=kb_config.get("name", "Knowledge Base"),
            max_results=kb_config.get("max_results", 5),
            max_entities=kb_config.get("max_entities", 10),
        )
        logger.info(f"Knowledge base search enabled: {kb_config.get('name', 'Knowledge Base')}")

        # Add relationship_search alongside knowledge_base
        executors["relationship_search"] = RelationshipSearchExecutor(
            database_url=kb_config["database_url"],
            name=kb_config.get("name", "Knowledge Base"),
        )
        logger.info("Relationship search enabled")

        # Add entity_info alongside knowledge_base
        executors["entity_info"] = EntityInfoExecutor(
            database_url=kb_config["database_url"],
            name=kb_config.get("name", "Knowledge Base"),
        )
        logger.info("Entity info enabled")

    return executors


async def execute_tool(
    tool_name: str, tool_executors: dict[str, Any], **kwargs
) -> str | list[dict]:
    """Execute a tool by name with given arguments."""
    executors = tool_executors

    if tool_name not in executors:
        raise ValueError(f"Unknown tool '{tool_name}'")

    executor = executors[tool_name]
    return await executor.execute(**kwargs)
