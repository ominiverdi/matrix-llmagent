"""Tests for the library search tool."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from matrix_llmagent.agentic_actor.library_tool import (
    LibraryResultsCache,
    LibrarySearchExecutor,
    fetch_library_image,
    format_results,
    get_best_image_path,
    get_citation_tag,
    library_search_tool_def,
    search_library_direct,
)


class TestLibraryResultsCache:
    """Tests for LibraryResultsCache."""

    def test_store_and_get(self):
        """Test basic store and get operations."""
        cache = LibraryResultsCache(ttl_hours=24, max_rooms=100)
        results = [{"id": 1, "content": "test"}]

        cache.store("room1", results)
        retrieved = cache.get("room1")

        assert retrieved == results

    def test_get_nonexistent(self):
        """Test getting from nonexistent room returns None."""
        cache = LibraryResultsCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        """Test that expired entries are not returned."""
        cache = LibraryResultsCache(ttl_hours=0.0001)  # Very short TTL
        results = [{"id": 1}]

        cache.store("room1", results)
        # Artificially expire the entry
        cache._cache["room1"].timestamp = time.time() - 3600

        assert cache.get("room1") is None

    def test_max_rooms_eviction(self):
        """Test that oldest entries are evicted when max_rooms is reached."""
        cache = LibraryResultsCache(ttl_hours=24, max_rooms=2)

        cache.store("room1", [{"id": 1}])
        time.sleep(0.01)  # Small delay to ensure different timestamps
        cache.store("room2", [{"id": 2}])
        time.sleep(0.01)
        cache.store("room3", [{"id": 3}])  # Should evict room1

        assert cache.get("room1") is None
        assert cache.get("room2") is not None
        assert cache.get("room3") is not None

    def test_clear(self):
        """Test clearing a room's cache."""
        cache = LibraryResultsCache()
        cache.store("room1", [{"id": 1}])

        cache.clear("room1")

        assert cache.get("room1") is None


class TestCitationTags:
    """Tests for citation tag generation."""

    def test_figure_tag(self):
        """Test figure citation tag."""
        result = {"source_type": "element", "element_type": "figure"}
        assert get_citation_tag(result) == "f"

    def test_table_tag(self):
        """Test table citation tag."""
        result = {"source_type": "element", "element_type": "table"}
        assert get_citation_tag(result) == "tb"

    def test_equation_tag(self):
        """Test equation citation tag."""
        result = {"source_type": "element", "element_type": "equation"}
        assert get_citation_tag(result) == "eq"

    def test_text_chunk_tag(self):
        """Test text chunk citation tag."""
        result = {"source_type": "chunk"}
        assert get_citation_tag(result) == "t"

    def test_unknown_element_tag(self):
        """Test unknown element type defaults to 'e'."""
        result = {"source_type": "element", "element_type": "unknown"}
        assert get_citation_tag(result) == "e"


class TestFormatResults:
    """Tests for result formatting."""

    def test_empty_results(self):
        """Test formatting empty results."""
        output = format_results([], "Test Library")
        assert "No results found" in output

    def test_format_element_result(self):
        """Test formatting an element result."""
        results = [
            {
                "source_type": "element",
                "element_type": "figure",
                "element_label": "Figure 1",
                "score_pct": 85.5,
                "page_number": 10,
                "document_title": "Test Doc",
                "content": "A test figure description",
            }
        ]
        output = format_results(results, "Test Library")

        assert "[RESULT #1]" in output  # Result number format
        assert "FIGURE" in output  # Element type
        assert "Figure 1" in output
        assert "p.10" in output
        assert "Test Doc" in output
        assert "show" in output.lower()  # Should suggest show command

    def test_format_text_chunk_result(self):
        """Test formatting a text chunk result."""
        results = [
            {
                "source_type": "chunk",
                "chunk_index": 5,
                "score_pct": 70.0,
                "page_number": 25,
                "document_title": "Test Doc",
                "content": "Some text content from the document.",
            }
        ]
        output = format_results(results, "Test Library")

        assert "[RESULT #1]" in output  # Result number format
        assert "TEXT" in output  # Text chunk indicator
        assert "p.25" in output
        assert "Some text content" in output

    def test_format_mixed_results(self):
        """Test formatting mixed element and chunk results."""
        results = [
            {
                "source_type": "element",
                "element_type": "equation",
                "element_label": "Eq. 1",
                "score_pct": 90.0,
                "page_number": 5,
                "document_title": "Math Doc",
                "content": "An equation",
            },
            {
                "source_type": "chunk",
                "chunk_index": 0,
                "score_pct": 60.0,
                "page_number": 1,
                "document_title": "Math Doc",
                "content": "Introduction text",
            },
        ]
        output = format_results(results, "Test Library")

        assert "[RESULT #1]" in output  # First result
        assert "[RESULT #2]" in output  # Second result
        assert "EQUATION" in output  # Equation type
        assert "TEXT" in output  # Text chunk type
        assert "Eq. 1" in output
        assert "Introduction text" in output


class TestGetBestImagePath:
    """Tests for best image path selection."""

    def test_chunk_has_no_image(self):
        """Test that chunks return None."""
        result = {"source_type": "chunk"}
        assert get_best_image_path(result) is None

    def test_figure_uses_crop_path(self):
        """Test that figures use crop_path."""
        result = {
            "source_type": "element",
            "element_type": "figure",
            "crop_path": "elements/figure.png",
        }
        assert get_best_image_path(result) == "elements/figure.png"

    def test_equation_prefers_rendered_path(self):
        """Test that equations prefer rendered_path over crop_path."""
        result = {
            "source_type": "element",
            "element_type": "equation",
            "crop_path": "elements/eq_crop.png",
            "rendered_path": "elements/eq_rendered.png",
        }
        assert get_best_image_path(result) == "elements/eq_rendered.png"

    def test_equation_falls_back_to_crop_path(self):
        """Test that equations fall back to crop_path if no rendered_path."""
        result = {
            "source_type": "element",
            "element_type": "equation",
            "crop_path": "elements/eq_crop.png",
            "rendered_path": "",
        }
        assert get_best_image_path(result) == "elements/eq_crop.png"

    def test_element_with_no_paths(self):
        """Test element with no image paths returns None."""
        result = {
            "source_type": "element",
            "element_type": "figure",
        }
        assert get_best_image_path(result) is None


class TestLibrarySearchToolDef:
    """Tests for tool definition generation."""

    def test_tool_def_structure(self):
        """Test that tool definition has required structure."""
        tool_def = library_search_tool_def("Test Library", "Search test documents.")

        assert tool_def["name"] == "library_search"
        assert "description" in tool_def
        assert "input_schema" in tool_def
        assert tool_def["persist"] == "summary"

    def test_tool_def_properties(self):
        """Test that expected properties exist."""
        tool_def = library_search_tool_def("Test Library", "Search test documents.")
        schema = tool_def["input_schema"]

        assert "query" in schema["properties"]
        assert "mode" in schema["properties"]
        assert "document_slug" in schema["properties"]
        assert "element_type" in schema["properties"]
        # No required fields - query is optional when mode='tour'
        assert schema["required"] == []

    def test_tool_def_mode_enum(self):
        """Test that mode has correct enum values."""
        tool_def = library_search_tool_def("Test Library", "Search test documents.")
        mode_prop = tool_def["input_schema"]["properties"]["mode"]

        assert mode_prop["type"] == "string"
        assert mode_prop["enum"] == ["search", "tour"]


class TestLibrarySearchExecutor:
    """Tests for LibrarySearchExecutor."""

    @pytest.fixture
    def cache(self):
        """Create a test cache."""
        return LibraryResultsCache(ttl_hours=24, max_rooms=100)

    @pytest.fixture
    def executor(self, cache):
        """Create a test executor."""
        return LibrarySearchExecutor(
            base_url="http://localhost:8095",
            cache=cache,
            arc="test#room",
            name="Test Library",
        )

    @pytest.mark.asyncio
    async def test_tour_mode(self, executor):
        """Test that mode='tour' returns the tour text."""
        result = await executor.execute(mode="tour")

        # Tour text should contain key sections
        assert "Welcome to the OSGeo Library" in result or "Library tour" in result
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_search_without_query(self, executor):
        """Test that search mode without query returns guidance."""
        result = await executor.execute()

        assert "search query" in result.lower() or "tour" in result.lower()

    @pytest.mark.asyncio
    async def test_successful_search(self, executor, cache):
        """Test successful search stores results in cache."""
        mock_response_data = {
            "query": "test query",
            "results": [
                {
                    "id": 1,
                    "source_type": "chunk",
                    "content": "Test content",
                    "score_pct": 80.0,
                    "page_number": 1,
                    "document_title": "Test Doc",
                    "chunk_index": 0,
                }
            ],
            "total": 1,
        }

        with patch(
            "matrix_llmagent.agentic_actor.library_tool.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data

            mock_post_ctx = AsyncMock()
            mock_post_ctx.__aenter__.return_value = mock_response
            mock_post_ctx.__aexit__.return_value = None

            mock_session.post = MagicMock(return_value=mock_post_ctx)

            result = await executor.execute("test query")

        # Hybrid format: text content for LLM (no [RESULT #N] format)
        assert "Test Library" in result
        assert "Test content" in result  # Content from text chunk
        assert "Test Doc" in result  # Document title

        # Check cache was populated
        cached = cache.get("test#room")
        assert cached is not None
        assert len(cached) == 1

    @pytest.mark.asyncio
    async def test_connection_error(self, executor):
        """Test handling of connection errors."""
        import aiohttp

        with patch(
            "matrix_llmagent.agentic_actor.library_tool.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session

            # Make post raise connection error
            mock_session.post = MagicMock(
                side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused"))
            )

            result = await executor.execute("test query")

        assert "Cannot connect" in result
        assert "administrator" in result

    @pytest.mark.asyncio
    async def test_http_error(self, executor):
        """Test handling of HTTP errors."""
        with patch(
            "matrix_llmagent.agentic_actor.library_tool.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"

            mock_post_ctx = AsyncMock()
            mock_post_ctx.__aenter__.return_value = mock_response
            mock_post_ctx.__aexit__.return_value = None

            mock_session.post = MagicMock(return_value=mock_post_ctx)

            result = await executor.execute("test query")

        assert "failed" in result.lower()
        assert "500" in result

    @pytest.mark.asyncio
    async def test_element_type_filter(self, executor):
        """Test that element_type filter disables chunks."""
        with patch(
            "matrix_llmagent.agentic_actor.library_tool.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"results": [], "total": 0}

            mock_post_ctx = AsyncMock()
            mock_post_ctx.__aenter__.return_value = mock_response
            mock_post_ctx.__aexit__.return_value = None

            mock_session.post = MagicMock(return_value=mock_post_ctx)

            await executor.execute("test", element_type="figure")

            # Verify the request payload had include_chunks=False
            call_args = mock_session.post.call_args
            payload = call_args.kwargs.get("json", {})
            assert payload.get("include_chunks") is False
            assert payload.get("element_type") == "figure"


class TestFetchLibraryImage:
    """Tests for fetch_library_image function."""

    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        """Test successful image fetch."""
        image_data = b"\x89PNG\r\n\x1a\n..."  # Fake PNG header

        with patch(
            "matrix_llmagent.agentic_actor.library_tool.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"content-type": "image/png"}
            mock_response.read.return_value = image_data

            mock_get_ctx = AsyncMock()
            mock_get_ctx.__aenter__.return_value = mock_response
            mock_get_ctx.__aexit__.return_value = None

            mock_session.get = MagicMock(return_value=mock_get_ctx)

            result = await fetch_library_image(
                "http://localhost:8095",
                "test_doc",
                "elements/figure.png",
            )

        assert result is not None
        assert result[0] == image_data
        assert result[1] == "image/png"

    @pytest.mark.asyncio
    async def test_not_found(self):
        """Test handling of 404 response."""
        with patch(
            "matrix_llmagent.agentic_actor.library_tool.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 404

            mock_get_ctx = AsyncMock()
            mock_get_ctx.__aenter__.return_value = mock_response
            mock_get_ctx.__aexit__.return_value = None

            mock_session.get = MagicMock(return_value=mock_get_ctx)

            result = await fetch_library_image(
                "http://localhost:8095",
                "test_doc",
                "elements/missing.png",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test handling of connection errors."""
        import aiohttp

        with patch(
            "matrix_llmagent.agentic_actor.library_tool.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session

            # Make get raise connection error
            mock_session.get = MagicMock(
                side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused"))
            )

            result = await fetch_library_image(
                "http://localhost:8095",
                "test_doc",
                "elements/figure.png",
            )

        assert result is None


class TestSearchLibraryDirect:
    """Tests for search_library_direct function."""

    @pytest.mark.asyncio
    async def test_successful_search(self):
        """Test successful direct library search."""
        mock_results = [
            {
                "source_type": "element",
                "element_type": "figure",
                "element_label": "Figure 1",
                "score_pct": 85.0,
                "page_number": 10,
                "document_title": "Test Doc",
                "content": "A test figure",
            }
        ]

        with patch(
            "matrix_llmagent.agentic_actor.library_tool.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"results": mock_results, "total": 1}

            mock_post_ctx = AsyncMock()
            mock_post_ctx.__aenter__.return_value = mock_response
            mock_post_ctx.__aexit__.return_value = None

            mock_session.post = MagicMock(return_value=mock_post_ctx)

            results, formatted = await search_library_direct(
                base_url="http://localhost:8095",
                query="test query",
                name="Test Library",
            )

        assert len(results) == 1
        assert results[0]["element_label"] == "Figure 1"
        assert "Test Library" in formatted
        assert "[RESULT #1]" in formatted  # Result number format
        assert "FIGURE" in formatted  # Figure type

    @pytest.mark.asyncio
    async def test_caches_results(self):
        """Test that results are cached when cache and arc are provided."""
        mock_results = [{"source_type": "chunk", "content": "test"}]
        cache = LibraryResultsCache()

        with patch(
            "matrix_llmagent.agentic_actor.library_tool.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"results": mock_results}

            mock_post_ctx = AsyncMock()
            mock_post_ctx.__aenter__.return_value = mock_response
            mock_post_ctx.__aexit__.return_value = None

            mock_session.post = MagicMock(return_value=mock_post_ctx)

            await search_library_direct(
                base_url="http://localhost:8095",
                query="test",
                cache=cache,
                arc="cli#test",
            )

        # Verify results were cached
        cached = cache.get("cli#test")
        assert cached is not None
        assert len(cached) == 1

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test handling of connection errors."""
        import aiohttp

        with patch(
            "matrix_llmagent.agentic_actor.library_tool.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session

            mock_session.post = MagicMock(
                side_effect=aiohttp.ClientConnectorError(MagicMock(), OSError("Connection refused"))
            )

            results, formatted = await search_library_direct(
                base_url="http://localhost:8095",
                query="test",
            )

        assert results == []
        assert "Cannot connect" in formatted

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """Test handling of empty results."""
        with patch(
            "matrix_llmagent.agentic_actor.library_tool.aiohttp.ClientSession"
        ) as mock_session_cls:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"results": []}

            mock_post_ctx = AsyncMock()
            mock_post_ctx.__aenter__.return_value = mock_response
            mock_post_ctx.__aexit__.return_value = None

            mock_session.post = MagicMock(return_value=mock_post_ctx)

            results, formatted = await search_library_direct(
                base_url="http://localhost:8095",
                query="nonexistent",
                name="Test Library",
            )

        assert results == []
        assert "No results" in formatted
