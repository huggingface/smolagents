"""Tests for async tool support (issue #334)."""

import asyncio

from smolagents import Tool, tool


class AsyncSearchTool(Tool):
    name = "async_search"
    description = "An async search tool for testing"
    inputs = {"query": {"type": "string", "description": "Search query"}}
    output_type = "string"

    async def forward(self, query: str) -> str:
        await asyncio.sleep(0.01)
        return f"Results for: {query}"


class SyncSearchTool(Tool):
    name = "sync_search"
    description = "A sync search tool for testing"
    inputs = {"query": {"type": "string", "description": "Search query"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        return f"Results for: {query}"


class TestAsyncToolSubclass:
    """Test async tools created by subclassing Tool."""

    def test_async_tool_returns_result(self):
        tool_instance = AsyncSearchTool()
        result = tool_instance("test query")
        assert result == "Results for: test query"

    def test_async_tool_with_kwargs(self):
        tool_instance = AsyncSearchTool()
        result = tool_instance(query="hello world")
        assert result == "Results for: hello world"

    def test_async_tool_with_dict_arg(self):
        tool_instance = AsyncSearchTool()
        result = tool_instance({"query": "dict input"})
        assert result == "Results for: dict input"

    def test_sync_tool_still_works(self):
        tool_instance = SyncSearchTool()
        result = tool_instance("test query")
        assert result == "Results for: test query"


class TestAsyncToolDecorator:
    """Test async tools created with the @tool decorator."""

    def test_async_decorated_tool(self):
        @tool
        async def async_greet(name: str) -> str:
            """Greets someone asynchronously.

            Args:
                name: The name to greet.
            """
            await asyncio.sleep(0.01)
            return f"Hello, {name}!"

        result = async_greet("World")
        assert result == "Hello, World!"

    def test_sync_decorated_tool_still_works(self):
        @tool
        def sync_greet(name: str) -> str:
            """Greets someone synchronously.

            Args:
                name: The name to greet.
            """
            return f"Hello, {name}!"

        result = sync_greet("World")
        assert result == "Hello, World!"


class TestAsyncToolInsideEventLoop:
    """Test that async tools work when called from within a running event loop."""

    def test_async_tool_inside_running_loop(self):
        tool_instance = AsyncSearchTool()

        async def run_in_loop():
            # This simulates calling a tool from within an async context
            # (e.g., an async agent or framework)
            return tool_instance("from async context")

        result = asyncio.run(run_in_loop())
        assert result == "Results for: from async context"

    def test_async_decorated_tool_inside_running_loop(self):
        @tool
        async def async_add(a: str, b: str) -> str:
            """Concatenates two strings asynchronously.

            Args:
                a: First string.
                b: Second string.
            """
            await asyncio.sleep(0.01)
            return a + b

        async def run_in_loop():
            return async_add("foo", "bar")

        result = asyncio.run(run_in_loop())
        assert result == "foobar"


class TestAsyncToolSerialization:
    """Test that async tools survive to_dict / from_dict round-tripping."""

    def test_async_subclass_round_trip(self):
        """Subclass tool whose forward has no extra imports."""

        class AsyncEchoTool(Tool):
            name = "async_echo"
            description = "Echoes input asynchronously"
            inputs = {"text": {"type": "string", "description": "Text to echo"}}
            output_type = "string"

            async def forward(self, text: str) -> str:
                return f"echo: {text}"

        original = AsyncEchoTool()
        data = original.to_dict()
        restored = Tool.from_dict(data)
        assert restored.name == original.name
        assert restored.description == original.description
        result = restored("round trip")
        assert result == "echo: round trip"

    def test_async_decorated_tool_round_trip(self):
        @tool
        async def async_upper(text: str) -> str:
            """Uppercases text asynchronously.

            Args:
                text: The text to uppercase.
            """
            return text.upper()

        data = async_upper.to_dict()
        restored = Tool.from_dict(data)
        assert restored.name == async_upper.name
        result = restored("hello")
        assert result == "HELLO"


class TestAsyncToolWithComplexReturn:
    """Test async tools returning various types."""

    def test_async_tool_returning_list(self):
        class AsyncListTool(Tool):
            name = "async_list"
            description = "Returns a list asynchronously"
            inputs = {"count": {"type": "integer", "description": "Number of items"}}
            output_type = "any"

            async def forward(self, count: int) -> list:
                await asyncio.sleep(0.01)
                return list(range(count))

        tool_instance = AsyncListTool()
        result = tool_instance(3)
        assert result == [0, 1, 2]

    def test_async_tool_returning_none(self):
        class AsyncNullTool(Tool):
            name = "async_null"
            description = "Returns nothing asynchronously"
            inputs = {}
            output_type = "null"

            async def forward(self) -> None:
                await asyncio.sleep(0.01)

        tool_instance = AsyncNullTool()
        result = tool_instance()
        assert result is None
