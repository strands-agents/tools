"""
Unit tests for A2AClientToolProvider.

Tests cover all public methods, error handling, resource management,
and async-to-sync conversion functionality.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from a2a.types import Role, SendMessageRequest
from strands_tools.a2a_client import DEFAULT_TIMEOUT, A2AClientToolProvider


class TestA2AClientToolProvider:
    """Test suite for A2AClientToolProvider."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        provider = A2AClientToolProvider()

        assert provider.timeout == DEFAULT_TIMEOUT
        assert provider._agent_urls == []
        assert provider._discovered_agents == {}
        assert provider._httpx_client is None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        agent_urls = ["http://agent1.com", "http://agent2.com"]
        timeout = 60

        provider = A2AClientToolProvider(agent_urls=agent_urls, timeout=timeout, discover_on_init=False)

        assert provider.timeout == timeout
        assert provider._agent_urls == agent_urls

    @patch.object(A2AClientToolProvider, "_run_async")
    def test_init_with_discover_on_init(self, mock_run_async):
        """Test initialization with discover_on_init=True."""
        agent_urls = ["http://agent1.com"]

        A2AClientToolProvider(agent_urls=agent_urls, discover_on_init=True)

        mock_run_async.assert_called_once()

    @patch("strands_tools.a2a_client.asyncio.get_event_loop")
    def test_run_async_existing_loop(self, mock_get_loop):
        """Test _run_async with existing event loop."""
        mock_loop = Mock()
        mock_get_loop.return_value = mock_loop
        mock_coro = Mock()

        provider = A2AClientToolProvider()
        provider._run_async(mock_coro)

        mock_loop.run_until_complete.assert_called_once_with(mock_coro)

    @patch("strands_tools.a2a_client.asyncio.set_event_loop")
    @patch("strands_tools.a2a_client.asyncio.new_event_loop")
    @patch("strands_tools.a2a_client.asyncio.get_event_loop")
    def test_run_async_no_existing_loop(self, mock_get_loop, mock_new_loop, mock_set_loop):
        """Test _run_async when no event loop exists."""
        mock_get_loop.side_effect = RuntimeError("No event loop")
        mock_loop = Mock()
        mock_new_loop.return_value = mock_loop
        mock_coro = Mock()

        provider = A2AClientToolProvider()
        provider._run_async(mock_coro)

        mock_new_loop.assert_called_once()
        mock_set_loop.assert_called_once_with(mock_loop)
        mock_loop.run_until_complete.assert_called_once_with(mock_coro)

    @pytest.mark.asyncio
    async def test_ensure_httpx_client_creates_new_client(self):
        """Test _ensure_httpx_client creates new client when none exists."""
        provider = A2AClientToolProvider(timeout=45)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            result = await provider._ensure_httpx_client()

            mock_client_class.assert_called_once_with(timeout=45)
            assert result == mock_client
            assert provider._httpx_client == mock_client

    @pytest.mark.asyncio
    async def test_ensure_httpx_client_reuses_existing(self):
        """Test _ensure_httpx_client reuses existing client."""
        provider = A2AClientToolProvider()
        existing_client = Mock()
        provider._httpx_client = existing_client

        result = await provider._ensure_httpx_client()

        assert result == existing_client

    @pytest.mark.asyncio
    @patch.object(A2AClientToolProvider, "_create_a2a_card_resolver")
    async def test_async_discover_agent_card_success(self, mock_create_resolver):
        """Test _async_discover_agent_card successfully discovers and caches agent."""
        provider = A2AClientToolProvider()
        mock_resolver = Mock()
        mock_agent_card = Mock()
        mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
        mock_create_resolver.return_value = mock_resolver

        result = await provider._async_discover_agent_card("http://test.com")

        assert result == mock_agent_card
        assert provider._discovered_agents["http://test.com"] == mock_agent_card
        mock_resolver.get_agent_card.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_discover_agent_card_cached(self):
        """Test _async_discover_agent_card returns cached agent."""
        provider = A2AClientToolProvider()
        cached_card = Mock()
        provider._discovered_agents["http://test.com"] = cached_card

        result = await provider._async_discover_agent_card("http://test.com")

        assert result == cached_card

    @patch.object(A2AClientToolProvider, "_run_async")
    def test_discover_agent_success(self, mock_run_async):
        """Test discover_agent tool returns success result."""
        provider = A2AClientToolProvider()
        expected_result = {"status": "success", "agent_card": {"name": "test_agent"}, "url": "http://test.com"}
        mock_run_async.return_value = expected_result

        result = provider.discover_agent("http://test.com")

        assert result == expected_result

    @pytest.mark.asyncio
    @patch.object(A2AClientToolProvider, "_async_discover_agent_card")
    async def test_async_discover_agent_card_tool_success(self, mock_discover):
        """Test _async_discover_agent_card_tool returns success result."""
        provider = A2AClientToolProvider()
        mock_agent_card = Mock()
        mock_agent_card.model_dump.return_value = {"name": "test_agent"}
        mock_discover.return_value = mock_agent_card

        result = await provider._async_discover_agent_card_tool("http://test.com")

        expected = {"status": "success", "agent_card": {"name": "test_agent"}, "url": "http://test.com"}
        assert result == expected

    @pytest.mark.asyncio
    @patch.object(A2AClientToolProvider, "_async_discover_agent_card")
    async def test_async_discover_agent_card_tool_error(self, mock_discover):
        """Test _async_discover_agent_card_tool handles errors."""
        provider = A2AClientToolProvider()
        mock_discover.side_effect = Exception("Network error")

        result = await provider._async_discover_agent_card_tool("http://test.com")

        expected = {"status": "error", "error": "Network error", "url": "http://test.com"}
        assert result == expected

    def test_list_discovered_agents_empty(self):
        """Test list_discovered_agents with no discovered agents."""
        provider = A2AClientToolProvider()

        result = provider.list_discovered_agents()

        expected = {"status": "success", "agents": [], "total_count": 0}
        assert result == expected

    def test_list_discovered_agents_with_agents(self):
        """Test list_discovered_agents with discovered agents."""
        provider = A2AClientToolProvider()
        mock_card1 = Mock()
        mock_card1.model_dump.return_value = {"name": "agent1"}
        mock_card2 = Mock()
        mock_card2.model_dump.return_value = {"name": "agent2"}

        provider._discovered_agents = {"http://agent1.com": mock_card1, "http://agent2.com": mock_card2}

        result = provider.list_discovered_agents()

        expected = {"status": "success", "agents": [{"name": "agent1"}, {"name": "agent2"}], "total_count": 2}
        assert result == expected

    def test_list_discovered_agents_error(self):
        """Test list_discovered_agents handles errors."""
        provider = A2AClientToolProvider()
        mock_card = Mock()
        mock_card.model_dump.side_effect = Exception("Serialization error")
        provider._discovered_agents = {"http://test.com": mock_card}

        result = provider.list_discovered_agents()

        expected = {"status": "error", "error": "Serialization error", "total_count": 0}
        assert result == expected

    @patch.object(A2AClientToolProvider, "_run_async")
    def test_send_message_with_message_id(self, mock_run_async):
        """Test send_message with provided message_id."""
        provider = A2AClientToolProvider()
        expected_result = {
            "status": "success",
            "response": {"result": "ok"},
            "message_id": "test_id",
            "target_agent_url": "http://test.com",
        }
        mock_run_async.return_value = expected_result

        result = provider.send_message("Hello", "http://test.com", "test_id")

        assert result == expected_result

    @patch.object(A2AClientToolProvider, "_run_async")
    def test_send_message_without_message_id(self, mock_run_async):
        """Test send_message without message_id (auto-generated)."""
        provider = A2AClientToolProvider()
        expected_result = {
            "status": "success",
            "response": {"result": "ok"},
            "message_id": "auto_generated",
            "target_agent_url": "http://test.com",
        }
        mock_run_async.return_value = expected_result

        result = provider.send_message("Hello", "http://test.com")

        assert result == expected_result

    @pytest.mark.asyncio
    @patch("strands_tools.a2a_client.uuid4")
    @patch.object(A2AClientToolProvider, "_create_a2a_client")
    async def test_async_send_message_success(self, mock_create_client, mock_uuid):
        """Test _async_send_message successful message sending."""
        provider = A2AClientToolProvider()

        # Mock UUID generation
        mock_message_uuid = Mock()
        mock_message_uuid.hex = "message_id_123"
        mock_request_uuid = Mock()
        mock_request_uuid.__str__ = Mock(return_value="request_id_456")
        mock_uuid.side_effect = [mock_message_uuid, mock_request_uuid]

        # Mock A2A client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.model_dump.return_value = {"result": "success"}
        mock_client.send_message = AsyncMock(return_value=mock_response)
        mock_create_client.return_value = mock_client

        result = await provider._async_send_message("Hello world", "http://test.com", None)

        expected = {
            "status": "success",
            "response": {"result": "success"},
            "message_id": "message_id_123",
            "target_agent_url": "http://test.com",
        }
        assert result == expected

        # Verify client was called with correct message structure
        mock_client.send_message.assert_called_once()
        call_args = mock_client.send_message.call_args[0][0]
        assert isinstance(call_args, SendMessageRequest)
        assert call_args.id == "request_id_456"
        assert call_args.params.message.role == Role.user
        assert call_args.params.message.messageId == "message_id_123"

    @pytest.mark.asyncio
    @patch.object(A2AClientToolProvider, "_create_a2a_client")
    async def test_async_send_message_error(self, mock_create_client):
        """Test _async_send_message handles errors."""
        provider = A2AClientToolProvider()
        mock_create_client.side_effect = Exception("Connection failed")

        result = await provider._async_send_message("Hello", "http://test.com", "test_id")

        expected = {
            "status": "error",
            "error": "Connection failed",
            "message_id": "test_id",
            "target_agent_url": "http://test.com",
        }
        assert result == expected

    @patch.object(A2AClientToolProvider, "_run_async")
    def test_close_with_client(self, mock_run_async):
        """Test close method with existing HTTP client."""
        provider = A2AClientToolProvider()
        mock_client = Mock()
        provider._httpx_client = mock_client
        provider._discovered_agents = {"http://test.com": Mock()}

        provider.close()

        mock_run_async.assert_called_once()
        assert provider._httpx_client is None
        assert provider._discovered_agents == {}

    def test_close_without_client(self):
        """Test close method without HTTP client (idempotent)."""
        provider = A2AClientToolProvider()
        provider._httpx_client = None

        provider.close()

        assert provider._httpx_client is None

    @pytest.mark.asyncio
    @patch.object(A2AClientToolProvider, "_async_discover_agent_card")
    async def test_discover_all_agents_success(self, mock_discover):
        """Test _discover_all_agents with successful discovery."""
        provider = A2AClientToolProvider()
        provider._agent_urls = ["http://agent1.com", "http://agent2.com"]

        mock_discover.return_value = Mock()

        await provider._discover_all_agents()

        assert mock_discover.call_count == 2
        mock_discover.assert_any_call("http://agent1.com")
        mock_discover.assert_any_call("http://agent2.com")

    @pytest.mark.asyncio
    @patch.object(A2AClientToolProvider, "_async_discover_agent_card")
    async def test_discover_all_agents_with_errors(self, mock_discover):
        """Test _discover_all_agents handles individual agent errors."""
        provider = A2AClientToolProvider()
        provider._agent_urls = ["http://agent1.com", "http://agent2.com"]

        # First agent fails, second succeeds
        mock_discover.side_effect = [
            Exception("Agent 1 failed"),
            Mock(),  # Agent 2 succeeds
        ]

        await provider._discover_all_agents()

        assert mock_discover.call_count == 2
