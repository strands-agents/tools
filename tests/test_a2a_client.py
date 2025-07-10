from unittest.mock import AsyncMock, Mock, patch

import pytest
from a2a.types import Role, SendMessageRequest
from strands_tools.a2a_client import DEFAULT_TIMEOUT, A2AClientToolProvider


def test_init_default_parameters():
    """Test initialization with default parameters."""
    provider = A2AClientToolProvider()

    assert provider.timeout == DEFAULT_TIMEOUT
    assert provider._known_agent_urls == []
    assert provider._discovered_agents == {}
    assert provider._httpx_client is None


def test_init_custom_parameters():
    """Test initialization with custom parameters."""
    agent_urls = ["http://agent1.com", "http://agent2.com"]
    timeout = 60

    provider = A2AClientToolProvider(known_agent_urls=agent_urls, timeout=timeout)

    assert provider.timeout == timeout
    assert provider._known_agent_urls == agent_urls


def test_tools_property():
    """Test that tools property returns decorated methods."""
    provider = A2AClientToolProvider()
    tools = provider.tools

    # Should have the three @tool decorated methods
    tool_names = [tool.tool_name for tool in tools]
    assert "discover_agent" in tool_names
    assert "list_discovered_agents" in tool_names
    assert "send_message" in tool_names


@patch("strands_tools.a2a_client.asyncio.get_event_loop")
@patch("strands_tools.a2a_client.asyncio.new_event_loop")
@patch("strands_tools.a2a_client.asyncio.set_event_loop")
def test_run_async_with_event_loop(mock_set_event_loop, mock_new_event_loop, mock_get_event_loop):
    """Test _run_async uses event loop directly."""
    mock_coro = Mock()
    mock_loop = Mock()
    mock_loop.run_until_complete.return_value = "test_result"
    mock_get_event_loop.return_value = mock_loop

    provider = A2AClientToolProvider()
    result = provider._run_async(mock_coro)

    assert result == "test_result"
    mock_get_event_loop.assert_called_once()
    mock_loop.run_until_complete.assert_called_once_with(mock_coro)
    mock_new_event_loop.assert_not_called()
    mock_set_event_loop.assert_not_called()


@patch("strands_tools.a2a_client.asyncio.get_event_loop")
@patch("strands_tools.a2a_client.asyncio.new_event_loop")
@patch("strands_tools.a2a_client.asyncio.set_event_loop")
def test_run_async_creates_new_event_loop(mock_set_event_loop, mock_new_event_loop, mock_get_event_loop):
    """Test _run_async creates new event loop when none exists."""
    mock_coro = Mock()
    mock_loop = Mock()
    mock_loop.run_until_complete.return_value = "test_result"
    mock_get_event_loop.side_effect = RuntimeError("No event loop")
    mock_new_event_loop.return_value = mock_loop

    provider = A2AClientToolProvider()
    result = provider._run_async(mock_coro)

    assert result == "test_result"
    mock_get_event_loop.assert_called_once()
    mock_new_event_loop.assert_called_once()
    mock_set_event_loop.assert_called_once_with(mock_loop)
    mock_loop.run_until_complete.assert_called_once_with(mock_coro)


@pytest.mark.asyncio
async def test_ensure_httpx_client_creates_new_client():
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
async def test_ensure_httpx_client_reuses_existing():
    """Test _ensure_httpx_client reuses existing client."""
    provider = A2AClientToolProvider()
    existing_client = Mock()
    provider._httpx_client = existing_client

    result = await provider._ensure_httpx_client()

    assert result == existing_client


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_create_a2a_card_resolver")
async def test_async_discover_agent_card_success(mock_create_resolver):
    """Test _async_discover_agent_card successfully discovers and caches agent."""
    provider = A2AClientToolProvider()
    provider._initial_discovery_done = False
    mock_resolver = Mock()
    mock_agent_card = Mock()
    mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
    mock_create_resolver.return_value = mock_resolver

    result = await provider._async_discover_agent_card("http://test.com")

    assert result == mock_agent_card
    assert provider._discovered_agents["http://test.com"] == mock_agent_card
    mock_resolver.get_agent_card.assert_called_once()


@pytest.mark.asyncio
async def test_async_discover_agent_card_cached():
    """Test _async_discover_agent_card returns cached agent."""
    provider = A2AClientToolProvider()
    provider._initial_discovery_done = False
    cached_card = Mock()
    provider._discovered_agents["http://test.com"] = cached_card

    result = await provider._async_discover_agent_card("http://test.com")

    assert result == cached_card


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_async_discover_agent_card")
async def test_discover_known_agents_success(mock_discover):
    """Test _discover_known_agents with successful discovery."""
    provider = A2AClientToolProvider()
    provider._known_agent_urls = ["http://agent1.com", "http://agent2.com"]
    provider._initial_discovery_done = False

    mock_discover.return_value = Mock()

    await provider._discover_known_agents()

    assert provider._initial_discovery_done is True
    assert mock_discover.call_count == 2
    mock_discover.assert_any_call("http://agent1.com")
    mock_discover.assert_any_call("http://agent2.com")


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_async_discover_agent_card")
async def test_discover_known_agents_with_errors(mock_discover):
    """Test _discover_known_agents handles individual agent errors."""
    provider = A2AClientToolProvider()
    provider._known_agent_urls = ["http://agent1.com", "http://agent2.com"]
    provider._initial_discovery_done = False

    # First agent fails, second succeeds
    mock_discover.side_effect = [
        Exception("Agent 1 failed"),
        Mock(),  # Agent 2 succeeds
    ]

    await provider._discover_known_agents()

    assert provider._initial_discovery_done is True
    assert mock_discover.call_count == 2


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_discover_known_agents")
async def test_ensure_discovered_known_agents_calls_discovery(mock_discover):
    """Test _ensure_discovered_known_agents calls discovery when needed."""
    provider = A2AClientToolProvider()
    provider._known_agent_urls = ["http://agent1.com"]
    provider._initial_discovery_done = False

    await provider._ensure_discovered_known_agents()

    mock_discover.assert_called_once()


@pytest.mark.asyncio
async def test_ensure_discovered_known_agents_skips_when_done():
    """Test _ensure_discovered_known_agents skips when already done."""
    provider = A2AClientToolProvider()
    provider._known_agent_urls = ["http://agent1.com"]
    provider._initial_discovery_done = True

    with patch.object(provider, "_discover_known_agents") as mock_discover:
        await provider._ensure_discovered_known_agents()
        mock_discover.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_discovered_known_agents_skips_when_no_urls():
    """Test _ensure_discovered_known_agents skips when no URLs provided."""
    provider = A2AClientToolProvider()
    provider._known_agent_urls = []
    provider._initial_discovery_done = False

    with patch.object(provider, "_discover_known_agents") as mock_discover:
        await provider._ensure_discovered_known_agents()
        mock_discover.assert_not_called()


@patch.object(A2AClientToolProvider, "_run_async")
def test_discover_agent_success(mock_run_async):
    """Test discover_agent tool returns success result."""
    provider = A2AClientToolProvider()
    expected_result = {"status": "success", "agent_card": {"name": "test_agent"}, "url": "http://test.com"}
    mock_run_async.return_value = expected_result

    result = provider.discover_agent("http://test.com")

    assert result == expected_result


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_async_discover_agent_card")
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_async_discover_agent_card_tool_success(mock_ensure, mock_discover):
    """Test _async_discover_agent_card_tool returns success result."""
    provider = A2AClientToolProvider()
    mock_agent_card = Mock()
    mock_agent_card.model_dump.return_value = {"name": "test_agent"}
    mock_discover.return_value = mock_agent_card

    result = await provider._async_discover_agent_card_tool("http://test.com")

    expected = {"status": "success", "agent_card": {"name": "test_agent"}, "url": "http://test.com"}
    assert result == expected
    mock_ensure.assert_called_once()


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_async_discover_agent_card")
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_async_discover_agent_card_tool_error(mock_ensure, mock_discover):
    """Test _async_discover_agent_card_tool handles errors."""
    provider = A2AClientToolProvider()
    mock_discover.side_effect = Exception("Network error")

    result = await provider._async_discover_agent_card_tool("http://test.com")

    expected = {"status": "error", "error": "Network error", "url": "http://test.com"}
    assert result == expected


def test_list_discovered_agents_empty():
    """Test list_discovered_agents with no discovered agents."""
    provider = A2AClientToolProvider()

    with patch.object(provider, "_run_async") as mock_run_async:
        expected = {"status": "success", "agents": [], "total_count": 0}
        mock_run_async.return_value = expected

        result = provider.list_discovered_agents()

        assert result == expected


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_async_list_discovered_agents_with_agents(mock_ensure):
    """Test _async_list_discovered_agents with discovered agents."""
    provider = A2AClientToolProvider()
    mock_card1 = Mock()
    mock_card1.model_dump.return_value = {"name": "agent1"}
    mock_card2 = Mock()
    mock_card2.model_dump.return_value = {"name": "agent2"}

    provider._discovered_agents = {"http://agent1.com": mock_card1, "http://agent2.com": mock_card2}

    result = await provider._async_list_discovered_agents()

    expected = {"status": "success", "agents": [{"name": "agent1"}, {"name": "agent2"}], "total_count": 2}
    assert result == expected
    mock_ensure.assert_called_once()


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_async_list_discovered_agents_error(mock_ensure):
    """Test _async_list_discovered_agents handles errors."""
    provider = A2AClientToolProvider()
    mock_card = Mock()
    mock_card.model_dump.side_effect = Exception("Serialization error")
    provider._discovered_agents = {"http://test.com": mock_card}

    result = await provider._async_list_discovered_agents()

    expected = {"status": "error", "error": "Serialization error", "total_count": 0}
    assert result == expected


@patch.object(A2AClientToolProvider, "_run_async")
def test_send_message_with_message_id(mock_run_async):
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
def test_send_message_without_message_id(mock_run_async):
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
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_async_send_message_success(mock_ensure, mock_create_client, mock_uuid):
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
    mock_ensure.assert_called_once()

    # Verify client was called with correct message structure
    mock_client.send_message.assert_called_once()
    call_args = mock_client.send_message.call_args[0][0]
    assert isinstance(call_args, SendMessageRequest)
    assert call_args.id == "request_id_456"
    assert call_args.params.message.role == Role.user
    assert call_args.params.message.messageId == "message_id_123"


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_create_a2a_client")
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_async_send_message_error(mock_ensure, mock_create_client):
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


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_ensure_httpx_client")
async def test_create_a2a_card_resolver(mock_ensure_client):
    """Test _create_a2a_card_resolver creates resolver with correct parameters."""
    provider = A2AClientToolProvider()
    mock_client = Mock()
    mock_ensure_client.return_value = mock_client

    with patch("strands_tools.a2a_client.A2ACardResolver") as mock_resolver_class:
        mock_resolver = Mock()
        mock_resolver_class.return_value = mock_resolver

        result = await provider._create_a2a_card_resolver("http://test.com")

        mock_resolver_class.assert_called_once_with(httpx_client=mock_client, base_url="http://test.com")
        assert result == mock_resolver


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_ensure_httpx_client")
@patch.object(A2AClientToolProvider, "_async_discover_agent_card")
async def test_create_a2a_client(mock_discover, mock_ensure_client):
    """Test _create_a2a_client creates client with correct parameters."""
    provider = A2AClientToolProvider()
    mock_client = Mock()
    mock_ensure_client.return_value = mock_client
    mock_agent_card = Mock()
    mock_discover.return_value = mock_agent_card

    with patch("strands_tools.a2a_client.A2AClient") as mock_client_class:
        mock_a2a_client = Mock()
        mock_client_class.return_value = mock_a2a_client

        result = await provider._create_a2a_client("http://test.com")

        mock_client_class.assert_called_once_with(httpx_client=mock_client, agent_card=mock_agent_card)
        assert result == mock_a2a_client


@patch.object(A2AClientToolProvider, "_run_async")
def test_close_with_client(mock_run_async):
    """Test close method with existing HTTP client."""
    provider = A2AClientToolProvider()
    mock_client = AsyncMock()
    provider._httpx_client = mock_client
    provider._discovered_agents = {"http://test.com": Mock()}

    provider.close()

    mock_run_async.assert_called_once()
    assert provider._httpx_client is None
    assert provider._discovered_agents == {}


def test_close_without_client():
    """Test close method without HTTP client (idempotent)."""
    provider = A2AClientToolProvider()
    provider._httpx_client = None

    provider.close()

    assert provider._httpx_client is None


@pytest.mark.asyncio
async def test_close_async_functionality():
    """Test the async close functionality directly."""
    provider = A2AClientToolProvider()
    mock_client = AsyncMock()
    provider._httpx_client = mock_client
    provider._discovered_agents = {"http://test.com": Mock()}

    # Call the internal async close method
    await provider._async_close()
    provider._httpx_client = None
    provider._discovered_agents.clear()

    mock_client.aclose.assert_called_once()
    assert provider._httpx_client is None
    assert provider._discovered_agents == {}
