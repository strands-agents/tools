from unittest.mock import AsyncMock, Mock, patch

import pytest
from a2a.types import Message
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
    assert "a2a_discover_agent" in tool_names
    assert "a2a_list_discovered_agents" in tool_names
    assert "a2a_send_message" in tool_names


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
async def test_discover_agent_card_success(mock_create_resolver):
    """Test _discover_agent_card successfully discovers and caches agent."""
    provider = A2AClientToolProvider()
    provider._initial_discovery_done = False
    mock_resolver = Mock()
    mock_agent_card = Mock()
    mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
    mock_create_resolver.return_value = mock_resolver

    result = await provider._discover_agent_card("http://test.com")

    assert result == mock_agent_card
    assert provider._discovered_agents["http://test.com"] == mock_agent_card
    mock_resolver.get_agent_card.assert_called_once()


@pytest.mark.asyncio
async def test_discover_agent_card_cached():
    """Test _discover_agent_card returns cached agent."""
    provider = A2AClientToolProvider()
    provider._initial_discovery_done = False
    cached_card = Mock()
    provider._discovered_agents["http://test.com"] = cached_card

    result = await provider._discover_agent_card("http://test.com")

    assert result == cached_card


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_discover_agent_card")
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
@patch.object(A2AClientToolProvider, "_discover_agent_card")
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


@pytest.mark.asyncio
async def test_discover_agent_success():
    """Test a2a_discover_agent tool returns success result."""
    provider = A2AClientToolProvider()

    with patch.object(provider, "_discover_agent_card_tool") as mock_discover_tool:
        expected_result = {"status": "success", "agent_card": {"name": "test_agent"}, "url": "http://test.com"}
        mock_discover_tool.return_value = expected_result

        result = await provider.a2a_discover_agent("http://test.com")

        assert result == expected_result
        mock_discover_tool.assert_called_once_with("http://test.com")


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_discover_agent_card")
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_discover_agent_card_tool_success(mock_ensure, mock_discover):
    """Test _discover_agent_card_tool returns success result."""
    provider = A2AClientToolProvider()
    mock_agent_card = Mock()
    mock_agent_card.model_dump.return_value = {"name": "test_agent"}
    mock_discover.return_value = mock_agent_card

    result = await provider._discover_agent_card_tool("http://test.com")

    expected = {"status": "success", "agent_card": {"name": "test_agent"}, "url": "http://test.com"}
    assert result == expected
    mock_ensure.assert_called_once()


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_discover_agent_card")
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_discover_agent_card_tool_error(mock_ensure, mock_discover):
    """Test _discover_agent_card_tool handles errors."""
    provider = A2AClientToolProvider()
    mock_discover.side_effect = Exception("Network error")

    result = await provider._discover_agent_card_tool("http://test.com")

    expected = {"status": "error", "error": "Network error", "url": "http://test.com"}
    assert result == expected


@pytest.mark.asyncio
async def test_list_discovered_agents_empty():
    """Test a2a_list_discovered_agents with no discovered agents."""
    provider = A2AClientToolProvider()

    with patch.object(provider, "_list_discovered_agents") as mock_list_agents:
        expected = {"status": "success", "agents": [], "total_count": 0}
        mock_list_agents.return_value = expected

        result = await provider.a2a_list_discovered_agents()

        assert result == expected
        mock_list_agents.assert_called_once()


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_list_discovered_agents_with_agents(mock_ensure):
    """Test _list_discovered_agents with discovered agents."""
    provider = A2AClientToolProvider()
    mock_card1 = Mock()
    mock_card1.model_dump.return_value = {"name": "agent1"}
    mock_card2 = Mock()
    mock_card2.model_dump.return_value = {"name": "agent2"}

    provider._discovered_agents = {"http://agent1.com": mock_card1, "http://agent2.com": mock_card2}

    result = await provider._list_discovered_agents()

    expected = {"status": "success", "agents": [{"name": "agent1"}, {"name": "agent2"}], "total_count": 2}
    assert result == expected
    mock_ensure.assert_called_once()


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_list_discovered_agents_error(mock_ensure):
    """Test _list_discovered_agents handles errors."""
    provider = A2AClientToolProvider()
    mock_card = Mock()
    mock_card.model_dump.side_effect = Exception("Serialization error")
    provider._discovered_agents = {"http://test.com": mock_card}

    result = await provider._list_discovered_agents()

    expected = {"status": "error", "error": "Serialization error", "total_count": 0}
    assert result == expected


@pytest.mark.asyncio
async def test_send_message_with_message_id():
    """Test a2a_send_message with provided message_id."""
    provider = A2AClientToolProvider()

    with patch.object(provider, "_send_message") as mock_send_message:
        expected_result = {
            "status": "success",
            "response": {"result": "ok"},
            "message_id": "test_id",
            "target_agent_url": "http://test.com",
        }
        mock_send_message.return_value = expected_result

        result = await provider.a2a_send_message("Hello", "http://test.com", "test_id")

        assert result == expected_result
        mock_send_message.assert_called_once_with("Hello", "http://test.com", "test_id")


@pytest.mark.asyncio
async def test_send_message_without_message_id():
    """Test a2a_send_message without message_id (auto-generated)."""
    provider = A2AClientToolProvider()

    with patch.object(provider, "_send_message") as mock_send_message:
        expected_result = {
            "status": "success",
            "response": {"result": "ok"},
            "message_id": "auto_generated",
            "target_agent_url": "http://test.com",
        }
        mock_send_message.return_value = expected_result

        result = await provider.a2a_send_message("Hello", "http://test.com")

        assert result == expected_result
        mock_send_message.assert_called_once_with("Hello", "http://test.com", None)


@pytest.mark.asyncio
@patch("strands_tools.a2a_client.uuid4")
@patch.object(A2AClientToolProvider, "_discover_agent_card")
@patch.object(A2AClientToolProvider, "_ensure_client_factory")
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_send_message_success(mock_ensure, mock_factory, mock_discover, mock_uuid):
    """Test _send_message successful message sending."""
    provider = A2AClientToolProvider()

    # Mock UUID generation
    mock_message_uuid = Mock()
    mock_message_uuid.hex = "message_id_123"
    mock_uuid.return_value = mock_message_uuid

    # Mock agent card
    mock_agent_card = Mock()
    mock_discover.return_value = mock_agent_card

    # Mock ClientFactory and Client
    mock_client_factory = Mock()
    mock_client = Mock()
    mock_factory.return_value = mock_client_factory
    mock_client_factory.create.return_value = mock_client

    # Mock client response - simulate Message response
    mock_response = Mock(spec=Message)
    mock_response.model_dump.return_value = {"result": "success"}

    async def mock_send_message_iter(message):
        yield mock_response

    mock_client.send_message = mock_send_message_iter

    result = await provider._send_message("Hello world", "http://test.com", None)

    expected = {
        "status": "success",
        "response": {"result": "success"},
        "message_id": "message_id_123",
        "target_agent_url": "http://test.com",
    }
    assert result == expected
    mock_ensure.assert_called_once()
    mock_discover.assert_called_once_with("http://test.com")
    mock_client_factory.create.assert_called_once_with(mock_agent_card)


@pytest.mark.asyncio
@patch.object(A2AClientToolProvider, "_discover_agent_card")
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_send_message_error(mock_ensure, mock_discover):
    """Test _send_message handles errors."""
    provider = A2AClientToolProvider()
    mock_discover.side_effect = Exception("Connection failed")

    result = await provider._send_message("Hello", "http://test.com", "test_id")

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
async def test_ensure_client_factory(mock_ensure_client):
    """Test _ensure_client_factory creates ClientFactory with correct parameters."""
    provider = A2AClientToolProvider()
    mock_client = Mock()
    mock_ensure_client.return_value = mock_client

    with patch("strands_tools.a2a_client.ClientFactory") as mock_factory_class:
        with patch("strands_tools.a2a_client.ClientConfig") as mock_config_class:
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_factory = Mock()
            mock_factory_class.return_value = mock_factory

            result = await provider._ensure_client_factory()

            mock_config_class.assert_called_once()
            mock_factory_class.assert_called_once_with(mock_config)
            assert result == mock_factory
            assert provider._client_factory == mock_factory


@pytest.mark.asyncio
@patch("strands_tools.a2a_client.uuid4")
@patch.object(A2AClientToolProvider, "_discover_agent_card")
@patch.object(A2AClientToolProvider, "_ensure_client_factory")
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_send_message_task_response(mock_ensure, mock_factory, mock_discover, mock_uuid):
    """Test _send_message handling task response from ClientFactory."""
    provider = A2AClientToolProvider()

    # Mock UUID generation
    mock_message_uuid = Mock()
    mock_message_uuid.hex = "message_id_123"
    mock_uuid.return_value = mock_message_uuid

    # Mock agent card
    mock_agent_card = Mock()
    mock_discover.return_value = mock_agent_card

    # Mock ClientFactory and Client
    mock_client_factory = Mock()
    mock_client = Mock()
    mock_factory.return_value = mock_client_factory
    mock_client_factory.create.return_value = mock_client

    # Mock client response - simulate (Task, UpdateEvent) tuple response
    mock_task = Mock()
    mock_task.model_dump.return_value = {"task_id": "123", "status": "completed"}
    mock_update_event = Mock()
    mock_update_event.model_dump.return_value = {"event": "finished"}

    async def mock_send_message_iter(message):
        yield (mock_task, mock_update_event)

    mock_client.send_message = mock_send_message_iter

    result = await provider._send_message("Hello world", "http://test.com", None)

    expected = {
        "status": "success",
        "response": {"task": {"task_id": "123", "status": "completed"}, "update": {"event": "finished"}},
        "message_id": "message_id_123",
        "target_agent_url": "http://test.com",
    }
    assert result == expected
    mock_ensure.assert_called_once()
    mock_discover.assert_called_once_with("http://test.com")
    mock_client_factory.create.assert_called_once_with(mock_agent_card)


@pytest.mark.asyncio
@patch("strands_tools.a2a_client.uuid4")
@patch.object(A2AClientToolProvider, "_discover_agent_card")
@patch.object(A2AClientToolProvider, "_ensure_client_factory")
@patch.object(A2AClientToolProvider, "_ensure_discovered_known_agents")
async def test_send_message_task_response_no_update(mock_ensure, mock_factory, mock_discover, mock_uuid):
    """Test _send_message handling task response with no update event."""
    provider = A2AClientToolProvider()

    # Mock UUID generation
    mock_message_uuid = Mock()
    mock_message_uuid.hex = "message_id_123"
    mock_uuid.return_value = mock_message_uuid

    # Mock agent card
    mock_agent_card = Mock()
    mock_discover.return_value = mock_agent_card

    # Mock ClientFactory and Client
    mock_client_factory = Mock()
    mock_client = Mock()
    mock_factory.return_value = mock_client_factory
    mock_client_factory.create.return_value = mock_client

    # Mock client response - simulate (Task, None) tuple response
    mock_task = Mock()
    mock_task.model_dump.return_value = {"task_id": "123", "status": "completed"}

    async def mock_send_message_iter(message):
        yield (mock_task, None)

    mock_client.send_message = mock_send_message_iter

    result = await provider._send_message("Hello world", "http://test.com", None)

    expected = {
        "status": "success",
        "response": {"task": {"task_id": "123", "status": "completed"}, "update": None},
        "message_id": "message_id_123",
        "target_agent_url": "http://test.com",
    }
    assert result == expected
