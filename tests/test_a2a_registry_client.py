from unittest.mock import AsyncMock, Mock, patch

import pytest
from strands_tools.a2a_registry_client import AgentRegistryToolProvider


def test_init_default_parameters():
    """Test initialization with default parameters."""
    provider = AgentRegistryToolProvider()

    assert provider.registry_url == "http://localhost:8000"
    assert provider.timeout == 30
    assert provider.agent_auth is None
    assert provider.transports == {}
    assert provider._httpx_client is None
    assert provider._client_factory is None
    assert provider._request_id == 0
    assert provider._agent_cache == {}


def test_init_custom_parameters():
    """Test initialization with custom parameters."""
    registry_url = "http://custom-registry.com"
    timeout = 60
    transports = {"custom": Mock()}

    provider = AgentRegistryToolProvider(registry_url=registry_url, timeout=timeout, transports=transports)

    assert provider.registry_url == registry_url
    assert provider.timeout == timeout
    assert provider.transports == transports


def test_tools_property():
    """Test that tools property returns decorated methods."""
    provider = AgentRegistryToolProvider()
    tools = provider.tools

    tool_names = [tool.tool_name for tool in tools]
    assert "registry_send_message_to_agent" in tool_names
    assert "registry_find_and_message_agent" in tool_names
    assert "registry_find_agents_by_skill" in tool_names
    assert "registry_get_all_agents" in tool_names
    assert "registry_find_best_agent_for_task" in tool_names
    assert "registry_find_similar_agents" in tool_names


@pytest.mark.asyncio
async def test_ensure_httpx_client_creates_new_client():
    """Test _ensure_httpx_client creates new client when none exists."""
    provider = AgentRegistryToolProvider(timeout=45)

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
    provider = AgentRegistryToolProvider()
    existing_client = Mock()
    provider._httpx_client = existing_client

    result = await provider._ensure_httpx_client()

    assert result == existing_client


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_ensure_httpx_client")
async def test_ensure_client_factory_with_auth(mock_ensure_client):
    """Test _ensure_client_factory with authentication."""
    provider = AgentRegistryToolProvider()
    mock_httpx_client = Mock()
    mock_ensure_client.return_value = mock_httpx_client
    mock_auth = Mock()
    mock_auth.__name__ = "MockAuth"
    provider.agent_auth = mock_auth

    mock_agent_card = Mock()
    mock_agent_card.name = "test_agent"

    with patch("strands_tools.a2a_registry_client.ClientFactory") as mock_factory_class:
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory

        result = await provider._ensure_client_factory(mock_agent_card)

        assert result == mock_factory
        assert mock_httpx_client.auth == mock_auth.return_value


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_jsonrpc_request")
async def test_get_agent_card_from_registry_success(mock_jsonrpc):
    """Test _get_agent_card_from_registry with successful response."""
    provider = AgentRegistryToolProvider()
    agent_data = {"name": "test_agent", "url": "http://test.com"}
    mock_jsonrpc.return_value = {"found": True, "agent_card": agent_data}

    with patch("strands_tools.a2a_registry_client.AgentCard") as mock_agent_card:
        mock_card = Mock()
        mock_agent_card.return_value = mock_card

        result = await provider._get_agent_card_from_registry("test_agent")

        assert result == mock_card
        mock_jsonrpc.assert_called_once_with("get_agent", {"agent_id": "test_agent"})
        mock_agent_card.assert_called_once_with(**agent_data)


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_jsonrpc_request")
async def test_get_agent_card_from_registry_not_found(mock_jsonrpc):
    """Test _get_agent_card_from_registry when agent not found."""
    provider = AgentRegistryToolProvider()
    mock_jsonrpc.return_value = {"found": False}

    result = await provider._get_agent_card_from_registry("test_agent")

    assert result is None


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_jsonrpc_request")
async def test_get_agent_card_from_registry_error(mock_jsonrpc):
    """Test _get_agent_card_from_registry handles errors."""
    provider = AgentRegistryToolProvider()
    mock_jsonrpc.side_effect = Exception("Network error")

    result = await provider._get_agent_card_from_registry("test_agent")

    assert result is None


def test_next_id():
    """Test _next_id increments request ID."""
    provider = AgentRegistryToolProvider()

    assert provider._next_id() == 1
    assert provider._next_id() == 2
    assert provider._request_id == 2


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_ensure_httpx_client")
async def test_jsonrpc_request_success(mock_ensure_client):
    """Test _jsonrpc_request with successful response."""
    provider = AgentRegistryToolProvider()
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"jsonrpc": "2.0", "result": {"data": "test"}, "id": 1}
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_ensure_client.return_value = mock_client

    result = await provider._jsonrpc_request("test_method", {"param": "value"})

    assert result == {"data": "test"}
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_ensure_httpx_client")
async def test_jsonrpc_request_error_response(mock_ensure_client):
    """Test _jsonrpc_request with error response."""
    provider = AgentRegistryToolProvider()
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"jsonrpc": "2.0", "error": {"code": -1, "message": "Test error"}, "id": 1}
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_ensure_client.return_value = mock_client

    with pytest.raises(Exception, match="JSON-RPC Error"):
        await provider._jsonrpc_request("test_method")


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_get_agent_card_from_registry")
@patch.object(AgentRegistryToolProvider, "_send_message_to_agent_direct")
async def test_registry_send_message_to_agent_success(mock_send_message, mock_get_agent):
    """Test registry_send_message_to_agent with successful flow."""
    provider = AgentRegistryToolProvider()
    mock_agent_card = Mock()
    mock_agent_card.model_dump.return_value = {"name": "test_agent"}
    mock_get_agent.return_value = mock_agent_card
    mock_send_message.return_value = {"status": "success", "response": "test response"}

    result = await provider.registry_send_message_to_agent("test_agent", "Hello")

    assert result["status"] == "success"
    mock_get_agent.assert_called_once_with("test_agent")
    mock_send_message.assert_called_once()


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_get_agent_card_from_registry")
async def test_registry_send_message_to_agent_not_found(mock_get_agent):
    """Test registry_send_message_to_agent when agent not found."""
    provider = AgentRegistryToolProvider()
    mock_get_agent.return_value = None

    result = await provider.registry_send_message_to_agent("test_agent", "Hello")

    assert result["status"] == "error"
    assert "not found in registry" in result["error"]


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_jsonrpc_request")
async def test_registry_find_agents_by_skill_success(mock_jsonrpc):
    """Test registry_find_agents_by_skill with successful response."""
    provider = AgentRegistryToolProvider()
    agents_data = [{"name": "agent1"}, {"name": "agent2"}]
    mock_jsonrpc.return_value = {"agents": agents_data}

    result = await provider.registry_find_agents_by_skill("python")

    assert result["status"] == "success"
    assert result["agents"] == agents_data
    assert result["skill_searched"] == "python"
    assert result["total_count"] == 2


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_jsonrpc_request")
async def test_registry_get_all_agents_success(mock_jsonrpc):
    """Test registry_get_all_agents with successful response."""
    provider = AgentRegistryToolProvider()
    agents_data = [{"name": "agent1"}, {"name": "agent2"}]
    mock_jsonrpc.return_value = {"agents": agents_data}

    result = await provider.registry_get_all_agents()

    assert result["status"] == "success"
    assert result["agents"] == agents_data
    assert result["total_count"] == 2


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_jsonrpc_request")
async def test_registry_find_best_agent_for_task_success(mock_jsonrpc):
    """Test registry_find_best_agent_for_task finds compatible agent."""
    provider = AgentRegistryToolProvider()
    agents_data = [
        {"name": "agent1", "skills": [{"id": "python"}, {"id": "web"}]},
        {"name": "agent2", "skills": [{"id": "python"}]},
    ]
    mock_jsonrpc.return_value = {"agents": agents_data}

    result = await provider.registry_find_best_agent_for_task(["python"])

    assert result["status"] == "success"
    assert result["best_agent"]["name"] == "agent1"  # Has more skills
    assert result["total_compatible"] == 2


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_jsonrpc_request")
async def test_registry_find_best_agent_for_task_no_match(mock_jsonrpc):
    """Test registry_find_best_agent_for_task when no agents match."""
    provider = AgentRegistryToolProvider()
    agents_data = [{"name": "agent1", "skills": [{"id": "java"}]}]
    mock_jsonrpc.return_value = {"agents": agents_data}

    result = await provider.registry_find_best_agent_for_task(["python"])

    assert result["status"] == "success"
    assert result["best_agent"] is None
    assert "No agents found" in result["message"]


@pytest.mark.asyncio
@patch.object(AgentRegistryToolProvider, "_jsonrpc_request")
async def test_registry_find_similar_agents_success(mock_jsonrpc):
    """Test registry_find_similar_agents finds similar agents."""
    provider = AgentRegistryToolProvider()
    reference_agent = {"name": "ref_agent", "skills": [{"id": "python"}, {"id": "web"}]}
    all_agents = [{"name": "agent1", "skills": [{"id": "python"}]}, {"name": "agent2", "skills": [{"id": "java"}]}]

    mock_jsonrpc.side_effect = [
        {"agent_card": reference_agent},  # get_agent call
        {"agents": all_agents},  # list_agents call
    ]

    result = await provider.registry_find_similar_agents("ref_agent")

    assert result["status"] == "success"
    assert len(result["similar_agents"]) == 1  # Only agent1 has overlap
    assert result["similar_agents"][0]["name"] == "agent1"
    assert "similarity_score" in result["similar_agents"][0]


@pytest.mark.asyncio
async def test_registry_find_and_message_agent_success():
    """Test registry_find_and_message_agent with successful flow."""
    provider = AgentRegistryToolProvider()
    best_agent_data = {"name": "best_agent"}

    with patch.object(provider, "registry_find_best_agent_for_task", new_callable=AsyncMock) as mock_find_best:
        with patch.object(provider, "_send_message_to_agent_direct", new_callable=AsyncMock) as mock_send_message:
            mock_find_best.return_value = {"status": "success", "best_agent": best_agent_data}
            mock_send_message.return_value = {"status": "success", "response": "test response"}

            result = await provider.registry_find_and_message_agent(["python"], "Hello")

            assert result["status"] == "success"
            assert result["selected_agent"] == "best_agent"
            mock_find_best.assert_called_once_with(["python"])
            mock_send_message.assert_called_once_with(best_agent_data, "Hello")


@pytest.mark.asyncio
async def test_registry_find_and_message_agent_no_agent_found():
    """Test registry_find_and_message_agent when no suitable agent found."""
    provider = AgentRegistryToolProvider()

    with patch.object(provider, "registry_find_best_agent_for_task", new_callable=AsyncMock) as mock_find_best:
        mock_find_best.return_value = {"status": "success", "best_agent": None}

        result = await provider.registry_find_and_message_agent(["python"], "Hello")

        assert result["status"] == "error"
        assert "No suitable agent found" in result["error"]
