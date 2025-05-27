from unittest.mock import MagicMock, patch

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--integration", action="store_true", default=False, help="Run integration tests that require Slack API access"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as requiring Slack API access")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --integration is specified."""
    if not config.getoption("--integration"):
        skip_integration = pytest.mark.skip(reason="Need --integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def mock_env_slack_tokens(monkeypatch):
    """Fixture to set mock Slack tokens in the environment."""
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test-token")
    monkeypatch.setenv("SLACK_TEST_CHANNEL", "C123TEST")

    # Automatically apply this fixture to any test containing "slack" in the module name
    yield
    # The monkeypatch fixture handles cleanup automatically


@pytest.fixture
def mock_slack_client():
    """
    Fixture to create a mock Slack client with dynamic attribute support.

    This is a more robust way to handle dynamic method access in Slack clients.
    """
    client = MagicMock()

    # Add dynamic method support without using __getattr__
    client.chat_postMessage = MagicMock()
    client.chat_postMessage.return_value = MagicMock(
        data={"ok": True, "ts": "1234.5678", "message": {"text": "Test message"}}
    )

    client.reactions_add = MagicMock()
    client.reactions_add.return_value = MagicMock(data={"ok": True})

    client.reactions_remove = MagicMock()
    client.reactions_remove.return_value = MagicMock(data={"ok": True})

    client.conversations_list = MagicMock()
    client.conversations_list.return_value = MagicMock(
        data={"ok": True, "channels": [{"id": "C123456", "name": "general"}]}
    )

    client.files_upload = MagicMock()
    client.files_upload.return_value = MagicMock(
        data={"ok": True, "file": {"id": "F123", "permalink": "https://test.com"}}
    )

    client.files_upload_v2 = MagicMock()
    client.files_upload_v2.return_value = MagicMock(
        data={"ok": True, "file": {"id": "F123", "permalink": "https://test.com"}}
    )

    client.auth_test = MagicMock()
    client.auth_test.return_value = {"ok": True, "user_id": "U123BOT", "bot_id": "B123"}

    # Add a special _method_missing attribute to handle dynamic method calls
    def _handle_method(name, *args, **kwargs):
        if not hasattr(client, name):
            dynamic_method = MagicMock()
            dynamic_method.return_value = MagicMock(data={"ok": True})
            setattr(client, name, dynamic_method)
        return getattr(client, name)(*args, **kwargs)

    client._method_missing = _handle_method

    return client


@pytest.fixture
def mock_slack_app():
    """Fixture to create a mock Slack app."""
    app = MagicMock()
    return app


@pytest.fixture
def mock_slack_socket_client():
    """Fixture to create a mock Slack socket client."""
    socket_client = MagicMock()
    socket_client.socket_mode_request_listeners = []
    return socket_client


@pytest.fixture
def mock_slack_response():
    """Fixture to create a mock Slack response."""
    response = MagicMock()
    response.data = {"ok": True, "ts": "1234.5678"}
    return response


@pytest.fixture(autouse=True)
def patch_slack_client_in_module():
    """
    Automatically patch the slack client in all tests.

    This is especially important for the slack module tests.
    """
    try:
        with (
            patch("strands_tools.slack.app", new=MagicMock()) as mock_app,
            patch("strands_tools.slack.client", new=MagicMock()) as mock_client,
            patch("strands_tools.slack.socket_client", new=MagicMock()) as mock_socket,
        ):
            # Configure client attributes
            mock_client.chat_postMessage = MagicMock()
            mock_client.chat_postMessage.return_value = MagicMock(
                data={"ok": True, "ts": "1234.5678", "message": {"text": "Test message"}}
            )

            mock_client.reactions_add = MagicMock()
            mock_client.reactions_add.return_value = MagicMock(data={"ok": True})

            mock_client.reactions_remove = MagicMock()
            mock_client.reactions_remove.return_value = MagicMock(data={"ok": True})

            mock_client.conversations_list = MagicMock()
            mock_client.conversations_list.return_value = MagicMock(
                data={"ok": True, "channels": [{"id": "C123456", "name": "general"}]}
            )

            mock_client.auth_test = MagicMock()
            mock_client.auth_test.return_value = {"user_id": "U123BOT", "bot_id": "B123"}

            # Configure socket client
            mock_socket.socket_mode_request_listeners = []

            yield mock_app, mock_client, mock_socket
    except (ImportError, AttributeError):
        # Module not loaded or attribute not found, skip patching
        yield None, None, None


@pytest.fixture
def mock_slack_initialize_clients():
    """Fixture to mock the initialize_slack_clients function."""
    with patch("strands_tools.slack.initialize_slack_clients") as mock_init:
        mock_init.return_value = (True, None)
        yield mock_init
