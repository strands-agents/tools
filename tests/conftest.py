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


@pytest.fixture(autouse=True)
def reset_workflow_global_state(request):
    """
    Comprehensive fixture to reset all workflow global state before each test.
    
    This fixture is automatically applied to all tests to prevent workflow tests
    from interfering with each other when run in parallel or in sequence.
    """
    # Only reset workflow state for workflow-related tests
    test_file = str(request.fspath)
    if 'workflow' not in test_file.lower():
        # Not a workflow test, skip the reset
        yield
        return
    
    # Import workflow module
    try:
        import strands_tools.workflow as workflow_module
        import src.strands_tools.workflow as src_workflow_module
    except ImportError:
        yield
        return
    
    # Aggressive cleanup of any existing state before test
    for module in [workflow_module, src_workflow_module]:
        try:
            # Force cleanup any existing managers and their resources
            if hasattr(module, '_manager') and module._manager:
                try:
                    if hasattr(module._manager, 'cleanup'):
                        module._manager.cleanup()
                    if hasattr(module._manager, '_executor'):
                        module._manager._executor.shutdown(wait=False)
                except:
                    pass
            
            if hasattr(module, 'WorkflowManager') and hasattr(module.WorkflowManager, '_instance') and module.WorkflowManager._instance:
                try:
                    if hasattr(module.WorkflowManager._instance, 'cleanup'):
                        module.WorkflowManager._instance.cleanup()
                    # Force stop any observers
                    if hasattr(module.WorkflowManager._instance, '_observer') and module.WorkflowManager._instance._observer:
                        try:
                            module.WorkflowManager._instance._observer.stop()
                            module.WorkflowManager._instance._observer.join(timeout=0.1)
                        except:
                            pass
                    # Force shutdown any executors
                    if hasattr(module.WorkflowManager._instance, '_executor'):
                        try:
                            module.WorkflowManager._instance._executor.shutdown(wait=False)
                        except:
                            pass
                except:
                    pass
        except:
            pass
    
    # Reset all global state variables for both import paths
    for module in [workflow_module, src_workflow_module]:
        if hasattr(module, '_manager'):
            module._manager = None
        if hasattr(module, '_last_request_time'):
            module._last_request_time = 0
        
        # Reset WorkflowManager class state if it exists
        if hasattr(module, 'WorkflowManager'):
            if hasattr(module.WorkflowManager, '_instance'):
                module.WorkflowManager._instance = None
            if hasattr(module.WorkflowManager, '_workflows'):
                module.WorkflowManager._workflows = {}
            if hasattr(module.WorkflowManager, '_observer'):
                module.WorkflowManager._observer = None
            if hasattr(module.WorkflowManager, '_watch_paths'):
                module.WorkflowManager._watch_paths = set()
        
        # Reset TaskExecutor class state if it exists
        if hasattr(module, 'TaskExecutor'):
            # Force cleanup any class-level executors
            try:
                if hasattr(module.TaskExecutor, '_executor'):
                    module.TaskExecutor._executor.shutdown(wait=False)
                    module.TaskExecutor._executor = None
            except:
                pass
    
    yield
    
    # Aggressive cleanup after test
    for module in [workflow_module, src_workflow_module]:
        try:
            # Cleanup any active managers
            if hasattr(module, '_manager') and module._manager:
                try:
                    if hasattr(module._manager, 'cleanup'):
                        module._manager.cleanup()
                    if hasattr(module._manager, '_executor'):
                        module._manager._executor.shutdown(wait=False)
                except:
                    pass
            
            if hasattr(module, 'WorkflowManager') and hasattr(module.WorkflowManager, '_instance') and module.WorkflowManager._instance:
                try:
                    if hasattr(module.WorkflowManager._instance, 'cleanup'):
                        module.WorkflowManager._instance.cleanup()
                    # Force stop any observers
                    if hasattr(module.WorkflowManager._instance, '_observer') and module.WorkflowManager._instance._observer:
                        try:
                            module.WorkflowManager._instance._observer.stop()
                            module.WorkflowManager._instance._observer.join(timeout=0.1)
                        except:
                            pass
                    # Force shutdown any executors
                    if hasattr(module.WorkflowManager._instance, '_executor'):
                        try:
                            module.WorkflowManager._instance._executor.shutdown(wait=False)
                        except:
                            pass
                except:
                    pass
        except Exception:
            pass
        
        # Reset state again after cleanup
        if hasattr(module, '_manager'):
            module._manager = None
        if hasattr(module, '_last_request_time'):
            module._last_request_time = 0
        
        if hasattr(module, 'WorkflowManager'):
            if hasattr(module.WorkflowManager, '_instance'):
                module.WorkflowManager._instance = None
            if hasattr(module.WorkflowManager, '_workflows'):
                module.WorkflowManager._workflows = {}
            if hasattr(module.WorkflowManager, '_observer'):
                module.WorkflowManager._observer = None
            if hasattr(module.WorkflowManager, '_watch_paths'):
                module.WorkflowManager._watch_paths = set()
        
        # Reset TaskExecutor class state if it exists
        if hasattr(module, 'TaskExecutor'):
            try:
                if hasattr(module.TaskExecutor, '_executor'):
                    module.TaskExecutor._executor.shutdown(wait=False)
                    module.TaskExecutor._executor = None
            except:
                pass
