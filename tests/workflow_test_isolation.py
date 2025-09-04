"""
Comprehensive workflow test isolation utilities.

This module provides utilities to completely isolate workflow tests
by mocking all threading and file system components that can cause hanging.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock


class MockObserver:
    """Mock Observer that doesn't create real threads."""
    
    def __init__(self):
        self.started = False
        self.stopped = False
    
    def schedule(self, *args, **kwargs):
        pass
    
    def start(self):
        self.started = True
    
    def stop(self):
        self.stopped = True
    
    def join(self, timeout=None):
        pass


class MockThreadPoolExecutor:
    """Mock ThreadPoolExecutor that doesn't create real threads."""
    
    def __init__(self, *args, **kwargs):
        self.shutdown_called = False
    
    def submit(self, fn, *args, **kwargs):
        # Execute immediately in the same thread for testing
        try:
            result = fn(*args, **kwargs)
            future = Mock()
            future.result.return_value = result
            future.done.return_value = True
            return future
        except Exception as e:
            future = Mock()
            future.exception.return_value = e
            future.done.return_value = True
            return future
    
    def shutdown(self, wait=True):
        self.shutdown_called = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()


class MockLock:
    """Mock lock that supports context manager protocol."""
    def __init__(self, *args, **kwargs):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def acquire(self, blocking=True, timeout=-1):
        return True
    
    def release(self):
        pass
    
    def wait(self, timeout=None):
        return True
    
    def notify(self, n=1):
        pass
    
    def notify_all(self):
        pass


class MockEvent:
    """Mock event for threading."""
    def __init__(self):
        self._is_set = False
    
    def set(self):
        self._is_set = True
    
    def clear(self):
        self._is_set = False
    
    def is_set(self):
        return self._is_set
    
    def wait(self, timeout=None):
        return self._is_set


class MockSemaphore:
    """Mock semaphore for threading."""
    def __init__(self, value=1):
        self._value = value
    
    def acquire(self, blocking=True, timeout=None):
        return True
    
    def release(self):
        pass
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class MockQueue:
    """Mock queue for threading."""
    def __init__(self, maxsize=0):
        self._items = []
    
    def put(self, item, block=True, timeout=None):
        self._items.append(item)
    
    def get(self, block=True, timeout=None):
        if self._items:
            return self._items.pop(0)
        raise Exception("Queue is empty")
    
    def empty(self):
        return len(self._items) == 0
    
    def qsize(self):
        return len(self._items)


@pytest.fixture(autouse=True)
def mock_workflow_threading_components(request):
    """
    Mock all threading components in workflow tests to prevent hanging.
    
    This fixture automatically mocks Observer, ThreadPoolExecutor, and other
    threading components that can cause tests to hang.
    """
    # Only apply to workflow tests
    test_file = str(request.fspath)
    if 'workflow' not in test_file.lower():
        yield
        return
    
    # Create a temporary directory for workflow files
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_workflow_dir = Path(temp_dir)
        
        # Mock all the threading components and WORKFLOW_DIR
        with patch('watchdog.observers.Observer', MockObserver), \
             patch('watchdog.observers.fsevents.FSEventsObserver', MockObserver), \
             patch('src.strands_tools.workflow.Observer', MockObserver), \
             patch('strands_tools.workflow.Observer', MockObserver), \
             patch('concurrent.futures.ThreadPoolExecutor', MockThreadPoolExecutor), \
             patch('src.strands_tools.workflow.ThreadPoolExecutor', MockThreadPoolExecutor), \
             patch('strands_tools.workflow.ThreadPoolExecutor', MockThreadPoolExecutor), \
             patch('src.strands_tools.workflow.WORKFLOW_DIR', temp_workflow_dir), \
             patch('strands_tools.workflow.WORKFLOW_DIR', temp_workflow_dir), \
             patch('threading.Lock', MockLock), \
             patch('threading.RLock', MockLock), \
             patch('threading.Event', MockEvent), \
             patch('threading.Semaphore', MockSemaphore), \
             patch('threading.Condition', MockLock), \
             patch('time.sleep', Mock()), \
             patch('queue.Queue', MockQueue):
            
            yield


@pytest.fixture
def isolated_workflow_environment():
    """
    Create a completely isolated workflow environment for testing.
    
    This fixture provides a clean environment with all global state reset
    and all threading components mocked.
    """
    # Import and reset workflow modules
    try:
        import strands_tools.workflow as workflow_module
        import src.strands_tools.workflow as src_workflow_module
    except ImportError:
        workflow_module = None
        src_workflow_module = None
    
    # Store original state
    original_state = {}
    
    for module in [workflow_module, src_workflow_module]:
        if module is None:
            continue
            
        original_state[module] = {}
        
        # Store and reset global state
        if hasattr(module, '_manager'):
            original_state[module]['_manager'] = module._manager
            module._manager = None
        
        if hasattr(module, '_last_request_time'):
            original_state[module]['_last_request_time'] = module._last_request_time
            module._last_request_time = 0
        
        # Store and reset WorkflowManager class state
        if hasattr(module, 'WorkflowManager'):
            wm = module.WorkflowManager
            original_state[module]['WorkflowManager'] = {
                '_instance': getattr(wm, '_instance', None),
                '_workflows': getattr(wm, '_workflows', {}).copy(),
                '_observer': getattr(wm, '_observer', None),
                '_watch_paths': getattr(wm, '_watch_paths', set()).copy(),
            }
            
            # Force cleanup of existing instance
            if hasattr(wm, '_instance') and wm._instance:
                try:
                    if hasattr(wm._instance, 'cleanup'):
                        wm._instance.cleanup()
                except:
                    pass
            
            wm._instance = None
            wm._workflows = {}
            wm._observer = None
            wm._watch_paths = set()
    
    yield
    
    # Restore original state
    for module in [workflow_module, src_workflow_module]:
        if module is None or module not in original_state:
            continue
            
        state = original_state[module]
        
        # Restore global state
        if '_manager' in state:
            module._manager = state['_manager']
        
        if '_last_request_time' in state:
            module._last_request_time = state['_last_request_time']
        
        # Restore WorkflowManager class state
        if 'WorkflowManager' in state and hasattr(module, 'WorkflowManager'):
            wm = module.WorkflowManager
            wm_state = state['WorkflowManager']
            
            wm._instance = wm_state['_instance']
            wm._workflows = wm_state['_workflows']
            wm._observer = wm_state['_observer']
            wm._watch_paths = wm_state['_watch_paths']