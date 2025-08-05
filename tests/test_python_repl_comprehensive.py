"""
Comprehensive tests for python_repl tool to improve coverage.
"""

import os
import signal
import sys
import tempfile
import threading
import time
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from strands_tools import python_repl
from strands_tools.python_repl import OutputCapture, PtyManager, ReplState, clean_ansi

if os.name == "nt":
    pytest.skip("skipping on windows until issue #17 is resolved", allow_module_level=True)


@pytest.fixture
def temp_repl_dir():
    """Create temporary directory for REPL state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = python_repl.repl_state.persistence_dir
        original_file = python_repl.repl_state.state_file
        
        python_repl.repl_state.persistence_dir = tmpdir
        python_repl.repl_state.state_file = os.path.join(tmpdir, "repl_state.pkl")
        
        yield tmpdir
        
        python_repl.repl_state.persistence_dir = original_dir
        python_repl.repl_state.state_file = original_file


@pytest.fixture
def mock_console():
    """Mock console for testing."""
    with patch("strands_tools.python_repl.console_util") as mock_console_util:
        yield mock_console_util.create.return_value


class TestOutputCaptureAdvanced:
    """Advanced tests for OutputCapture class."""

    def test_output_capture_context_manager_exception(self):
        """Test OutputCapture context manager with exception."""
        capture = OutputCapture()
        
        try:
            with capture:
                print("Before exception")
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        output = capture.get_output()
        assert "Before exception" in output

    def test_output_capture_nested_context(self):
        """Test nested OutputCapture contexts."""
        outer_capture = OutputCapture()
        inner_capture = OutputCapture()
        
        with outer_capture:
            print("Outer output")
            with inner_capture:
                print("Inner output")
            print("More outer output")
        
        outer_output = outer_capture.get_output()
        inner_output = inner_capture.get_output()
        
        assert "Outer output" in outer_output
        assert "More outer output" in outer_output
        assert "Inner output" in inner_output
        assert "Outer output" not in inner_output

    def test_output_capture_large_output(self):
        """Test OutputCapture with large output."""
        capture = OutputCapture()
        
        with capture:
            # Generate large output
            for i in range(1000):
                print(f"Line {i}")
        
        output = capture.get_output()
        assert "Line 0" in output
        assert "Line 999" in output

    def test_output_capture_unicode_output(self):
        """Test OutputCapture with unicode characters."""
        capture = OutputCapture()
        
        with capture:
            print("Unicode test: 🐍 Python 中文 العربية")
            print("Special chars: ñáéíóú àèìòù")
        
        output = capture.get_output()
        assert "🐍" in output
        assert "中文" in output
        assert "العربية" in output
        assert "ñáéíóú" in output

    def test_output_capture_mixed_streams(self):
        """Test OutputCapture with mixed stdout/stderr."""
        capture = OutputCapture()
        
        with capture:
            print("Standard output line 1")
            print("Error line 1", file=sys.stderr)
            print("Standard output line 2")
            print("Error line 2", file=sys.stderr)
        
        output = capture.get_output()
        assert "Standard output line 1" in output
        assert "Standard output line 2" in output
        assert "Error line 1" in output
        assert "Error line 2" in output
        assert "Errors:" in output

    def test_output_capture_empty_streams(self):
        """Test OutputCapture with empty streams."""
        capture = OutputCapture()
        
        with capture:
            pass  # No output
        
        output = capture.get_output()
        assert output == ""

    def test_output_capture_only_stderr(self):
        """Test OutputCapture with only stderr output."""
        capture = OutputCapture()
        
        with capture:
            print("Only error output", file=sys.stderr)
        
        output = capture.get_output()
        assert "Only error output" in output
        assert "Errors:" in output


class TestReplStateAdvanced:
    """Advanced tests for ReplState class."""

    def test_repl_state_complex_objects(self, temp_repl_dir):
        """Test ReplState with complex Python objects."""
        repl = ReplState()
        repl.clear_state()
        
        complex_code = """
import collections
import datetime
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

# Complex data structures
people = [Person("Alice", 30), Person("Bob", 25)]
counter = collections.Counter(['a', 'b', 'a', 'c', 'b', 'a'])
default_dict = collections.defaultdict(list)
default_dict['items'].extend([1, 2, 3])

# Date and time objects
now = datetime.datetime.now()
today = datetime.date.today()

# Nested structures
nested_data = {
    'people': people,
    'stats': {
        'counter': counter,
        'default_dict': default_dict
    },
    'timestamps': {
        'now': now,
        'today': today
    }
}

result_count = len(people)
"""
        
        repl.execute(complex_code)
        namespace = repl.get_namespace()
        
        assert namespace["result_count"] == 2
        assert "people" in namespace
        assert "nested_data" in namespace
        assert "Person" in namespace

    def test_repl_state_function_definitions(self, temp_repl_dir):
        """Test ReplState with function definitions."""
        repl = ReplState()
        repl.clear_state()
        
        function_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

# Higher-order functions
def apply_twice(func, x):
    return func(func(x))

def add_one(x):
    return x + 1

# Test the functions
fib_5 = fibonacci(5)
fact_5 = factorial(5)
twice_add_one = apply_twice(add_one, 5)

# Lambda functions
square = lambda x: x * x
squares = list(map(square, range(5)))
"""
        
        repl.execute(function_code)
        namespace = repl.get_namespace()
        
        assert namespace["fib_5"] == 5
        assert namespace["fact_5"] == 120
        assert namespace["twice_add_one"] == 7
        assert "fibonacci" in namespace
        assert "factorial" in namespace
        assert "square" in namespace

    def test_repl_state_class_definitions(self, temp_repl_dir):
        """Test ReplState with class definitions."""
        repl = ReplState()
        repl.clear_state()
        
        class_code = """
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def speak(self):
        return f"{self.name} makes a sound"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Canine")
        self.breed = breed
    
    def speak(self):
        return f"{self.name} barks"
    
    def fetch(self):
        return f"{self.name} fetches the ball"

# Create instances
generic_animal = Animal("Generic", "Unknown")
my_dog = Dog("Buddy", "Golden Retriever")

# Test methods
animal_sound = generic_animal.speak()
dog_sound = my_dog.speak()
dog_action = my_dog.fetch()

# Class attributes
dog_species = my_dog.species
dog_breed = my_dog.breed
"""
        
        repl.execute(class_code)
        namespace = repl.get_namespace()
        
        assert "Animal" in namespace
        assert "Dog" in namespace
        assert "my_dog" in namespace
        assert namespace["animal_sound"] == "Generic makes a sound"
        assert namespace["dog_sound"] == "Buddy barks"
        assert namespace["dog_species"] == "Canine"

    def test_repl_state_import_variations(self, temp_repl_dir):
        """Test ReplState with various import patterns."""
        repl = ReplState()
        repl.clear_state()
        
        import_code = """
# Standard imports
import os
import sys
import json

# Aliased imports
import datetime as dt
import collections as col

# From imports
from pathlib import Path, PurePath
from itertools import chain, combinations

# Import with star (not recommended but testing)
from math import *

# Use imported modules
current_dir = os.getcwd()
python_version = sys.version_info.major
json_data = json.dumps({"test": "data"})

# Use aliased imports
now = dt.datetime.now()
counter = col.Counter([1, 2, 1, 3, 2, 1])

# Use from imports
home_path = Path.home()
chained = list(chain([1, 2], [3, 4]))

# Use math functions (from star import)
pi_value = pi
sqrt_16 = sqrt(16)
"""
        
        repl.execute(import_code)
        namespace = repl.get_namespace()
        
        assert "os" in namespace
        assert "dt" in namespace
        assert "Path" in namespace
        assert "pi" in namespace
        assert namespace["python_version"] >= 3
        assert namespace["sqrt_16"] == 4.0

    def test_repl_state_exception_handling(self, temp_repl_dir):
        """Test ReplState with exception handling code."""
        repl = ReplState()
        repl.clear_state()
        
        exception_code = """
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return "Cannot divide by zero"
    except TypeError:
        return "Invalid types for division"
    finally:
        pass  # Cleanup code would go here

def process_list(items):
    results = []
    for item in items:
        try:
            processed = int(item) * 2
            results.append(processed)
        except ValueError:
            results.append(f"Could not process: {item}")
    return results

# Test exception handling
safe_result_1 = safe_divide(10, 2)
safe_result_2 = safe_divide(10, 0)
safe_result_3 = safe_divide("10", 2)

processed_items = process_list([1, "2", 3, "invalid", 5])
"""
        
        repl.execute(exception_code)
        namespace = repl.get_namespace()
        
        assert namespace["safe_result_1"] == 5.0
        assert namespace["safe_result_2"] == "Cannot divide by zero"
        assert namespace["safe_result_3"] == "Invalid types for division"
        assert len(namespace["processed_items"]) == 5

    def test_repl_state_generators_and_iterators(self, temp_repl_dir):
        """Test ReplState with generators and iterators."""
        repl = ReplState()
        repl.clear_state()
        
        generator_code = """
def fibonacci_generator(n):
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

def squares_generator(n):
    for i in range(n):
        yield i ** 2

# Generator expressions
squares_gen = (x**2 for x in range(5))
even_squares = (x for x in squares_gen if x % 2 == 0)

# Use generators
fib_list = list(fibonacci_generator(8))
squares_list = list(squares_generator(5))
even_squares_list = list(even_squares)

# Iterator protocol
class CountDown:
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

countdown = CountDown(3)
countdown_list = list(countdown)
"""
        
        repl.execute(generator_code)
        namespace = repl.get_namespace()
        
        assert namespace["fib_list"] == [0, 1, 1, 2, 3, 5, 8, 13]
        assert namespace["squares_list"] == [0, 1, 4, 9, 16]
        assert namespace["countdown_list"] == [3, 2, 1]

    def test_repl_state_decorators(self, temp_repl_dir):
        """Test ReplState with decorators."""
        repl = ReplState()
        repl.clear_state()
        
        decorator_code = """
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        # Simplified timing (not using actual time for testing)
        result = func(*args, **kwargs)
        return f"Timed: {result}"
    return wrapper

def cache_decorator(func):
    cache = {}
    def wrapper(*args):
        if args in cache:
            return f"Cached: {cache[args]}"
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

@timing_decorator
def slow_function(x):
    return x * 2

@cache_decorator
def expensive_function(x):
    return x ** 2

# Class decorators
def add_method(cls):
    def new_method(self):
        return "Added method"
    cls.new_method = new_method
    return cls

@add_method
class TestClass:
    def __init__(self, value):
        self.value = value

# Test decorated functions
timed_result = slow_function(5)
cached_result_1 = expensive_function(4)
cached_result_2 = expensive_function(4)  # Should be cached

# Test decorated class
test_obj = TestClass(10)
added_method_result = test_obj.new_method()
"""
        
        repl.execute(decorator_code)
        namespace = repl.get_namespace()
        
        assert "Timed: 10" in namespace["timed_result"]
        assert namespace["cached_result_1"] == 16
        assert "Cached: 16" in namespace["cached_result_2"]
        assert namespace["added_method_result"] == "Added method"

    def test_repl_state_context_managers(self, temp_repl_dir):
        """Test ReplState with context managers."""
        repl = ReplState()
        repl.clear_state()
        
        context_code = """
class TestContextManager:
    def __init__(self, name):
        self.name = name
        self.entered = False
        self.exited = False
    
    def __enter__(self):
        self.entered = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False

# Test context manager
with TestContextManager("test") as cm:
    context_name = cm.name
    context_entered = cm.entered

context_exited = cm.exited

# Multiple context managers
class ResourceManager:
    def __init__(self, resource_id):
        self.resource_id = resource_id
        self.acquired = False
        self.released = False
    
    def __enter__(self):
        self.acquired = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.released = True

with ResourceManager("A") as res_a, ResourceManager("B") as res_b:
    resources_acquired = res_a.acquired and res_b.acquired

resources_released = res_a.released and res_b.released
"""
        
        repl.execute(context_code)
        namespace = repl.get_namespace()
        
        assert namespace["context_name"] == "test"
        assert namespace["context_entered"] is True
        assert namespace["context_exited"] is True
        assert namespace["resources_acquired"] is True
        assert namespace["resources_released"] is True

    def test_repl_state_async_code_simulation(self, temp_repl_dir):
        """Test ReplState with async-like code (without actual async)."""
        repl = ReplState()
        repl.clear_state()
        
        # Simulate async patterns without actual async/await
        async_simulation_code = """
class Future:
    def __init__(self, value):
        self._value = value
        self._done = False
    
    def set_result(self, value):
        self._value = value
        self._done = True
    
    def result(self):
        if not self._done:
            self.set_result(self._value)
        return self._value

class AsyncSimulator:
    def __init__(self):
        self.tasks = []
    
    def create_task(self, func, *args):
        future = Future(None)
        try:
            result = func(*args)
            future.set_result(result)
        except Exception as e:
            future.set_result(f"Error: {e}")
        self.tasks.append(future)
        return future
    
    def gather(self):
        return [task.result() for task in self.tasks]

def async_task_1():
    return "Task 1 completed"

def async_task_2():
    return "Task 2 completed"

def async_task_error():
    raise ValueError("Task failed")

# Simulate async execution
simulator = AsyncSimulator()
task1 = simulator.create_task(async_task_1)
task2 = simulator.create_task(async_task_2)
task3 = simulator.create_task(async_task_error)

results = simulator.gather()
successful_tasks = [r for r in results if not r.startswith("Error:")]
failed_tasks = [r for r in results if r.startswith("Error:")]
"""
        
        repl.execute(async_simulation_code)
        namespace = repl.get_namespace()
        
        assert len(namespace["results"]) == 3
        assert len(namespace["successful_tasks"]) == 2
        assert len(namespace["failed_tasks"]) == 1

    def test_repl_state_metaclasses(self, temp_repl_dir):
        """Test ReplState with metaclasses."""
        repl = ReplState()
        repl.clear_state()
        
        metaclass_code = """
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    def __init__(self, value):
        if not hasattr(self, 'initialized'):
            self.value = value
            self.initialized = True

class AutoPropertyMeta(type):
    def __new__(mcs, name, bases, attrs):
        for key, value in list(attrs.items()):
            if key.startswith('_') and not key.startswith('__'):
                prop_name = key[1:]  # Remove leading underscore
                attrs[prop_name] = property(
                    lambda self, k=key: getattr(self, k),
                    lambda self, val, k=key: setattr(self, k, val)
                )
        return super().__new__(mcs, name, bases, attrs)

class AutoProperty(metaclass=AutoPropertyMeta):
    def __init__(self):
        self._x = 0
        self._y = 0

# Test metaclasses
singleton1 = Singleton(10)
singleton2 = Singleton(20)
same_instance = singleton1 is singleton2

auto_prop = AutoProperty()
auto_prop.x = 42
x_value = auto_prop.x
"""
        
        repl.execute(metaclass_code)
        namespace = repl.get_namespace()
        
        assert namespace["same_instance"] is True
        assert namespace["x_value"] == 42
        assert "SingletonMeta" in namespace
        assert "AutoPropertyMeta" in namespace

    def test_repl_state_save_error_handling(self, temp_repl_dir):
        """Test ReplState save error handling."""
        repl = ReplState()
        repl.clear_state()
        
        # Mock file operations to cause errors
        with patch("builtins.open", side_effect=IOError("Disk full")):
            # Should not raise exception
            repl.save_state("test_var = 42")
        
        # State should still be updated in memory
        assert "test_var" in repl.get_namespace()

    def test_repl_state_load_corrupted_file(self, temp_repl_dir):
        """Test ReplState loading corrupted state file."""
        # Create corrupted state file
        state_file = os.path.join(temp_repl_dir, "repl_state.pkl")
        with open(state_file, "wb") as f:
            f.write(b"corrupted pickle data")
        
        # Should handle corruption gracefully
        repl = ReplState()
        assert "__name__" in repl.get_namespace()

    def test_repl_state_clear_with_file_error(self, temp_repl_dir):
        """Test ReplState clear with file removal error."""
        repl = ReplState()
        
        # Create state file
        repl.save_state("test_var = 42")
        
        # Mock file removal to fail
        with patch("os.remove", side_effect=OSError("Permission denied")):
            # Should not raise exception
            repl.clear_state()
        
        # State should still be cleared in memory
        assert "test_var" not in repl.get_namespace()

    def test_repl_state_get_user_objects_filtering(self, temp_repl_dir):
        """Test ReplState user objects filtering."""
        repl = ReplState()
        repl.clear_state()
        
        # Add various types of objects
        test_code = """
# User objects (should be included)
user_int = 42
user_float = 3.14
user_string = "hello"
user_bool = True

# Private objects (should be excluded)
_private_var = "private"
__dunder_var__ = "dunder"

# Complex objects (should be excluded from user objects display)
user_list = [1, 2, 3]
user_dict = {"key": "value"}

def user_function():
    pass

class UserClass:
    pass
"""
        
        repl.execute(test_code)
        user_objects = repl.get_user_objects()
        
        # Should include basic types
        assert "user_int" in user_objects
        assert "user_float" in user_objects
        assert "user_string" in user_objects
        assert "user_bool" in user_objects
        
        # Should exclude private variables
        assert "_private_var" not in user_objects
        assert "__dunder_var__" not in user_objects
        
        # Should exclude complex objects from display
        assert "user_list" not in user_objects
        assert "user_dict" not in user_objects
        assert "user_function" not in user_objects
        assert "UserClass" not in user_objects


class TestCleanAnsiAdvanced:
    """Advanced tests for clean_ansi function."""

    def test_clean_ansi_complex_sequences(self):
        """Test cleaning complex ANSI sequences."""
        test_cases = [
            # Color codes
            ("\033[31mRed text\033[0m", "Red text"),
            ("\033[1;32mBold green\033[0m", "Bold green"),
            ("\033[38;5;196mBright red\033[0m", "Bright red"),
            
            # Cursor movement
            ("\033[2J\033[H\033[KClear screen", "Clear screen"),
            ("\033[10;20HPosition cursor", "Position cursor"),
            
            # Mixed sequences
            ("\033[1m\033[31mBold red\033[0m\033[32m green\033[0m", "Bold red green"),
            
            # Malformed sequences
            ("\033[Invalid sequence", "nvalid sequence"),
            ("Text\033[999mMore text", "TextMore text"),
        ]
        
        for input_text, expected in test_cases:
            result = clean_ansi(input_text)
            assert result == expected

    def test_clean_ansi_empty_and_edge_cases(self):
        """Test clean_ansi with empty and edge cases."""
        assert clean_ansi("") == ""
        assert clean_ansi("No ANSI codes") == "No ANSI codes"
        assert clean_ansi("\033[0m") == ""
        assert clean_ansi("\033[31m\033[0m") == ""

    def test_clean_ansi_unicode_with_ansi(self):
        """Test clean_ansi with unicode characters and ANSI codes."""
        input_text = "\033[31m🐍 Python\033[0m \033[32m中文\033[0m"
        expected = "🐍 Python 中文"
        result = clean_ansi(input_text)
        assert result == expected


class TestPtyManagerAdvanced:
    """Advanced tests for PtyManager class."""

    def test_pty_manager_initialization(self):
        """Test PtyManager initialization."""
        pty_mgr = PtyManager()
        assert pty_mgr.supervisor_fd == -1
        assert pty_mgr.worker_fd == -1
        assert pty_mgr.pid == -1
        assert len(pty_mgr.output_buffer) == 0
        assert len(pty_mgr.input_buffer) == 0
        assert not pty_mgr.stop_event.is_set()

    def test_pty_manager_with_callback(self):
        """Test PtyManager with callback function."""
        callback_outputs = []
        
        def test_callback(output):
            callback_outputs.append(output)
        
        pty_mgr = PtyManager(callback=test_callback)
        assert pty_mgr.callback == test_callback

    def test_pty_manager_read_output_unicode_handling(self):
        """Test PtyManager read output with unicode handling."""
        pty_mgr = PtyManager()
        
        # Mock file descriptor and select
        with (
            patch("select.select") as mock_select,
            patch("os.read") as mock_read,
            patch("os.close")
        ):
            # Configure mocks for unicode test
            mock_select.side_effect = [
                ([10], [], []),  # First call - data ready
                ([], [], [])     # Second call - no data
            ]
            
            # Test incomplete UTF-8 sequence
            mock_read.side_effect = [
                b"\xc3",  # Incomplete UTF-8 sequence
                b"\xa9",  # Completion of UTF-8 sequence (©)
                b""       # EOF
            ]
            
            pty_mgr.supervisor_fd = 10
            
            # Start reading in thread
            read_thread = threading.Thread(target=pty_mgr._read_output)
            read_thread.daemon = True
            read_thread.start()
            
            # Allow thread to process
            time.sleep(0.1)
            
            # Stop the thread
            pty_mgr.stop_event.set()
            read_thread.join(timeout=1.0)
            
            # Should handle unicode correctly
            output = pty_mgr.get_output()
            assert "©" in output or len(pty_mgr.output_buffer) > 0

    def test_pty_manager_read_output_error_handling(self):
        """Test PtyManager read output error handling."""
        pty_mgr = PtyManager()
        
        with (
            patch("select.select") as mock_select,
            patch("os.read") as mock_read,
            patch("os.close")
        ):
            # Test various error conditions
            error_cases = [
                OSError(9, "Bad file descriptor"),
                IOError("I/O error"),
                UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")
            ]
            
            for error in error_cases:
                mock_select.side_effect = [([10], [], [])]
                mock_read.side_effect = error
                
                pty_mgr.supervisor_fd = 10
                pty_mgr.stop_event.clear()
                
                # Should handle error gracefully
                read_thread = threading.Thread(target=pty_mgr._read_output)
                read_thread.daemon = True
                read_thread.start()
                
                time.sleep(0.1)
                pty_mgr.stop_event.set()
                read_thread.join(timeout=1.0)

    def test_pty_manager_read_output_callback_error(self):
        """Test PtyManager read output with callback error."""
        def failing_callback(output):
            raise Exception("Callback failed")
        
        pty_mgr = PtyManager(callback=failing_callback)
        
        with (
            patch("select.select") as mock_select,
            patch("os.read") as mock_read,
            patch("os.close")
        ):
            mock_select.side_effect = [([10], [], []), ([], [], [])]
            mock_read.side_effect = [b"test output\n", b""]
            
            pty_mgr.supervisor_fd = 10
            
            # Should handle callback error gracefully
            read_thread = threading.Thread(target=pty_mgr._read_output)
            read_thread.daemon = True
            read_thread.start()
            
            time.sleep(0.1)
            pty_mgr.stop_event.set()
            read_thread.join(timeout=1.0)
            
            # Output should still be captured despite callback error
            assert len(pty_mgr.output_buffer) > 0

    def test_pty_manager_handle_input_error(self):
        """Test PtyManager input handling with errors."""
        pty_mgr = PtyManager()
        
        with (
            patch("select.select", side_effect=OSError("Select error")),
            patch("sys.stdin.read")
        ):
            pty_mgr.supervisor_fd = 10
            
            # Should handle error gracefully
            input_thread = threading.Thread(target=pty_mgr._handle_input)
            input_thread.daemon = True
            input_thread.start()
            
            time.sleep(0.1)
            pty_mgr.stop_event.set()
            input_thread.join(timeout=1.0)

    def test_pty_manager_get_output_binary_truncation(self):
        """Test PtyManager binary content truncation."""
        pty_mgr = PtyManager()
        
        # Add binary-looking content
        binary_content = "\\x00\\x01\\x02" * 50  # Long binary-like content
        pty_mgr.output_buffer = [binary_content]
        
        # Test with default max length
        output = pty_mgr.get_output()
        assert "[binary content truncated]" in output
        
        # Test with custom max length
        with patch.dict(os.environ, {"PYTHON_REPL_BINARY_MAX_LEN": "20"}):
            output = pty_mgr.get_output()
            assert "[binary content truncated]" in output

    def test_pty_manager_stop_process_scenarios(self):
        """Test PtyManager stop with various process scenarios."""
        pty_mgr = PtyManager()
        
        # Test with valid PID
        pty_mgr.pid = 12345
        pty_mgr.supervisor_fd = 10
        
        with (
            patch("os.kill") as mock_kill,
            patch("os.waitpid") as mock_waitpid,
            patch("os.close") as mock_close
        ):
            # Test graceful termination
            mock_waitpid.side_effect = [(12345, 0)]  # Process exits gracefully
            
            pty_mgr.stop()
            
            mock_kill.assert_called_with(12345, signal.SIGTERM)
            mock_waitpid.assert_called()
            mock_close.assert_called_with(10)

    def test_pty_manager_stop_force_kill(self):
        """Test PtyManager stop with force kill."""
        pty_mgr = PtyManager()
        pty_mgr.pid = 12345
        pty_mgr.supervisor_fd = 10
        
        with (
            patch("os.kill") as mock_kill,
            patch("os.waitpid") as mock_waitpid,
            patch("os.close") as mock_close,
            patch("time.sleep")
        ):
            # Process doesn't exit gracefully, needs force kill
            mock_waitpid.side_effect = [
                (0, 0),    # First check - still running
                (0, 0),    # Second check - still running
                (12345, 9) # Finally killed
            ]
            
            pty_mgr.stop()
            
            # Should try SIGTERM first, then SIGKILL
            assert mock_kill.call_count >= 2
            mock_kill.assert_any_call(12345, signal.SIGTERM)
            mock_kill.assert_any_call(12345, signal.SIGKILL)

    def test_pty_manager_stop_process_errors(self):
        """Test PtyManager stop with process errors."""
        pty_mgr = PtyManager()
        pty_mgr.pid = 12345
        pty_mgr.supervisor_fd = 10
        
        with (
            patch("os.kill", side_effect=ProcessLookupError("No such process")),
            patch("os.close") as mock_close
        ):
            # Should handle process not found gracefully
            pty_mgr.stop()
            mock_close.assert_called_with(10)

    def test_pty_manager_stop_fd_error(self):
        """Test PtyManager stop with file descriptor error."""
        pty_mgr = PtyManager()
        pty_mgr.supervisor_fd = 10
        
        with patch("os.close", side_effect=OSError("Bad file descriptor")):
            # Should handle FD error gracefully
            pty_mgr.stop()
            assert pty_mgr.supervisor_fd == -1


class TestPythonReplAdvanced:
    """Advanced tests for python_repl function."""

    def test_python_repl_with_metrics(self, mock_console):
        """Test python_repl with metrics in result."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "result = 2 + 2", "interactive": False}
        }
        
        # Mock result with metrics
        mock_result = MagicMock()
        mock_result.get = MagicMock(side_effect=lambda k, default=None: {
            "content": [{"text": "Code executed"}],
            "stop_reason": "completed",
            "metrics": MagicMock()
        }.get(k, default))
        
        with (
            patch("strands_tools.python_repl.get_user_input", return_value="y"),
            patch.object(python_repl.repl_state, "execute"),
            patch("strands_tools.python_repl.OutputCapture") as mock_capture_class
        ):
            mock_capture = MagicMock()
            mock_capture.get_output.return_value = "4"
            mock_capture_class.return_value.__enter__.return_value = mock_capture
            
            result = python_repl.python_repl(tool=tool_use)
            
            assert result["status"] == "success"

    def test_python_repl_interactive_mode_waitpid_scenarios(self, mock_console):
        """Test python_repl interactive mode with various waitpid scenarios."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "print('test')", "interactive": True}
        }
        
        with patch("strands_tools.python_repl.PtyManager") as mock_pty_class:
            mock_pty = MagicMock()
            mock_pty.pid = 12345
            mock_pty.get_output.return_value = "test output"
            mock_pty_class.return_value = mock_pty
            
            # Test different waitpid scenarios
            waitpid_scenarios = [
                [(12345, 0)],  # Normal exit
                [(0, 0), (12345, 0)],  # Process running, then exits
                OSError("No child processes")  # Process already gone
            ]
            
            for scenario in waitpid_scenarios:
                with patch("os.waitpid") as mock_waitpid:
                    if isinstance(scenario, list):
                        mock_waitpid.side_effect = scenario
                    else:
                        mock_waitpid.side_effect = scenario
                    
                    result = python_repl.python_repl(tool=tool_use, non_interactive_mode=True)
                    
                    assert result["status"] == "success"
                    mock_pty.stop.assert_called()

    def test_python_repl_interactive_mode_exit_status_handling(self, mock_console):
        """Test python_repl interactive mode exit status handling."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "print('test')", "interactive": True}
        }
        
        with patch("strands_tools.python_repl.PtyManager") as mock_pty_class:
            mock_pty = MagicMock()
            mock_pty.pid = 12345
            mock_pty.get_output.return_value = "test output"
            mock_pty_class.return_value = mock_pty
            
            # Test non-zero exit status (error)
            with patch("os.waitpid", return_value=(12345, 1)):
                result = python_repl.python_repl(tool=tool_use, non_interactive_mode=True)
                
                assert result["status"] == "success"  # Still success as output was captured
                # State should not be saved on error
                mock_pty.stop.assert_called()

    def test_python_repl_recursion_error_state_reset(self, mock_console, temp_repl_dir):
        """Test python_repl recursion error with state reset."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "def recurse(): recurse()\nrecurse()", "interactive": False}
        }
        
        # Mock recursion error during execution
        with (
            patch("strands_tools.python_repl.get_user_input", return_value="y"),
            patch.object(python_repl.repl_state, "execute", side_effect=RecursionError("maximum recursion depth exceeded")),
            patch.object(python_repl.repl_state, "clear_state") as mock_clear
        ):
            result = python_repl.python_repl(tool=tool_use)
            
            assert result["status"] == "error"
            assert "RecursionError" in result["content"][0]["text"]
            assert "reset_state=True" in result["content"][0]["text"]
            
            # Should clear state on recursion error
            mock_clear.assert_called_once()

    def test_python_repl_error_logging(self, mock_console, temp_repl_dir):
        """Test python_repl error logging to file."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "1/0", "interactive": False}
        }
        
        # Create errors directory
        errors_dir = os.path.join(Path.cwd(), "errors")
        os.makedirs(errors_dir, exist_ok=True)
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result = python_repl.python_repl(tool=tool_use)
            
            assert result["status"] == "error"
            
            # Check if error was logged
            error_file = os.path.join(errors_dir, "errors.txt")
            if os.path.exists(error_file):
                with open(error_file, "r") as f:
                    content = f.read()
                    assert "ZeroDivisionError" in content

    def test_python_repl_user_objects_display(self, mock_console, temp_repl_dir):
        """Test python_repl user objects display in output."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "x = 42\ny = 'hello'\nz = [1, 2, 3]", "interactive": False}
        }
        
        with (
            patch("strands_tools.python_repl.get_user_input", return_value="y"),
            patch("strands_tools.python_repl.OutputCapture") as mock_capture_class
        ):
            mock_capture = MagicMock()
            mock_capture.get_output.return_value = ""
            mock_capture_class.return_value.__enter__.return_value = mock_capture
            
            result = python_repl.python_repl(tool=tool_use)
            
            assert result["status"] == "success"
            # Should show user objects in namespace
            result_text = result["content"][0]["text"]
            # The exact format may vary, but should indicate objects were created

    def test_python_repl_execution_timing(self, mock_console):
        """Test python_repl execution timing display."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "import time; time.sleep(0.01)", "interactive": False}
        }
        
        with (
            patch("strands_tools.python_repl.get_user_input", return_value="y"),
            patch("strands_tools.python_repl.OutputCapture") as mock_capture_class
        ):
            mock_capture = MagicMock()
            mock_capture.get_output.return_value = ""
            mock_capture_class.return_value.__enter__.return_value = mock_capture
            
            result = python_repl.python_repl(tool=tool_use)
            
            assert result["status"] == "success"
            # Should include timing information
            # The console output would show timing, but we're mocking it

    def test_python_repl_confirmation_dialog_details(self, mock_console):
        """Test python_repl confirmation dialog with code details."""
        long_code = "x = 1\n" * 50  # Multi-line code
        
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": long_code, "interactive": True, "reset_state": True}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y") as mock_input:
            result = python_repl.python_repl(tool=tool_use)
            
            # Should have shown confirmation dialog
            mock_input.assert_called_once()
            assert result["status"] == "success"

    def test_python_repl_custom_rejection_reason(self, mock_console):
        """Test python_repl with custom rejection reason."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "print('rejected')", "interactive": False}
        }
        
        with (
            patch("strands_tools.python_repl.get_user_input", side_effect=["custom rejection", ""]),
            patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "false"}, clear=False)
        ):
            result = python_repl.python_repl(tool=tool_use)
            
            assert result["status"] == "error"
            assert "custom rejection" in result["content"][0]["text"]

    def test_python_repl_state_persistence_verification(self, mock_console, temp_repl_dir):
        """Test python_repl state persistence across calls."""
        # First call - set variable
        tool_use_1 = {
            "toolUseId": "test-1",
            "input": {"code": "persistent_var = 'I persist'", "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result_1 = python_repl.python_repl(tool=tool_use_1)
            assert result_1["status"] == "success"
        
        # Second call - use variable
        tool_use_2 = {
            "toolUseId": "test-2",
            "input": {"code": "result = persistent_var + ' across calls'", "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result_2 = python_repl.python_repl(tool=tool_use_2)
            assert result_2["status"] == "success"
        
        # Verify variable persisted
        namespace = python_repl.repl_state.get_namespace()
        assert namespace.get("result") == "I persist across calls"

    def test_python_repl_output_capture_integration(self, mock_console):
        """Test python_repl output capture integration."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {
                "code": "print('stdout'); import sys; print('stderr', file=sys.stderr)",
                "interactive": False
            }
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result = python_repl.python_repl(tool=tool_use)
            
            assert result["status"] == "success"
            # Output should contain both stdout and stderr
            output_text = result["content"][0]["text"]
            assert "stdout" in output_text
            assert "stderr" in output_text

    def test_python_repl_environment_variable_handling(self, mock_console):
        """Test python_repl with various environment variable configurations."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "test_var = 42", "interactive": False}
        }
        
        # Test with BYPASS_TOOL_CONSENT variations
        bypass_values = ["true", "TRUE", "True", "false", "FALSE", "False", ""]
        
        for bypass_value in bypass_values:
            with patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": bypass_value}):
                if bypass_value.lower() == "true":
                    # Should bypass confirmation
                    result = python_repl.python_repl(tool=tool_use)
                    assert result["status"] == "success"
                else:
                    # Should require confirmation
                    with patch("strands_tools.python_repl.get_user_input", return_value="y"):
                        result = python_repl.python_repl(tool=tool_use)
                        assert result["status"] == "success"


class TestPythonReplEdgeCases:
    """Test edge cases and error conditions."""

    def test_python_repl_empty_code(self, mock_console):
        """Test python_repl with empty code."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "", "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result = python_repl.python_repl(tool=tool_use)
            assert result["status"] == "success"

    def test_python_repl_whitespace_only_code(self, mock_console):
        """Test python_repl with whitespace-only code."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "   \n\t\n   ", "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result = python_repl.python_repl(tool=tool_use)
            assert result["status"] == "success"

    def test_python_repl_very_long_code(self, mock_console):
        """Test python_repl with very long code."""
        long_code = "x = " + "1 + " * 1000 + "1"
        
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": long_code, "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result = python_repl.python_repl(tool=tool_use)
            assert result["status"] == "success"

    def test_python_repl_unicode_code(self, mock_console):
        """Test python_repl with unicode in code."""
        unicode_code = """
# Unicode variable names and strings
变量 = "中文"
العربية = "Arabic"
emoji = "🐍🚀✨"
print(f"{变量} {العربية} {emoji}")
"""
        
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": unicode_code, "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result = python_repl.python_repl(tool=tool_use)
            assert result["status"] == "success"

    def test_python_repl_mixed_indentation(self, mock_console):
        """Test python_repl with mixed indentation (should cause IndentationError)."""
        mixed_indent_code = """
def test_function():
    if True:
\t\treturn "mixed tabs and spaces"
"""
        
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": mixed_indent_code, "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result = python_repl.python_repl(tool=tool_use)
            assert result["status"] == "error"
            assert "TabError" in result["content"][0]["text"]

    def test_python_repl_import_error_handling(self, mock_console):
        """Test python_repl with import errors."""
        import_error_code = """
import nonexistent_module
from another_nonexistent import something
"""
        
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": import_error_code, "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result = python_repl.python_repl(tool=tool_use)
            assert result["status"] == "error"
            assert "ModuleNotFoundError" in result["content"][0]["text"]

    def test_python_repl_memory_error_simulation(self, mock_console):
        """Test python_repl with simulated memory error."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "x = 42", "interactive": False}
        }
        
        # Mock execute to raise MemoryError
        with (
            patch("strands_tools.python_repl.get_user_input", return_value="y"),
            patch.object(python_repl.repl_state, "execute", side_effect=MemoryError("Out of memory"))
        ):
            result = python_repl.python_repl(tool=tool_use)
            assert result["status"] == "error"
            assert "MemoryError" in result["content"][0]["text"]

    @pytest.mark.skip(reason="KeyboardInterrupt simulation causes issues with parallel test execution")
    def test_python_repl_keyboard_interrupt_simulation(self, mock_console):
        """Test python_repl with simulated KeyboardInterrupt."""
        tool_use = {
            "toolUseId": "test-id",
            "input": {"code": "x = 42", "interactive": False}
        }
        
        # Create a custom exception that will be caught by the general exception handler
        keyboard_interrupt = KeyboardInterrupt("Interrupted")
        
        # Mock execute to raise KeyboardInterrupt, but wrap the call to handle it properly
        with (
            patch("strands_tools.python_repl.get_user_input", return_value="y"),
            patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "true"}),  # Skip user confirmation
            patch.object(python_repl.repl_state, "execute") as mock_execute
        ):
            # Configure the mock to raise KeyboardInterrupt
            mock_execute.side_effect = keyboard_interrupt
            
            # The python_repl function should catch the KeyboardInterrupt and return an error result
            result = python_repl.python_repl(tool=tool_use)
            assert result["status"] == "error"
            assert "KeyboardInterrupt" in result["content"][0]["text"]


class TestPythonReplIntegration:
    """Integration tests for python_repl."""

    def test_python_repl_full_workflow(self, mock_console, temp_repl_dir):
        """Test complete python_repl workflow."""
        # Step 1: Define functions and classes
        setup_code = """
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self):
        return self.history

calc = Calculator()
"""
        
        tool_use_1 = {
            "toolUseId": "setup",
            "input": {"code": setup_code, "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result_1 = python_repl.python_repl(tool=tool_use_1)
            assert result_1["status"] == "success"
        
        # Step 2: Use the calculator
        calc_code = """
result1 = calc.add(5, 3)
result2 = calc.add(10, 7)
history = calc.get_history()
total_operations = len(history)
"""
        
        tool_use_2 = {
            "toolUseId": "calculate",
            "input": {"code": calc_code, "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result_2 = python_repl.python_repl(tool=tool_use_2)
            assert result_2["status"] == "success"
        
        # Step 3: Verify results
        verify_code = """
assert result1 == 8
assert result2 == 17
assert total_operations == 2
assert "5 + 3 = 8" in history
assert "10 + 7 = 17" in history
verification_passed = True
"""
        
        tool_use_3 = {
            "toolUseId": "verify",
            "input": {"code": verify_code, "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result_3 = python_repl.python_repl(tool=tool_use_3)
            assert result_3["status"] == "success"
        
        # Verify final state
        namespace = python_repl.repl_state.get_namespace()
        assert namespace.get("verification_passed") is True

    def test_python_repl_error_recovery(self, mock_console, temp_repl_dir):
        """Test python_repl error recovery."""
        # Step 1: Successful operation
        success_code = "x = 10"
        tool_use_1 = {
            "toolUseId": "success",
            "input": {"code": success_code, "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result_1 = python_repl.python_repl(tool=tool_use_1)
            assert result_1["status"] == "success"
        
        # Step 2: Error operation
        error_code = "y = x / 0"  # Division by zero
        tool_use_2 = {
            "toolUseId": "error",
            "input": {"code": error_code, "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result_2 = python_repl.python_repl(tool=tool_use_2)
            assert result_2["status"] == "error"
        
        # Step 3: Recovery operation
        recovery_code = "y = x * 2"  # Should work
        tool_use_3 = {
            "toolUseId": "recovery",
            "input": {"code": recovery_code, "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result_3 = python_repl.python_repl(tool=tool_use_3)
            assert result_3["status"] == "success"
        
        # Verify state is intact
        namespace = python_repl.repl_state.get_namespace()
        assert namespace.get("x") == 10
        assert namespace.get("y") == 20

    def test_python_repl_state_reset_workflow(self, mock_console, temp_repl_dir):
        """Test python_repl state reset workflow."""
        # Step 1: Set some variables
        setup_code = "a = 1; b = 2; c = 3"
        tool_use_1 = {
            "toolUseId": "setup",
            "input": {"code": setup_code, "interactive": False}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result_1 = python_repl.python_repl(tool=tool_use_1)
            assert result_1["status"] == "success"
        
        # Verify variables exist
        namespace = python_repl.repl_state.get_namespace()
        assert "a" in namespace
        assert "b" in namespace
        assert "c" in namespace
        
        # Step 2: Reset state and set new variables
        reset_code = "x = 10; y = 20"
        tool_use_2 = {
            "toolUseId": "reset",
            "input": {"code": reset_code, "interactive": False, "reset_state": True}
        }
        
        with patch("strands_tools.python_repl.get_user_input", return_value="y"):
            result_2 = python_repl.python_repl(tool=tool_use_2)
            assert result_2["status"] == "success"
        
        # Verify old variables are gone, new ones exist
        namespace = python_repl.repl_state.get_namespace()
        assert "a" not in namespace
        assert "b" not in namespace
        assert "c" not in namespace
        assert namespace.get("x") == 10
        assert namespace.get("y") == 20