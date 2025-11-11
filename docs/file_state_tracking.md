# File State Tracking

This document describes the file state tracking feature in the `file_read`, `file_write`, and `editor` tools, inspired by the [LangGraph deepagents filesystem middleware](https://github.com/langchain-ai/deepagents/blob/master/libs/deepagents/middleware/filesystem.py) pattern.

## Overview

All file tools (`file_read`, `file_write`, `editor`) automatically track file state using Strands' built-in `agent.state` management via `ToolContext`. Each file operation stores comprehensive metadata for state tracking.

### State Structure

File information is stored in multiple formats for flexibility:

**Dictionary by Path (LangGraph-style):**
```python
files = agent.state.get("files") or {}
# Structure: {"/path/to/file.py": FileInfo, ...}
file_info = files["/path/to/file.py"]
```

**Sequential Lists (for history tracking):**
- `agent.state.get("files_written")` - List of all files written
- `agent.state.get("last_file_written")` - Last file written
- `agent.state.get("files_read")` - List of all files read
- `agent.state.get("last_files_read")` - Last batch of files read

### FileInfo Structure

Each `FileInfo` object contains:
```python
{
    "path": str,                    # Absolute file path
    "is_dir": bool,                 # Whether it's a directory
    "size": int,                    # File size in bytes
    "created_at": str,              # ISO 8601 timestamp
    "modified_at": str,             # ISO 8601 timestamp
    "mode": Optional[str],          # Operation mode (e.g., "view", "write", "search")
    "mode_info": Optional[dict]     # Mode-specific parameters (e.g., {"chunk_size": 1024})
}
```

This enables agents to:

- Maintain a session history of file operations
- Track file modifications for audit purposes
- Build stateful agents that are aware of their file interactions
- Monitor file changes across agent operations
- Implement file-based decision making
- Integrate with graph-based agent frameworks (LangGraph, etc.)

## File Info Structure

The file information follows a standardized structure:

```python
class FileInfo(TypedDict):
    path: str                    # Absolute path to the file
    is_dir: bool                 # Whether the path is a directory
    size: int                    # File size in bytes
    created_at: str              # ISO format timestamp of file creation
    modified_at: str             # ISO format timestamp of last modification
    mode: Optional[str]          # Operation mode (e.g., "view", "write", "search", "chunk")
    mode_info: Optional[dict]    # Mode-specific parameters (e.g., {"chunk_size": 1024, "chunk_offset": 0})
```

## Usage

### File Write Tool

When writing a file, the file information is automatically stored in `agent.state`:

```python
from strands import Agent
from strands_tools import file_write

agent = Agent(tools=[file_write])

response = agent.tool.file_write(
    path="/tmp/example.txt",
    content="Hello, World!\nThis is line 2."
)

# Access via files dict (LangGraph pattern)
files = agent.state.get("files") or {}
if "/tmp/example.txt" in files:
    file_info = files["/tmp/example.txt"]
    print(f"Path: {file_info['path']}")
    print(f"Size: {file_info['size']} bytes")
    print(f"Created: {file_info['created_at']}")
    print(f"Modified: {file_info['modified_at']}")
    print(f"Mode: {file_info.get('mode')}")  # "write"

# Or access via sequential list
files_written = agent.state.get("files_written") or []
print(f"Total files written: {len(files_written)}")

# Get the last file written
last_file = agent.state.get("last_file_written")
if last_file:
    print(f"Last file: {last_file['path']}")
    print(f"Size: {last_file['size']} bytes")
    print(f"Mode: {last_file.get('mode')}")  # "write"
```

### File Read Tool

When reading files, the file information is automatically stored in `agent.state`:

```python
from strands import Agent
from strands_tools import file_read

agent = Agent(tools=[file_read])

# Read a single file
response = agent.tool.file_read(
    path="/tmp/example.txt",
    mode="view"
)

# Access via files dict (LangGraph pattern)
files = agent.state.get("files") or {}
if "/tmp/example.txt" in files:
    file_info = files["/tmp/example.txt"]
    print(f"Path: {file_info['path']}")
    print(f"Size: {file_info['size']} bytes")
    print(f"Created: {file_info['created_at']}")
    print(f"Modified: {file_info['modified_at']}")
    print(f"Mode: {file_info.get('mode')}")  # "view"

# Or access via sequential list
files_read = agent.state.get("files_read") or []
print(f"Total files read: {len(files_read)}")

# Get the last files read
last_files = agent.state.get("last_files_read") or []
for file_info in last_files:
    print(f"Read from: {file_info['path']}")
    print(f"Size: {file_info['size']} bytes")
    print(f"Modified: {file_info['modified_at']}")

# Find multiple files
response = agent.tool.file_read(
    path="/tmp/*.txt",
    mode="find",
    recursive=True
)

# Check updated state - all found files are now in the dict
files = agent.state.get("files") or {}
print(f"Total files in state: {len(files)}")
for path, info in files.items():
    print(f"  {path}: {info['size']} bytes, modified: {info['modified_at']}")
```

### Reading with Mode Info

Different reading modes store specific parameters in `mode_info`:

```python
# Read a chunk of a file
response = agent.tool.file_read(
    path="/tmp/large_file.txt",
    mode="chunk",
    chunk_size=2048,
    chunk_offset=1024
)

# Access the file info with mode details
files = agent.state.get("files") or {}
file_info = files["/tmp/large_file.txt"]
print(f"Mode: {file_info.get('mode')}")  # "chunk"
print(f"Chunk Size: {file_info.get('mode_info', {}).get('chunk_size')}")  # 2048
print(f"Chunk Offset: {file_info.get('mode_info', {}).get('chunk_offset')}")  # 1024

# Read specific lines
response = agent.tool.file_read(
    path="/tmp/code.py",
    mode="lines",
    start_line=10,
    end_line=20
)

file_info = files["/tmp/code.py"]
print(f"Mode: {file_info.get('mode')}")  # "lines"
print(f"Start Line: {file_info.get('mode_info', {}).get('start_line')}")  # 10
print(f"End Line: {file_info.get('mode_info', {}).get('end_line')}")  # 20

# Search in a file
response = agent.tool.file_read(
    path="/tmp/log.txt",
    mode="search",
    search_pattern="ERROR",
    context_lines=3
)

file_info = files["/tmp/log.txt"]
print(f"Mode: {file_info.get('mode')}")  # "search"
print(f"Pattern: {file_info.get('mode_info', {}).get('search_pattern')}")  # "ERROR"
print(f"Context Lines: {file_info.get('mode_info', {}).get('context_lines')}")  # 3
```

## State Tracking Pattern

Here's a complete example of tracking file operations in an agent session:

```python
from strands import Agent
from strands_tools import file_read, file_write

class FileStateTracker:
    """Track file operations during an agent session."""
    
    def __init__(self):
        self.files_written = []
        self.files_read = []
        self.file_metadata = {}
    
    def track_write(self, response):
        """Track a file write operation."""
        if response["status"] == "success" and "metadata" in response:
            file_info = response["metadata"]["file_info"]
            path = file_info["path"]
            
            self.files_written.append(path)
            self.file_metadata[path] = file_info
    
    def track_read(self, response):
        """Track file read operations."""
        if response["status"] == "success" and "metadata" in response:
            for file_info in response["metadata"]["files_info"]:
                path = file_info["path"]
                
                if path not in self.files_read:
                    self.files_read.append(path)
                self.file_metadata[path] = file_info
    
    def get_summary(self):
        """Get a summary of file operations."""
        return {
            "total_files_written": len(self.files_written),
            "total_files_read": len(self.files_read),
            "total_unique_files": len(self.file_metadata),
            "total_size_bytes": sum(
                info["size"] for info in self.file_metadata.values()
            )
        }

# Usage
agent = Agent(tools=[file_read, file_write])
tracker = FileStateTracker()

# Write a file
write_response = agent.tool.file_write(
    path="/tmp/data.txt",
    content="Important data"
)
tracker.track_write(write_response)

# Read files
read_response = agent.tool.file_read(
    path="/tmp/*.txt",
    mode="find"
)
tracker.track_read(read_response)

# Get summary
summary = tracker.get_summary()
print(f"Session summary: {summary}")
```

## Integration with LangGraph

For agents using LangGraph, file information can be stored in the graph state. The tools accept an optional `state` parameter for seamless integration:

### Method 1: Using State Parameter (Recommended)

```python
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    files_accessed: List[dict]
    last_file_operation: dict

def file_operation_node(state: AgentState):
    """Node that performs file operations with state tracking."""
    from strands_tools import file_write
    
    # Call tool with state parameter
    result = file_write(
        tool={
            "toolUseId": "write-1",
            "input": {
                "path": "/tmp/data.txt",
                "content": "Important data"
            }
        },
        state=state  # Pass state for integration
    )
    
    # Extract file info from metadata and update state
    if result.get("metadata") and "file_info" in result["metadata"]:
        file_info = result["metadata"]["file_info"]
        state["files_accessed"].append(file_info)
        state["last_file_operation"] = {
            "type": "write",
            "file": file_info["path"],
            "timestamp": file_info["modified_at"]
        }
    
    return state

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("file_ops", file_operation_node)
# ... add more nodes and edges
app = workflow.compile()
```

### Method 2: Using Metadata Return Pattern

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    messages: List[dict]
    files_accessed: List[dict]
    last_file_operation: dict

def process_file_tool_result(state: AgentState, tool_result: dict) -> AgentState:
    """Update state with file information from tool results."""
    
    # Track files from write operations
    if "metadata" in tool_result and "file_info" in tool_result["metadata"]:
        file_info = tool_result["metadata"]["file_info"]
        state["files_accessed"].append(file_info)
        state["last_file_operation"] = {
            "type": "write",
            "file": file_info["path"],
            "timestamp": file_info["modified_at"]
        }
    
    # Track files from read operations
    if "metadata" in tool_result and "files_info" in tool_result["metadata"]:
        for file_info in tool_result["metadata"]["files_info"]:
            state["files_accessed"].append(file_info)
        
        if tool_result["metadata"]["files_info"]:
            last_file = tool_result["metadata"]["files_info"][-1]
            state["last_file_operation"] = {
                "type": "read",
                "file": last_file["path"],
                "timestamp": last_file["modified_at"]
            }
    
    return state
```

## Best Practices

1. **Always Check Metadata Presence**: Not all tool results will have metadata (e.g., on errors)
   ```python
   if response.get("metadata") and "file_info" in response["metadata"]:
       # Safe to access file_info
   ```

2. **Handle Multiple Files**: The `file_read` tool can process multiple files, always iterate through `files_info`
   ```python
   for file_info in response["metadata"]["files_info"]:
       # Process each file
   ```

3. **Track Timestamps**: Use the `modified_at` field to detect changes
   ```python
   if current_modified != previous_modified:
       print("File has been modified since last access")
   ```

4. **Size Monitoring**: Track cumulative file sizes to monitor resource usage
   ```python
   total_size = sum(f["size"] for f in session_files)
   if total_size > MAX_SIZE:
       print("Warning: Session file size limit exceeded")
   ```

## API Reference

### File Write Response

```python
{
    "toolUseId": str,
    "status": "success" | "error",
    "content": [{"text": str}],
    "metadata": {
        "file_info": {
            "path": str,
            "is_dir": bool,
            "size": int,
            "modified_at": str  # ISO format
        }
    }
}
```

### File Read Response

```python
{
    "toolUseId": str,
    "status": "success" | "error",
    "content": List[dict],
    "metadata": {
        "files_info": [
            {
                "path": str,
                "is_dir": bool,
                "size": int,
                "modified_at": str  # ISO format
            }
        ]
    }
}
```

## Utility Functions

The `strands_tools.utils.file_info` module provides utilities for working with file information:

```python
from strands_tools.utils.file_info import get_file_info, format_file_info

# Get file info manually
file_info = get_file_info("/path/to/file.txt")

# Format for display
formatted = format_file_info(file_info)
print(formatted)
```

## See Also

- [deepagents filesystem backend](https://github.com/langchain-ai/deepagents/blob/master/libs/deepagents/backends/filesystem.py)
- [File State Tracking Example](../examples/file_state_tracking.py)
- [File Read Tool Documentation](../src/strands_tools/file_read.py)
- [File Write Tool Documentation](../src/strands_tools/file_write.py)

