"""
JSON processing tool for comprehensive JSON data manipulation and analysis.

This module provides a powerful JSON processing capability with validation,
formatting, querying, and manipulation operations. It's designed to handle
various JSON processing scenarios from simple validation to complex data extraction.

Key Features:
1. JSON Validation:
   • Syntax validation with detailed error reporting
   • Type detection and structure analysis
   • Schema validation support

2. Formatting Operations:
   • Pretty printing with configurable indentation
   • Minification for compact representation
   • Custom formatting options

3. Data Querying:
   • Dot notation path querying (e.g., 'user.profile.name')
   • Key extraction and enumeration
   • Size and structure analysis

4. Data Manipulation:
   • Safe data extraction and transformation
   • Type-aware processing
   • Error handling with graceful fallbacks

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import json_processor

agent = Agent(tools=[json_processor])

# Validate JSON
agent.tool.json_processor(action="validate", data='{"name": "John"}')

# Format JSON with custom indentation
agent.tool.json_processor(action="format", data={"name": "John", "age": 30}, indent=4)

# Query nested data
agent.tool.json_processor(action="query", data={"user": {"name": "John"}}, query_path="user.name")
```
"""

import json
from typing import Any, Dict, List, Union

from strands import tool


@tool
def json_processor(
    action: str,
    data: Union[str, Dict[str, Any], List[Any]] = None,
    query_path: str = None,
    indent: int = 2,
) -> Dict[str, Any]:
    """
    Process JSON data with validation, formatting, and querying capabilities.

    This tool provides comprehensive JSON processing including validation,
    pretty formatting, querying, and basic manipulation operations.

    Args:
        action (str): The operation to perform:
            - 'validate': Check if JSON is valid and return type information
            - 'format': Pretty print JSON with specified indentation
            - 'minify': Remove whitespace from JSON for compact representation
            - 'query': Extract value using dot notation path (e.g., 'user.name')
            - 'keys': Get all keys from JSON object at root or specified path
            - 'size': Get size/length information about the JSON structure
            - 'type': Get detailed type information about JSON data
        data (Union[str, Dict, List]): JSON string or Python object to process
        query_path (str, optional): Dot notation path for querying (e.g., 'user.profile.name')
        indent (int, optional): Indentation spaces for formatting (default: 2)

    Returns:
        Dict[str, Any]: Result containing the processed data and metadata

    Raises:
        ValueError: If invalid action or required parameters are missing

    Examples:
        >>> json_processor(action="validate", data='{"name": "John", "age": 30}')
        {'valid': True, 'message': 'Valid JSON', 'type': 'dict', 'size': 2}

        >>> json_processor(action="format", data='{"name":"John","age":30}', indent=4)
        {'formatted': '{\n    "name": "John",\n    "age": 30\n}', 'size': 32}

        >>> json_processor(action="query", data={"user": {"name": "John"}}, query_path="user.name")
        {'result': 'John', 'path': 'user.name', 'found': True}

        >>> json_processor(action="keys", data={"name": "John", "age": 30})
        {'keys': ['name', 'age'], 'count': 2, 'type': 'object'}
    """

    if not action:
        raise ValueError("Action parameter is required")

    if data is None and action != "validate":
        raise ValueError(f"Data parameter is required for action '{action}'")

    try:
        # Parse JSON string to Python object if needed
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError as e:
                if action == "validate":
                    return {
                        "valid": False,
                        "error": f"Invalid JSON: {str(e)}",
                        "line": getattr(e, "lineno", None),
                        "column": getattr(e, "colno", None),
                    }
                else:
                    # For non-validate actions, treat as a primitive string value
                    parsed_data = data
        else:
            parsed_data = data

        if action == "validate":
            return {
                "valid": True,
                "message": "Valid JSON",
                "type": type(parsed_data).__name__,
                "size": len(parsed_data) if hasattr(parsed_data, "__len__") else 1,
            }

        elif action == "format":
            try:
                formatted = json.dumps(parsed_data, indent=indent, ensure_ascii=False, sort_keys=True)
                return {"formatted": formatted, "size": len(formatted), "lines": formatted.count("\n") + 1}
            except TypeError as e:
                raise ValueError(f"Cannot format data as JSON: {str(e)}") from e

        elif action == "minify":
            try:
                minified = json.dumps(parsed_data, separators=(",", ":"), ensure_ascii=False)
                return {
                    "minified": minified,
                    "size": len(minified),
                    "compression_ratio": round(len(minified) / len(json.dumps(parsed_data, indent=2)), 2),
                }
            except TypeError as e:
                raise ValueError(f"Cannot minify data as JSON: {str(e)}") from e

        elif action == "query":
            if not query_path:
                raise ValueError("query_path is required for query action")

            # Simple dot notation querying
            result = parsed_data
            path_parts = query_path.split(".")

            try:
                for i, key in enumerate(path_parts):
                    if isinstance(result, dict):
                        if key in result:
                            result = result[key]
                        else:
                            return {
                                "result": None,
                                "path": query_path,
                                "found": False,
                                "error": f"Key '{key}' not found at path '{'.'.join(path_parts[: i + 1])}'",
                            }
                    elif isinstance(result, list):
                        try:
                            index = int(key)
                            if 0 <= index < len(result):
                                result = result[index]
                            else:
                                return {
                                    "result": None,
                                    "path": query_path,
                                    "found": False,
                                    "error": f"Index {index} out of range for array of length {len(result)}",
                                }
                        except ValueError:
                            return {
                                "result": None,
                                "path": query_path,
                                "found": False,
                                "error": f"Cannot use key '{key}' on array, expected numeric index",
                            }
                    else:
                        return {
                            "result": None,
                            "path": query_path,
                            "found": False,
                            "error": (
                                f"Cannot query further into {type(result).__name__} "
                                f"at '{'.'.join(path_parts[:i])}'"
                            ),
                        }

                return {
                    "result": result,
                    "path": query_path,
                    "found": True,
                    "type": type(result).__name__,
                }
            except Exception as e:
                return {
                    "result": None,
                    "path": query_path,
                    "found": False,
                    "error": f"Query error: {str(e)}",
                }

        elif action == "keys":
            if isinstance(parsed_data, dict):
                keys = list(parsed_data.keys())
                return {
                    "keys": keys,
                    "count": len(keys),
                    "type": "object",
                    "nested_objects": sum(1 for v in parsed_data.values() if isinstance(v, dict)),
                    "nested_arrays": sum(1 for v in parsed_data.values() if isinstance(v, list)),
                }
            elif isinstance(parsed_data, list):
                return {
                    "keys": list(range(len(parsed_data))),
                    "count": len(parsed_data),
                    "type": "array",
                    "nested_objects": sum(1 for item in parsed_data if isinstance(item, dict)),
                    "nested_arrays": sum(1 for item in parsed_data if isinstance(item, list)),
                }
            else:
                return {
                    "keys": [],
                    "count": 0,
                    "type": type(parsed_data).__name__,
                    "message": f"Data is a {type(parsed_data).__name__}, not an object or array",
                }

        elif action == "size":

            def calculate_deep_size(obj):
                """Calculate the deep size of nested JSON structures."""
                if isinstance(obj, dict):
                    return sum(calculate_deep_size(v) for v in obj.values()) + len(obj)
                elif isinstance(obj, list):
                    return sum(calculate_deep_size(item) for item in obj) + len(obj)
                else:
                    return 1

            if isinstance(parsed_data, dict):
                return {
                    "keys": len(parsed_data),
                    "type": "object",
                    "deep_size": calculate_deep_size(parsed_data),
                    "max_depth": _calculate_depth(parsed_data),
                }
            elif isinstance(parsed_data, list):
                return {
                    "items": len(parsed_data),
                    "type": "array",
                    "deep_size": calculate_deep_size(parsed_data),
                    "max_depth": _calculate_depth(parsed_data),
                }
            else:
                return {
                    "size": 1,
                    "type": type(parsed_data).__name__,
                    "deep_size": 1,
                    "max_depth": 0,
                }

        elif action == "type":
            return {
                "type": type(parsed_data).__name__,
                "python_type": str(type(parsed_data)),
                "is_container": isinstance(parsed_data, (dict, list)),
                "is_empty": len(parsed_data) == 0 if hasattr(parsed_data, "__len__") else False,
                "json_serializable": _is_json_serializable(parsed_data),
            }

        else:
            raise ValueError(
                f"Unknown action: {action}. Supported actions: validate, format, minify, query, keys, size, type"
            )

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Error processing JSON: {str(e)}") from e


def _calculate_depth(obj, current_depth=0):
    """Calculate the maximum depth of nested JSON structures."""
    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(_calculate_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(_calculate_depth(item, current_depth + 1) for item in obj)
    else:
        return current_depth


def _is_json_serializable(obj):
    """Check if an object is JSON serializable."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False
