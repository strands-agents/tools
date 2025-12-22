"""Tests for the json_processor tool."""

import json

import pytest

from strands_tools.json_processor import json_processor


class TestJsonProcessor:
    """Test cases for json_processor tool."""

    def test_validate_valid_json_string(self):
        """Test validation of valid JSON string."""
        result = json_processor(action="validate", data='{"name": "John", "age": 30}')

        assert result["valid"] is True
        assert result["message"] == "Valid JSON"
        assert result["type"] == "dict"
        assert result["size"] == 2

    def test_validate_valid_json_object(self):
        """Test validation of valid Python dict."""
        result = json_processor(action="validate", data={"name": "John", "age": 30})

        assert result["valid"] is True
        assert result["message"] == "Valid JSON"
        assert result["type"] == "dict"
        assert result["size"] == 2

    def test_validate_invalid_json_string(self):
        """Test validation of invalid JSON string."""
        result = json_processor(action="validate", data='{"name": "John", "age":}')

        assert result["valid"] is False
        assert "Invalid JSON" in result["error"]
        assert "line" in result or "column" in result

    def test_format_json_object(self):
        """Test formatting of JSON object."""
        data = {"name": "John", "age": 30}
        result = json_processor(action="format", data=data, indent=2)

        assert "formatted" in result
        assert "size" in result
        assert "lines" in result
        assert result["lines"] > 1  # Should be multi-line

        # Verify it's valid JSON
        parsed = json.loads(result["formatted"])
        assert parsed == data

    def test_format_with_custom_indent(self):
        """Test formatting with custom indentation."""
        data = {"name": "John", "age": 30}
        result = json_processor(action="format", data=data, indent=4)

        formatted = result["formatted"]
        lines = formatted.split("\n")
        # Check that indentation is 4 spaces
        assert any("    " in line for line in lines if line.strip())

    def test_minify_json(self):
        """Test JSON minification."""
        data = {"name": "John", "age": 30}
        result = json_processor(action="minify", data=data)

        assert "minified" in result
        assert "size" in result
        assert "compression_ratio" in result

        # Verify no unnecessary whitespace
        minified = result["minified"]
        assert "\n" not in minified
        assert "  " not in minified  # No double spaces

        # Verify it's still valid JSON
        parsed = json.loads(minified)
        assert parsed == data

    def test_query_simple_path(self):
        """Test querying with simple dot notation."""
        data = {"user": {"name": "John", "age": 30}}
        result = json_processor(action="query", data=data, query_path="user.name")

        assert result["found"] is True
        assert result["result"] == "John"
        assert result["path"] == "user.name"
        assert result["type"] == "str"

    def test_query_nested_path(self):
        """Test querying with deeply nested path."""
        data = {"user": {"profile": {"personal": {"name": "John"}}}}
        result = json_processor(action="query", data=data, query_path="user.profile.personal.name")

        assert result["found"] is True
        assert result["result"] == "John"
        assert result["path"] == "user.profile.personal.name"

    def test_query_array_index(self):
        """Test querying array elements by index."""
        data = {"users": [{"name": "John"}, {"name": "Jane"}]}
        result = json_processor(action="query", data=data, query_path="users.1.name")

        assert result["found"] is True
        assert result["result"] == "Jane"

    def test_query_nonexistent_key(self):
        """Test querying non-existent key."""
        data = {"user": {"name": "John"}}
        result = json_processor(action="query", data=data, query_path="user.email")

        assert result["found"] is False
        assert result["result"] is None
        assert "not found" in result["error"].lower()

    def test_query_out_of_bounds_array(self):
        """Test querying array with out-of-bounds index."""
        data = {"users": [{"name": "John"}]}
        result = json_processor(action="query", data=data, query_path="users.5.name")

        assert result["found"] is False
        assert "out of range" in result["error"].lower()

    def test_keys_object(self):
        """Test getting keys from JSON object."""
        data = {"name": "John", "age": 30, "city": "NYC"}
        result = json_processor(action="keys", data=data)

        assert result["type"] == "object"
        assert set(result["keys"]) == {"name", "age", "city"}
        assert result["count"] == 3
        assert "nested_objects" in result
        assert "nested_arrays" in result

    def test_keys_array(self):
        """Test getting keys (indices) from JSON array."""
        data = ["a", "b", "c"]
        result = json_processor(action="keys", data=data)

        assert result["type"] == "array"
        assert result["keys"] == [0, 1, 2]
        assert result["count"] == 3

    def test_keys_primitive(self):
        """Test getting keys from primitive value."""
        result = json_processor(action="keys", data="hello")

        assert result["type"] == "str"
        assert result["keys"] == []
        assert result["count"] == 0
        assert "not an object or array" in result["message"]

    def test_size_object(self):
        """Test size calculation for JSON object."""
        data = {"user": {"name": "John", "details": {"age": 30}}}
        result = json_processor(action="size", data=data)

        assert result["type"] == "object"
        assert result["keys"] == 1  # Top-level keys
        assert result["deep_size"] > result["keys"]  # Should count nested elements
        assert result["max_depth"] > 0

    def test_size_array(self):
        """Test size calculation for JSON array."""
        data = [1, 2, [3, 4]]
        result = json_processor(action="size", data=data)

        assert result["type"] == "array"
        assert result["items"] == 3
        assert result["deep_size"] > result["items"]
        assert result["max_depth"] > 0

    def test_type_information(self):
        """Test type information extraction."""
        data = {"name": "John"}
        result = json_processor(action="type", data=data)

        assert result["type"] == "dict"
        assert "python_type" in result
        assert result["is_container"] is True
        assert result["is_empty"] is False
        assert result["json_serializable"] is True

    def test_type_empty_container(self):
        """Test type information for empty container."""
        result = json_processor(action="type", data=[])

        assert result["type"] == "list"
        assert result["is_container"] is True
        assert result["is_empty"] is True

    def test_missing_action_parameter(self):
        """Test error handling for missing action parameter."""
        with pytest.raises(ValueError, match="Action parameter is required"):
            json_processor(action="", data="{}")

    def test_missing_data_parameter(self):
        """Test error handling for missing data parameter."""
        with pytest.raises(ValueError, match="Data parameter is required"):
            json_processor(action="format", data=None)

    def test_missing_query_path(self):
        """Test error handling for missing query_path in query action."""
        with pytest.raises(ValueError, match="query_path is required"):
            json_processor(action="query", data={"name": "John"}, query_path=None)

    def test_invalid_action(self):
        """Test error handling for invalid action."""
        with pytest.raises(ValueError, match="Unknown action"):
            json_processor(action="invalid_action", data={"name": "John"})

    def test_complex_nested_structure(self):
        """Test with complex nested JSON structure."""
        data = {
            "users": [
                {
                    "id": 1,
                    "profile": {
                        "name": "John",
                        "contacts": {"email": "john@example.com", "phones": ["123-456-7890", "098-765-4321"]},
                    },
                }
            ],
            "metadata": {"total": 1, "page": 1},
        }

        # Test validation
        validate_result = json_processor(action="validate", data=data)
        assert validate_result["valid"] is True

        # Test complex query
        query_result = json_processor(action="query", data=data, query_path="users.0.profile.contacts.phones.1")
        assert query_result["found"] is True
        assert query_result["result"] == "098-765-4321"

        # Test size calculation
        size_result = json_processor(action="size", data=data)
        assert size_result["deep_size"] > 10  # Should count all nested elements
        assert size_result["max_depth"] >= 4  # Deep nesting

    def test_json_string_input(self):
        """Test processing JSON string input."""
        json_string = '{"name": "John", "age": 30}'
        result = json_processor(action="keys", data=json_string)

        assert result["type"] == "object"
        assert set(result["keys"]) == {"name", "age"}

    def test_malformed_json_string_for_non_validate_action(self):
        """Test handling of malformed JSON strings in non-validate actions."""
        # Now malformed JSON strings are treated as primitive strings
        result = json_processor(action="type", data='{"name": "John", "age":}')

        # Should treat it as a string, not fail
        assert result["type"] == "str"
        assert result["json_serializable"] is True  # String itself is JSON serializable

    def test_unicode_handling(self):
        """Test proper Unicode handling in JSON processing."""
        data = {"name": "José", "city": "São Paulo", "emoji": "🚀"}

        # Test formatting preserves Unicode
        format_result = json_processor(action="format", data=data)
        formatted = format_result["formatted"]
        assert "José" in formatted
        assert "São Paulo" in formatted
        assert "🚀" in formatted

        # Test minification preserves Unicode
        minify_result = json_processor(action="minify", data=data)
        minified = minify_result["minified"]
        parsed = json.loads(minified)
        assert parsed == data
