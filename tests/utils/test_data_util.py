"""
Tests for data utility functions.
"""

from datetime import datetime, timezone

import pytest

from strands_tools.utils.data_util import convert_datetime_to_str, to_snake_case


class TestConvertDatetimeToStr:
    """Test convert_datetime_to_str function."""

    def test_convert_single_datetime(self):
        """Test converting a single datetime object."""
        dt = datetime(2025, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        result = convert_datetime_to_str(dt)
        assert result == "2025-01-15 14:30:45+0000"

    def test_convert_datetime_without_timezone(self):
        """Test converting datetime without timezone info."""
        dt = datetime(2025, 1, 15, 14, 30, 45)
        result = convert_datetime_to_str(dt)
        assert result == "2025-01-15 14:30:45"

    def test_convert_dict_with_datetime(self):
        """Test converting dictionary containing datetime objects."""
        dt = datetime(2025, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        data = {
            "timestamp": dt,
            "name": "test",
            "count": 42
        }
        
        result = convert_datetime_to_str(data)
        
        assert result["timestamp"] == "2025-01-15 14:30:45+0000"
        assert result["name"] == "test"
        assert result["count"] == 42

    def test_convert_nested_dict_with_datetime(self):
        """Test converting nested dictionary with datetime objects."""
        dt1 = datetime(2025, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        dt2 = datetime(2025, 1, 16, 10, 15, 30)
        
        data = {
            "event": {
                "start_time": dt1,
                "metadata": {
                    "created_at": dt2,
                    "status": "active"
                }
            },
            "id": 123
        }
        
        result = convert_datetime_to_str(data)
        
        assert result["event"]["start_time"] == "2025-01-15 14:30:45+0000"
        assert result["event"]["metadata"]["created_at"] == "2025-01-16 10:15:30"
        assert result["event"]["metadata"]["status"] == "active"
        assert result["id"] == 123

    def test_convert_list_with_datetime(self):
        """Test converting list containing datetime objects."""
        dt1 = datetime(2025, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        dt2 = datetime(2025, 1, 16, 10, 15, 30)
        
        data = [dt1, "string", 42, dt2]
        
        result = convert_datetime_to_str(data)
        
        assert result[0] == "2025-01-15 14:30:45+0000"
        assert result[1] == "string"
        assert result[2] == 42
        assert result[3] == "2025-01-16 10:15:30"

    def test_convert_list_of_dicts_with_datetime(self):
        """Test converting list of dictionaries with datetime objects."""
        dt1 = datetime(2025, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        dt2 = datetime(2025, 1, 16, 10, 15, 30)
        
        data = [
            {"timestamp": dt1, "value": 100},
            {"timestamp": dt2, "value": 200}
        ]
        
        result = convert_datetime_to_str(data)
        
        assert result[0]["timestamp"] == "2025-01-15 14:30:45+0000"
        assert result[0]["value"] == 100
        assert result[1]["timestamp"] == "2025-01-16 10:15:30"
        assert result[1]["value"] == 200

    def test_convert_complex_nested_structure(self):
        """Test converting complex nested structure with datetime objects."""
        dt = datetime(2025, 1, 15, 14, 30, 45, tzinfo=timezone.utc)
        
        data = {
            "events": [
                {
                    "timestamp": dt,
                    "details": {
                        "logs": [
                            {"time": dt, "message": "Started"},
                            {"time": dt, "message": "Completed"}
                        ]
                    }
                }
            ],
            "metadata": {
                "created": dt
            }
        }
        
        result = convert_datetime_to_str(data)
        
        expected_time_str = "2025-01-15 14:30:45+0000"
        assert result["events"][0]["timestamp"] == expected_time_str
        assert result["events"][0]["details"]["logs"][0]["time"] == expected_time_str
        assert result["events"][0]["details"]["logs"][1]["time"] == expected_time_str
        assert result["metadata"]["created"] == expected_time_str

    def test_convert_non_datetime_objects(self):
        """Test that non-datetime objects are returned unchanged."""
        data = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        
        result = convert_datetime_to_str(data)
        
        # Should be identical since no datetime objects
        assert result == data

    def test_convert_empty_structures(self):
        """Test converting empty structures."""
        assert convert_datetime_to_str({}) == {}
        assert convert_datetime_to_str([]) == []
        assert convert_datetime_to_str(None) is None
        assert convert_datetime_to_str("") == ""

    def test_convert_datetime_with_microseconds(self):
        """Test converting datetime with microseconds."""
        dt = datetime(2025, 1, 15, 14, 30, 45, 123456, tzinfo=timezone.utc)
        result = convert_datetime_to_str(dt)
        assert result == "2025-01-15 14:30:45+0000"  # Microseconds are not included in format


class TestToSnakeCase:
    """Test to_snake_case function."""

    def test_camel_case_conversion(self):
        """Test converting camelCase to snake_case."""
        assert to_snake_case("camelCase") == "camel_case"
        assert to_snake_case("myVariableName") == "my_variable_name"
        assert to_snake_case("getUserData") == "get_user_data"

    def test_pascal_case_conversion(self):
        """Test converting PascalCase to snake_case."""
        assert to_snake_case("PascalCase") == "pascal_case"
        assert to_snake_case("MyClassName") == "my_class_name"
        assert to_snake_case("HTTPResponseCode") == "h_t_t_p_response_code"

    def test_already_snake_case(self):
        """Test that snake_case strings remain unchanged."""
        assert to_snake_case("snake_case") == "snake_case"
        assert to_snake_case("already_snake_case") == "already_snake_case"
        assert to_snake_case("my_variable") == "my_variable"

    def test_single_word(self):
        """Test single word conversions."""
        assert to_snake_case("word") == "word"
        assert to_snake_case("Word") == "word"
        assert to_snake_case("WORD") == "w_o_r_d"

    def test_empty_string(self):
        """Test empty string conversion."""
        assert to_snake_case("") == ""

    def test_numbers_in_string(self):
        """Test strings with numbers."""
        assert to_snake_case("version2API") == "version2_a_p_i"
        assert to_snake_case("myVar123") == "my_var123"
        assert to_snake_case("API2Version") == "a_p_i2_version"

    def test_consecutive_capitals(self):
        """Test strings with consecutive capital letters."""
        assert to_snake_case("XMLHttpRequest") == "x_m_l_http_request"
        assert to_snake_case("JSONData") == "j_s_o_n_data"
        assert to_snake_case("HTTPSConnection") == "h_t_t_p_s_connection"

    def test_special_cases(self):
        """Test special edge cases."""
        assert to_snake_case("A") == "a"
        assert to_snake_case("AB") == "a_b"
        assert to_snake_case("ABC") == "a_b_c"
        assert to_snake_case("aB") == "a_b"
        assert to_snake_case("aBC") == "a_b_c"

    def test_mixed_patterns(self):
        """Test mixed patterns with underscores and capitals."""
        assert to_snake_case("my_CamelCase") == "my__camel_case"
        assert to_snake_case("snake_CaseExample") == "snake__case_example"
        assert to_snake_case("API_Version2") == "a_p_i__version2"

    def test_leading_capital(self):
        """Test strings starting with capital letters."""
        assert to_snake_case("ClassName") == "class_name"
        assert to_snake_case("MyFunction") == "my_function"
        assert to_snake_case("APIEndpoint") == "a_p_i_endpoint"

    def test_all_caps(self):
        """Test all caps strings."""
        assert to_snake_case("CONSTANT") == "c_o_n_s_t_a_n_t"
        assert to_snake_case("API") == "a_p_i"
        assert to_snake_case("HTTP") == "h_t_t_p"