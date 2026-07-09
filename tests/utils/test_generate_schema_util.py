"""
Tests for the generate_schema_util.py module.

This test suite covers the utility functions that generate JSON schemas
for AWS service operations.
"""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import UnknownServiceError
from botocore.model import Shape

from strands_tools.utils.generate_schema_util import (
    check_boto3_validity,
    clean_and_trim_description,
    generate_input_schema,
    generate_schema,
    to_pascal_case,
    to_snake_case,
)


class TestCleanAndTrimDescription:
    """Tests for the clean_and_trim_description function."""

    def test_html_removal(self):
        """Test that HTML tags are removed."""
        description = "<p>This is a <b>test</b> description.</p>"
        result = clean_and_trim_description(description)
        assert result == "This is a test description."
        assert "<" not in result
        assert ">" not in result

    def test_whitespace_normalization(self):
        """Test that whitespace is normalized."""
        description = "This   has    multiple    spaces."
        result = clean_and_trim_description(description)
        assert result == "This has multiple spaces."

    def test_max_length_respected(self):
        """Test that the max_length parameter is respected."""
        description = "This is a very long description that should be trimmed."
        max_length = 20
        result = clean_and_trim_description(description, max_length)
        assert len(result) <= max_length
        # Only check that it starts with the expected content, as implementation
        # might handle trimming slightly differently
        assert result.startswith("This is a very long")


class TestToSnakeCase:
    """Tests for the to_snake_case function."""

    def test_pascal_case_conversion(self):
        """Test conversion from PascalCase to snake_case."""
        assert to_snake_case("DescribeInstances") == "describe_instances"
        assert to_snake_case("CreateUser") == "create_user"

    def test_camel_case_conversion(self):
        """Test conversion from camelCase to snake_case."""
        assert to_snake_case("describeInstances") == "describe_instances"
        assert to_snake_case("createUser") == "create_user"

    def test_kebab_case_conversion(self):
        """Test conversion from kebab-case to snake_case."""
        assert to_snake_case("describe-instances") == "describe_instances"
        assert to_snake_case("create-user") == "create_user"

    def test_already_snake_case(self):
        """Test that snake_case remains unchanged."""
        assert to_snake_case("describe_instances") == "describe_instances"
        assert to_snake_case("create_user") == "create_user"

    def test_acronym_handling(self):
        """Test that acronyms are properly handled."""
        assert to_snake_case("DescribeDBInstances") == "describe_db_instances"
        assert to_snake_case("GetURLPath") == "get_url_path"


class TestToPascalCase:
    """Tests for the to_pascal_case function."""

    @patch("boto3.client")
    def test_snake_case_conversion(self, mock_boto_client):
        """Test conversion from snake_case to PascalCase."""
        # Setup mock boto3 client
        mock_service_model = MagicMock()
        mock_boto_client.return_value.meta.service_model = mock_service_model

        # Test cases
        with patch(
            "strands_tools.utils.generate_schema_util.boto3.client",
            return_value=mock_boto_client.return_value,
        ):
            assert to_pascal_case("s3", "list_buckets") == "ListBuckets"
            assert to_pascal_case("dynamodb", "create_table") == "CreateTable"

    @patch("boto3.client")
    def test_kebab_case_conversion(self, mock_boto_client):
        """Test conversion from kebab-case to PascalCase."""
        # Setup mock boto3 client
        mock_service_model = MagicMock()
        mock_boto_client.return_value.meta.service_model = mock_service_model

        with patch(
            "strands_tools.utils.generate_schema_util.boto3.client",
            return_value=mock_boto_client.return_value,
        ):
            assert to_pascal_case("s3", "list-buckets") == "ListBuckets"
            assert to_pascal_case("dynamodb", "create-table") == "CreateTable"

    @patch("boto3.client")
    def test_already_pascal_case(self, mock_boto_client):
        """Test that PascalCase remains unchanged."""
        # Setup mock boto3 client
        mock_service_model = MagicMock()
        mock_boto_client.return_value.meta.service_model = mock_service_model

        with patch(
            "strands_tools.utils.generate_schema_util.boto3.client",
            return_value=mock_boto_client.return_value,
        ):
            assert to_pascal_case("s3", "ListBuckets") == "ListBuckets"
            assert to_pascal_case("dynamodb", "CreateTable") == "CreateTable"

    @patch("boto3.client")
    def test_validation_exception_fallback(self, mock_boto_client):
        """Test fallback when operation validation fails."""
        # Setup mock to raise exception on first validation but succeed on fallback
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        # First validation throws exception
        mock_client.meta.service_model.operation_model.side_effect = Exception("Not found")

        # But operations list contains the correct operation
        mock_client.meta.service_model.operation_names = ["ListBuckets", "CreateTable"]

        with patch(
            "strands_tools.utils.generate_schema_util.boto3.client",
            return_value=mock_client,
        ):
            # Should find match in operation_names
            assert to_pascal_case("s3", "list_buckets") == "ListBuckets"

    @patch("boto3.client")
    def test_fallback_with_complete_exception(self, mock_boto_client):
        """Test the fallback mechanism when both validations fail."""
        # Setup the first validation to fail
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.meta.service_model.operation_model.side_effect = Exception("Not found")

        # And the second validation to also fail by providing an empty list
        mock_client.meta.service_model.operation_names = []

        # Mock the logger directly
        with patch("strands_tools.utils.generate_schema_util.logger.debug"):
            result = to_pascal_case("dynamodb", "list_tables")
            assert result == "ListTables"  # Should convert despite validation failures


class TestCheckBoto3Validity:
    """Tests for the check_boto3_validity function."""

    @patch("boto3.client")
    def test_valid_service_and_operation(self, mock_boto_client):
        """Test checking a valid service and operation."""
        # Setup mock client
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        # Make sure hasattr returns True for the operation
        mock_client.__class__ = type("MockClient", (), {"list_buckets": lambda: None})

        with patch(
            "strands_tools.utils.generate_schema_util.to_pascal_case",
            return_value="ListBuckets",
        ):
            with patch(
                "strands_tools.utils.generate_schema_util.to_snake_case",
                return_value="list_buckets",
            ):
                is_valid, error_message = check_boto3_validity("s3", "list_buckets")
                assert is_valid is True
                assert error_message == ""

    @patch("boto3.client")
    def test_invalid_service(self, mock_boto_client):
        """Test checking an invalid service."""
        mock_boto_client.side_effect = UnknownServiceError(
            service_name="invalid_service", known_service_names=["s3", "ec2"]
        )

        is_valid, error_message = check_boto3_validity("invalid_service", "some_operation")
        assert is_valid is False
        assert "Unknown service: 'invalid_service'" == error_message

    @patch("boto3.client")
    def test_invalid_operation(self, mock_boto_client):
        """Test checking an invalid operation."""
        # Create a mock client with no operations
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        # Patch hasattr to return False for the operation
        with patch(
            "strands_tools.utils.generate_schema_util.hasattr",
            side_effect=lambda obj, name: (False if name in ["invalid_operation", "InvalidOperation"] else True),
        ):
            # Mock the conversion functions
            with patch(
                "strands_tools.utils.generate_schema_util.to_pascal_case",
                return_value="InvalidOperation",
            ):
                with patch(
                    "strands_tools.utils.generate_schema_util.to_snake_case",
                    return_value="invalid_operation",
                ):
                    is_valid, error_message = check_boto3_validity("s3", "invalid_operation")
                    assert is_valid is False
                    assert "not found in service" in error_message


class TestGenerateSchema:
    """Tests for the generate_schema function."""

    def test_structure_shape(self):
        """Test generating a schema from a structure shape."""
        # Create a mock Shape object representing a structure
        mock_shape = MagicMock(spec=Shape)
        mock_shape.type_name = "structure"
        mock_shape.required_members = ["required_field"]

        # Create mock member shapes
        string_shape = MagicMock(spec=Shape)
        string_shape.type_name = "string"

        integer_shape = MagicMock(spec=Shape)
        integer_shape.type_name = "integer"

        # Set up the members dictionary
        mock_shape.members = {
            "required_field": string_shape,
            "optional_field": integer_shape,
        }

        # Generate the schema
        schema = generate_schema(mock_shape)

        # Check schema structure
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "required_field" in schema["required"]
        assert len(schema["required"]) == 1

    def test_list_shape(self):
        """Test generating a schema from a list shape."""
        # Create mock shapes
        list_shape = MagicMock(spec=Shape)
        list_shape.type_name = "list"

        member_shape = MagicMock(spec=Shape)
        member_shape.type_name = "string"
        list_shape.member = member_shape

        # Generate schema
        schema = generate_schema(list_shape)

        # Check schema structure
        assert schema["type"] == "array"
        assert "items" in schema
        assert schema["items"]["type"] == "string"

    def test_map_shape(self):
        """Test generating a schema from a map shape."""
        # Create mock shapes
        map_shape = MagicMock(spec=Shape)
        map_shape.type_name = "map"

        value_shape = MagicMock(spec=Shape)
        value_shape.type_name = "string"
        map_shape.value = value_shape

        # Generate schema
        schema = generate_schema(map_shape)

        # Check schema structure
        assert schema["type"] == "object"
        assert "additionalProperties" in schema
        assert schema["additionalProperties"]["type"] == "string"

    def test_primitive_shapes(self):
        """Test generating schemas from primitive shapes."""
        for shape_type, expected_json_type in [
            ("string", "string"),
            ("integer", "integer"),
            ("boolean", "boolean"),
            ("float", "number"),
            ("double", "number"),
            ("long", "integer"),
        ]:
            # Create mock shape
            mock_shape = MagicMock(spec=Shape)
            mock_shape.type_name = shape_type

            # Generate schema
            schema = generate_schema(mock_shape)

            # Check schema type
            assert schema["type"] == expected_json_type

    def test_max_depth_respected(self):
        """Test that the max_depth parameter is respected."""
        # Create a recursive structure
        root_shape = MagicMock(spec=Shape)
        root_shape.type_name = "structure"
        root_shape.required_members = []

        # Create a child shape that references the root shape
        child_shape = MagicMock(spec=Shape)
        child_shape.type_name = "structure"
        child_shape.required_members = []
        child_shape.members = {"parent": root_shape}

        # Set up the recursive relationship
        root_shape.members = {"child": child_shape}

        # Generate schema with max_depth=1
        schema = generate_schema(root_shape, max_depth=1)

        # Check that recursion was limited
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "child" in schema["properties"]

        # The child's properties should exist but be limited due to max_depth
        child_schema = schema["properties"]["child"]
        assert "type" in child_schema  # Should at least have a type

        # Generate at higher depth to compare
        full_schema = generate_schema(root_shape, max_depth=3)
        # The nested structure should be deeper in the full schema
        assert "properties" in full_schema["properties"]["child"]


class TestGenerateInputSchema:
    """Tests for the generate_input_schema function."""

    @patch("strands_tools.utils.generate_schema_util.check_boto3_validity")
    @patch("strands_tools.utils.generate_schema_util.boto3.client")
    def test_valid_service_and_operation(self, mock_boto_client, mock_check_validity):
        """Test generating an input schema for a valid service and operation."""
        # Setup mocks
        mock_check_validity.return_value = (True, "")

        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        mock_service_model = MagicMock()
        mock_client.meta.service_model = mock_service_model

        mock_operation_model = MagicMock()
        mock_service_model.operation_model.return_value = mock_operation_model

        # Mock the documentation
        mock_operation_model.documentation = "This is a test operation."

        # Mock the input shape
        mock_input_shape = MagicMock(spec=Shape)
        mock_input_shape.type_name = "structure"
        mock_input_shape.required_members = ["RequiredParam"]

        mock_param_shape = MagicMock(spec=Shape)
        mock_param_shape.type_name = "string"
        mock_input_shape.members = {"RequiredParam": mock_param_shape}

        mock_operation_model.input_shape = mock_input_shape

        # Test function
        with patch(
            "strands_tools.utils.generate_schema_util.to_pascal_case",
            return_value="TestOperation",
        ):
            with patch("strands_tools.utils.generate_schema_util.generate_schema") as mock_generate_schema:
                # Mock the schema generation result
                mock_generate_schema.return_value = {
                    "type": "object",
                    "properties": {"RequiredParam": {"type": "string"}},
                    "required": ["RequiredParam"],
                }

                result = generate_input_schema("test-service", "test_operation")

                # Verify the result
                assert result["result"] == "success"
                assert result["name"] == "test_operation"
                assert result["description"] == "This is a test operation."
                assert "inputSchema" in result
                assert "json" in result["inputSchema"]

                # Verify mocks were called correctly
                mock_check_validity.assert_called_once_with("test-service", "test_operation")
                mock_service_model.operation_model.assert_called_once_with("TestOperation")

    @patch("strands_tools.utils.generate_schema_util.check_boto3_validity")
    def test_invalid_service_or_operation(self, mock_check_validity):
        """Test handling invalid service or operation."""
        # Mock validity check to fail
        mock_check_validity.return_value = (False, "Unknown service: 'invalid-service'")

        # Test function
        result = generate_input_schema("invalid-service", "invalid_operation")

        # Verify the result indicates an error
        assert result["result"] == "error"
        assert result["name"] == "invalid_operation"
        assert "Error: Unknown service" in result["description"]
        # Should still have a schema, but it's empty
        assert result["inputSchema"]["json"]["type"] == "object"
        assert result["inputSchema"]["json"]["properties"] == {}

    @patch("strands_tools.utils.generate_schema_util.check_boto3_validity")
    @patch("strands_tools.utils.generate_schema_util.boto3.client")
    def test_exception_handling(self, mock_boto_client, mock_check_validity):
        """Test exception handling in generate_input_schema."""
        # Setup mocks for a valid service/operation but with an error during schema generation
        mock_check_validity.return_value = (True, "")
        mock_boto_client.side_effect = Exception("Unexpected error")

        # Test function - should raise RuntimeError
        with pytest.raises(RuntimeError) as excinfo:
            generate_input_schema("test-service", "test_operation")

        # Verify the error message
        assert "Error generating input schema" in str(excinfo.value)
