"""
Tests for the use_aws tool using the Agent interface.
"""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from botocore.config import Config as BotocoreConfig
from strands import Agent

from strands_tools import use_aws
from strands_tools.utils import data_util, user_input


@pytest.fixture
def agent():
    """Create an agent with the use_aws tool loaded."""
    return Agent(tools=[use_aws])


def extract_result_text(result):
    """Extract the result text from the agent response."""
    if isinstance(result, dict) and "content" in result and isinstance(result["content"], list):
        return result["content"][0]["text"]
    return str(result)


@pytest.fixture
def mock_boto3_client():
    """Create a mock boto3 client for testing."""
    with patch("strands_tools.use_aws.get_boto3_client") as mock_get_client:
        # Create a mock client with a mock operation
        mock_client = MagicMock()
        mock_operation = MagicMock()
        mock_operation.return_value = {
            "ResponseMetadata": {"RequestId": "test-request-id"},
            "Items": [{"id": "item1"}],
        }
        mock_client.describe_instances = mock_operation
        mock_client.list_buckets = MagicMock(return_value={"Buckets": [{"Name": "test-bucket"}]})

        # Configure the mock to return our mock client
        mock_get_client.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_available_services():
    """Create a mock for available AWS services."""
    with patch("strands_tools.use_aws.get_available_services") as mock_services:
        mock_services.return_value = ["ec2", "s3", "lambda", "dynamodb"]
        yield mock_services


@pytest.fixture
def mock_available_operations():
    """Create a mock for available operations."""
    with patch("strands_tools.use_aws.get_available_operations") as mock_operations:
        mock_operations.return_value = [
            "describe_instances",
            "list_buckets",
            "create_bucket",
        ]
        yield mock_operations


@pytest.fixture
def mock_boto3_session():
    """Create a mock boto3 session."""
    with patch("boto3.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session
        yield mock_session


def test_use_aws_direct_success(mock_boto3_client, mock_available_services, mock_available_operations):
    """Test direct invocation of the use_aws tool with a successful operation."""
    # Create a tool use dictionary similar to how the agent would call it
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "service_name": "ec2",
            "operation_name": "describe_instances",
            "parameters": {"InstanceIds": ["i-123456789"]},
            "region": "us-west-2",
            "label": "Test EC2 Instance Description",
        },
    }

    # Call the use_aws function directly
    result = use_aws.use_aws(tool=tool_use)

    # Verify the result has the expected structure
    assert result["toolUseId"] == "test-tool-use-id"
    assert result["status"] == "success"
    assert "Success:" in result["content"][0]["text"]
    assert "test-request-id" in result["content"][0]["text"]


def test_use_aws_invalid_service(mock_available_services, mock_available_operations):
    """Test use_aws with an invalid service name."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "service_name": "invalid_service",
            "operation_name": "describe_instances",
            "parameters": {},
            "region": "us-west-2",
            "label": "Invalid Service Test",
        },
    }

    result = use_aws.use_aws(tool=tool_use)

    assert result["status"] == "error"
    assert "Invalid AWS service: invalid_service" in result["content"][0]["text"]


def test_use_aws_invalid_operation(mock_available_services, mock_available_operations):
    """Test use_aws with an invalid operation name."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "service_name": "ec2",
            "operation_name": "invalid_operation",
            "parameters": {},
            "region": "us-west-2",
            "label": "Invalid Operation Test",
        },
    }

    result = use_aws.use_aws(tool=tool_use)

    assert result["status"] == "error"
    assert "Invalid AWS operation: invalid_operation" in result["content"][0]["text"]


def test_use_aws_validation_error(mock_boto3_client, mock_available_services, mock_available_operations):
    """Test use_aws with a parameter validation error."""
    # Configure the mock to raise a validation error
    from botocore.exceptions import ParamValidationError

    mock_boto3_client.describe_instances.side_effect = ParamValidationError(report="Invalid parameter format")

    # Mock the schema generation function
    with patch("strands_tools.use_aws.generate_input_schema") as mock_schema:
        mock_schema.return_value = {
            "type": "object",
            "properties": {"InstanceIds": {"type": "array"}},
        }

        tool_use = {
            "toolUseId": "test-tool-use-id",
            "input": {
                "service_name": "ec2",
                "operation_name": "describe_instances",
                "parameters": {"InstanceIds": "not-an-array"},
                "region": "us-west-2",
                "label": "Validation Error Test",
            },
        }

        result = use_aws.use_aws(tool=tool_use)

        assert result["status"] == "error"
        assert "Validation error:" in result["content"][0]["text"]
        assert "Expected input schema" in result["content"][1]["text"]


def test_use_aws_exception_handling(mock_boto3_client, mock_available_services, mock_available_operations):
    """Test use_aws with a generic exception."""
    # Configure the mock to raise an exception
    mock_boto3_client.describe_instances.side_effect = Exception("Test exception")

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "service_name": "ec2",
            "operation_name": "describe_instances",
            "parameters": {},
            "region": "us-west-2",
            "label": "Exception Test",
        },
    }

    result = use_aws.use_aws(tool=tool_use)

    assert result["status"] == "error"
    assert "AWS call threw exception: Test exception" in result["content"][0]["text"]


def test_use_aws_streaming_body_handling(mock_boto3_client, mock_available_services, mock_available_operations):
    """Test use_aws with a streaming body response."""
    from io import BytesIO

    from botocore.response import StreamingBody

    # Create a mock streaming body
    mock_stream = BytesIO(b'{"Result": "streaming data"}')
    mock_streaming_body = StreamingBody(mock_stream, len(mock_stream.getvalue()))

    # Configure the mock to return a response with a streaming body
    mock_boto3_client.describe_instances.return_value = {
        "ResponseMetadata": {"RequestId": "test-request-id"},
        "StreamData": mock_streaming_body,
    }

    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "service_name": "ec2",
            "operation_name": "describe_instances",
            "parameters": {},
            "region": "us-west-2",
            "label": "Streaming Body Test",
        },
    }

    result = use_aws.use_aws(tool=tool_use)

    assert result["status"] == "success"
    assert "Success:" in result["content"][0]["text"]
    assert "streaming data" in result["content"][0]["text"]


@patch("strands_tools.use_aws.get_user_input")
def test_use_aws_mutative_operation_confirm(
    mock_user_input,
    mock_boto3_client,
    mock_available_services,
    mock_available_operations,
):
    """Test use_aws with a mutative operation that requires confirmation."""
    # Mock the user input to confirm the operation
    mock_user_input.return_value = "y"

    # Configure environment variable
    with patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "false"}):
        tool_use = {
            "toolUseId": "test-tool-use-id",
            "input": {
                "service_name": "s3",
                "operation_name": "create_bucket",
                "parameters": {"Bucket": "test-bucket"},
                "region": "us-west-2",
                "label": "Mutative Operation Test",
            },
        }

        result = use_aws.use_aws(tool=tool_use)

        # Verify user was prompted for confirmation
        mock_user_input.assert_called_once()
        assert result["status"] == "success"


@patch("strands_tools.use_aws.get_user_input")
def test_use_aws_mutative_operation_cancel(
    mock_user_input,
    mock_boto3_client,
    mock_available_services,
    mock_available_operations,
):
    """Test use_aws with a mutative operation that's canceled by the user."""
    # Mock the user input to cancel the operation
    mock_user_input.return_value = "n"

    # Configure environment variable
    with patch.dict("os.environ", {"BYPASS_TOOL_CONSENT": "false"}):
        tool_use = {
            "toolUseId": "test-tool-use-id",
            "input": {
                "service_name": "s3",
                "operation_name": "create_bucket",
                "parameters": {"Bucket": "test-bucket"},
                "region": "us-west-2",
                "label": "Mutative Operation Cancel Test",
            },
        }

        result = use_aws.use_aws(tool=tool_use)

        # Verify user was prompted for confirmation
        mock_user_input.assert_called_once()
        assert result["status"] == "error"
        assert "Operation canceled by user" in result["content"][0]["text"]


def test_use_aws_with_profile(mock_boto3_client, mock_available_services, mock_available_operations):
    """Test use_aws with a specified AWS profile."""
    tool_use = {
        "toolUseId": "test-tool-use-id",
        "input": {
            "service_name": "ec2",
            "operation_name": "describe_instances",
            "parameters": {},
            "region": "us-west-2",
            "label": "Profile Test",
            "profile_name": "test-profile",
        },
    }

    with patch("strands_tools.use_aws.get_boto3_client") as mock_get_client:
        mock_get_client.return_value = mock_boto3_client
        result = use_aws.use_aws(tool=tool_use)

        # Verify profile was passed to the client
        mock_get_client.assert_called_once_with("ec2", "us-west-2", "test-profile")
        assert result["status"] == "success"


def test_get_available_operations():
    """Test get_available_operations with a valid service."""
    with patch("boto3.client") as mock_client_func:
        # Create a mock client
        mock_client = MagicMock()

        # Add methods directly to the mock
        mock_client.describe_instances = MagicMock()
        mock_client.list_buckets = MagicMock()
        mock_client._private_method = MagicMock()

        mock_client_func.return_value = mock_client

        # When dir() is called, only the non-private methods should be included
        result = use_aws.get_available_operations("ec2")

        # Check results
        assert "_private_method" not in result
        assert "describe_instances" in result
        assert "list_buckets" in result


def test_get_boto3_client():
    """Test that get_boto3_client calls boto3.Session and session.client correctly."""
    with patch("boto3.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        use_aws.get_boto3_client("s3", "us-east-1", "test-profile")

        mock_session_class.assert_called_once_with(profile_name="test-profile")

        # Verify the client was called with the correct arguments
        mock_session.client.assert_called_once()
        args, kwargs = mock_session.client.call_args

        assert kwargs["service_name"] == "s3"
        assert kwargs["region_name"] == "us-east-1"
        assert "config" in kwargs
        config = kwargs["config"]
        assert isinstance(config, BotocoreConfig)
        assert config.user_agent_extra == "strands-agents-use-aws"


def test_handle_streaming_body_non_json():
    """Test handle_streaming_body with non-JSON content."""
    from botocore.response import StreamingBody

    # Create a mock streaming body with non-JSON content
    mock_stream = BytesIO(b"This is not JSON content")
    mock_streaming_body = StreamingBody(mock_stream, len(mock_stream.getvalue()))

    response = {"StreamData": mock_streaming_body}
    result = use_aws.handle_streaming_body(response)

    assert result["StreamData"] == "This is not JSON content"


def test_get_available_services():
    """Test get_available_services calls boto3.Session().get_available_services()."""
    with patch("boto3.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session.get_available_services.return_value = ["s3", "ec2", "lambda"]
        mock_session_class.return_value = mock_session

        result = use_aws.get_available_services()

        mock_session.get_available_services.assert_called_once()
        assert result == ["s3", "ec2", "lambda"]


def test_get_available_operations_exception():
    """Test get_available_operations when boto3.client raises an exception."""
    with patch("boto3.client") as mock_client_func:
        mock_client_func.side_effect = Exception("Test exception")

        result = use_aws.get_available_operations("invalid-service")

        assert result == []


def test_use_aws_schema_generation_exception():
    """Test use_aws when schema generation raises an exception."""
    # Mock the dependencies
    mock_available_services = ["ec2"]
    mock_available_operations = ["describe_instances"]

    with (
        patch(
            "strands_tools.use_aws.get_available_services",
            return_value=mock_available_services,
        ),
        patch(
            "strands_tools.use_aws.get_available_operations",
            return_value=mock_available_operations,
        ),
        patch("strands_tools.use_aws.get_boto3_client") as mock_get_client,
        patch("strands_tools.use_aws.generate_input_schema") as mock_generate_schema,
    ):
        # Configure mocks
        mock_client = MagicMock()
        mock_describe = MagicMock()
        mock_describe.side_effect = use_aws.ParamValidationError(report="Invalid parameter")
        mock_client.describe_instances = mock_describe
        mock_get_client.return_value = mock_client

        # Make generate_input_schema raise an exception
        mock_generate_schema.side_effect = Exception("Schema generation failed")

        # Create a tool use dictionary
        tool_use = {
            "toolUseId": "test-tool-use-id",
            "input": {
                "service_name": "ec2",
                "operation_name": "describe_instances",
                "parameters": {"InvalidParam": "value"},
                "region": "us-west-2",
                "label": "Schema Exception Test",
            },
        }

        result = use_aws.use_aws(tool=tool_use)

        assert result["status"] == "error"
        assert "Validation error:" in result["content"][0]["text"]
        # Confirm it doesn't include the schema since generation failed
        assert len(result["content"]) == 1


def test_convert_datetime_to_str():
    """Test convert_datetime_to_str with various data types."""
    from datetime import datetime, timezone

    # Test with a datetime object
    dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert data_util.convert_datetime_to_str(dt) == "2023-01-01 12:00:00+0000"

    # Test with a dictionary containing datetime
    data = {"date": dt, "name": "test"}
    result = data_util.convert_datetime_to_str(data)
    assert result["date"] == "2023-01-01 12:00:00+0000"
    assert result["name"] == "test"

    # Test with a list containing datetime
    data = [dt, "test", 123]
    result = data_util.convert_datetime_to_str(data)
    assert result[0] == "2023-01-01 12:00:00+0000"
    assert result[1] == "test"
    assert result[2] == 123

    # Test with a regular value
    assert data_util.convert_datetime_to_str("test") == "test"
    assert data_util.convert_datetime_to_str(123) == 123


def test_to_snake_case():
    """Test to_snake_case function."""
    assert data_util.to_snake_case("HelloWorld") == "hello_world"
    assert data_util.to_snake_case("helloWorld") == "hello_world"
    assert data_util.to_snake_case("hello") == "hello"
    assert data_util.to_snake_case("hello_world") == "hello_world"


@patch("asyncio.get_event_loop")
@patch("asyncio.new_event_loop")
@patch("asyncio.set_event_loop")
def test_get_user_input_new_loop(mock_set_event_loop, mock_new_event_loop, mock_get_event_loop):
    """Test get_user_input when there's no existing event loop."""
    # Setup mocks
    mock_get_event_loop.side_effect = RuntimeError("No running event loop")
    mock_loop = MagicMock()
    mock_new_event_loop.return_value = mock_loop
    mock_loop.run_until_complete.return_value = "y"

    # Call function
    with patch("strands_tools.utils.user_input.get_user_input_async") as mock_get_async:
        mock_get_async.return_value = "y"
        result = user_input.get_user_input("Prompt", "n")

    # Verify a new loop was created and set
    mock_new_event_loop.assert_called_once()
    mock_set_event_loop.assert_called_once_with(mock_loop)
    assert result == "y"
