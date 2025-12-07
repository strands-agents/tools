from unittest.mock import MagicMock, patch

import pytest

from strands_tools.redshift_query import redshift_query


@pytest.fixture
def tool_input_success():
    return {
        "toolUseId": "mock-id-1",
        "input": {
            "sql": "SELECT 1 as test_value",
            "database": "dev",
            "secretArn": "arn:test",
            "clusterIdentifier": "cluster-1",
        },
    }


@patch("strands_tools.redshift_query.boto3.client")
def test_redshift_query_success(mock_boto_client, tool_input_success):
    # Mock Redshift Data API client
    mock_client = MagicMock()
    mock_boto_client.return_value = mock_client

    # Mock execute -> returns fake statement ID
    mock_client.execute_statement.return_value = {"Id": "stmt-123"}

    # Mock polling -> immediately returns FINISHED
    mock_client.describe_statement.return_value = {"Status": "FINISHED"}

    # Mock result set
    mock_client.get_statement_result.return_value = {
        "ColumnMetadata": [{"label": "test_value"}],
        "Records": [[{"stringValue": "1"}]],
    }

    result = redshift_query(tool_input_success)

    assert result["status"] == "success"
    assert "test_value" in result["content"][0]["text"]
    assert "1" in result["content"][0]["text"]


@patch("strands_tools.redshift_query.boto3.client")
def test_redshift_query_failure(mock_boto_client, tool_input_success):
    mock_client = MagicMock()
    mock_boto_client.return_value = mock_client

    mock_client.execute_statement.side_effect = Exception("Execution failed")

    result = redshift_query(tool_input_success)

    assert result["status"] == "error"
    assert "Execution failed" in result["content"][0]["text"]


@patch("strands_tools.redshift_query.boto3.client")
def test_redshift_query_missing_inputs(mock_boto_client):
    bad_input = {
        "toolUseId": "mock-id-2",
        "input": {
            "sql": "SELECT 1",
            "database": "dev",
            "secretArn": "arn:test",
            # Missing clusterIdentifier and workgroupName
        },
    }

    mock_client = MagicMock()
    mock_boto_client.return_value = mock_client

    result = redshift_query(bad_input)

    assert result["status"] == "error"
    assert "clusterIdentifier" in result["content"][0]["text"]
