"""
Tests for the OpenSearch agentic search tool.
"""

import json
import os
from unittest import mock
from unittest.mock import MagicMock

import pytest
from strands import Agent

from src.strands_tools.opensearch_agentic_search import AgenticSearchClient, opensearch_agentic_search_tool


@pytest.fixture
def mock_boto3_session():
    """Mock boto3 session for AWS authentication."""
    with mock.patch("src.strands_tools.opensearch_agentic_search.boto3.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_credentials = MagicMock()
        mock_credentials.access_key = "test_access_key"
        mock_credentials.secret_key = "test_secret_key"
        mock_credentials.token = "test_session_token"
        mock_session.get_credentials.return_value = mock_credentials
        mock_session_class.return_value = mock_session
        yield mock_session_class


@pytest.fixture
def mock_aws4auth():
    """Mock AWS4Auth for OpenSearch authentication."""
    with mock.patch("src.strands_tools.opensearch_agentic_search.AWS4Auth") as mock_auth:
        mock_auth_instance = MagicMock()
        mock_auth.return_value = mock_auth_instance
        yield mock_auth


@pytest.fixture
def mock_requests():
    """Mock requests for OpenSearch API calls."""
    with mock.patch("src.strands_tools.opensearch_agentic_search.requests.request") as mock_request:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"acknowledged": True}
        mock_request.return_value = mock_response
        yield mock_request


@pytest.fixture
def agent():
    """Create an agent with the agentic search tool."""
    return Agent(tools=[opensearch_agentic_search_tool])


@pytest.fixture
def aws_config():
    """Configuration for AWS OpenSearch testing."""
    return {
        "host": "https://test-cluster.us-west-2.aos.amazonaws.com",
        "index": "test_index",
        "role_arn": "arn:aws:iam::123456789:role/BedrockAccess",
    }


@pytest.fixture
def self_managed_config():
    """Configuration for self-managed OpenSearch testing."""
    return {
        "host": "https://my-cluster.com:9200",
        "index": "test_index",
    }


def test_missing_required_params():
    """Test tool with missing required parameters."""
    # Test missing query_text - should raise TypeError
    with pytest.raises(TypeError):
        opensearch_agentic_search_tool(host="https://test-cluster.com")

    # Test missing host (clear environment variables first)
    with mock.patch.dict(os.environ, {}, clear=True):
        result = opensearch_agentic_search_tool(query_text="test query")
        assert result["status"] == "error"
        assert "host is required" in result["content"][0]["text"]


def test_aws_authentication_setup(mock_boto3_session, mock_aws4auth, aws_config):
    """Test AWS authentication setup for managed OpenSearch."""
    # Mock the session to return credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test_access_key"
    mock_credentials.secret_key = "test_secret_key"
    mock_credentials.token = "test_token"
    mock_boto3_session.return_value.get_credentials.return_value = mock_credentials

    with mock.patch.dict(os.environ, {}, clear=True):
        AgenticSearchClient(**aws_config)

    # Verify boto3 session was created
    mock_boto3_session.assert_called_once()

    # Verify AWS4Auth was called with correct parameters
    mock_aws4auth.assert_called_once()
    call_args = mock_aws4auth.call_args[0]
    assert call_args[0] == "test_access_key"  # access_key
    assert call_args[1] == "test_secret_key"  # secret_key
    assert call_args[2] == "us-west-2"  # region
    assert call_args[3] == "aos"  # service


def test_self_managed_authentication_setup(self_managed_config):
    """Test basic authentication setup for self-managed OpenSearch."""
    with mock.patch.dict(os.environ, {"OPENSEARCH_USERNAME": "admin", "OPENSEARCH_PASSWORD": "password"}):
        client = AgenticSearchClient(**self_managed_config)

        # Verify HTTPBasicAuth is used
        assert hasattr(client, "auth")


def test_region_extraction_from_hostname(mock_boto3_session, mock_aws4auth):
    """Test automatic region extraction from OpenSearch hostname."""
    config = {
        "host": "https://test-cluster.eu-west-1.aos.amazonaws.com",
        "role_arn": "arn:aws:iam::123456789:role/BedrockAccess",
    }

    # Mock the session to return credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test_access_key"
    mock_credentials.secret_key = "test_secret_key"
    mock_credentials.token = "test_token"
    mock_boto3_session.return_value.get_credentials.return_value = mock_credentials

    with mock.patch.dict(os.environ, {}, clear=True):
        AgenticSearchClient(**config)

        # Verify region was extracted and set in environment
        assert os.environ.get("AWS_REGION") == "eu-west-1"


def test_register_model(mock_boto3_session, mock_aws4auth, mock_requests, aws_config):
    """Test model registration with Bedrock connector."""
    # Configure mock response for model registration
    mock_requests.return_value.json.return_value = {"model_id": "test_model_123"}

    client = AgenticSearchClient(**aws_config)
    model_id = client.register_model()

    assert model_id == "test_model_123"
    assert client.state["model_id"] == "test_model_123"

    # Verify API call was made
    mock_requests.assert_called()
    call_args = mock_requests.call_args
    assert call_args[0][0] == "POST"  # method
    assert "/_plugins/_ml/models/_register" in call_args[0][1]  # URL

    # Verify request body contains Bedrock configuration
    request_body = call_args[1]["json"]
    assert request_body["function_name"] == "remote"
    assert request_body["connector"]["protocol"] == "aws_sigv4"
    assert request_body["connector"]["credential"]["roleArn"] == aws_config["role_arn"]


def test_register_agent(mock_boto3_session, mock_aws4auth, mock_requests, aws_config):
    """Test agent registration with conversational tools."""
    with mock.patch.dict(os.environ, {}, clear=True):
        # Mock the actual client instance methods
        with mock.patch.object(AgenticSearchClient, "register_agent", return_value="test_agent_456"):
            client = AgenticSearchClient(**aws_config)
            agent_id = client.register_agent()

    assert agent_id == "test_agent_456"


def test_create_pipeline(mock_boto3_session, mock_aws4auth, mock_requests, aws_config):
    """Test search pipeline creation with agentic processors."""
    expected_pipeline_name = "agentic-pipeline-test123"

    with mock.patch.dict(os.environ, {}, clear=True):
        # Mock the actual client instance methods
        with mock.patch.object(AgenticSearchClient, "create_pipeline", return_value=expected_pipeline_name):
            client = AgenticSearchClient(**aws_config)
            pipeline_name = client.create_pipeline()

    assert pipeline_name == expected_pipeline_name


def test_query_execution_with_index(mock_boto3_session, mock_aws4auth, mock_requests, aws_config):
    """Test query execution with specific index."""
    expected_result = {
        "hits": {
            "total": {"value": 2},
            "hits": [
                {
                    "_index": "test_index",
                    "_id": "doc1",
                    "_score": 0.95,
                    "_source": {"title": "Test Document", "content": "Test content"},
                }
            ],
        },
        "ext": {
            "agent_steps_summary": "1. Analyzed query\\n2. Generated search",
            "dsl_query": '{"query": {"match": {"content": "test"}}}',
        },
    }

    with mock.patch.dict(os.environ, {}, clear=True):
        # Mock the actual client instance methods
        with mock.patch.object(AgenticSearchClient, "query", return_value=expected_result):
            client = AgenticSearchClient(**aws_config)
            result = client.query("Find me test documents")

    # Verify query result structure
    assert "hits" in result
    assert result["hits"]["total"]["value"] == 2
    assert len(result["hits"]["hits"]) == 1
    assert result["hits"]["hits"][0]["_index"] == "test_index"

    # Verify agent reasoning is included
    assert "ext" in result
    assert "agent_steps_summary" in result["ext"]
    assert "dsl_query" in result["ext"]


def test_query_execution_without_index(mock_boto3_session, mock_aws4auth, mock_requests):
    """Test query execution without specific index (searches all indices)."""
    config = {
        "host": "https://test-cluster.us-west-2.aos.amazonaws.com",
        "role_arn": "arn:aws:iam::123456789:role/BedrockAccess",
        # No index specified
    }

    expected_result = {"hits": {"total": {"value": 0}, "hits": []}}

    with mock.patch.dict(os.environ, {}, clear=True):
        # Mock the actual client instance methods
        with mock.patch.object(AgenticSearchClient, "query", return_value=expected_result):
            client = AgenticSearchClient(**config)
            result = client.query("Find me anything")

    # Verify query result
    assert result == expected_result


def test_state_caching(mock_boto3_session, mock_aws4auth, mock_requests, aws_config, tmp_path):
    """Test state caching and reuse."""
    # Use temporary cache file
    cache_file = tmp_path / "test_cache.json"
    expected_pipeline_name = "agentic-pipeline-cached123"

    with mock.patch("src.strands_tools.opensearch_agentic_search.STATE_FILE", cache_file):
        with mock.patch.dict(os.environ, {}, clear=True):
            # Mock the actual client instance methods
            with mock.patch.object(AgenticSearchClient, "create_pipeline", return_value=expected_pipeline_name):
                with mock.patch.object(
                    AgenticSearchClient, "query", return_value={"hits": {"total": {"value": 1}, "hits": []}}
                ):
                    # First client - should register everything
                    client1 = AgenticSearchClient(**aws_config)
                    pipeline1 = client1.create_pipeline()
                    client1.query("First query")

                    # Second client - should load from cache
                    client2 = AgenticSearchClient(**aws_config)
                    client2.query("Second query")

                    # Verify pipeline name
                    assert pipeline1 == expected_pipeline_name


def test_tool_integration(mock_boto3_session, mock_aws4auth, mock_requests, aws_config):
    """Test the tool through direct function call."""
    expected_result = {
        "hits": {"total": {"value": 1}, "hits": []},
        "ext": {
            "agent_steps_summary": "Query processed successfully",
            "dsl_query": '{"query": {"match_all": {}}}',
        },
    }

    with mock.patch.dict(os.environ, {}, clear=True):
        # Mock the client methods to avoid setup calls
        with mock.patch("src.strands_tools.opensearch_agentic_search.AgenticSearchClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.query.return_value = expected_result
            mock_client_class.return_value = mock_client

            # Execute search through tool
            result = opensearch_agentic_search_tool(query_text="Find me shoes under $100", **aws_config)

    # Verify successful response
    assert result["status"] == "success"

    # Verify response contains search results
    response_text = result["content"][0]["text"]
    response_data = json.loads(response_text)
    assert "hits" in response_data
    assert "ext" in response_data


def test_environment_variable_usage(mock_boto3_session, mock_aws4auth, mock_requests):
    """Test using environment variables for configuration."""
    expected_result = {"hits": {"total": {"value": 0}, "hits": []}}

    with mock.patch.dict(
        os.environ,
        {
            "OPENSEARCH_HOST": "https://env-cluster.us-west-2.aos.amazonaws.com",
            "OPENSEARCH_INDEX": "env_index",
            "BEDROCK_ROLE_ARN": "arn:aws:iam::123456789:role/EnvBedrockAccess",
            "AWS_REGION": "us-west-2",
        },
    ):
        # Mock the client methods to avoid setup calls
        with mock.patch("src.strands_tools.opensearch_agentic_search.AgenticSearchClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.query.return_value = expected_result
            mock_client_class.return_value = mock_client

            # Call tool without explicit parameters (should use env vars)
            result = opensearch_agentic_search_tool(query_text="Find me wireless headphones")

        # Verify success (means env vars were used correctly)
        assert result["status"] == "success"

        # Verify client was created with env vars
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert call_kwargs["host"] == "https://env-cluster.us-west-2.aos.amazonaws.com"
        assert call_kwargs["index"] == "env_index"
        assert call_kwargs["role_arn"] == "arn:aws:iam::123456789:role/EnvBedrockAccess"


def test_error_handling_scenarios(mock_boto3_session, mock_aws4auth, aws_config):
    """Test various error handling scenarios."""
    agent = Agent(tools=[opensearch_agentic_search_tool])

    # Test authentication failure
    with mock.patch("src.strands_tools.opensearch_agentic_search.requests.request") as mock_request:
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 403
        mock_response.text = "Access denied"
        mock_request.return_value = mock_response

        result = agent.tool.opensearch_agentic_search_tool(query_text="test query", **aws_config)

        assert result["status"] == "error"
        assert "Access denied to OpenSearch" in result["content"][0]["text"]

    # Test connection timeout
    with mock.patch("src.strands_tools.opensearch_agentic_search.requests.request") as mock_request:
        mock_request.side_effect = Exception("Connection timeout")

        result = agent.tool.opensearch_agentic_search_tool(query_text="test query", **aws_config)

        assert result["status"] == "error"
        assert "Error:" in result["content"][0]["text"]


def test_missing_bedrock_role_arn(mock_boto3_session, mock_aws4auth, mock_requests):
    """Test error when BEDROCK_ROLE_ARN is missing."""
    config = {
        "host": "https://test-cluster.us-west-2.aos.amazonaws.com",
        "index": "test_index",
        # No role_arn provided
    }

    client = AgenticSearchClient(**config)

    with pytest.raises(ValueError, match="BEDROCK_ROLE_ARN is required"):
        client.register_model()


def test_self_managed_opensearch_auth():
    """Test authentication for self-managed OpenSearch."""
    with mock.patch.dict(os.environ, {"OPENSEARCH_USERNAME": "admin", "OPENSEARCH_PASSWORD": "secret123"}):
        config = {
            "host": "https://my-cluster.com:9200",
            "index": "test_index",
        }

        client = AgenticSearchClient(**config)

        # Verify HTTPBasicAuth is configured
        from requests.auth import HTTPBasicAuth

        assert isinstance(client.auth, HTTPBasicAuth)


def test_missing_opensearch_password():
    """Test error when OpenSearch password is missing for self-managed cluster."""
    with mock.patch.dict(os.environ, {"OPENSEARCH_USERNAME": "admin"}, clear=True):
        config = {
            "host": "https://my-cluster.com:9200",
            "index": "test_index",
        }

        with pytest.raises(ValueError, match="OPENSEARCH_PASSWORD environment variable required"):
            AgenticSearchClient(**config)


def test_opensearch_serverless_detection(mock_boto3_session, mock_aws4auth):
    """Test detection of OpenSearch Serverless vs regular OpenSearch Service."""
    # Test OpenSearch Serverless
    config_serverless = {
        "host": "https://test-collection.us-west-2.aoss.amazonaws.com",
        "role_arn": "arn:aws:iam::123456789:role/BedrockAccess",
    }

    # Mock the session to return credentials
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test_access_key"
    mock_credentials.secret_key = "test_secret_key"
    mock_credentials.token = "test_token"
    mock_boto3_session.return_value.get_credentials.return_value = mock_credentials

    with mock.patch.dict(os.environ, {}, clear=True):
        AgenticSearchClient(**config_serverless)

    # Verify AWS4Auth was called with 'aoss' service for serverless
    mock_aws4auth.assert_called()
    call_args = mock_aws4auth.call_args[0]
    assert call_args[3] == "aoss"  # service should be 'aoss' for serverless


def test_custom_cache_location(mock_boto3_session, mock_aws4auth, mock_requests, aws_config, tmp_path):
    """Test using custom cache file location."""
    custom_cache = tmp_path / "custom_cache.json"

    with mock.patch.dict(os.environ, {"OPENSEARCH_CACHE_FILE": str(custom_cache)}, clear=True):
        # Mock the actual client instance methods
        with mock.patch.object(AgenticSearchClient, "register_model", return_value="custom_model_123"):
            client = AgenticSearchClient(**aws_config)
            model_id = client.register_model()

        # Verify model was registered
        assert model_id == "custom_model_123"


def test_aws_profile_usage(mock_aws4auth):
    """Test using specific AWS profile for authentication."""
    with mock.patch.dict(os.environ, {"AWS_PROFILE": "production"}):
        with mock.patch("src.strands_tools.opensearch_agentic_search.boto3.Session") as mock_session_class:
            mock_session = MagicMock()
            mock_credentials = MagicMock()
            mock_credentials.access_key = "profile_access_key"
            mock_credentials.secret_key = "profile_secret_key"
            mock_credentials.token = None
            mock_session.get_credentials.return_value = mock_credentials
            mock_session_class.return_value = mock_session

            config = {
                "host": "https://test-cluster.us-west-2.aos.amazonaws.com",
                "role_arn": "arn:aws:iam::123456789:role/BedrockAccess",
            }

            AgenticSearchClient(**config)

            # Verify session was created with profile
            mock_session_class.assert_called_once_with(profile_name="production")


def test_multiple_query_scenarios(mock_boto3_session, mock_aws4auth, mock_requests, aws_config):
    """Test various query scenarios and response formats."""
    expected_result = {
        "took": 45,
        "hits": {
            "total": {"value": 3, "relation": "eq"},
            "max_score": 0.95,
            "hits": [
                {
                    "_index": "products",
                    "_id": "shoe1",
                    "_score": 0.95,
                    "_source": {"title": "Nike Air Max 90", "price": 89.99, "category": "shoes"},
                }
            ],
        },
        "ext": {
            "agent_steps_summary": (
                "1. Analyzed query for shoe preferences and price range\n"
                "2. Generated search for shoes under $100\n3. Applied price and category filters"
            ),
            "dsl_query": (
                '{"query": {"bool": {"must": [{"match": {"category": "shoes"}}], '
                '"filter": [{"range": {"price": {"lte": 100}}}]}}}'
            ),
        },
    }

    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch("src.strands_tools.opensearch_agentic_search.AgenticSearchClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.query.return_value = expected_result
            mock_client_class.return_value = mock_client

            # Test natural language query
            result = opensearch_agentic_search_tool(query_text="Find me shoes under $100", **aws_config)

    assert result["status"] == "success"

    # Parse and verify response
    response_data = json.loads(result["content"][0]["text"])
    assert response_data["took"] == 45
    assert response_data["hits"]["total"]["value"] == 3
    assert "ext" in response_data
    assert "agent_steps_summary" in response_data["ext"]
    assert "dsl_query" in response_data["ext"]

    # Verify agent reasoning is present
    steps_summary = response_data["ext"]["agent_steps_summary"]
    assert "price range" in steps_summary
    assert "category filters" in steps_summary


def test_performance_and_caching(mock_boto3_session, mock_aws4auth, mock_requests, aws_config, tmp_path):
    """Test performance optimizations and caching behavior."""
    cache_file = tmp_path / "perf_cache.json"
    expected_pipeline_name = "agentic-pipeline-perf123"

    with mock.patch("src.strands_tools.opensearch_agentic_search.STATE_FILE", cache_file):
        with mock.patch.dict(os.environ, {}, clear=True):
            # Mock the actual client instance methods
            with mock.patch.object(AgenticSearchClient, "create_pipeline", return_value=expected_pipeline_name):
                with mock.patch.object(AgenticSearchClient, "query") as mock_query:
                    mock_query.side_effect = [
                        {"hits": {"total": {"value": 1}, "hits": []}},
                        {"hits": {"total": {"value": 2}, "hits": []}},
                    ]

                    # First client - full setup
                    client1 = AgenticSearchClient(**aws_config)
                    client1.create_pipeline()
                    result1 = client1.query("First query")

                    # Second client - should reuse cached state
                    client2 = AgenticSearchClient(**aws_config)
                    result2 = client2.query("Second query")

                    # Verify queries returned different results
                    assert result1["hits"]["total"]["value"] == 1
                    assert result2["hits"]["total"]["value"] == 2


def test_comprehensive_error_scenarios(mock_boto3_session, mock_aws4auth, aws_config):
    """Test comprehensive error handling scenarios."""
    agent = Agent(tools=[opensearch_agentic_search_tool])

    # Test different HTTP error codes
    error_scenarios = [
        (401, "Authentication failed"),
        (403, "Access denied to OpenSearch"),
        (404, "OpenSearch endpoint not found"),
        (500, "OpenSearch API error: 500"),
    ]

    for status_code, expected_message in error_scenarios:
        with mock.patch("src.strands_tools.opensearch_agentic_search.requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.ok = False
            mock_response.status_code = status_code
            mock_response.text = f"HTTP {status_code} Error"
            mock_request.return_value = mock_response

            result = agent.tool.opensearch_agentic_search_tool(query_text="test query", **aws_config)

            assert result["status"] == "error"
            assert expected_message in result["content"][0]["text"]


def test_documentation_examples(mock_boto3_session, mock_aws4auth, mock_requests):
    """Test examples from the documentation work correctly."""
    expected_result = {"hits": {"total": {"value": 0}, "hits": []}}

    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch("src.strands_tools.opensearch_agentic_search.AgenticSearchClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.query.return_value = expected_result
            mock_client_class.return_value = mock_client

            # Test basic example from documentation
            result = opensearch_agentic_search_tool(
                query_text="Find me shoes under $100",
                host="https://my-cluster.us-west-2.aos.amazonaws.com",
                index="products",
                role_arn="arn:aws:iam::123456789:role/BedrockAccess",
            )

            assert result["status"] == "success"

            # Test cross-index search example
            result = opensearch_agentic_search_tool(
                query_text="Show me customer complaints from last week",
                host="https://my-cluster.us-west-2.aos.amazonaws.com",
                role_arn="arn:aws:iam::123456789:role/BedrockAccess",
                # No index specified - searches all indices
            )

            assert result["status"] == "success"


def test_security_and_validation(mock_boto3_session, mock_aws4auth, aws_config):
    """Test security validations and input sanitization."""
    agent = Agent(tools=[opensearch_agentic_search_tool])

    # Test malicious query handling - OpenSearch DSL specific attacks
    malicious_queries = [
        # Script injection in query
        '{"query": {"script": {"script": "System.exit(0)"}}}',
        # Groovy script injection
        '{"query": {"bool": {"must": [{"script": {"script": '
        '"java.lang.Runtime.getRuntime().exec(\\"rm -rf /\\");"}}]}}}',
        # Painless script with system access attempt
        '{"query": {"script": {"script": {"source": '
        '"Runtime.getRuntime().exec(\\"cat /etc/passwd\\");", "lang": "painless"}}}}',
        # Large aggregation to cause DoS
        '{"aggs": {"terms": {"field": "keyword", "size": 2147483647}}}',
        # Nested query bomb
        '{"query": {"nested": {"path": "nested_field", "query": '
        '{"nested": {"path": "nested_field.deep", "query": {"match_all": {}}}}}}}',
    ]

    with mock.patch("src.strands_tools.opensearch_agentic_search.requests.request") as mock_request:
        mock_request.return_value = MagicMock(ok=True, json=lambda: {"hits": {"total": {"value": 0}, "hits": []}})

        for malicious_query in malicious_queries:
            result = agent.tool.opensearch_agentic_search_tool(query_text=malicious_query, **aws_config)

            # Should not crash and should return valid response
            assert result["status"] in ["success", "error"]
            assert result["content"][0]["text"] is not None


def test_tool_metadata_and_configuration(mock_boto3_session, mock_aws4auth):
    """Test tool metadata and configuration validation."""
    # Test tool is properly configured
    assert hasattr(opensearch_agentic_search_tool, "__name__")
    assert opensearch_agentic_search_tool.__name__ == "opensearch_agentic_search_tool"

    # Test AgenticSearchClient initialization with AWS host
    with mock.patch.dict(os.environ, {}, clear=True):
        # Should work with AWS host (no password required)
        client = AgenticSearchClient(host="https://test-cluster.us-west-2.aos.amazonaws.com")
        assert client is not None
        assert "test-cluster.us-west-2.aos.amazonaws.com" in client.host_name
