"""
Tool for performing agentic search using OpenSearch and Bedrock Claude.

This module provides end-to-end capabilities to register models, deploy
agents, create search pipelines, and run natural language queries
against OpenSearch indices using an agentic approach.

Key Features:
-------------
1. Model & Connector Management:
   • register_model: Register a Bedrock Claude model with OpenSearch ML
   • Supports AWS SigV4 authentication with IAM roles
   • Automatic model deployment and timestamped naming

2. Agent Management:
   • register_agent: Create a local conversational agent in OpenSearch
   • Built-in tools: ListIndexTool, IndexMappingTool, WebSearchTool, QueryPlanningTool
   • Conversation memory and configurable max iterations

3. Pipeline Management:
   • create_pipeline: Create agentic search pipeline with processors
   • Request processor: agentic_query_translator for natural language queries
   • Response processor: agentic_context with steps summary and DSL queries

4. Query Execution:
   • query: Execute natural language queries against OpenSearch indices
   • Supports single index or cross-index search
   • Returns structured results with agent reasoning and generated DSL

5. Authentication & State Management:
   • AWS IAM authentication for managed OpenSearch clusters
   • Basic auth fallback for self-managed clusters
   • Local state caching for model, agent, and pipeline reuse

6. Error Handling:
   • Validates required environment variables and parameters
   • Graceful error messages for authentication and API failures
   • Automatic region detection from OpenSearch hostnames

Environment Variables:
----------------------
Required:
- OPENSEARCH_HOST: OpenSearch cluster endpoint
- BEDROCK_ROLE_ARN: AWS IAM role ARN with Bedrock access

Optional:
- OPENSEARCH_INDEX: Default index to search (can search all if not set)
- AWS_REGION: AWS region (auto-detected from hostname if not set)
- AWS_PROFILE: AWS profile for authentication
- OPENSEARCH_CACHE_FILE: Custom cache file location

Usage Examples:
---------------
```python
from agent import AgenticSearchClient, opensearch_agentic_search_tool

# Direct client usage
client = AgenticSearchClient(
    host="https://my-cluster.us-west-2.aos.amazonaws.com",
    index="index-name",  # optional
    role_arn="arn:aws:iam::123456789:role/BedrockAccess"
)

# Full setup (model → agent → pipeline)
model_id = client.register_model()
agent_id = client.register_agent()
pipeline_name = client.create_pipeline()

# Execute agentic search
results = client.query("Find documents about machine learning")

# Using the Strands tool
from strands import Agent
agent = Agent(tools=[opensearch_agentic_search_tool])
agent("Find me red color shoes to gift under $200")
```

Authentication:
---------------
- AWS OpenSearch: Uses AWS4Auth with boto3 session credentials
- Self-managed: Uses HTTPBasicAuth with OPENSEARCH_USERNAME/PASSWORD
- Automatic detection based on hostname (.aos.amazonaws.com or .aoss.amazonaws.com)
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import boto3
import requests
from requests.auth import HTTPBasicAuth
from requests_aws4auth import AWS4Auth
from strands import tool
from strands.types.tools import ToolResult, ToolResultContent

# Path to cache state (model, agent, pipeline)
cache_file = os.environ.get("OPENSEARCH_CACHE_FILE")
if cache_file:
    STATE_FILE = Path(cache_file)
else:
    STATE_FILE = Path.home() / ".opensearch_agentic_cache.json"


class AgenticSearchClient:
    """Agentic Search client for OpenSearch and Bedrock Claude."""

    def __init__(self, host: str, index: str = None, role_arn: str = None):
        # Normalize host URL
        if host.startswith("https://"):
            self.host_url = host
            self.host_name = host.replace("https://", "")
        elif host.startswith("http://"):
            self.host_url = host
            self.host_name = host.replace("http://", "")
        else:
            self.host_name = host
            self.host_url = f"https://{host}"

        self.index = index  # Can be None for searching all indices
        self.role_arn = role_arn
        self.state = self._load_state() or {}
        self.auth = self._setup_auth()

    def _setup_auth(self):
        """Set up authentication for OpenSearch."""
        try:
            if (
                "aos.amazonaws.com" in self.host_name
                or "aoss.amazonaws.com" in self.host_name
                or "es.amazonaws.com" in self.host_name
            ):
                # AWS OpenSearch - use AWS4Auth
                region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")

                # Extract region from host if not set in environment
                if not os.environ.get("AWS_REGION") and not os.environ.get("AWS_DEFAULT_REGION"):
                    match = re.search(r"\.([a-z0-9-]+)\.(aos|aoss|es)\.amazonaws\.com", self.host_name)
                    if match:
                        region = match.group(1)
                        os.environ["AWS_REGION"] = region

                # Use AWS profile if specified, otherwise use default session
                aws_profile = os.environ.get("AWS_PROFILE")
                session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()

                credentials = session.get_credentials()
                if not credentials:
                    raise ValueError("No AWS credentials found. Please configure AWS CLI or set environment variables.")

                if "aoss.amazonaws.com" in self.host_name:
                    service = "aoss"
                elif "es.amazonaws.com" in self.host_name:
                    service = "es"  # Legacy ES service
                elif "aos.amazonaws.com" in self.host_name:
                    service = "aos"  # New AOS service
                else:
                    service = "es"  # fallback for legacy clusters
                auth = AWS4Auth(
                    credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token
                )
                return auth
            else:
                # Self-managed OpenSearch - use basic auth
                username = os.getenv("OPENSEARCH_USERNAME", "admin")
                password = os.getenv("OPENSEARCH_PASSWORD")

                if not password:
                    raise ValueError("OPENSEARCH_PASSWORD environment variable required for basic auth")

                return HTTPBasicAuth(username, password)

        except Exception as e:
            raise ValueError(f"Failed to set up authentication: {e}") from e

    @staticmethod
    def _request(method: str, url: str, body=None, auth=None):
        headers = {"Content-Type": "application/json"}

        try:
            resp = requests.request(method, url, headers=headers, json=body, auth=auth, timeout=180)

            if not resp.ok:
                error_text = resp.text
                if resp.status_code == 403:
                    raise RuntimeError("Access denied to OpenSearch. Check your AWS credentials and IAM permissions.")
                elif resp.status_code == 401:
                    raise RuntimeError("Authentication failed. Check your AWS credentials.")
                elif resp.status_code == 404:
                    raise RuntimeError(f"OpenSearch endpoint not found: {url}")
                else:
                    raise RuntimeError(f"OpenSearch API error: {resp.status_code} - {error_text}")

            return resp.json()

        except requests.exceptions.Timeout:
            raise RuntimeError(f"Request timeout after 30 seconds to {url}") from None
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Connection error to {url}. Check if the OpenSearch cluster is accessible.") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed to {url}: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from {url}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during request to {url}: {str(e)}") from e

    def _load_state(self) -> Dict[str, Any]:
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except Exception:
                return {}
        return {}

    def _save_state(self):
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def register_model(self):
        """
        Register and deploy a Bedrock Claude 4 Sonnet model with OpenSearch ML.

        Returns:
            str: Model ID for use with agents

        Raises:
            ValueError: If BEDROCK_ROLE_ARN is not provided
        """
        if not self.role_arn:
            raise ValueError("BEDROCK_ROLE_ARN is required for model registration")

        timestamp = int(time.time())
        model_name = f"agentic-search-model-{timestamp}"
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")

        body = {
            "name": model_name,
            "function_name": "remote",
            "connector": {
                "name": f"Bedrock Claude 4 Sonnet Connector {timestamp}",
                "description": "Amazon Bedrock connector for Claude 4 Sonnet - Agentic Search",
                "version": 1,
                "protocol": "aws_sigv4",
                "parameters": {
                    "region": region,
                    "service_name": "bedrock",
                    "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                },
                "credential": {"roleArn": self.role_arn},
                "actions": [
                    {
                        "action_type": "predict",
                        "method": "POST",
                        "url": "https://bedrock-runtime.${parameters.region}.amazonaws.com/model/${parameters.model}/converse",
                        "headers": {"content-type": "application/json"},
                        "request_body": (
                            '{ "system": [{"text": "${parameters.system_prompt}"}], '
                            '"messages": [${parameters._chat_history:-}{"role":"user",'
                            '"content":[{"text":"${parameters.user_prompt}"}]}${parameters._interactions:-}]'
                            "${parameters.tool_configs:-} }"
                        ),
                    }
                ],
            },
        }

        resp = self._request("POST", f"{self.host_url}/_plugins/_ml/models/_register?deploy=true", body, auth=self.auth)
        model_id = resp.get("model_id") or resp.get("modelId")

        if not model_id:
            raise RuntimeError(f"Model registration failed - no model_id returned: {resp}")

        self.state["model_id"] = model_id
        self._save_state()
        return model_id

    def register_agent(self):
        """
        Register a local conversational agent in OpenSearch for agentic search.

        Creates an agent with tools: ListIndexTool, IndexMappingTool, WebSearchTool, QueryPlanningTool.

        Returns:
            str: Agent ID for use with search pipelines

        Raises:
            ValueError: If no model is available for agent creation
        """
        # Ensure we have a valid model
        model_id = self.state.get("model_id")
        if model_id:
            try:
                # Verify the model still exists and is deployed
                model_resp = self._request("GET", f"{self.host_url}/_plugins/_ml/models/{model_id}", auth=self.auth)
                model_state = model_resp.get("model_state", "")
                if model_state != "DEPLOYED":
                    model_id = None
            except Exception:
                model_id = None

        if not model_id:
            model_id = self.register_model()

        if not model_id:
            raise ValueError("No model available for agent creation")

        timestamp = int(time.time())
        agent_name = f"agentic-search-agent-{timestamp}"

        body = {
            "name": agent_name,
            "type": "conversational",
            "description": "Conversational agent using Bedrock Claude for agentic search",
            "llm": {"model_id": model_id, "parameters": {"max_iteration": 10}},
            "memory": {"type": "conversation_index"},
            "tools": [
                {"type": "ListIndexTool", "name": "ListIndexTool"},
                {"type": "IndexMappingTool", "name": "IndexMappingTool"},
                {"type": "WebSearchTool", "name": "DuckduckgoWebSearchTool", "parameters": {"engine": "duckduckgo"}},
                {"type": "QueryPlanningTool", "name": "QueryPlanningTool"},
            ],
            "app_type": "os_chat",
            "parameters": {"_llm_interface": "bedrock/converse/claude"},
        }
        resp = self._request("POST", f"{self.host_url}/_plugins/_ml/agents/_register", body, auth=self.auth)
        agent_id = resp.get("agent_id") or resp.get("agentId")

        if not agent_id:
            raise RuntimeError(f"Agent registration failed - no agent_id returned: {resp}")

        self.state["agent_id"] = agent_id
        self._save_state()
        return agent_id

    def create_pipeline(self):
        """
        Create an agentic search pipeline with query translation and context processors.

        Returns:
            str: Pipeline name for use with search queries

        Raises:
            RuntimeError: If no agent is available for pipeline creation
        """
        # Ensure we have a valid agent
        agent_id = self.state.get("agent_id")
        if agent_id:
            try:
                # Verify the agent still exists
                agent_resp = self._request("GET", f"{self.host_url}/_plugins/_ml/agents/{agent_id}", auth=self.auth)
                if agent_resp.get("agent_id") != agent_id:
                    agent_id = None
            except Exception:
                agent_id = None

        if not agent_id:
            agent_id = self.register_agent()

        if not agent_id:
            raise RuntimeError("No agent available for pipeline creation")

        timestamp = int(time.time())
        pipeline_name = f"agentic-pipeline-{timestamp}"

        body = {
            "request_processors": [{"agentic_query_translator": {"agent_id": agent_id}}],
            "response_processors": [{"agentic_context": {"agent_steps_summary": True, "dsl_query": True}}],
        }
        self._request("PUT", f"{self.host_url}/_search/pipeline/{pipeline_name}", body, auth=self.auth)

        self.state["pipeline"] = pipeline_name
        self._save_state()
        return pipeline_name

    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Execute an agentic search query using natural language.

        Args:
            query_text: Natural language search query

        Returns:
            Dict[str, Any]: Search results with agent steps summary and generated DSL query
        """
        pipeline = self.state.get("pipeline") or self.create_pipeline()
        body = {"query": {"agentic": {"query_text": query_text}}}

        # Build the search URL - use specific index if provided, otherwise search all indices
        if self.index:
            search_url = f"{self.host_url}/{self.index}/_search?search_pipeline={pipeline}"
        else:
            search_url = f"{self.host_url}/_search?search_pipeline={pipeline}"

        return self._request("GET", search_url, body, auth=self.auth)


@tool
def opensearch_agentic_search_tool(
    query_text: str, host: str = None, index: str = None, pipeline: str = None, role_arn: str = None
) -> ToolResult:
    """
    Perform agentic search using OpenSearch and Bedrock Claude.

    The agent uses ListIndexTool, IndexMappingTool, WebSearchTool and QueryPlanningTool of local agent to intelligently
    generate and execute search queries against OpenSearch indices.

    Args:
        query_text: The search query text (required)
        host: OpenSearch host URL (optional, uses OPENSEARCH_HOST env var if not provided)
        index: OpenSearch index name (optional, uses OPENSEARCH_INDEX env var if not provided.
               If not provided, searches all indices)
        pipeline: OpenSearch pipeline name (optional, uses OPENSEARCH_PIPELINE env var if not provided)
        role_arn: AWS Bedrock role ARN (optional, uses BEDROCK_ROLE_ARN env var if not provided)
    """
    try:
        if not query_text:
            raise ValueError("query_text is required")

        # Use provided parameters or fall back to environment variables
        host = host or os.getenv("OPENSEARCH_HOST")
        index = index or os.getenv("OPENSEARCH_INDEX")  # Can be None
        pipeline = pipeline or os.getenv("OPENSEARCH_PIPELINE")
        role_arn = role_arn or os.getenv("BEDROCK_ROLE_ARN")

        if not host:
            raise ValueError("host is required (via parameter or OPENSEARCH_HOST environment variable)")

        if not role_arn:
            raise ValueError("role_arn is required (via parameter or BEDROCK_ROLE_ARN environment variable)")

        client = AgenticSearchClient(host=host, index=index, role_arn=role_arn)
        client.create_pipeline()
        result = client.query(query_text)

        # Display essential agentic features after search
        if "ext" in result:
            ext = result["ext"]

            if "agent_steps_summary" in ext:
                print("\n📋 Agent Steps Summary:")
                print(f"{ext['agent_steps_summary']}")

            if "dsl_query" in ext:
                print("\n🔧 Generated DSL Query:")
                try:
                    dsl_json = json.loads(ext["dsl_query"])
                    print(json.dumps(dsl_json, indent=2))
                except (json.JSONDecodeError, TypeError):
                    print(ext["dsl_query"])

        # Display search results summary
        if "hits" in result:
            hits = result["hits"]
            total = hits.get("total", {})
            if isinstance(total, dict):
                total_value = total.get("value", 0)
            else:
                total_value = total
            print(f"\n📊 Found {total_value} results")

        return ToolResult(
            toolUseId="opensearch-agentic-search-tool",
            status="success",
            content=[ToolResultContent(text=json.dumps(result, indent=2))],
        )

    except Exception as e:
        return ToolResult(
            toolUseId="opensearch-agentic-search-tool",
            status="error",
            content=[ToolResultContent(text=f"Error: {str(e)}")],
        )
