"""
Amazon Bedrock Knowledge Base retrieval tool for Strands Agent.

This module provides functionality to perform semantic search against Amazon Bedrock
Knowledge Bases, enabling natural language queries against your organization's documents.
It uses vector-based similarity matching to find relevant information and returns results
ordered by relevance score.

Key Features:
1. Semantic Search:
   • Vector-based similarity matching
   • Relevance scoring (0.0-1.0)
   • Score-based filtering

2. Advanced Configuration:
   • Custom result limits
   • Score thresholds
   • Regional support
   • Multiple knowledge bases

3. Response Format:
   • Sorted by relevance
   • Includes metadata
   • Source tracking
   • Score visibility

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import retrieve

agent = Agent(tools=[retrieve])

# Basic search with default knowledge base and region
results = agent.tool.retrieve(text="What is the STRANDS SDK?")

# Advanced search with custom parameters
results = agent.tool.retrieve(
    text="deployment steps for production",
    numberOfResults=5,
    score=0.7,
    knowledgeBaseId="custom-kb-id",
    region="us-east-1",
    retrieveFilter={
        "andAll": [
            {"equals": {"key": "category", "value": "security"}},
            {"greaterThan": {"key": "year", "value": "2022"}}
        ]
    }
)
```

See the retrieve function docstring for more details on available parameters and options.
"""

import os
from typing import Any, Dict, List

import boto3
from botocore.config import Config as BotocoreConfig
from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
    "name": "retrieve",
    "description": """Retrieves knowledge based on the provided text from Amazon Bedrock Knowledge Bases.

Key Features:
1. Semantic Search:
   - Vector-based similarity matching
   - Relevance scoring (0.0-1.0)
   - Score-based filtering

2. Advanced Configuration:
   - Custom result limits
   - Score thresholds
   - Regional support
   - Multiple knowledge bases

3. Response Format:
   - Sorted by relevance
   - Includes metadata
   - Source tracking
   - Score visibility

4. Example Response:
   {
     "content": {
       "text": "Document content...",
       "type": "TEXT"
     },
     "location": {
       "customDocumentLocation": {
         "id": "document_id"
       },
       "type": "CUSTOM"
     },
     "metadata": {
       "x-amz-bedrock-kb-source-uri": "source_uri",
       "x-amz-bedrock-kb-chunk-id": "chunk_id",
       "x-amz-bedrock-kb-data-source-id": "data_source_id"
     },
     "score": 0.95
   }

Usage Examples:
1. Basic search:
   retrieve(text="What is STRANDS?")

2. With score threshold:
   retrieve(text="deployment steps", score=0.7)

3. Limited results:
   retrieve(text="best practices", numberOfResults=3)

4. Custom knowledge base:
   retrieve(text="query", knowledgeBaseId="custom-kb-id")""",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The query to retrieve relevant knowledge.",
                },
                "numberOfResults": {
                    "type": "integer",
                    "description": "The maximum number of results to return. Default is 5.",
                },
                "knowledgeBaseId": {
                    "type": "string",
                    "description": "The ID of the knowledge base to retrieve from.",
                },
                "region": {
                    "type": "string",
                    "description": "The AWS region name. Default is 'us-west-2'.",
                },
                "score": {
                    "type": "number",
                    "description": (
                        "Minimum relevance score threshold (0.0-1.0). Results below this score will be filtered out. "
                        "Default is 0.4."
                    ),
                    "default": 0.4,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "profile_name": {
                    "type": "string",
                    "description": (
                        "Optional: AWS profile name to use from ~/.aws/credentials. Defaults to default profile if not "
                        "specified."
                    ),
                },
            },
            "required": ["text"],
        }
    },
}


def filter_results_by_score(results: List[Dict[str, Any]], min_score: float) -> List[Dict[str, Any]]:
    """
    Filter results based on minimum score threshold.

    This function takes the raw results from a knowledge base query and removes
    any items that don't meet the minimum relevance score threshold.

    Args:
        results: List of retrieval results from Bedrock Knowledge Base
        min_score: Minimum score threshold (0.0-1.0). Only results with scores
            greater than or equal to this value will be returned.

    Returns:
        List of filtered results that meet or exceed the score threshold
    """
    return [result for result in results if result.get("score", 0.0) >= min_score]


def format_results_for_display(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieval results for readable display.

    This function takes the raw results from a knowledge base query and formats
    them into a human-readable string with scores, document IDs, and content.

    Args:
        results: List of retrieval results from Bedrock Knowledge Base

    Returns:
        Formatted string containing the results in a readable format
    """
    if not results:
        return "No results found above score threshold."

    formatted = []
    for result in results:
        doc_id = result.get("location", {}).get("customDocumentLocation", {}).get("id", "Unknown")
        score = result.get("score", 0.0)
        formatted.append(f"\nScore: {score:.4f}")
        formatted.append(f"Document ID: {doc_id}")

        content = result.get("content", {})
        if content and isinstance(content.get("text"), str):
            text = content["text"]
            formatted.append(f"Content: {text}\n")

    return "\n".join(formatted)


def retrieve(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Retrieve relevant knowledge from Amazon Bedrock Knowledge Base.

    This tool uses Amazon Bedrock Knowledge Bases to perform semantic search against your
    organization's documents. It returns results sorted by relevance score, with the ability
    to filter results that don't meet a minimum score threshold.

    How It Works:
    ------------
    1. The provided query text is sent to Amazon Bedrock Knowledge Base
    2. The service performs vector-based semantic search against indexed documents
    3. Results are returned with relevance scores (0.0-1.0) indicating match quality
    4. Results below the minimum score threshold are filtered out
    5. Remaining results are formatted for readability and returned

    Common Usage Scenarios:
    ---------------------
    - Answering user questions from product documentation
    - Finding relevant information in company policies
    - Retrieving context from technical manuals
    - Searching for relevant sections in research papers
    - Looking up information in legal documents

    Args:
        tool: Tool use information containing input parameters:
            text: The query text to search for in the knowledge base
            numberOfResults: Maximum number of results to return (default: 10)
            knowledgeBaseId: The ID of the knowledge base to query (default: from environment)
            region: AWS region where the knowledge base is located (default: us-west-2)
            score: Minimum relevance score threshold (default: 0.4)
            profile_name: Optional AWS profile name to use
            retrieveFilter: Optional filter to apply to the retrieval results

    Returns:
        Dictionary containing status and response content in the format:
        {
            "toolUseId": "unique_id",
            "status": "success|error",
            "content": [{"text": "Retrieved results or error message"}]
        }

        Success case: Returns formatted results from the knowledge base
        Error case: Returns information about what went wrong during retrieval

    Notes:
        - The knowledge base ID can be set via the KNOWLEDGE_BASE_ID environment variable
        - The AWS region can be set via the AWS_REGION environment variable
        - The minimum score threshold can be set via the MIN_SCORE environment variable
        - Results are automatically filtered based on the minimum score threshold
        - AWS credentials must be configured properly for this tool to work
    """
    default_knowledge_base_id = os.getenv("KNOWLEDGE_BASE_ID")
    default_aws_region = os.getenv("AWS_REGION", "us-west-2")
    default_min_score = float(os.getenv("MIN_SCORE", "0.4"))
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    try:
        # Extract parameters
        query = tool_input["text"]
        number_of_results = tool_input.get("numberOfResults", 10)
        kb_id = tool_input.get("knowledgeBaseId", default_knowledge_base_id)
        region_name = tool_input.get("region", default_aws_region)
        min_score = tool_input.get("score", default_min_score)
        retrieve_filter = tool_input.get("retrieveFilter")

        # Initialize Bedrock client with optional profile name
        profile_name = tool_input.get("profile_name")
        config = BotocoreConfig(user_agent_extra="strands-agents-retrieve")
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
            bedrock_agent_runtime_client = session.client(
                "bedrock-agent-runtime", region_name=region_name, config=config
            )
        else:
            bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=region_name, config=config)

        # Default retrieval configuration
        retrieval_config = {"vectorSearchConfiguration": {"numberOfResults": number_of_results}}

        if retrieve_filter:
            try:
                if _validate_filter(retrieve_filter):
                    retrieval_config["vectorSearchConfiguration"]["filter"] = retrieve_filter
            except ValueError as e:
                return {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": str(e)}],
                }

        # Perform retrieval
        response = bedrock_agent_runtime_client.retrieve(
            retrievalQuery={"text": query}, knowledgeBaseId=kb_id, retrievalConfiguration=retrieval_config
        )

        # Get and filter results
        all_results = response.get("retrievalResults", [])
        filtered_results = filter_results_by_score(all_results, min_score)

        # Format results for display
        formatted_results = format_results_for_display(filtered_results)

        # Return success with formatted results
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {"text": f"Retrieved {len(filtered_results)} results with score >= {min_score}:\n{formatted_results}"}
            ],
        }

    except Exception as e:
        # Return error with details
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error during retrieval: {str(e)}"}],
        }


# A simple validator to check filter is in valid shape
def _validate_filter(retrieve_filter):
    """Validate the structure of a retrieveFilter."""
    try:
        if not isinstance(retrieve_filter, dict):
            raise ValueError("retrieveFilter must be a dictionary")

        # Valid operators according to AWS Bedrock documentation
        valid_operators = [
            "equals",
            "greaterThan",
            "greaterThanOrEquals",
            "in",
            "lessThan",
            "lessThanOrEquals",
            "listContains",
            "notEquals",
            "notIn",
            "orAll",
            "andAll",
            "startsWith",
            "stringContains",
        ]

        # Validate each operator in the filter
        for key, value in retrieve_filter.items():
            if key not in valid_operators:
                raise ValueError(f"Invalid operator: {key}")

            # Validate operator value structure
            if key in ["orAll", "andAll"]:  # Both orAll and andAll require arrays
                if not isinstance(value, list):
                    raise ValueError(f"Value for '{key}' operator must be a list")
                if len(value) < 2:  # Both require minimum 2 items
                    raise ValueError(f"Value for '{key}' operator must contain at least 2 items")
                for sub_filter in value:
                    _validate_filter(sub_filter)
            else:
                if not isinstance(value, dict):
                    raise ValueError(f"Value for '{key}' operator must be a dictionary")
        return True
    except Exception as e:
        raise Exception(f"Unexpected error while validating retrieve filter: {str(e)}") from e
