"""
Tool for managing memories using Elasticsearch with semantic search capabilities.

This module provides comprehensive memory management capabilities using
Elasticsearch as the backend with vector embeddings for semantic search.

Key Features:
------------
1. Memory Management:
   • record: Store new memories with automatic embedding generation
   • retrieve: Semantic search using vector embeddings and k-NN search
   • list: List all memories with pagination support
   • get: Retrieve specific memories by memory ID
   • delete: Remove specific memories by memory ID

2. Semantic Search:
   • Automatic embedding generation using Amazon Bedrock Titan
   • Vector similarity search with cosine similarity
   • Hybrid search combining vector and text search
   • Namespace-based filtering

3. Index Management:
   • Automatic index creation with proper mappings
   • Vector field configuration for k-NN search
   • Optimized for semantic search performance

4. Error Handling:
   • Connection validation
   • Parameter validation
   • Graceful API error handling
   • Clear error messages

Usage Examples:
--------------
```python
from strands import Agent
from strands_tools.elasticsearch_memory import elasticsearch_memory

# Create agent with direct tool usage
agent = Agent(tools=[elasticsearch_memory])

# Store a memory with semantic embeddings
elasticsearch_memory(
    action="record",
    content="User prefers vegetarian pizza with extra cheese",
    metadata={"category": "food_preferences", "type": "dietary"},
    cloud_id="your-elasticsearch-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)

# Search memories using semantic similarity (vector search)
elasticsearch_memory(
    action="retrieve",
    query="food preferences and dietary restrictions",
    max_results=5,
    cloud_id="your-elasticsearch-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)

# List all memories with pagination
elasticsearch_memory(
    action="list",
    max_results=10,
    cloud_id="your-elasticsearch-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)

# Get specific memory by ID
elasticsearch_memory(
    action="get",
    memory_id="mem_1234567890_abcd1234",
    cloud_id="your-elasticsearch-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)

# Delete a memory
elasticsearch_memory(
    action="delete",
    memory_id="mem_1234567890_abcd1234",
    cloud_id="your-elasticsearch-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)
```

Environment Variables:
---------------------
```bash
# Connection (choose one)
export ELASTICSEARCH_CLOUD_ID="your-cloud-id"           # For Elasticsearch Cloud
export ELASTICSEARCH_URL="https://your-serverless-url"  # For Elasticsearch Serverless

# Required
export ELASTICSEARCH_API_KEY="your-api-key"

# Optional
export ELASTICSEARCH_INDEX_NAME="custom_memories"       # Default: "strands_memory"
export ELASTICSEARCH_NAMESPACE="custom_namespace"       # Default: "default"
export ELASTICSEARCH_EMBEDDING_MODEL="amazon.titan-embed-text-v2:0"
export AWS_REGION="us-east-1"                          # Default: "us-west-2"
```
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import boto3
from elasticsearch import Elasticsearch, NotFoundError
from strands import tool

# Set up logging
logger = logging.getLogger(__name__)


# Custom exceptions for better error handling
class ElasticsearchMemoryError(Exception):
    """Base exception for Elasticsearch memory operations."""

    pass


class ElasticsearchConnectionError(ElasticsearchMemoryError):
    """Raised when connection to Elasticsearch fails."""

    pass


class ElasticsearchMemoryNotFoundError(ElasticsearchMemoryError):
    """Raised when a memory record is not found."""

    pass


class ElasticsearchEmbeddingError(ElasticsearchMemoryError):
    """Raised when embedding generation fails."""

    pass


class ElasticsearchValidationError(ElasticsearchMemoryError):
    """Raised when parameter validation fails."""

    pass


# Define memory actions as an Enum
class MemoryAction(str, Enum):
    """Enum for memory actions."""

    RECORD = "record"
    RETRIEVE = "retrieve"
    LIST = "list"
    GET = "get"
    DELETE = "delete"


# Define required parameters for each action
REQUIRED_PARAMS = {
    MemoryAction.RECORD: ["content"],
    MemoryAction.RETRIEVE: ["query"],
    MemoryAction.LIST: [],
    MemoryAction.GET: ["memory_id"],
    MemoryAction.DELETE: ["memory_id"],
}

# Default settings
DEFAULT_INDEX_NAME = "strands_memory"
DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_EMBEDDING_DIMS = 1024  # Titan v2 returns 1024 dimensions
DEFAULT_MAX_RESULTS = 10


def _ensure_index_exists(es_client: Elasticsearch, index_name: str, es_url: Optional[str] = None):
    """Create index with proper mappings if it doesn't exist."""
    try:
        if not es_client.indices.exists(index=index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "standard"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": DEFAULT_EMBEDDING_DIMS,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "namespace": {"type": "keyword"},
                        "memory_id": {"type": "keyword"},
                        "timestamp": {"type": "date"},
                        "metadata": {"type": "object", "enabled": True},
                    }
                }
            }

            # Add settings only if not using serverless (URL-based connection)
            if not es_url:
                mapping["settings"] = {"number_of_shards": 1, "number_of_replicas": 0, "index.knn": True}

            es_client.indices.create(index=index_name, body=mapping)

    except Exception as e:
        logger.error(f"Failed to create index {index_name}: {str(e)}")
        raise ElasticsearchConnectionError(f"Failed to create index {index_name}: {str(e)}") from e


def _generate_embedding(bedrock_runtime, text: str, embedding_model: str) -> List[float]:
    """
    Generate embeddings for text using Amazon Bedrock Titan.

    This method generates 1024-dimensional vector embeddings using Amazon Bedrock's
    Titan embedding model. These embeddings are used for semantic similarity search.

    Args:
        bedrock_runtime: Bedrock runtime client
        text: Text to generate embeddings for
        embedding_model: Model ID for embedding generation

    Returns:
        List of 1024 float values representing the text embedding

    Raises:
        Exception: If embedding generation fails
    """
    try:
        response = bedrock_runtime.invoke_model(modelId=embedding_model, body=json.dumps({"inputText": text}))

        try:
            response_body = json.loads(response["body"].read())
        except json.JSONDecodeError as e:
            raise ElasticsearchEmbeddingError(f"Invalid JSON response from Bedrock: {str(e)}") from e

        embedding = response_body["embedding"]

        # Validate embedding dimensions
        if len(embedding) != DEFAULT_EMBEDDING_DIMS:
            raise ElasticsearchEmbeddingError(f"Expected {DEFAULT_EMBEDDING_DIMS} dimensions, got {len(embedding)}")

        return embedding

    except ElasticsearchEmbeddingError:
        raise
    except Exception as e:
        raise ElasticsearchEmbeddingError(f"Embedding generation failed: {str(e)}") from e


def _generate_memory_id() -> str:
    """Generate a unique memory ID."""
    timestamp = int(time.time() * 1000)  # milliseconds
    unique_id = str(uuid.uuid4())[:8]
    return f"mem_{timestamp}_{unique_id}"


def _record_memory(
    es_client: Elasticsearch,
    bedrock_runtime,
    index_name: str,
    namespace: str,
    embedding_model: str,
    content: str,
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Store a memory in Elasticsearch with embedding.

    Args:
        es_client: Elasticsearch client
        bedrock_runtime: Bedrock runtime client
        index_name: Elasticsearch index name
        namespace: Memory namespace
        embedding_model: Embedding model ID
        content: Text content to store
        metadata: Optional metadata dictionary

    Returns:
        Dict containing the stored memory information
    """
    # Generate unique memory ID
    memory_id = _generate_memory_id()

    # Generate embedding for semantic search
    embedding = _generate_embedding(bedrock_runtime, content, embedding_model)

    # Prepare document
    doc = {
        "memory_id": memory_id,
        "content": content,
        "embedding": embedding,
        "namespace": namespace,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }

    # Store in Elasticsearch
    response = es_client.index(index=index_name, id=memory_id, body=doc)

    # Return filtered response with embedding metadata
    return {
        "memory_id": memory_id,
        "content": content,
        "namespace": namespace,
        "timestamp": doc["timestamp"],
        "result": response["result"],
        "embedding_info": {"model": embedding_model, "dimensions": len(embedding), "generated": True},
    }


def _retrieve_memories(
    es_client: Elasticsearch,
    bedrock_runtime,
    index_name: str,
    namespace: str,
    embedding_model: str,
    query: str,
    max_results: int,
    next_token: Optional[str] = None,
) -> Dict:
    """
    Retrieve memories using semantic search.

    Args:
        es_client: Elasticsearch client
        bedrock_runtime: Bedrock runtime client
        index_name: Elasticsearch index name
        namespace: Memory namespace
        embedding_model: Embedding model ID
        query: Search query
        max_results: Maximum number of results
        next_token: Pagination token (from parameter for Elasticsearch)

    Returns:
        Dict containing search results
    """
    # Generate embedding for query
    query_embedding = _generate_embedding(bedrock_runtime, query, embedding_model)

    # Calculate offset from next_token
    from_offset = int(next_token) if next_token else 0

    # Perform semantic search using k-NN
    search_body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": max_results,
            "num_candidates": max_results * 3,
            "filter": {"term": {"namespace": namespace}},
        },
        "from": from_offset,
        "size": max_results,
        "_source": ["memory_id", "content", "timestamp", "metadata"],
    }

    response = es_client.search(index=index_name, body=search_body)

    # Format results
    memories = []
    for hit in response["hits"]["hits"]:
        memory = {
            "memory_id": hit["_source"]["memory_id"],
            "content": hit["_source"]["content"],
            "timestamp": hit["_source"]["timestamp"],
            "metadata": hit["_source"].get("metadata", {}),
            "score": hit["_score"],
        }
        memories.append(memory)

    result = {
        "memories": memories,
        "total": response["hits"]["total"]["value"],
        "max_score": response["hits"]["max_score"],
        "search_info": {
            "query_embedding_generated": True,
            "search_type": "k-NN vector similarity",
            "embedding_model": embedding_model,
            "embedding_dimensions": DEFAULT_EMBEDDING_DIMS,
            "similarity_function": "cosine",
        },
    }

    # Add next_token if there are more results
    if from_offset + max_results < response["hits"]["total"]["value"]:
        result["next_token"] = str(from_offset + max_results)

    return result


def _list_memories(
    es_client: Elasticsearch, index_name: str, namespace: str, max_results: int, next_token: Optional[str] = None
) -> Dict:
    """
    List all memories in the namespace.

    Args:
        es_client: Elasticsearch client
        index_name: Elasticsearch index name
        namespace: Memory namespace
        max_results: Maximum number of results
        next_token: Pagination token

    Returns:
        Dict containing all memories
    """
    # Calculate offset from next_token
    from_offset = int(next_token) if next_token else 0

    search_body = {
        "query": {"term": {"namespace": namespace}},
        "sort": [{"timestamp": {"order": "desc"}}],
        "from": from_offset,
        "size": max_results,
        "_source": ["memory_id", "content", "timestamp", "metadata"],
    }

    response = es_client.search(index=index_name, body=search_body)

    # Format results
    memories = []
    for hit in response["hits"]["hits"]:
        memory = {
            "memory_id": hit["_source"]["memory_id"],
            "content": hit["_source"]["content"],
            "timestamp": hit["_source"]["timestamp"],
            "metadata": hit["_source"].get("metadata", {}),
        }
        memories.append(memory)

    result = {"memories": memories, "total": response["hits"]["total"]["value"]}

    # Add next_token if there are more results
    if from_offset + max_results < response["hits"]["total"]["value"]:
        result["next_token"] = str(from_offset + max_results)

    return result


def _get_memory(es_client: Elasticsearch, index_name: str, namespace: str, memory_id: str) -> Dict:
    """
    Get a specific memory by ID.

    Args:
        es_client: Elasticsearch client
        index_name: Elasticsearch index name
        namespace: Memory namespace
        memory_id: Memory ID to retrieve

    Returns:
        Dict containing the memory

    Raises:
        Exception: If memory not found or not in correct namespace
    """
    try:
        response = es_client.get(index=index_name, id=memory_id)
        source = response["_source"]

        # Verify namespace
        if source.get("namespace") != namespace:
            raise ElasticsearchMemoryNotFoundError(f"Memory {memory_id} not found in namespace {namespace}")

        return {
            "memory_id": source["memory_id"],
            "content": source["content"],
            "timestamp": source["timestamp"],
            "metadata": source.get("metadata", {}),
            "namespace": source["namespace"],
        }

    except NotFoundError:
        raise ElasticsearchMemoryNotFoundError(f"Memory {memory_id} not found") from None
    except ElasticsearchMemoryNotFoundError:
        raise
    except Exception as e:
        raise ElasticsearchMemoryError(f"Failed to get memory {memory_id}: {str(e)}") from e


def _delete_memory(es_client: Elasticsearch, index_name: str, namespace: str, memory_id: str) -> Dict:
    """
    Delete a specific memory by ID.

    Args:
        es_client: Elasticsearch client
        index_name: Elasticsearch index name
        namespace: Memory namespace
        memory_id: Memory ID to delete

    Returns:
        Dict containing deletion result

    Raises:
        Exception: If memory not found or deletion fails
    """
    try:
        # First verify the memory exists and is in correct namespace
        _get_memory(es_client, index_name, namespace, memory_id)

        # Delete the memory
        response = es_client.delete(index=index_name, id=memory_id)

        return {"memory_id": memory_id, "result": response["result"]}

    except ElasticsearchMemoryNotFoundError:
        raise
    except NotFoundError:
        raise ElasticsearchMemoryNotFoundError(f"Memory {memory_id} not found") from None
    except Exception as e:
        raise ElasticsearchMemoryError(f"Failed to delete memory {memory_id}: {str(e)}") from e


@tool
def elasticsearch_memory(
    action: str,
    content: Optional[str] = None,
    query: Optional[str] = None,
    memory_id: Optional[str] = None,
    max_results: Optional[int] = None,
    next_token: Optional[str] = None,
    metadata: Optional[Dict] = None,
    cloud_id: Optional[str] = None,
    api_key: Optional[str] = None,
    es_url: Optional[str] = None,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
    embedding_model: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict:
    """
    Work with Elasticsearch memories - create, search, retrieve, list, and manage memory records.

    This tool helps agents store and access memories using Elasticsearch with semantic search
    capabilities, allowing them to remember important information across conversations.

    Key Capabilities:
    - Store new memories with automatic embedding generation
    - Search for memories using semantic similarity
    - Browse and list all stored memories
    - Retrieve specific memories by ID
    - Delete unwanted memories

    Supported Actions:
    -----------------
    Memory Management:
    - record: Store a new memory with semantic embeddings
      Use this when you need to save information for later semantic recall.

    - retrieve: Find relevant memories using semantic search
      Use this when searching for information related to a topic or concept.
      This performs vector similarity search for the most relevant matches.

    - list: Browse all stored memories with pagination
      Use this to see all available memories without filtering.

    - get: Fetch a specific memory by ID
      Use this when you already know the exact memory ID.

    - delete: Remove a specific memory
      Use this to delete memories that are no longer needed.

    Args:
        action: The memory operation to perform (one of: "record", "retrieve", "list", "get", "delete")
        content: For record action: Text content to store as a memory
        query: Search terms for semantic search (required for retrieve action)
        memory_id: ID of a specific memory (required for get and delete actions)
        max_results: Maximum number of results to return (optional, default: 10)
        next_token: Pagination token for list action (optional)
        metadata: Additional metadata to store with the memory (optional)
        cloud_id: Elasticsearch Cloud ID for connection (optional if es_url provided)
        api_key: Elasticsearch API key for authentication
        es_url: Elasticsearch URL for serverless connection (optional if cloud_id provided)
        index_name: Name of the Elasticsearch index (defaults to 'strands_memory')
        namespace: Namespace for memory operations (defaults to 'default')
        embedding_model: Amazon Bedrock model for embeddings (defaults to Titan)
        region: AWS region for Bedrock service (defaults to 'us-west-2')

    Returns:
        Dict: Response containing the requested memory information or operation status
    """
    try:
        # Get values from environment variables if not provided
        cloud_id = cloud_id or os.getenv("ELASTICSEARCH_CLOUD_ID")
        es_url = es_url or os.getenv("ELASTICSEARCH_URL")
        api_key = api_key or os.getenv("ELASTICSEARCH_API_KEY")

        # Validate required parameters
        if not api_key:
            return {"status": "error", "content": [{"text": "api_key is required"}]}
        if not cloud_id and not es_url:
            return {"status": "error", "content": [{"text": "Either cloud_id or es_url is required"}]}

        # Set defaults
        index_name = index_name or os.getenv("ELASTICSEARCH_INDEX_NAME", DEFAULT_INDEX_NAME)
        namespace = namespace or os.getenv("ELASTICSEARCH_NAMESPACE", "default")
        embedding_model = embedding_model or os.getenv("ELASTICSEARCH_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        region = region or os.getenv("AWS_REGION", "us-west-2")
        max_results = max_results or DEFAULT_MAX_RESULTS

        # Initialize Elasticsearch client
        try:
            if es_url:
                # Use URL-based connection (for serverless)
                es_client = Elasticsearch(
                    hosts=[es_url],
                    api_key=api_key,
                    request_timeout=30,
                    retry_on_timeout=True,
                    max_retries=3,
                )
            else:
                # Use cloud_id connection
                es_client = Elasticsearch(
                    cloud_id=cloud_id,
                    api_key=api_key,
                    request_timeout=30,
                    retry_on_timeout=True,
                    max_retries=3,
                )

            # Test connection
            if not es_client.ping():
                return {"status": "error", "content": [{"text": "Unable to connect to Elasticsearch cluster"}]}

        except Exception as e:
            return {"status": "error", "content": [{"text": f"Failed to initialize Elasticsearch client: {str(e)}"}]}

        # Initialize Amazon Bedrock client for embeddings
        try:
            bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Failed to initialize Bedrock client: {str(e)}"}]}

        # Ensure index exists with proper mappings
        _ensure_index_exists(es_client, index_name, es_url)

        # Validate action
        try:
            action_enum = MemoryAction(action)
        except ValueError:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"Action '{action}' is not supported. "
                        f"Supported actions: {', '.join([a.value for a in MemoryAction])}"
                    }
                ],
            }

        # Validate required parameters
        param_values = {
            "content": content,
            "query": query,
            "memory_id": memory_id,
        }

        missing_params = [param for param in REQUIRED_PARAMS[action_enum] if param_values.get(param) is None]

        if missing_params:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"The following parameters are required for {action_enum.value} action: "
                            f"{', '.join(missing_params)}"
                        )
                    }
                ],
            }

        # Execute the appropriate action
        try:
            if action_enum == MemoryAction.RECORD:
                response = _record_memory(
                    es_client, bedrock_runtime, index_name, namespace, embedding_model, content, metadata
                )
                return {
                    "status": "success",
                    "content": [{"text": f"Memory stored successfully: {json.dumps(response, default=str)}"}],
                }

            elif action_enum == MemoryAction.RETRIEVE:
                response = _retrieve_memories(
                    es_client, bedrock_runtime, index_name, namespace, embedding_model, query, max_results, next_token
                )
                return {
                    "status": "success",
                    "content": [{"text": f"Memories retrieved successfully: {json.dumps(response, default=str)}"}],
                }

            elif action_enum == MemoryAction.LIST:
                response = _list_memories(es_client, index_name, namespace, max_results, next_token)
                return {
                    "status": "success",
                    "content": [{"text": f"Memories listed successfully: {json.dumps(response, default=str)}"}],
                }

            elif action_enum == MemoryAction.GET:
                response = _get_memory(es_client, index_name, namespace, memory_id)
                return {
                    "status": "success",
                    "content": [{"text": f"Memory retrieved successfully: {json.dumps(response, default=str)}"}],
                }

            elif action_enum == MemoryAction.DELETE:
                response = _delete_memory(es_client, index_name, namespace, memory_id)
                return {
                    "status": "success",
                    "content": [{"text": f"Memory deleted successfully: {memory_id}"}],
                }

        except Exception as e:
            error_msg = f"API error: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "content": [{"text": error_msg}]}

    except Exception as e:
        logger.error(f"Unexpected error in elasticsearch_memory tool: {str(e)}")
        return {"status": "error", "content": [{"text": str(e)}]}
