"""
Tool for managing memories using MongoDB Atlas with semantic search capabilities.

This module provides comprehensive memory management capabilities using
MongoDB Atlas as the backend with vector embeddings for semantic search.

Key Features:
------------
1. Memory Management:
   • record: Store new memories with automatic embedding generation
   • retrieve: Semantic search using vector embeddings and MongoDB Atlas Vector Search
   • list: List all memories with pagination support
   • get: Retrieve specific memories by memory ID
   • delete: Remove specific memories by memory ID

2. Semantic Search:
   • Automatic embedding generation using Amazon Bedrock Titan
   • Vector similarity search with cosine similarity
   • MongoDB Atlas Vector Search with $vectorSearch aggregation
   • Namespace-based filtering

3. Collection Management:
   • Automatic collection creation with proper structure
   • Vector search index configuration for semantic search
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
from strands_tools.mongodb_memory import mongodb_memory

# Create agent with direct tool usage
agent = Agent(tools=[mongodb_memory])

# Store a memory with semantic embeddings
mongodb_memory(
    action="record",
    content="User prefers vegetarian pizza with extra cheese",
    metadata={"category": "food_preferences", "type": "dietary"},
    cluster_uri="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="memories_db",
    collection_name="memories",
    namespace="user_123"
)

# Search memories using semantic similarity (vector search)
mongodb_memory(
    action="retrieve",
    query="food preferences and dietary restrictions",
    max_results=5,
    cluster_uri="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="memories_db",
    collection_name="memories",
    namespace="user_123"
)

# List all memories with pagination
mongodb_memory(
    action="list",
    max_results=10,
    cluster_uri="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="memories_db",
    collection_name="memories",
    namespace="user_123"
)

# Get specific memory by ID
mongodb_memory(
    action="get",
    memory_id="mem_1234567890_abcd1234",
    cluster_uri="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="memories_db",
    collection_name="memories",
    namespace="user_123"
)

# Delete a memory
mongodb_memory(
    action="delete",
    memory_id="mem_1234567890_abcd1234",
    cluster_uri="mongodb+srv://user:pass@cluster.mongodb.net/",
    database_name="memories_db",
    collection_name="memories",
    namespace="user_123"
)
```

Environment Variables:
---------------------
```bash
# Required
export MONGODB_ATLAS_CLUSTER_URI="mongodb+srv://user:pass@cluster.mongodb.net/"

# Optional
export MONGODB_DATABASE_NAME="custom_memories_db"        # Default: "strands_memory"
export MONGODB_COLLECTION_NAME="custom_memories"        # Default: "memories"
export MONGODB_NAMESPACE="custom_namespace"             # Default: "default"
export MONGODB_EMBEDDING_MODEL="amazon.titan-embed-text-v2:0"
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
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from strands import tool

# Set up logging
logger = logging.getLogger(__name__)


# Custom exceptions for better error handling
class MongoDBMemoryError(Exception):
    """Base exception for MongoDB memory operations."""

    pass


class MongoDBConnectionError(MongoDBMemoryError):
    """Raised when connection to MongoDB fails."""

    pass


class MongoDBMemoryNotFoundError(MongoDBMemoryError):
    """Raised when a memory record is not found."""

    pass


class MongoDBEmbeddingError(MongoDBMemoryError):
    """Raised when embedding generation fails."""

    pass


class MongoDBValidationError(MongoDBMemoryError):
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
DEFAULT_DATABASE_NAME = "strands_memory"
DEFAULT_COLLECTION_NAME = "memories"
DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_EMBEDDING_DIMS = 1024  # Titan v2 returns 1024 dimensions
DEFAULT_MAX_RESULTS = 10
DEFAULT_VECTOR_INDEX_NAME = "vector_index"


def _ensure_vector_search_index(collection, index_name: str = DEFAULT_VECTOR_INDEX_NAME):
    """Create vector search index if it doesn't exist."""
    try:
        # Check if index exists
        existing_indexes = list(collection.list_search_indexes())
        index_exists = any(idx.get("name") == index_name for idx in existing_indexes)

        if not index_exists:
            # Create vector search index with proper mappings
            index_definition = {
                "name": index_name,
                "definition": {
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": DEFAULT_EMBEDDING_DIMS,
                                "similarity": "cosine",
                            },
                            "namespace": {"type": "filter"},
                        },
                    }
                },
            }

            collection.create_search_index(index_definition)
            logger.info(f"Created vector search index: {index_name}")

            # Wait a moment for index to be ready
            import time

            time.sleep(2)

    except Exception as e:
        logger.warning(f"Could not create vector search index {index_name}: {str(e)}")
        logger.info("Vector search index should be created manually in MongoDB Atlas UI")
        # Don't raise exception - allow the tool to work without vector search


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
            raise MongoDBEmbeddingError(f"Invalid JSON response from Bedrock: {str(e)}") from e

        embedding = response_body["embedding"]

        # Validate embedding dimensions
        if len(embedding) != DEFAULT_EMBEDDING_DIMS:
            raise MongoDBEmbeddingError(f"Expected {DEFAULT_EMBEDDING_DIMS} dimensions, got {len(embedding)}")

        return embedding

    except MongoDBEmbeddingError:
        raise
    except Exception as e:
        raise MongoDBEmbeddingError(f"Embedding generation failed: {str(e)}") from e


def _generate_memory_id() -> str:
    """Generate a unique memory ID."""
    timestamp = int(time.time() * 1000)  # milliseconds
    unique_id = str(uuid.uuid4())[:8]
    return f"mem_{timestamp}_{unique_id}"


def _record_memory(
    collection,
    bedrock_runtime,
    namespace: str,
    embedding_model: str,
    content: str,
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Store a memory in MongoDB with embedding.

    Args:
        collection: MongoDB collection
        bedrock_runtime: Bedrock runtime client
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

    # Store in MongoDB
    result = collection.insert_one(doc)

    # Return filtered response with embedding metadata
    return {
        "memory_id": memory_id,
        "content": content,
        "namespace": namespace,
        "timestamp": doc["timestamp"],
        "result": "created" if result.inserted_id else "failed",
        "embedding_info": {"model": embedding_model, "dimensions": len(embedding), "generated": True},
    }


def _retrieve_memories(
    collection,
    bedrock_runtime,
    namespace: str,
    embedding_model: str,
    query: str,
    max_results: int,
    next_token: Optional[str] = None,
    index_name: str = DEFAULT_VECTOR_INDEX_NAME,
) -> Dict:
    """
    Retrieve memories using semantic search.

    Args:
        collection: MongoDB collection
        bedrock_runtime: Bedrock runtime client
        namespace: Memory namespace
        embedding_model: Embedding model ID
        query: Search query
        max_results: Maximum number of results
        next_token: Pagination token (skip count for MongoDB)
        index_name: Vector search index name

    Returns:
        Dict containing search results
    """
    # Generate embedding for query
    query_embedding = _generate_embedding(bedrock_runtime, query, embedding_model)

    # Calculate skip from next_token
    skip_count = int(next_token) if next_token else 0

    # Perform semantic search using MongoDB Atlas Vector Search
    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": max_results * 3,
                "limit": max_results,
                "filter": {"namespace": {"$eq": namespace}},
            }
        },
        {"$skip": skip_count},
        {"$limit": max_results},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"memory_id": 1, "content": 1, "timestamp": 1, "metadata": 1, "score": 1, "_id": 0}},
    ]

    results = list(collection.aggregate(pipeline))

    # Get total count for pagination
    total_pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 1000,  # Higher limit for count
                "limit": 1000,
                "filter": {"namespace": {"$eq": namespace}},
            }
        },
        {"$count": "total"},
    ]

    try:
        total_result = list(collection.aggregate(total_pipeline))
        total_count = total_result[0]["total"] if total_result and len(total_result) > 0 else len(results)
    except Exception:
        # Fallback to result count if aggregation fails
        total_count = len(results)

    # Format results
    memories = []
    max_score = 0
    for doc in results:
        memory = {
            "memory_id": doc["memory_id"],
            "content": doc["content"],
            "timestamp": doc["timestamp"],
            "metadata": doc.get("metadata", {}),
            "score": doc.get("score", 0),
        }
        memories.append(memory)
        max_score = max(max_score, doc.get("score", 0))

    result = {
        "memories": memories,
        "total": total_count,
        "max_score": max_score,
        "search_info": {
            "query_embedding_generated": True,
            "search_type": "MongoDB Atlas Vector Search",
            "embedding_model": embedding_model,
            "embedding_dimensions": DEFAULT_EMBEDDING_DIMS,
            "similarity_function": "cosine",
        },
    }

    # Add next_token if there are more results
    if skip_count + max_results < total_count:
        result["next_token"] = str(skip_count + max_results)

    return result


def _list_memories(collection, namespace: str, max_results: int, next_token: Optional[str] = None) -> Dict:
    """
    List all memories in the namespace.

    Args:
        collection: MongoDB collection
        namespace: Memory namespace
        max_results: Maximum number of results
        next_token: Pagination token

    Returns:
        Dict containing all memories
    """
    # Calculate skip from next_token
    skip_count = int(next_token) if next_token else 0

    # Query for memories in namespace
    cursor = (
        collection.find(
            {"namespace": namespace}, {"memory_id": 1, "content": 1, "timestamp": 1, "metadata": 1, "_id": 0}
        )
        .sort("timestamp", -1)
        .skip(skip_count)
        .limit(max_results)
    )

    memories = list(cursor)

    # Get total count
    total_count = collection.count_documents({"namespace": namespace})

    result = {"memories": memories, "total": total_count}

    # Add next_token if there are more results
    if skip_count + max_results < total_count:
        result["next_token"] = str(skip_count + max_results)

    return result


def _get_memory(collection, namespace: str, memory_id: str) -> Dict:
    """
    Get a specific memory by ID.

    Args:
        collection: MongoDB collection
        namespace: Memory namespace
        memory_id: Memory ID to retrieve

    Returns:
        Dict containing the memory

    Raises:
        Exception: If memory not found or not in correct namespace
    """
    try:
        doc = collection.find_one(
            {"memory_id": memory_id},
            {"memory_id": 1, "content": 1, "timestamp": 1, "metadata": 1, "namespace": 1, "_id": 0},
        )

        if not doc:
            raise MongoDBMemoryNotFoundError(f"Memory {memory_id} not found")

        # Verify namespace
        if doc.get("namespace") != namespace:
            raise MongoDBMemoryNotFoundError(f"Memory {memory_id} not found in namespace {namespace}")

        return {
            "memory_id": doc["memory_id"],
            "content": doc["content"],
            "timestamp": doc["timestamp"],
            "metadata": doc.get("metadata", {}),
            "namespace": doc["namespace"],
        }

    except MongoDBMemoryNotFoundError:
        raise
    except Exception as e:
        raise MongoDBMemoryError(f"Failed to get memory {memory_id}: {str(e)}") from e


def _delete_memory(collection, namespace: str, memory_id: str) -> Dict:
    """
    Delete a specific memory by ID.

    Args:
        collection: MongoDB collection
        namespace: Memory namespace
        memory_id: Memory ID to delete

    Returns:
        Dict containing deletion result

    Raises:
        Exception: If memory not found or deletion fails
    """
    try:
        # First verify the memory exists and is in correct namespace
        _get_memory(collection, namespace, memory_id)

        # Delete the memory
        result = collection.delete_one({"memory_id": memory_id, "namespace": namespace})

        if result.deleted_count == 0:
            raise MongoDBMemoryNotFoundError(f"Memory {memory_id} not found")

        return {"memory_id": memory_id, "result": "deleted"}

    except MongoDBMemoryNotFoundError:
        raise
    except Exception as e:
        raise MongoDBMemoryError(f"Failed to delete memory {memory_id}: {str(e)}") from e


@tool
def mongodb_memory(
    action: str,
    content: Optional[str] = None,
    query: Optional[str] = None,
    memory_id: Optional[str] = None,
    max_results: Optional[int] = None,
    next_token: Optional[str] = None,
    metadata: Optional[Dict] = None,
    cluster_uri: Optional[str] = None,
    database_name: Optional[str] = None,
    collection_name: Optional[str] = None,
    namespace: Optional[str] = None,
    embedding_model: Optional[str] = None,
    region: Optional[str] = None,
    vector_index_name: Optional[str] = None,
) -> Dict:
    """
    Work with MongoDB Atlas memories - create, search, retrieve, list, and manage memory records.

    This tool helps agents store and access memories using MongoDB Atlas with semantic search
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
        cluster_uri: MongoDB Atlas cluster URI for connection
        database_name: Name of the MongoDB database (defaults to 'strands_memory')
        collection_name: Name of the MongoDB collection (defaults to 'memories')
        namespace: Namespace for memory operations (defaults to 'default')
        embedding_model: Amazon Bedrock model for embeddings (defaults to Titan)
        region: AWS region for Bedrock service (defaults to 'us-west-2')
        vector_index_name: Name of the vector search index (defaults to 'vector_index')

    Returns:
        Dict: Response containing the requested memory information or operation status
    """
    try:
        # Get values from environment variables if not provided
        cluster_uri = cluster_uri or os.getenv("MONGODB_ATLAS_CLUSTER_URI")

        # Validate required parameters
        if not cluster_uri:
            return {"status": "error", "content": [{"text": "cluster_uri is required"}]}

        # Set defaults
        database_name = database_name or os.getenv("MONGODB_DATABASE_NAME", DEFAULT_DATABASE_NAME)
        collection_name = collection_name or os.getenv("MONGODB_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
        namespace = namespace or os.getenv("MONGODB_NAMESPACE", "default")
        embedding_model = embedding_model or os.getenv("MONGODB_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        region = region or os.getenv("AWS_REGION", "us-west-2")
        max_results = max_results or DEFAULT_MAX_RESULTS
        vector_index_name = vector_index_name or DEFAULT_VECTOR_INDEX_NAME

        # Initialize MongoDB client
        try:
            client = MongoClient(cluster_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            client.admin.command("ping")

            database = client[database_name]
            collection = database[collection_name]

        except ConnectionFailure as e:
            return {"status": "error", "content": [{"text": f"Unable to connect to MongoDB cluster: {str(e)}"}]}
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Failed to initialize MongoDB client: {str(e)}"}]}

        # Initialize Amazon Bedrock client for embeddings
        try:
            bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Failed to initialize Bedrock client: {str(e)}"}]}

        # Ensure vector search index exists for retrieve operations
        if action in [MemoryAction.RETRIEVE.value]:
            _ensure_vector_search_index(collection, vector_index_name)

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
                response = _record_memory(collection, bedrock_runtime, namespace, embedding_model, content, metadata)
                return {
                    "status": "success",
                    "content": [{"text": f"Memory stored successfully: {json.dumps(response, default=str)}"}],
                }

            elif action_enum == MemoryAction.RETRIEVE:
                response = _retrieve_memories(
                    collection,
                    bedrock_runtime,
                    namespace,
                    embedding_model,
                    query,
                    max_results,
                    next_token,
                    vector_index_name,
                )
                return {
                    "status": "success",
                    "content": [{"text": f"Memories retrieved successfully: {json.dumps(response, default=str)}"}],
                }

            elif action_enum == MemoryAction.LIST:
                response = _list_memories(collection, namespace, max_results, next_token)
                return {
                    "status": "success",
                    "content": [{"text": f"Memories listed successfully: {json.dumps(response, default=str)}"}],
                }

            elif action_enum == MemoryAction.GET:
                response = _get_memory(collection, namespace, memory_id)
                return {
                    "status": "success",
                    "content": [{"text": f"Memory retrieved successfully: {json.dumps(response, default=str)}"}],
                }

            elif action_enum == MemoryAction.DELETE:
                response = _delete_memory(collection, namespace, memory_id)
                return {
                    "status": "success",
                    "content": [{"text": f"Memory deleted successfully: {memory_id}"}],
                }

        except Exception as e:
            error_msg = f"API error: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "content": [{"text": error_msg}]}

    except Exception as e:
        logger.error(f"Unexpected error in mongodb_memory tool: {str(e)}")
        return {"status": "error", "content": [{"text": str(e)}]}
