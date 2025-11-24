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

# Store a memory
result = mongodb_memory(
    action="record",
    content="User prefers vegetarian pizza with extra cheese",
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123"
)

# Search memories
result = mongodb_memory(
    action="retrieve",
    query="food preferences",
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123",
    max_results=5
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
import re
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import boto3
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.cursor import Cursor
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
DEFAULT_AWS_REGION = "us-west-2"
DEFAULT_NAMESPACE = "default"

# MongoDB projection constants
INCLUDE_FIELD = 1
EXCLUDE_FIELD = 0

# Response size limits to prevent "tool result too large" errors
MAX_RESPONSE_SIZE = 70000  # Maximum characters in response (70K total safety margin)
MAX_CONTENT_LENGTH = 12000  # Maximum content length per memory (12K per memory)
MAX_MEMORIES_IN_RESPONSE = 5  # Maximum memories to include in responses

# Index creation settings
INDEX_CREATION_TIMEOUT = 5  # seconds to wait for index creation


def _ensure_vector_search_index(collection: Collection, index_name: str = DEFAULT_VECTOR_INDEX_NAME) -> None:
    """
    Create vector search index if it doesn't exist.

    This function ensures that the required vector search index exists for semantic search operations.
    If the index doesn't exist, it creates one with the proper configuration for 1024-dimensional
    Titan embeddings using cosine similarity.

    Args:
        collection: MongoDB collection to create index on
        index_name: Name of the vector search index to create
    """
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
                            "namespace": {"type": "string"},
                        },
                    }
                },
            }

            collection.create_search_index(index_definition)
            logger.info(f"Created vector search index: {index_name}")
            logger.info("Index creation initiated - it may take a few minutes to become available")

    except Exception as e:
        logger.warning(f"Could not create vector search index {index_name}: {str(e)}")
        logger.info("Vector search index should be created manually in MongoDB Atlas UI")
        # Don't raise exception - allow the tool to work without vector search


def _generate_embedding(bedrock_runtime: Any, text: str, embedding_model: str) -> List[float]:
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

        # Extract embedding from Bedrock response
        # According to Amazon Bedrock Titan Embedding API documentation:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
        # The response contains an "embedding" field with the vector values
        embedding = response_body["embedding"]

        # Validate embedding dimensions
        if len(embedding) != DEFAULT_EMBEDDING_DIMS:
            raise MongoDBEmbeddingError(f"Expected {DEFAULT_EMBEDDING_DIMS} dimensions, got {len(embedding)}")

        return embedding

    except MongoDBEmbeddingError:
        raise
    except Exception as e:
        raise MongoDBEmbeddingError(f"Embedding generation failed: {str(e)}") from e


def _truncate_content(content: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """Truncate content to prevent large responses."""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def _optimize_response_size(response: Dict, action: str) -> Dict:
    """Optimize response size to prevent 'tool result too large' errors."""

    # For list and retrieve operations, limit the number of memories and truncate content
    if action in ["list", "retrieve"] and "memories" in response:
        memories = response["memories"]

        # Limit number of memories in response
        if len(memories) > MAX_MEMORIES_IN_RESPONSE:
            memories = memories[:MAX_MEMORIES_IN_RESPONSE]
            response["memories"] = memories
            response["truncated"] = True
            response["showing"] = len(memories)

        # Truncate content in each memory
        for memory in memories:
            if "content" in memory:
                memory["content"] = _truncate_content(memory["content"])

        # Remove verbose search_info for retrieve operations to save space
        if action == "retrieve" and "search_info" in response:
            response["search_info"] = {"type": "vector_search", "model": "titan-v2"}

    return response


def _validate_response_size(response_text: str) -> str:
    """Validate and truncate response if it exceeds size limits."""
    if len(response_text) <= MAX_RESPONSE_SIZE:
        return response_text

    # If response is too large, truncate and add warning
    truncated = response_text[: MAX_RESPONSE_SIZE - 100]  # Leave room for warning
    return f"{truncated}... [Response truncated due to size limit]"


def _validate_namespace(namespace: Any) -> str:
    """
    Validate and sanitize namespace parameter to prevent NoSQL injection.

    This function treats namespace as a trusted identifier by requiring it to be
    a simple string matching the pattern ^[A-Za-z0-9_-]{1,64}$ before including
    it in MongoDB queries. This prevents MongoDB operator injection attacks where
    dictionaries like {"$ne": ""} could be used to bypass tenant isolation.

    Args:
        namespace: The namespace value to validate (can be any type)

    Returns:
        str: A validated string namespace (1-64 chars, alphanumeric + underscore + hyphen only)

    Raises:
        MongoDBValidationError: If namespace cannot be converted to a safe string
    """
    if namespace is None:
        return DEFAULT_NAMESPACE

    if not isinstance(namespace, str):
        raise MongoDBValidationError(f"Namespace must be a string, got {type(namespace).__name__}. ")

    clean_namespace = str(namespace).strip()

    if not clean_namespace:
        raise MongoDBValidationError("Invalid namespace: Namespace cannot be empty.")

    if len(clean_namespace) > 64:
        raise MongoDBValidationError("Invalid namespace: Namespace too long. Maximum 64 characters allowed.")

    if not re.match(r"^[A-Za-z0-9_-]{1,64}$", clean_namespace):
        raise MongoDBValidationError(
            f"Invalid namespace: Namespace '{clean_namespace}' contains invalid characters. "
            "Must match pattern ^[A-Za-z0-9_-]{1,64}$"
        )

    return clean_namespace


def _mask_connection_string(connection_string: str) -> str:
    """
    Mask sensitive information in MongoDB connection string for logging/error messages.

    This function helps prevent credential exposure in logs and error messages by
    masking the username and password portions of MongoDB connection strings.

    Args:
        connection_string: MongoDB connection string that may contain credentials

    Returns:
        Masked connection string safe for logging
    """
    if not connection_string:
        return "[EMPTY]"

    try:
        # Pattern to match mongodb+srv://username:password@host/...
        pattern = r"mongodb\+srv://([^:]+):([^@]+)@(.+)"
        match = re.match(pattern, connection_string)

        if match:
            username, password, rest = match.groups()
            masked_username = username[:2] + "***" if len(username) > 2 else "***"
            return f"mongodb+srv://{masked_username}:***@{rest}"

        # Fallback for other patterns
        if "@" in connection_string:
            parts = connection_string.split("@")
            if len(parts) >= 2:
                return f"***@{parts[-1]}"

        return "***[MASKED_CONNECTION_STRING]***"
    except Exception:
        return "***[MASKED_CONNECTION_STRING]***"


def _validate_connection_string(cluster_uri: str) -> bool:
    """
    Validate MongoDB connection string format.

    Args:
        cluster_uri: MongoDB connection string to validate

    Returns:
        True if connection string appears valid, False otherwise
    """
    if not cluster_uri or not isinstance(cluster_uri, str):
        return False

    # Basic validation for MongoDB Atlas connection strings
    return (cluster_uri.startswith("mongodb+srv://") or cluster_uri.startswith("mongodb://")) and "@" in cluster_uri


def _generate_memory_id() -> str:
    """Generate a unique memory ID."""
    timestamp = int(time.time() * 1000)  # milliseconds
    unique_id = str(uuid.uuid4())[:8]
    return f"mem_{timestamp}_{unique_id}"


def _record_memory(
    collection: Collection,
    bedrock_runtime: Any,
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
    # Validate namespace to prevent NoSQL injection
    safe_namespace = _validate_namespace(namespace)

    # Generate unique memory ID
    memory_id = _generate_memory_id()

    # Generate embedding for semantic search
    embedding = _generate_embedding(bedrock_runtime, content, embedding_model)

    # Prepare document
    doc = {
        "memory_id": memory_id,
        "content": content,
        "embedding": embedding,
        "namespace": safe_namespace,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }

    # Store in MongoDB
    result = collection.insert_one(doc)

    # Return filtered response without embedding vectors (only metadata)
    return {
        "memory_id": memory_id,
        "content": content,
        "namespace": safe_namespace,
        "timestamp": doc["timestamp"],
        "result": "created" if result.inserted_id else "failed",
        "embedding_info": {"model": embedding_model, "dimensions": len(embedding), "generated": True},
    }


def _retrieve_memories(
    collection: Collection,
    bedrock_runtime: Any,
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
    # Validate namespace to prevent NoSQL injection
    safe_namespace = _validate_namespace(namespace)

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
                "numCandidates": max_results * 3,  # Use 3x candidates for better search quality
                "limit": max_results,
                "filter": {"namespace": {"$eq": safe_namespace}},
            }
        },
        {"$skip": skip_count},
        {"$limit": max_results},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        # MongoDB projection syntax: INCLUDE_FIELD = include field, EXCLUDE_FIELD = exclude field
        # We exclude _id (MongoDB's internal ObjectId) and embedding vectors, include only the fields we need
        {
            "$project": {
                "memory_id": INCLUDE_FIELD,
                "content": INCLUDE_FIELD,
                "timestamp": INCLUDE_FIELD,
                "metadata": INCLUDE_FIELD,
                "score": INCLUDE_FIELD,
                "_id": EXCLUDE_FIELD,
            }
        },
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
                "filter": {"namespace": {"$eq": safe_namespace}},
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


def _list_memories(collection: Collection, namespace: str, max_results: int, next_token: Optional[str] = None) -> Dict:
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
    # Validate namespace to prevent NoSQL injection
    safe_namespace = _validate_namespace(namespace)

    # Calculate skip from next_token
    skip_count = int(next_token) if next_token else 0

    # Query for memories in namespace
    # MongoDB projection syntax: INCLUDE_FIELD = include field, EXCLUDE_FIELD = exclude field
    # We exclude _id (MongoDB's internal ObjectId) and embedding vectors, include only the fields we need
    cursor: Cursor = (
        collection.find(
            {"namespace": safe_namespace},
            {
                "memory_id": INCLUDE_FIELD,
                "content": INCLUDE_FIELD,
                "timestamp": INCLUDE_FIELD,
                "metadata": INCLUDE_FIELD,
                "_id": EXCLUDE_FIELD,
            },
        )
        .sort("timestamp", -1)
        .skip(skip_count)
        .limit(max_results)
    )

    memories = list(cursor)

    # Get total count
    total_count = collection.count_documents({"namespace": safe_namespace})

    result = {"memories": memories, "total": total_count}

    # Add next_token if there are more results
    if skip_count + max_results < total_count:
        result["next_token"] = str(skip_count + max_results)

    return result


def _get_memory(collection: Collection, namespace: str, memory_id: str) -> Dict:
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
    # Validate namespace to prevent NoSQL injection
    safe_namespace = _validate_namespace(namespace)

    try:
        # MongoDB projection syntax: INCLUDE_FIELD = include field, EXCLUDE_FIELD = exclude field
        # We exclude _id (MongoDB's internal ObjectId) and embedding vectors, include only the fields we need
        doc = collection.find_one(
            {"memory_id": memory_id, "namespace": safe_namespace},
            {
                "memory_id": INCLUDE_FIELD,
                "content": INCLUDE_FIELD,
                "timestamp": INCLUDE_FIELD,
                "metadata": INCLUDE_FIELD,
                "namespace": INCLUDE_FIELD,
                "_id": EXCLUDE_FIELD,
            },
        )

        if not doc:
            raise MongoDBMemoryNotFoundError(f"Memory {memory_id} not found in namespace {safe_namespace}")

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


def _delete_memory(collection: Collection, namespace: str, memory_id: str) -> Dict:
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
    # Validate namespace to prevent NoSQL injection
    safe_namespace = _validate_namespace(namespace)

    try:
        # First verify the memory exists and is in correct namespace
        _get_memory(collection, safe_namespace, memory_id)

        # Delete the memory
        result = collection.delete_one({"memory_id": memory_id, "namespace": safe_namespace})

        if result.deleted_count == 0:
            raise MongoDBMemoryNotFoundError(f"Memory {memory_id} not found")

        # Return minimal response to avoid size issues
        return {"memory_id": memory_id, "result": "deleted"}

    except MongoDBMemoryNotFoundError:
        raise
    except Exception as e:
        raise MongoDBMemoryError(f"Failed to delete memory {memory_id}: {str(e)}") from e


class MongoDBMemoryTool:
    """
    MongoDB Atlas Memory Tool with secure credential management.

    This class encapsulates MongoDB Atlas connection credentials and configuration,
    preventing agents from accessing sensitive information like passwords and connection strings.
    """

    def __init__(
        self,
        cluster_uri: Optional[str] = None,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        region: Optional[str] = None,
        vector_index_name: Optional[str] = None,
    ):
        """
        Initialize MongoDB Memory Tool with secure credential storage.

        Args:
            cluster_uri: MongoDB Atlas cluster URI (kept private from agents)
            database_name: Name of the MongoDB database
            collection_name: Name of the MongoDB collection
            embedding_model: Amazon Bedrock model for embeddings
            region: AWS region for Bedrock service
            vector_index_name: Name of the vector search index
        """
        # Private attributes - not accessible to agents
        self._cluster_uri = cluster_uri or os.getenv("MONGODB_ATLAS_CLUSTER_URI")
        self._database_name = database_name or os.getenv("MONGODB_DATABASE_NAME", DEFAULT_DATABASE_NAME)
        self._collection_name = collection_name or os.getenv("MONGODB_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
        self._embedding_model = embedding_model or os.getenv("MONGODB_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self._region = region or os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
        self._vector_index_name = vector_index_name or DEFAULT_VECTOR_INDEX_NAME

        # Validate credentials during initialization
        if not self._cluster_uri:
            raise MongoDBValidationError("cluster_uri is required for MongoDB Memory Tool initialization")

        if not _validate_connection_string(self._cluster_uri):
            raise MongoDBValidationError("Invalid MongoDB connection string format")

    @tool
    def mongodb_memory(
        self,
        action: str,
        content: Optional[str] = None,
        query: Optional[str] = None,
        memory_id: Optional[str] = None,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
        metadata: Optional[Dict] = None,
        namespace: Optional[str] = None,
    ) -> Dict:
        """
        Work with MongoDB Atlas memories - create, search, retrieve, list, and manage memory records.

        This tool helps agents store and access memories using MongoDB Atlas with semantic search
        capabilities, allowing them to remember important information across conversations.

        Note: Credentials are securely managed by the class and not exposed to agents.

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
        - retrieve: Find relevant memories using semantic search
        - list: Browse all stored memories with pagination
        - get: Fetch a specific memory by ID
        - delete: Remove a specific memory

        Args:
            action: The memory operation to perform (one of: "record", "retrieve", "list", "get", "delete")
            content: For record action: Text content to store as a memory
            query: Search terms for semantic search (required for retrieve action)
            memory_id: ID of a specific memory (required for get and delete actions)
            max_results: Maximum number of results to return (optional, default: 10)
            next_token: Pagination token for list action (optional)
            metadata: Additional metadata to store with the memory (optional)
            namespace: Namespace for memory operations (defaults to 'default')

        Returns:
            Dict: Response containing the requested memory information or operation status
        """
        try:
            # Use private configuration (credentials not exposed to agents)
            if namespace is None:
                namespace = os.getenv("MONGODB_NAMESPACE", DEFAULT_NAMESPACE)
            max_results = max_results or DEFAULT_MAX_RESULTS

            try:
                safe_namespace = _validate_namespace(namespace)
            except MongoDBValidationError as e:
                return {
                    "status": "error",
                    "content": [{"text": f"Invalid namespace: {str(e)}"}],
                }

            # Initialize MongoDB client with secure error handling
            try:
                client = MongoClient(self._cluster_uri, serverSelectionTimeoutMS=5000)
                # Test connection
                client.admin.command("ping")

                database = client[self._database_name]
                collection = database[self._collection_name]

            except ConnectionFailure as e:
                # Use masked connection string in error messages for security
                masked_uri = _mask_connection_string(self._cluster_uri)
                logger.error(f"MongoDB connection failed for {masked_uri}: {str(e)}")
                return {
                    "status": "error",
                    "content": [{"text": f"Unable to connect to MongoDB cluster at {masked_uri}"}],
                }
            except Exception as e:
                # Use masked connection string in error messages for security
                masked_uri = _mask_connection_string(self._cluster_uri)
                logger.error(f"MongoDB client initialization failed for {masked_uri}: {str(e)}")
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to initialize MongoDB client for {masked_uri}"}],
                }

            # Initialize Amazon Bedrock client for embeddings
            try:
                bedrock_runtime = boto3.client("bedrock-runtime", region_name=self._region)
            except Exception as e:
                return {"status": "error", "content": [{"text": f"Failed to initialize Bedrock client: {str(e)}"}]}

            # Ensure vector search index exists for retrieve operations
            if action in [MemoryAction.RETRIEVE.value]:
                _ensure_vector_search_index(collection, self._vector_index_name)

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
                        collection, bedrock_runtime, safe_namespace, self._embedding_model, content, metadata
                    )
                    return {
                        "status": "success",
                        "content": [{"text": "Memory stored successfully"}, {"json": response}],
                    }

                elif action_enum == MemoryAction.RETRIEVE:
                    response = _retrieve_memories(
                        collection,
                        bedrock_runtime,
                        safe_namespace,
                        self._embedding_model,
                        query,
                        max_results,
                        next_token,
                        self._vector_index_name,
                    )
                    # Optimize response size for retrieve operations
                    optimized_response = _optimize_response_size(response, "retrieve")
                    return {
                        "status": "success",
                        "content": [{"text": "Memories retrieved successfully"}, {"json": optimized_response}],
                    }

                elif action_enum == MemoryAction.LIST:
                    response = _list_memories(collection, safe_namespace, max_results, next_token)
                    # Optimize response size for list operations
                    optimized_response = _optimize_response_size(response, "list")
                    return {
                        "status": "success",
                        "content": [{"text": "Memories listed successfully"}, {"json": optimized_response}],
                    }

                elif action_enum == MemoryAction.GET:
                    response = _get_memory(collection, safe_namespace, memory_id)
                    return {
                        "status": "success",
                        "content": [{"text": "Memory retrieved successfully"}, {"json": response}],
                    }

                elif action_enum == MemoryAction.DELETE:
                    response = _delete_memory(collection, safe_namespace, memory_id)
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
    - retrieve: Find relevant memories using semantic search
    - list: List all memories with pagination support
    - get: Retrieve specific memories by memory ID
    - delete: Remove specific memories by memory ID

    Args:
        action: The memory operation to perform (one of: "record", "retrieve", "list", "get", "delete")
        content: For record action: Text content to store as a memory
        query: Search terms for semantic search (required for retrieve action)
        memory_id: ID of a specific memory (required for get and delete actions)
        max_results: Maximum number of results to return (optional, default: 10)
        next_token: Pagination token for list action (optional)
        metadata: Additional metadata to store with the memory (optional)
        cluster_uri: MongoDB Atlas cluster URI (optional if set via environment)
        database_name: Name of the MongoDB database (optional, defaults to 'strands_memory')
        collection_name: Name of the MongoDB collection (optional, defaults to 'memories')
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
        database_name = database_name or os.getenv("MONGODB_DATABASE_NAME", DEFAULT_DATABASE_NAME)
        collection_name = collection_name or os.getenv("MONGODB_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)
        embedding_model = embedding_model or os.getenv("MONGODB_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        region = region or os.getenv("AWS_REGION", DEFAULT_AWS_REGION)
        vector_index_name = vector_index_name or DEFAULT_VECTOR_INDEX_NAME
        if namespace is None:
            namespace = os.getenv("MONGODB_NAMESPACE", DEFAULT_NAMESPACE)
        max_results = max_results or DEFAULT_MAX_RESULTS

        try:
            safe_namespace = _validate_namespace(namespace)
        except MongoDBValidationError as e:
            return {
                "status": "error",
                "content": [{"text": f"Invalid namespace: {str(e)}"}],
            }

        # Validate required parameters
        if not cluster_uri:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "cluster_uri is required for MongoDB Memory Tool. "
                            "Set MONGODB_ATLAS_CLUSTER_URI environment variable or provide cluster_uri parameter."
                        )
                    }
                ],
            }

        if not _validate_connection_string(cluster_uri):
            return {"status": "error", "content": [{"text": "Invalid MongoDB connection string format"}]}

        # Initialize MongoDB client with secure error handling
        try:
            client = MongoClient(cluster_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            client.admin.command("ping")

            database = client[database_name]
            collection = database[collection_name]

        except ConnectionFailure as e:
            # Use masked connection string in error messages for security
            masked_uri = _mask_connection_string(cluster_uri)
            logger.error(f"MongoDB connection failed for {masked_uri}: {str(e)}")
            return {"status": "error", "content": [{"text": f"Unable to connect to MongoDB cluster at {masked_uri}"}]}
        except Exception as e:
            # Use masked connection string in error messages for security
            masked_uri = _mask_connection_string(cluster_uri)
            logger.error(f"MongoDB client initialization failed for {masked_uri}: {str(e)}")
            return {"status": "error", "content": [{"text": f"Failed to initialize MongoDB client for {masked_uri}"}]}

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
                response = _record_memory(
                    collection, bedrock_runtime, safe_namespace, embedding_model, content, metadata
                )
                return {
                    "status": "success",
                    "content": [{"text": "Memory stored successfully"}, {"json": response}],
                }

            elif action_enum == MemoryAction.RETRIEVE:
                response = _retrieve_memories(
                    collection,
                    bedrock_runtime,
                    safe_namespace,
                    embedding_model,
                    query,
                    max_results,
                    next_token,
                    vector_index_name,
                )
                # Optimize response size for retrieve operations
                optimized_response = _optimize_response_size(response, "retrieve")
                return {
                    "status": "success",
                    "content": [{"text": "Memories retrieved successfully"}, {"json": optimized_response}],
                }

            elif action_enum == MemoryAction.LIST:
                response = _list_memories(collection, safe_namespace, max_results, next_token)
                # Optimize response size for list operations
                optimized_response = _optimize_response_size(response, "list")

                # For empty results, return minimal response to prevent size issues
                if not optimized_response.get("memories"):
                    return {
                        "status": "success",
                        "content": [{"text": "No memories found"}],
                    }

                return {
                    "status": "success",
                    "content": [{"text": "Memories listed successfully"}, {"json": optimized_response}],
                }

            elif action_enum == MemoryAction.GET:
                response = _get_memory(collection, safe_namespace, memory_id)
                return {
                    "status": "success",
                    "content": [{"text": "Memory retrieved successfully"}, {"json": response}],
                }

            elif action_enum == MemoryAction.DELETE:
                response = _delete_memory(collection, safe_namespace, memory_id)
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
