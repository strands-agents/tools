"""
Tool for managing memories using Elasticsearch with semantic search capabilities.

This module provides comprehensive memory management capabilities using
Elasticsearch as the backend with vector embeddings for semantic search.
It follows the same pattern as agent_core_memory.py.

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
from strands_tools.elasticsearch_memory import ElasticsearchMemoryToolProvider

# Method 1: Elasticsearch Cloud (traditional)
provider = ElasticsearchMemoryToolProvider(
    cloud_id="your-elasticsearch-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)

# Method 2: Elasticsearch Serverless (URL-based)
provider = ElasticsearchMemoryToolProvider(
    es_url="https://your-serverless-cluster.es.region.aws.elastic.cloud:443",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)

# Method 3: Using environment variables
# Set: ELASTICSEARCH_CLOUD_ID or ELASTICSEARCH_URL
# Set: ELASTICSEARCH_API_KEY, ELASTICSEARCH_INDEX_NAME, etc.
provider = ElasticsearchMemoryToolProvider()

agent = Agent(tools=provider.tools)

# Store a memory with semantic embeddings
agent.tool.elasticsearch_memory(
    action="record",
    content="User prefers vegetarian pizza with extra cheese",
    metadata={"category": "food_preferences", "type": "dietary"}
)

# Search memories using semantic similarity (vector search)
agent.tool.elasticsearch_memory(
    action="retrieve",
    query="food preferences and dietary restrictions",
    max_results=5
)

# List all memories with pagination
agent.tool.elasticsearch_memory(
    action="list",
    max_results=10
)

# Get specific memory by ID
agent.tool.elasticsearch_memory(
    action="get",
    memory_id="mem_1234567890_abcd1234"
)

# Delete a memory
agent.tool.elasticsearch_memory(
    action="delete",
    memory_id="mem_1234567890_abcd1234"
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
from typing import Dict, List, Optional, Any

import boto3
from elasticsearch import Elasticsearch
from strands import tool
from strands.types.tools import AgentTool

# Set up logging
logger = logging.getLogger(__name__)

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


class ElasticsearchMemoryToolProvider:
    """Provider for Elasticsearch Memory Service tools."""

    def __init__(
        self,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        es_url: Optional[str] = None,
        index_name: Optional[str] = None,
        namespace: Optional[str] = None,
        embedding_model: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """
        Initialize the Elasticsearch Memory tool provider.

        Args:
            cloud_id: Elasticsearch Cloud ID for connection (optional if es_url provided)
            api_key: Elasticsearch API key for authentication
            es_url: Elasticsearch URL for serverless connection (optional if cloud_id provided)
            index_name: Name of the Elasticsearch index (defaults to 'strands_memory')
            namespace: Namespace for memory operations (defaults to 'default')
            embedding_model: Amazon Bedrock model for embeddings (defaults to Titan)
            region: AWS region for Bedrock service (defaults to 'us-west-2')

        Raises:
            ValueError: If required parameters are missing or invalid
            ConnectionError: If unable to connect to Elasticsearch
        """
        # Get values from environment variables if not provided
        self.cloud_id = cloud_id or os.getenv("ELASTICSEARCH_CLOUD_ID")
        self.es_url = es_url or os.getenv("ELASTICSEARCH_URL")
        self.api_key = api_key or os.getenv("ELASTICSEARCH_API_KEY")
        
        # Validate required parameters
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.cloud_id and not self.es_url:
            raise ValueError("Either cloud_id or es_url is required")
        self.index_name = index_name or os.getenv("ELASTICSEARCH_INDEX_NAME", DEFAULT_INDEX_NAME)
        self.namespace = namespace or os.getenv("ELASTICSEARCH_NAMESPACE", "default")
        self.embedding_model = embedding_model or os.getenv("ELASTICSEARCH_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.region = region or os.getenv("AWS_REGION", "us-west-2")

        # Initialize Elasticsearch client
        try:
            if self.es_url:
                # Use URL-based connection (for serverless)
                self.es_client = Elasticsearch(
                    hosts=[self.es_url],
                    api_key=self.api_key,
                    request_timeout=30,
                    retry_on_timeout=True,
                    max_retries=3
                )
            else:
                # Use cloud_id connection
                self.es_client = Elasticsearch(
                    cloud_id=self.cloud_id,
                    api_key=self.api_key,
                    request_timeout=30,
                    retry_on_timeout=True,
                    max_retries=3
                )
            
            # Test connection
            if not self.es_client.ping():
                raise ConnectionError("Unable to connect to Elasticsearch cluster")
                
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Elasticsearch client: {str(e)}")

        # Initialize Amazon Bedrock client for embeddings
        try:
            self.bedrock_runtime = boto3.client(
                'bedrock-runtime',
                region_name=self.region
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Bedrock client: {str(e)}")

        # Ensure index exists with proper mappings
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Create index with proper mappings if it doesn't exist."""
        try:
            if not self.es_client.indices.exists(index=self.index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "content": {
                                "type": "text",
                                "analyzer": "standard"
                            },
                            "embedding": {
                                "type": "dense_vector",
                                "dims": DEFAULT_EMBEDDING_DIMS,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "namespace": {
                                "type": "keyword"
                            },
                            "memory_id": {
                                "type": "keyword"
                            },
                            "timestamp": {
                                "type": "date"
                            },
                            "metadata": {
                                "type": "object",
                                "enabled": True
                            }
                        }
                    }
                }
                
                # Add settings only if not using serverless (URL-based connection)
                if not self.es_url:
                    mapping["settings"] = {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "index.knn": True
                    }
                
                self.es_client.indices.create(index=self.index_name, body=mapping)
                logger.info(f"Created Elasticsearch index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Failed to create index {self.index_name}: {str(e)}")
            raise

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for text using Amazon Bedrock Titan.

        This method generates 1024-dimensional vector embeddings using Amazon Bedrock's
        Titan embedding model. These embeddings are used for semantic similarity search.

        Args:
            text: Text to generate embeddings for

        Returns:
            List of 1024 float values representing the text embedding

        Raises:
            Exception: If embedding generation fails
        """
        try:
            logger.debug(f"Generating embedding for text: {text[:100]}...")
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.embedding_model,
                body=json.dumps({"inputText": text})
            )
            
            response_body = json.loads(response['body'].read())
            embedding = response_body['embedding']
            
            # Validate embedding dimensions
            if len(embedding) != DEFAULT_EMBEDDING_DIMS:
                raise Exception(f"Expected {DEFAULT_EMBEDDING_DIMS} dimensions, got {len(embedding)}")
            
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text '{text[:50]}...': {str(e)}")
            raise Exception(f"Embedding generation failed: {str(e)}")

    def _generate_memory_id(self) -> str:
        """Generate a unique memory ID."""
        timestamp = int(time.time() * 1000)  # milliseconds
        unique_id = str(uuid.uuid4())[:8]
        return f"mem_{timestamp}_{unique_id}"

    @property
    def tools(self) -> list[AgentTool]:
        """Extract all @tool decorated methods from this instance."""
        tools = []
        for attr_name in dir(self):
            if attr_name == "tools":
                continue
            attr = getattr(self, attr_name)
            if isinstance(attr, AgentTool):
                tools.append(attr)
        return tools

    @tool
    def elasticsearch_memory(
        self,
        action: str,
        content: Optional[str] = None,
        query: Optional[str] = None,
        memory_id: Optional[str] = None,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
        metadata: Optional[Dict] = None,
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

        Returns:
            Dict: Response containing the requested memory information or operation status
        """
        try:
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

            missing_params = [
                param for param in REQUIRED_PARAMS[action_enum] 
                if not param_values.get(param)
            ]

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

            # Set defaults
            max_results = max_results or DEFAULT_MAX_RESULTS

            # Execute the appropriate action
            try:
                if action_enum == MemoryAction.RECORD:
                    response = self._record_memory(content, metadata)
                    return {
                        "status": "success",
                        "content": [{"text": f"Memory stored successfully: {json.dumps(response, default=str)}"}],
                    }

                elif action_enum == MemoryAction.RETRIEVE:
                    response = self._retrieve_memories(query, max_results, next_token)
                    return {
                        "status": "success",
                        "content": [{"text": f"Memories retrieved successfully: {json.dumps(response, default=str)}"}],
                    }

                elif action_enum == MemoryAction.LIST:
                    response = self._list_memories(max_results, next_token)
                    return {
                        "status": "success",
                        "content": [{"text": f"Memories listed successfully: {json.dumps(response, default=str)}"}],
                    }

                elif action_enum == MemoryAction.GET:
                    response = self._get_memory(memory_id)
                    return {
                        "status": "success",
                        "content": [{"text": f"Memory retrieved successfully: {json.dumps(response, default=str)}"}],
                    }

                elif action_enum == MemoryAction.DELETE:
                    response = self._delete_memory(memory_id)
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

    def _record_memory(self, content: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Store a memory in Elasticsearch with embedding.

        Args:
            content: Text content to store
            metadata: Optional metadata dictionary

        Returns:
            Dict containing the stored memory information
        """
        # Generate unique memory ID
        memory_id = self._generate_memory_id()
        
        # Generate embedding for semantic search
        embedding = self._generate_embedding(content)
        
        # Prepare document
        doc = {
            "memory_id": memory_id,
            "content": content,
            "embedding": embedding,
            "namespace": self.namespace,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        
        # Store in Elasticsearch
        response = self.es_client.index(
            index=self.index_name,
            id=memory_id,
            body=doc
        )
        
        # Return filtered response with embedding metadata
        return {
            "memory_id": memory_id,
            "content": content,
            "namespace": self.namespace,
            "timestamp": doc["timestamp"],
            "result": response["result"],
            "embedding_info": {
                "model": self.embedding_model,
                "dimensions": len(embedding),
                "generated": True
            }
        }

    def _retrieve_memories(self, query: str, max_results: int, next_token: Optional[str] = None) -> Dict:
        """
        Retrieve memories using semantic search.

        Args:
            query: Search query
            max_results: Maximum number of results
            next_token: Pagination token (from parameter for Elasticsearch)

        Returns:
            Dict containing search results
        """
        # Generate embedding for query
        query_embedding = self._generate_embedding(query)
        
        # Calculate offset from next_token
        from_offset = int(next_token) if next_token else 0
        
        # Perform semantic search using k-NN
        search_body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": max_results,
                "num_candidates": max_results * 3,
                "filter": {
                    "term": {"namespace": self.namespace}
                }
            },
            "from": from_offset,
            "size": max_results,
            "_source": ["memory_id", "content", "timestamp", "metadata"]
        }
        
        response = self.es_client.search(index=self.index_name, body=search_body)
        
        # Format results
        memories = []
        for hit in response["hits"]["hits"]:
            memory = {
                "memory_id": hit["_source"]["memory_id"],
                "content": hit["_source"]["content"],
                "timestamp": hit["_source"]["timestamp"],
                "metadata": hit["_source"].get("metadata", {}),
                "score": hit["_score"]
            }
            memories.append(memory)
        
        result = {
            "memories": memories,
            "total": response["hits"]["total"]["value"],
            "max_score": response["hits"]["max_score"],
            "search_info": {
                "query_embedding_generated": True,
                "search_type": "k-NN vector similarity",
                "embedding_model": self.embedding_model,
                "embedding_dimensions": DEFAULT_EMBEDDING_DIMS,
                "similarity_function": "cosine"
            }
        }
        
        # Add next_token if there are more results
        if from_offset + max_results < response["hits"]["total"]["value"]:
            result["next_token"] = str(from_offset + max_results)
            
        return result

    def _list_memories(self, max_results: int, next_token: Optional[str] = None) -> Dict:
        """
        List all memories in the namespace.

        Args:
            max_results: Maximum number of results
            next_token: Pagination token

        Returns:
            Dict containing all memories
        """
        # Calculate offset from next_token
        from_offset = int(next_token) if next_token else 0
        
        search_body = {
            "query": {
                "term": {"namespace": self.namespace}
            },
            "sort": [
                {"timestamp": {"order": "desc"}}
            ],
            "from": from_offset,
            "size": max_results,
            "_source": ["memory_id", "content", "timestamp", "metadata"]
        }
        
        response = self.es_client.search(index=self.index_name, body=search_body)
        
        # Format results
        memories = []
        for hit in response["hits"]["hits"]:
            memory = {
                "memory_id": hit["_source"]["memory_id"],
                "content": hit["_source"]["content"],
                "timestamp": hit["_source"]["timestamp"],
                "metadata": hit["_source"].get("metadata", {})
            }
            memories.append(memory)
        
        result = {
            "memories": memories,
            "total": response["hits"]["total"]["value"]
        }
        
        # Add next_token if there are more results
        if from_offset + max_results < response["hits"]["total"]["value"]:
            result["next_token"] = str(from_offset + max_results)
            
        return result

    def _get_memory(self, memory_id: str) -> Dict:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            Dict containing the memory

        Raises:
            Exception: If memory not found or not in correct namespace
        """
        try:
            response = self.es_client.get(index=self.index_name, id=memory_id)
            source = response["_source"]
            
            # Verify namespace
            if source.get("namespace") != self.namespace:
                raise Exception(f"Memory {memory_id} not found in namespace {self.namespace}")
            
            return {
                "memory_id": source["memory_id"],
                "content": source["content"],
                "timestamp": source["timestamp"],
                "metadata": source.get("metadata", {}),
                "namespace": source["namespace"]
            }
            
        except Exception as e:
            # Handle Elasticsearch NotFoundError
            if hasattr(e, 'status_code') and e.status_code == 404:
                raise Exception(f"Memory {memory_id} not found")
            elif "not_found" in str(e).lower() or "NotFoundError" in str(type(e)):
                raise Exception(f"Memory {memory_id} not found")
            raise

    def _delete_memory(self, memory_id: str) -> Dict:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: Memory ID to delete

        Returns:
            Dict containing deletion result

        Raises:
            Exception: If memory not found or deletion fails
        """
        try:
            # First verify the memory exists and is in correct namespace
            self._get_memory(memory_id)
            
            # Delete the memory
            response = self.es_client.delete(index=self.index_name, id=memory_id)
            
            return {
                "memory_id": memory_id,
                "result": response["result"]
            }
            
        except Exception as e:
            if "not_found" in str(e).lower():
                raise Exception(f"Memory {memory_id} not found")
            raise
