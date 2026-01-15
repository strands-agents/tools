"""
S3 Vector Memory Tool for Strands Agents

AWS-native memory management using Amazon S3 Vectors with automatic user isolation.
Provides persistent memory storage and retrieval for Strands agents.

Usage:
    from strands import Agent
    from strands_tools.s3_vector_memory import s3_vector_memory

    agent = Agent(tools=[s3_vector_memory])

    # Store a memory
    agent.tool.s3_vector_memory(
        action="store",
        content="User prefers vegetarian food",
        user_id="user123"
    )

    # Retrieve relevant memories
    agent.tool.s3_vector_memory(
        action="retrieve",
        query="food preferences",
        user_id="user123",
        top_k=5
    )

    # List all memories
    agent.tool.s3_vector_memory(
        action="list",
        user_id="user123"
    )

Environment Variables:
    VECTOR_BUCKET_NAME: S3 Vector bucket name (default: multimodal-vector-store)
    VECTOR_INDEX_NAME: Vector index name (default: strands-multimodal)
    AWS_REGION: AWS region (default: us-east-1)
    EMBEDDING_MODEL: Bedrock embedding model (default: amazon.titan-embed-text-v2:0)

Requirements:
    - AWS credentials configured
    - Amazon S3 Vectors service access
    - Amazon Bedrock access for embeddings
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List

import boto3
from strands import tool


@tool
def s3_vector_memory(
    action: str,
    content: str = None,
    query: str = None,
    user_id: str = None,
    vector_bucket_name: str = None,
    index_name: str = None,
    top_k: int = 20,
    region_name: str = None,
    embedding_model: str = None,
    min_score: float = 0.1,
) -> Dict:
    """
    AWS-native memory management using Amazon S3 Vectors.

    Actions:
    - store: Store new memory content
    - retrieve: Search and retrieve relevant memories
    - list: List all user memories

    Args:
        action: Operation to perform (store/retrieve/list)
        content: Content to store (required for store action)
        query: Search query (required for retrieve action)
        user_id: User identifier for memory isolation (required)
        vector_bucket_name: S3 Vector bucket (env: VECTOR_BUCKET_NAME)
        index_name: Vector index name (env: VECTOR_INDEX_NAME)
        top_k: Maximum results to return (default: 20)
        region_name: AWS region (env: AWS_REGION, default: us-east-1)
        embedding_model: Bedrock embedding model (env: EMBEDDING_MODEL)
        min_score: Minimum similarity threshold (default: 0.1)

    Returns:
        Dict with operation results and status
    """

    # Validate required user_id for security
    if not user_id:
        return {"status": "error", "message": "user_id is required for memory isolation"}

    try:
        # Load configuration from environment or parameters
        config = {
            "bucket_name": vector_bucket_name or os.environ.get("VECTOR_BUCKET_NAME", "multimodal-vector-store"),
            "index_name": index_name or os.environ.get("VECTOR_INDEX_NAME", "strands-multimodal"),
            "region": region_name or os.environ.get("AWS_REGION", "us-east-1"),
            "model_id": embedding_model or os.environ.get("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"),
        }

        # Initialize AWS clients
        bedrock = boto3.client("bedrock-runtime", region_name=config["region"])
        s3vectors = boto3.client("s3vectors", region_name=config["region"])

        # Route to appropriate action
        if action == "store":
            return _store_memory(s3vectors, bedrock, config, content, user_id)
        elif action == "retrieve":
            return _retrieve_memories(s3vectors, bedrock, config, query, user_id, top_k, min_score)
        elif action == "list":
            return _list_memories(s3vectors, bedrock, config, user_id, top_k)
        else:
            return {"status": "error", "message": f"Invalid action: {action}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def _generate_embedding(bedrock, model_id: str, text: str) -> List[float]:
    """Generate text embedding using Amazon Bedrock."""
    # Truncate text if exceeds model limit
    if len(text) > 8000:
        text = text[:8000]

    response = bedrock.invoke_model(modelId=model_id, body=json.dumps({"inputText": text}))

    return json.loads(response["body"].read())["embedding"]


def _store_memory(s3vectors, bedrock, config: Dict, content: str, user_id: str) -> Dict:
    """Store memory with user isolation."""
    if not content:
        return {"status": "error", "message": "content is required for store action"}

    # Generate embedding
    embedding = _generate_embedding(bedrock, config["model_id"], content)

    # Create unique memory key with user prefix
    memory_key = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Prepare vector data with metadata
    vector_data = {
        "key": memory_key,
        "data": {"float32": [float(x) for x in embedding]},
        "metadata": {"user_id": user_id, "content": content, "timestamp": datetime.now().isoformat()},
    }

    # Store in S3 Vectors
    s3vectors.put_vectors(vectorBucketName=config["bucket_name"], indexName=config["index_name"], vectors=[vector_data])

    return {"status": "success", "message": "Memory stored successfully", "memory_key": memory_key}


def _retrieve_memories(
    s3vectors, bedrock, config: Dict, query: str, user_id: str, top_k: int, min_score: float
) -> Dict:
    """Retrieve relevant memories for user."""
    if not query:
        return {"status": "error", "message": "query is required for retrieve action"}

    # Generate query embedding
    query_embedding = _generate_embedding(bedrock, config["model_id"], query)

    # Search with user filter for isolation
    response = s3vectors.query_vectors(
        vectorBucketName=config["bucket_name"],
        indexName=config["index_name"],
        queryVector={"float32": query_embedding},
        topK=top_k,
        filter={"user_id": user_id},
        returnDistance=True,
        returnMetadata=True,
    )

    # Process results
    memories = []
    for vector in response.get("vectors", []):
        # Verify user isolation
        if vector["metadata"].get("user_id") != user_id:
            continue

        # Calculate similarity score
        similarity = 1.0 - vector.get("distance", 1.0)

        # Filter by minimum score
        if similarity >= min_score:
            memories.append(
                {
                    "id": vector["key"],
                    "memory": vector["metadata"].get("content", ""),
                    "similarity": round(similarity, 3),
                    "created_at": vector["metadata"].get("timestamp", ""),
                }
            )

    # Sort by similarity
    memories.sort(key=lambda x: x["similarity"], reverse=True)

    return {"status": "success", "memories": memories[:top_k], "total_found": len(memories), "query": query}


def _list_memories(s3vectors, bedrock, config: Dict, user_id: str, top_k: int) -> Dict:
    """List all user memories."""
    # Use generic embedding for listing
    generic_embedding = _generate_embedding(bedrock, config["model_id"], "user memories")

    # Query all user vectors
    response = s3vectors.query_vectors(
        vectorBucketName=config["bucket_name"],
        indexName=config["index_name"],
        queryVector={"float32": generic_embedding},
        topK=top_k,
        filter={"user_id": user_id},
        returnMetadata=True,
    )

    # Process memories
    memories = []
    for vector in response.get("vectors", []):
        # Verify user isolation
        if vector["metadata"].get("user_id") != user_id:
            continue

        memories.append(
            {
                "id": vector["key"],
                "memory": vector["metadata"].get("content", ""),
                "created_at": vector["metadata"].get("timestamp", ""),
            }
        )

    # Sort by creation time
    memories.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return {"status": "success", "memories": memories, "total_found": len(memories)}
