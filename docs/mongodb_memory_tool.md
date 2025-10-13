# MongoDB Atlas Memory Tool

The MongoDB Atlas Memory Tool provides comprehensive memory management capabilities using MongoDB Atlas as the backend with vector embeddings for semantic search. It uses the direct tool pattern where tools are imported and used directly with agents.

## Features

- **Semantic Search**: Automatic embedding generation using Amazon Bedrock Titan for vector similarity search
- **Memory Management**: Store, retrieve, list, get, and delete memory operations
- **Index Management**: Automatic vector search index creation with proper configuration
- **Namespace Support**: Organize memories by namespace for multi-user scenarios
- **Pagination**: Support for paginated results in list and retrieve operations
- **Error Handling**: Comprehensive error handling with clear error messages

## Installation

Install the required dependencies:

```bash
pip install strands-agents-tools[mongodb_memory]
```

This will install:
- `pymongo>=4.0.0,<5.0.0` - MongoDB Python client

## Prerequisites

1. **MongoDB Atlas**: You need a MongoDB Atlas cluster with:
   - Connection URI (mongodb+srv format)
   - Database user with read/write permissions
   - Vector Search enabled (Atlas Search)

2. **Amazon Bedrock**: Access to Amazon Bedrock for embedding generation:
   - AWS credentials configured
   - Access to `amazon.titan-embed-text-v2:0` model (or custom embedding model)

## Quick Start

### Basic Setup

```python
from strands import Agent
from strands_tools.mongodb_memory import mongodb_memory

# Create an agent with the direct tool
agent = Agent(tools=[mongodb_memory])

# Use the tool with configuration parameters
result = agent.tool.mongodb_memory(
    action="record",
    content="User prefers vegetarian pizza with extra cheese",
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123"
)
```

### Environment Variables

You can also use environment variables for configuration:

```bash
export MONGODB_ATLAS_CLUSTER_URI="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority"
export MONGODB_DATABASE_NAME="memory_db"
export MONGODB_COLLECTION_NAME="memories"
export MONGODB_NAMESPACE="user_123"
export MONGODB_EMBEDDING_MODEL="amazon.titan-embed-text-v2:0"
export AWS_REGION="us-west-2"
```

Then use the tool with minimal parameters (environment variables will be used):

```python
result = agent.tool.mongodb_memory(
    action="record",
    content="User prefers vegetarian pizza"
    # cluster_uri, database_name, etc. will be read from environment variables
)
```

## Usage Examples

### 1. Store Memories

```python
# Store a simple memory
result = agent.tool.mongodb_memory(
    action="record",
    content="User prefers vegetarian pizza with extra cheese and no onions",
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123"
)

# Store a memory with metadata
result = agent.tool.mongodb_memory(
    action="record",
    content="Meeting scheduled for next Tuesday at 2 PM with the development team",
    metadata={
        "category": "meetings",
        "priority": "high",
        "participants": ["dev_team"],
        "date": "2024-01-16"
    },
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123"
)
```

### 2. Semantic Search

```python
# Search for food-related memories
result = agent.tool.mongodb_memory(
    action="retrieve",
    query="food preferences and dietary restrictions",
    max_results=5,
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123"
)

# Search for meeting information
result = agent.tool.mongodb_memory(
    action="retrieve",
    query="upcoming meetings and appointments",
    max_results=10,
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123"
)
```

### 3. List All Memories

```python
# List recent memories
result = agent.tool.mongodb_memory(
    action="list",
    max_results=20,
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123"
)

# List with pagination
result = agent.tool.mongodb_memory(
    action="list",
    max_results=10,
    next_token="10",  # Start from the 11th result
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123"
)
```

### 4. Get Specific Memory

```python
# Retrieve a specific memory by ID
result = agent.tool.mongodb_memory(
    action="get",
    memory_id="mem_1704567890123_abc12345",
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123"
)
```

### 5. Delete Memory

```python
# Delete a specific memory
result = agent.tool.mongodb_memory(
    action="delete",
    memory_id="mem_1704567890123_abc12345",
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123"
)
```

## Advanced Configuration

### Using Configuration Dictionary

For cleaner code, you can use a configuration dictionary:

```python
config = {
    "cluster_uri": "mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    "database_name": "memory_db",
    "collection_name": "memories",
    "namespace": "user_123",
    "region": "us-east-1"
}

# Store memory
result = agent.tool.mongodb_memory(
    action="record",
    content="User prefers vegetarian pizza",
    **config
)

# Search memories
result = agent.tool.mongodb_memory(
    action="retrieve",
    query="food preferences",
    max_results=5,
    **config
)
```

### Custom Embedding Model

```python
result = agent.tool.mongodb_memory(
    action="record",
    content="User prefers vegetarian pizza",
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    database_name="memory_db",
    collection_name="memories",
    embedding_model="amazon.titan-embed-text-v1:0",  # Different model
    region="us-east-1"
)
```

### Multiple Namespaces

```python
# User-specific memories
result = agent.tool.mongodb_memory(
    action="record",
    content="Alice likes Italian food",
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_alice"
)

# System-wide memories
result = agent.tool.mongodb_memory(
    action="record",
    content="System maintenance scheduled",
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    database_name="memory_db",
    collection_name="memories",
    namespace="system_global"
)
```

## Response Format

All operations return a standardized response format:

```python
{
    "status": "success",  # or "error"
    "content": [
        {
            "text": "Memory stored successfully: {...}"
        }
    ]
}
```

### Successful Record Response

```json
{
    "status": "success",
    "content": [
        {
            "text": "Memory stored successfully: {\"memory_id\": \"mem_1704567890123_abc12345\", \"content\": \"User prefers vegetarian pizza\", \"namespace\": \"user_123\", \"timestamp\": \"2024-01-06T20:31:30.123456Z\", \"result\": \"created\"}"
        }
    ]
}
```

### Successful Retrieve Response

```json
{
    "status": "success",
    "content": [
        {
            "text": "Memories retrieved successfully: {\"memories\": [{\"memory_id\": \"mem_123\", \"content\": \"User prefers vegetarian pizza\", \"timestamp\": \"2024-01-06T20:31:30Z\", \"metadata\": {\"category\": \"food\"}, \"score\": 0.95}], \"total\": 1, \"max_score\": 0.95}"
        }
    ]
}
```

## Collection Structure

The tool automatically creates a MongoDB collection with documents structured as follows:

```json
{
    "_id": "ObjectId",
    "memory_id": "mem_1704567890123_abc12345",
    "content": "User prefers vegetarian pizza with extra cheese",
    "embedding": [0.1, 0.2, 0.3, ...],  // 1024-dimensional vector
    "namespace": "user_123",
    "timestamp": "2024-01-06T20:31:30.123456Z",
    "metadata": {
        "category": "food",
        "priority": "medium"
    }
}
```

### Vector Search Index

The tool automatically creates a vector search index with the following configuration:

```json
{
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 1024,
            "similarity": "cosine"
        },
        {
            "type": "filter",
            "path": "namespace"
        }
    ]
}
```

## Error Handling

The tool provides comprehensive error handling:

### Connection Errors

```python
# Invalid connection URI
result = agent.tool.mongodb_memory(
    action="record",
    content="test",
    cluster_uri="mongodb+srv://invalid:credentials@invalid.mongodb.net/"
)
# Returns: {"status": "error", "content": [{"text": "Unable to connect to MongoDB Atlas cluster"}]}
```

### Missing Parameters

```python
# Missing required content for record action
result = agent.tool.mongodb_memory(
    action="record",
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/"
)
# Returns: {"status": "error", "content": [{"text": "The following parameters are required for record action: content"}]}

# Missing connection parameters
result = agent.tool.mongodb_memory(action="record", content="test")
# Returns: {"status": "error", "content": [{"text": "cluster_uri is required"}]}
```

### Memory Not Found

```python
# Non-existent memory ID
result = agent.tool.mongodb_memory(
    action="get",
    memory_id="nonexistent",
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/"
)
# Returns: {"status": "error", "content": [{"text": "API error: Memory nonexistent not found"}]}
```

## Performance Considerations

### Embedding Generation

- Embeddings are generated using Amazon Bedrock Titan model
- Each record and retrieve operation requires embedding generation
- Consider caching strategies for frequently accessed queries

### Index Optimization

- The tool creates optimized vector search indices
- Uses cosine similarity for semantic matching
- Configures appropriate index settings for performance

### Pagination

- Use pagination for large result sets
- `max_results` parameter controls batch size
- `next_token` enables efficient pagination using skip/limit

## Best Practices

### 1. Configuration Management

Create reusable configuration objects:

```python
# Create a base configuration
base_config = {
    "cluster_uri": "mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    "database_name": "memory_db",
    "region": "us-east-1"
}

# User-specific configuration
def get_user_config(user_id):
    return {
        **base_config,
        "collection_name": "user_memories",
        "namespace": f"user_{user_id}"
    }

# Usage
user_config = get_user_config("alice")
result = agent.tool.mongodb_memory(
    action="record",
    content="Alice likes Italian food",
    **user_config
)
```

### 2. Namespace Organization

The namespace parameter is crucial for data isolation and multi-tenant memory management:

```python
# User-based namespaces
user_namespace = f"user_{user_id}"

# Session-based namespaces
session_namespace = f"session_{session_id}"

# Hierarchical namespaces
org_user_namespace = f"org_{org_id}_user_{user_id}"

# Feature-based namespaces
chat_namespace = "feature_chat"
task_namespace = "feature_tasks"
```

### 3. Metadata Usage

```python
# Use structured metadata for better organization
result = agent.tool.mongodb_memory(
    action="record",
    content="Important project deadline",
    metadata={
        "type": "deadline",
        "project": "project_alpha",
        "priority": "high",
        "due_date": "2024-02-01",
        "assigned_to": ["alice", "bob"]
    },
    **config
)
```

### 4. Error Handling

```python
def safe_memory_operation(agent, action, **kwargs):
    try:
        result = agent.tool.mongodb_memory(action=action, **kwargs)
        if result["status"] == "error":
            logger.error(f"Memory operation failed: {result['content'][0]['text']}")
            return None
        return result
    except Exception as e:
        logger.error(f"Unexpected error in memory operation: {e}")
        return None
```

### 5. Batch Operations

```python
# Store multiple related memories
memories = [
    "User likes Italian food",
    "User is allergic to nuts", 
    "User prefers evening meetings"
]

config = {
    "cluster_uri": "mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority",
    "database_name": "memory_db",
    "collection_name": "memories",
    "namespace": "user_123"
}

for content in memories:
    agent.tool.mongodb_memory(
        action="record",
        content=content,
        metadata={"batch": "user_preferences", "timestamp": datetime.now().isoformat()},
        **config
    )
```

## Troubleshooting

### Common Issues

1. **Connection Timeout**
   - Check MongoDB Atlas cluster status
   - Verify network connectivity and IP whitelist
   - Increase connection timeout settings

2. **Authentication Errors**
   - Verify connection URI format
   - Check database user credentials
   - Ensure user has proper permissions

3. **Vector Search Index Issues**
   - Verify Atlas Search is enabled
   - Check index creation status
   - Ensure proper index configuration

4. **Embedding Generation Failures**
   - Verify AWS credentials
   - Check Bedrock model access
   - Ensure proper IAM permissions

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed MongoDB and Bedrock API calls
result = agent.tool.mongodb_memory(
    action="record",
    content="test",
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/"
)
```

### Vector Search Index Creation

If vector search is not working, manually create the index in MongoDB Atlas:

1. Go to Atlas Search in your MongoDB Atlas dashboard
2. Create a new search index on your collection
3. Use the JSON configuration provided in the Collection Structure section
4. Wait for the index to build (this can take several minutes)

## Security Considerations

### Connection Security

- Use strong passwords for database users
- Enable IP whitelisting in MongoDB Atlas
- Use connection string with SSL/TLS enabled
- Store connection URIs securely (environment variables, secrets manager)

### Data Privacy

- Use appropriate namespaces for data isolation
- Consider encryption at rest (MongoDB Atlas feature)
- Implement proper access controls
- Regular security audits

### Network Security

- Use VPC peering for production environments
- Implement proper firewall rules
- Monitor database access logs
- Use private endpoints when available

## Support and Resources

- [MongoDB Atlas Documentation](https://docs.atlas.mongodb.com/)
- [MongoDB Atlas Vector Search](https://docs.atlas.mongodb.com/atlas-search/vector-search/)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Strands Agents Framework](https://strandsagents.com/)
- [GitHub Issues](https://github.com/strands-agents/tools/issues)
