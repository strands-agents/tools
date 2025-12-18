# Elasticsearch Memory Tool

The Elasticsearch Memory Tool provides comprehensive memory management capabilities using Elasticsearch as the backend with vector embeddings for semantic search. It uses the direct tool pattern where tools are imported and used directly with agents.

## Features

- **Semantic Search**: Automatic embedding generation using Amazon Bedrock Titan for vector similarity search
- **Memory Management**: Store, retrieve, list, get, and delete memory operations
- **Index Management**: Automatic index creation with proper mappings for vector search
- **Namespace Support**: Organize memories by namespace for multi-user scenarios
- **Pagination**: Support for paginated results in list and retrieve operations
- **Error Handling**: Comprehensive error handling with clear error messages

## Installation

Install the required dependencies:

```bash
pip install strands-agents-tools[elasticsearch_memory]
```

This will install:
- `elasticsearch>=8.0.0,<9.0.0` - Elasticsearch Python client

## Prerequisites

1. **Elasticsearch Cloud**: You need an Elasticsearch Cloud deployment with:
   - Cloud ID
   - API Key with appropriate permissions

2. **Amazon Bedrock**: Access to Amazon Bedrock for embedding generation:
   - AWS credentials configured
   - Access to `amazon.titan-embed-text-v2:0` model (or custom embedding model)

## Quick Start

### Basic Setup

```python
from strands import Agent
from strands_tools.elasticsearch_memory import elasticsearch_memory

# Create an agent with the direct tool
agent = Agent(tools=[elasticsearch_memory])

# Use the tool with configuration parameters
result = agent.tool.elasticsearch_memory(
    action="record",
    content="User prefers vegetarian pizza with extra cheese",
    cloud_id="your-elasticsearch-cloud-id",
    api_key="your-elasticsearch-api-key",
    index_name="my_memories",
    namespace="user_123"
)
```

### Environment Variables

You can also use environment variables for configuration:

```bash
export ELASTICSEARCH_CLOUD_ID="your-cloud-id"
export ELASTICSEARCH_API_KEY="your-api-key"
export ELASTICSEARCH_INDEX_NAME="my_memories"
export ELASTICSEARCH_NAMESPACE="user_123"
export ELASTICSEARCH_EMBEDDING_MODEL="amazon.titan-embed-text-v2:0"
export AWS_REGION="us-west-2"
```

Then use the tool with minimal parameters (environment variables will be used):

```python
result = agent.tool.elasticsearch_memory(
    action="record",
    content="User prefers vegetarian pizza"
    # cloud_id, api_key, etc. will be read from environment variables
)
```

## Usage Examples

### 1. Store Memories

```python
# Store a simple memory
result = agent.tool.elasticsearch_memory(
    action="record",
    content="User prefers vegetarian pizza with extra cheese and no onions",
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)

# Store a memory with metadata
result = agent.tool.elasticsearch_memory(
    action="record",
    content="Meeting scheduled for next Tuesday at 2 PM with the development team",
    metadata={
        "category": "meetings",
        "priority": "high",
        "participants": ["dev_team"],
        "date": "2024-01-16"
    },
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)
```

### 2. Semantic Search

```python
# Search for food-related memories
result = agent.tool.elasticsearch_memory(
    action="retrieve",
    query="food preferences and dietary restrictions",
    max_results=5,
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)

# Search for meeting information
result = agent.tool.elasticsearch_memory(
    action="retrieve",
    query="upcoming meetings and appointments",
    max_results=10,
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)
```

### 3. List All Memories

```python
# List recent memories
result = agent.tool.elasticsearch_memory(
    action="list",
    max_results=20,
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)

# List with pagination
result = agent.tool.elasticsearch_memory(
    action="list",
    max_results=10,
    next_token="10",  # Start from the 11th result
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)
```

### 4. Get Specific Memory

```python
# Retrieve a specific memory by ID
result = agent.tool.elasticsearch_memory(
    action="get",
    memory_id="mem_1704567890123_abc12345",
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)
```

### 5. Delete Memory

```python
# Delete a specific memory
result = agent.tool.elasticsearch_memory(
    action="delete",
    memory_id="mem_1704567890123_abc12345",
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)
```

## Advanced Configuration

### Using Configuration Dictionary

For cleaner code, you can use a configuration dictionary:

```python
config = {
    "cloud_id": "your-cloud-id",
    "api_key": "your-api-key",
    "index_name": "memories",
    "namespace": "user_123",
    "region": "us-east-1"
}

# Store memory
result = agent.tool.elasticsearch_memory(
    action="record",
    content="User prefers vegetarian pizza",
    **config
)

# Search memories
result = agent.tool.elasticsearch_memory(
    action="retrieve",
    query="food preferences",
    max_results=5,
    **config
)
```

### Custom Embedding Model

```python
result = agent.tool.elasticsearch_memory(
    action="record",
    content="User prefers vegetarian pizza",
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    embedding_model="amazon.titan-embed-text-v1:0",  # Different model
    region="us-east-1"
)
```

### Elasticsearch Serverless (URL-based connection)

```python
result = agent.tool.elasticsearch_memory(
    action="record",
    content="User prefers vegetarian pizza",
    es_url="https://your-serverless-cluster.es.region.aws.elastic.cloud:443",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123"
)
```

### Multiple Namespaces

```python
# User-specific memories
result = agent.tool.elasticsearch_memory(
    action="record",
    content="Alice likes Italian food",
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    namespace="user_alice"
)

# System-wide memories
result = agent.tool.elasticsearch_memory(
    action="record",
    content="System maintenance scheduled",
    cloud_id="your-cloud-id",
    api_key="your-api-key",
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

## Index Structure

The tool automatically creates an Elasticsearch index with the following mapping:

```json
{
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "standard"
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "index": true,
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
                "enabled": true
            }
        }
    }
}
```

## Error Handling

The tool provides comprehensive error handling:

### Connection Errors

```python
# Invalid credentials
result = agent.tool.elasticsearch_memory(
    action="record",
    content="test",
    cloud_id="invalid-cloud-id",
    api_key="invalid-api-key"
)
# Returns: {"status": "error", "content": [{"text": "Unable to connect to Elasticsearch cluster"}]}
```

### Missing Parameters

```python
# Missing required content for record action
result = agent.tool.elasticsearch_memory(
    action="record",
    cloud_id="your-cloud-id",
    api_key="your-api-key"
)
# Returns: {"status": "error", "content": [{"text": "The following parameters are required for record action: content"}]}

# Missing connection parameters
result = agent.tool.elasticsearch_memory(action="record", content="test")
# Returns: {"status": "error", "content": [{"text": "Either cloud_id or es_url is required"}]}
```

### Memory Not Found

```python
# Non-existent memory ID
result = agent.tool.elasticsearch_memory(
    action="get",
    memory_id="nonexistent",
    cloud_id="your-cloud-id",
    api_key="your-api-key"
)
# Returns: {"status": "error", "content": [{"text": "API error: Memory nonexistent not found"}]}
```

## Performance Considerations

### Embedding Generation

- Embeddings are generated using Amazon Bedrock Titan model
- Each record and retrieve operation requires embedding generation
- Consider caching strategies for frequently accessed queries

### Index Optimization

- The tool creates optimized indices for vector search
- Uses cosine similarity for semantic matching
- Configures appropriate shard and replica settings

### Pagination

- Use pagination for large result sets
- `max_results` parameter controls batch size
- `next_token` enables efficient pagination

## Best Practices

### 1. Configuration Management

Create reusable configuration objects:

```python
# Create a base configuration
base_config = {
    "cloud_id": "your-cloud-id",
    "api_key": "your-api-key",
    "region": "us-east-1"
}

# User-specific configuration
def get_user_config(user_id):
    return {
        **base_config,
        "index_name": "user_memories",
        "namespace": f"user_{user_id}"
    }

# Usage
user_config = get_user_config("alice")
result = agent.tool.elasticsearch_memory(
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
result = agent.tool.elasticsearch_memory(
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
        result = agent.tool.elasticsearch_memory(action=action, **kwargs)
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
    "cloud_id": "your-cloud-id",
    "api_key": "your-api-key",
    "index_name": "memories",
    "namespace": "user_123"
}

for content in memories:
    agent.tool.elasticsearch_memory(
        action="record",
        content=content,
        metadata={"batch": "user_preferences", "timestamp": datetime.now().isoformat()},
        **config
    )
```

## Troubleshooting

### Common Issues

1. **Connection Timeout**
   - Check Elasticsearch Cloud status
   - Verify network connectivity
   - Increase timeout settings

2. **Authentication Errors**
   - Verify Cloud ID format
   - Check API key permissions
   - Ensure API key is not expired

3. **Embedding Generation Failures**
   - Verify AWS credentials
   - Check Bedrock model access
   - Ensure proper IAM permissions

4. **Index Creation Failures**
   - Check Elasticsearch cluster resources
   - Verify index naming conventions
   - Review cluster settings

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed Elasticsearch and Bedrock API calls
result = agent.tool.elasticsearch_memory(
    action="record",
    content="test",
    cloud_id="your-cloud-id",
    api_key="your-api-key"
)
```

## Security Considerations

### API Key Management

- Store API keys securely (environment variables, secrets manager)
- Use least-privilege API keys
- Rotate API keys regularly
- Monitor API key usage

### Data Privacy

- Use appropriate namespaces for data isolation
- Consider encryption at rest (Elasticsearch feature)
- Implement proper access controls
- Regular security audits

### Network Security

- Use HTTPS for all connections
- Consider VPC/private networking for production
- Implement proper firewall rules
- Monitor network traffic

## Support and Resources

- [Elasticsearch Documentation](https://www.elastic.co/guide/index.html)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Strands Agents Framework](https://strandsagents.com/)
- [GitHub Issues](https://github.com/strands-agents/tools/issues)
