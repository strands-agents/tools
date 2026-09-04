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

## Security Model

Connection credentials (`cloud_id`/`es_url`/`api_key`), the target `index_name`, and the tenant `namespace` are **never** exposed as agent-facing tool parameters. The agent only chooses the `action` and its `content`/`query`/`memory_id`. This prevents a model (or prompt-injected content) from redirecting the memory layer at another cluster or index, authenticating with its own `api_key`, or reading, writing, or deleting another tenant's memories by supplying a different `namespace`.

There are two supported patterns:

- **Class-based (recommended, required for multi-tenant):** construct one `ElasticsearchMemoryTool` per authenticated principal, binding the connection, index, and `namespace` at construction time.
- **Standalone function (single-tenant):** the module-level `elasticsearch_memory` tool reads all connection, index, namespace, and embedding configuration from environment variables only.

## Quick Start

### Class-Based Usage (Recommended)

Bind the connection, index, and tenant `namespace` per authenticated principal. They are kept out of the agent-facing tool, so the model cannot change them.

```python
from strands import Agent
from strands_tools.elasticsearch_memory import ElasticsearchMemoryTool

# Operator code, per authenticated request:
memory_tool = ElasticsearchMemoryTool(
    cloud_id="your-elasticsearch-cloud-id",  # or es_url="https://...:443" for Serverless
    api_key="your-elasticsearch-api-key",
    index_name="my_memories",
    namespace=f"user_{authenticated_user_id}",  # bound, not LLM-controllable
)
agent = Agent(tools=[memory_tool.elasticsearch_memory])

# The agent only chooses the action and its content/query/memory_id:
result = agent.tool.elasticsearch_memory(action="record", content="User prefers vegetarian pizza")
result = agent.tool.elasticsearch_memory(action="retrieve", query="food preferences", max_results=5)
```

### Standalone Function Usage

Single-tenant convenience. All connection, index, namespace, and embedding configuration comes from environment variables; there are no connection or namespace parameters for the agent to supply.

```python
from strands import Agent
from strands_tools.elasticsearch_memory import elasticsearch_memory

agent = Agent(tools=[elasticsearch_memory])
result = agent.tool.elasticsearch_memory(action="record", content="User prefers vegetarian pizza")
```

### Environment Variables

Configure the standalone function (and the fallbacks for the class) with environment variables:

```bash
export ELASTICSEARCH_CLOUD_ID="your-cloud-id"
export ELASTICSEARCH_API_KEY="your-api-key"
export ELASTICSEARCH_INDEX_NAME="my_memories"
export ELASTICSEARCH_NAMESPACE="user_123"
export ELASTICSEARCH_EMBEDDING_MODEL="amazon.titan-embed-text-v2:0"
export AWS_REGION="us-west-2"
```

The standalone `elasticsearch_memory` function is single-tenant: it always uses `ELASTICSEARCH_NAMESPACE` (defaulting to `default`). To serve multiple principals, use the class-based approach and construct one `ElasticsearchMemoryTool` per principal with an explicit `namespace`.

```python
agent = Agent(tools=[elasticsearch_memory])
result = agent.tool.elasticsearch_memory(action="record", content="User prefers vegetarian pizza")
```

## Usage Examples

The examples below use a per-principal `ElasticsearchMemoryTool` (see Quick Start) and assume `agent = Agent(tools=[memory_tool.elasticsearch_memory])`.

### 1. Store Memories

```python
# Store a simple memory
result = agent.tool.elasticsearch_memory(
    action="record",
    content="User prefers vegetarian pizza with extra cheese and no onions",
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
)
```

### 2. Semantic Search

```python
# Search for food-related memories
result = agent.tool.elasticsearch_memory(
    action="retrieve",
    query="food preferences and dietary restrictions",
    max_results=5,
)

# Search for meeting information
result = agent.tool.elasticsearch_memory(
    action="retrieve",
    query="upcoming meetings and appointments",
    max_results=10,
)
```

### 3. List All Memories

```python
# List recent memories
result = agent.tool.elasticsearch_memory(
    action="list",
    max_results=20,
)

# List with pagination
result = agent.tool.elasticsearch_memory(
    action="list",
    max_results=10,
    next_token="10",  # Start from the 11th result
)
```

### 4. Get Specific Memory

```python
# Retrieve a specific memory by ID
result = agent.tool.elasticsearch_memory(
    action="get",
    memory_id="mem_1704567890123_abc12345",
)
```

### 5. Delete Memory

```python
# Delete a specific memory
result = agent.tool.elasticsearch_memory(
    action="delete",
    memory_id="mem_1704567890123_abc12345",
)
```

## Advanced Configuration

### Reusing a Connection Configuration

For cleaner code, collect the connection settings once and bind them to a per-principal tool at construction time:

```python
config = {
    "cloud_id": "your-cloud-id",
    "api_key": "your-api-key",
    "index_name": "memories",
    "region": "us-east-1",
}

# Bind the connection plus the authenticated principal's namespace.
memory_tool = ElasticsearchMemoryTool(namespace=f"user_{authenticated_user_id}", **config)
agent = Agent(tools=[memory_tool.elasticsearch_memory])

# Store memory
result = agent.tool.elasticsearch_memory(action="record", content="User prefers vegetarian pizza")

# Search memories
result = agent.tool.elasticsearch_memory(action="retrieve", query="food preferences", max_results=5)
```

### Custom Embedding Model

The embedding model and region are bound at construction (or via `ELASTICSEARCH_EMBEDDING_MODEL` / `AWS_REGION` for the standalone function):

```python
memory_tool = ElasticsearchMemoryTool(
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    namespace="user_123",
    embedding_model="amazon.titan-embed-text-v1:0",  # Different model
    region="us-east-1",
)
```

### Elasticsearch Serverless (URL-based connection)

Use `es_url` instead of `cloud_id` for Serverless deployments:

```python
memory_tool = ElasticsearchMemoryTool(
    es_url="https://your-serverless-cluster.es.region.aws.elastic.cloud:443",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123",
)
agent = Agent(tools=[memory_tool.elasticsearch_memory])
result = agent.tool.elasticsearch_memory(action="record", content="User prefers vegetarian pizza")
```

### Multiple Namespaces

The `namespace` is bound per tool instance. To serve multiple principals (or logical groupings), construct one `ElasticsearchMemoryTool` per namespace — never let the agent choose the namespace on a call.

```python
# One tool per user, each bound to that user's namespace
alice_tool = ElasticsearchMemoryTool(
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    namespace="user_alice",
)

# A separate tool for system-wide memories
system_tool = ElasticsearchMemoryTool(
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    namespace="system_global",
)

# Wire the tool that matches the authenticated principal into that principal's agent.
alice_agent = Agent(tools=[alice_tool.elasticsearch_memory])
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
# Invalid credentials (configured at construction)
memory_tool = ElasticsearchMemoryTool(
    cloud_id="invalid-cloud-id",
    api_key="invalid-api-key",
    namespace="user_123",
)
agent = Agent(tools=[memory_tool.elasticsearch_memory])
result = agent.tool.elasticsearch_memory(action="record", content="test")
# Returns: {"status": "error", "content": [{"text": "Unable to connect to Elasticsearch cluster"}]}
```

### Missing Parameters

```python
# Missing required content for record action
result = agent.tool.elasticsearch_memory(action="record")
# Returns: {"status": "error", "content": [{"text": "The following parameters are required for record action: content"}]}

# Missing connection configuration (no api_key at construction and no
# ELASTICSEARCH_API_KEY environment variable) raises at construction time:
ElasticsearchMemoryTool(cloud_id="your-cloud-id", namespace="user_123")
# Raises: ElasticsearchValidationError("api_key is required for Elasticsearch Memory Tool initialization")
```

### Memory Not Found

```python
# Non-existent memory ID
result = agent.tool.elasticsearch_memory(action="get", memory_id="nonexistent")
# Returns: {"status": "error", "content": [{"text": "Memory nonexistent not found in namespace user_123"}]}
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

Build a per-principal tool from reusable connection settings, binding the authenticated user's namespace at construction:

```python
# Create a base connection configuration
base_config = {
    "cloud_id": "your-cloud-id",
    "api_key": "your-api-key",
    "index_name": "user_memories",
    "region": "us-east-1",
}

# Build one tool per authenticated principal
def build_user_agent(user_id):
    memory_tool = ElasticsearchMemoryTool(namespace=f"user_{user_id}", **base_config)
    return Agent(tools=[memory_tool.elasticsearch_memory])

# Usage
alice_agent = build_user_agent("alice")
result = alice_agent.tool.elasticsearch_memory(action="record", content="Alice likes Italian food")
```

### 2. Namespace Organization

The `namespace` is crucial for data isolation and multi-tenant memory management. It is bound per tool instance and is not agent-controllable:

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

# Bind the chosen namespace when constructing the tool
memory_tool = ElasticsearchMemoryTool(
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    namespace=user_namespace,
)
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
# Store multiple related memories with a single per-principal tool
memories = [
    "User likes Italian food",
    "User is allergic to nuts", 
    "User prefers evening meetings"
]

memory_tool = ElasticsearchMemoryTool(
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    index_name="memories",
    namespace="user_123",
)
agent = Agent(tools=[memory_tool.elasticsearch_memory])

for content in memories:
    agent.tool.elasticsearch_memory(
        action="record",
        content=content,
        metadata={"batch": "user_preferences", "timestamp": datetime.now().isoformat()},
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
memory_tool = ElasticsearchMemoryTool(
    cloud_id="your-cloud-id",
    api_key="your-api-key",
    namespace="user_123",
)
agent = Agent(tools=[memory_tool.elasticsearch_memory])
result = agent.tool.elasticsearch_memory(action="record", content="test")
```

## Security Considerations

### API Key Management

- Store API keys securely (environment variables, secrets manager)
- Use least-privilege API keys
- Rotate API keys regularly
- Monitor API key usage

### Data Privacy

- Bind the `namespace` per authenticated principal at construction; never expose it as an agent-controllable parameter (see Security Model)
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
