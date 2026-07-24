# MongoDB Atlas Memory Tool

The MongoDB Atlas Memory Tool provides comprehensive memory management capabilities using MongoDB Atlas as the backend with vector embeddings for semantic search. It uses the direct tool pattern where tools are imported and used directly with agents.

## Features

- **Semantic Search**: Automatic embedding generation using Amazon Bedrock Titan for vector similarity search
- **Memory Management**: Store, retrieve, list, get, and delete memory operations
- **Index Management**: Automatic vector search index creation with proper configuration
- **Namespace Field**: Organize and filter memories using a `namespace` document field for logical grouping within a collection
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
   - Connection URI (mongodb+srv format) - [How to find your connection string](https://www.mongodb.com/docs/atlas/connect-to-database-deployment/)
   - Database user with read/write permissions - [Create database user](https://www.mongodb.com/docs/atlas/security-add-mongodb-users/)
   - Vector Search enabled (Atlas Search) - [Enable Atlas Search](https://www.mongodb.com/docs/atlas/atlas-search/create-index/)

2. **Amazon Bedrock**: Access to Amazon Bedrock for embedding generation:
   - AWS credentials configured
   - Access to `amazon.titan-embed-text-v2:0` model (or custom embedding model)

### Getting Your MongoDB Atlas Connection URI

If you're new to MongoDB Atlas:

1. **Sign up for MongoDB Atlas**: Visit [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and create a free account
2. **Create a cluster**: Follow the setup wizard to create your first cluster (free tier available)
3. **Create a database user**: Go to Database Access → Add New Database User with read/write permissions
4. **Configure network access**: Go to Network Access → Add IP Address (add your current IP or 0.0.0.0/0 for testing)
5. **Get connection string**: 
   - Go to your cluster in the Atlas dashboard
   - Click "Connect" button
   - Choose "Connect your application"
   - Select "Python" as the driver
   - Copy the connection string (it will look like: `mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/`)
   - Replace `<password>` with your actual database user password

**Important**: Your connection URI should be in the format `mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/` without additional query parameters. The tool will handle SSL and other connection settings automatically.

For detailed instructions, see the [official MongoDB Atlas documentation](https://www.mongodb.com/docs/atlas/connect-to-database-deployment/).

## Security Model

Connection credentials, the target database/collection, and the tenant `namespace` are **never** exposed as agent-facing tool parameters. The agent only chooses the `action` and its `content`/`query`/`memory_id`. This prevents a model (or prompt-injected content) from redirecting the memory layer at another cluster or reading, writing, or deleting another tenant's memories by supplying a different `namespace`.

There are two supported patterns:

- **Class-based (recommended, required for multi-tenant):** construct one `MongoDBMemoryTool` per authenticated principal, binding the connection and `namespace` at construction time.
- **Standalone function (single-tenant):** the module-level `mongodb_memory` tool reads all connection, collection, namespace, and embedding configuration from environment variables only.

## Quick Start

### Class-Based Usage (Recommended)

Bind the connection, collection, and tenant `namespace` per authenticated principal. They are kept out of the agent-facing tool, so the model cannot change them.

```python
from strands import Agent
from strands_tools.mongodb_memory import MongoDBMemoryTool

# Operator code, per authenticated request:
memory_tool = MongoDBMemoryTool(
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="memory_db",
    collection_name="memories",
    namespace=f"user_{authenticated_user_id}",  # bound, not LLM-controllable
)
agent = Agent(tools=[memory_tool.mongodb_memory])

# The agent only chooses the action and its content/query/memory_id:
result = agent.tool.mongodb_memory(action="record", content="User prefers vegetarian pizza")
result = agent.tool.mongodb_memory(action="retrieve", query="food preferences", max_results=5)
```

### Standalone Function Usage

Single-tenant convenience. All connection, collection, namespace, and embedding configuration comes from environment variables; there are no connection or namespace parameters for the agent to supply.

```python
from strands import Agent
from strands_tools.mongodb_memory import mongodb_memory

agent = Agent(tools=[mongodb_memory])
result = agent.tool.mongodb_memory(action="record", content="User prefers vegetarian pizza")
```

### Environment Variables

Configure the standalone function (and the fallbacks for the class) with environment variables:

```bash
export MONGODB_ATLAS_CLUSTER_URI="mongodb+srv://user:password@cluster.mongodb.net/"
export MONGODB_DATABASE_NAME="memory_db"
export MONGODB_COLLECTION_NAME="memories"
export MONGODB_NAMESPACE="user_123"
export MONGODB_EMBEDDING_MODEL="amazon.titan-embed-text-v2:0"
export AWS_REGION="us-west-2"
```

The standalone `mongodb_memory` function is single-tenant: it always uses `MONGODB_NAMESPACE` (defaulting to `default`). To serve multiple principals, use the class-based approach and construct one `MongoDBMemoryTool` per principal with an explicit `namespace`.

```python
# Standalone function usage (configuration entirely from environment variables)
agent = Agent(tools=[mongodb_memory])
result = agent.tool.mongodb_memory(action="record", content="User prefers vegetarian pizza")
```

## Usage Examples

The examples below use a per-principal `MongoDBMemoryTool` (see Quick Start) and assume `agent = Agent(tools=[memory_tool.mongodb_memory])`.

### 1. Store Memories

```python
# Store a simple memory
result = agent.tool.mongodb_memory(
    action="record",
    content="User prefers vegetarian pizza with extra cheese and no onions",
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
)
```

### 2. Semantic Search

```python
# Search for food-related memories
result = agent.tool.mongodb_memory(
    action="retrieve",
    query="food preferences and dietary restrictions",
    max_results=5,
)

# Search for meeting information
result = agent.tool.mongodb_memory(
    action="retrieve",
    query="upcoming meetings and appointments",
    max_results=10,
)
```

### 3. List All Memories

```python
# List recent memories
result = agent.tool.mongodb_memory(
    action="list",
    max_results=20,
)

# List with pagination
result = agent.tool.mongodb_memory(
    action="list",
    max_results=10,
    next_token="10",  # Start from the 11th result
)
```

### 4. Get Specific Memory

```python
# Retrieve a specific memory by ID
result = agent.tool.mongodb_memory(
    action="get",
    memory_id="mem_1704567890123_abc12345",
)
```

### 5. Delete Memory

```python
# Delete a specific memory
result = agent.tool.mongodb_memory(
    action="delete",
    memory_id="mem_1704567890123_abc12345",
)
```

## Advanced Configuration

### Reusing a Connection Configuration

For cleaner code, collect the connection settings once and bind them to a per-principal tool at construction time:

```python
config = {
    "cluster_uri": "mongodb+srv://user:password@cluster.mongodb.net/",
    "database_name": "memory_db",
    "collection_name": "memories",
    "region": "us-east-1",
}

# Bind the connection plus the authenticated principal's namespace.
memory_tool = MongoDBMemoryTool(namespace=f"user_{authenticated_user_id}", **config)
agent = Agent(tools=[memory_tool.mongodb_memory])

# Store memory
result = agent.tool.mongodb_memory(action="record", content="User prefers vegetarian pizza")

# Search memories
result = agent.tool.mongodb_memory(action="retrieve", query="food preferences", max_results=5)
```

### Custom Embedding Model

The embedding model and region are bound at construction (or via `MONGODB_EMBEDDING_MODEL` / `AWS_REGION` for the standalone function):

```python
memory_tool = MongoDBMemoryTool(
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123",
    embedding_model="amazon.titan-embed-text-v1:0",  # Different model
    region="us-east-1",
)
```

### Multiple Namespaces

The `namespace` is bound per tool instance. To serve multiple principals (or logical groupings), construct one `MongoDBMemoryTool` per namespace — never let the agent choose the namespace on a call.

```python
# One tool per user, each bound to that user's namespace
alice_tool = MongoDBMemoryTool(
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_alice",
)

# A separate tool for system-wide memories
system_tool = MongoDBMemoryTool(
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="memory_db",
    collection_name="memories",
    namespace="system_global",
)

# Wire the tool that matches the authenticated principal into that principal's agent.
alice_agent = Agent(tools=[alice_tool.mongodb_memory])
```

## Response Format

All operations return a standardized response format:

```python
{
    "status": "success",  # or "error"
    "content": [
        {
            "text": "Memory stored successfully"
        },
        {
            "json": {...}
        }
    ]
}
```

### Successful Record Response

```json
{
    "status": "success",
    "content": [
        {"text": "Memory stored successfully"},
        {"json": {"memory_id": "mem_1704567890123_abc12345", "content": "User prefers vegetarian pizza", "namespace": "user_123", "timestamp": "2024-01-06T20:31:30.123456Z", "result": "created"}}
    ]
}
```

### Successful Retrieve Response

```json
{
    "status": "success",
    "content": [
        {"text": "Memories retrieved successfully"},
        {"json": {"memories": [{"memory_id": "mem_123", "content": "User prefers vegetarian pizza", "timestamp": "2024-01-06T20:31:30Z", "metadata": {"category": "food"}, "score": 0.95}], "total": 1, "max_score": 0.95}}
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
# Invalid connection URI (configured at construction)
memory_tool = MongoDBMemoryTool(
    cluster_uri="mongodb+srv://invalid:credentials@invalid.mongodb.net/",
    namespace="user_123",
)
agent = Agent(tools=[memory_tool.mongodb_memory])
result = agent.tool.mongodb_memory(action="record", content="test")
# Returns: {"status": "error", "content": [{"text": "Unable to connect to MongoDB cluster ..."}]}
```

### Missing Parameters

```python
# Missing required content for record action
result = agent.tool.mongodb_memory(action="record")
# Returns: {"status": "error", "content": [{"text": "The following parameters are required for record action: content"}]}

# Missing connection configuration (no cluster_uri at construction and no
# MONGODB_ATLAS_CLUSTER_URI environment variable) raises at construction time:
MongoDBMemoryTool(namespace="user_123")
# Raises: MongoDBValidationError("cluster_uri is required for MongoDB Memory Tool initialization")
```

### Memory Not Found

```python
# Non-existent memory ID
result = agent.tool.mongodb_memory(action="get", memory_id="nonexistent")
# Returns: {"status": "error", "content": [{"text": "API error: Memory nonexistent not found ..."}]}
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

Build a per-principal tool from reusable connection settings, binding the authenticated user's namespace at construction:

```python
# Create a base connection configuration
base_config = {
    "cluster_uri": "mongodb+srv://user:password@cluster.mongodb.net/",
    "database_name": "memory_db",
    "collection_name": "user_memories",
    "region": "us-east-1",
}

# Build one tool per authenticated principal
def build_user_agent(user_id):
    memory_tool = MongoDBMemoryTool(namespace=f"user_{user_id}", **base_config)
    return Agent(tools=[memory_tool.mongodb_memory])

# Usage
alice_agent = build_user_agent("alice")
result = alice_agent.tool.mongodb_memory(action="record", content="Alice likes Italian food")
```

### 2. Namespace Organization

The `namespace` is a document field used for logical grouping and query filtering within a collection. It is bound per tool instance and is not agent-controllable.

> **Note:** This differs from [MongoDB's glossary definition of "namespace"](https://www.mongodb.com/docs/manual/reference/glossary/#std-term-namespace), which refers to the combination of database and collection names.

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
memory_tool = MongoDBMemoryTool(
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="memory_db",
    collection_name="memories",
    namespace=user_namespace,
)
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

# Usage example:
result = safe_memory_operation(agent, "record", content="Test memory")
```

### 5. Batch Operations

```python
# Store multiple related memories with a single per-principal tool
memories = [
    "User likes Italian food",
    "User is allergic to nuts", 
    "User prefers evening meetings"
]

memory_tool = MongoDBMemoryTool(
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/",
    database_name="memory_db",
    collection_name="memories",
    namespace="user_123",
)
agent = Agent(tools=[memory_tool.mongodb_memory])

for content in memories:
    agent.tool.mongodb_memory(
        action="record",
        content=content,
        metadata={"batch": "user_preferences", "timestamp": datetime.now().isoformat()},
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
memory_tool = MongoDBMemoryTool(
    cluster_uri="mongodb+srv://user:password@cluster.mongodb.net/",
    namespace="user_123",
)
agent = Agent(tools=[memory_tool.mongodb_memory])
result = agent.tool.mongodb_memory(action="record", content="test")
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

- Bind the `namespace` per authenticated principal at construction; never expose it as an agent-controllable parameter (see Security Model)
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
