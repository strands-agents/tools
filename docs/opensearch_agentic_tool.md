# OpenSearch Agentic Search Tool

The OpenSearch Agentic Search Tool provides intelligent search capabilities using OpenSearch and Amazon Bedrock Claude for natural language query processing. It creates a complete agentic search pipeline with model registration, local agent deployment, and intelligent query translation.

## Features

- **Agentic Search**: Natural language queries translated to optimized OpenSearch DSL
- **Model Management**: Register and deploy Bedrock Claude models with OpenSearch ML
- **Agent Management**: Local conversational agents with specialized search tools
- **Pipeline Management**: Automated search pipeline creation with request/response processors
- **AWS Integration**: Native AWS authentication for managed OpenSearch clusters
- **State Caching**: Local caching of models, agents, and pipelines for reuse
- **Cross-Index Search**: Support for single index or cluster-wide search operations

## Installation

Install the required dependencies:

```bash
pip install strands-agents-tools[opensearch_agentic]
```

This will install:
- `requests>=2.25.0` - HTTP client for OpenSearch API calls
- `boto3>=1.26.0` - AWS SDK for authentication and Bedrock access
- `requests-aws4auth>=1.1.0` - AWS signature authentication

## Prerequisites

1. **OpenSearch Cluster**: Either AWS-managed or self-managed OpenSearch cluster
   - **Version Requirement**: OpenSearch 3.3+ (for ML plugin support)
   - AWS OpenSearch Service or OpenSearch Serverless
   - Or self-managed OpenSearch with basic authentication

2. **Amazon Bedrock**: Access to Amazon Bedrock for Claude model:
   - AWS credentials configured (IAM role, profile, or environment variables)
   - IAM role with Bedrock access permissions
   - Access to `us.anthropic.claude-sonnet-4-20250514-v1:0` model

3. **OpenSearch ML Plugin**: Ensure ML plugin is enabled on your cluster
   - Required for model registration and agent functionality
   - Available by default in AWS OpenSearch Service 3.3+

## Quick Start

### Basic Setup

```python
from strands import Agent
from strands_tools.opensearch_agentic_search import opensearch_agentic_search_tool

# Create an agent with the agentic search tool
agent = Agent(tools=[opensearch_agentic_search_tool])

# Use the tool for natural language search
result = agent.tool.opensearch_agentic_search_tool(
    query_text="Find me shoes under $100",
    host="https://my-cluster.us-west-2.aos.amazonaws.com",
    index="products",  # optional - index is selected by the local agent based on indices name
    role_arn="arn:aws:iam::123456789:role/BedrockAccess"
)
```

### Environment Variables

Configure using environment variables for cleaner code:

```bash
export OPENSEARCH_HOST="https://my-cluster.us-west-2.aos.amazonaws.com"
export OPENSEARCH_INDEX="documents"  # optional
export BEDROCK_ROLE_ARN="arn:aws:iam::123456789:role/BedrockAccess"
export AWS_REGION="us-west-2"
export AWS_PROFILE="my-profile"  # optional
```

Then use the tool with minimal parameters:

```python
result = agent.tool.opensearch_agentic_search_tool(
    query_text="Show me wireless headphones with good reviews"
    # host, role_arn, etc. will be read from environment variables
)
```

## Usage Examples

### 1. Basic Natural Language Search

```python
# Search specific index
result = agent.tool.opensearch_agentic_search_tool(
    query_text="Find me a laptop for gaming under $1500",
    host="https://my-cluster.us-west-2.aos.amazonaws.com",
    index="electronics",
    role_arn="arn:aws:iam::123456789:role/BedrockAccess"
)

# Search across all indices
result = agent.tool.opensearch_agentic_search_tool(
    query_text="Show me customer complaints from last week",
    host="https://my-cluster.us-west-2.aos.amazonaws.com",
    role_arn="arn:aws:iam::123456789:role/BedrockAccess"
    # No index specified - searches all indices
)
```

### 2. Complex Query Examples

```python
# Shopping query
result = agent.tool.opensearch_agentic_search_tool(
    query_text="Find me sunglasses for my dad who likes classic styles",
    host="https://my-cluster.us-west-2.aos.amazonaws.com",
    index="products"
)

# Travel planning query
result = agent.tool.opensearch_agentic_search_tool(
    query_text="Show me hotels in Paris with good breakfast and near the Eiffel Tower under 200 euros per night",
    host="https://my-cluster.us-west-2.aos.amazonaws.com",
    index="hotels"
)

# Recipe search query
result = agent.tool.opensearch_agentic_search_tool(
    query_text="Find me easy dinner recipes for kids that don't have nuts or dairy",
    host="https://my-cluster.us-west-2.aos.amazonaws.com",
    index="recipes"
)
```

### 3. Direct Client Usage

```python
from agent import AgenticSearchClient

# Initialize client
client = AgenticSearchClient(
    host="https://my-cluster.us-west-2.aos.amazonaws.com",
    index="documents",  # optional
    role_arn="arn:aws:iam::123456789:role/BedrockAccess"
)

# Step-by-step setup
model_id = client.register_model()
print(f"Model registered: {model_id}")

agent_id = client.register_agent()
print(f"Agent registered: {agent_id}")

pipeline_name = client.create_pipeline()
print(f"Pipeline created: {pipeline_name}")

# Execute search
results = client.query("Find me running shoes for flat feet")
print(f"Search results: {results}")
```

### 4. Self-Managed OpenSearch

```python
# For self-managed OpenSearch clusters
import os
os.environ["OPENSEARCH_USERNAME"] = "admin"
os.environ["OPENSEARCH_PASSWORD"] = "your-password"

result = agent.tool.opensearch_agentic_search_tool(
    query_text="Find me error logs from yesterday",
    host="https://my-self-managed-cluster.com:9200",
    index="logs"
    # No role_arn needed for self-managed clusters
)
```

## Advanced Configuration

### Using Configuration Dictionary

```python
config = {
    "host": "https://my-cluster.us-west-2.aos.amazonaws.com",
    "role_arn": "arn:aws:iam::123456789:role/BedrockAccess",
    "index": "documents"
}

# Search with configuration
result = agent.tool.opensearch_agentic_search_tool(
    query_text="Find me a winter jacket for hiking",
    **config
)
```

### Custom Cache Location

```python
import os
os.environ["OPENSEARCH_CACHE_FILE"] = "/path/to/custom/cache.json"

# The tool will use the custom cache location for state persistence
result = agent.tool.opensearch_agentic_search_tool(
    query_text="Show me books about cooking",
    host="https://my-cluster.us-west-2.aos.amazonaws.com"
)
```

### Multiple Clusters

```python
# Production cluster
prod_config = {
    "host": "https://prod-cluster.us-west-2.aos.amazonaws.com",
    "role_arn": "arn:aws:iam::123456789:role/ProdBedrockAccess",
    "index": "prod_documents"
}

# Development cluster
dev_config = {
    "host": "https://dev-cluster.us-west-2.aos.amazonaws.com",
    "role_arn": "arn:aws:iam::123456789:role/DevBedrockAccess",
    "index": "dev_documents"
}

# Search production
prod_result = agent.tool.opensearch_agentic_search_tool(
    query_text="Show me system alerts from today",
    **prod_config
)

# Search development
dev_result = agent.tool.opensearch_agentic_search_tool(
    query_text="Find me test results that failed",
    **dev_config
)
```

## Response Format

The tool returns structured responses with agent reasoning and generated queries:

```python
{
    "status": "success",  # or "error"
    "content": [
        {
            "text": "{\"hits\": {...}, \"ext\": {\"agent_steps_summary\": \"...\", \"dsl_query\": \"...\"}}"
        }
    ]
}
```

### Successful Search Response

```json
{
    "status": "success",
    "content": [
        {
            "text": "{
                \"took\": 45,
                \"hits\": {
                    \"total\": {\"value\": 3, \"relation\": \"eq\"},
                    \"hits\": [
                        {
                            \"_index\": \"documents\",
                            \"_id\": \"doc1\",
                            \"_score\": 0.95,
                            \"_source\": {
                                \"title\": \"Nike Air Max 90\",
                                \"content\": \"Comfortable running shoes with air cushioning, perfect for daily wear...\"
                            }
                        }
                    ]
                },
                \"ext\": {
                    \"agent_steps_summary\": \"1. Analyzed query for shoe preferences and price range\\n2. Generated search for running shoes under $100\\n3. Applied price and category filters\",
                    \"dsl_query\": \"{\\\"query\\\": {\\\"bool\\\": {\\\"should\\\": [...]}}}\"
                }
            }"
        }
    ]
}
```

## Local Agent Tools and Capabilities

The agentic search system includes four specialized tools:

### 1. ListIndexTool
- Discovers available OpenSearch indices
- Provides index metadata and statistics
- Helps agent understand data structure

### 2. IndexMappingTool
- Analyzes field mappings and data types
- Understands document structure
- Optimizes query generation based on schema

### 3. WebSearchTool (DuckDuckGo)
- Performs external web searches when needed
- Enriches context for complex queries
- Provides additional information sources

### 4. QueryPlanningTool
- Generates optimized OpenSearch DSL queries
- Applies semantic search strategies
- Handles complex query logic and filtering

## Pipeline Architecture

The agentic search pipeline consists of:

### Request Processor: agentic_query_translator
- Receives natural language queries
- Uses the registered agent to analyze and plan
- Translates to optimized OpenSearch DSL

### Response Processor: agentic_context
- Processes search results
- Generates agent steps summary
- Includes generated DSL query in response
- Provides reasoning transparency

## Authentication Methods

### AWS OpenSearch Service

Automatic detection for hostnames containing:
- `.aos.amazonaws.com` (OpenSearch Service)
- `.aoss.amazonaws.com` (OpenSearch Serverless)

Uses AWS4Auth with:
- Boto3 session credentials
- IAM roles, profiles, or environment variables
- Automatic region detection from hostname

```python
# AWS authentication (automatic)
result = agent.tool.opensearch_agentic_search_tool(
    query_text="search query",
    host="https://my-cluster.us-west-2.aos.amazonaws.com",
    role_arn="arn:aws:iam::123456789:role/BedrockAccess"
)
```

### Self-Managed OpenSearch

Uses HTTP Basic Authentication:

```python
import os
os.environ["OPENSEARCH_USERNAME"] = "admin"
os.environ["OPENSEARCH_PASSWORD"] = "your-password"

result = agent.tool.opensearch_agentic_search_tool(
    query_text="search query",
    host="https://my-cluster.com:9200"
)
```

## State Management

The tool maintains local state for efficient reuse:

### Cache Location
- Default: `~/.opensearch_agentic_cache.json`
- Custom: Set `OPENSEARCH_CACHE_FILE` environment variable

### Cached Information
- Model ID and registration details
- Agent ID and configuration
- Pipeline name and settings
- Timestamps for uniqueness

### Cache Benefits
- Avoids re-registering models and agents
- Faster subsequent searches
- Persistent across sessions
- Automatic cleanup and regeneration

## Error Handling

### Common Error Scenarios

#### Authentication Errors
```python
# Invalid AWS credentials
result = agent.tool.opensearch_agentic_search_tool(
    query_text="test",
    host="https://cluster.us-west-2.aos.amazonaws.com",
    role_arn="arn:aws:iam::123456789:role/InvalidRole"
)
# Returns: {"status": "error", "content": [{"text": "Access denied to OpenSearch. Check your AWS credentials and IAM permissions."}]}
```

#### Missing Parameters
```python
# Missing required host
result = agent.tool.opensearch_agentic_search_tool(query_text="test")
# Returns: {"status": "error", "content": [{"text": "host is required (via parameter or OPENSEARCH_HOST environment variable)"}]}
```

#### Connection Issues
```python
# Invalid host
result = agent.tool.opensearch_agentic_search_tool(
    query_text="test",
    host="https://invalid-cluster.com"
)
# Returns: {"status": "error", "content": [{"text": "OpenSearch API error: 404 - Not Found"}]}
```

## Performance Considerations

### Model Registration
- Models are registered once and cached
- Automatic deployment with timestamped names
- Reuse across multiple search sessions

### Agent Creation
- Agents are created once per model
- Local execution within OpenSearch
- No external API calls during search

### Pipeline Optimization
- Pipelines are created once and reused
- Request/response processors optimize query flow
- Efficient query translation and result processing

### Search Performance
- Cross-index search when no index specified
- Semantic search with relevance scoring
- Agent-optimized DSL query generation

## Best Practices

### 1. Environment Configuration

```python
# Create environment configuration file
# .env
OPENSEARCH_HOST=https://my-cluster.us-west-2.aos.amazonaws.com
BEDROCK_ROLE_ARN=arn:aws:iam::123456789:role/BedrockAccess
AWS_REGION=us-west-2
AWS_PROFILE=production

# Load in application
from dotenv import load_dotenv
load_dotenv()

# Use with minimal parameters
result = agent.tool.opensearch_agentic_search_tool(
    query_text="Find customer feedback"
)
```

#### Alternative: Export Environment Variables

```bash
# OpenSearch configuration
export OPENSEARCH_HOST="https://my-cluster.us-west-2.aos.amazonaws.com"
export OPENSEARCH_INDEX="products"  # optional
export BEDROCK_ROLE_ARN="arn:aws:iam::123456789:role/BedrockAccess"

# AWS authentication - choose one method:

# Method 1: AWS Profile (recommended)
export AWS_PROFILE="production"
export AWS_REGION="us-west-2"

# Method 2: AWS Access Keys (for CI/CD or containers)
export AWS_ACCESS_KEY_ID="access key"
export AWS_SECRET_ACCESS_KEY="secret key"
export AWS_SESSION_TOKEN="optional-session-token-for-temporary-credentials"
export AWS_REGION="us-west-2"

# Method 3: AWS Default Region (uses default credentials)
export AWS_DEFAULT_REGION="us-west-2"
```

### 2. Query Optimization

```python
# Specific and detailed queries work best
good_query = "Find me wireless earbuds with noise cancellation under $200"

# Vague queries may produce less relevant results
avoid_query = "find stuff"

# Use natural language with specific requirements
natural_query = "Show me Italian restaurants near downtown with outdoor seating and good pasta"
```

### 3. Index Strategy

```python
# Search specific index for focused results
focused_result = agent.tool.opensearch_agentic_search_tool(
    query_text="Find me hiking boots for winter",
    index="outdoor_gear"
)

# Search all indices for comprehensive results
comprehensive_result = agent.tool.opensearch_agentic_search_tool(
    query_text="Show me anything related to camping"
    # No index specified - searches all
)
```

### 4. Error Handling

```python
def safe_opensearch_agentic_search(agent, query_text, **kwargs):
    try:
        result = agent.tool.opensearch_agentic_search_tool(
            query_text=query_text,
            **kwargs
        )
        if result["status"] == "error":
            logger.error(f"Search failed: {result['content'][0]['text']}")
            return None
        return result
    except Exception as e:
        logger.error(f"Unexpected error in agentic search: {e}")
        return None

# Usage
result = safe_agentic_search(
    agent,
    "Find me a coffee maker with a timer",
    host="https://my-cluster.us-west-2.aos.amazonaws.com"
)
```

### 5. Multi-Environment Setup

```python
class OpenSearchConfig:
    def __init__(self, environment):
        self.configs = {
            "development": {
                "host": "https://dev-cluster.us-west-2.aos.amazonaws.com",
                "role_arn": "arn:aws:iam::123456789:role/DevBedrockAccess",
                "index": "dev_documents"
            },
            "staging": {
                "host": "https://staging-cluster.us-west-2.aos.amazonaws.com",
                "role_arn": "arn:aws:iam::123456789:role/StagingBedrockAccess",
                "index": "staging_documents"
            },
            "production": {
                "host": "https://prod-cluster.us-west-2.aos.amazonaws.com",
                "role_arn": "arn:aws:iam::123456789:role/ProdBedrockAccess",
                "index": "prod_documents"
            }
        }
    
    def get_config(self, environment):
        return self.configs.get(environment, self.configs["development"])

# Usage
config = OpenSearchConfig("production")
result = agent.tool.opensearch_agentic_search_tool(
    query_text="Find me deals on smartphones",
    **config.get_config("production")
)
```

## Troubleshooting

### Common Issues

1. **OpenSearch Version Compatibility**
   - **Minimum Version**: OpenSearch 3.3+ required for ML plugin support
   - **AWS OpenSearch Service**: Use version 3.3 or higher
   - **Self-managed**: Ensure ML plugin is installed and enabled
   - **Check Version**: Use `GET /` API to verify cluster version

2. **Model Registration Failures**
   - Verify BEDROCK_ROLE_ARN has proper permissions
   - Check AWS region configuration
   - Ensure Bedrock model access is enabled
   - Confirm OpenSearch version supports ML models (3.3+)

3. **Agent Creation Errors**
   - Verify OpenSearch ML plugin is enabled
   - Check cluster resources and capacity
   - Review agent tool configurations
   - Ensure OpenSearch version 3.3+ for agent support

4. **Pipeline Creation Issues**
   - Ensure agent is successfully registered
   - Check OpenSearch version compatibility (3.3+)
   - Verify pipeline naming conventions
   - Confirm ML plugin is active

5. **Search Query Failures**
   - Validate index names and existence
   - Check query syntax and complexity
   - Review authentication and permissions
   - Verify agentic search pipeline is created

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed OpenSearch and Bedrock API calls
result = agent.tool.opensearch_agentic_search_tool(
    query_text="Find me a birthday gift for my mom",
    host="https://my-cluster.us-west-2.aos.amazonaws.com"
)
```

### Cache Management

```python
import os
from pathlib import Path

# Clear cache for fresh setup
cache_file = Path.home() / ".opensearch_agentic_cache.json"
if cache_file.exists():
    cache_file.unlink()
    print("Cache cleared")

# Use custom cache location
os.environ["OPENSEARCH_CACHE_FILE"] = "/tmp/custom_cache.json"
```

## Security Considerations

### IAM Permissions

The Bedrock role requires both a trust policy and permissions policy:

#### Trust Policy (Required)
The role must allow OpenSearch service to assume it:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "es.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        },
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "es.aws.internal"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```

#### Permissions Policy (Required)
The role must have permissions to invoke Bedrock models:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/us.anthropic.claude-sonnet-4-20250514-v1:0"
            ]
        }
    ]
}
```

### OpenSearch Permissions

Required OpenSearch permissions:
- Cluster: `cluster:admin/opensearch/ml/*`
- Index: `indices:admin/*`, `indices:data/*`
- Search: `indices:data/read/search*`

### Network Security

- Use HTTPS for all connections
- Consider VPC endpoints for AWS OpenSearch
- Implement proper security groups and NACLs
- Monitor API access and usage

### Data Privacy

- Use appropriate index naming for data classification
- Implement field-level security if needed
- Consider encryption at rest and in transit
- Regular security audits and access reviews

## Support and Resources

- [OpenSearch Documentation](https://opensearch.org/docs/)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS OpenSearch Service](https://docs.aws.amazon.com/opensearch-service/)
- [Strands Agents Framework](https://strandsagents.com/)
- [GitHub Issues](https://github.com/strands-agents/tools/issues)