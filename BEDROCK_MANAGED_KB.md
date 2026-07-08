# Bedrock Managed Knowledge Base Support

## Changes
- Updated Bedrock Knowledge Base tool/retriever to default to `managedSearchConfiguration`
- Added `KNOWLEDGE_BASE_TYPE` environment variable to control retrieval mode
- Tool retrieval logic branches on type: MANAGED uses `managedSearchConfiguration`, VECTOR uses `vectorSearchConfiguration`
- Added `AgenticRetrieveStream` path when `USE_AGENTIC_RETRIEVAL=true`
- All existing VECTOR retrieval paths unchanged

## Design
- VECTOR is the default; MANAGED via `KNOWLEDGE_BASE_TYPE=MANAGED` env var
- AgenticRetrieveStream for agentic retrieval with managed reranking
- Backward compatible: existing VECTOR configurations continue working
- Configuration via environment variables for deployment flexibility

## API Shapes
- KB Creation: `type: MANAGED` + `managedKnowledgeBaseConfiguration.embeddingModelType: MANAGED`
- Retrieval: `managedSearchConfiguration` (not `vectorSearchConfiguration`)
- Agentic: `AgenticRetrieveStream` with `foundationModelType: MANAGED`, `rerankingModelType: MANAGED`

## Configuration
| Variable | Description | Default |
|---|---|---|
| KNOWLEDGE_BASE_TYPE | MANAGED or VECTOR | VECTOR |
| USE_AGENTIC_RETRIEVAL | Enable agentic retrieval | true |
| KNOWLEDGE_BASE_ID | KB identifier | (required) |

## SDK Requirements
- boto3 >= 1.43 for managed search and agentic retrieval

## Required IAM Permissions
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:Retrieve",
    "bedrock:AgenticRetrieve"
  ],
  "Resource": "arn:aws:bedrock:<region>:<account-id>:knowledge-base/<kb-id>"
}
```
