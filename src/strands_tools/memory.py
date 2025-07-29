"""Tool for managing data in Bedrock Knowledge Base (store, delete, list, get, and retrieve)

This module provides comprehensive Knowledge Base management capabilities for
Amazon Bedrock Knowledge Bases. It handles all aspects of document management with
a user-friendly interface and proper error handling.

Key Features:
------------
1. Content Management:
   • store: Add new content with automatic ID generation and metadata
   • delete: Remove existing documents using document IDs
   • list: Retrieve all documents with optional pagination
   • get: Retrieve specific documents by document ID
   • retrieve: Perform semantic search across all documents

2. Data Source Support:
   • Detects CUSTOM data source types
   • Falls back to first available data source if no CUSTOM found
   • Provides clear error messages for unsupported data source types
   • Currently supports CUSTOM data sources for direct ingestion
   • S3 and other data source types show clear error messages

3. Safety Features:
   • User confirmation for mutative operations
   • Content previews before storage
   • Warning messages before deletion
   • BYPASS_TOOL_CONSENT mode for bypassing confirmations in tests

4. Advanced Capabilities:
   • Automatic document ID generation
   • Structured content storage with metadata
   • Semantic search with relevance filtering
   • Rich output formatting
   • Pagination support

5. Error Handling:
   • Knowledge Base ID validation
   • Parameter validation
   • Data source type detection and validation
   • Graceful API error handling
   • Clear error messages

Usage Examples:
--------------
```python
from strands import Agent
from strands_tools.memory import memory

agent = Agent(tools=[memory])

# Store content in Knowledge Base
agent.tool.memory(
    action="store",
    content="Important information to remember",
    title="Meeting Notes",
    STRANDS_KNOWLEDGE_BASE_ID="my1234kb"
)

# Retrieve content using semantic search
agent.tool.memory(
    action="retrieve",
    query="meeting information",
    min_score=0.7,
    STRANDS_KNOWLEDGE_BASE_ID="my1234kb"
)

# List all documents
agent.tool.memory(
    action="list",
    max_results=50,
    STRANDS_KNOWLEDGE_BASE_ID="my1234kb"
)
```

Notes:
-----
Knowledge base IDs must contain only alphanumeric characters (no hyphens or special characters).
ENV variable STRANDS_KNOWLEDGE_BASE_ID can be used instead of passing the ID to each call.
"""

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config as BotocoreConfig
from rich.panel import Panel
from strands import tool

from strands_tools.utils import console_util
from strands_tools.utils.user_input import get_user_input

# Set up logging
logger = logging.getLogger(__name__)


class MemoryServiceClient:
    """
    Client for interacting with Bedrock Knowledge Base service.

    This client handles all API interactions with AWS Bedrock Knowledge Bases,
    including document storage, retrieval, listing, and deletion. It provides
    a simplified interface for common operations and handles session management.

    Attributes:
        region: AWS region where the Knowledge Base is located
        profile_name: Optional AWS profile name for credentials
        session: The boto3 session used for API calls
    """

    def __init__(self, region: str = None, profile_name: Optional[str] = None):
        """
        Initialize the memory service client.

        Args:
            region: AWS region name (defaults to AWS_REGION env var or "us-west-2")
            profile_name: Optional AWS profile name for credentials
        """
        self.region = region or os.getenv("AWS_REGION", "us-west-2")
        self.profile_name = profile_name
        self._agent_client = None
        self._runtime_client = None

        # Set up session if profile is provided
        if profile_name:
            self.session = boto3.Session(profile_name=profile_name)
        else:
            self.session = boto3.Session()

    @property
    def agent_client(self):
        """
        Lazy-loaded agent client for Bedrock Agent API.

        Returns:
            boto3.client: A boto3 client for the bedrock-agent service
        """
        if not self._agent_client:
            config = BotocoreConfig(user_agent_extra="strands-agents-memory")
            self._agent_client = self.session.client("bedrock-agent", region_name=self.region, config=config)
        return self._agent_client

    @property
    def runtime_client(self):
        """
        Lazy-loaded runtime client for Bedrock Agent Runtime API.

        Returns:
            boto3.client: A boto3 client for the bedrock-agent-runtime service
        """
        if not self._runtime_client:
            config = BotocoreConfig(user_agent_extra="strands-agents-memory")
            self._runtime_client = self.session.client("bedrock-agent-runtime", region_name=self.region, config=config)
        return self._runtime_client

    def _detect_data_source_type(self, kb_id: str):
        """
        Helper method to detect data source type for a knowledge base.

        This method implements the same logic as store_in_kb tool:
        1. Look for CUSTOM data source first (preferred)
        2. Fall back to first available data source if no CUSTOM found
        3. Log appropriate messages for debugging

        Args:
            kb_id: Knowledge Base ID

        Returns:
            Tuple of (data_source_id, source_type)

        Raises:
            ValueError: If no data sources are found
        """
        # Get data source details to determine the type
        data_sources = self.agent_client.list_data_sources(knowledgeBaseId=kb_id)

        if data_sources and not data_sources.get("dataSourceSummaries"):
            raise ValueError(f"No data sources found for knowledge base {kb_id}")

        # Look for CUSTOM data source first, then fallback
        data_source_id = None
        source_type = None

        for ds in data_sources["dataSourceSummaries"]:
            # Get the data source details to check its type
            ds_detail = self.agent_client.get_data_source(knowledgeBaseId=kb_id, dataSourceId=ds["dataSourceId"])

            # Check if this is a CUSTOM type data source
            if ds_detail["dataSource"]["dataSourceConfiguration"]["type"] == "CUSTOM":
                data_source_id = ds["dataSourceId"]
                source_type = "CUSTOM"
                logger.debug(f"Found CUSTOM data source: {data_source_id}")
                break

        # If no CUSTOM data source found, use the first available one but log a warning
        if not data_source_id and data_sources["dataSourceSummaries"]:
            data_source_id = data_sources["dataSourceSummaries"][0]["dataSourceId"]
            ds_detail = self.agent_client.get_data_source(knowledgeBaseId=kb_id, dataSourceId=data_source_id)
            source_type = ds_detail["dataSource"]["dataSourceConfiguration"]["type"]
            logger.debug(f"No CUSTOM data source found. Using {source_type} data source: {data_source_id}")

        if not data_source_id:
            raise ValueError(f"No suitable data source found for knowledge base {kb_id}")

        return data_source_id, source_type

    def get_data_source_id(self, kb_id: str) -> str:
        """
        Get the data source ID for a knowledge base.

        Args:
            kb_id: Knowledge Base ID

        Returns:
            The data source ID string

        Raises:
            ValueError: If no data sources are found for the knowledge base
        """
        data_sources = self.agent_client.list_data_sources(knowledgeBaseId=kb_id)
        if not data_sources.get("dataSourceSummaries"):
            raise ValueError(f"No data sources found for knowledge base {kb_id}")
        return data_sources["dataSourceSummaries"][0]["dataSourceId"]

    def list_documents(
        self,
        kb_id: str,
        data_source_id: str = None,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
    ):
        """
        List documents in the knowledge base.

        Args:
            kb_id: Knowledge Base ID
            data_source_id: Optional data source ID (will be retrieved if not provided)
            max_results: Maximum number of results to return
            next_token: Pagination token for subsequent requests

        Returns:
            Response from the list_knowledge_base_documents API call
        """
        # Get the data source ID if not provided
        if not data_source_id:
            data_source_id = self.get_data_source_id(kb_id)

        # Build parameters for the list_knowledge_base_documents call
        params = {"knowledgeBaseId": kb_id, "dataSourceId": data_source_id}

        if max_results:
            params["maxResults"] = max_results

        if next_token:
            params["nextToken"] = next_token

        return self.agent_client.list_knowledge_base_documents(**params)

    def get_document(self, kb_id: str, data_source_id: str = None, document_id: str = None):
        """
        Get a document by ID.

        Args:
            kb_id: Knowledge Base ID
            data_source_id: Optional data source ID (will be retrieved if not provided)
            document_id: ID of the document to retrieve

        Returns:
            Response from the get_knowledge_base_documents API call
        """
        # Use helper method to detect data source type
        data_source_id, source_type = self._detect_data_source_type(kb_id)

        # Prepare get request based on the data source type
        if source_type == "CUSTOM":
            get_request = {
                "knowledgeBaseId": kb_id,
                "dataSourceId": data_source_id,
                "documentIdentifiers": [{"dataSourceType": "CUSTOM", "custom": {"id": document_id}}],
            }
        elif source_type == "S3":
            # For S3, we would need to construct the S3 URI identifier
            # This is more complex and may require additional logic
            raise ValueError("S3 data source type is not fully supported for document retrieval.")
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")

        return self.agent_client.get_knowledge_base_documents(**get_request)

    def store_document(self, kb_id: str, data_source_id: str = None, content: str = None, title: str = None):
        """
        Store a document in the knowledge base.

        Args:
            kb_id: Knowledge Base ID
            data_source_id: Optional data source ID (will be retrieved if not provided)
            content: Document content to store
            title: Optional document title

        Returns:
            Tuple of (response, document_id, document_title)
        """
        # Generate document ID with timestamp for traceability
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        doc_id = f"memory_{timestamp}_{str(uuid.uuid4())[:8]}"

        # Create a document title if not provided
        doc_title = title or f"Strands Memory {timestamp}"

        # Package content with metadata for better organization
        content_with_metadata = {
            "title": doc_title,
            "action": "store",
            "content": content,
        }

        # Use helper method to detect data source type
        data_source_id, source_type = self._detect_data_source_type(kb_id)

        # Prepare document for ingestion based on the data source type
        if source_type == "CUSTOM":
            ingest_request = {
                "knowledgeBaseId": kb_id,
                "dataSourceId": data_source_id,
                "documents": [
                    {
                        "content": {
                            "dataSourceType": "CUSTOM",
                            "custom": {
                                "customDocumentIdentifier": {"id": doc_id},
                                "inlineContent": {
                                    "textContent": {"data": json.dumps(content_with_metadata)},
                                    "type": "TEXT",
                                },
                                "sourceType": "IN_LINE",
                            },
                        }
                    }
                ],
            }
        elif source_type == "S3":
            # S3 source types need a different ingestion approach
            raise ValueError("S3 data source type is not supported for direct ingestion with this tool.")
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")

        # Ingest document into knowledge base
        response = self.agent_client.ingest_knowledge_base_documents(**ingest_request)

        # Log success
        logger.debug(f"Successfully ingested document into knowledge base {kb_id}: {doc_id}")

        return response, doc_id, doc_title

    def delete_document(self, kb_id: str, data_source_id: str = None, document_id: str = None):
        """
        Delete a document from the knowledge base.
        FIXED: Now handles multiple data source types like store_in_kb tool.

        Args:
            kb_id: Knowledge Base ID
            data_source_id: Optional data source ID (will be retrieved if not provided)
            document_id: ID of the document to delete

        Returns:
            Response from the delete_knowledge_base_documents API call
        """
        # Use helper method to detect data source type
        data_source_id, source_type = self._detect_data_source_type(kb_id)

        # Prepare delete request based on the data source type
        if source_type == "CUSTOM":
            delete_request = {
                "knowledgeBaseId": kb_id,
                "dataSourceId": data_source_id,
                "documentIdentifiers": [{"dataSourceType": "CUSTOM", "custom": {"id": document_id}}],
            }
        elif source_type == "S3":
            # For S3, we would need to construct the S3 URI identifier
            # This is more complex and may require additional logic
            raise ValueError("S3 data source type is not fully supported for document deletion.")
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")

        # Delete document from knowledge base
        return self.agent_client.delete_knowledge_base_documents(**delete_request)

    def retrieve(self, kb_id: str, query: str, max_results: int = 5, next_token: str = None):
        """
        Retrieve documents based on search query.

        Args:
            kb_id: Knowledge Base ID
            query: Search query text
            max_results: Maximum number of results to return
            next_token: Pagination token for subsequent requests

        Returns:
            Response from the retrieve API call
        """
        # Always include retrievalConfiguration with a default from environment if not specified
        params = {
            "retrievalQuery": {"text": query},
            "knowledgeBaseId": kb_id,
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {"numberOfResults": max_results},
            },
        }

        # Add pagination token if provided
        if next_token:
            params["nextToken"] = next_token

        return self.runtime_client.retrieve(**params)


class MemoryFormatter:
    """
    Formats memory tool responses for display.

    This class handles formatting the raw API responses into user-friendly
    output with proper structure, emoji indicators, and readable formatting.
    Each method corresponds to a specific action type's response format.
    """

    def format_list_response(self, response: Dict) -> List[Dict]:
        """
        Format list documents response.

        Args:
            response: Raw API response from list_knowledge_base_documents

        Returns:
            List of formatted content dictionaries for display
        """
        content = []
        document_details = response.get("documentDetails", [])

        if not document_details:
            content.append({"text": "No documents found."})
            return content

        result_text = f"Found {len(document_details)} documents:"

        for i, doc in enumerate(document_details, 1):
            doc_id = None
            # Extract document ID based on the identifier structure
            if doc.get("identifier") and doc["identifier"].get("custom"):
                doc_id = doc["identifier"]["custom"].get("id")
            elif doc.get("identifier") and doc["identifier"].get("s3"):
                doc_id = doc["identifier"]["s3"].get("uri")

            if doc_id:
                status = doc.get("status", "UNKNOWN")
                updated_at = doc.get("updatedAt", "Unknown")
                result_text += f"\n{i}. 🔖 ID: {doc_id}"
                result_text += f"\n   📊 Status: {status}"
                result_text += f"\n   🕒 Updated: {updated_at}"

        content.append({"text": result_text})

        # Add next token if available
        if "nextToken" in response:
            content.append({"text": "➡️ More results available. Use next_token parameter to continue."})
            content.append({"text": f"next_token: {response['nextToken']}"})

        return content

    def format_get_response(self, document_id: str, kb_id: str, content_data: Dict) -> List[Dict]:
        """
        Format get document response.

        Args:
            document_id: ID of the retrieved document
            kb_id: Knowledge Base ID
            content_data: Parsed content data from the document

        Returns:
            List of formatted content dictionaries for display
        """
        result = [
            {"text": "✅ Document retrieved successfully:"},
            {"text": f"📝 Title: {content_data.get('title', 'Unknown')}"},
            {"text": f"🔑 Document ID: {document_id}"},
            {"text": f"🗄️ Knowledge Base ID: {kb_id}"},
            {"text": f"\n📄 Content:\n\n{content_data.get('content', 'No content available')}"},
        ]
        return result

    def format_store_response(self, doc_id: str, kb_id: str, title: str) -> List[Dict]:
        """
        Format store document response.

        Args:
            doc_id: ID of the newly stored document
            kb_id: Knowledge Base ID
            title: Title of the stored document

        Returns:
            List of formatted content dictionaries for display
        """
        content = [
            {"text": "✅ Successfully stored content in knowledge base:"},
            {"text": f"📝 Title: {title}"},
            {"text": f"🔑 Document ID: {doc_id}"},
            {"text": f"🗄️ Knowledge Base ID: {kb_id}"},
        ]
        return content

    def format_delete_response(self, status: str, doc_id: str, kb_id: str) -> List[Dict]:
        """
        Format delete document response.

        Args:
            status: Status of the deletion operation
            doc_id: ID of the deleted document
            kb_id: Knowledge Base ID

        Returns:
            List of formatted content dictionaries for display
        """
        if status in ["DELETED", "DELETING", "DELETE_IN_PROGRESS"]:
            content = [
                {"text": f"✅ Document deletion {status.lower().replace('_', ' ')}:"},
                {"text": f"🔑 Document ID: {doc_id}"},
                {"text": f"🗄️ Knowledge Base ID: {kb_id}"},
            ]
        else:
            content = [
                {"text": f"❌ Document deletion failed with status: {status}"},
                {"text": f"🔑 Document ID: {doc_id}"},
                {"text": f"🗄️ Knowledge Base ID: {kb_id}"},
            ]
        return content

    def format_retrieve_response(self, response: Dict, min_score: float = 0.0) -> List[Dict]:
        """
        Format retrieve response.

        Args:
            response: Raw API response from retrieve
            min_score: Minimum relevance score threshold for filtering results

        Returns:
            List of formatted content dictionaries for display
        """
        content = []
        results = response.get("retrievalResults", [])

        # Filter by score
        filtered_results = [r for r in results if r.get("score", 0) >= min_score]

        if not filtered_results:
            content.append({"text": "No results found that meet the score threshold."})
            return content

        result_text = f"Retrieved {len(filtered_results)} results with score >= {min_score}:"

        for result in filtered_results:
            score = result.get("score", 0)
            doc_id = "unknown"
            text = "No content available"
            title = None

            # Extract document ID
            if "location" in result and "customDocumentLocation" in result["location"]:
                doc_id = result["location"]["customDocumentLocation"].get("id", "unknown")

            # Extract content text
            if "content" in result and "text" in result["content"]:
                text = result["content"]["text"]

            result_text += f"\n\nScore: {score:.4f}"
            result_text += f"\nDocument ID: {doc_id}"

            # Try to parse content as JSON for better display
            try:
                if text.strip().startswith("{"):
                    content_obj = json.loads(text)
                    if isinstance(content_obj, dict) and "title" in content_obj:
                        title = content_obj.get("title")
                        result_text += f"\nTitle: {title}"
            except json.JSONDecodeError:
                pass

            # Add content preview
            preview = text[:150]
            if len(text) > 150:
                preview += "..."
            result_text += f"\nContent Preview: {preview}"

        content.append({"text": result_text})

        # Add next token if available
        if "nextToken" in response:
            content.append({"text": "\n➡️ More results available. Use next_token parameter to continue."})
            content.append({"text": f"next_token: {response['nextToken']}"})

        return content


# Factory functions for dependency injection
def get_memory_service_client(region: str = None, profile_name: str = None) -> MemoryServiceClient:
    """
    Factory function to create a memory service client.

    This function can be mocked in tests for better testability.

    Args:
        region: Optional AWS region
        profile_name: Optional AWS profile name

    Returns:
        An initialized MemoryServiceClient instance
    """
    return MemoryServiceClient(region=region, profile_name=profile_name)


def get_memory_formatter() -> MemoryFormatter:
    """
    Factory function to create a memory formatter.

    This function can be mocked in tests for better testability.

    Returns:
        An initialized MemoryFormatter instance
    """
    return MemoryFormatter()


@tool
def memory(
    action: str,
    content: Optional[str] = None,
    title: Optional[str] = None,
    document_id: Optional[str] = None,
    query: Optional[str] = None,
    STRANDS_KNOWLEDGE_BASE_ID: Optional[str] = None,
    max_results: int = None,
    next_token: Optional[str] = None,
    min_score: float = None,
    region_name: str = None,
) -> Dict[str, Any]:
    """
    Manage content in a Bedrock Knowledge Base (store, delete, list, get, or retrieve).

    This tool provides a user-friendly interface for managing knowledge base content
    with built-in safety measures for mutative operations. For operations that modify
    data (store, delete), users will be shown a preview and asked for explicit confirmation
    before changes are made, unless the BYPASS_TOOL_CONSENT environment variable is set to "true".

    Args:
        action: The action to perform ('store', 'delete', 'list', 'get', or 'retrieve').
        content: The text content to store in the knowledge base (required for 'store' action).
        title: Optional title for the content when storing. If not provided, a timestamp will be used.
        document_id: The ID of the document to delete or get (required for 'delete' and 'get' actions).
        STRANDS_KNOWLEDGE_BASE_ID: Optional knowledge base ID. If not provided, will use the
            STRANDS_KNOWLEDGE_BASE_ID env variable. Note: Knowledge base ID must match pattern
            [0-9a-zA-Z]+ (alphanumeric characters only).
        max_results: Maximum number of results to return for 'list' or 'retrieve' action (default: 50, max: 1000).
        next_token: Token for pagination in 'list' or 'retrieve' action (optional).
        query: The search query for semantic search (required for 'retrieve' action).
        min_score: Minimum relevance score threshold (0.0-1.0) for 'retrieve' action. Default is 0.4.
        region_name: Optional AWS region name. If not provided, will use the AWS_REGION env variable.
            If AWS_REGION is not specified, it will default to us-west-2.

    Returns:
        A dictionary containing the result of the operation.

    Notes:
        - Store and delete operations require user confirmation (unless in BYPASS_TOOL_CONSENT mode)
        - Content previews are shown before storage to verify accuracy
        - Warning messages are provided before document deletion
        - Operation can be cancelled by the user during confirmation
        - Retrieve provides semantic search across all documents in the knowledge base
        - Knowledge base IDs must contain only alphanumeric characters (no hyphens or special characters)
    """
    console = console_util.create()

    # Initialize the client and formatter using factory functions
    client = get_memory_service_client(region=region_name)
    formatter = get_memory_formatter()

    # Get environment variables at runtime
    max_results = int(os.getenv("MEMORY_DEFAULT_MAX_RESULTS", "50")) if max_results is None else max_results
    min_score = float(os.getenv("MEMORY_DEFAULT_MIN_SCORE", "0.4")) if min_score is None else min_score
    kb_id = STRANDS_KNOWLEDGE_BASE_ID or os.getenv("STRANDS_KNOWLEDGE_BASE_ID")

    # Validate required inputs
    if not kb_id:
        return {
            "status": "error",
            "content": [
                {"text": "❌ No knowledge base ID provided or found in environment variables STRANDS_KNOWLEDGE_BASE_ID"}
            ],
        }

    # Validate action
    if action not in ["store", "delete", "list", "get", "retrieve"]:
        return {
            "status": "error",
            "content": [
                {"text": f"❌ Invalid action: {action}. Must be 'store', 'delete', 'list', 'get', or 'retrieve'"}
            ],
        }

    # Try to validate KB ID format
    if not re.match(r"^[0-9a-zA-Z]+$", kb_id):
        return {
            "status": "error",
            "content": [
                {"text": f"❌ Invalid knowledge base ID format: '{kb_id}'"},
                {
                    "text": "Knowledge base IDs must contain only alphanumeric characters (no hyphens or special "
                    "characters)"
                },
            ],
        }

    # Try to get the data source ID associated with the knowledge base
    data_source_id = None
    try:
        data_source_id = client.get_data_source_id(kb_id)
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Failed to get data source ID: {str(e)}"}],
        }

    # Define mutative actions that need confirmation
    mutative_actions = {"store", "delete"}
    strands_dev = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"
    needs_confirmation = action in mutative_actions and not strands_dev

    # Show confirmation dialog for mutative operations
    if needs_confirmation:
        if action == "store":
            # Validate content
            if not content or not content.strip():
                return {"status": "error", "content": [{"text": "❌ Content cannot be empty"}]}

            # Preview what will be stored
            doc_title = title or f"Memory {time.strftime('%Y%m%d_%H%M%S')}"
            content_preview = content[:15000] + "..." if len(content) > 15000 else content

            console.print(Panel(content_preview, title=f"[bold green]{doc_title}", border_style="green"))

        elif action == "delete":
            # Validate document_id
            if not document_id:
                return {"status": "error", "content": [{"text": "❌ Document ID cannot be empty for delete operation"}]}

            # Try to get document info first for better context
            try:
                get_response = client.get_document(kb_id, data_source_id, document_id)
                document_details = get_response.get("documentDetails", [])
                document_status = document_details[0].get("status", "UNKNOWN") if document_details else "UNKNOWN"

                # For better context, try to get title if possible
                title_info = ""
                try:
                    retrieval_result = client.retrieve(
                        kb_id=kb_id,
                        query=f"documentId:{document_id}",
                        # Explicitly set max_results to ensure retrievalConfiguration is included
                        max_results=max_results,
                    )

                    retrieved_results = retrieval_result.get("retrievalResults", [])
                    if retrieved_results:
                        result = retrieved_results[0]
                        text = result.get("content", {}).get("text", "")
                        try:
                            content_data = json.loads(text) if text.strip().startswith("{") else {}
                            if "title" in content_data:
                                title_info = f"\nTitle: {content_data['title']}"
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    # Ignore errors in title retrieval
                    pass

                console.print(
                    Panel(
                        f"Document ID: {document_id}{title_info}\nKnowledge Base: {kb_id}\nStatus: {document_status}",
                        title="[bold red]⚠️ Document to be permanently deleted",
                        border_style="red",
                    )
                )
            except Exception:
                # Fall back to basic info if we can't get document details
                console.print(
                    Panel(
                        f"Document ID: {document_id}\nKnowledge Base: {kb_id}",
                        title="[bold red]⚠️ Document to be permanently deleted",
                        border_style="red",
                    )
                )

        # Get user confirmation
        user_input = get_user_input(
            f"<yellow><bold>Do you want to proceed with the {action} operation?</bold> [y/*]</yellow>"
        )
        if user_input.lower().strip() != "y":
            cancellation_reason = (
                user_input if user_input.strip() != "n" else get_user_input("Please provide a reason for cancellation:")
            )
            error_message = f"Operation cancelled by the user. Reason: {cancellation_reason}"
            return {
                "status": "error",
                "content": [{"text": error_message}],
            }

    # Validate action-specific requirements before making API calls
    try:
        if action == "store":
            # Validate content if not already done in confirmation step
            if not needs_confirmation and (not content or not content.strip()):
                return {"status": "error", "content": [{"text": "❌ Content cannot be empty"}]}

            # Generate a title if none provided
            store_title = title
            if not store_title:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                store_title = f"Memory Entry {timestamp}"

            # Store the document
            _, doc_id, doc_title = client.store_document(kb_id, data_source_id, content, store_title)
            formatted_content = formatter.format_store_response(doc_id, kb_id, doc_title)
            return {"status": "success", "content": formatted_content}

        elif action == "delete":
            # Validate document_id if not already done in confirmation step
            if not needs_confirmation and not document_id:
                return {"status": "error", "content": [{"text": "❌ Document ID cannot be empty for delete operation"}]}

            # Delete the document
            response = client.delete_document(kb_id, data_source_id, document_id)

            # Check status
            document_details = response.get("documentDetails", [])
            if document_details:
                status = document_details[0].get("status", "UNKNOWN")
                formatted_content = formatter.format_delete_response(status, document_id, kb_id)
                return {"status": "success", "content": formatted_content}

            # If no document details, assume success based on API call completing
            return {
                "status": "success",
                "content": [
                    {"text": "✅ Document deletion request accepted:"},
                    {"text": f"🔑 Document ID: {document_id}"},
                    {"text": f"🗄️ Knowledge Base ID: {kb_id}"},
                ],
            }

        elif action == "get":
            # Validate document_id
            if not document_id:
                return {"status": "error", "content": [{"text": "❌ Document ID cannot be empty for get operation"}]}

            try:
                # Get document
                response = client.get_document(kb_id, data_source_id, document_id)

                # Check if document exists
                document_details = response.get("documentDetails", [])
                if not document_details:
                    return {"status": "error", "content": [{"text": f"❌ Document not found: {document_id}"}]}

                # Get the first document detail
                document_detail = document_details[0]
                status = document_detail.get("status", "UNKNOWN")

                # Check if document is indexed
                if status != "INDEXED":
                    # If document exists but isn't indexed yet, we can try a few retries
                    # This helps when document was just created and is still being processed
                    max_retries = 3
                    retry_delay = 2  # seconds

                    for _retry in range(max_retries):
                        # Wait before retry
                        time.sleep(retry_delay)

                        # Check status again
                        retry_response = client.get_document(kb_id, data_source_id, document_id)
                        retry_details = retry_response.get("documentDetails", [])

                        if retry_details and retry_details[0].get("status") == "INDEXED":
                            # Document is now indexed, proceed with retrieval
                            status = "INDEXED"
                            break

                    # If still not indexed after retries
                    if status != "INDEXED":
                        return {
                            "status": "error",
                            "content": [
                                {"text": f"❌ Document is not indexed (status: {status}):"},
                                {"text": f"🔑 Document ID: {document_id}"},
                                {"text": f"🗄️ Knowledge Base ID: {kb_id}"},
                            ],
                        }

                # Query for document content using retrieve
                try:
                    # First try using documentId prefix which is the most accurate way
                    retrieval_result = client.retrieve(
                        kb_id=kb_id,
                        query=f"documentId:{document_id}",
                        # Explicitly set max_results to ensure retrievalConfiguration is included
                        max_results=max_results,
                    )

                    # Check if we got results
                    retrieved_results = retrieval_result.get("retrievalResults", [])

                    # If first query fails, try alternative queries
                    if not retrieved_results:
                        # Try with the raw ID
                        alt_retrieval_result = client.retrieve(
                            kb_id=kb_id,
                            query=document_id,
                            max_results=max_results,  # Use a higher value to increase chances
                        )
                        retrieved_results = alt_retrieval_result.get("retrievalResults", [])

                        # Filter for exact document ID match
                        if retrieved_results:
                            matching_results = []
                            for result in retrieved_results:
                                result_doc_id = "unknown"
                                if "location" in result and "customDocumentLocation" in result["location"]:
                                    result_doc_id = result["location"]["customDocumentLocation"].get("id", "unknown")

                                if result_doc_id == document_id:
                                    matching_results.append(result)

                            if matching_results:
                                # Use the first match
                                retrieved_results = [matching_results[0]]
                            else:
                                # No exact matches found
                                retrieved_results = []

                    if not retrieved_results:
                        # If no results, the document might be indexed but the content isn't available yet
                        # Try again with a direct retrieve using a more general query to improve chances of match
                        try:
                            # Try a more general retrieval approach
                            alt_query = f"id:{document_id}"
                            alt_retrieval_result = client.retrieve(
                                kb_id=kb_id,
                                query=alt_query,
                                # Try a slightly higher max_results to increase chances of finding it
                                max_results=max_results,
                            )
                            alt_results = alt_retrieval_result.get("retrievalResults", [])
                            if alt_results:
                                # We found some results with the alternative approach
                                for alt_result in alt_results:
                                    alt_doc_id = "unknown"
                                    if "location" in alt_result and "customDocumentLocation" in alt_result["location"]:
                                        alt_doc_id = alt_result["location"]["customDocumentLocation"].get(
                                            "id", "unknown"
                                        )

                                    if alt_doc_id == document_id:
                                        # Found the right document
                                        result = alt_result
                                        text = result.get("content", {}).get("text", "")
                                        # Continue with processing this result
                                        break
                                else:
                                    # Didn't find the document in the results
                                    return {
                                        "status": "error",
                                        "content": [
                                            {
                                                "text": f"❌ Document found but content could not be retrieved: "
                                                f"{document_id}"
                                            }
                                        ],
                                    }
                            else:
                                return {
                                    "status": "error",
                                    "content": [
                                        {"text": f"❌ Document found but content could not be retrieved: {document_id}"}
                                    ],
                                }
                        except Exception:
                            # If the alternate approach fails, return the original error
                            return {
                                "status": "error",
                                "content": [
                                    {"text": f"❌ Document found but content could not be retrieved: {document_id}"}
                                ],
                            }

                    # Extract content
                    result = retrieved_results[0]
                    text = result.get("content", {}).get("text", "")

                    try:
                        # Try to parse as JSON if it looks like our format
                        content_data = json.loads(text) if text.strip().startswith("{") else {"content": text}

                        if "title" in content_data and "content" in content_data:
                            return {
                                "status": "success",
                                "content": formatter.format_get_response(document_id, kb_id, content_data),
                            }
                        else:
                            return {
                                "status": "success",
                                "content": [
                                    {"text": "✅ Document retrieved successfully:"},
                                    {"text": f"🔑 Document ID: {document_id}"},
                                    {"text": f"🗄️ Knowledge Base ID: {kb_id}"},
                                    {"text": f"\n📄 Content:\n\n{text}"},
                                ],
                            }
                    except json.JSONDecodeError:
                        # If not JSON, return raw content
                        return {
                            "status": "success",
                            "content": [
                                {"text": "✅ Document retrieved successfully:"},
                                {"text": f"🔑 Document ID: {document_id}"},
                                {"text": f"🗄️ Knowledge Base ID: {kb_id}"},
                                {"text": f"\n📄 Content:\n\n{text}"},
                            ],
                        }
                except Exception as e:
                    return {"status": "error", "content": [{"text": f"❌ Error retrieving document content: {str(e)}"}]}

            except Exception as e:
                return {"status": "error", "content": [{"text": f"❌ Error retrieving document: {str(e)}"}]}

        elif action == "list":
            # Validate max_results
            if max_results < 1 or max_results > 1000:
                return {"status": "error", "content": [{"text": "❌ max_results must be between 1 and 1000"}]}

            response = client.list_documents(kb_id, data_source_id, max_results, next_token)
            formatted_content = formatter.format_list_response(response)

            result = {
                "status": "success",
                "content": formatted_content,
            }

            # Handle next_token properly (embed it in content instead of adding directly to result)
            if "nextToken" in response:
                # The next token is already included in the formatted_content
                pass

            return result

        elif action == "retrieve":
            if not query:
                return {"status": "error", "content": [{"text": "❌ No query provided for retrieval."}]}

            # Validate parameters
            if min_score < 0.0 or min_score > 1.0:
                return {"status": "error", "content": [{"text": "❌ min_score must be between 0.0 and 1.0"}]}

            if max_results < 1 or max_results > 1000:
                return {"status": "error", "content": [{"text": "❌ max_results must be between 1 and 1000"}]}

            # Set default max results if not provided
            if max_results is None:
                max_results = 5

            try:
                # Perform retrieval
                response = client.retrieve(kb_id=kb_id, query=query, max_results=max_results, next_token=next_token)

                # Format and filter response
                formatted_content = formatter.format_retrieve_response(response, min_score)

                result = {
                    "status": "success",
                    "content": formatted_content,
                }

                return result

            except Exception as e:
                error_msg = str(e).lower()
                if "validationexception" in error_msg and "knowledgebaseid" in error_msg:
                    return {
                        "status": "error",
                        "content": [
                            {"text": f"❌ Invalid knowledge base ID format: '{kb_id}'"},
                            {
                                "text": "Knowledge base IDs must contain only alphanumeric characters "
                                "(no hyphens or special characters)"
                            },
                        ],
                    }
                return {"status": "error", "content": [{"text": f"❌ Error during retrieval: {str(e)}"}]}

    except Exception as e:
        return {"status": "error", "content": [{"text": f"❌ Error during {action} operation: {str(e)}"}]}
