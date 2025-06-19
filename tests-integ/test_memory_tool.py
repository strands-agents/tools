"""
Integration test for the Bedrock Knowledge Base (memory) tool.

This test creates real AWS resources (IAM Role, OpenSearch Collection, Bedrock KB)
to validate the end-to-end functionality of the memory tool. It is designed to
be resilient and re-use existing resources to speed up local development.

Prerequisites:
- AWS credentials must be configured in the environment.
- The AWS identity must have permissions to manage IAM, OpenSearch Serverless,
  and Bedrock KB resources.

Configuration:
- STRANDS_TEARDOWN_RESOURCES (env var): Set to "false" to prevent the automatic
  deletion of AWS resources after the test run. This is useful for speeding
  up local development by re-using the same Knowledge Base.
"""

import json
import logging
import os
import time
import uuid
from unittest.mock import patch

import boto3
import pytest
from botocore.exceptions import ClientError
from opensearchpy import (
    AuthenticationException,
    AuthorizationException,
    AWSV4SignerAuth,
    OpenSearch,
    RequestsHttpConnection,
)
from strands import Agent
from strands_tools import memory

logger = logging.getLogger(__name__)

AWS_REGION = "us-east-1"
# Use a standard embedding model for the KB
EMBEDDING_MODEL_ARN = f"arn:aws:bedrock:{AWS_REGION}::foundation-model/amazon.titan-embed-text-v1"
EMBEDDING_DIMENSION = 1536  # Dimension for amazon.titan-embed-text-v1


def _get_boto_clients():
    """Returns a dictionary of boto3 clients needed for the test."""
    return {
        "iam": boto3.client("iam", region_name=AWS_REGION),
        "bedrock-agent": boto3.client("bedrock-agent", region_name=AWS_REGION),
        "opensearchserverless": boto3.client("opensearchserverless", region_name=AWS_REGION),
        "sts": boto3.client("sts", region_name=AWS_REGION),
    }


def _wait_for_resource(
    poll_function,
    resource_name,
    success_status="ACTIVE",
    failure_status="FAILED",
    timeout_seconds=600,
    delay_seconds=30,
):
    """Generic waiter for AWS asynchronous operations with detailed logging."""
    logger.info(f"Waiting for '{resource_name}' to become '{success_status}'...")
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            response = poll_function()
            status = response.get("status") or response.get("Status")
            logger.info(f"Polling status for '{resource_name}': {status} (Elapsed: {int(time.time() - start_time)}s)")
            if status == success_status:
                logger.info(f"SUCCESS: Resource '{resource_name}' reached '{success_status}' state.")
                return response
            if status == failure_status:
                logger.error(
                    f"FAILURE: Resource '{resource_name}' entered failure state: {failure_status}-Response: {response}"
                )
                raise Exception(f"Resource '{resource_name}' entered failure state: {failure_status}")
        except ClientError as e:
            if "ResourceNotFoundException" not in str(e):
                raise e
            logger.info(f"Resource '{resource_name}' not found yet, continuing to wait...")
        time.sleep(delay_seconds)
    raise TimeoutError(f"Timed out waiting for resource '{resource_name}' to become '{success_status}'.")


@pytest.fixture(scope="module")
def managed_knowledge_base():
    """
    Pytest fixture to create and tear down a Bedrock Knowledge Base and its dependencies.
    It will re-use existing resources if found to speed up local test runs.
    Teardown can be skipped by setting the STRANDS_TEARDOWN_RESOURCES env var to "false".
    """
    clients = _get_boto_clients()

    resource_names = {
        "role_name": "StrandsMemoryIntegTestRole",
        "kb_name": "strands-memory-integ-test-kb",
        "ds_name": "strands-memory-integ-test-ds",
        "policy_name": "StrandsMemoryIntegTestPolicy",
        "collection_name": "strands-memory-integ-coll",
        "enc_policy_name": "strands-memory-enc-policy",
        "net_policy_name": "strands-memory-net-policy",
        "access_policy_name": "strands-memory-access-policy",
        "vector_index_name": "bedrock-kb-index",
    }
    created_resources = {}

    # Check if Knowledge Base already exists
    try:
        kbs = clients["bedrock-agent"].list_knowledge_bases()
        existing_kb = next(
            (kb for kb in kbs.get("knowledgeBaseSummaries", []) if kb["name"] == resource_names["kb_name"]), None
        )
        if existing_kb and existing_kb["status"] == "ACTIVE":
            logger.info(f"Found existing and ACTIVE Knowledge Base '{resource_names['kb_name']}'. Re-using for test.")
            yield existing_kb["knowledgeBaseId"]
            return  # Skip creation if we are reusing
    except ClientError as e:
        logger.error(f"Error checking for existing Knowledge Bases: {e}")
        raise

    try:
        logger.info("No active Knowledge Base found. Creating or validating all resources from scratch...")

        # STEP 1: Create OpenSearch Security Policies
        try:
            clients["opensearchserverless"].create_security_policy(
                name=resource_names["enc_policy_name"],
                type="encryption",
                policy=json.dumps(
                    {
                        "Rules": [
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{resource_names['collection_name']}"],
                            }
                        ],
                        "AWSOwnedKey": True,
                    }
                ),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConflictException":
                raise
            logger.info(f"Encryption policy '{resource_names['enc_policy_name']}' already exists.")

        try:
            clients["opensearchserverless"].create_security_policy(
                name=resource_names["net_policy_name"],
                type="network",
                policy=json.dumps(
                    [
                        {
                            "Rules": [
                                {
                                    "ResourceType": "collection",
                                    "Resource": [f"collection/{resource_names['collection_name']}"],
                                }
                            ],
                            "AllowFromPublic": True,
                        }
                    ]
                ),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConflictException":
                raise
            logger.info(f"Network policy '{resource_names['net_policy_name']}' already exists.")
        time.sleep(10)

        # STEP 2: Create OpenSearch Serverless Collection
        try:
            collection_res = clients["opensearchserverless"].create_collection(
                name=resource_names["collection_name"], type="VECTORSEARCH"
            )
            collection_id = collection_res["createCollectionDetail"]["id"]
            collection_arn = collection_res["createCollectionDetail"]["arn"]
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConflictException":
                raise
            logger.info(f"Collection '{resource_names['collection_name']}' already exists. Fetching details.")
            collection_details = clients["opensearchserverless"].list_collections(
                collectionFilters={"name": resource_names["collection_name"]}
            )["collectionSummaries"][0]
            collection_id = collection_details["id"]
            collection_arn = collection_details["arn"]
        created_resources["collection_id"] = collection_id

        # STEP 3: Create IAM Role and Policies
        try:
            role_res = clients["iam"].get_role(RoleName=resource_names["role_name"])
            logger.info(f"IAM Role '{resource_names['role_name']}' already exists.")
            created_resources["role_arn"] = role_res["Role"]["Arn"]
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchEntity":
                raise
            iam_policy_doc = {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Action": "bedrock:InvokeModel", "Resource": EMBEDDING_MODEL_ARN},
                    {"Effect": "Allow", "Action": "aoss:APIAccessAll", "Resource": collection_arn},
                ],
            }
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Principal": {"Service": "bedrock.amazonaws.com"}, "Action": "sts:AssumeRole"}
                ],
            }
            role_res = clients["iam"].create_role(
                RoleName=resource_names["role_name"], AssumeRolePolicyDocument=json.dumps(assume_role_policy)
            )
            created_resources["role_arn"] = role_res["Role"]["Arn"]
            clients["iam"].put_role_policy(
                RoleName=resource_names["role_name"],
                PolicyName=resource_names["policy_name"],
                PolicyDocument=json.dumps(iam_policy_doc),
            )
            time.sleep(15)

        # STEP 4: Create OpenSearch Data Access Policy
        try:
            user_arn = clients["sts"].get_caller_identity()["Arn"]
            access_policy_doc = [
                {
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [f"collection/{resource_names['collection_name']}"],
                            "Permission": ["aoss:*"],
                        },
                        {
                            "ResourceType": "index",
                            "Resource": [f"index/{resource_names['collection_name']}/*"],
                            "Permission": ["aoss:*"],
                        },
                    ],
                    "Principal": [created_resources["role_arn"], user_arn],
                }
            ]
            clients["opensearchserverless"].create_access_policy(
                name=resource_names["access_policy_name"], type="data", policy=json.dumps(access_policy_doc)
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConflictException":
                raise
            logger.info(f"Access policy '{resource_names['access_policy_name']}' already exists.")

        # STEP 5: Wait for OpenSearch Collection to be Active
        collection_details = _wait_for_resource(
            lambda: clients["opensearchserverless"].batch_get_collection(ids=[collection_id])["collectionDetails"][0],
            resource_name=f"OpenSearch Collection ({collection_id})",
        )

        # STEP 6: Create the vector index
        collection_endpoint = collection_details["collectionEndpoint"]
        host = collection_endpoint.replace("https://", "")
        auth = AWSV4SignerAuth(boto3.Session().get_credentials(), AWS_REGION, "aoss")
        os_client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
        index_body = {
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 512}},
            "mappings": {
                "properties": {
                    "bedrock-kb-vector": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIMENSION,
                        "method": {"name": "hnsw", "engine": "faiss"},
                    },
                    "AMAZON_BEDROCK_TEXT_CHUNK": {"type": "text"},
                    "AMAZON_BEDROCK_METADATA": {"type": "text"},
                }
            },
        }
        if not os_client.indices.exists(resource_names["vector_index_name"]):
            time.sleep(20)  # Wait for access policy to propagate
            for i in range(5):
                try:
                    os_client.indices.create(resource_names["vector_index_name"], body=index_body)
                    logger.info("SUCCESS: Vector index created.")
                    break
                except (AuthenticationException, AuthorizationException) as e:
                    if i < 4:
                        logger.warning(f"Auth error creating index (Attempt {i+1}). Waiting 30s...")
                        time.sleep(30)
                    else:
                        logger.error("Authorization error persisted after multiple retries.")
                        raise e
            time.sleep(10)

        # STEP 7: Create Knowledge Base
        kb_res = clients["bedrock-agent"].create_knowledge_base(
            name=resource_names["kb_name"],
            roleArn=created_resources["role_arn"],
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {"embeddingModelArn": EMBEDDING_MODEL_ARN},
            },
            storageConfiguration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": collection_arn,
                    "vectorIndexName": resource_names["vector_index_name"],
                    "fieldMapping": {
                        "vectorField": "bedrock-kb-vector",
                        "textField": "AMAZON_BEDROCK_TEXT_CHUNK",
                        "metadataField": "AMAZON_BEDROCK_METADATA",
                    },
                },
            },
        )
        created_resources["kb_id"] = kb_res["knowledgeBase"]["knowledgeBaseId"]
        _wait_for_resource(
            lambda: clients["bedrock-agent"].get_knowledge_base(knowledgeBaseId=created_resources["kb_id"])[
                "knowledgeBase"
            ],
            resource_name=f"Knowledge Base ({created_resources['kb_id']})",
        )

        # STEP 8: Create Data Source
        ds_res = clients["bedrock-agent"].create_data_source(
            knowledgeBaseId=created_resources["kb_id"],
            name=resource_names["ds_name"],
            dataSourceConfiguration={"type": "CUSTOM"},
        )
        created_resources["ds_id"] = ds_res["dataSource"]["dataSourceId"]

        #  Do not need to wait for CUSTOM data source to become ACTIVE
        time.sleep(10)

        logger.info("All new resources are ready.")
        yield created_resources["kb_id"]

    finally:
        if os.environ.get("STRANDS_TEARDOWN_RESOURCES", "true").lower() == "true":
            logger.info("Starting teardown of AWS resources...")
            # Use try/except for each deletion to make teardown more resilient
            if "ds_id" in created_resources and "kb_id" in created_resources:
                try:
                    clients["bedrock-agent"].delete_data_source(
                        knowledgeBaseId=created_resources["kb_id"], dataSourceId=created_resources["ds_id"]
                    )
                except ClientError:
                    logger.warning("Could not delete data source.")
            if "kb_id" in created_resources:
                try:
                    clients["bedrock-agent"].delete_knowledge_base(knowledgeBaseId=created_resources["kb_id"])
                    time.sleep(30)
                except ClientError:
                    logger.warning("Could not delete knowledge base.")
            if "collection_id" in created_resources:
                try:
                    clients["opensearchserverless"].delete_collection(id=created_resources["collection_id"])
                except ClientError:
                    logger.warning("Could not delete OpenSearch collection.")
            if "access_policy_name" in resource_names:
                try:
                    clients["opensearchserverless"].delete_access_policy(
                        name=resource_names["access_policy_name"], type="data"
                    )
                except ClientError:
                    logger.warning("Could not delete access policy.")
            if "net_policy_name" in resource_names:
                try:
                    clients["opensearchserverless"].delete_security_policy(
                        name=resource_names["net_policy_name"], type="network"
                    )
                except ClientError:
                    logger.warning("Could not delete network policy.")
            if "enc_policy_name" in resource_names:
                try:
                    clients["opensearchserverless"].delete_security_policy(
                        name=resource_names["enc_policy_name"], type="encryption"
                    )
                except ClientError:
                    logger.warning("Could not delete encryption policy.")
            if "role_name" in resource_names:
                try:
                    clients["iam"].delete_role_policy(
                        RoleName=resource_names["role_name"], PolicyName=resource_names["policy_name"]
                    )
                    clients["iam"].delete_role(RoleName=resource_names["role_name"])
                except ClientError:
                    logger.warning("Could not delete IAM role.")
            logger.info("Teardown complete.")
        else:
            logger.info("Skipping teardown of AWS resources as per STRANDS_TEARDOWN_RESOURCES setting.")


@patch.dict(os.environ, {"BYPASS_TOOL_CONSENT": "true"})
def test_memory_integration_store_and_retrieve(managed_knowledge_base):
    """
    End-to-end test for Bedrock Knowledge Base memory tool:
      - Store a unique document
      - Poll until it is INDEXED
      - Retrieve via semantic search and verify presence
    """
    kb_id = managed_knowledge_base
    agent = Agent(tools=[memory])
    clients = _get_boto_clients()

    unique_content = f"The secret password for the test is {uuid.uuid4()}."
    store_result = agent.tool.memory(
        action="store",
        content=unique_content,
        title="Integration Test Document",
        STRANDS_KNOWLEDGE_BASE_ID=kb_id,
        region_name=AWS_REGION,
    )
    assert store_result["status"] == "success", f"Store failed: {store_result}"

    # Extract document ID
    doc_id = next(
        (
            item["text"].split(":", 1)[1].strip()
            for item in store_result["content"]
            if "Document ID" in item.get("text", "")
        ),
        None,
    )
    assert doc_id, f"No document_id returned from store operation. Got: {store_result}"

    # Wait up to 3 minutes for document to be INDEXED
    ds_id = clients["bedrock-agent"].list_data_sources(knowledgeBaseId=kb_id)["dataSourceSummaries"][0]["dataSourceId"]
    for _ in range(18):  # 18 * 10s = 180s
        docs = clients["bedrock-agent"].list_knowledge_base_documents(knowledgeBaseId=kb_id, dataSourceId=ds_id)[
            "documentDetails"
        ]
        found = next((d for d in docs if d.get("identifier", {}).get("custom", {}).get("id") == doc_id), None)
        if found and found.get("status") == "INDEXED":
            break
        time.sleep(10)
    else:
        raise AssertionError("Stored document did not become INDEXED in time.")

    # Try up to 2 minutes for the content to appear in retrieval results
    for _ in range(12):
        retrieve_result = agent.tool.memory(
            action="retrieve",
            query="The secret password for the test is",
            STRANDS_KNOWLEDGE_BASE_ID=kb_id,
            region_name=AWS_REGION,
            min_score=0.0,
            max_results=5,
        )
        assert retrieve_result["status"] == "success", f"Retrieve failed: {retrieve_result}"

        full_retrieved_text = " ".join(item.get("text", "") for item in retrieve_result.get("content", []))
        if unique_content in full_retrieved_text:
            break
        time.sleep(10)
    else:
        raise AssertionError("Stored content not found in retrieval after waiting.")
