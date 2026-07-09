import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError
from opensearchpy import (
    AuthenticationException,
    AuthorizationException,
    AWSV4SignerAuth,
    OpenSearch,
    RequestsHttpConnection,
)

AWS_REGION = "us-east-1"
EMBEDDING_MODEL_ARN = f"arn:aws:bedrock:{AWS_REGION}::foundation-model/amazon.titan-embed-text-v1"
EMBEDDING_DIMENSION = 1536  # Dimension for amazon.titan-embed-text-v1

logger = logging.getLogger(__name__)


class KnowledgeBaseHelper:
    def __init__(self):
        self.clients = self.get_boto_clients()
        self.index = {
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
        self.resource_names = self.index
        self.created_resources = {}
        self.should_teardown = os.environ.get("STRANDS_TEARDOWN_RESOURCES", "false").lower() == "true"

    @staticmethod
    def get_boto_clients():
        return {
            "iam": boto3.client("iam", region_name=AWS_REGION),
            "bedrock-agent": boto3.client("bedrock-agent", region_name=AWS_REGION),
            "opensearchserverless": boto3.client("opensearchserverless", region_name=AWS_REGION),
            "sts": boto3.client("sts", region_name=AWS_REGION),
        }

    def try_get_existing(self):
        try:
            kbs = self.clients["bedrock-agent"].list_knowledge_bases()
            kb = next(
                (kb for kb in kbs.get("knowledgeBaseSummaries", []) if kb["name"] == self.resource_names["kb_name"]),
                None,
            )
            if kb and kb["status"] == "ACTIVE":
                return kb["knowledgeBaseId"]
        except ClientError as e:
            logger.error(f"Error checking for existing Knowledge Bases: {e}")
        return None

    def create_resources(self):
        resources = self.resource_names
        client = self.clients

        # 1. OpenSearch Security Policies
        try:
            client["opensearchserverless"].create_security_policy(
                name=resources["enc_policy_name"],
                type="encryption",
                policy=json.dumps(
                    {
                        "Rules": [
                            {"ResourceType": "collection", "Resource": [f"collection/{resources['collection_name']}"]}
                        ],
                        "AWSOwnedKey": True,
                    }
                ),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConflictException":
                raise

        try:
            client["opensearchserverless"].create_security_policy(
                name=resources["net_policy_name"],
                type="network",
                policy=json.dumps(
                    [
                        {
                            "Rules": [
                                {
                                    "ResourceType": "collection",
                                    "Resource": [f"collection/{resources['collection_name']}"],
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
        time.sleep(10)

        # 2. OpenSearch Collection
        try:
            collection_res = client["opensearchserverless"].create_collection(
                name=resources["collection_name"], type="VECTORSEARCH"
            )
            collection_id = collection_res["createCollectionDetail"]["id"]
            collection_arn = collection_res["createCollectionDetail"]["arn"]
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConflictException":
                raise
            collection_details = client["opensearchserverless"].list_collections(
                collectionFilters={"name": resources["collection_name"]}
            )["collectionSummaries"][0]
            collection_id = collection_details["id"]
            collection_arn = collection_details["arn"]
        self.created_resources["collection_id"] = collection_id

        # 3. IAM Role and Policies (create first to get ARN)
        try:
            role_res = client["iam"].get_role(RoleName=resources["role_name"])
            self.created_resources["role_arn"] = role_res["Role"]["Arn"]
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchEntity":
                raise
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Principal": {"Service": "bedrock.amazonaws.com"}, "Action": "sts:AssumeRole"}
                ],
            }
            role_res = client["iam"].create_role(
                RoleName=resources["role_name"], AssumeRolePolicyDocument=json.dumps(assume_role_policy)
            )
            self.created_resources["role_arn"] = role_res["Role"]["Arn"]
            time.sleep(10)  # Wait for role to propagate

        # 4. OpenSearch Data Access Policy (create before collection is active)
        try:
            user_arn = client["sts"].get_caller_identity()["Arn"]
            access_policy_doc = [
                {
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [f"collection/{resources['collection_name']}"],
                            "Permission": ["aoss:*"],
                        },
                        {
                            "ResourceType": "index",
                            "Resource": [f"index/{resources['collection_name']}/*"],
                            "Permission": ["aoss:*"],
                        },
                    ],
                    "Principal": [self.created_resources["role_arn"], user_arn],
                }
            ]
            client["opensearchserverless"].create_access_policy(
                name=resources["access_policy_name"], type="data", policy=json.dumps(access_policy_doc)
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConflictException":
                raise
        # 5. Wait for OpenSearch Collection to be Active
        collection_details = self._wait_for_resource(
            lambda: client["opensearchserverless"].batch_get_collection(ids=[collection_id])["collectionDetails"][0],
            resource_name=f"OpenSearch Collection ({collection_id})",
        )
        
        # 6. Add IAM policy after collection is ready
        try:
            iam_policy_doc = {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Action": "bedrock:InvokeModel", "Resource": EMBEDDING_MODEL_ARN},
                    {"Effect": "Allow", "Action": "aoss:APIAccessAll", "Resource": collection_arn},
                ],
            }
            client["iam"].put_role_policy(
                RoleName=resources["role_name"],
                PolicyName=resources["policy_name"],
                PolicyDocument=json.dumps(iam_policy_doc),
            )
            time.sleep(30)  # Wait for policy to propagate
        except ClientError as e:
            if "EntityAlreadyExists" not in str(e):
                raise

        # 7. Create vector index
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
        if not os_client.indices.exists(resources["vector_index_name"]):
            time.sleep(20)
            for i in range(5):
                try:
                    os_client.indices.create(resources["vector_index_name"], body=index_body)
                    break
                except (AuthenticationException, AuthorizationException) as e:
                    if i < 4:
                        time.sleep(30)
                    else:
                        raise e
            time.sleep(10)

        # 8. Knowledge Base
        kb_res = client["bedrock-agent"].create_knowledge_base(
            name=resources["kb_name"],
            roleArn=self.created_resources["role_arn"],
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {"embeddingModelArn": EMBEDDING_MODEL_ARN},
            },
            storageConfiguration={
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": collection_arn,
                    "vectorIndexName": resources["vector_index_name"],
                    "fieldMapping": {
                        "vectorField": "bedrock-kb-vector",
                        "textField": "AMAZON_BEDROCK_TEXT_CHUNK",
                        "metadataField": "AMAZON_BEDROCK_METADATA",
                    },
                },
            },
        )
        self.created_resources["kb_id"] = kb_res["knowledgeBase"]["knowledgeBaseId"]
        self._wait_for_resource(
            lambda: client["bedrock-agent"].get_knowledge_base(knowledgeBaseId=self.created_resources["kb_id"])[
                "knowledgeBase"
            ],
            resource_name=f"Knowledge Base ({self.created_resources['kb_id']})",
        )
        # 9. Data Source
        ds_resource = client["bedrock-agent"].create_data_source(
            knowledgeBaseId=self.created_resources["kb_id"],
            name=resources["ds_name"],
            dataSourceConfiguration={"type": "CUSTOM"},
        )
        self.created_resources["ds_id"] = ds_resource["dataSource"]["dataSourceId"]
        time.sleep(10)
        return self.created_resources["kb_id"]

    def destroy(self):
        client = self.clients
        resources = self.resource_names
        cr = self.created_resources
        try:
            if "ds_id" in cr and "kb_id" in cr:
                client["bedrock-agent"].delete_data_source(knowledgeBaseId=cr["kb_id"], dataSourceId=cr["ds_id"])
        except ClientError:
            pass
        try:
            if "kb_id" in cr:
                client["bedrock-agent"].delete_knowledge_base(knowledgeBaseId=cr["kb_id"])
                time.sleep(30)
        except ClientError:
            pass
        try:
            if "collection_id" in cr:
                client["opensearchserverless"].delete_collection(id=cr["collection_id"])
        except ClientError:
            pass
        try:
            client["opensearchserverless"].delete_access_policy(name=resources["access_policy_name"], type="data")
        except ClientError:
            pass
        try:
            client["opensearchserverless"].delete_security_policy(name=resources["net_policy_name"], type="network")
        except ClientError:
            pass
        try:
            client["opensearchserverless"].delete_security_policy(name=resources["enc_policy_name"], type="encryption")
        except ClientError:
            pass
        try:
            client["iam"].delete_role_policy(RoleName=resources["role_name"], PolicyName=resources["policy_name"])
            client["iam"].delete_role(RoleName=resources["role_name"])
        except ClientError:
            pass

    def _wait_for_resource(
        self,
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
                logger.info(
                    f"Polling status for '{resource_name}': {status} (Elapsed: {int(time.time() - start_time)}s)"
                )
                if status == success_status:
                    logger.info(f"SUCCESS: Resource '{resource_name}' reached '{success_status}' state.")
                    return response
                if status == failure_status:
                    logger.error(
                        f"FAILURE: Resource '{resource_name}' failed with status: {failure_status}-Response:{response}"
                    )
                    raise Exception(f"Resource '{resource_name}' entered failure state: {failure_status}")
            except ClientError as e:
                if "ResourceNotFoundException" not in str(e):
                    raise e
                logger.info(f"Resource '{resource_name}' not found yet, continuing to wait...")
            time.sleep(delay_seconds)
        raise TimeoutError(f"Timed out waiting for resource '{resource_name}' to become '{success_status}'.")
