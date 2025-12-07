"""
Amazon Redshift SQL Query Tool for Strands Agents.

This tool enables Strands Agents to execute SQL queries directly on
Amazon Redshift using the Redshift Data API. It supports both
PROVISIONED clusters (via clusterIdentifier) and SERVERLESS workgroups
(via workgroupName). The responses include rows, column metadata, and
a consistent ToolResult format compatible with Strands Agent workflows.

-------------------------------------------------------------------------------
Key Features
-------------------------------------------------------------------------------
1. SQL Execution
   - Execute SQL queries on Redshift Provisioned or Serverless.
   - Supports parameterized SQL queries.
   - Fully managed via Redshift Data API (no JDBC driver required).

2. AWS Config Support
   - Works with AWS CLI profiles (profile_name).
   - Allows explicit AWS Region overrides.
   - Uses Secrets Manager for DB credentials (secretArn).

3. Output Format
   - Returns list of records as JSON rows.
   - Includes column metadata.
   - Always returns a structured ToolResult:
        {
            "toolUseId": "...",
            "status": "success" | "error",
            "content": [{"text": "..."}]
        }

-------------------------------------------------------------------------------
Basic Usage Example
-------------------------------------------------------------------------------
from strands import Agent
from strands_tools import redshift_query

agent = Agent(tools=[redshift_query])

result = agent.tool.redshift_query(
    sql="SELECT * FROM users LIMIT 5",
    clusterIdentifier="my-redshift-cluster",
    database="dev",
    secretArn="arn:aws:secretsmanager:us-east-1:123456789:secret:mysecret",
)

-------------------------------------------------------------------------------
Serverless Usage Example
-------------------------------------------------------------------------------
result = agent.tool.redshift_query(
    sql="SELECT COUNT(*) FROM orders",
    workgroupName="my-serverless-workgroup",
    database="mydb",
    secretArn="arn:aws:secretsmanager:us-east-1:xxx:secret:mysecret",
)

-------------------------------------------------------------------------------
Parameterized SQL Example
-------------------------------------------------------------------------------
result = agent.tool.redshift_query(
    sql="SELECT * FROM users WHERE user_id = :uid",
    parameters={"uid": 1001},
    clusterIdentifier="cluster-1",
    database="dev",
    secretArn="arn:aws:secretsmanager:us-east-1:xxx:secret:secret123",
)

-------------------------------------------------------------------------------
Notes
-------------------------------------------------------------------------------
- You must provide either clusterIdentifier (Provisioned) OR workgroupName (Serverless).
- AWS credentials must be configured correctly.
- Uses Redshift Data API polling to wait for query completion.

-------------------------------------------------------------------------------
"""

import os
import time
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config as BotocoreConfig
from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
    "name": "redshift_query",
    "description": "Executes SQL queries on Amazon Redshift using the Redshift Data API.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to execute.",
                },
                "clusterIdentifier": {
                    "type": "string",
                    "description": "Provisioned Redshift cluster identifier.",
                },
                "workgroupName": {
                    "type": "string",
                    "description": "Serverless Redshift workgroup name.",
                },
                "database": {
                    "type": "string",
                    "description": "Database to execute the query against.",
                },
                "secretArn": {
                    "type": "string",
                    "description": "AWS Secrets Manager ARN for DB credentials.",
                },
                "region": {
                    "type": "string",
                    "description": "AWS Region. Default is 'us-east-1'.",
                },
                "parameters": {
                    "type": "object",
                    "description": "Optional dictionary of SQL parameters.",
                },
                "profile_name": {
                    "type": "string",
                    "description": "Optional AWS CLI profile to use.",
                },
            },
            "required": ["sql", "database", "secretArn"],
        }
    },
}


def _rs_client(profile: Optional[str], region: str):
    """Create a Redshift Data API client, with custom User-Agent."""
    config = BotocoreConfig(user_agent_extra="strands-agents-redshift")
    if profile:
        session = boto3.Session(profile_name=profile)
        return session.client("redshift-data", region_name=region, config=config)
    return boto3.client("redshift-data", region_name=region, config=config)


def _poll_status(client, statement_id: str) -> Dict[str, Any]:
    """Poll Redshift Data API until the SQL query finishes execution."""
    while True:
        status_info = client.describe_statement(Id=statement_id)
        if status_info["Status"] in ("FINISHED", "FAILED", "ABORTED"):
            return status_info
        time.sleep(0.5)


def _convert_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert Redshift Data API row format into Python dict rows."""
    columns = [col["label"] for col in result.get("ColumnMetadata", [])]
    rows = []

    for record in result.get("Records", []):
        row = {}
        for col_name, col_val in zip(columns, record, strict=False):
            value = next(iter(col_val.values()), None)
            row[col_name] = value
        rows.append(row)

    return rows


def redshift_query(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Execute a SQL query on Amazon Redshift using the Redshift Data API.

    The function supports:
    - Provisioned Redshift (clusterIdentifier)
    - Redshift Serverless (workgroupName)
    - Parameterized SQL queries
    - AWS profile-based sessions

    Returns a standard ToolResult with:
    - toolUseId
    - status ("success" or "error")
    - content (response text with JSON payload)
    """
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    try:
        sql = tool_input["sql"]
        database = tool_input["database"]
        secret_arn = tool_input["secretArn"]

        cluster_id = tool_input.get("clusterIdentifier")
        workgroup = tool_input.get("workgroupName")
        parameters = tool_input.get("parameters")
        profile = tool_input.get("profile_name")

        region = tool_input.get("region", os.getenv("AWS_REGION", "us-east-1"))
        client = _rs_client(profile, region)

        exec_args = {
            "Sql": sql,
            "Database": database,
            "SecretArn": secret_arn,
        }

        if parameters:
            exec_args["Parameters"] = [{"name": k, "value": str(v)} for k, v in parameters.items()]

        if workgroup:
            exec_args["WorkgroupName"] = workgroup
        else:
            if not cluster_id:
                raise ValueError("Either 'clusterIdentifier' or 'workgroupName' must be provided.")
            exec_args["ClusterIdentifier"] = cluster_id

        response = client.execute_statement(**exec_args)
        statement_id = response["Id"]

        status_info = _poll_status(client, statement_id)
        if status_info["Status"] != "FINISHED":
            raise RuntimeError(f"Query execution failed with status: {status_info['Status']}")

        result_page = client.get_statement_result(Id=statement_id)
        records = _convert_rows(result_page)

        final_output = {
            "records": records,
            "column_metadata": result_page.get("ColumnMetadata", []),
        }

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": f"Query executed successfully:\n{final_output}"}],
        }

    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error during Redshift query: {str(e)}"}],
        }
