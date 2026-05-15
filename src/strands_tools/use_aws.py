"""AWS service integration tool for Strands Agent.

This module provides a comprehensive interface to AWS services through boto3,
allowing you to invoke any AWS API operation directly from your Strands Agent.
The tool handles authentication, parameter validation, response formatting,
and provides user-friendly error messages with input schema recommendations.

Key Features:

1. Universal AWS Access:
   • Access to all boto3-supported AWS services
   • Support for all service operations in snake_case format
   • Region-specific API calls
   • AWS profile support for credential management

2. Safety Features:
   • Confirmation prompts for mutative operations (create, update, delete)
   • Confirmation prompts for credential-returning operations (STS, Secrets Manager, ECR)
   • Response redaction of known sensitive keys (SecretAccessKey, SessionToken, etc.)
   • Parameter validation with helpful error messages
   • Automatic schema generation for invalid requests
   • Error handling with detailed feedback

3. Response Handling:
   • JSON formatting of responses
   • Special handling for streaming responses
   • DateTime object conversion for JSON compatibility
   • Pretty printing of operation details

4. Usage Examples:
   ```python
   from strands import Agent
   from strands_tools import use_aws

   agent = Agent(tools=[use_aws])

   # List S3 buckets
   result = agent.tool.use_aws(
       service_name="s3",
       operation_name="list_buckets",
       parameters={},
       region="us-west-2",
       label="List all S3 buckets"
   )
   ```

Security Recommendations:

   The IAM role attached to the agent's execution environment should follow
   least-privilege principles. Grant only the services and operations the
   agent needs to fulfill its purpose.

   • Scope the role policy to specific resources and actions rather than
     using wildcards.
   • Add explicit Deny statements for operations the agent should never
     call (e.g., credential-returning or destructive APIs).
   • Use IAM condition keys (e.g., aws:SourceIp, aws:RequestedRegion) to
     further constrain what the role can do and from where.

   The tool provides response redaction and consent prompts as defense-in-depth,
   but IAM policy scoping remains the primary mechanism for controlling what
   an agent can access.

See the use_aws function docstring for more details on parameters and usage.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ParamValidationError, ValidationError
from botocore.response import StreamingBody
from rich import box
from rich.panel import Panel
from rich.table import Table
from strands.types.tools import ToolResult, ToolUse

from strands_tools.utils import console_util
from strands_tools.utils.data_util import convert_datetime_to_str
from strands_tools.utils.generate_schema_util import generate_input_schema
from strands_tools.utils.user_input import get_user_input

logger = logging.getLogger(__name__)

MUTATIVE_OPERATIONS = [
    "create",
    "put",
    "delete",
    "update",
    "terminate",
    "revoke",
    "disable",
    "deregister",
    "stop",
    "add",
    "modify",
    "remove",
    "attach",
    "detach",
    "start",
    "enable",
    "register",
    "set",
    "associate",
    "disassociate",
    "allocate",
    "release",
    "cancel",
    "reboot",
    "accept",
]

# Operations that return credentials or secrets and require user consent
# even though they are not mutative. These operations disclose sensitive
# material that should not be exposed to the LLM context without explicit
# human approval.
SENSITIVE_OPERATIONS = {
    ("sts", "get_session_token"),
    ("sts", "assume_role"),
    ("sts", "assume_role_with_saml"),
    ("sts", "assume_role_with_web_identity"),
    ("sts", "get_federation_token"),
    ("secretsmanager", "get_secret_value"),
    ("secretsmanager", "batch_get_secret_value"),
    ("ecr", "get_authorization_token"),
    ("ecr-public", "get_authorization_token"),
    ("redshift", "get_cluster_credentials"),
    ("redshift", "get_cluster_credentials_with_iam"),
    ("codeartifact", "get_authorization_token"),
    ("cognito-identity", "get_credentials_for_identity"),
    ("cognito-identity", "get_open_id_token"),
    ("cognito-identity", "get_open_id_token_for_developer_identity"),
    ("cognito-idp", "initiate_auth"),
    ("cognito-idp", "admin_initiate_auth"),
}

# Response keys whose values must be redacted before returning to LLM context.
# These are exact key names (compared case-insensitively) known to carry secrets.
# Avoid overly generic names like "token" which collide with pagination tokens.
SENSITIVE_RESPONSE_KEYS = {
    "secretaccesskey",
    "sessiontoken",
    "secretstring",
    "secretbinary",
    "authorizationtoken",
    "password",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "apikey",
    "clientsecret",
    "keymaterial",
    "privatekey",
    "sharedsecret",
    "dbpassword",
    "masteruserpassword",
}

def redact_sensitive_values(obj: Any) -> Any:
    """Recursively redact values of known sensitive keys from an AWS response.

    Args:
        obj: The object (dict, list, or scalar) to scrub.

    Returns:
        A copy of the object with sensitive values replaced by a placeholder.
    """
    if isinstance(obj, dict):
        redacted = {}
        for key, value in obj.items():
            if key.lower() in SENSITIVE_RESPONSE_KEYS:
                redacted[key] = "**REDACTED**"
            else:
                redacted[key] = redact_sensitive_values(value)
        return redacted
    elif isinstance(obj, list):
        return [redact_sensitive_values(item) for item in obj]
    return obj


def get_boto3_client(
    service_name: str,
    region_name: str,
    profile_name: Optional[str] = None,
) -> Any:
    """Create an AWS boto3 client for the specified service and region.

    Args:
        service_name: Name of the AWS service (e.g., 's3', 'ec2', 'dynamodb')
        region_name: AWS region name (e.g., 'us-west-2', 'us-east-1')
        profile_name: Optional AWS profile name from ~/.aws/credentials

    Returns:
        A boto3 client object for the specified service
    """
    session = boto3.Session(profile_name=profile_name)

    config = BotocoreConfig(user_agent_extra="strands-agents-use-aws")

    return session.client(service_name=service_name, region_name=region_name, config=config)


def handle_streaming_body(response: Dict[str, Any]) -> Dict[str, Any]:
    """Process streaming body responses from AWS into regular Python objects.

    Some AWS APIs return StreamingBody objects that need special handling to
    convert them into regular Python dictionaries or strings for proper JSON serialization.

    Args:
        response: AWS API response that may contain StreamingBody objects

    Returns:
        Processed response with StreamingBody objects converted to Python objects
    """
    for key, value in response.items():
        if isinstance(value, StreamingBody):
            content = value.read()
            try:
                response[key] = json.loads(content.decode("utf-8"))
            except json.JSONDecodeError:
                response[key] = content.decode("utf-8")
    return response


def get_available_services() -> List[str]:
    """Get a list of all available AWS services supported by boto3.

    Returns:
        List of service names as strings
    """
    services = boto3.Session().get_available_services()
    return list(services)


def get_available_operations(service_name: str) -> List[str]:
    """Get a list of all available operations for a specific AWS service.

    Args:
        service_name: Name of the AWS service (e.g., 's3', 'ec2')

    Returns:
        List of operation names as strings
    """

    aws_region = os.environ.get("AWS_REGION", "us-west-2")
    try:
        client = boto3.client(service_name, region_name=aws_region)
        return [op for op in dir(client) if not op.startswith("_")]
    except Exception as e:
        logger.error(f"Error getting operations for service {service_name}: {str(e)}")
        return []


TOOL_SPEC = {
    "name": "use_aws",
    "description": (
        "Make a boto3 client call with the specified service, operation, and parameters. "
        "Boto3 operations are snake_case."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "The name of the AWS service",
                },
                "operation_name": {
                    "type": "string",
                    "description": "The name of the operation to perform",
                },
                "parameters": {
                    "type": "object",
                    "description": "The parameters for the operation",
                },
                "region": {
                    "type": "string",
                    "description": "Region name for calling the operation on AWS boto3",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Label of AWS API operations human readable explanation. "
                        "This is useful for communicating with human."
                    ),
                },
                "profile_name": {
                    "type": "string",
                    "description": (
                        "Optional: AWS profile name to use from ~/.aws/credentials. "
                        "Defaults to default profile if not specified."
                    ),
                },
            },
            "required": [
                "region",
                "service_name",
                "operation_name",
                "label",
            ],
        }
    },
}


def use_aws(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Execute AWS service operations using boto3 with comprehensive error handling and validation.

    This tool provides a universal interface to AWS services, allowing you to execute
    any operation supported by boto3. It handles authentication, parameter validation,
    response formatting, and provides helpful error messages with schema recommendations
    when invalid parameters are provided.

    How It Works:
    ------------
    1. The tool validates the provided service and operation names against available APIs
    2. For potentially disruptive operations (create, delete, etc.), it prompts for confirmation
    3. It sets up a boto3 client with appropriate region and credentials
    4. The requested operation is executed with the provided parameters
    5. Responses are processed to handle special data types (e.g., streaming bodies)
    6. If errors occur, helpful messages and expected parameter schemas are returned

    Common Usage Scenarios:
    ---------------------
    - Resource Management: Create, list, modify or delete AWS resources
    - Data Operations: Store, retrieve, or process data in AWS services
    - Configuration: Update settings or permissions for AWS services
    - Monitoring: Retrieve metrics, logs or status information
    - Security Operations: Manage IAM roles, policies or security settings

    Args:
        tool: The ToolUse object containing:
            - toolUseId: Unique identifier for this tool invocation
            - input: Dictionary containing:
                - service_name: AWS service name (e.g., 's3', 'ec2', 'dynamodb')
                - operation_name: Operation to perform in snake_case (e.g., 'list_buckets')
                - parameters: Dictionary of parameters for the operation
                - region: AWS region (e.g., 'us-west-2')
                - label: Human-readable description of the operation
                - profile_name: Optional AWS profile name for credentials
        **kwargs: Additional keyword arguments (unused)

    Returns:
        ToolResult dictionary with:
        - toolUseId: Same ID from the request
        - status: 'success' or 'error'
        - content: List of content dictionaries with response text

    Notes:
        - Mutative operations (create, delete, etc.) require user confirmation in non-dev environments
        - You can disable confirmation by setting the environment variable BYPASS_TOOL_CONSENT=true
        - The tool automatically handles special response types like streaming bodies
        - For validation errors, the tool attempts to generate the correct input schema
        - All datetime objects are automatically converted to strings for proper JSON serialization
    """
    aws_region = os.environ.get("AWS_REGION", "us-west-2")
    console = console_util.create()

    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    service_name = tool_input["service_name"]
    operation_name = tool_input["operation_name"]
    parameters = tool_input.get("parameters", {})
    region = tool_input.get("region", aws_region)
    label = tool_input.get("label", "AWS Operation Details")

    STRANDS_BYPASS_TOOL_CONSENT = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

    # Create a panel for AWS Operation Details using Rich's native styling
    details_table = Table(show_header=False, box=box.SIMPLE, pad_edge=False)
    details_table.add_column("Property", style="cyan", justify="left", min_width=12)
    details_table.add_column("Value", style="white", justify="left")

    details_table.add_row("Service:", service_name)
    details_table.add_row("Operation:", operation_name)
    details_table.add_row("Region:", region)

    if parameters:
        details_table.add_row("Parameters:", "")
        for key, value in parameters.items():
            details_table.add_row(f"  • {key}:", str(value))
    else:
        details_table.add_row("Parameters:", "None")

    console.print(Panel(details_table, title=f"[bold blue]🚀 {label}[/bold blue]", border_style="blue", expand=False))

    logger.debug(
        "Invoking: service_name = %s, operation_name = %s, parameters = %s" % (service_name, operation_name, parameters)
    )

    # Check if the operation is potentially mutative
    is_mutative = any(op in operation_name.lower() for op in MUTATIVE_OPERATIONS)

    # Check if the operation returns credentials or secrets
    is_sensitive = (service_name.lower(), operation_name.lower()) in SENSITIVE_OPERATIONS

    if (is_mutative or is_sensitive) and not STRANDS_BYPASS_TOOL_CONSENT:
        if is_sensitive:
            prompt_message = (
                f"<yellow><bold>The operation '{service_name}.{operation_name}' may return "
                f"sensitive credentials or secrets. Do you want to proceed?</bold> [y/*]</yellow>"
            )
        else:
            prompt_message = (
                f"<yellow><bold>The operation '{operation_name}' is potentially mutative. "
                f"Do you want to proceed?</bold> [y/*]</yellow>"
            )

        confirm = get_user_input(prompt_message)
        if confirm.lower() != "y":
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Operation canceled by user. Reason: {confirm}."}],
            }

    # Check AWS service
    available_services = get_available_services()
    if service_name not in available_services:
        logger.debug(f"Invalid AWS service: {service_name}")
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [
                {"text": f"Invalid AWS service: {service_name}\nAvailable services: {str(available_services)}"}
            ],
        }

    # Check AWS operation
    available_operations = get_available_operations(service_name)
    if operation_name not in available_operations:
        logger.debug(f"Invalid AWS operation: {operation_name}")
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [
                {"text": f"Invalid AWS operation: {operation_name}, Available operations:\n{available_operations}\n"}
            ],
        }

    # Set up the boto3 client
    profile_name = tool_input.get("profile_name")
    client = get_boto3_client(service_name, region, profile_name)
    operation_method = getattr(client, operation_name)

    try:
        response = operation_method(**parameters)
        response = handle_streaming_body(response)
        response = convert_datetime_to_str(response)
        response = redact_sensitive_values(response)

        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": f"Success: {str(response)}"}],
        }
    except (ValidationError, ParamValidationError) as val_ex:
        # Handle validation errors with schema
        try:
            schema = generate_input_schema(service_name, operation_name)
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [
                    {"text": f"Validation error: {str(val_ex)}"},
                    {"text": f"Expected input schema for {operation_name}:"},
                    {"text": json.dumps(schema, indent=2)},
                ],
            }
        except Exception as schema_ex:
            logger.error(f"Failed to generate schema: {str(schema_ex)}")
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Validation error: {str(val_ex)}"}],
            }
    except Exception as ex:
        logger.warning(f"AWS call threw exception: {type(ex).__name__}")
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"AWS call threw exception: {str(ex)}"}],
        }
