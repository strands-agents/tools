"""
This module provides utility functions for generating JSON schemas for AWS service operations.

It includes functions for:
1. Generating schemas from boto3 shapes
2. Cleaning and trimming descriptions
3. Converting between Pascal, snake, and kebab case
4. Checking boto3 validity of service and operation names
5. Generating input schemas for AWS service operations

The main function, generate_input_schema, combines these utilities to create a complete
input schema for a given AWS service operation.

Example usage:
schema = generate_input_schema('s3', 'list_buckets')

"""

import logging
import re
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import boto3
from botocore.exceptions import UnknownServiceError
from botocore.model import Shape

# Initialize logging and set paths
logger = logging.getLogger(__name__)

# Precompile regex patterns for improved performance
CLEAN_HTML_PATTERN = re.compile("<[^<]+?>")
SNAKE_CASE_PATTERN1 = re.compile(r"([A-Z])([A-Z])([a-z])")
SNAKE_CASE_PATTERN2 = re.compile(r"([a-z0-9])([A-Z])")
WORD_SPLIT_PATTERN = re.compile(r"[_-]")

# Shape type mapping for efficient lookup
SHAPE_TYPE_MAP = {
    "string": {"type": "string"},
    "integer": {"type": "integer"},
    "boolean": {"type": "boolean"},
    "float": {"type": "number"},
    "double": {"type": "number"},
    "long": {"type": "integer"},
}


@lru_cache(maxsize=128)
def generate_schema(shape: Optional[Shape], depth: int = 0, max_depth: int = 5) -> Dict[str, Any]:
    """
    Recursively generate a JSON schema from a boto3 shape.

    This function creates a JSON schema representation of the given boto3 shape,
    handling nested structures up to a specified maximum depth.

    Args:
        shape (Optional[Shape]): The boto3 shape to generate a schema from.
        depth (int): Current depth in the recursion. Defaults to 0.
        max_depth (int): Maximum depth to recurse. Defaults to 5.

    Returns:
        Dict[str, Any]: A dictionary representing the JSON schema.
    """
    if depth > max_depth or shape is None:
        return {}

    shape_type = shape.type_name

    if shape_type == "structure":
        schema = {
            "type": "object",
            "properties": (
                {}
                if not hasattr(shape, "members")
                else {
                    member_name: generate_schema(member_shape, depth + 1, max_depth)
                    for member_name, member_shape in shape.members.items()
                }
            ),
        }
        if hasattr(shape, "required_members") and shape.required_members:
            schema["required"] = list(shape.required_members)
        return schema
    elif shape_type == "list":
        return {
            "type": "array",
            "items": generate_schema(getattr(shape, "member", None), depth + 1, max_depth),
        }
    elif shape_type == "map":
        return {
            "type": "object",
            "additionalProperties": generate_schema(getattr(shape, "value", None), depth + 1, max_depth),
        }
    else:
        return SHAPE_TYPE_MAP.get(shape_type, {"type": "object"})


def clean_and_trim_description(description: str, max_length: int = 2000) -> str:
    """
    Clean and trim a description string by removing HTML tags and limiting length.

    Args:
        description (str): The description to clean and trim.
        max_length (int): Maximum length of the resulting string. Defaults to 2000.

    Returns:
        str: Cleaned and trimmed description.

    Example:
        >>> desc = "<p>This is a <b>sample</b> description.</p>"
        >>> clean_and_trim_description(desc, max_length=30)
        'This is a sample description.'
    """
    # Remove HTML tags
    clean_description = CLEAN_HTML_PATTERN.sub("", description)
    # Remove extra whitespace and limit length
    result = " ".join(clean_description.split())[:max_length]
    return result


def to_snake_case(input_str: str) -> str:
    """
    Convert a PascalCase, camelCase, or kebab-case string to snake_case.

    This function handles acronyms correctly (e.g., "DescribeDBInstances" -> "describe_db_instances").

    Args:
        input_str (str): The string to convert.

    Returns:
        str: The string in snake_case.

    Example:
        >>> to_snake_case("DescribeDBInstances")
        'describe_db_instances'
        >>> to_snake_case("createUser")
        'create_user'
        >>> to_snake_case("api-gateway")
        'api_gateway'
    """
    # Replace hyphens with underscores
    s1 = input_str.replace("-", "_")
    # Handle uppercase acronyms
    s2 = SNAKE_CASE_PATTERN1.sub(r"\1_\2\3", s1)
    # Insert underscore between lowercase and uppercase letters
    s3 = SNAKE_CASE_PATTERN2.sub(r"\1_\2", s2)
    result = s3.lower().lstrip("_")
    return result


@lru_cache(maxsize=128)
def to_pascal_case(service_name: str, input_str: str) -> str:
    """
    Convert a snake_case, kebab-case, or camelCase string to PascalCase.

    This function uses boto3 to get the correct PascalCase for AWS operation names.

    Args:
        service_name (str): The name of the AWS service.
        input_str (str): The input string to convert.

    Returns:
        str: The string in PascalCase.

    Example:
        >>> to_pascal_case("s3", "list_buckets")
        'ListBuckets'
        >>> to_pascal_case("dynamodb", "create-table")
        'CreateTable'
    """

    # Check if the input is already in PascalCase
    if input_str and input_str[0].isupper() and "_" not in input_str and "-" not in input_str:
        return input_str

    # Convert to PascalCase
    pascal_case = "".join(word.capitalize() for word in WORD_SPLIT_PATTERN.split(input_str))

    try:
        # Validate using boto3
        client = boto3.client(service_name, region_name="us-east-1")
        service_model = client.meta.service_model
        service_model.operation_model(pascal_case)
        return pascal_case
    except Exception:
        try:
            # Fallback: search for matching operation name
            client = boto3.client(service_name, region_name="us-east-1")
            operations = client.meta.service_model.operation_names
            snake_case = to_snake_case(input_str)
            result = next(
                (op for op in operations if to_snake_case(op) == snake_case),
                pascal_case,
            )
            return result
        except Exception:  # pragma: no cover
            logger.debug(f"Could not validate PascalCase for '{input_str}', using: '{pascal_case}'")
            return pascal_case


@lru_cache(maxsize=128)
def check_boto3_validity(service_name: str, operation_name: str) -> Tuple[bool, str]:
    """
    Check if a given service and operation are valid in boto3.

    Args:
        service_name (str): The name of the AWS service.
        operation_name (str): The name of the operation to check.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if the service and operation are valid, False otherwise.
            - str: An error message if the check fails, empty string otherwise.

    Example:
        >>> check_boto3_validity("s3", "list_buckets")
        (True, '')
        >>> check_boto3_validity("invalid_service", "invalid_operation")
        (False, "Unknown service: 'invalid_service'")
    """
    try:
        client = boto3.client(service_name, region_name="us-east-1")
        pascal_operation_name = to_pascal_case(service_name, operation_name)
        snake_operation_name = to_snake_case(pascal_operation_name)

        if hasattr(client, snake_operation_name) or hasattr(client, pascal_operation_name):
            return True, ""
        else:
            return (
                False,
                f"Operation '{operation_name}' not found in service '{service_name}'",
            )
    except UnknownServiceError:
        return False, f"Unknown service: '{service_name}'"
    except Exception as e:  # pragma: no cover
        return False, str(e)


def generate_input_schema(service_name: str, operation_name: str) -> Dict[str, Any]:
    """
    Generate an input schema for a given AWS service operation.

    This function combines all the utility functions to create a complete input schema
    for the specified AWS service operation.

    Args:
        service_name (str): The name of the AWS service.
        operation_name (str): The name of the operation.

    """

    # Check if the service and operation are valid
    is_valid, error_message = check_boto3_validity(service_name, operation_name)
    if not is_valid:
        return {
            "result": "error",
            "name": operation_name,
            "description": f"Error: {error_message}",
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }

    try:
        # Create a boto3 client and get the service model
        client = boto3.client(service_name, region_name="us-east-1")
        service_model = client.meta.service_model
        pascal_operation_name = to_pascal_case(service_name, operation_name)
        operation_model = service_model.operation_model(pascal_operation_name)

        # Generate the schema
        result = {
            "result": "success",
            "name": operation_name,
            "description": clean_and_trim_description(operation_model.documentation),
            "inputSchema": {"json": generate_schema(operation_model.input_shape)},
        }
        return result
    except Exception as e:
        raise RuntimeError(f"Error generating input schema: {str(e)}") from e
