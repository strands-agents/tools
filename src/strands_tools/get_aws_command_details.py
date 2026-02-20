"""AWS CLI command details retrieval tool."""

from typing import Any, Dict

import requests
from bs4 import BeautifulSoup
from strands import tool

from strands_tools.get_aws_service_commands import get_aws_service_commands


@tool
def get_aws_command_details(service: str, command: str) -> Dict[str, Any]:
    """
    Retrieves the documentation details for a specific AWS CLI command of a given service.

    Args:
        service (str): The AWS service name (e.g., 'ec2', 'lambda', 's3', 'dynamodb').
        command (str): The AWS CLI command name to get details for (e.g., 'list-buckets', 'describe-instances').

    Returns:
        Dict[str, Any]: A dictionary with 'command', 'link', and 'details' keys.
            Returns error dictionary if command not found.

    Examples:
        >>> get_aws_command_details("s3", "ls")
        {
            'command': 'ls',
            'link': 'https://docs.aws.amazon.com/cli/latest/reference/s3/ls.html',
            'details': 'List S3 objects and common prefixes...'
        }
    """
    commands = get_aws_service_commands(service)
    command_info = next((cmd for cmd in commands if cmd["command"] == command), None)

    if not command_info:
        return {"error": "Command not found"}

    response = requests.get(command_info["link"])
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    main_content = soup.find("div", {"id": command})
    if not main_content:
        return {"command": command, "details": "No details found"}

    details = main_content.get_text(separator="\n", strip=True)
    return {"command": command, "link": command_info["link"], "details": details}
