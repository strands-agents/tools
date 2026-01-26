"""AWS CLI service commands retrieval tool."""

from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from strands import tool


@tool
def get_aws_service_commands(service: str) -> List[Dict[str, str]]:
    """
    Retrieves the list of available AWS CLI commands for a given AWS service.

    Args:
        service (str): The AWS service name (e.g., 'ec2', 'lambda', 's3', 'dynamodb').

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing 'command' and 'link' keys
            for the available commands. Returns empty list if no commands found.

    Examples:
        >>> get_aws_service_commands("s3")
        [
            {'command': 'cp', 'link': 'https://awscli.amazonaws.com/v2/documentation/api/latest/reference/s3/cp.html'},
            {'command': 'ls', 'link': 'https://awscli.amazonaws.com/v2/documentation/api/latest/reference/s3/ls.html'},
            ...
        ]
    """
    url = f"https://docs.aws.amazon.com/cli/latest/reference/{service}/#available-commands"
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    header = soup.find("div", {"id": "available-commands"})
    if not header:
        return []

    ul = header.find_next("ul")
    if not ul:
        return []

    commands = []
    for li in ul.find_all("li"):
        link = li.find("a")
        if link:
            commands.append(
                {
                    "command": link.text.strip(),
                    "link": f"https://docs.aws.amazon.com/cli/latest/reference/{service}/{link['href']}",
                }
            )
    return commands
