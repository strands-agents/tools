"""
CI environment configurations for skipping tests in specific CI environments.
"""

import os
from dataclasses import dataclass

from pytest import mark


@dataclass
class CIEnvironmentInfo:
    """Information about CI environments for skipping tests."""

    def __init__(self, id: str, environment_variable: str, value: str = "true") -> None:
        self.id = id
        self.environment_variable = environment_variable
        self.value = value
        self.mark = mark.skipif(
            os.environ.get(self.environment_variable) == self.value,
            reason=f"Test skipped in {self.id} environment to limit utilization",
            )


skip_if_github_action = CIEnvironmentInfo(id="Skip GitHub Actions", environment_variable="GITHUB_ACTIONS", value="true")
