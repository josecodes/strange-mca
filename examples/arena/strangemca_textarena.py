"""
TextArena integration for Strange MCA.

This module provides agent classes and utilities for using Strange MCA with TextArena.
"""

import os
import sys
from typing import Optional

import textarena as ta

# Add the project root to the Python path to allow importing from src
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.strange_mca.run_strange_mca import run_strange_mca


class StrangeMCAAgent(ta.Agent):
    """
    Custom agent wrapper for the Strange MCA multi-agent system.
    This agent uses a team of LLMs to analyze and solve tasks.
    """

    def __init__(
        self,
        child_per_parent: int = 2,
        depth: int = 2,
        model: str = "gpt-3.5-turbo",
        viz: bool = False,
        print_details: bool = False,
        task_template: Optional[str] = None,
        domain_specific_instructions: Optional[str] = "",
        strange_loop_count: int = 0,
    ):
        """
        Initialize the Strange MCA agent.

        Args:
            child_per_parent: Number of children per parent node in the agent tree.
            depth: Depth of the agent tree.
            model: The model to use for the agents.
            viz: Whether to generate visualizations.
            print_details: Whether to print detailed information.
            task_template: Optional template for formatting the task. If None, the observation is used as is.
        """
        self.child_per_parent = child_per_parent
        self.depth = depth
        self.model = model
        self.viz = viz
        self.print_details = print_details
        self.task_template = task_template
        self.domain_specific_instructions = domain_specific_instructions
        self.strange_loop_count = strange_loop_count

    def __call__(self, observation: str) -> str:
        """
        Process the observation and return an action using the Strange MCA system.

        Args:
            observation: The observation from the environment.

        Returns:
            The action to take.
        """
        # Format the task if a template is provided
        if self.task_template:
            task = self.task_template.format(observation=observation)
        else:
            task = observation
        result = run_strange_mca(
            task=task,
            child_per_parent=self.child_per_parent,
            depth=self.depth,
            model=self.model,
            viz=self.viz,
            print_details=self.print_details,
            domain_specific_instructions=self.domain_specific_instructions,
            strange_loop_count=self.strange_loop_count,
        )
        final_response = result.get("final_response", "")
        return final_response
