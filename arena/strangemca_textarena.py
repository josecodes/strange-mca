"""
TextArena integration for Strange MCA.

This module provides agent classes and utilities for using Strange MCA with TextArena.
"""

import os
import re
import sys
import textwrap
from typing import Dict, Any, Optional

import textarena as ta

# Add the project root to the Python path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strange_mca.run_strange_mca import run_strange_mca

class StrangeMCAAgent(ta.Agent):
    """
    Custom agent wrapper for the Strange MCA multi-agent system.
    This agent uses a team of LLMs to analyze and solve tasks.
    """
    
    def __init__(
        self,
        child_per_parent: int = 3,
        depth: int = 2,
        model: str = "gpt-3.5-turbo",
        viz: bool = False,
        print_details: bool = False,
        task_template: Optional[str] = None,
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
        
        # Run the Strange MCA system
        print(f"Running Strange MCA with {self.child_per_parent} children per parent, depth {self.depth}, model {self.model}")
        result = run_strange_mca(
            task=task,
            child_per_parent=self.child_per_parent,
            depth=self.depth,
            model=self.model,
            viz=self.viz,
            print_details=self.print_details,
        )
        
        # Get the final response
        final_response = result.get("final_response", "")
        
        # Print the response for debugging
        print(f"Response: {final_response}")
        
        return final_response

