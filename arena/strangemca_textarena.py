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
        max_iterations: int = 1,
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
            max_iterations: Maximum number of iterations to refine the response.
        """
        self.child_per_parent = child_per_parent
        self.depth = depth
        self.model = model
        self.viz = viz
        self.print_details = print_details
        self.task_template = task_template
        self.max_iterations = max_iterations
        
    def process_response(self, response: str, iteration: int, original_task: str) -> str:
        """
        Process the response from the Strange MCA system.
        This method can be overridden by subclasses to implement custom processing.
        
        Args:
            response: The response from the Strange MCA system.
            iteration: The current iteration number (0-based).
            original_task: The original task given to the system.
            
        Returns:
            The processed response.
        """
        # By default, just return the response as is
        return response
        
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
        response = result.get("final_response", "")
        
        # Process the response in a loop for refinement
        for iteration in range(self.max_iterations):
            print(f"Processing response (iteration {iteration+1}/{self.max_iterations})")
            
            # Process the response
            processed_response = self.process_response(response, iteration, task)
            
            # If the response didn't change, no need to continue
            if processed_response == response:
                print(f"Response unchanged after iteration {iteration+1}, stopping")
                break
                
            # Update the response for the next iteration
            response = processed_response
            print(f"Updated response: {response}")
        
        return response

class StrangeMCAChessAgent(StrangeMCAAgent):
    """
    Specialized Strange MCA agent for playing chess.
    """
    
    @classmethod
    def _extract_chess_move(cls, text: str) -> str:
        """
        Extract a chess move from the text.
        Looks for common move patterns and returns the first match.
        
        Args:
            text: The text to extract a move from
            
        Returns:
            The extracted move, formatted with square brackets
        """
        # Try to find a move in UCI format (e.g., e2e4) or algebraic notation (e.g., Nf3)
        # First look for moves already in brackets
        bracket_pattern = r'\[([a-h][1-8][a-h][1-8]|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8]|O-O(?:-O)?)\]'
        # Then look for UCI format moves
        uci_pattern = r'\b([a-h][1-8][a-h][1-8])\b'
        # Then look for algebraic notation moves
        algebraic_pattern = r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8]|O-O(?:-O)?)\b'
        
        # Check for moves in brackets first
        bracket_match = re.search(bracket_pattern, text)
        if bracket_match:
            return f"[{bracket_match.group(1)}]"
        
        # Check for UCI format moves
        uci_match = re.search(uci_pattern, text)
        if uci_match:
            return f"[{uci_match.group(1)}]"
        
        # Check for algebraic notation moves
        algebraic_match = re.search(algebraic_pattern, text)
        if algebraic_match:
            return f"[{algebraic_match.group(1)}]"
        
        # If no pattern matches, return the original text
        return text
    
    @classmethod
    def extract_move(cls, text: str) -> str:
        """
        Extract a chess move from the text.
        Looks for common move patterns and returns the first match.
        
        Args:
            text: The text to extract a move from
            
        Returns:
            The extracted move, formatted with square brackets
        """
        return cls._extract_chess_move(text)
    
    def process_response(self, response: str, iteration: int, original_task: str) -> str:
        """
        Process the response to ensure it's a valid chess move.
        
        Args:
            response: The response from the Strange MCA system.
            iteration: The current iteration number (0-based).
            original_task: The original task given to the system.
            
        Returns:
            The processed response with a properly formatted chess move.
        """
        move = self.extract_move(response)
        if move != response:
            print(f"Iteration {iteration+1}: Reformatted move from '{response}' to '{move}'")
            return move
        return response
    
    def __init__(
        self,
        child_per_parent: int = 2,
        depth: int = 2,
        model: str = "gpt-3.5-turbo",
        viz: bool = False,
        print_details: bool = False,
        max_iterations: int = 1,
    ):
        """
        Initialize the Strange MCA Chess agent.
        
        Args:
            child_per_parent: Number of children per parent node in the agent tree.
            depth: Depth of the agent tree.
            model: The model to use for the agents.
            viz: Whether to generate visualizations.
            print_details: Whether to print detailed information.
            max_iterations: Maximum number of iterations to refine the response.
        """
        # Create a chess-specific task template
        chess_template = textwrap.dedent("""
        You are playing a game of chess. Analyze the board carefully and make a strategic move.
        
        Current game state:
        
        {observation}
        
        Your task is to decide on the best chess move to make in this position.
        Provide your move in standard algebraic notation (e.g., 'e2e4', 'Nf3', etc.) or in UCI format.
        Your final answer should be just the move notation, without any additional text.
        """)
        
        super().__init__(
            child_per_parent=child_per_parent,
            depth=depth,
            model=model,
            viz=viz,
            print_details=print_details,
            task_template=chess_template,
            max_iterations=max_iterations,
        )

