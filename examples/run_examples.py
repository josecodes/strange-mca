"""
Example script demonstrating how to use the run_strange_mca function.

This script shows how to run the Strange MCA system with different configurations.
"""

import os
import sys
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.strange_mca.run_strange_mca import run_strange_mca


def run_examples() -> List[Dict[str, Any]]:
    """
    Run a series of examples with different configurations.
    
    Returns:
        A list of results from each example run.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("examples")
    
    results = []
    
    # Example 1: Basic configuration
    logger.info("Running Example 1: Basic configuration")
    result1 = run_strange_mca(
        task="Explain the concept of recursion in programming",
        child_per_parent=2,
        depth=2,
        model="gpt-3.5-turbo",
        log_level="info",
        viz=True,
    )
    results.append(result1)
    
    # Example 2: More children per parent
    logger.info("\n\nRunning Example 2: More children per parent")
    result2 = run_strange_mca(
        task="Describe the benefits and challenges of remote work",
        child_per_parent=3,
        depth=2,
        model="gpt-3.5-turbo",
        log_level="info",
        viz=True,
    )
    results.append(result2)
    
    # Example 3: Deeper tree
    logger.info("\n\nRunning Example 3: Deeper tree")
    result3 = run_strange_mca(
        task="Analyze the impact of artificial intelligence on society",
        child_per_parent=2,
        depth=3,
        model="gpt-3.5-turbo",
        log_level="info",
        viz=True,
    )
    results.append(result3)
    
    # Example 4: Custom output directory
    logger.info("\n\nRunning Example 4: Custom output directory")
    custom_output_dir = "output/custom_example"
    os.makedirs(custom_output_dir, exist_ok=True)
    result4 = run_strange_mca(
        task="Summarize the history of computing",
        child_per_parent=2,
        depth=2,
        model="gpt-3.5-turbo",
        log_level="info",
        viz=True,
        output_dir=custom_output_dir,
    )
    results.append(result4)
    
    # Example 5: No visualizations
    logger.info("\n\nRunning Example 5: No visualizations")
    result5 = run_strange_mca(
        task="Provide tips for effective time management",
        child_per_parent=2,
        depth=2,
        model="gpt-3.5-turbo",
        log_level="info",
        viz=False,
    )
    results.append(result5)
    
    return results


if __name__ == "__main__":
    results = run_examples()
    print(f"\nCompleted {len(results)} example runs.") 