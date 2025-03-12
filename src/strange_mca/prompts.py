"""
Prompt-related functions for the Strange MCA system.

This module contains functions for creating various prompts used in the system,
as well as parsing responses from those prompts.
"""

import re
from typing import Dict, List


def create_task_decomposition_prompt(task: str, context: str, child_nodes: List[str]) -> str:
    """Create a prompt for task decomposition.
    
    Args:
        task: The original task to decompose.
        context: The context for the agent that will decompose the task.
        child_nodes: List of child node names that will receive the subtasks.
        
    Returns:
        A prompt for task decomposition.
    """
    child_nodes_str = "\n".join([f"- {node}" for node in child_nodes])
    
    return f"""

{context}

Your task is to break down the following task into subtasks for your team members:

{task}

You have the following team members that need assignments:
{child_nodes_str}

For each team member, create a specific subtask that:
1. Is clearly described and actionable
2. Includes any specific instructions or constraints
3. Contributes meaningfully to the overall task

Format your response with one subtask per team member, using their exact name as the prefix:

{child_nodes[0]}: [Subtask description for this team member]
{child_nodes[1] if len(child_nodes) > 1 else "[Next team member]"}: [Subtask description for this team member]
...and so on for all team members.

Make sure each team member has exactly one subtask assigned to them."""


def create_synthesis_prompt(child_responses: Dict[str, str]) -> str:
    """Create a prompt for synthesizing responses from child nodes.
    
    Args:
        child_responses: Dictionary mapping child node names to their responses.
        
    Returns:
        A prompt for synthesizing responses.
    """
    formatted_responses = "\n\n".join([
        f"Agent {child}: {response}"
        for child, response in child_responses.items()
    ])
    
    return f"""Synthesize the following responses from your team members:

{formatted_responses}

Your task is to:
1. Integrate the key insights from each response
2. Resolve any contradictions or inconsistencies
3. Provide a coherent and concise answer

Format your response as a well-structured summary."""


def create_strange_loop_prompt(original_task: str, tentative_response: str) -> str:
    """Create a prompt for the strange loop.
    
    Args:
        original_task: The original task to complete.
        tentative_response: The tentative response from the team.
        
    Returns:
        A prompt for the strange loop.
    """
    return f"""
    
 
    I was given the following task to complete:

    Task:     
    **************************************************
    {original_task}
    **************************************************

    I produced this response:

     Response: 
    **************************************************
    {tentative_response}
    **************************************************
    
    Is this the best response I can provide for the task? If so, then I'll simply provide that is the final response.

    If it could be improved upon, I will make revisions and produce the final response.

    I will format the revised final response  with in the following format:
    
    Final Response:
    **************************************************    
    [Final response]
    **************************************************

    After this section, I'll provide a brief explanation of reasoning for revisions made (or lack thereof).

    """



def parse_strange_loop_response(response: str) -> str:
    """Extract the final response from the strange loop output.
    
    Args:
        response: The full response from the strange loop prompt.
        
    Returns:
        The extracted final response text, or the original response if no final response section found.
    """
    # Handle empty or None response
    if not response:
        return ""
    
    # Try to find the final response section using regex pattern matching
    pattern = r"Final Response:\s*\n\*{10,}\s*\n(.*?)\n\*{10,}"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # # If regex fails, try line-by-line parsing
    # lines = response.split('\n')
    # final_response_lines = []
    # in_final_response = False
    # found_asterisks = False  
    # for line in lines:
    #     line_stripped = line.strip()
    #     if line_stripped == "Final Response:" or line_stripped.startswith("Final Response:"):
    #         in_final_response = False
    #         found_asterisks = False
    #         continue
    #     if not in_final_response and not found_asterisks and line_stripped.startswith('*****'):
    #         found_asterisks = True
    #         continue
    #     if found_asterisks and not in_final_response:
    #         in_final_response = True
    #     if in_final_response and line_stripped.startswith('*****'):
    #         break
    #     if in_final_response:
    #         final_response_lines.append(line)
    # if final_response_lines:
    #     return '\n'.join(final_response_lines).strip()
    # return response.strip()