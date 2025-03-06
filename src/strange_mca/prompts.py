"""System prompts for the multiagent system."""
#todo: refactor all this
from typing import Dict, List


def get_supervisor_prompt(child_per_parent: int, depth: int) -> str:
    """Get the system prompt for the supervisor agent.
    
    Args:
        child_per_parent: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
        
    Returns:
        The system prompt for the supervisor agent.
    """
    # Generate a list of child names
    child_names = [f"L2N{i}" for i in range(1, child_per_parent + 1)]
    child_list = ", ".join(child_names)
    
    return (
        f"You are the supervisor agent (L1N1) responsible for coordinating a team of {child_per_parent} agents: {child_list}. "
        "Your role is to break down complex tasks into simpler subtasks and assign them "
        "to your team members. You will receive their responses and synthesize a final answer."
    )


def get_team_member_prompts(child_per_parent: int) -> Dict[str, str]:
    """Get the system prompts for team member agents.
    
    Args:
        child_per_parent: The number of team members.
        
    Returns:
        A dictionary mapping agent names to their system prompts.
    """
    prompts = {}
    
    # Define specialized roles based on the agent's position
    specializations = {
        1: "analyzing complex concepts and breaking them down into simpler terms",
        2: "providing detailed explanations with examples",
        3: "connecting ideas to real-world applications",
        4: "evaluating different perspectives and providing balanced views",
        5: "researching historical context and development of ideas",
        6: "identifying potential challenges and limitations",
        7: "proposing innovative solutions and future directions",
        8: "summarizing and synthesizing information concisely",
        9: "explaining technical concepts to non-experts",
    }
    
    # Create prompts for each team member
    for i in range(1, min(child_per_parent + 1, 10)):
        specialization = specializations.get(i, "providing expert analysis and insights")
        prompts[f"L2N{i}"] = (
            f"You are agent L2N{i}, a specialized agent working as part of a team. "
            "Your parent agent is L1N1. "
            f"You excel at {specialization}. "
            "You will receive tasks from your parent and should complete them to the best of your ability."
        )
    
    return prompts


def update_agent_prompts(
    agent_configs: Dict[str, "AgentConfig"],
    child_per_parent: int,
    depth: int,
) -> Dict[str, "AgentConfig"]:
    """Update the system prompts for all agents.
    
    Args:
        agent_configs: The agent configurations to update.
        child_per_parent: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
        
    Returns:
        The updated agent configurations.
    """
    # Import here to avoid circular imports
    from src.strange_mca.agents import AgentConfig
    
    # Get the supervisor prompt
    supervisor_prompt = get_supervisor_prompt(child_per_parent, depth)
    
    # Update the supervisor prompt
    agent_configs["L1N1"].system_prompt = supervisor_prompt
    
    # Get team member prompts
    team_member_prompts = get_team_member_prompts(child_per_parent)
    
    # Update team member prompts
    for name, prompt in team_member_prompts.items():
        if name in agent_configs:
            agent_configs[name].system_prompt = prompt
    
    return agent_configs 