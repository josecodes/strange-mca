"""System prompts for the multiagent system."""

from typing import Dict, List


def get_supervisor_prompt(team_size: int, depth: int) -> str:
    """Get the system prompt for the supervisor agent.
    
    Args:
        team_size: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
        
    Returns:
        The system prompt.
    """
    child_names = [f"L2N{i}" for i in range(1, team_size + 1)]
    child_list = ", ".join(child_names)
    
    return (
        f"You are the supervisor agent (L1N1) responsible for coordinating a team of {team_size} agents: {child_list}. "
        "Your role is to break down complex tasks into simpler subtasks and assign them to your team members. "
        "You will receive their responses and synthesize a final answer. "
        "You should consider the strengths and specialties of each team member when assigning tasks."
    )


def get_team_member_prompts(team_size: int) -> Dict[str, str]:
    """Get system prompts for team members.
    
    Args:
        team_size: The number of team members.
        
    Returns:
        A dictionary mapping agent names to their system prompts.
    """
    specialties = {
        1: "research and information gathering",
        2: "critical analysis and evaluation",
        3: "creative problem-solving and innovation",
        4: "technical implementation and execution",
        5: "communication and explanation",
        6: "planning and organization",
        7: "risk assessment and mitigation",
        8: "ethical considerations and implications",
        9: "user experience and design",
    }
    
    prompts = {}
    for i in range(1, min(team_size + 1, 10)):
        specialty = specialties.get(i, "general problem-solving")
        prompts[f"L2N{i}"] = (
            f"You are agent L2N{i}, a specialized agent working as part of a team under the supervision of L1N1. "
            f"Your specialty is {specialty}. "
            "You will receive tasks from your supervisor and should complete them to the best of your ability, "
            f"focusing on your expertise in {specialty}."
        )
    
    return prompts


def update_agent_prompts(
    agent_configs: Dict[str, "AgentConfig"],
    team_size: int,
    depth: int,
) -> Dict[str, "AgentConfig"]:
    """Update agent prompts with more specific system prompts.
    
    Args:
        agent_configs: The agent configurations to update.
        team_size: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
        
    Returns:
        The updated agent configurations.
    """
    # Import here to avoid circular imports
    from strange_mca.agents import AgentConfig
    
    # Update the supervisor prompt
    supervisor_prompt = get_supervisor_prompt(team_size, depth)
    agent_configs["L1N1"].system_prompt = supervisor_prompt
    
    # Update team member prompts
    team_member_prompts = get_team_member_prompts(team_size)
    for name, prompt in team_member_prompts.items():
        if name in agent_configs:
            agent_configs[name].system_prompt = prompt
    
    return agent_configs 