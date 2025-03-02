"""Agent definitions for the multiagent system."""

from typing import Dict, List, Optional

from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for an agent in the system."""
    
    name: str
    level: int
    node_number: int
    system_prompt: str
    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    
    @property
    def full_name(self) -> str:
        """Return the full name of the agent (e.g., L1N1)."""
        return f"L{self.level}N{self.node_number}"


class Agent:
    """An agent in the multiagent system."""
    
    def __init__(self, config: AgentConfig, model_name: str = "gpt-3.5-turbo"):
        """Initialize the agent.
        
        Args:
            config: The agent configuration.
            model_name: The name of the LLM model to use.
        """
        self.config = config
        # Use a real OpenAI LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        
        # Create the prompt template for this agent
        self.prompt_template = PromptTemplate.from_template(
            template=(
                "Context: {context}\n\n"
                "Task: {task}\n\n"
                "Your response:"
            )
        )
    
    def run(self, context: str, task: str) -> str:
        """Run the agent on a task.
        
        Args:
            context: The context for the task.
            task: The task to perform.
            
        Returns:
            The agent's response.
        """
        # Format the prompt
        prompt = self.prompt_template.format(context=context, task=task)
        
        # Create messages
        messages = [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        # Invoke the LLM
        response = self.llm.invoke(messages)
        return response.content
    
    def __repr__(self) -> str:
        return f"Agent({self.config.full_name})"


def create_agent_configs(child_per_parent: int, depth: int) -> Dict[str, AgentConfig]:
    """Create agent configurations for a tree with the given parameters.
    
    Args:
        child_per_parent: The number of children each non-leaf node has.
        depth: The number of levels in the tree.
        
    Returns:
        A dictionary mapping agent names to their configurations.
    """
    configs = {}
    
    # Create the root node
    root_name = "L1N1"
    root_config = AgentConfig(
        name=root_name,
        level=1,
        node_number=1,
        system_prompt=(
            "You are the supervisor agent responsible for coordinating a team of agents. "
            "Your role is to break down complex tasks into simpler subtasks and assign them "
            "to your team members. You will receive their responses and synthesize a final answer."
        )
    )
    configs[root_name] = root_config
    
    # Create the rest of the tree
    for level in range(2, depth + 1):
        parent_level = level - 1
        parent_count = child_per_parent ** (parent_level - 1)
        
        for parent_idx in range(1, parent_count + 1):
            parent_name = f"L{parent_level}N{parent_idx}"
            
            for child_idx in range(1, child_per_parent + 1):
                node_number = ((parent_idx - 1) * child_per_parent) + child_idx
                child_name = f"L{level}N{node_number}"
                
                child_config = AgentConfig(
                    name=child_name,
                    level=level,
                    node_number=node_number,
                    parent=parent_name,
                    system_prompt=(
                        f"You are agent {child_name}, a specialized agent working as part of a team. "
                        f"Your parent agent is {parent_name}. "
                        "You will receive tasks from your parent and should complete them to the best of your ability."
                    )
                )
                
                configs[child_name] = child_config
                configs[parent_name].children.append(child_name)
    
    return configs 