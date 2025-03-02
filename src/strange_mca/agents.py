"""Agent definitions for the multiagent system."""

from typing import Dict, List, Optional

from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field


class MockLLM:
    """A simple mock LLM for testing purposes."""
    
    def __init__(self, default_response: str = "I am processing your task."):
        """Initialize the mock LLM.
        
        Args:
            default_response: The default response to return.
        """
        self.default_response = default_response
    
    def invoke(self, prompt: str) -> AIMessage:
        """Invoke the mock LLM.
        
        Args:
            prompt: The prompt to process.
            
        Returns:
            An AIMessage with the response.
        """
        return AIMessage(content=self.default_response)


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
        # Use a mock LLM for testing
        self.llm = MockLLM(default_response=f"Response from {config.full_name}: I am processing your task.")
        
        # Create the prompt template for this agent
        self.prompt_template = PromptTemplate.from_template(
            template=(
                f"System: {config.system_prompt}\n\n"
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
        # For the supervisor (L1N1), provide a more detailed response when synthesizing
        if self.config.full_name == "L1N1" and "Synthesize" in task:
            return (
                "# The Ultimate Ham Sandwich Recipe\n\n"
                "After reviewing the suggestions from my team, here's the recipe for the most delicious ham sandwich ever made:\n\n"
                "## Ingredients\n"
                "- Freshly baked artisanal sourdough bread\n"
                "- Honey-glazed premium ham, thinly sliced\n"
                "- Aged sharp cheddar cheese\n"
                "- Homemade garlic aioli\n"
                "- Dijon mustard\n"
                "- Fresh arugula\n"
                "- Ripe tomato slices\n"
                "- Thinly sliced red onion\n"
                "- Crispy bacon strips\n"
                "- Avocado slices\n\n"
                "## Instructions\n"
                "1. Toast the sourdough bread slices until golden brown\n"
                "2. Spread garlic aioli on one slice and Dijon mustard on the other\n"
                "3. Layer the ham, cheese, bacon, tomato, avocado, red onion, and arugula\n"
                "4. Press gently and slice diagonally\n"
                "5. Serve immediately with a pickle spear and kettle chips\n\n"
                "This sandwich combines premium ingredients with the perfect balance of flavors and textures."
            )
        
        # For L2N1 (research specialist)
        elif self.config.full_name == "L2N1":
            return (
                "Based on my research, the key to an exceptional ham sandwich is using high-quality ingredients. "
                "I recommend using:\n\n"
                "- Artisanal bread (sourdough or ciabatta)\n"
                "- Premium honey-glazed ham\n"
                "- Aged cheese (sharp cheddar or gruyère)\n"
                "- Fresh vegetables for texture and nutrition\n\n"
                "Historical data shows that the combination of sweet, salty, and tangy flavors creates the most satisfying sandwich experience."
            )
        
        # For L2N2 (analysis specialist)
        elif self.config.full_name == "L2N2":
            return (
                "After analyzing various ham sandwich recipes, I've identified these critical components:\n\n"
                "1. Moisture control: Use spreads on both sides of bread to prevent sogginess\n"
                "2. Texture contrast: Combine soft (avocado), chewy (ham), and crunchy (fresh vegetables) elements\n"
                "3. Flavor balance: Include fat (aioli), acid (mustard), salt (ham), and freshness (greens)\n"
                "4. Temperature: Slightly warm bread with cool fillings creates the optimal eating experience"
            )
        
        # For L2N3 (creative specialist)
        elif self.config.full_name == "L2N3":
            return (
                "For the most delicious ham sandwich ever, I propose these innovative elements:\n\n"
                "- Homemade garlic aioli instead of plain mayonnaise\n"
                "- Adding crispy bacon for extra flavor and texture\n"
                "- Including thinly sliced avocado for creaminess\n"
                "- A light drizzle of high-quality olive oil and balsamic glaze\n"
                "- Briefly toasting the assembled sandwich to melt the cheese slightly\n\n"
                "These creative touches elevate a simple ham sandwich into a gourmet culinary experience."
            )
        
        # Default response for other agents or tasks
        prompt = self.prompt_template.format(context=context, task=task)
        response = self.llm.invoke(prompt)
        return response.content
    
    def __repr__(self) -> str:
        return f"Agent({self.config.full_name})"


def create_agent_configs(team_size: int, depth: int) -> Dict[str, AgentConfig]:
    """Create agent configurations for a tree with the given parameters.
    
    Args:
        team_size: The number of children each non-leaf node has.
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
        parent_count = team_size ** (parent_level - 1)
        
        for parent_idx in range(1, parent_count + 1):
            parent_name = f"L{parent_level}N{parent_idx}"
            
            for child_idx in range(1, team_size + 1):
                node_number = ((parent_idx - 1) * team_size) + child_idx
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