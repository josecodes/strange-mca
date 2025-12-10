"""Agent definitions for the multiagent system."""

from typing import Optional

from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Configuration for an agent in the system."""

    name: str
    level: int
    node_number: int
    system_prompt: Optional[str] = ""

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
            template=("Task: {task}\n\n" "Your response:")
        )

    def run(self, task: str) -> str:
        """Run the agent on a task.

        Args:
            task: The task to perform.

        Returns:
            The agent's response.
        """
        # Format the prompt
        prompt = self.prompt_template.format(task=task)

        # Create messages
        messages = [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=prompt),
        ]

        # Invoke the LLM
        response = self.llm.invoke(messages)
        return response.content

    def __repr__(self) -> str:
        return f"Agent({self.config.full_name})"
