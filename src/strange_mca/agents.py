"""Agent definitions for the emergent MCA system.

Provides topology-aware AgentConfig, Agent wrapper around ChatOpenAI,
perspective assignment, and build_agent_tree() for constructing the full
agent hierarchy with competency prompts.
"""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.strange_mca.prompts import create_competency_prompt
from src.strange_mca.tree_helpers import (
    generate_all_nodes,
    get_children,
    get_parent,
    get_siblings,
    parse_node_name,
)

PERSPECTIVES = [
    "analytical",
    "creative",
    "critical",
    "practical",
    "theoretical",
    "empirical",
    "ethical",
    "systemic",
]


class AgentConfig(BaseModel):
    """Configuration for an agent in the MCA hierarchy."""

    name: str
    level: int
    node_number: int
    depth: int
    system_prompt: Optional[str] = ""
    siblings: list[str] = []
    children: list[str] = []
    parent: Optional[str] = None
    perspective: str = ""

    @property
    def full_name(self) -> str:
        """Return the full name of the agent (e.g., L1N1)."""
        return f"L{self.level}N{self.node_number}"

    @property
    def is_leaf(self) -> bool:
        """Check if this agent is a leaf node."""
        return self.level == self.depth

    @property
    def is_root(self) -> bool:
        """Check if this agent is the root node."""
        return self.level == 1

    @property
    def role(self) -> str:
        """Return the role of this agent."""
        if self.is_root:
            return "integrator"
        if self.is_leaf:
            return "specialist"
        return "coordinator"


class Agent:
    """An agent in the MCA system."""

    def __init__(self, config: AgentConfig, model_name: str = "gpt-4o-mini"):
        """Initialize the agent.

        Args:
            config: The agent configuration.
            model_name: The name of the LLM model to use.
        """
        self.config = config
        self.system_prompt = config.system_prompt or ""
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)

    def invoke(self, prompt: str) -> str:
        """Pass prompt directly as a HumanMessage.

        Args:
            prompt: The prompt to send.

        Returns:
            The LLM response content.
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content

    def __repr__(self) -> str:
        return f"Agent({self.config.full_name})"


def build_agent_tree(
    cpp: int,
    depth: int,
    model_name: str,
    perspectives: Optional[list[str]] = None,
) -> dict[str, "Agent"]:
    """Build all MCA agents with topology-aware configs and competency prompts.

    Args:
        cpp: Children per parent.
        depth: Total tree depth.
        model_name: LLM model name.
        perspectives: Optional custom perspectives for leaf agents. If None, uses PERSPECTIVES.

    Returns:
        Dict mapping node name (e.g., "L2N3") to Agent instance.
    """
    perspective_pool = perspectives if perspectives else PERSPECTIVES
    agents = {}

    all_nodes = generate_all_nodes(cpp, depth)

    for node_name in all_nodes:
        level, num = parse_node_name(node_name)

        children_names = get_children(node_name, cpp, depth)
        parent_name = get_parent(node_name, cpp)
        sibling_names = get_siblings(node_name, cpp, depth)

        # Assign perspective to leaf agents
        perspective = ""
        if level == depth:
            perspective = perspective_pool[(num - 1) % len(perspective_pool)]

        # Determine role and create competency prompt
        if level == 1:
            role = "integrator"
        elif level == depth:
            role = "specialist"
        else:
            role = "coordinator"

        system_prompt = create_competency_prompt(role, perspective)

        config = AgentConfig(
            name=node_name,
            level=level,
            node_number=num,
            depth=depth,
            system_prompt=system_prompt,
            siblings=sibling_names,
            children=children_names,
            parent=parent_name,
            perspective=perspective,
        )

        agents[node_name] = Agent(config, model_name=model_name)

    return agents
