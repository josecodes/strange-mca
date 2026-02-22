"""
Prompt functions for the emergent MCA system.

Provides prompts for competency (system prompts), initial response, lateral
communication, parent observation, downward signals, and strange loop
self-reflection.
"""

import re
from typing import Optional


def create_competency_prompt(role: str, perspective: str = "") -> str:
    """Create a competency system prompt for an agent based on its role.

    Args:
        role: One of 'specialist', 'coordinator', or 'integrator'.
        perspective: The agent's unique perspective (specialists only).

    Returns:
        A system prompt string.
    """
    if role == "specialist":
        return (
            f"You are a specialist analyst in a collaborative research team. Your unique "
            f"perspective is: {perspective}.\n\n"
            f"Your goal is to contribute the most valuable insight you can from your area "
            f"of expertise. You are NOT being assigned a subtask — you independently analyze "
            f"the full problem through your lens.\n\n"
            f"When communicating with peers, maintain your perspective while being open to "
            f"integration. Be concise and substantive."
        )
    elif role == "coordinator":
        return (
            "You are a mid-level coordinator in a collaborative research team. You observe "
            "the outputs of several specialist analysts (your team).\n\n"
            "Your goal is to identify emergent patterns, contradictions, and synergies in "
            "their work — things that arise from the COMBINATION of perspectives that no "
            "individual captured alone. You do NOT assign tasks or tell your team what to do. "
            "You observe and synthesize.\n\n"
            "When communicating with peer coordinators, share your synthesis and look for "
            "cross-team patterns. Focus on what emerges from the combination."
        )
    else:  # integrator
        return (
            "You are the lead integrator of a collaborative research team. You observe the "
            "outputs of all coordinators and their teams.\n\n"
            "Your goal is to produce a coherent, holistic response that captures the best "
            "emergent insights from across the entire team. You do NOT direct the process. "
            "You make sense of what has emerged from below."
        )


def create_initial_response_prompt(
    task: str,
    perspective: str,
    round_num: int,
    previous_response: Optional[str] = None,
    parent_signal: Optional[str] = None,
) -> str:
    """Create a prompt for a leaf agent's independent response.

    Args:
        task: The original task.
        perspective: The agent's perspective.
        round_num: Current round number.
        previous_response: Agent's response from previous round (round 2+).
        parent_signal: Downward signal from parent (round 2+, if enabled).

    Returns:
        A prompt string.
    """
    parts = [f"TASK: {task}"]

    if round_num > 1 and previous_response:
        parts.append(
            f"\nYOUR PREVIOUS RESPONSE (round {round_num - 1}):\n{previous_response}"
        )

    if round_num > 1 and parent_signal:
        parts.append(
            f"\nSIGNAL FROM YOUR COORDINATOR:\n{parent_signal}\n\n"
            "This signal highlights areas your coordinator noticed. You may incorporate "
            "it or stay with your current analysis if you believe it's stronger."
        )

    if round_num == 1:
        parts.append(
            f"\nAnalyze this task from your {perspective} perspective. "
            "Provide your independent analysis."
        )
    else:
        parts.append(
            f"\nProvide your updated analysis from your {perspective} perspective, "
            "building on your previous response and any signals received."
        )

    return "\n".join(parts)


def create_lateral_prompt(
    task: str,
    own_response: str,
    peer_responses: dict[str, str],
    round_num: int,
) -> str:
    """Create a prompt for lateral communication with peers.

    Args:
        task: The original task.
        own_response: The agent's own response this round.
        peer_responses: Dict mapping peer names to their responses.
        round_num: Current round number.

    Returns:
        A prompt string.
    """
    parts = [
        f"You previously responded to this task:\n\nTASK: {task}\n\n"
        f"YOUR RESPONSE:\n{own_response}\n\n"
        "Your peer specialists have also responded. Here are their perspectives:\n"
    ]

    for name, response in peer_responses.items():
        parts.append(f"--- {name} ---\n{response}\n")

    parts.append(
        "\nConsider your peers' contributions. You should:\n"
        "- MAINTAIN your unique perspective — do not abandon your viewpoint\n"
        "- IDENTIFY contradictions or tensions between your response and others\n"
        "- FILL GAPS that others may have missed from your vantage point\n"
        "- SHARPEN your contribution by noting where it adds something others don't cover\n"
        "- NOTE any genuine disagreements — disagreement is valuable, not a problem to fix\n\n"
        "Provide your revised response. If your original response already captures "
        "your best contribution given what your peers have said, you may restate it "
        "with minor adjustments."
    )

    return "\n".join(parts)


def create_observation_prompt(
    task: str,
    child_responses: dict[str, str],
    own_previous: Optional[str] = None,
    round_num: int = 1,
) -> str:
    """Create a prompt for a parent agent to observe children's outputs.

    Args:
        task: The original task.
        child_responses: Dict mapping child names to their latest responses.
        own_previous: Parent's own synthesis from the previous round.
        round_num: Current round number.

    Returns:
        A prompt string.
    """
    parts = [
        f"You are observing the outputs of your team in response to:\n\nTASK: {task}\n\n"
        "TEAM RESPONSES:\n"
    ]

    for name, response in child_responses.items():
        parts.append(f"--- {name} ---\n{response}\n")

    if own_previous:
        parts.append(
            f"\nYOUR PREVIOUS SYNTHESIS (round {round_num - 1}):\n{own_previous}\n"
        )

    parts.append(
        "\nSynthesize what you observe. Focus specifically on:\n"
        "1. What EMERGES from the combination of these perspectives that no individual "
        "captured alone?\n"
        "2. What contradictions or tensions exist between perspectives? Are these "
        "productive tensions (revealing genuine complexity) or errors?\n"
        "3. What is MISSING — what question does this collective response fail to address?\n\n"
        "Do NOT simply summarize each person's contribution. Your value is in seeing "
        "patterns, connections, and gaps across the whole."
    )

    return "\n".join(parts)


def create_signal_prompt(
    task: str,
    child_responses: dict[str, str],
    own_synthesis: str,
) -> str:
    """Create a prompt for generating a downward signal to children.

    Args:
        task: The original task.
        child_responses: Dict mapping child names to their responses.
        own_synthesis: The parent's synthesis.

    Returns:
        A prompt string.
    """
    parts = [
        f"Based on your synthesis of your team's work on:\n\nTASK: {task}\n\n"
        "You notice the following gaps or tensions that your team might address "
        "in their next round:\n\n"
        "Generate a brief (2-3 sentence) signal to your team. This should:\n"
        "- Highlight blind spots or underexplored areas\n"
        "- Note productive tensions worth developing further\n"
        "- Be SUGGESTIVE, not DIRECTIVE — your team decides how to respond\n\n"
        "TEAM RESPONSES:\n"
    ]

    for name, response in child_responses.items():
        parts.append(f"--- {name} ---\n{response}\n")

    parts.append(f"\nYOUR SYNTHESIS:\n{own_synthesis}")

    return "\n".join(parts)


def create_signal_response_prompt(
    task: str,
    own_response: str,
    parent_signal: str,
    round_num: int,
) -> str:
    """Create a prompt for responding to a parent's downward signal.

    Args:
        task: The original task.
        own_response: The agent's current response.
        parent_signal: The signal from the parent.
        round_num: Current round number.

    Returns:
        A prompt string.
    """
    return (
        f"TASK: {task}\n\n"
        f"YOUR CURRENT RESPONSE:\n{own_response}\n\n"
        f"SIGNAL FROM YOUR COORDINATOR:\n{parent_signal}\n\n"
        "Your coordinator has highlighted areas for attention. Consider whether "
        "this signal reveals gaps in your analysis. You may adjust your response "
        "or maintain it if you believe your current analysis is stronger.\n\n"
        "Provide your response."
    )


def create_strange_loop_prompt(
    original_task: str, tentative_response: str, domain_specific_instructions: str = ""
) -> str:
    """Create a prompt for the strange loop.

    Args:
        original_task: The original task to complete.
        tentative_response: The tentative response from the team.
        domain_specific_instructions: Optional domain-specific instructions.

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

    {domain_specific_instructions}

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

    # If regex fails, try line-by-line parsing
    lines = response.split("\n")
    final_response_lines = []
    in_final_response = False
    found_asterisks = False
    for line in lines:
        line_stripped = line.strip()
        if line_stripped == "Final Response:" or line_stripped.startswith(
            "Final Response:"
        ):
            in_final_response = False
            found_asterisks = False
            continue
        if (
            not in_final_response
            and not found_asterisks
            and line_stripped.startswith("*****")
        ):
            found_asterisks = True
            continue
        if found_asterisks and not in_final_response:
            in_final_response = True
        if in_final_response and line_stripped.startswith("*****"):
            break
        if in_final_response:
            final_response_lines.append(line)
    if final_response_lines:
        return "\n".join(final_response_lines).strip()

    # If all else fails, return the original response
    return response.strip()
