# Design: Emergent MCA Redesign

**Author**: Jose Cortez + Claude
**Date**: 2026-02-01
**Status**: Draft
**Supersedes**: `gt-mca-rearchitecture-design.md`, `rfc-bottom-up-emergent-mca.md`

---

## 1. Summary

This document specifies a redesign of strange-mca to test whether LLM agents arranged in a hierarchy can produce emergent behavior through local interaction — the core claim of a Multiscale Competency Architecture. The current system is a top-down task decomposition tree. This redesign adds a round-based, bottom-up execution mode (`mca_mode=True`) alongside the existing system, preserving backward compatibility.

The key hypothesis: **when agents respond independently, communicate laterally with peers, and have their outputs observed (not directed) by parent agents, the collective output will exhibit properties that no individual agent produced.** This design optimizes for making that hypothesis testable.

Design decisions draw from both prior documents. The execution model follows the RFC's round-based bottom-up approach. Backward compatibility follows the rearchitecture design's dual-mode strategy. Convergence uses the RFC's deterministic metric. Observability is a first-class concern throughout, since the point is to evaluate whether emergence actually occurs.

---

## 2. Problem Statement

The current system implements top-down task decomposition: root decomposes → children execute assignments → root synthesizes. This is not an MCA. In Levin's MCA, each scale has its own competency, agents respond to local signals rather than directives, and higher-order behavior emerges from collective interaction.

The specific gaps (identified in both prior documents):
- Parents assign subtasks; children have no autonomy
- Siblings cannot communicate
- Synthesis is aggregation by a single LLM call, not emergence from interaction
- Self-reflection exists only at root
- No bottom-up information flow

See `rfc-bottom-up-emergent-mca.md` Section 2 and `gt-mca-rearchitecture-design.md` Section 1 for detailed analysis.

### Relevant GitHub Issues

| Issue | Title | Relationship |
|-------|-------|-------------|
| #5 | Explore bottom-up emergent behavior vs top-down decomposition | Core motivation |
| #6 | Add basal cognition / cultural system prompts | Competency prompts enable this |
| #7 | Improve multiscale layer interactions | Lateral communication addresses this |
| #10 | Add dynamic routing and structure | Round-based loop is a step toward this |
| #11 | Explore hierarchical strange loop layers | Deferred; round-based refinement covers this for v1 |

---

## 3. Design Principles

**Stimulus over assignment.** Every agent sees the original task. No agent is told what to do — each responds according to its assigned perspective and competency.

**Local communication.** Agents communicate with immediate neighbors: siblings (same parent) and parent. No global broadcast, no cross-subtree communication. This mirrors biological locality.

**Observation over direction.** Parent agents observe what children produce and identify emergent patterns. They do not prescribe, assign, or evaluate. Downward signals are optional nudges that children may ignore.

**Convergence over single-pass.** The system runs multiple rounds until the root's output stabilizes, rather than making a single traversal.

**Preserved hierarchy.** The tree structure is maintained. Each level represents a different scale of organization with qualitatively different competency prompts — specialists, coordinators, integrator.

**Backward compatibility.** The existing execution model, API, CLI, TextArena integration, and tests continue to work unchanged. MCA behavior is activated via `mca_mode=True`.

**Observability as a feature.** Every round's state is captured. The system produces enough data to determine whether emergence is happening or whether the output is just averaged-out mush.

---

## 4. Architecture Overview

Two execution modes sharing the same agent tree, state management, and output format:

```
                     run_strange_mca(task, mca_mode=False)
                              │
                     create_execution_graph()          ← existing nested subgraphs
                              │
                     Sequential down-up pass
                              │
                     Result + final_state.json


                     run_strange_mca(task, mca_mode=True)
                              │
                     create_mca_execution_graph()      ← NEW flat graph
                              │
                     Round-based bottom-up processing
                              │
                     Result + final_state.json + mca_report.json
```

The two graph builders share tree helper functions from `graph.py` and the `Agent` class from `agents.py`. They produce output in the same format (`final_response`, `strange_loops`), with MCA mode adding supplementary observability data.

---

## 5. Execution Model

### 5.1 Round-Based Processing

Each execution consists of multiple rounds. A round processes the full tree bottom-up with lateral communication at each level:

```
ROUND N:

  1. LEAF RESPOND
     All leaf agents generate responses to the original task.
     Round 1: independent response from each agent's perspective.
     Round 2+: agent has access to its own previous response and any parent signal.

  2. LEAF LATERAL
     Each leaf agent sees all siblings' responses and revises its own.
     Agents maintain their perspective while integrating peer insights.

  3. INTERNAL OBSERVE (per level, bottom to top)
     Each internal agent reads all its children's latest responses.
     Produces a synthesis: emergent patterns, contradictions, synergies.

  4. INTERNAL LATERAL (per level, bottom to top)
     Each internal agent sees sibling coordinators' syntheses and revises.

  5. ROOT OBSERVE
     Root reads all children's syntheses. Produces holistic integration.

  6. SIGNAL DOWN (optional, configurable)
     Each non-leaf agent generates a brief downward nudge for its children:
     gaps noticed, tensions identified, areas needing depth.
     Children may incorporate or ignore these in the next round.

  7. CONVERGENCE CHECK
     Compare root's output to previous round via Jaccard similarity.
     If converged or max_rounds reached → finalize.
     Otherwise → next round.

FINALIZE:
  Root applies strange loop self-reflection (preserved from current design).
  Emit final_response + observability data.
```

### 5.2 Leaf Node Processing

Leaf agents are specialists. Each has a unique perspective (e.g., "analytical", "critical", "practical"). In round 1, they respond independently. In round 2+, they have access to:
- Their own previous response
- Parent signal (if downward signals are enabled)

After initial response, each leaf sees all siblings' responses and can revise. The lateral prompt instructs agents to maintain their viewpoint while constructively engaging with peers — not to seek consensus, but to sharpen their contribution.

### 5.3 Internal Node Processing

Internal agents are coordinators. They observe their children's outputs and synthesize what emerges from the combination. The observation prompt specifically asks: "What arises from the combination of these responses that no individual captured?" This is the critical prompt for emergence — it pushes the coordinator beyond summarization.

After observing, coordinators see sibling coordinators' syntheses and can revise. This enables cross-subtree pattern recognition at each level.

### 5.4 Root Node Processing

The root is an integrator. It observes its children's syntheses and produces a holistic response. After the round loop converges, the existing strange loop self-reflection is applied (if `strange_loop_count > 0`).

---

## 6. Agent Model

### 6.1 AgentConfig (extended)

```python
class AgentConfig(BaseModel):
    # Existing fields — unchanged
    name: str
    level: int
    node_number: int
    system_prompt: Optional[str] = ""

    # New fields for MCA mode
    depth: int = 0                     # Total tree depth (0 = legacy mode)
    siblings: list[str] = []           # Peer node names (same parent)
    children: list[str] = []           # Child node names
    parent: Optional[str] = None       # Parent node name (None for root)
    perspective: str = ""              # Unique perspective (leaf agents only)

    @property
    def full_name(self) -> str:
        return f"L{self.level}N{self.node_number}"

    @property
    def is_leaf(self) -> bool:
        return self.depth > 0 and self.level == self.depth

    @property
    def is_root(self) -> bool:
        return self.level == 1

    @property
    def role(self) -> str:
        if self.depth == 0:
            return "legacy"
        if self.is_root:
            return "integrator"
        if self.is_leaf:
            return "specialist"
        return "coordinator"
```

New fields all have defaults that preserve existing behavior. Existing tests that construct `AgentConfig(name=..., level=..., node_number=...)` continue to work.

### 6.2 Agent Class

The `Agent` class gains one new method. The existing `run()` is unchanged.

```python
class Agent:
    def __init__(self, config: AgentConfig, model_name: str = "gpt-3.5-turbo"):
        self.config = config
        self.system_prompt = config.system_prompt or ""
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)

    def run(self, task: str) -> str:
        """Legacy mode — unchanged."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Task: {task}\n\nYour response:"),
        ]
        response = self.llm.invoke(messages)
        return response.content

    def invoke(self, prompt: str) -> str:
        """MCA mode — passes prompt directly without 'Task:' wrapper."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content
```

The `invoke()` method is the MCA entry point. It passes the prompt directly as a HumanMessage, without the `"Task: {task}\n\nYour response:"` wrapper that `run()` uses. MCA prompts are richer and self-contained — they don't need the wrapper.

The system prompt for MCA agents is the competency prompt (see Section 10), set at construction time via `AgentConfig.system_prompt`.

### 6.3 Perspective Assignment

Leaf agents each receive a unique perspective to ensure response diversity:

```python
PERSPECTIVES = [
    "analytical",    # Systematic, logical decomposition
    "creative",      # Novel connections, lateral thinking
    "critical",      # Assumptions, weaknesses, counterarguments
    "practical",     # Implementation, feasibility, real-world constraints
    "theoretical",   # First principles, formal frameworks
    "empirical",     # Evidence, data, observations
    "ethical",       # Values, stakeholders, consequences
    "systemic",      # Interactions, feedback loops, emergent effects
]
```

Assignment is deterministic: `perspective = PERSPECTIVES[(node_number - 1) % len(PERSPECTIVES)]`. Custom perspectives can be passed via API to override the defaults.

### 6.4 Topology Construction

```python
def build_agent_tree(
    cpp: int,
    depth: int,
    model_name: str,
    perspectives: Optional[list[str]] = None,
) -> dict[str, Agent]:
    """
    Build all MCA agents with topology-aware configs and competency prompts.

    Returns dict mapping node name (e.g., "L2N3") to Agent instance.
    """
```

This function:
1. Iterates through all levels and nodes using existing tree helpers
2. Computes `children`, `siblings`, `parent` for each node
3. Assigns `perspective` to leaf nodes
4. Generates competency system prompt via `create_competency_prompt(role, perspective)`
5. Creates `AgentConfig` with full topology
6. Instantiates `Agent` for each config
7. Returns `dict[str, Agent]`

---

## 7. Communication Model

### 7.1 Channels

```
        L1N1 (Integrator)
       /    \
   L2N1 <-> L2N2  (Coordinators — lateral within level)
   / \       / \
 L3N1 L3N2 L3N3 L3N4  (Specialists — lateral within sibling group)
 <------>   <-------->
```

**Lateral (peer-to-peer):** Agents at the same level sharing the same parent exchange full responses. Siblings under L2N1 see each other; siblings under L2N2 see each other. Cross-subtree lateral communication (L3N1 seeing L3N3) does not occur — this mirrors biological locality.

Exception: internal-level lateral communication crosses subtrees. L2N1 and L2N2 are siblings of L1N1, so they see each other's syntheses. This is analogous to how organ systems influence each other through shared organismal context.

**Upward (child → parent):** Parents read all children's latest responses. The parent does not issue commands. It observes and synthesizes.

**Downward (parent → child):** After synthesizing, a parent generates a brief signal — not a task assignment, but a nudge highlighting gaps, tensions, or areas needing depth. Children may incorporate this or ignore it. This channel is optional and disabled via `enable_downward_signals=False`.

### 7.2 Visibility Rules

| Agent Role | Can See | Cannot See |
|---|---|---|
| Specialist (leaf) | Original task, own history, sibling responses, parent signal | Cousins, grandparent, other subtrees |
| Coordinator (internal) | Original task, own history, sibling syntheses, all children's responses, parent signal | Agents beyond immediate neighborhood |
| Integrator (root) | Original task, own history, all children's syntheses | — |

### 7.3 Information Flow Per Round

```
Round N data flow:

  Leaf L3N1 ──response──► Siblings L3N2    (lateral)
  Leaf L3N2 ──response──► Siblings L3N1    (lateral)
       │                       │
       └──────────┬────────────┘
                  │
                  ▼
  Coordinator L2N1 ──observation──► Sibling L2N2  (lateral)
  Coordinator L2N2 ──observation──► Sibling L2N1  (lateral)
       │                       │
       └──────────┬────────────┘
                  │
                  ▼
  Integrator L1N1 (root observation)
                  │
                  ▼
  L2N1 ◄──signal── L1N1 ──signal──► L2N2   (downward, optional)
    │                                  │
    ▼                                  ▼
  L3N1 ◄──signal── L2N1          L2N2 ──signal──► L3N3
  L3N2 ◄──signal── L2N1          L2N2 ──signal──► L3N4
```

---

## 8. Convergence

### 8.1 Metric

Jaccard token similarity between the root's response in round N and round N-1:

```python
def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
```

**Why Jaccard over alternatives:**
- **LLM-as-judge**: Adds cost and is unreliable (the rearchitecture design acknowledged this — "LLMs are not good at self-evaluation")
- **Embedding cosine similarity**: More semantically accurate but adds an embedding model dependency
- **Jaccard**: Zero-cost, deterministic, sufficient for detecting stabilization. Known weakness: semantically identical responses with different wording score low. This causes extra rounds but not incorrect behavior.

Can be swapped for a better metric later via a `convergence_method` parameter.

### 8.2 Scope

**Root convergence only.** If the root's synthesis has stabilized, the system output has stabilized regardless of leaf-level variation. This is simpler, cheaper, and sufficient. Per-agent convergence is tracked in observability data but does not gate termination.

### 8.3 Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_rounds` | 3 | Safety valve — stop after this many rounds regardless |
| `convergence_threshold` | 0.85 | Jaccard similarity threshold for convergence |

`max_rounds` defaults to 3 (not 5 as in the RFC) to keep initial cost reasonable while still allowing meaningful iteration. Users can increase for deeper convergence exploration.

---

## 9. State Design

### 9.1 MCA State

A new `MCAState` TypedDict for the flat MCA graph. The existing `State` in `graph.py` is unchanged — it continues to serve the legacy nested subgraph execution.

```python
class AgentRoundData(TypedDict, total=False):
    response: str              # Agent's response this round
    lateral_response: str      # Agent's response after lateral communication
    revised: bool              # Whether lateral communication changed the response
    signal_sent: str           # Downward signal sent (non-leaf only)
    signal_received: str       # Downward signal received (non-root only)

class MCAState(TypedDict, total=False):
    # Task
    original_task: str

    # Agent data: agent_name -> list of AgentRoundData (index = round - 1)
    agent_history: dict[str, list[AgentRoundData]]

    # Round tracking
    current_round: int
    max_rounds: int
    convergence_threshold: float
    enable_downward_signals: bool

    # Convergence
    converged: bool
    convergence_scores: list[float]     # Root similarity per round

    # Final output (same keys as legacy mode)
    final_response: str
    strange_loops: list[dict[str, str]]
```

### 9.2 Design Rationale

- **`agent_history`** is a dict of lists, not a flat dict. Each agent's response history is indexed by round. This makes round-over-round comparison trivial.
- **`AgentRoundData`** captures both the initial response and the post-lateral response for each agent each round. The `revised` flag indicates whether lateral communication actually changed anything — critical for evaluating whether lateral communication is adding value.
- **`convergence_scores`** is a list of floats, one per round (starting from round 2). This gives a clear trajectory: is the system converging smoothly, oscillating, or stuck?
- The final output fields (`final_response`, `strange_loops`) match the legacy `State` so that downstream consumers (TextArena, output JSON) work identically.

---

## 10. Prompt Design

### 10.1 Competency System Prompts

Generated at build time per agent. Stored in `AgentConfig.system_prompt`.

**Specialist (leaf):**
```
You are a specialist analyst in a collaborative research team. Your unique
perspective is: {perspective}.

Your goal is to contribute the most valuable insight you can from your area
of expertise. You are NOT being assigned a subtask — you independently analyze
the full problem through your lens.

When communicating with peers, maintain your perspective while being open to
integration. Be concise and substantive.
```

**Coordinator (internal):**
```
You are a mid-level coordinator in a collaborative research team. You observe
the outputs of several specialist analysts (your team).

Your goal is to identify emergent patterns, contradictions, and synergies in
their work — things that arise from the COMBINATION of perspectives that no
individual captured alone. You do NOT assign tasks or tell your team what to do.
You observe and synthesize.

When communicating with peer coordinators, share your synthesis and look for
cross-team patterns. Focus on what emerges from the combination.
```

**Integrator (root):**
```
You are the lead integrator of a collaborative research team. You observe the
outputs of all coordinators and their teams.

Your goal is to produce a coherent, holistic response that captures the best
emergent insights from across the entire team. You do NOT direct the process.
You make sense of what has emerged from below.
```

### 10.2 Round Prompt Functions

Six new functions in `prompts.py`. All existing functions are unchanged.

| Function | Used By | Purpose |
|---|---|---|
| `create_competency_prompt(role, perspective)` | `build_agent_tree()` | Generate system prompt from role and perspective |
| `create_initial_response_prompt(task, perspective, round_num, previous_response)` | Leaf respond | First/subsequent independent response |
| `create_lateral_prompt(task, own_response, peer_responses, round_num)` | All levels, lateral phase | Revise after reading peers |
| `create_observation_prompt(task, child_responses, own_previous, round_num)` | Non-leaf, observe phase | Synthesize emergent patterns from children |
| `create_signal_prompt(task, child_responses, own_synthesis)` | Non-leaf, signal phase | Generate downward nudge |
| `create_signal_response_prompt(task, own_response, parent_signal, round_num)` | Non-root, round 2+ | Optionally adjust based on parent nudge |

### 10.3 Key Prompt: Lateral Communication

This prompt is the most important for emergence. It must balance two tensions: maintaining perspective diversity (don't converge to mush) and enabling genuine integration (don't just repeat yourself).

```
You previously responded to this task:

TASK: {task}

YOUR RESPONSE:
{own_response}

Your peer specialists have also responded. Here are their perspectives:

{for name, response in peer_responses:}
--- {name} ---
{response}

{end for}

Consider your peers' contributions. You should:
- MAINTAIN your unique perspective — do not abandon your viewpoint
- IDENTIFY contradictions or tensions between your response and others
- FILL GAPS that others may have missed from your vantage point
- SHARPEN your contribution by noting where it adds something others don't cover
- NOTE any genuine disagreements — disagreement is valuable, not a problem to fix

Provide your revised response. If your original response already captures
your best contribution given what your peers have said, you may restate it
with minor adjustments.
```

### 10.4 Key Prompt: Parent Observation

This prompt is critical for the emergence hypothesis. It must push the parent beyond summarization.

```
You are observing the outputs of your team in response to:

TASK: {task}

TEAM RESPONSES:
{for name, response in child_responses:}
--- {name} ---
{response}

{end for}

{if own_previous:}
YOUR PREVIOUS SYNTHESIS (round {round_num - 1}):
{own_previous}
{end if}

Synthesize what you observe. Focus specifically on:
1. What EMERGES from the combination of these perspectives that no individual
   captured alone?
2. What contradictions or tensions exist between perspectives? Are these
   productive tensions (revealing genuine complexity) or errors?
3. What is MISSING — what question does this collective response fail to address?

Do NOT simply summarize each person's contribution. Your value is in seeing
patterns, connections, and gaps across the whole.
```

### 10.5 Key Prompt: Downward Signal

Brief and non-directive. The signal is a nudge, not a command.

```
Based on your synthesis of your team's work on:

TASK: {task}

You notice the following gaps or tensions that your team might address
in their next round:

Generate a brief (2-3 sentence) signal to your team. This should:
- Highlight blind spots or underexplored areas
- Note productive tensions worth developing further
- Be SUGGESTIVE, not DIRECTIVE — your team decides how to respond

TEAM RESPONSES:
{child_responses}

YOUR SYNTHESIS:
{own_synthesis}
```

---

## 11. LangGraph Integration

### 11.1 Graph Topology (MCA Mode)

A single flat `StateGraph` with a conditional back-edge for the round loop. This replaces the nested subgraph pattern for MCA mode only.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  init ──► leaf_respond ──► leaf_lateral ──► [observe_L{n} ──►       │
│    ↑                                         lateral_L{n}]          │
│    │                                         ... per internal       │
│    │                                         level, bottom to top   │
│    │                                              │                 │
│    │                                              ▼                 │
│    │                                        observe_root            │
│    │                                              │                 │
│    │                                              ▼                 │
│    │                                        signal_down             │
│    │                                        (conditional)           │
│    │                                              │                 │
│    │                                              ▼                 │
│    │                                     check_convergence          │
│    │                                        │          │            │
│    │           (not converged) ◄────────────┘          │            │
│    │                │                            (converged)        │
│    └────────────────┘                                  │            │
│                                                   finalize ──► END  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

For `depth=2, cpp=3`:
```
init → leaf_respond → leaf_lateral → observe_root → signal_down → check_convergence → finalize
```

For `depth=3, cpp=2`:
```
init → leaf_respond → leaf_lateral → observe_L2 → lateral_L2 → observe_root → signal_down → check_convergence → finalize
```

### 11.2 Why a Flat Graph for MCA Mode

The RFC's argument is correct: nested subgraphs were designed for single-pass down-up traversal. For iterative rounds where every agent re-enters every round:
- Re-invoking compiled nested subgraphs each round adds overhead
- Peer state sharing across nested subgraphs requires complex plumbing
- A flat graph with Python loops inside node functions is simpler to debug and visualize
- LLM calls are the bottleneck, not graph traversal

The existing nested subgraph architecture is preserved for legacy mode. The choice of graph architecture is an implementation detail hidden behind `mca_mode`.

### 11.3 Node Functions

Each graph node function iterates over the relevant agents. Example:

```python
def leaf_respond(state: MCAState) -> dict:
    round_num = state["current_round"]
    task = state["original_task"]
    updates = {}

    for name in leaf_names:
        agent = agents[name]
        history = state["agent_history"].get(name, [])
        previous = history[-1]["response"] if history else None

        # Build prompt
        if round_num == 1:
            prompt = create_initial_response_prompt(
                task, agent.config.perspective, round_num, previous_response=None
            )
        else:
            signal = None
            if state.get("enable_downward_signals"):
                parent_name = agent.config.parent
                if history:
                    signal = history[-1].get("signal_received")
            prompt = create_initial_response_prompt(
                task, agent.config.perspective, round_num,
                previous_response=previous, parent_signal=signal
            )

        response = agent.invoke(prompt)
        round_data: AgentRoundData = {"response": response, "revised": False}
        updates[name] = history + [round_data]

    return {"agent_history": updates}
```

```python
def leaf_lateral(state: MCAState) -> dict:
    round_num = state["current_round"]
    task = state["original_task"]
    updates = {}

    for name in leaf_names:
        agent = agents[name]
        history = state["agent_history"][name]
        own_response = history[-1]["response"]

        # Gather sibling responses
        peer_responses = {}
        for sibling in agent.config.siblings:
            sib_history = state["agent_history"][sibling]
            peer_responses[sibling] = sib_history[-1]["response"]

        if not peer_responses:
            # Only child — no lateral communication
            history[-1]["lateral_response"] = own_response
            history[-1]["revised"] = False
            updates[name] = history
            continue

        prompt = create_lateral_prompt(task, own_response, peer_responses, round_num)
        lateral_response = agent.invoke(prompt)

        revised = lateral_response.strip() != own_response.strip()
        history[-1]["lateral_response"] = lateral_response
        history[-1]["revised"] = revised
        updates[name] = history

    return {"agent_history": updates}
```

### 11.4 Convergence as Conditional Edge

```python
def check_convergence(state: MCAState) -> dict:
    round_num = state["current_round"]
    threshold = state["convergence_threshold"]
    max_rounds = state["max_rounds"]

    root_history = state["agent_history"][root_name]
    scores = list(state.get("convergence_scores", []))

    if len(root_history) >= 2:
        prev = root_history[-2].get("lateral_response", root_history[-2]["response"])
        curr = root_history[-1].get("lateral_response", root_history[-1]["response"])
        score = compute_jaccard_similarity(prev, curr)
        scores.append(score)
        converged = score >= threshold
    else:
        converged = False

    if round_num >= max_rounds:
        converged = True  # Force termination

    return {
        "converged": converged,
        "convergence_scores": scores,
        "current_round": round_num + 1,
    }

# Conditional edge
builder.add_conditional_edges(
    "check_convergence",
    lambda state: "finalize" if state.get("converged", False) else "leaf_respond",
    {"leaf_respond": "leaf_respond", "finalize": "finalize"},
)
```

### 11.5 Finalize Node

```python
def finalize(state: MCAState) -> dict:
    # Get root's latest response
    root_history = state["agent_history"][root_name]
    root_response = root_history[-1].get("lateral_response", root_history[-1]["response"])

    # Apply strange loop if configured (reuse existing _apply_strange_loop)
    if strange_loop_count > 0:
        root_agent = agents[root_name]
        final_response, loops = _apply_strange_loop(
            root_agent, root_response, state["original_task"],
            strange_loop_count, domain_specific_instructions
        )
    else:
        final_response = root_response
        loops = []

    return {
        "final_response": final_response,
        "strange_loops": loops,
    }
```

### 11.6 Recursion Limit

```python
num_graph_nodes = 2 + (2 * num_internal_levels) + 2 + 1  # init, leaf pair, internal pairs, signal+convergence, finalize
recursion_limit = max_rounds * (num_graph_nodes + 5)
```

Minimum 50 to provide headroom.

### 11.7 Future: Parallel LLM Calls

Within `leaf_respond` and `leaf_lateral`, agents at the same level are independent. The architecture supports `asyncio.gather()` for parallel invocation. LangGraph supports `async def` node functions natively. Deferred to follow-up.

---

## 12. Observability & Evaluation Framework

This is the most important section. The entire redesign exists to test whether emergence happens. Without good observability, we can't answer the question.

### 12.1 MCA Report

Every MCA execution produces `mca_report.json` alongside `final_state.json`:

```json
{
  "task": "Explain the implications of quantum computing for cryptography",
  "config": {
    "cpp": 3,
    "depth": 2,
    "model": "gpt-4o-mini",
    "max_rounds": 3,
    "convergence_threshold": 0.85,
    "enable_downward_signals": true,
    "perspectives": ["analytical", "creative", "critical"]
  },
  "rounds": [
    {
      "round": 1,
      "agents": {
        "L2N1": {
          "role": "specialist",
          "perspective": "analytical",
          "response": "...",
          "lateral_response": "...",
          "revised": true,
          "signal_received": null
        },
        "L2N2": {
          "role": "specialist",
          "perspective": "creative",
          "response": "...",
          "lateral_response": "...",
          "revised": false,
          "signal_received": null
        },
        "L2N3": {
          "role": "specialist",
          "perspective": "critical",
          "response": "...",
          "lateral_response": "...",
          "revised": true,
          "signal_received": null
        },
        "L1N1": {
          "role": "integrator",
          "response": "...",
          "lateral_response": null,
          "revised": false,
          "signal_sent": "Consider the tension between..."
        }
      },
      "convergence_score": null
    },
    {
      "round": 2,
      "agents": { "..." : "..." },
      "convergence_score": 0.78
    },
    {
      "round": 3,
      "agents": { "..." : "..." },
      "convergence_score": 0.91
    }
  ],
  "convergence": {
    "converged": true,
    "rounds_used": 3,
    "score_trajectory": [0.78, 0.91]
  },
  "summary_metrics": {
    "total_llm_calls": 21,
    "lateral_revision_rate": 0.56,
    "per_agent_revision_counts": {
      "L2N1": 2, "L2N2": 1, "L2N3": 2
    }
  },
  "final_response": "..."
}
```

### 12.2 Emergence Evaluation Protocol

The report data supports answering these questions:

**Q1: Does lateral communication change anything?**
- Metric: `lateral_revision_rate` — fraction of lateral phases where agents actually revised their response.
- Compare: Run with `max_rounds=1` (no iteration) vs `max_rounds=3`. Is the output qualitatively different?
- Red flag: If `lateral_revision_rate` is near 0, agents aren't engaging with peers.

**Q2: Does the root's output contain something no child produced?**
- Analysis: Compare root's synthesis against each child's response. Look for claims, connections, or framings that appear only in the root's output.
- This requires human evaluation or an LLM-as-judge comparison (separate from the system itself).

**Q3: Do agents maintain perspective diversity across rounds?**
- Metric: Pairwise Jaccard similarity between sibling responses at end of each round. If siblings converge toward identical responses, diversity is collapsing.
- Red flag: If sibling similarity exceeds 0.8 by round 3, perspectives are washing out.

**Q4: Does iteration improve output quality?**
- Comparison: Run same task with `max_rounds=1` and `max_rounds=3`. Compare final output quality (human eval or LLM-as-judge).
- If `max_rounds=1` output is just as good, the round-based model isn't adding value.

**Q5: How does MCA compare to legacy mode?**
- Comparison: Run same task in both modes. Compare output quality, diversity of perspectives, and depth of analysis.
- This is the fundamental test. If legacy mode (simple decomposition + synthesis) produces equivalent output, MCA isn't earning its cost.

### 12.3 Comparison Script

A utility script `scripts/compare_modes.py` that runs a task in both modes and produces a side-by-side comparison:

```bash
poetry run python scripts/compare_modes.py \
    --task "Explain the implications of quantum computing for cryptography" \
    --cpp 3 --depth 2 --model gpt-4o-mini --max_rounds 3
```

Output: two `final_state.json` files, one `mca_report.json`, and a summary comparing:
- Final response text (both modes)
- Total LLM calls (both modes)
- MCA-specific metrics (round count, revision rate, convergence trajectory)

---

## 13. API Changes

All new parameters have defaults that preserve existing behavior.

### 13.1 CLI (`main.py`)

```
--mca_mode                       Enable MCA execution mode (default: off)
--max_rounds N                   Max rounds for MCA convergence (default: 3)
--convergence_threshold F        Jaccard similarity threshold 0-1 (default: 0.85)
--enable_downward_signals        Enable parent-to-child signals (default: on)
--no_downward_signals            Disable parent-to-child signals
--perspectives P [P ...]         Custom perspectives for leaf agents
```

### 13.2 Programmatic (`run_strange_mca()`)

```python
run_strange_mca(
    task="...",
    mca_mode=False,                      # NEW — activate MCA execution
    max_rounds=3,                        # NEW — MCA round cap
    convergence_threshold=0.85,          # NEW — MCA convergence threshold
    enable_downward_signals=True,        # NEW — parent-to-child signals
    perspectives=None,                   # NEW — custom leaf perspectives
    # ... all existing params unchanged ...
)
```

### 13.3 TextArena (`StrangeMCAAgent`)

```python
StrangeMCAAgent(
    mca_mode=False,                      # NEW
    max_rounds=3,                        # NEW
    convergence_threshold=0.85,          # NEW
    enable_downward_signals=True,        # NEW
    perspectives=None,                   # NEW
    # ... all existing params unchanged ...
)
```

---

## 14. LLM Cost Analysis

For `cpp=3, depth=2` (4 agents: 1 root + 3 leaves):

| Mode | Calls per unit | Total | Notes |
|------|---------------|-------|-------|
| Legacy | — | 5 + N | 1 decompose + 3 leaf + 1 synthesize + N strange loops |
| MCA, 1 round, signals on | 3 respond + 3 lateral + 1 observe + 1 signal = 8 | 8 + N | 1.6x legacy |
| MCA, 3 rounds, signals on | 8 * 3 = 24 | 24 + N | 4.8x legacy |
| MCA, 3 rounds, signals off | (3 + 3 + 1) * 3 = 21 | 21 + N | 4.2x legacy |
| MCA, 1 round, signals off | 3 + 3 + 1 = 7 | 7 + N | 1.4x legacy |

For `cpp=2, depth=3` (7 agents: 1 root + 2 internal + 4 leaves):

| Mode | Calls per round | Total (3 rounds) | Notes |
|------|----------------|-------------------|-------|
| Legacy | — | 8 + N | 1 root decompose + 2 internal decompose + 4 leaf + 1 synthesize + N |
| MCA, signals on | 4 respond + 4 lateral + 2 observe + 2 lateral + 1 observe + 3 signal = 16 | 48 + N | 6x legacy |
| MCA, signals off | 4 + 4 + 2 + 2 + 1 = 13 | 39 + N | 4.9x legacy |

The cost multiplier is the price of genuine multi-perspective iterative processing. Cost-sensitive configurations:
- `max_rounds=1, enable_downward_signals=False`: ~1.4x legacy (minimal MCA)
- `max_rounds=2, enable_downward_signals=True`: ~3.2x legacy (moderate)
- `max_rounds=3, enable_downward_signals=True`: ~4.8x legacy (full)

---

## 15. Backward Compatibility

**Why keep the legacy mode:** This is an exploratory project testing a hypothesis. If MCA doesn't produce meaningfully better results, the legacy mode remains a working baseline. The cost of an `if mca_mode` branch is small; the cost of losing a working baseline during research is high.

**What stays unchanged:**
- All existing source files continue to work without modification when `mca_mode=False`
- `State` TypedDict in `graph.py` is unchanged
- `create_execution_graph()` and `run_execution_graph()` are unchanged
- All existing prompt functions are unchanged
- All existing tests pass without modification
- `final_state.json` output format is unchanged in legacy mode

**What's new (additive only):**
- `MCAState` TypedDict alongside existing `State`
- `create_mca_execution_graph()` alongside existing `create_execution_graph()`
- New prompt functions alongside existing ones
- New `AgentConfig` fields with defaults
- New `Agent.invoke()` method alongside existing `run()`
- `mca_report.json` output (MCA mode only)

---

## 16. File-by-File Change Specification

### New Files

| File | Lines (est.) | Contents |
|------|-------------|----------|
| `src/strange_mca/convergence.py` | ~50 | `compute_jaccard_similarity()`, `check_convergence()` |
| `tests/test_convergence.py` | ~60 | Jaccard similarity edge cases, convergence detection |
| `scripts/compare_modes.py` | ~80 | Side-by-side legacy vs MCA comparison utility |

### Modified Files

**`src/strange_mca/agents.py`** (~150 lines, up from 68)
- `AgentConfig`: Add `depth`, `siblings`, `children`, `parent`, `perspective` fields. Add `is_leaf`, `is_root`, `role` properties. All new fields have defaults — existing construction unchanged.
- `Agent`: Add `invoke(prompt: str) -> str` method.
- `PERSPECTIVES`: List of 8 perspective strings.
- `build_agent_tree(cpp, depth, model_name, perspectives) -> dict[str, Agent]`: New function.

**`src/strange_mca/prompts.py`** (~350 lines, up from 175)
- Keep all existing functions unchanged.
- Add: `create_competency_prompt()`, `create_initial_response_prompt()`, `create_lateral_prompt()`, `create_observation_prompt()`, `create_signal_prompt()`, `create_signal_response_prompt()`.

**`src/strange_mca/graph.py`** (~750 lines, up from 522)
- Keep all existing code unchanged: `State`, `merge_dicts`, tree helpers, `create_agent_subgraph()`, `create_execution_graph()`, `run_execution_graph()`, `_apply_strange_loop()`, `_parse_subtask_for_child()`.
- Add: `AgentRoundData`, `MCAState`, `create_mca_execution_graph()`, `run_mca_execution_graph()`, all MCA node functions (`init_node`, `leaf_respond`, `leaf_lateral`, `observe_level`, `lateral_level`, `observe_root`, `signal_down`, `check_convergence`, `finalize`).

**`src/strange_mca/run_strange_mca.py`** (~200 lines, up from 162)
- `run_strange_mca()` gains: `mca_mode`, `max_rounds`, `convergence_threshold`, `enable_downward_signals`, `perspectives`.
- When `mca_mode=True`, calls `create_mca_execution_graph()` and `run_mca_execution_graph()` instead of legacy functions.
- Writes `mca_report.json` alongside `final_state.json` when in MCA mode.

**`src/strange_mca/main.py`** (~260 lines, up from 231)
- Add argparse arguments: `--mca_mode`, `--max_rounds`, `--convergence_threshold`, `--enable_downward_signals`/`--no_downward_signals`, `--perspectives`.
- Pass new args to `run_strange_mca()`.
- Print MCA summary metrics when in MCA mode.

**`examples/arena/strangemca_textarena.py`** (~110 lines, up from 96)
- `StrangeMCAAgent.__init__()` gains: `mca_mode`, `max_rounds`, `convergence_threshold`, `enable_downward_signals`, `perspectives`.
- Pass through to `run_strange_mca()`.

**`src/strange_mca/visualization.py`** (~280 lines, up from 242)
- `visualize_agent_tree()`: Add dashed edges between siblings to show lateral channels when MCA mode.

### Test Changes

| File | Change |
|------|--------|
| `tests/test_agents.py` | Add tests for new `AgentConfig` properties, `Agent.invoke()`, `build_agent_tree()`. Existing tests unchanged. |
| `tests/test_prompts.py` | Add tests for 6 new prompt functions. Existing tests unchanged. |
| `tests/test_graph.py` | Add tests for `MCAState`, `create_mca_execution_graph()` structure, MCA node functions. Existing tests unchanged. |
| `tests/test_run_strange_mca.py` | Add tests for MCA mode parameters. Existing tests unchanged. |
| `tests/test_main.py` | Add tests for new CLI args. Existing tests unchanged. |
| `tests/test_strangemca_textarena.py` | Add tests for new params. Existing tests unchanged. |

---

## 17. Implementation Phases

### Phase 1: Infrastructure

- Create `convergence.py` with `compute_jaccard_similarity()` and `check_convergence()`
- Extend `AgentConfig` with new fields and properties
- Add `Agent.invoke()` method
- Add `PERSPECTIVES` list and `build_agent_tree()` function
- Write `test_convergence.py` and extend `test_agents.py`
- **Verification**: `poetry run pytest` — all existing + new tests pass

### Phase 2: Prompts

- Add 6 new prompt functions to `prompts.py`
- Write tests for all new prompts in `test_prompts.py`
- **Verification**: `poetry run pytest` — all tests pass

### Phase 3: MCA Execution Graph

- Add `AgentRoundData`, `MCAState` to `graph.py`
- Implement `create_mca_execution_graph()` with all node functions
- Implement `run_mca_execution_graph()`
- Wire: `init → leaf_respond → leaf_lateral → [observe/lateral per internal level] → observe_root → signal_down → check_convergence → [loop or finalize]`
- Write MCA graph tests in `test_graph.py`
- **Verification**: `poetry run pytest` — all tests pass. MCA graph can be constructed with mocked agents.

### Phase 4: Integration

- Update `run_strange_mca.py` with MCA parameters and mode branching
- Update `main.py` with new CLI flags
- Update TextArena adapter with new params
- Write `mca_report.json` output logic
- Update integration tests
- **Verification**: `poetry run pytest` — all tests pass. Smoke test with real LLM:
  ```
  poetry run python -m src.strange_mca.main --task "Explain recursion" --depth 2 --cpp 3 --mca_mode --max_rounds 2
  ```

### Phase 5: Observability & Evaluation

- Create `scripts/compare_modes.py`
- Update `visualization.py` for lateral edges
- Run evaluation protocol (Section 12.2) on 3-5 diverse tasks
- Document findings
- **Verification**: Full test suite passes. Comparison script produces valid output. MCA report JSON validates.

---

## 18. Risks

### LLM cost scales with rounds and tree size
Each round invokes every agent at least twice (respond + lateral). For deep trees with multiple rounds, costs compound. **Mitigation**: `max_rounds` defaults to 3. `enable_downward_signals=False` reduces calls per round. `max_rounds=1` with no signals is only ~1.4x legacy cost for quick experiments.

### Convergence may not occur
Agents may oscillate, especially on creative or open-ended tasks. **Mitigation**: `max_rounds` guarantees termination. Root-only convergence checking (vs. all-agent) is more likely to stabilize since the root sees the big picture. The threshold (0.85) is permissive — it detects stabilization, not identity.

### Emergence may be shallow
LLM agents share the same underlying capability. Unlike biological cells with genuinely different competencies, LLMs are general-purpose. "Emergence" may reduce to averaging. **Mitigation**: This is the hypothesis we're testing. Diverse perspectives force different initial responses. Lateral prompts instruct agents to maintain their viewpoint. Observation prompts ask specifically for combination effects. If emergence is shallow despite these measures, that's a valid finding.

### Jaccard similarity is crude
Semantically identical responses with different wording score low, causing unnecessary extra rounds. **Mitigation**: Extra rounds cost LLM calls but don't produce incorrect results. The metric can be swapped later via a `convergence_method` parameter. Embedding cosine similarity is the most likely upgrade.

### Prompt sensitivity
The quality of MCA behavior depends heavily on prompt engineering. Small prompt changes could significantly affect whether emergence occurs. **Mitigation**: All prompts are centralized in `prompts.py`. The evaluation protocol (Section 12.2) provides quantitative signals for prompt iteration. The lateral and observation prompts are the most critical and should be iterated first.

### Perspective assignment is still top-down
Assigning perspectives to leaf agents is arguably a form of top-down direction — we're telling agents how to think. **Mitigation**: The perspective is a starting orientation, not a task assignment. Agents respond to the full original task through their lens, not to a decomposed subtask. The alternative (no perspectives, all agents respond identically) would eliminate diversity entirely. Dynamic perspective generation is a follow-on experiment.

---

## 19. Verification Plan

After each phase:

1. `poetry run pytest` — all existing + new tests pass
2. `./scripts/lint.sh` — passes Ruff and Black
3. Legacy smoke test: `poetry run python -m src.strange_mca.main --task "Explain recursion" --depth 2 --child_per_parent 2` — output unchanged
4. MCA smoke test (Phase 4+): `poetry run python -m src.strange_mca.main --task "Explain recursion" --depth 2 --child_per_parent 3 --mca_mode --max_rounds 2`
5. Inspect `output/*/final_state.json` — verify format matches legacy when `mca_mode=False`
6. Inspect `output/*/mca_report.json` — verify round history, convergence scores, revision rates
7. TextArena (Phase 4+): `poetry run python examples/arena/strange_basic_twoplayer.py` — works in both modes
8. Comparison (Phase 5): `poetry run python scripts/compare_modes.py --task "..." --cpp 3 --depth 2` — produces valid comparison

---

## 20. Open Questions

1. **Should the lateral prompt encourage consensus or productive disagreement?** The current prompt leans toward "maintain your perspective" (productive disagreement). This may prevent convergence. Testing will reveal whether the balance needs adjusting.

2. **Is Jaccard sufficient, or will we need embedding similarity sooner?** If convergence false-negatives (unnecessary extra rounds) are frequent, we should upgrade the metric earlier than planned.

3. **Should perspectives be task-dependent?** The fixed perspective pool works for general tasks. For domain-specific tasks (e.g., "Debug this SQL query"), generic perspectives like "ethical" or "creative" may not be useful. A `task_type` parameter that selects a relevant perspective pool is a potential follow-on.

4. **How should the strange loop interact with MCA convergence?** Currently, the strange loop runs after MCA convergence, at finalization. An alternative: the strange loop replaces MCA convergence entirely — the root self-reflects each round instead of checking Jaccard. This would make the strange loop the convergence mechanism. Worth exploring if the current approach feels redundant.

5. **What tasks best reveal (or fail to reveal) emergence?** The evaluation protocol needs a curated set of tasks that span: factual questions (where emergence should NOT help), analytical questions (where multiple perspectives should help), creative questions (where diversity should help most), and adversarial questions (where contradictions between perspectives are productive). Defining this task battery is part of Phase 5.
