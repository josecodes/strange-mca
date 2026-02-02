# RFC: Bottom-Up Emergent Multiscale Competency Architecture

**Author:** strange-mca team
**Status:** Draft
**Issue:** [#5 — Explore bottom-up emergent behavior vs top-down decomposition](https://github.com/josecodes/strange-mca/issues/5)

---

## 1. Problem Statement

The current strange-mca system implements a strictly top-down execution model: the root agent receives a task, decomposes it into subtasks, assigns those subtasks to children, and synthesizes responses on the way back up. This is fundamentally a command-and-control pattern — the root dictates what each agent does.

This is antithetical to the Multiscale Competency Architecture (MCA) concept the project is named after. In biological MCAs (as described in Michael Levin's work and Phillip Ball's *How Life Works*), intelligence at each scale emerges from the interactions of components at the scale below. Cells don't receive task assignments from the organism — they have their own local competencies, respond to local signals, and higher-order behavior emerges from their collective activity.

The current architecture has no mechanism for:
- Agents developing their own goals or perspectives
- Lateral (peer-to-peer) communication between agents at the same level
- Emergent behavior arising from local interactions
- Iterative refinement through multi-round communication

This RFC proposes a complete replacement of the execution model with a biologically-inspired bottom-up architecture.

---

## 2. Current Architecture Summary

### 2.1 Agent Model (`agents.py`, 67 lines)

`AgentConfig` holds `name`, `level`, `node_number`, and an optional `system_prompt`. `Agent` wraps `ChatOpenAI` with a single `run(task: str) -> str` method. Agents are stateless task executors — they receive a task string and return a response. They have no awareness of their position in the hierarchy, no persistent state across calls, and no concept of peers.

### 2.2 Execution Model (`graph.py`, 522 lines)

The system builds a nested LangGraph subgraph structure. Each agent gets its own compiled subgraph containing:
- A **down node**: non-leaf agents decompose tasks via `create_task_decomposition_prompt()`; leaf agents execute directly
- **Child invocation nodes**: wrapper functions that parse subtasks from the parent's decomposition, invoke child subgraphs, and store results
- An **up node**: non-leaf agents synthesize child responses via `create_synthesis_prompt()`; root additionally applies strange loop self-reflection

Execution is strictly sequential: `L1N1_down → child_L2N1 → child_L2N2 → ... → L1N1_up → END`. There is no parallelism, no iteration, and no lateral communication.

### 2.3 State (`graph.py`)

```python
class State(TypedDict, total=False):
    task: str
    original_task: str
    response: str
    decomposition: str
    child_tasks: Annotated[dict[str, str], merge_dicts]
    child_responses: Annotated[dict[str, str], merge_dicts]
    final_response: str
    strange_loops: list[dict[str, str]]
    nodes: dict[str, dict[str, str]]       # legacy
    current_node: str                       # legacy
```

State is scoped per-subgraph. Each agent subgraph has its own `State` instance. Cross-agent communication happens only through the parent invoking child subgraphs and reading their return values.

### 2.4 Prompts (`prompts.py`, 175 lines)

Three prompt types exist:
- `create_task_decomposition_prompt()` — instructs parent to break a task into subtasks formatted as `{child_name}: [subtask]`
- `create_synthesis_prompt()` — instructs parent to integrate child responses
- `create_strange_loop_prompt()` / `parse_strange_loop_response()` — self-reflection at root

### 2.5 Public API (`run_strange_mca.py`, 162 lines; `main.py`, 231 lines)

`run_strange_mca()` accepts `task`, `child_per_parent`, `depth`, `model`, `strange_loop_count`, `domain_specific_instructions`, plus logging/viz options. CLI mirrors this via argparse.

### 2.6 External Integration (`examples/arena/strangemca_textarena.py`, 96 lines)

`StrangeMCAAgent` extends `textarena.Agent`, wrapping `run_strange_mca()` in a `__call__(observation: str) -> str` interface. Used for chess games.

---

## 3. Proposed Architecture

### 3.1 Design Principles

**Stimulus over assignment.** Every agent sees the original task. No agent is told what to do — each responds according to its own competency.

**Local communication.** Agents communicate with their immediate neighborhood: siblings (same parent), children, and parent. No global broadcast.

**Emergence over synthesis.** Parent agents observe what their children produce and identify emergent patterns. They do not prescribe or assign — they make sense of what arises below.

**Convergence over single-pass.** The system runs multiple rounds of communication until agent outputs stabilize, rather than making a single pass through the tree.

**Preserved hierarchy.** The tree structure is maintained. Like biological systems (cells → tissues → organs → organisms), each level represents a different scale of organization with qualitatively different competencies.

### 3.2 Biological Analogy

| Tree Level | Biological Analogy | Agent Role | Behavior |
|---|---|---|---|
| Leaves (deepest) | Cells / Tissues | **Specialist** | Responds to the task from a unique narrow perspective. Coordinates with siblings. |
| Internal | Organs | **Coordinator** | Observes children's outputs. Identifies patterns, contradictions, synergies across specialist outputs. Coordinates with sibling coordinators. |
| Root (L1) | Organism | **Integrator** | Observes the full subtree. Produces holistic sense-making. Applies strange loop self-reflection. |

### 3.3 Execution Model: Round-Based Iterative Processing

Each execution consists of multiple **rounds**. A round processes the full tree bottom-up with lateral communication at each level:

```
INITIALIZATION
  All agents are created with topology-aware configs (siblings, children, parent)
  Each leaf agent is assigned a unique perspective from a predefined pool

ROUND 1: Seeding
  Step 1 — Leaf Response:     All leaf agents generate independent responses
  Step 2 — Leaf Lateral:      Leaf agents read siblings' responses, revise their own
  Step 3 — Internal Observe:  Internal agents read children's responses, synthesize patterns
  Step 4 — Internal Lateral:  Internal agents read siblings' syntheses, revise
       (Steps 3-4 repeat for each internal level, bottom to top)
  Step 5 — Root Observe:      Root reads children's syntheses
  Step 6 — Signal Down:       Non-leaf agents send brief guidance signals to children
  Step 7 — Convergence Check: Compare root's output to previous round

ROUND 2+: Refinement
  Same as Round 1, but:
  - Leaf agents incorporate parent signals before responding
  - All agents have access to their own previous response
  - Convergence is checked at end of each round

FINALIZATION (after convergence)
  Root applies strange loop self-reflection (preserved from current design)
  Return final response
```

### 3.4 Agent Model

#### AgentConfig (extended)

```python
class AgentConfig(BaseModel):
    name: str                          # e.g., "L3N4"
    level: int                         # Tree depth level (1 = root)
    node_number: int                   # Position within level (1-indexed)
    depth: int                         # Total tree depth
    siblings: list[str] = []           # Peer node names (same parent)
    children: list[str] = []           # Child node names
    parent: Optional[str] = None       # Parent node name (None for root)
    competency: str = ""               # Auto-generated system prompt
    perspective: str = ""              # Unique perspective (leaf agents only)

    @property
    def full_name(self) -> str: ...
    @property
    def is_leaf(self) -> bool: ...
    @property
    def is_root(self) -> bool: ...
    @property
    def role(self) -> str:             # "specialist" | "coordinator" | "integrator"
```

The key addition is topology awareness. Each agent knows its siblings, children, and parent at build time. The `competency` field holds a system prompt generated from the agent's role and perspective, which defines its goal-seeking behavior.

#### Agent (multi-method)

The current single `run(task)` method is replaced with purpose-specific methods. Each method constructs the appropriate prompt (via `prompts.py`) and calls the LLM:

| Method | Used By | Purpose |
|---|---|---|
| `respond(task, round_num)` | Leaves (round 1) | Independent initial response |
| `respond_with_peers(task, own_response, peer_responses, round_num)` | All levels | Revise after reading siblings |
| `observe_children(task, child_responses, own_previous, round_num)` | Non-leaves | Synthesize emergent patterns from children |
| `receive_signal(task, parent_signal, own_response, round_num)` | Non-root | Optionally adjust based on parent guidance |
| `generate_signal(task, child_responses, own_synthesis)` | Non-leaves | Produce downward guidance nudge |

All methods use the agent's `competency` as the system prompt, maintaining consistent persona across rounds.

#### Perspective Assignment

Leaf agents each receive a unique perspective to ensure diversity. A predefined pool is used:

```python
PERSPECTIVES = [
    "analytical", "creative", "critical", "practical",
    "theoretical", "empirical", "ethical", "systemic",
]
```

Assignment is deterministic: `perspective = PERSPECTIVES[node_number % len(PERSPECTIVES)]`. This avoids an extra LLM call at build time and keeps tests reproducible. Custom perspectives can be passed via API.

#### Topology Construction

A new `build_agent_tree(cpp, depth, perspectives)` function creates all `AgentConfig` objects with proper topology. It iterates through all levels, computes children/siblings/parent for each node using the existing tree helper functions, generates competency prompts, and returns a `dict[str, AgentConfig]`.

### 3.5 Communication Model

#### Channels

```
        L1N1 (Integrator)
       /    \
   L2N1 ←→ L2N2  (Coordinators — lateral)
   / | \    / | \
 L3N1-N3  L3N4-N6  (Specialists — lateral within siblings)
 ←----→    ←------→
```

**Lateral (peer-to-peer):** Agents at the same level sharing the same parent exchange full responses. This is the primary coordination mechanism. Siblings under `L2N1` can see each other; siblings under `L2N2` can see each other. Cross-subtree lateral communication (e.g., `L3N1` seeing `L3N4`) does not occur in v1 — this mirrors biological locality where cells in one tissue don't directly signal cells in another organ.

**Upward (child → parent):** Parents read all children's latest responses. The parent does NOT issue commands. It observes and synthesizes. This is analogous to how organ-level function emerges from tissue behavior without the organ "telling" tissues what to do.

**Downward (parent → child):** After synthesizing, a parent generates a brief "signal" — not a task assignment, but a nudge highlighting gaps, tensions, or areas needing depth. Children may incorporate this or ignore it. This is analogous to hormonal or nervous system feedback in biology. This channel is optional and can be disabled via config.

#### Visibility Rules

| Agent | Can See | Cannot See |
|---|---|---|
| Leaf | Original task, own history, sibling responses, parent signal | Cousins, grandparent, other subtrees |
| Internal | Original task, own history, sibling syntheses, all children's responses, parent signal | Agents beyond immediate neighborhood |
| Root | Original task, own history, all children's syntheses | — |

### 3.6 Convergence Detection

#### Metric

Jaccard token similarity between an agent's response in round N and round N-1:

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

This was chosen over alternatives for v1:
- **LLM-as-judge** (ask LLM "are these the same?"): more semantically accurate but adds LLM cost per agent per round
- **Embedding cosine similarity**: better than token overlap but requires an embedding model dependency

Jaccard is zero-cost, deterministic, and sufficient for detecting stabilization. It can be swapped for a more sophisticated metric later.

#### Convergence Scope

**Root convergence** is the primary check. If the root's synthesis has stabilized, the system output has stabilized regardless of leaf-level variation. This is both simpler and more efficient than requiring all agents to converge.

```python
def check_global_convergence(
    agent_states: dict[str, AgentState],
    root_name: str,
    threshold: float,
) -> bool:
    root_responses = agent_states[root_name]["responses"]
    if len(root_responses) < 2:
        return False
    return compute_jaccard_similarity(
        root_responses[-1], root_responses[-2]
    ) >= threshold
```

#### Parameters

| Parameter | Default | Description |
|---|---|---|
| `max_rounds` | 5 | Safety valve — stop after this many rounds regardless |
| `convergence_threshold` | 0.85 | Jaccard similarity threshold for convergence |

### 3.7 State Design

```python
class AgentState(TypedDict, total=False):
    responses: list[str]           # Response history indexed by round
    signals_received: list[str]    # Parent signals received each round
    converged: bool                # Per-agent convergence flag (informational)

class State(TypedDict, total=False):
    # Task
    original_task: str

    # Per-agent state (keyed by node name, e.g., "L3N2")
    agent_states: Annotated[dict[str, AgentState], merge_dicts]

    # Latest parent signals (keyed by parent node name)
    parent_signals: Annotated[dict[str, str], merge_dicts]

    # Round tracking
    current_round: int
    max_rounds: int
    convergence_threshold: float

    # Convergence
    converged: bool

    # Final output
    final_response: str
    strange_loops: list[dict[str, str]]

    # Debug / observability
    round_history: Annotated[list[dict[str, Any]], append_to_list]
```

Key design decisions:
- **Flat agent_states dict** with per-agent response lists (index = round). Avoids deep nesting.
- **merge_dicts reducer** on `agent_states` and `parent_signals` allows each graph node to return partial updates that merge into the global state.
- **append_to_list reducer** on `round_history` accumulates per-round debug snapshots.
- The old `task`, `response`, `decomposition`, `child_tasks`, `child_responses`, `nodes`, `current_node` fields are all removed.

---

## 4. LangGraph Integration

### 4.1 Graph Topology

The current system uses nested subgraphs (one per agent). The new system uses a **single flat StateGraph** with a conditional back-edge for the round loop.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  init ──→ leaf_respond ──→ lateral_leaves ──→ [observe_L{n} ──→ lateral_L{n} │
│               ↑                                         ... per internal     │
│               │                                         level, bottom-up]    │
│               │                                              │               │
│               │                                              ↓               │
│               │                                      observe_root            │
│               │                                              │               │
│               │                                              ↓               │
│               │                                      signal_down             │
│               │                                              │               │
│               │                                              ↓               │
│               │                                    check_convergence         │
│               │                                       │           │          │
│               │              (not converged)──────────┘           │          │
│               │                                          (converged)         │
│               └──────────────────────────────────────         │              │
│                                                           finalize ──→ END   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

For `depth=2, cpp=3` the graph nodes are:
`init → leaf_respond → lateral_leaves → observe_root → signal_down → check_convergence → finalize`

For `depth=3, cpp=2` the graph nodes are:
`init → leaf_respond → lateral_leaves → observe_L2 → lateral_L2 → observe_root → signal_down → check_convergence → finalize`

### 4.2 Why a Flat Graph

Nested subgraphs were appropriate for the sequential down-up pattern where each agent was a self-contained unit. For the new round-based pattern:

- Each round re-enters every agent. Nested subgraphs would require re-invoking every subgraph every round.
- The conditional convergence loop must be at the outer level regardless.
- State passing between nested subgraphs becomes complex when agents need to read peers' state.
- A flat graph with Python loops inside each node function is simpler to debug, visualize, and test.

The trade-off is that per-agent logic lives in Python loops rather than graph topology. Since LLM calls are the bottleneck (not graph traversal), this is the correct trade-off.

### 4.3 Node Functions

Each graph node function iterates over the relevant agents at its level. Example for `leaf_respond`:

```python
def leaf_respond(state: State) -> dict:
    round_num = state["current_round"]
    task = state["original_task"]
    updated_states = {}

    for name in leaf_agent_names:
        agent = agents[name]
        agent_state = state["agent_states"][name]
        previous = agent_state["responses"]

        if round_num == 1:
            response = agent.respond(task, round_num)
        else:
            parent_signal = state.get("parent_signals", {}).get(
                agent_configs[name].parent, "")
            if parent_signal:
                response = agent.receive_signal(task, parent_signal, previous[-1], round_num)
            else:
                response = previous[-1]

        updated_states[name] = {
            "responses": previous + [response],
            "signals_received": agent_state["signals_received"],
            "converged": agent_state["converged"],
        }

    return {"agent_states": updated_states}
```

### 4.4 Convergence as Conditional Edge

```python
builder.add_conditional_edges(
    "check_convergence",
    lambda state: "finalize" if state.get("converged", False) else "next_round",
    {"next_round": "leaf_respond", "finalize": "finalize"},
)
```

### 4.5 Recursion Limit

The round loop means the graph traverses many edges. Set `recursion_limit = max_rounds * (num_graph_nodes + 5)` to provide headroom.

### 4.6 Future: Parallel LLM Calls

Within `leaf_respond` and `lateral_leaves`, agents at the same level are independent and could be called in parallel via `asyncio.gather()`. LangGraph supports `async def` node functions natively. This is deferred to a follow-up but the architecture supports it cleanly.

---

## 5. Prompt Design

### 5.1 Competency System Prompts

Generated at build time per agent. Stored in `AgentConfig.competency`.

**Specialist (leaf):**
> You are a specialist analyst in a collaborative research team. Your unique perspective is: {perspective}. Your goal is to contribute the most valuable insight you can from your area of expertise. You are NOT being assigned a subtask — you independently analyze the problem through your lens. When communicating with peers, maintain your perspective while being open to integration. Be concise and substantive.

**Coordinator (internal):**
> You are a mid-level coordinator in a collaborative research team. You observe the outputs of several specialist analysts (your team). Your goal is to identify emergent patterns, contradictions, and synergies in their work. You do NOT assign tasks or tell them what to do. You observe and synthesize. When communicating with peer coordinators, share your synthesis and look for cross-team patterns. Focus on what emerges from the combination that no individual noticed.

**Integrator (root):**
> You are the lead integrator of a collaborative research team. You observe the outputs of all coordinators and their teams. Your goal is to produce a coherent, holistic response that captures the best emergent insights. You do NOT direct the process. You make sense of what has emerged from below.

### 5.2 Round Prompts

| Function | Purpose | Key Instructions |
|---|---|---|
| `create_initial_response_prompt(task, round_num)` | Leaf first response | "Provide your independent analysis from your unique perspective" |
| `create_lateral_prompt(task, own_response, peer_responses, round_num)` | Peer revision | "Maintain your viewpoint but integrate peer insights. Address contradictions constructively." |
| `create_observation_prompt(task, child_responses, own_previous, round_num)` | Parent observing children | "What themes emerge? What contradictions exist? What arises from the COMBINATION?" |
| `create_signal_prompt(task, child_responses, own_synthesis)` | Downward guidance | "Highlight gaps or blind spots. Keep it suggestive, not directive. 2-3 sentences." |
| `create_signal_response_prompt(task, own_response, parent_signal, round_num)` | Child adjusting to signal | "Consider whether the coordinator's observation reveals something you should address. You are free to adjust or maintain your position." |

### 5.3 Preserved Prompts

`create_strange_loop_prompt()` and `parse_strange_loop_response()` are kept unchanged. The strange loop is orthogonal to the top-down vs bottom-up question — it is self-reflection at the root after the system has converged, and remains valuable.

---

## 6. Alternatives Considered

### 6.1 Keep Top-Down as an Option

Considered adding bottom-up as a second execution mode alongside the existing top-down. Rejected because:
- Maintaining two execution engines doubles the surface area
- The goal is to explore whether bottom-up is a better model for MCA, not to provide optionality
- The top-down code remains in git history if needed

### 6.2 Fully Connected Communication

Considered allowing any agent to communicate with any other agent (not just immediate neighbors). Rejected because:
- Quadratic LLM cost scaling (every agent reads every other agent)
- Contradicts the biological model where communication is local
- More noise than signal — agents at distant positions lack shared context

### 6.3 LLM-Based Convergence Detection

Considered using an LLM call to judge whether responses have stabilized. Deferred because:
- Adds significant cost (one LLM call per agent per round)
- Jaccard similarity is sufficient for detecting stabilization in v1
- Can be added as a `--convergence_method` flag later if needed

### 6.4 Dynamic Perspective Generation

Considered having the LLM generate unique perspectives for leaf agents at build time. Deferred because:
- Adds unpredictable LLM calls during graph construction
- Makes tests non-deterministic
- The predefined pool is sufficient and customizable via API

### 6.5 Cross-Subtree Lateral Communication

Considered allowing agents to communicate with cousins (peers under different parents). Deferred because:
- Significantly increases message volume
- The current design can be extended to support this via a `lateral_scope` parameter
- Worth exploring after validating that within-subtree lateral communication produces meaningful emergence

---

## 7. Risk Analysis

### 7.1 LLM Cost

**Risk:** Multiple rounds with lateral communication at every level means significantly more LLM calls per execution.

**Mitigation:**
- Current system: For `cpp=3, depth=2`, makes 1 (root decompose) + 3 (leaf execute) + 1 (root synthesize) = **5 LLM calls**.
- New system: For `cpp=3, depth=2`, one round makes 3 (leaf respond) + 3 (lateral) + 1 (root observe) + 1 (signal) = **8 calls per round**. With 3 rounds, that's **~24 calls**.
- This is ~5x the current cost. The `max_rounds` safety valve and convergence threshold keep this bounded. For cost-sensitive usage, `max_rounds=2` and `--no_downward_signals` reduces to ~12 calls.

### 7.2 Convergence May Not Occur

**Risk:** Agents may oscillate rather than converge, especially with creative or open-ended tasks.

**Mitigation:**
- `max_rounds` safety valve ensures termination
- Root convergence (not all-agent convergence) is more likely to stabilize since the root sees the big picture
- The convergence threshold (0.85 Jaccard) is relatively permissive — it detects stabilization, not identity

### 7.3 Emergent Behavior May Be Shallow

**Risk:** LLM agents are powerful general-purpose reasoners. Unlike biological cells with genuinely different capabilities, LLM agents all have the same underlying competency. "Emergence" may reduce to simple averaging.

**Mitigation:**
- Diverse perspectives for leaf agents force genuinely different initial responses
- Lateral communication prompts explicitly instruct agents to maintain their perspective while integrating
- The observation prompts for parent agents specifically ask "what arises from the COMBINATION that no individual captured?" — pushing toward genuine synthesis rather than summarization
- This is an inherent limitation of the LLM-as-agent paradigm and a key thing to evaluate during testing

### 7.4 Prompt Sensitivity

**Risk:** The quality of emergence depends heavily on prompt engineering. Small prompt changes could significantly affect behavior.

**Mitigation:**
- All prompts are centralized in `prompts.py` and easy to iterate on
- The competency system prompt is generated from templates, not hardcoded per agent
- Integration tests with real LLM calls will validate prompt effectiveness

---

## 8. File-by-File Change Specification

### 8.1 `src/strange_mca/convergence.py` — NEW

~80 lines. Contains:
- `compute_jaccard_similarity(text_a: str, text_b: str) -> float`
- `check_agent_convergence(responses: list[str], threshold: float) -> bool`
- `check_global_convergence(agent_states: dict, root_name: str, threshold: float) -> bool`

### 8.2 `src/strange_mca/agents.py` — REWRITE

~200 lines. Changes:
- `AgentConfig`: Add `depth`, `siblings`, `children`, `parent`, `competency`, `perspective`. Add properties `is_leaf`, `is_root`, `role`.
- `Agent`: Replace single `run()` with `respond()`, `respond_with_peers()`, `observe_children()`, `receive_signal()`, `generate_signal()`. Each constructs appropriate prompt via `prompts.py` and calls LLM.
- `PERSPECTIVES`: Predefined list of perspective strings.
- `build_agent_tree(cpp, depth, perspectives) -> dict[str, AgentConfig]`: Constructs full topology.

### 8.3 `src/strange_mca/prompts.py` — REWRITE

~300 lines. Changes:
- **Remove**: `create_task_decomposition_prompt()`
- **Keep**: `create_strange_loop_prompt()`, `parse_strange_loop_response()`
- **Add**: `create_competency_prompt()`, `create_initial_response_prompt()`, `create_lateral_prompt()`, `create_observation_prompt()`, `create_signal_prompt()`, `create_signal_response_prompt()`

### 8.4 `src/strange_mca/graph.py` — REWRITE

~400 lines. Changes:
- **Keep**: Tree helpers (`parse_node_name`, `make_node_name`, `get_children`, `is_leaf`, `is_root`, `count_nodes_at_level`, `total_nodes`), `merge_dicts`, `_apply_strange_loop`
- **Remove**: `create_agent_subgraph`, `_parse_subtask_for_child`, old `State`
- **Add**: New `State`/`AgentState` TypedDicts, `append_to_list` reducer, new `create_execution_graph()` building flat graph with round loop
- `create_execution_graph()` signature gains: `max_rounds`, `convergence_threshold`, `enable_downward_signals`, `perspectives`
- `run_execution_graph()`: minimal changes — passes new params into initial state, adjusts recursion limit

### 8.5 `src/strange_mca/run_strange_mca.py` — MODIFY

~170 lines. Changes:
- `run_strange_mca()` gains: `max_rounds=5`, `convergence_threshold=0.85`, `enable_downward_signals=True`, `perspectives=None`
- Pass new params to `create_execution_graph()` and `run_execution_graph()`
- Update result output to include `round_history` and `agent_states`

### 8.6 `src/strange_mca/main.py` — MODIFY

~250 lines. Changes:
- Add argparse args: `--max_rounds`, `--convergence_threshold`, `--enable_downward_signals`/`--no_downward_signals`, `--perspectives`
- Update execution summary to show round count and convergence info

### 8.7 `src/strange_mca/visualization.py` — MODIFY

~280 lines. Changes:
- `visualize_agent_tree()`: Add dashed edges between siblings to show lateral channels
- New `visualize_round_history(round_history, output_path)`: Per-round visualization of response evolution

### 8.8 `src/strange_mca/logging_utils.py` — MINOR MODIFY

Add `[Round N]` prefix to log entries. No structural changes.

### 8.9 `examples/arena/strangemca_textarena.py` — MODIFY

Pass through `max_rounds`, `convergence_threshold`, `enable_downward_signals` to `run_strange_mca()`.

### 8.10 Tests — REWRITE

| File | Changes |
|---|---|
| `tests/test_convergence.py` | **NEW.** Jaccard similarity, per-agent convergence, global convergence edge cases. |
| `tests/test_agents.py` | New `AgentConfig` properties, `build_agent_tree()` topology correctness, Agent method mocking. |
| `tests/test_prompts.py` | All new prompt functions. Remove decomposition test. Keep strange loop tests. |
| `tests/test_graph.py` | Keep tree helper tests. New: round-based graph structure, convergence loop, state transitions. |
| `tests/test_run_strange_mca.py` | Update mocks for new params. |
| `tests/test_main.py` | Update argparse tests for new CLI args. |
| `tests/test_strangemca_textarena.py` | Update for new params. |

---

## 9. Implementation Sequence

**Phase 1 — Core infrastructure (no LLM calls)**
1. Create `convergence.py` with similarity functions
2. Rewrite `agents.py` — `AgentConfig`, `Agent` (stub LLM methods), `build_agent_tree()`
3. Write `test_convergence.py` and `test_agents.py`

**Phase 2 — Prompts**
4. Rewrite `prompts.py` with all new prompt functions
5. Write `test_prompts.py`

**Phase 3 — Execution graph**
6. Rewrite `graph.py` — new `State`, new `create_execution_graph()` with round loop
7. Write `test_graph.py`

**Phase 4 — Integration**
8. Update `run_strange_mca.py` with new params
9. Update `main.py` CLI
10. Update TextArena adapter
11. Write/update integration tests

**Phase 5 — Polish**
12. Update `visualization.py` for lateral connections and round history
13. Update `logging_utils.py` for round-aware output
14. Update `CLAUDE.md` documentation
15. End-to-end testing with real LLM calls

---

## 10. Verification Plan

1. **Unit tests**: `poetry run pytest` — all tests pass
2. **Lint**: `./scripts/lint.sh` — passes Ruff and Black
3. **Basic execution**: `poetry run python -m src.strange_mca.main --task "Explain photosynthesis" --child_per_parent 3 --depth 2 --max_rounds 3` — runs, shows multiple rounds, converges
4. **Deep tree**: `poetry run python -m src.strange_mca.main --task "Explain photosynthesis" --depth 3 --child_per_parent 2` — internal level coordination works
5. **Convergence behavior**: Run with `--max_rounds 10` and verify convergence occurs before max
6. **Signals disabled**: `poetry run python -m src.strange_mca.main --task "Explain photosynthesis" --no_downward_signals` — works without signals
7. **TextArena**: `poetry run python examples/arena/strange_basic_twoplayer.py` — adapter still works
8. **Visualization**: `--viz` flag produces tree with lateral connections
