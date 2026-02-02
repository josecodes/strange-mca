# Design: MCA Rearchitecture of strange-mca

**Author**: mayor (AI agent)
**Date**: 2026-02-01
**Status**: Draft — awaiting review
**Reviewers**: Jose Cortez

---

## 1. Problem Statement

strange-mca was built to model Levin's Multiscale Competency Architecture using LLM agents. The current implementation does not achieve this. It implements a top-down task decomposition tree — a corporate org chart, not an MCA.

Specifically:

- **Parents assign subtasks**. The `create_task_decomposition_prompt` tells the parent to break work into pieces and hand them to named children. Children have no say in what they work on.
- **Siblings are walled off**. The decomposition prompt explicitly states: *"Each team member works independently (they cannot see each other's work)."* There is zero lateral communication.
- **Synthesis is aggregation, not emergence**. The `up` node concatenates child responses and asks a single LLM to summarize. Nothing emerges from the children's interaction — a manager writes a report.
- **Self-reflection is bolted onto the root**. The strange loop runs once, at the top, after everything is done. No other level self-evaluates.
- **No bottom-up signaling**. Information flows strictly downward (tasks) and then upward (responses). Children cannot signal that a goal is misframed or that they discovered something unexpected.

In Levin's MCA, each scale has its own competency. Cells solve problems. Tissues coordinate. Organs integrate. The organism's behavior emerges from their collective interaction — it is not dictated from above. The current system has none of this.

## 2. Goals

1. **Goal broadcast replaces task assignment**. Parents communicate what needs to be achieved; children decide how to contribute.
2. **Lateral communication between siblings**. After producing initial responses, siblings see each other's work and coordinate — filling gaps, resolving contradictions, avoiding duplication.
3. **Fractal self-reflection**. Every node evaluates its own output against its goal before passing it upward. The strange loop becomes a property of the system, not just the root.
4. **Bottom-up signaling**. Children can flag goal mismatches, unexpected findings, or resource needs. Critical signals trigger re-decomposition at the parent.
5. **Backward compatibility**. The existing API, CLI, TextArena integration, and tests continue to work unchanged. New behavior is activated via `mca_mode=True`.

Non-goal for this iteration: adaptive tree structure (dynamic spawning/pruning of children at runtime). This is deferred as a follow-on.

## 3. Relevant GitHub Issues

| Issue | Title | Relationship |
|-------|-------|-------------|
| #5 | Explore bottom-up emergent behavior vs top-down decomposition | Core motivation |
| #6 | Add basal cognition / cultural system prompts | Goal broadcast enables this |
| #7 | Improve multiscale layer interactions | Lateral communication addresses this directly |
| #10 | Add dynamic routing and structure | Evaluate loop is a step toward this |
| #11 | Explore hierarchical strange loop layers | Fractal self-reflection implements this |

## 4. Design

### 4.1 Execution Model

The current execution model for a non-leaf node:

```
down (assign subtasks) → child_1 → child_2 → ... → child_N → up (summarize) → END
```

Children execute **sequentially** and **in isolation**. The parent controls both the task and the synthesis.

The proposed model:

```
broadcast ──→ execute_children (parallel) ──→ coordinate ──→ evaluate ──┐
    ↑                                                                    │
    │                                                              [conditional]
    │                                                                    │
    └──── re_broadcast ←── not satisfied ────────────────────────────────┤
                                                                         │
                                                                   satisfied
                                                                         │
                                                          synthesize ──→ self_reflect ──→ END
```

For leaf nodes:

```
execute ──→ self_reflect ──→ END
```

#### Node Descriptions

**`broadcast`**: The parent formulates a goal broadcast. Unlike the current decomposition, this does not assign specific subtasks to named children. It states the overall objective, provides context and constraints, and lets children self-select their contribution. On re-broadcast (iteration > 1), it incorporates signals from the previous round.

**`execute_children`**: Invokes all child subgraphs with the broadcast goal. Children run in parallel using `ThreadPoolExecutor`. Each child returns a structured response: text, satisfaction score (0–1), and optional signals.

**`coordinate`**: The lateral communication phase. For each child, constructs a prompt containing that child's response plus all siblings' responses. Re-invokes each child subgraph with this enriched context. Children can revise their contribution: adjust to avoid duplication, fill gaps, flag contradictions.

**`evaluate`**: The parent examines the collective child output. Checks average satisfaction, scans for critical signals (`goal_mismatch`, `needs_reframing`, `contradiction_detected`). If the collective is below the satisfaction threshold or critical signals are present, routes to `re_broadcast`. Otherwise routes to `synthesize`. Capped by `max_iterations`.

**`re_broadcast`**: Increments the iteration counter and returns to `broadcast` with signal context from the previous round.

**`synthesize`**: Combines child responses into a coherent parent response. At the root, also runs the existing strange loop mechanism.

**`self_reflect`**: Fractal strange loop. The node evaluates its synthesized response against the goal. Aggregates child signals with its own assessment and passes the combined signals upward. At the root, this is supplementary to the existing strange loop.

### 4.2 Lateral Communication Mechanism

This is the hardest design problem. LangGraph nested subgraphs are opaque — a parent invokes `child_subgraph.invoke(state)` and gets back a result dict. There is no shared memory between siblings.

**Approach**: Parent-mediated coordination. The parent's `coordinate` node re-invokes each child with an enriched prompt that contains all siblings' responses.

**Why not LangGraph's `Send` API?** The `Send` API enables fan-out within a flat graph. Our children are compiled nested subgraphs — they cannot be wired as `Send` targets. The fan-out must happen in Python code within a single LangGraph node.

**Biological analogy**: This is how gap junctions work. Neighboring cells don't communicate by reaching into each other — signals propagate through the shared extracellular environment. The parent node is that environment.

**Cost**: Coordination doubles child invocations per parent (from N to 2N). Configurable via `max_coordination_rounds`. Setting it to 0 disables lateral communication entirely.

### 4.3 Bottom-Up Signaling

Each agent can attach signals to its response:

| Signal | Meaning | Parent Action |
|--------|---------|---------------|
| `unexpected_finding` | Child discovered something the goal didn't anticipate | Informational — propagate upward |
| `goal_mismatch` | Child's expertise suggests the goal is framed wrong | **Critical** — triggers re-broadcast |
| `needs_reframing` | Child cannot contribute meaningfully with current framing | **Critical** — triggers re-broadcast |
| `contradiction_detected` | Findings contradict a sibling's (discovered during coordination) | Propagate upward, parent evaluates |
| `resource_request` | Child needs context or data it doesn't have | Propagate upward |

Signals propagate through the tree. The `self_reflect` node at each level aggregates its children's signals with its own assessment before passing them upward. The root's output includes a full signal chain.

### 4.4 State Schema

Extending the existing `State` TypedDict. All new fields are optional (`total=False`), so existing code that constructs partial state dicts is unaffected.

```python
class State(TypedDict, total=False):
    # Existing fields — unchanged
    task: str
    original_task: str
    response: str
    decomposition: str
    child_tasks: Annotated[dict[str, str], merge_dicts]
    child_responses: Annotated[dict[str, str], merge_dicts]
    final_response: str
    strange_loops: list[dict[str, str]]
    nodes: dict[str, dict[str, str]]
    current_node: str

    # New MCA fields
    mca_mode: bool
    goal: str                                                    # Broadcast goal
    parent_context: str                                          # Parent's guidance text
    child_agent_responses: Annotated[dict[str, dict], merge_dicts]  # Rich child data
    satisfaction: float                                          # 0–1 self-assessed quality
    signals: list[str]                                           # Bottom-up signal tags
    iteration: int                                               # Current broadcast-evaluate loop
    max_iterations: int                                          # Loop cap
```

### 4.5 Agent Response Structure

New dataclass for structured agent output:

```python
@dataclass
class AgentResponse:
    response: str
    satisfaction: float = 1.0
    signals: list[str] = field(default_factory=list)
    refined: bool = False
    iterations: int = 1
```

The `Agent` class gains two new methods alongside the existing `run()`:

- `run_with_signals(task, mode) -> AgentResponse` — returns structured output
- `self_reflect(goal, response) -> AgentResponse` — fractal strange loop

The existing `run()` method is unchanged. Legacy code paths call `run()`. MCA code paths call `run_with_signals()` and `self_reflect()`.

### 4.6 New Prompts

Six new prompt functions in `prompts.py`. Existing functions are untouched.

| Function | Purpose |
|----------|---------|
| `create_goal_broadcast_prompt` | Parent broadcasts goal+constraints (replaces `create_task_decomposition_prompt` in MCA mode) |
| `create_coordination_prompt` | Lateral communication — "here's what you said, here's what siblings said, revise if needed" |
| `create_self_reflection_prompt` | Fractal strange loop — "does your output serve the goal? Rate 0–1, revise if not" |
| `create_evaluation_prompt` | Parent evaluates collective — "are children's outputs sufficient? Any critical signals?" |
| `parse_agent_response` | Extract `AgentResponse` fields from LLM output |
| `parse_evaluation_response` | Extract satisfied/feedback/signals from evaluation output |

### 4.7 Output Format

The `final_state.json` output is extended (not replaced) when `mca_mode=True`:

```json
{
  "task": "...",
  "final_response": "...",
  "strange_loops": [...],
  "child_responses": {...},

  "mca_metadata": {
    "mode": "mca",
    "total_iterations": 2,
    "total_coordination_rounds": 1,
    "root_satisfaction": 0.85,
    "signal_chain": ["unexpected_finding (L3N2)", "contradiction_detected (L2N1)"],
    "satisfaction_map": {
      "L1N1": 0.85, "L2N1": 0.78, "L2N2": 0.92,
      "L3N1": 0.88, "L3N2": 0.65, "L3N3": 0.91, "L3N4": 0.95
    }
  }
}
```

### 4.8 API Changes

All new parameters have defaults that preserve existing behavior.

**CLI** (`main.py`):
```
--mca_mode                     Enable MCA mode (default: off)
--max_coordination_rounds N    Lateral communication rounds (default: 1)
--max_iterations N             Max broadcast-evaluate loops (default: 2)
--satisfaction_threshold F     Homeostatic threshold 0–1 (default: 0.7)
```

**Programmatic** (`run_strange_mca()`):
```python
run_strange_mca(
    task="...",
    mca_mode=True,                    # NEW
    max_coordination_rounds=1,        # NEW
    max_iterations=2,                 # NEW
    satisfaction_threshold=0.7,       # NEW
    # ... all existing params unchanged ...
)
```

**TextArena** (`StrangeMCAAgent`):
```python
StrangeMCAAgent(
    mca_mode=True,                    # NEW
    max_coordination_rounds=1,        # NEW
    max_iterations=2,                 # NEW
    satisfaction_threshold=0.7,       # NEW
    # ... all existing params unchanged ...
)
```

## 5. LLM Call Budget

For `cpp=2, depth=2` (3 agents total):

| Mode | Calls | Breakdown |
|------|-------|-----------|
| Legacy | 4 + N | 1 decompose + 2 leaf + 1 synthesize + N strange loops |
| MCA (coord=1, iter=1) | ~10 + N | 1 broadcast + 2 execute + 2 self-reflect + 2 coordinate + 1 evaluate + 1 synthesize + 1 self-reflect + N strange loops |
| MCA (coord=0, iter=1) | ~7 + N | No coordination phase |

The 2.5x cost is the price of genuine multi-perspective processing. Users can tune down with `max_coordination_rounds=0` or `max_iterations=1`.

The recursion limit in `run_execution_graph` should scale: `total_nodes(cpp, depth) * 10 * max_iterations`, with a minimum of 200 for MCA mode.

## 6. Implementation Phases

### Phase 1: Foundation

Add `AgentResponse` to `agents.py`. Add 6 new prompt/parse functions to `prompts.py`. Extend `State` in `graph.py`. No behavioral changes. All existing tests pass.

### Phase 2: Leaf Self-Reflection

Add `self_reflect` node to leaf subgraphs in MCA mode. Leaves evaluate their own output before returning. First observable MCA behavior.

### Phase 3: Goal Broadcast

Add `create_mca_agent_subgraph()` to `graph.py`. Implement `broadcast_node` and `execute_children_node`. Wire: `broadcast -> execute_children -> synthesize -> self_reflect -> END`. Thread `mca_mode` through `run_strange_mca.py` and `main.py`. Children receive goals, not assignments.

### Phase 4: Lateral Communication

Add `coordinate_node`. Wire between `execute_children` and `synthesize`. Children see siblings' responses and can revise. The defining MCA feature.

### Phase 5: Evaluate Loop + Bottom-Up Signals

Add `evaluate_node`, `re_broadcast_node`, conditional edge. Critical child signals trigger re-broadcast. `max_iterations` caps the loop. Full MCA topology operational.

### Phase 6: Non-Leaf Fractal Loops + Signal Propagation

Wire `self_reflect_node` for intermediate nodes. Aggregate child signals + own evaluation. Signals propagate from leaves through intermediates to root.

### Phase 7: Integration

Update `StrangeMCAAgent` for TextArena. Extend `final_state.json` with `mca_metadata`. Add lateral edges to visualization. Update tests.

## 7. Files Modified

| File | Phase | Change Summary |
|------|-------|---------------|
| `src/strange_mca/agents.py` | 1 | `AgentResponse`, `run_with_signals()`, `self_reflect()` |
| `src/strange_mca/prompts.py` | 1 | 6 new functions, existing unchanged |
| `src/strange_mca/graph.py` | 1–6 | State extension, `create_mca_agent_subgraph()`, all new nodes |
| `src/strange_mca/run_strange_mca.py` | 3 | Thread new parameters |
| `src/strange_mca/main.py` | 3 | CLI flags |
| `examples/arena/strangemca_textarena.py` | 7 | `mca_mode` parameter |
| `src/strange_mca/visualization.py` | 7 | Lateral edges |
| `tests/test_agents.py` | 1 | `AgentResponse`, new methods |
| `tests/test_prompts.py` | 1 | New prompt functions |
| `tests/test_graph.py` | 2–6 | MCA topology, signals, evaluate loop |

All paths relative to `/Users/jcortez/gt/strangemca/crew/jose/`.

## 8. Risks and Mitigations

**LLM satisfaction scores are unreliable.** LLMs are not good at self-evaluation. The satisfaction threshold should be treated as a soft signal, not a hard gate. The `max_iterations` cap prevents runaway loops regardless of satisfaction values. Over time, prompt tuning will improve signal quality.

**Coordination cost scales with tree size.** Each coordination round doubles child invocations. For deep trees (depth=4+), this compounds. Mitigation: `max_coordination_rounds` defaults to 1 and can be set to 0. Coordination can also be skipped when all children report high satisfaction (>0.9) and no signals — an optimization for Phase 7+.

**Nested subgraph opacity limits coordination depth.** The parent-mediated coordination model means children coordinate through their parent, not directly. For a depth=3 tree, L3 siblings can coordinate, and L2 siblings can coordinate, but L3 nodes in different L2 subtrees cannot see each other. This is actually biologically correct — cells in one tissue don't directly signal cells in a distant tissue — but it's worth naming as a constraint.

**Prompt engineering is critical.** The quality of the MCA behavior depends heavily on how well the prompts elicit goal-oriented (vs. task-following) behavior, meaningful self-evaluation, and useful lateral coordination. Expect iteration on prompt design through Phases 1–5.

## 9. Verification Plan

After each phase:

1. `cd /Users/jcortez/gt/strangemca/crew/jose && poetry run pytest` — existing tests pass
2. Legacy mode smoke test: `poetry run python -m src.strange_mca.main --task "Explain recursion" --depth 2 --child_per_parent 2` — output unchanged
3. MCA mode smoke test (Phase 3+): `poetry run python -m src.strange_mca.main --task "Explain recursion" --depth 2 --child_per_parent 2 --mca_mode` — produces output with MCA metadata
4. Inspect `output/*/final_state.json` — verify new fields present in MCA mode, absent in legacy
5. TextArena (Phase 7): `poetry run python examples/arena/strange_basic_twoplayer.py` — both modes work

## 10. Open Questions

1. **Should the coordination prompt encourage consensus or productive disagreement?** Consensus-seeking may flatten useful diversity. Productive disagreement may prevent convergence. The prompt design in Phase 4 needs to balance these.

2. **Should signals have severity levels?** The current design uses flat string tags. A `(signal, severity)` tuple would let the evaluate node make finer decisions but adds parsing complexity.

3. **How should the strange loop interact with MCA self-reflection?** Currently the root runs both `synthesize` (with the existing strange loop) and `self_reflect`. These may overlap. One option: in MCA mode, the fractal self-reflection replaces the classic strange loop entirely, and `strange_loop_count` controls the number of self-reflect iterations instead.
