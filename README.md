# **strange-mca**: MCA with a strange loop

<img src="assets/strange-mca-logo.png" alt="strange-mca Logo" width="150" align="left" style="margin-right: 20px; margin-bottom: 10px;">

This multiagent system is a simplified model of a Multiscale Competency Architecture (MCA). Since reading about MCAs in [one](https://pubmed.ncbi.nlm.nih.gov/37156924/) of Michael Levin's recent papers, I see MCAs everywhere :).  Levin poses them as a framework to conceptualize how biological systems exhibit competence (i.e. problem solving skills) across multiple scales - from cells to tissues to organs to organisms to social groups. I've been looking for a way to model the concept in software, and using LLMs as information processing nodes in an MCA seemed like a fun experiment.

This system is also inspired by Hofstadter's [Strange Loop](https://en.wikipedia.org/wiki/Strange_loop). There is a bit of (configurable) self-reflection when giving a response to a prompt at the root node in this system. Perhaps this is similar to what today's reasoning models do (albeit in more sophisticated form)

For both MCA and Strange Loop concepts, this system is a minimal (but fun) attempt at running a multiagent system that models these concepts. In the *Future Ideas and Improvements* section near the end, I share some thoughts for iteration on the system.

I thought it would make for interesting behavior and comparisons to point this at an OpenAI Gym style environment like TextArena to see it play chess and other games against other LLMs. So I have included TextArena integration code in the `examples` section.

Probably the most fun thing to do with it right now, besides asking ambiguous or absurd questions, is to have it play a game of chess against a single LLM of the same spec. They play about as well as you'd expect a LLM to play chess, but it is interesting to see agents independently respond from unique perspectives, communicate laterally with peers, and iterate in rounds until the root's output converges — then see the strange loop do its thing on the final response. In the `assets` directory there is an example output `final_state.json` that shows the first turn `state` object for the strange-mca set at 2 child-per-node, 3 levels, and gpt-4o-mini. All the messy chatter in its full glory to look through if it sounds interesting. This json is produced in the output dir for every execution of strange-mca. The json file is the main way to inspect how the agents behaved. You can also play with cmd line args to get more info in stdout.

## High Level Architecture

The system uses a single **flat LangGraph `StateGraph`** with round-based bottom-up processing. Rather than top-down task decomposition, each agent responds independently from a unique perspective, communicates laterally with peers, and iterates in rounds until the root's output converges. Higher-order behavior emerges from local interaction, not top-down assignment.

```
init → leaf_respond → leaf_lateral → [observe_L{n} → lateral_L{n}]* → observe_root → signal_down → check_convergence → [loop | finalize] → END
```

### Agent Roles

- **Specialist** (leaf): Responds to the full task from a unique perspective (analytical, creative, critical, etc.)
- **Coordinator** (internal): Observes children's outputs and identifies emergent patterns
- **Integrator** (root): Produces holistic synthesis from coordinator/specialist outputs

### Processing Flow Per Round

1. **Leaf Respond**: Each leaf generates an independent response from its assigned perspective
2. **Leaf Lateral**: Leaves see siblings' responses and revise while maintaining their viewpoint
3. **Internal Observe/Lateral**: Per internal level (bottom to top), coordinators observe children and communicate laterally with peers
4. **Root Observe**: Root synthesizes children's outputs into a unified response
5. **Signal Down** (optional): Non-leaf agents send brief nudges to children highlighting gaps or tensions
6. **Convergence Check**: Jaccard similarity on root output across rounds; loop or finalize

The strange loop self-reflection occurs at finalization, after convergence.

## Features

- **Hierarchical Agent Structure**: Configurable tree of agents with customizable depth and branching factor
- **Emergent Bottom-Up Processing**: Agents respond independently; coherence emerges from local interactions rather than top-down decomposition
- **Lateral Peer Communication**: Agents at the same level see siblings' responses and revise their own, maintaining their unique viewpoint
- **Round-Based Convergence**: The system iterates until the root's output stabilizes (Jaccard similarity threshold) or max rounds reached
- **Downward Signals**: Parent agents send brief nudges to children highlighting gaps or tensions (configurable)
- **Multiple Perspectives**: Leaf agents are assigned from a pool of 8 default perspectives — analytical, creative, critical, practical, theoretical, empirical, ethical, systemic — or custom perspectives via CLI/API
- **Strange Loop Refinement**: Optional self-critique and refinement of the final response at the root
- **Visualization Tools**: Generate visual representations of the agent tree and execution graph
- **Observability Reports**: JSON reports (`mca_report.json`) with per-round agent data, convergence trajectory, LLM call counts, and lateral revision rates
- **TextArena Integration**: Run strange-mca agents in game environments using TextArena

## Installation

### Prerequisites

- Python 3.9+
- [Poetry](https://python-poetry.org/) for dependency management

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/josecodes/strange-mca.git
   cd strange-mca
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. (Optional) Install TextArena dependencies:
   ```bash
   poetry install --with arena
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Running the Multi-Agent System

You can run the strange-mca system using the command-line interface:

```bash
poetry run python -m src.strange_mca.main --task "Your task here" --child_per_parent 3 --depth 2 --model "gpt-4o-mini"
```

#### Command-line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `"Explain the concept of multi-agent systems"` | The task to perform |
| `--child_per_parent` | `3` | Number of children per parent node |
| `--depth` | `2` | Number of levels in the tree |
| `--model` | `"gpt-4o-mini"` | LLM model to use |
| `--max_rounds` | `3` | Maximum number of rounds for convergence |
| `--convergence_threshold` | `0.85` | Jaccard similarity threshold for convergence (0-1) |
| `--enable_downward_signals` | on | Enable parent-to-child signals |
| `--no_downward_signals` | — | Disable parent-to-child signals |
| `--perspectives` | 8 defaults | Custom perspectives for leaf agents (space-separated) |
| `--strange_loop_count` | `0` | Number of strange loop iterations to perform |
| `--domain_specific_instructions` | `""` | Domain-specific instructions for the strange loop prompt |
| `--log_level` | `"info"` | Logging level (debug, info, warning, error, critical) |
| `--local-logs-only` | off | Show only logs from strange_mca, suppress dependency logs |
| `--viz` | off | Generate visualizations of the agent tree and execution graph |
| `--dry_run` | off | Don't run the system, just show the configuration |
| `--print_tree` | off | Print the agent tree structure |
| `--print_details` | off | Print full state details after execution |

### Programmatic Usage

You can also use strange-mca programmatically in your Python code:

```python
from src.strange_mca.run_strange_mca import run_strange_mca

result = run_strange_mca(
    task="Explain the concept of multi-agent systems",
    child_per_parent=2,
    depth=2,
    model="gpt-4o-mini",
    max_rounds=3,
    convergence_threshold=0.85,
    enable_downward_signals=True,
    perspectives=None,  # uses 8 defaults
    strange_loop_count=1,
    domain_specific_instructions="Focus on practical applications",
)

print(result["final_response"])
```

The return value is a dict containing `final_response`, `converged`, `convergence_scores`, `current_round`, `agent_history`, and `strange_loops`.

For structured observability data, use `build_mca_report()`:

```python
from src.strange_mca.run_strange_mca import build_mca_report

report = build_mca_report(result, task="...", config={...})
# report contains: task, config, rounds (per-agent data), convergence, summary_metrics, final_response
```

### TextArena Integration


![Curses Chess Interface](assets/curses-chess.png)
*Figure: strange-mca vs. single LLM agent playing chess*



strange-mca can be used with [TextArena](https://github.com/LeonGuertler/TextArena) to create agents that play games and solve interactive tasks. The integration is available in the `examples/arena` directory.

#### Running a TextArena Game

```bash
poetry run python examples/arena/strange_basic_twoplayer.py
```

This will run a two-player game (SpellingBee by default) with a strange-mca agent competing against an OpenAI agent.

#### Available Game Scripts

- `strange_basic_twoplayer.py`: Runs a basic two-player game without rendering
- `strange_rendered_twoplayer.py`: Runs a two-player game with visual rendering

#### Creating Custom Game Scripts

You can create custom game scripts by following the template in the existing scripts. The key components are:

1. Initialize the strange-mca agent with appropriate parameters
2. Set up the game environment using TextArena
3. Create a game loop to manage turns and actions

Example:

```python
import textarena as ta
from strangemca_textarena import StrangeMCAAgent

# Initialize agents
agents = {
    0: StrangeMCAAgent(
        child_per_parent=2,
        depth=2,
        model="gpt-4o-mini",
        domain_specific_instructions="Your game-specific instructions here",
        strange_loop_count=2
    ),
    1: ta.agents.OpenAIAgent(
        model_name="gpt-4o-mini",
        system_prompt="Your system prompt here",
    ),
}

# Initialize environment
env = ta.make(env_id="Chess-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)

# Game loop
env.reset(num_players=len(agents))
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
```

## Project Structure

- `src/strange_mca/`: Core implementation
  - `main.py`: CLI entry point with MCA parameters
  - `run_strange_mca.py`: Programmatic API and `build_mca_report()` observability
  - `graph.py`: Flat `StateGraph`, `MCAState`, `create_execution_graph()`, `run_execution_graph()`
  - `agents.py`: `AgentConfig` (topology-aware), `Agent` (wraps ChatOpenAI), `build_agent_tree()`, `PERSPECTIVES`
  - `prompts.py`: MCA prompt functions (competency, initial response, lateral, observation, signal, strange loop)
  - `tree_helpers.py`: Pure functions for node naming, parent/child/sibling relationships, tree traversal
  - `convergence.py`: Jaccard token similarity for convergence detection
  - `visualization.py`: Agent tree and execution graph visualization
  - `logging_utils.py`: Detailed logging utilities
- `tests/`: Test suite (91 unit tests + 5 live integration tests)
  - `test_agents.py`, `test_graph.py`, `test_prompts.py`, `test_tree_helpers.py`, `test_convergence.py`: Unit tests for core modules
  - `test_main.py`, `test_run_strange_mca.py`, `test_visualization.py`: Integration tests
  - `test_emergent_properties.py`: Automated tests for emergent behavior properties
  - `test_live_integration.py`: Live tests requiring `OPENAI_API_KEY` (marked `@pytest.mark.live`)
- `examples/arena/`: TextArena integration
  - `strangemca_textarena.py`: strange-mca / TextArena adapter
  - `strange_basic_twoplayer.py`: Basic two-player game script
  - `strange_rendered_twoplayer.py`: Two-player game with rendering
- `docs/`: Design documents and RFCs
- `scripts/`: Linting and development scripts
- `output/`: Generated outputs, reports, and visualizations

## Testing

Run all unit tests:
```bash
poetry run pytest
```

Run a single test file or test by name:
```bash
poetry run pytest tests/test_agents.py
poetry run pytest -k "test_name"
```

Run live integration tests (requires `OPENAI_API_KEY`):
```bash
poetry run pytest -m live -v
```

Live tests hit the OpenAI API and verify end-to-end behavior including lateral communication, convergence trajectories, downward signals, and report output.

## Future Ideas and Improvements

This system mainly serves as a conceptual playground to model MCA and StrangeLoop in a multiagent system; this will be the focus, not building a production system or things like scale and performance. It may solve puzzles better than its non-MCA competitors one day; the current level of chess competition is about even and is about who can send an invalid response last, as you might expect. Future more powerful agents will make this a more interesting competition.

See [GitHub Issues](https://github.com/josecodes/strange-mca/issues) for planned improvements, organized by label:
- [`mca`](https://github.com/josecodes/strange-mca/labels/mca) - Multiscale Competency Architecture enhancements
- [`strange-loop`](https://github.com/josecodes/strange-mca/labels/strange-loop) - Strange Loop self-reflection improvements
- [`tech`](https://github.com/josecodes/strange-mca/labels/tech) - Technical improvements and maintenance

## Development

### Adding New Features

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Implement your changes
4. Run tests: `poetry run pytest`
5. Run linting: `./scripts/lint.sh`
6. Submit a pull request

### Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) and [Black](https://black.readthedocs.io/) for code linting and formatting:

```bash
# Run both Ruff and Black
./scripts/lint.sh

# Auto-fix issues
./scripts/lint.sh --fix
```

See `scripts/README.md` for more details on the linting scripts.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses [LangGraph](https://github.com/langchain-ai/langgraph) for graph-based execution
- TextArena integration is based on [TextArena](https://github.com/LeonGuertler/TextArena)
- Inspired by Michael Levin's work on [Multiscale Competency Architectures](https://pubmed.ncbi.nlm.nih.gov/37156924/) and Douglas Hofstadter's [Strange Loop](https://en.wikipedia.org/wiki/Strange_loop)
