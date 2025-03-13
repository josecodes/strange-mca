# Strange MCA: Multi-Agent Conversation Architecture

Strange MCA is a hierarchical multi-agent system that uses a team of language models to solve complex tasks through task decomposition and response synthesis. The system creates a tree of agents that work together to break down problems, solve sub-problems, and synthesize a final response.

## Features

- **Hierarchical Agent Structure**: Configurable tree of agents with customizable depth and branching factor
- **Bidirectional Processing**: Top-down task decomposition and bottom-up response synthesis
- **Strange Loop Refinement**: Optional refinement of the final response through self-critique
- **Visualization Tools**: Generate visual representations of the agent tree and execution graph
- **TextArena Integration**: Run Strange MCA agents in game environments using TextArena

## Installation

### Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/) for dependency management

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/strange-mca.git
   cd strange-mca
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Running the Multi-Agent System

You can run the Strange MCA system using the command-line interface:

```bash
poetry run python -m src.strange_mca.main --task "Your task here" --child_per_parent 3 --depth 2 --model "gpt-3.5-turbo"
```

#### Command-line Options

- `--task`: The task to perform
- `--child_per_parent`: Number of children per parent node (default: 3)
- `--depth`: Number of levels in the tree (default: 2)
- `--model`: LLM model to use (default: "gpt-3.5-turbo")
- `--log_level`: Logging level (debug, info, warning, error, critical)
- `--all_logs`: Show logs from dependencies in addition to local logs
- `--viz`: Generate visualizations of the agent tree and execution graph
- `--dry_run`: Don't run the system, just show the configuration
- `--print_tree`: Print the agent tree structure
- `--print_details`: Print details about each agent
- `--domain_specific_instructions`: Domain-specific instructions for the strange loop prompt
- `--strange_loop_count`: Number of strange loop iterations to perform

### Programmatic Usage

You can also use Strange MCA programmatically in your Python code:

```python
from src.strange_mca.run_strange_mca import run_strange_mca

result = run_strange_mca(
    task="Explain the concept of multi-agent systems",
    child_per_parent=2,
    depth=2,
    model="gpt-3.5-turbo",
    viz=True,
    domain_specific_instructions="Focus on practical applications",
    strange_loop_count=1
)

print(result["final_response"])
```

### TextArena Integration

Strange MCA can be used with [TextArena](https://github.com/microsoft/TextWorld) to create agents that play games and solve interactive tasks. The integration is available in the `arena` directory.

#### Running a TextArena Game

```bash
poetry run python arena/strange_basic_twoplayer.py
```

This will run a two-player game (SpellingBee by default) with a Strange MCA agent competing against an OpenAI agent.

#### Available Game Scripts

- `strange_basic_twoplayer.py`: Runs a basic two-player game without rendering
- `strange_rendered_twoplayer.py`: Runs a two-player game with visual rendering

#### Creating Custom Game Scripts

You can create custom game scripts by following the template in the existing scripts. The key components are:

1. Initialize the Strange MCA agent with appropriate parameters
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
env = ta.make(env_id="YourGame-v0")
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

- `src/strange_mca/`: Core implementation of the Strange MCA system
  - `main.py`: Command-line interface and main execution script
  - `run_strange_mca.py`: Programmatic API for running the system
  - `graph.py`: Implementation of the execution graph using LangGraph
  - `agents.py`: Agent definitions and tree structure
  - `prompts.py`: Prompt templates for various stages of processing
  - `visualization.py`: Tools for visualizing the agent tree and execution
  - `logging_utils.py`: Utilities for detailed logging
- `arena/`: TextArena integration
  - `strangemca_textarena.py`: Integration of Strange MCA with TextArena
  - `strange_basic_twoplayer.py`: Basic two-player game script
  - `strange_rendered_twoplayer.py`: Two-player game with rendering
- `output/`: Generated outputs and visualizations

## Development

### Adding New Features

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Implement your changes
4. Run tests: `poetry run pytest`
5. Submit a pull request

### Code Style

This project follows PEP 8 guidelines. You can check your code style with:

```bash
poetry run flake8
```

### Linting

This project uses Ruff and Black for code linting and formatting. You can run the linting scripts with:

```bash
# Run both Ruff and Black on the default directories
./scripts/lint.sh

# Fix issues automatically
./scripts/lint.sh --fix
```

See `scripts/README.md` for more details on the linting scripts.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses [LangGraph](https://github.com/langchain-ai/langgraph) for graph-based execution
- TextArena integration is based on [TextArena](https://github.com/microsoft/TextWorld) 