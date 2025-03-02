# Strange MCA

A multiagent system built with LangChain that implements a configurable tree-like structure.

## Project Overview

This project implements a hierarchical multiagent system where:
- Each node in the graph (except leaf nodes) has X child nodes that act as a team
- The tree can have Y number of levels
- Default configuration: X=3, Y=2 (1 supervisor node + 3 children nodes)
- Naming convention: L1N1 for the top node, L2N1, L2N2, L2N3 for children nodes
- Each parent node defines the system prompts for its child team

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. Make sure you have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then, install the project dependencies:

```bash
poetry install
```

For visualization support, install the optional dependencies:

```bash
poetry install --with viz
```

### Setting up OpenAI API Key

This project uses OpenAI's API for the language models. You need to set up your API key:

1. Copy the `.env.example` file to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and replace `your_openai_api_key_here` with your actual OpenAI API key.

3. The application will automatically load this key when it runs.

## Development

Activate the virtual environment:

```bash
poetry shell
```

Run tests:

```bash
poetry run pytest
```

## Usage

To run the multiagent system with default configuration (X=3, Y=2):

```bash
poetry run python -m src.strange_mca.main
```

To customize the configuration:

```bash
poetry run python -m src.strange_mca.main --child_per_parent 4 --depth 3
```

### Visualization Options

Print the agent tree structure:

```bash
poetry run python -m src.strange_mca.main --print_tree
```

Print detailed information about each agent:

```bash
poetry run python -m src.strange_mca.main --print_details
```

Generate a visual graph representation (requires graphviz):

```bash
poetry run python -m src.strange_mca.main --visualize --output viz_outputs/agent_graph --format png
```

Run a dry run (only show configuration without executing the agents):

```bash
poetry run python -m src.strange_mca.main --dry_run --print_tree
```

### Example Agent Tree (X=3, Y=2)

```
Agent Tree:
└─ L1N1 (Level 1)
  └─ L2N1 (Level 2)
  └─ L2N2 (Level 2)
  └─ L2N3 (Level 2)
```

### Example Agent Tree (X=2, Y=3)

```
Agent Tree:
└─ L1N1 (Level 1)
  └─ L2N1 (Level 2)
    └─ L3N1 (Level 3)
    └─ L3N2 (Level 3)
  └─ L2N2 (Level 2)
    └─ L3N3 (Level 3)
    └─ L3N4 (Level 3)
```

## License

[MIT](https://choosealicense.com/licenses/mit/) 