# Strange MCA Examples

This directory contains example scripts that demonstrate how to use the Strange MCA system programmatically.

## Prerequisites

Before running any of the examples, make sure you have set up your environment variables. Create a `.env` file in the project root directory with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
```

Replace `your_openai_api_key` with your actual OpenAI API key.

## Available Scripts

### 1. `run_examples.py`

This script runs a series of examples with different configurations to demonstrate the capabilities of the Strange MCA system.

To run all examples:

```bash
python examples/run_examples.py
```

This will run 5 different examples with various configurations:
- Basic configuration (2 children per parent, 2 levels deep)
- More children per parent (3 children per parent, 2 levels deep)
- Deeper tree (2 children per parent, 3 levels deep)
- Custom output directory
- No visualizations

### 2. `run_single_task.py`

This script allows you to run a single task with custom parameters.

Example usage:

```bash
python examples/run_single_task.py --task "Explain the concept of recursion" --child_per_parent 2 --depth 2 --viz
```

Available parameters:

- `--task`: The task to run (required)
- `--child_per_parent`: Number of children per parent node (default: 2)
- `--depth`: Depth of the tree (default: 2)
- `--model`: Model to use (default: "gpt-3.5-turbo")
- `--log_level`: Log level (default: "info")
- `--viz`: Generate visualizations
- `--no_viz`: Do not generate visualizations
- `--only_local_logs`: Only use local logs
- `--print_details`: Print detailed information
- `--no_print_details`: Do not print detailed information
- `--output_dir`: Output directory to use

## Using the API in Your Own Code

You can also import the `run_strange_mca` function directly in your own code:

```python
from dotenv import load_dotenv
from src.strange_mca.run_strange_mca import run_strange_mca

# Load environment variables
load_dotenv()

# Run a task with custom parameters
result = run_strange_mca(
    task="Your task here",
    child_per_parent=2,
    depth=2,
    model="gpt-3.5-turbo",
    log_level="info",
    viz=True,
)

# Access the final response
final_response = result.get("final_response", "No final response generated")
print(final_response)
```

This allows you to integrate the Strange MCA system into your own applications and workflows. 