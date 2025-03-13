# Strange MCA Arena

This directory contains scripts for running the Strange MCA agent team in an OpenAI Gym style environment using TextArena. TextArena provides a collection of competitive text-based games for language model evaluation and reinforcement learning.

## Prerequisites

Before running any of the scripts, make sure you have set up your environment variables. Create a `.env` file in the project root directory with the following variables:

```
OPENAI_API_KEY=your_openai_api_key
```

Replace `your_openai_api_key` with your actual OpenAI API key.

This project uses a local installation of TextArena from `/Users/jcortez/dev/TextArena`. The dependency is configured in the Poetry configuration, so it should be installed automatically when you run `poetry install`.

## Available Scripts

### 1. `strange_basic_twoplayer.py`

This script demonstrates how to use TextArena to run a Chess game between a Strange MCA agent team and an OpenAI model.

To run the script:

```bash
poetry run python examples/arena/strange_basic_twoplayer.py
```

### 2. `strange_rendered_twoplayer.py`

This script demonstrates how to use TextArena to run a Chess game between a Strange MCA agent team and an OpenAI model, with visual rendering.

To run the script:

```bash
poetry run python examples/arena/strange_rendered_twoplayer.py
```

## Game State Display

The example scripts have been enhanced to display detailed game state information without relying on render wrappers. This includes:

- Current player information
- Game-specific state (e.g., board state in Chess)
- Player observations and actions
- Game results with player rewards

This approach provides a clean and informative display that works in any terminal size.

## Render Wrapper Options

While the `strange_basic_twoplayer.py` script uses no render wrapper for reliability, the `strange_rendered_twoplayer.py` script uses the CursesRenderWrapper for a more interactive and visually appealing interface.

You can use one of the following render wrappers in your own scripts:

1. **SimpleRenderWrapper**: A basic render wrapper that displays the game state in a formatted box. This can cause display issues in some terminals.

```python
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "Strange MCA Team", 1: "OpenAI"},
)
```

2. **CursesRenderWrapper**: A more advanced wrapper that uses the curses library for terminal rendering. This provides a nicer UI but requires a terminal that supports curses.

```python
env = ta.wrappers.RenderWrappers.CursesRenderWrapper(
    env=env,
    player_names={0: "Strange MCA Team", 1: "OpenAI"},
)
```

## Available Games

TextArena provides a wide range of games, including:

- Single-player games: Crosswords, FifteenPuzzle, GuessTheNumber, Hangman, Mastermind, Sudoku, Wordle, etc.
- Two-player games: Chess, ConnectFour, Debate, TicTacToe, Battleship, Checkers, etc.
- Multi-player games: Poker, LiarsDice, Negotiation, etc.

For a complete list of available games, refer to the [TextArena GitHub repository](https://github.com/LeonGuertler/TextArena).

## Creating Custom Scripts

To create a custom script for a different game, follow this template:

```python
import os
import sys
import textarena as ta
from dotenv import load_dotenv

# Add the project root to the Python path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from examples.arena.strangemca_textarena import StrangeMCAAgent

# Load environment variables
load_dotenv()

# Initialize agents
agents = {
    0: StrangeMCAAgent(
        child_per_parent=2,
        depth=2,
        model="gpt-4o-mini",
        domain_specific_instructions="Your game-specific instructions here",
    ),
    1: ta.agents.OpenAIAgent(
        model_name="gpt-4o-mini",
        system_prompt="Your system prompt here",
    ),
}

# Initialize environment and wrap it
env = ta.make(env_id="GameName-v0")  # Replace with the desired game
env = ta.wrappers.LLMObservationWrapper(env=env)

# Reset the environment
env.reset(num_players=len(agents))

# Game loop
done = False
while not done:
    # Get the current player ID and observation
    player_id, observation = env.get_observation()
    
    # Get action from the agent
    action = agents[player_id](observation)
    
    # Call step and handle the return values
    done, info = env.step(action=action)

# Close the environment and get rewards
rewards = env.close()

# Print the results
print("Rewards:")
if rewards:
    print(rewards)
else:
    print("No results available.")
```

Replace `"GameName-v0"` with the ID of the game you want to play, and adjust the agent parameters as needed. 