# Strange MCA Examples

This directory contains example use cases and integrations for the Strange MCA system.

## Arena

The `arena` directory contains examples of using Strange MCA with TextArena to create agents that play games and solve interactive tasks.

### Running a TextArena Game

```bash
poetry run python examples/arena/strange_basic_twoplayer.py
```

This will run a two-player Chess game with a Strange MCA agent competing against an OpenAI agent.

### Available Game Scripts

- `strange_basic_twoplayer.py`: Runs a basic two-player Chess game without rendering
- `strange_rendered_twoplayer.py`: Runs a two-player Chess game with visual rendering

See the [arena README](arena/README.md) for more details on the TextArena integration.

## Adding New Examples

To add a new example:

1. Create a new directory or file in the `examples` directory
2. Add a README.md file explaining the example
3. Update the main README.md to reference the new example 