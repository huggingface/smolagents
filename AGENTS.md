# Contributor Guidelines

## Setup

Python 3.10 or newer (CI runs 3.10 and 3.12).

```bash
pip install -e ".[dev]"
```

## Checks

Run these before opening a PR. Both are enforced in CI.

```bash
make quality   # ruff check + ruff format --check on examples/ src/ tests/
make test      # pytest ./tests/
```

`make style` auto-fixes lint and formatting — don't hand-format code.

## Repository layout

- `src/smolagents/` — library code
- `tests/` — pytest suite
- `examples/` — runnable examples
- `docs/` — documentation sources

## Code style

- Line length is 119; ruff enforces `E`, `F`, `I`, `W`.
- Imports are sorted by ruff's isort with `smolagents` as first-party.

## Guidelines

- Follow OOP principles
- Be Pythonic: follow Python best practices and idiomatic patterns
- Write unit tests for new functionality
