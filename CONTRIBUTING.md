# Contributing

Thanks for contributing to the Waste Classification project.

## Development setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements-dev.txt
pre-commit install
```

The project targets **Python 3.12** with **TensorFlow 2.16.1**.

## Branching strategy

`main` is protected. Create a branch per phase or feature:

```
feat/<short-description>
fix/<short-description>
docs/<short-description>
```

Open a pull request into `main`; CI must pass before merge.

## Commit messages — Conventional Commits

```
<type>(<scope>): <subject>

<body>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

Example:

```
feat(data): add DataLoader with class distribution logging
```

Keep one logical change per commit; never commit broken code to `main`.

## Code style

- **Formatter:** `black --line-length 100`
- **Imports:** `isort --profile black`
- **Linter:** `flake8` (max line length 100)
- **Types:** `mypy` (lenient); type hints required on public functions
- **Docstrings:** Google style on all public functions and classes
- **Logging:** use the project logger (`src/logger.py`); no `print()` inside
  `src/` (CLI scripts may print user-facing output)
- **Errors:** raise custom exceptions from `src/exceptions.py`; log before
  raising; never silently swallow exceptions

Run all checks locally before pushing:

```bash
black --check --line-length 100 src/ scripts/ tests/
isort --check-only --profile black src/ scripts/ tests/
flake8 src/ scripts/ tests/
mypy src/
pytest --cov=src
```

## Testing

- Add a unit test in `tests/unit/` for every new module.
- TensorFlow-dependent tests must use `pytest.importorskip("tensorflow")` so
  they skip cleanly when TensorFlow is unavailable.
- Integration tests live in `tests/integration/` and use the `integration`
  marker; slow tests use the `slow` marker.

## Module conventions

- One responsibility per module; keep modules under ~300 lines.
- Public APIs get type hints and Google-style docstrings.
