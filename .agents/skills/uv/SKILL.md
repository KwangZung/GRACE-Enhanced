---
name: uv
description: Common tasks and best practices for using the uv Python package and project manager.
---

# `uv` Python Package and Project Manager Skill

`uv` is an extremely fast Python package and project manager written in Rust. It replaces tools like `pip`, `poetry`, `pipx`, and `pyenv`.

## 1. Project Initialization

To create a new Python project or initialize an existing directory:

```bash
# Create a new project in a new directory
uv init my-project
cd my-project

# Or initialize in the current directory
uv init
```
This sets up a standard structure including `pyproject.toml` and allows for easy virtual environment management.

## 2. Managing Dependencies

`uv` manages project dependencies and automatically synchronizes lockfiles (`uv.lock`) and virtual environments (`.venv`).

```bash
# Add a dependency
uv add requests

# Add a specific version or constraint
uv add 'requests==2.31.0'
uv add 'flask>=2.0,<3.0'

# Add a git dependency
uv add git+https://github.com/psf/requests

# Add optional dependencies (extras)
uv add 'flask[async]'

# Add a development dependency
uv add --dev pytest

# Remove a dependency
uv remove requests

# Sync the virtual environment with the lockfile
uv sync

# Lock dependencies without syncing
uv lock

# Upgrade a specific package or all packages
uv lock --upgrade-package requests
uv lock --upgrade
```

## 3. Running Scripts and Commands

You can execute Python scripts and CLI tools within the managed virtual environment without needing to activate it manually.

```bash
# Run a Python script
uv run script.py

# Run a script with ad-hoc dependencies (without modifying the project)
uv run --with rich script.py
uv run --with 'rich>12,<13' script.py

# Run a CLI command provided by a dependency (e.g., pytest, flask)
uv run pytest tests
uv run -- flask run -p 3000

# Run with a specific Python version
uv run --python 3.10 script.py

# Skip project installation for standalone scripts
uv run --no-project script.py
```

## 4. Managing Python Versions

`uv` can download and manage Python interpreters, replacing tools like `pyenv`.

```bash
# Install specific Python versions
uv python install 3.12
uv python install 3.11 3.12

# Install the latest Python version
uv python install

# List available and installed versions
uv python list

# Pin Python version for the current directory (creates/updates .python-version file)
uv python pin 3.11

# Upgrade installed Python versions
uv python upgrade 3.12
uv python upgrade  # Upgrade all
```

## Best Practices for Agents

1. **Avoid `pip` and `venv` directly**: When working in a repository that uses `uv` (identified by `uv.lock` or `pyproject.toml` managed by `uv`), always prefer `uv add`, `uv sync`, and `uv run` over `pip install` or manually activating virtual environments.
2. **Use `uv run` for execution**: Instead of activating a virtual environment, simply prepend your command with `uv run` (e.g., `uv run python script.py` or `uv run pytest`).
3. **Ad-hoc dependencies**: When you need a package just to run a one-off script, use `uv run --with <package> script.py` instead of adding it to the project dependencies.
4. **Environment Syncing**: After pulling changes or manually editing `pyproject.toml`, run `uv sync` to ensure the environment matches the lockfile.
