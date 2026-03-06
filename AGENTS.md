# Repository Agent Instructions

## Environment Management
- This project uses `uv` with a committed `uv.lock`.
- Do not change anything directly inside `.venv`.
- Do not use `pip` against `.venv` for installs, upgrades, or removals.
- Manage dependency changes via `uv` and project metadata (`pyproject.toml`), then update `uv.lock`.
- Preferred commands:
  - `uv sync`
  - `uv add <package>`
  - `uv remove <package>`
