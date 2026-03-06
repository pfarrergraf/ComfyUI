# AI Copilot Instructions — ComfyUI Workspace

## Virtual Environment: HANDS OFF

- The project uses a **uv-managed** virtual environment at **`.venv`** (not `venv`).
- **NEVER** create a new virtual environment (no `python -m venv`, no `virtualenv`, no `conda create`).
- **NEVER** install, uninstall, upgrade, or downgrade any package inside `.venv` without **explicit, written user approval** of the exact command(s), package name(s), version(s), and index URL(s).
- **NEVER** run `pip install`, `pip uninstall`, `uv pip install`, `uv pip sync`, or any similar command against `.venv` on your own initiative.
- This applies **especially** to `torch`, `torchvision`, `torchaudio`, and any CUDA-related packages. These are installed from a specific PyTorch index with a specific CUDA version. A wrong install will break the entire setup.

## Package Management: UV Only

- All package operations **must** use `uv` (e.g., `uv pip install`, `uv pip freeze`, `uv pip sync`).
- Never use bare `pip` or `python -m pip`. The environment is managed by `uv`.
- The frozen state of `.venv` is recorded in `requirements-frozen.txt` at the repo root (plain pip-freeze format — **not** a TOML `uv.lock`). Before ANY approved change, create a fresh backup:
  ```powershell
  uv pip freeze --python .venv\Scripts\python.exe > requirements-frozen.bak
  ```
- After any approved change, update the lock:
  ```powershell
  uv pip freeze --python .venv\Scripts\python.exe > requirements-frozen.txt
  ```
- **NEVER** create a file called `uv.lock` — `uv run` expects that name to be TOML and it will break.

## Before Recommending CUDA / PyTorch Changes

1. Run and present results of:
   ```powershell
   nvidia-smi
   .venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), getattr(torch.version, 'cuda', None))"
   ```
2. Show the exact install command, index URL, and target CUDA version.
3. **Wait for explicit user confirmation** before executing anything.

## Code & Git Safety

- Always create a backup branch (`git checkout -b backup/auto-<timestamp>`) and commit uncommitted changes before any destructive operation.
- Note: git backups do **not** protect `.venv`. If `.venv` changes are approved, the `uv.lock.bak` backup above is the recovery path.

## General Rules

- If unsure about any environment, package, or CUDA compatibility question: **stop and ask**. Do not guess.
- Never run destructive commands against system-level Python, CUDA drivers, or global packages.
- Always use `.venv` (dot-prefix) — never `venv`.
- Always use `uv` — never bare `pip`.
