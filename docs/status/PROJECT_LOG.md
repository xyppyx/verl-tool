# Project Log

## 2026-07-17 - Initial status files and upstream remote

Created the status fact source required by `AGENTS.md`:

- `docs/status/PROJECT_COMPLETED.md`
- `docs/status/PROJECT_TODO.md`
- `docs/status/PROJECT_LOG.md`

Configured official upstream:

- `git remote add upstream https://github.com/TIGER-AI-Lab/verl-tool.git`
- Verified upstream `main` currently points to `383d4b1539ba387f94c3a117d3edc06b467c09d1`.

## 2026-07-17 - Local environment audit

Local command results:

- `python --version`: `Python 3.14.4`
- Installed Python packages checked through `importlib.metadata`:
  - `torch`: not installed
  - `vllm`: not installed
  - `transformers`: not installed
  - `ray`: not installed
  - `verl`: not installed
- Visible GPU:
  - `NVIDIA GeForce RTX 4060 Laptop GPU`
  - Memory: `8188 MiB`
  - Driver: `610.62`
- `nvcc` was not found in `PATH` during the quick check.

Interpretation:

- This local machine is suitable for documentation, static audit, small CPU tests, and mock integration.
- It is not suitable for real VerlTool GRPO training.
- First real training should happen on a clean remote GPU environment.

## 2026-07-17 - Dependency notes from repo files

Observed dependency signals:

- Root `pyproject.toml` requires Python `>=3.10`.
- Root `requirements.txt` pins notable training dependencies:
  - `torch==2.6.0`
  - `ray==2.43.0`
  - `transformers==4.51.3`
  - `vllm==0.8.4`
  - CUDA 12 package pins such as `nvidia-cuda-runtime-cu12==12.4.127`
- Root `pyproject.toml` optional `vllm` extra allows `vllm<=0.11.0`.
- README says the reorganized codebase supports latest verl `0.6.0` and vLLM `0.11.0`, while `requirements.txt` still pins `vllm==0.8.4`.

Interpretation:

- Do not assume one dependency path is correct before testing.
- Phase 0 should choose one reproducible environment recipe and record it.
- For a quick L1 reproduction, prefer an upstream known-good recipe over opportunistic upgrades.

## 2026-07-17 - Script audit notes

Math-TIR script issues observed:

- `examples/train/math_tir/train_1.5b_grpo.sh` contains a hardcoded `WANDB_API_KEY`; the value must be removed from the repo and rotated externally.
- Math-TIR scripts include fixed or assumption-heavy settings such as GPU IDs/counts, local `$(pwd)` data paths, random port selection, workers per tool, and WandB project/run naming.
- `ipython_code` is used as a tool type in Math-TIR scripts; generated code execution must be sandboxed before real model runs.

Interpretation:

- Do not run the upstream Math-TIR scripts as-is.
- First implementation work should create project-owned, parameterized smoke scripts and configs under the LiteVerlTool namespace.

## 2026-07-17 - Remote AutoDL server audit

Configured local passwordless SSH:

- SSH alias: `verltool-autodl`
- Key file: `~/.ssh/id_ed25519`
- Verification command: `ssh -o BatchMode=yes verltool-autodl 'echo SSH_OK; hostname; whoami; date; pwd'`
- Result: login succeeded as `root`.

Server audit summary:

- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- VRAM: `97887 MiB`
- Driver: `595.71.05`
- NVIDIA-SMI CUDA Version: `13.2`
- Torch CUDA version in base env: `12.8`
- Torch CUDA available: `True`
- BF16 supported: `True`
- Miniconda path: `/root/miniconda3`
- Base Python: `Python 3.12.3`
- Base package state:
  - `torch==2.8.0+cu128`
  - `vllm`: not installed
  - `transformers`: not installed
  - `ray`: not installed
  - `verl`: not installed
- Disk:
  - `/`: 30GB overlay
  - `/root/autodl-tmp`: 550GB local data disk

Interpretation:

- Hardware is sufficient for L1 work.
- Do not install large dependencies or store checkpoints on `/`; use `/root/autodl-tmp`.
- Dependency compatibility remains open because upstream repo pins differ from the base environment.
