# Project TODO

Last updated: 2026-07-17 15:22:29 +0800

Current target: L1 internship-ready project from `docs/design/LITE_VERLTOOL_INTERNSHIP_PROJECT.md`.

## Phase 0 - Repository and environment audit

- [ ] Remove or quarantine hardcoded secrets from Math-TIR scripts.
  - `examples/train/math_tir/train_1.5b_grpo.sh` contains a hardcoded `WANDB_API_KEY`.
  - Do not copy the value into logs, docs, commits, or issue text.
  - Rotate the exposed WandB key before using this repo for any public work.
  - Replace with `${WANDB_API_KEY:?set WANDB_API_KEY}` or make WandB optional.
- [ ] Convert personal or machine-specific script settings to parameters.
  - Audit fixed `CUDA_VISIBLE_DEVICES`, fixed GPU counts, random port ranges, `$(pwd)` data paths, run names, and log paths.
  - Prefer project-owned scripts under `scripts/lite_verltool/` and configs under `configs/lite_verltool/`.
- [ ] Decide the first reproducible environment target.
  - Recommended for local work: docs, static checks, CPU unit tests, mock Tool Server.
  - Recommended for remote GPU: 4x24GB minimum for 1.5B smoke/pilot; 4x40GB or 4x48GB for fewer OOM iterations.
- [ ] Create a clean environment on the remote machine.
  - Remote server login is verified; see `docs/server/AUTODL_RTX_PRO_6000.md`.
  - Base environment has Python 3.12.3 and `torch==2.8.0+cu128`.
  - Decide whether to use Python 3.12 base or create a Python 3.10/3.11 project env for upstream compatibility.
  - Install `vllm`, `transformers`, `ray`, and project packages in an isolated environment.
  - Record exact package versions after install.
- [ ] Initialize only needed benchmark submodules.
  - For Math-TIR L1 work, do not initialize every benchmark unless required.
  - Record any submodule initialization or pointer changes before committing.
- [ ] Run a secret and path audit before the first public commit.
  - Search for API keys, tokens, absolute home paths, fixed usernames, and remote service URLs.
  - If a secret is found, remove it and rotate externally.

## Phase 0 - Minimal validation checklist

- [ ] Verify `python -m compileall` or targeted imports after dependency install.
- [ ] Run the existing Tool Server tests that do not require external services.
- [ ] Run a CPU/mock two-turn agent loop test before any GPU training.
- [ ] Run one 1-step remote GPU smoke test with logs, config, peak VRAM, and wall-clock recorded.

## Phase 1 - L1 project implementation

- [ ] Implement deterministic transient fault injection.
  - Stable SHA256-based decision key.
  - One injected fault per trajectory.
  - Parser accepted action only.
  - Observation has structured metadata.
  - Fault returns `done=false`.
- [ ] Add fault-aware metrics.
  - `CleanAcc`
  - `TriggeredFaultAcc`
  - `MatchedCleanAcc`
  - `ConditionalRecovery`
  - `MatchedRobustnessGap`
- [ ] Implement Static-Fault training config.
- [ ] Save small, sanitized trajectory samples for interview/demo use.
- [ ] Write the final project report with resource cost, negative results, and limitations.

## Current blockers

- Local machine is not a valid training environment yet: core packages are not installed, and only one 8GB laptop GPU is visible.
- Remote GPU hardware is provisioned and login works, but the project Python environment is not installed yet.
- Hardcoded WandB credential must be removed and rotated before public sharing.
