# Project Completed

Last updated: 2026-07-17 15:22:29 +0800

This file records only completed facts with evidence. Planned work and unverified claims belong in `PROJECT_TODO.md`.

## 2026-07-17 - Repository baseline recorded

- Current fork branch: `main`, tracking `origin/main`.
- Current fork commit:
  - Full SHA: `383d4b1539ba387f94c3a117d3edc06b467c09d1`
  - Short SHA: `383d4b1`
  - Commit date: `2026-07-15 11:11:03 -0700`
  - Subject: `Update README.md`
- Remote configuration:
  - `origin`: `https://github.com/xyppyx/verl-tool.git`
  - `upstream`: `https://github.com/TIGER-AI-Lab/verl-tool.git`
- Upstream verification:
  - Command: `git ls-remote --heads upstream main`
  - Result: upstream `main` points to `383d4b1539ba387f94c3a117d3edc06b467c09d1`.

## 2026-07-17 - Benchmark submodule pins recorded

The benchmark directories are Git submodules. They are pinned in `HEAD`, but are not initialized in the current working tree, as indicated by leading `-` in `git submodule status --recursive`.

| Path | Pinned SHA |
| --- | --- |
| `benchmarks/LiveCodeBench` | `125a8de39b2d88f214fbde8a22d034a04f774d11` |
| `benchmarks/MCP-Universe` | `326d29e2c05c900fd45eef524700ecc04c142b8a` |
| `benchmarks/bigcodebench` | `eb400cdfaba72ae9868338a3710459034211469f` |
| `benchmarks/evalplus` | `221984301dc2cace122baf11a937e5b7fe3069fb` |
| `benchmarks/math-evaluation-harness` | `9271e69bece4d14b33340df050c469996f1d6ab1` |

## 2026-07-17 - Project design and collaboration docs present

- Current design document exists at `docs/design/LITE_VERLTOOL_INTERNSHIP_PROJECT.md`.
- Project collaboration rules exist at `AGENTS.md`.
- `docs/raw-design/` is retained as historical context, not the active implementation source.

## 2026-07-17 - Remote GPU server login verified

- Server record: `docs/server/AUTODL_RTX_PRO_6000.md`.
- SSH alias configured locally: `verltool-autodl`.
- Passwordless login verified with `ssh -o BatchMode=yes verltool-autodl 'echo SSH_OK'`.
- GPU observed: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, `97887 MiB` VRAM.
- Driver observed: `595.71.05`; NVIDIA-SMI reports CUDA Version `13.2`.
- Miniconda observed at `/root/miniconda3`.
- Base Python observed: `Python 3.12.3`.
- Base Torch observed: `torch==2.8.0+cu128`; CUDA available; BF16 supported.
- `vllm`, `transformers`, `ray`, and `verl` were not installed in the base environment during audit.
