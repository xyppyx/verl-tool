# AutoDL RTX PRO 6000 Server

Last updated: 2026-07-17 15:22:29 +0800

This file records the remote GPU environment used for LiteVerlTool L1 reproduction and experiments. It intentionally does not store passwords, API keys, or service tokens.

## SSH access

- SSH alias: `verltool-autodl`
- Local SSH config: `~/.ssh/config`
- Key: `~/.ssh/id_ed25519`
- Passwordless login: verified with `ssh -o BatchMode=yes verltool-autodl 'echo SSH_OK'`
- Login user: `root`
- Hostname observed on server: `autodl-container-cvxyhxpra6-a27fb485`

Example:

```bash
ssh verltool-autodl
```

## Hardware

GPU:

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition
VRAM: 97887 MiB
Driver: 595.71.05
NVIDIA-SMI CUDA Version: 13.2
Current GPU processes: none during audit
```

CPU and RAM observed by container:

```text
CPU model: Intel(R) Xeon(R) Platinum 8470Q
Logical CPUs visible: 208
NUMA nodes: 2
RAM: 1.0 TiB total, 974 GiB available during audit
Swap: none
```

The purchased instance page may show a smaller logical allocation. Use the observed values as the runtime audit result, and re-check before long experiments.

## Disk

```text
/                 overlay   30G   53M   30G   1%
/root/autodl-tmp  /dev/md0  550G  12K   550G  1%
```

Recommended project layout:

```text
/root/autodl-tmp/FaultAwareVerlTool
/root/autodl-tmp/hf_cache
/root/autodl-tmp/data
/root/autodl-tmp/checkpoints
/root/autodl-tmp/runs
```

Keep large model caches, datasets, checkpoints, and rollout traces under `/root/autodl-tmp`, not the 30GB system disk.

## Python and CUDA

Miniconda exists at:

```text
/root/miniconda3
```

Conda environments:

```text
base  /root/miniconda3
```

Base Python:

```text
Python 3.12.3
/root/miniconda3/bin/python
```

Installed packages in base during audit:

```text
torch==2.8.0+cu128
vllm: not installed
transformers: not installed
ray: not installed
verl: not installed
```

Torch CUDA check:

```text
torch import: ok
torch version: 2.8.0+cu128
cuda available: True
cuda version: 12.8
gpu count: 1
gpu name: NVIDIA RTX PRO 6000 Blackwell Server Edition
bf16 supported: True
```

## Interpretation for LiteVerlTool

This server is suitable for L1 work:

- Qwen2.5-Math-1.5B GRPO smoke and pilot runs.
- Fault benchmark and Static-Fault training.
- Conservative 7B smoke experiments if dependency compatibility is solved.

Open environment questions before training:

- Whether VerlTool pinned dependencies work cleanly with Python 3.12 and Torch 2.8.
- Whether to create a project conda env with Python 3.10 or 3.11 to match the upstream dependency set more closely.
- Which vLLM version to use on Blackwell RTX PRO 6000: repo `requirements.txt` pins `vllm==0.8.4`, while project docs mention newer vLLM support.

Recommended next step:

Create an isolated conda environment under `/root/miniconda3/envs/lite-verltool` or install into a project venv under `/root/autodl-tmp`, then record exact package versions after installation.

