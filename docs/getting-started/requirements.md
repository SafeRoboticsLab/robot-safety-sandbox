# Requirements

Read this before installing. The sim-stack version pins are **mandatory**, not
conservative — an unpinned install breaks at the first environment build.

## Compatibility matrix

| Component | Supported / tested version |
|---|---|
| OS | **Linux only** |
| Python | ≥ 3.10 (3.10 documented; 3.11 also used by the maintainers) |
| GPU | **NVIDIA required**; tested on RTX 4070 / Ada `sm_89` and RTX 5090 / Blackwell `sm_120` |
| NVIDIA driver | CUDA 12.8+ runtime compatibility |
| PyTorch | cu128 wheels — validated with `2.10.0+cu128` and `2.11.0+cu128` |
| mjlab | `1.2.0` (the 1.2 API; 1.3+ **not** validated) |
| MuJoCo | `3.6.0` |
| mujoco-warp | `3.6.0` |
| warp-lang | `1.12.0` |
| `safety_sb3` | `v0.4.0` (pinned git dependency) |
| Digit simulation | **custom mjlab fork required** (imports on stock mjlab, but stepping the sim needs the fork's entity patch) |

"Supported" here means "the version this release was written and tested against."
Other versions may work but are not validated; the sim-stack triple
(mjlab / MuJoCo / mujoco-warp) in particular must match.

## Why the pins are mandatory

mjlab 1.2.0 does not pin its own sim stack and sets simulation options (e.g.
`ls_parallel`) that were **removed** in MuJoCo Warp 3.9.1. An unpinned
`mujoco-warp` therefore raises
`AttributeError: ls_parallel was removed in MuJoCo Warp 3.9.1` at the first
environment build. mjlab 1.2.0 also imports `scipy` without declaring it. The
pinned triple above is the validated set.

## No PyPI release

There is **no PyPI package**. Install both repositories from git:

- `robot-safety-sandbox` (this package) is installed editable: `pip install -e .`.
- `safety_sb3` is a **pinned git dependency** in `pyproject.toml`
  (`safety_sb3 @ git+https://github.com/SafeRoboticsLab/safety-stable-baselines.git@v0.4.0`),
  so a single editable install of this package pulls it in.

The editable install is currently the supported path (the package is not yet
published as a wheel). See [installation](../installation.md) for the full
procedure, including the CPU-only import check and the GPU simulation smoke test.

## Headless machines

Video rendering (the eval `--video` flag and wandb training videos) needs an
offscreen GL context: `export MUJOCO_GL=egl` plus `moviepy`. Without them a run
crashes at the first video interval with an OpenGL-context error. Training
itself does not need a display.
