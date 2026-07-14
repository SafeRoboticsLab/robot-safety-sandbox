# Installation

Two packages, one benchmark:

| repo | provides | depends on |
|---|---|---|
| [`safety-stable-baselines`](https://github.com/SafeRoboticsLab/safety-stable-baselines) (`safety_sb3`) | the ALGORITHMS: SafetyPPO / ReachAvoidPPO / IsaacsPPO (+ SAC variants), tensor path, buffers | SB3, torch |
| `robot-safety-sandbox` (this repo) | the ENVIRONMENTS: task registry (safety + nominal layers), margins, bridges, trainers | mjlab, safety_sb3 |

Neither depends on `unitree_rl_mjlab` (the legacy research repo) — the zoo is
self-contained and this independence is CI-tested by installing on a clean
machine (see Verification).

## Requirements

- Linux, NVIDIA GPU (tested: RTX 4070 / Ada `sm_89`, RTX 5090 / Blackwell
  `sm_120`), recent driver (CUDA 12.8+ runtime compatibility).
- Python ≥ 3.10, conda recommended.

## Install (from scratch)

```bash
conda create -y -n mjlab python=3.10
conda activate mjlab

# 1. torch — cu128 wheels cover BOTH Ada (sm_89) and Blackwell (sm_120)
#    (validated with 2.10.0+cu128 and 2.11.0+cu128):
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. STOCK mjlab, pinned to the API the zoo targets — WITH its sim stack
#    pinned too. mjlab 1.2.0 does not pin mujoco/warp itself, and it sets
#    sim options (e.g. ls_parallel) that were REMOVED in MuJoCo Warp 3.9.1,
#    so an unpinned install breaks at env build. It also imports scipy
#    without declaring it.
pip install "mjlab==1.2.0" "mujoco==3.6.0" "mujoco-warp==3.6.0" \
            "warp-lang==1.12.0" scipy

# 3. this package. safety_sb3 is a declared pip dependency (pinned release
#    tag from GitHub), so one editable install pulls both:
pip install wandb tensorboard imageio moviepy
pip install -e path/to/robot-safety-sandbox

#    developing BOTH packages at once? install safety_sb3 editable FIRST and
#    it satisfies the requirement:
# pip install -e path/to/safety-stable-baselines
# pip install -e path/to/robot-safety-sandbox
```

**Headless machines** (no display — clusters, ssh boxes): eval-video rendering
needs `export MUJOCO_GL=egl` (NVIDIA EGL; verified on a headless RTX 5090) and
`moviepy` (wandb's video encoder). Without them training crashes at the first
video interval with an OpenGL-context / `wandb.Video requires moviepy` error.

## Verification

```bash
# registry imports; Digit tasks warn-and-skip on stock mjlab (expected — see Notes)
python -c "from robot_safety_sandbox import list_tasks; print(list_tasks())"

# algorithm math (CPU, no GPU/simulator needed):
python path/to/safety-stable-baselines/tests/test_tensor_sac.py

# end-to-end GPU training smoke (~1 min; warp JIT-compiles kernels for your
# arch on the FIRST env build — expect a one-time pause):
python examples/train.py --task hopper_safety --num-envs 256 --steps 500000 \
    --net 128,128,128 --ent-coef 0 --no-adaptive-lr --no-wandb
```

Trainers: `examples/train.py` (safety layer, safety_sb3 learners),
`examples/train_nominal.py` (nominal task policies, vanilla SB3),
`examples/eval_filter.py` (value-filter composition). See README/PORTING.md.

## Notes & known deviations

- **Digit tasks** (`digit_stabilize*`) need the Digit asset from the
  SafeRoboticsLab mjlab fork; on stock mjlab they are skipped with a warning
  at import (everything else works). Vendoring the asset is tracked for a
  future release.
- **mjlab version**: the zoo targets the mjlab **1.2** API. Newer mjlab
  (1.3–1.5) has not been validated; known 1.1↔1.2 drift already required
  compat shims once, so keep the pin unless you are prepared to re-run the
  verification suite.
- **The sim-stack pins are mandatory, not conservative**: unpinned
  mujoco-warp (3.10 at time of writing) removes options mjlab 1.2.0 sets →
  `AttributeError: ls_parallel was removed in MuJoCo Warp 3.9.1` at the first
  env build. The pinned triple above is the validated set.
- **GPU memory**: per-task footprints scale with `--num-envs`; see the
  task docstrings. The classic-control ports (hopper etc.) are tiny (<1 GB);
  the Go2 tasks want 4–9 GB at 2–3k envs.
- **Cross-arch**: no code changes needed between sm_89 and sm_120 — only the
  warp kernel cache differs (per-machine, auto-built).
