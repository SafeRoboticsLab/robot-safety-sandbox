# Quick start

This is the five-minute path: install, verify, smoke-train the tutorial task, and
inspect the result. For a from-scratch build of a task, see the
[car-goal tutorial](../tutorial-car-goal.md).

!!! info "Before you start"
    You need **Linux + an NVIDIA GPU** and the [pinned sim stack](requirements.md).
    Follow [installation](../installation.md) first. The **first** environment
    build JIT-compiles Warp kernels for your GPU — expect a one-time pause of up
    to a minute before the first run prints anything.

## 1. Verify the install (CPU only)

No GPU or simulator needed — this just confirms the registry imports:

```bash
python -c "from robot_safety_sandbox import list_tasks; print(list_tasks())"
```

You should see a list of ~45 task IDs (`car_goal`, `go2_stabilize`, …). Digit
tasks warn-and-skip on stock mjlab; that is expected.

## 2. Smoke-train the tutorial task

`car_goal` is a small differential-drive reach-avoid task. This short run trains
a real policy for a few seconds and writes a checkpoint:

```bash
python examples/train.py --config configs/car_goal.yaml \
    --num-envs 256 --steps 200000 --no-wandb
```

- **Expected runtime:** a few seconds of training once Warp is compiled (the
  first-ever build adds a one-time compilation pause).
- **Output:** written to `runs/car_goal/` — `final_model.zip` (the policy),
  `tensornormalize.pt` (the observation normalizer), `config.yaml` (the fully
  resolved recipe), and a `PPO_1/` TensorBoard directory.
- **No checkpoint required** to start — training creates one.

!!! note "This is a verification run, not the full recipe"
    The full `car_goal` recipe (`configs/car_goal.yaml`) trains for **25M**
    env-steps and reaches ≈60–67% of goals with near-zero violations. The
    `--num-envs 256 --steps 200000` overrides above shrink it to a smoke test.
    Drop the overrides (and add wandb) to run it for real:
    `python examples/train.py --config configs/car_goal.yaml`.

!!! warning "`--smoke` is SAC-only"
    The `--smoke` flag exists on the **off-policy (SAC)** trainer only. `car_goal`
    is an on-policy (PPO) task, so shrink it with `--num-envs` / `--steps` as
    above rather than `--smoke`.

## 3. Evaluate the checkpoint

Report the policy's reach and safety rates on a batch of randomized spawns:

```bash
python examples/eval.py --task car_goal \
    --safety-policy runs/car_goal/final_model.zip --safety-only \
    --no-filter --num-envs 512 --steps 700
```

This prints `task_success` (reach rate), `safe_rate`, and `violation_rate`, among
other metrics. `--safety-only` drives the policy with its own control actor;
`--no-filter` runs it unfiltered. (A smoke-trained checkpoint will score low —
that is expected; train it for 25M steps to reproduce the tutorial numbers.)

## 4. Watch it (optional)

If you have a display, render the trained policy in an interactive viewer:

```bash
python examples/play.py --task car_goal --algo ReachAvoidPPO1P --run runs/car_goal
```

On a headless box use `--viewer viser` (prints a browser URL) and set
`MUJOCO_GL=egl`. See the [CLI reference](../reference/cli.md) for `play.py`
options.

## What success looks like

| Stage | Success signal |
|---|---|
| Verify | `list_tasks()` prints the task IDs without error |
| Train | telemetry table streams; `ep_rew_mean` climbs; `runs/car_goal/final_model.zip` is written |
| Evaluate | a metrics summary prints with `task_success`, `safe_rate`, `violation_rate` |

## Next steps

- **[Choose a workflow](../guide/workflows.md)** — safety-policy training,
  task-policy training, filter evaluation, or filtered training.
- **[Environments catalog](../environments/index.md)** — the full task list with
  maturity labels.
- **[Tutorial: build the car-goal task](../tutorial-car-goal.md)** — construct a
  task from scratch, theory-first.
- **[Troubleshooting](troubleshooting.md)** — if the smoke run failed.
