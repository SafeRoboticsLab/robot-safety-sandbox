# Task registry and Python API

The stable public surface for defining and building tasks. For the generated
docstrings, see the [code reference](../reference.md); for the conceptual contract,
see [margins](../concepts/margins.md) and [the MAP](../concepts/map.md).

## A task = cfg_builder + margin_fn + TaskSpec

```python
TaskSpec(
    task_id,
    cfg_builder,          # (play: bool) -> ManagerBasedRlEnvCfg   (plain mjlab)
    margin_fn,            # (env) -> (g, l) batched tensors      (None for cumulative)
    mode,                            # REQUIRED: which BACKUP values it
    end_criterion="failure",         # when the episode ends
    supports_adversary=False,        # can this task take a --adversary run?
    ctrl_dim=12, dstb_dim=3,
    description="",
)
```

- **`cfg_builder`** is plain mjlab — terrain, spawn events, curricula,
  terminations. Algorithm-agnostic.
- **`margin_fn`** composes from `margins.py`. For an avoid-only task pass
  `compose(g_fn)` (no `l`); it carries `has_target = (l_fn is not None)`.
- **`mode`** is the task's single axis and the only thing it says about the
  learner (the [MAP](../concepts/map.md)'s **M**): the `safety_sb3.backups` mode
  it is trained under.

| `mode` | backup | margins | learner |
|---|---|---|---|
| `"safety"` | `V = min(g, γV′)` | `margin_fn` required, no `l` | `Safety*` |
| `"reach-avoid"` | `V = min(g, max(l, γV′))` | `margin_fn` required, real `l` | `ReachAvoid*` |
| `"cumulative"` | `V = r + γ(1−d)V′` | none (`margin_fn=None`) | stock `PPO` / `SAC` |

`mode="cumulative"` is plain reward-maximizing RL — the task policy `π_task` a
safety filter wraps. Its envs are auto-built in dense-reward mode and it trains
with **stock SB3**, keeping the checkpoint a vanilla SB3 zip that loads without
`safety_sb3`. There is no `default_algo` and no `warmstart_from` field: the learner
name is computed (see [the MAP](../concepts/map.md)); warm-start lineage is a
run-level `--load` choice.

```python
from robot_safety_sandbox import register, TaskSpec
from robot_safety_sandbox.margins import compose
register(TaskSpec(task_id="my_task", cfg_builder=..., margin_fn=compose(g, l),
                  mode="reach-avoid"))
```

## Registry functions

```python
from robot_safety_sandbox import (
    make_tensor, make_numpy, list_tasks, spec, register, algo_name, TaskSpec,
)

list_tasks(mode=None) -> list[str]        # mode: one of the modes, or None (all)
spec(task_id) -> TaskSpec
register(TaskSpec) -> None
algo_name(task_id, adversary=False, family="on_policy") -> str   # the MAP formula
```

## The bridges

Register once, and both bridges build the task:

```python
make_tensor(task_id, num_envs=2048, device="cuda:0", adversary=False, **kw)  # GPU, PPO family
make_numpy (task_id, num_envs=64,   device="cuda:0", adversary=False, **kw)  # SB3 VecEnv, SAC family
```

- **`MjlabTensorSafetyEnv`** (GPU, primary) implements the tensor path:
  `step_tensor(actions) -> (obs, reward_g, dones, timeouts, l_x)` — all device
  tensors.
- **`MjlabNumpySafetyEnv`** implements the classic SB3 `VecEnv` path (`g` on
  reward, `l` on `info["l_x"]`) for the SAC family and stock SB3 tooling.
- `metrics()` forwards curriculum levels and task metrics to the logger every
  rollout — watch the `env/Curriculum/*` keys; a stalled curriculum looks exactly
  like converged training in the reward curve.

## Configuration recipes

Training recipes live in `configs/` — YAML files whose keys are the trainer's
argparse flag names, plus a reserved `family:` key. Precedence is
**argparse defaults < `--config` file < explicit CLI flags**. A reserved
`env_overrides:` dict forwards params to the task's `cfg_builder`. Every run
writes its fully resolved config to `<outdir>/config.yaml` for exact reproduction.
See [training](../guide/training.md#config-recipes-recommended) for the shipped
recipe list.

## See also

- [Code reference](../reference.md) — the generated docstrings for the registry,
  margins, and bridges.
- [Environments catalog](../environments/index.md) — every registered task with
  its mode and maturity.
- [Extending the sandbox](../EXTENDING.md) — worked examples for adding a task or
  robot.
