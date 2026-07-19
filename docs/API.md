# robot-safety-sandbox — API reference

The canonical contract for the **environment layer**: tasks, margins, the
registry, and the two bridges to `safety_sb3`. The algorithm layer (learners,
backups, `terminal_type`) is documented in
[safety-stable-baselines `docs/API.md`](https://github.com/SafeRoboticsLab/safety-stable-baselines/blob/main/docs/API.md),
which is the source of truth for the shared `g`/`l` channel contract restated in §1.

- Orientation: [Home](index.md)
- Adding a task / robot: [Extending](EXTENDING.md), [Porting a task](porting.md)
- Install & pins: [Installation](installation.md)

---

## 1. The `g`/`l` contract (shared with safety_sb3)

A task hands the learner two margins per step:

| symbol | meaning | rides on |
|---|---|---|
| `g(s)` | **safety margin**; `g ≥ 0` ⟺ outside the failure set | the reward channel — **never normalized** |
| `l(s)` | **target margin**; `l ≥ 0` ⟺ inside the target set (zeros for avoid-only) | `info["l_x"]` (numpy) / `step_tensor`'s 5th return (tensor) |

`min` = AND, `max` = OR; normalize every margin term to O(1) and clamp to
`±CLAMP` (see `margins.py`). The env **terminates on `g < 0`** — enforced by
`end_criterion` (§4).

---

## 2. A task = cfg_builder + margin_fn + TaskSpec

```
TaskSpec(
    task_id,
    cfg_builder,          # (play: bool) -> ManagerBasedRlEnvCfg   (plain mjlab)
    margin_fn,            # (env) -> (g, l) batched tensors          (None for nominal)
    default_algo="SafetyPPO",
    end_criterion="failure",         # when the episode ends (§4)
    warmstart_from=None,             # previous pipeline-stage task_id
    supports_adversary=False,        # can this task take a --adversary run?
    ctrl_dim=12, dstb_dim=3,
    kind="safety",                   # "safety" (margins) | "nominal" (dense task)
    description="",
)
```

- **`cfg_builder`** is plain mjlab — terrain, spawn events, curricula, terminations.
  Algorithm-agnostic.
- **`margin_fn`** composes from `margins.py`. For an **avoid-only** task pass
  `compose(g_fn)` (no `l`) — see §5. It carries `has_target = (l_fn is not None)`.
- **`default_algo`** picks the **column** (avoid vs reach-avoid) — this is the task's
  *problem*. The **row** (single- vs two-player) is a property of the *run*
  (`--adversary`), resolved by `algo_name()` (§3).

Register once, and both bridges work:

```python
from robot_safety_sandbox import register, TaskSpec
register(TaskSpec(task_id="my_task", cfg_builder=..., margin_fn=compose(g, l)))
```

---

## 3. Registry API

```python
from robot_safety_sandbox import (
    make_tensor, make_numpy, list_tasks, spec, register, algo_name, TaskSpec,
)

list_tasks(kind=None) -> list[str]        # kind: "safety" | "nominal" | None (all)
spec(task_id) -> TaskSpec
register(TaskSpec) -> None
algo_name(task_id, adversary=False) -> str   # the learner CLASS NAME to use

make_tensor(task_id, num_envs=2048, device="cuda:0", adversary=False, **kw)  # GPU, PPO family
make_numpy (task_id, num_envs=64,   device="cuda:0", adversary=False, **kw)  # SB3 VecEnv, SAC family
```

`algo_name` resolves both axes and is the one place the 2×2 is kept honest — the
task's margins fix the *problem*, `--adversary` fixes the *player count*:

| task problem | 1-player | 2-player (`adversary=True`) |
|---|---|---|
| avoid (`default_algo=SafetyPPO`) | `SafetyPPO` | `IsaacsPPO` |
| reach-avoid (`default_algo=ReachAvoidPPO`) | `ReachAvoidPPO` | `GameplayPPO` |

It **refuses** a reach-avoid learner on an avoid-only task (no target set) — the
guard against the retired `l_neg` pattern. The registry never imports `safety_sb3`;
`algo_name` returns names only, so the two layers stay decoupled.

> `Isaacs*` = two-player **avoid** (ISAACS eq. 7); `Gameplay*` = two-player
> **reach-avoid** (Gameplay Filters). These names changed meaning in safety_sb3
> v0.2.0 — see its RELEASE_NOTES.

---

## 4. `end_criterion` — when the episode ends

A `TaskSpec` field (and a `--end-criterion` override in `examples/train.py`),
one of:

| `end_criterion` | terminates when | use |
|---|---|---|
| `"failure"` (default) | `g < 0` (+ timeout). **Never on reach.** | reach *deeper* — the agent keeps going after reaching, so the reach-avoid value climbs with `l` up to the `g` ceiling |
| `"reach-avoid"` | `g < 0` **or** (`g ≥ 0` **and** `l ≥ 0`) | reach and stop — the episode ends at the target boundary |
| `"timeout"` | only the env timeout | diagnostic / pure value-learning |

This is the **environment half** of a pairing whose algorithm half is the
learner's `terminal_type` (safety_sb3 §4). They are orthogonal; all pairings are
constructible. The pairing that learns to reach deeper into the target is
`end_criterion="failure"` + `terminal_type="all"`.

Implemented as a mjlab `DoneTerm` (`zoo_reach_success`, fires on `g ≥ 0 ∧ l ≥ 0`)
added only in `reach-avoid` mode — a real termination term, so mjlab auto-resets
on the same step rather than one step late. Default `"failure"` adds no term.

Defaults reproduce prior behavior exactly: an audit of all 20 safety tasks found
**none currently terminates on success**, so every task stays `"failure"` and is
bit-identical. Switch a task to reach-and-stop by setting `end_criterion="reach-avoid"`
on its `TaskSpec`, or per-run with `--end-criterion`.

---

## 5. margins.py

Compose a `margin_fn` from a `g` term and an optional `l` term:

```python
from robot_safety_sandbox.margins import compose, avoid_only

compose(g_fn, l_fn)      # reach-avoid task: has_target=True
compose(g_fn)            # avoid-only task:  l is a zero placeholder, has_target=False
avoid_only(margin_fn)    # strip the target off an existing (g, l) builder
```

**Avoid is not a reach-avoid instance** — do not emulate an avoid task by pinning
`l` to a constant (`l_neg`/`l_zero`, both removed). It cannot work: a negative
constant empties the safe set, a non-negative one strips the lookahead. An
avoid-only task declares no `l` (`compose(g_fn)`) and runs on an avoid learner,
which ignores `l`. See safety_sb3 API §5 for the proof, and `margins.py` for the
in-code note.

Available terms (see `margins.py` for the full list): `g_terrain_relative`,
reach terms `l_rest` / `l_gap_foothold` / `l_launch_basin`, and per-robot terms
under `envs/*/margins.py`.

---

## 6. The bridges

`MjlabTensorSafetyEnv` (GPU, primary) implements the tensor path:

```
step_tensor(actions) -> (obs, reward_g, dones, timeouts, l_x)     # all device tensors
```

`MjlabNumpySafetyEnv` implements the classic SB3 `VecEnv` path (`g` on reward,
`l` on `info["l_x"]`) for the SAC family and stock SB3 tooling.

`metrics()` forwards curriculum levels and task metrics to the logger every
rollout — watch the `env/Curriculum/*` keys; a stalled curriculum looks exactly
like converged training in the reward curve.

---

## 7. Training

Two entrypoints, split by algorithm family (both resolve the learner from
*(task margins × `--adversary`)*):

- **`examples/train.py`** — the **PPO** family (SafetyPPO / ReachAvoidPPO /
  IsaacsPPO / GameplayPPO).
- **`examples/train_sac.py`** — the **SAC** family (SafetySAC / ReachAvoidSAC /
  IsaacsSAC / GameplaySAC), the off-policy analog. Same `--task` / `--adversary`
  resolution; two-player-only knobs (`ctrl_action_dim`, leaderboard, per-agent
  LRs, adversary force ramp) are gated behind `--adversary`.

```bash
# PPO family
python examples/train.py     --task go2_gap_chain --terminal-type all      # reach-avoid PPO
python examples/train.py     --task digit_stabilize_avoid --adversary      # two-player avoid (IsaacsPPO)
# SAC family
python examples/train_sac.py --task go2_stabilize                          # 1-player reach-avoid (ReachAvoidSAC)
python examples/train_sac.py --task go2_stabilize --adversary --num-envs 1024   # 2-player reach-avoid (GameplaySAC)
python examples/train_sac.py --task digit_stabilize_avoid --adversary      # two-player avoid (IsaacsSAC)
```

- `--terminal-type {all,g}` — forwarded to reach-avoid learners; ignored (with a
  notice) on avoid tasks.
- `--end-criterion {failure,reach-avoid,timeout}` — overrides the task's default
  for this run.
- `--adversary` — two-player run; the learner is resolved by `algo_name`.
- `--config <file.yaml>` — a reusable **recipe** (keys = flag names) that sets
  defaults; explicit CLI flags still override it (`argparse defaults < config <
  CLI`). Every run also dumps its fully-resolved config to `<outdir>/config.yaml`
  (re-run with `--config <that file>` to reproduce). Recipes live in `configs/`.

```bash
python examples/train_sac.py --config configs/go2_stabilize_gameplaysac.yaml         # the E042 recipe
python examples/train_sac.py --config configs/go2_stabilize_gameplaysac.yaml --seed 3  # override one knob
```
- **Env/task overrides** — a config `env_overrides:` dict (or `--env-override KEY=VAL`,
  repeatable) forwards params to the task's `cfg_builder`, overriding values baked into
  its registration (e.g. `gate_close_rate`, `bar_clearance`) — so a recipe can define the
  *environment* too, no argparse edit. An unaccepted key fails loud.

`train_sac.py` exposes the reference-faithful controls (see safety_sb3
[hyperparameters](https://saferoboticslab.github.io/safety-stable-baselines/hyperparameters/)):
`--gamma-schedule` (discount anneal, default the discrete-jump schedule),
`--min-alpha`, per-agent `--critic-lr/--dstb-lr/--ent-coef-lr/--dstb-ent-coef-lr`,
`--eval-rollouts` (safe/success-rate to wandb), and **throughput** leaderboard
defaults (`--leaderboard-freq 2_000_000 --leaderboard-episodes 3` + an on-device
tensor eval env — ~30× over the old settings at 1024 envs).

PPO recipe that works (hard-won): `normalize_obs=True` (obs only), `ent_coef=1e-4`,
`log_std_init=ln(0.3)`, `adaptive_lr=True`, `n_steps=48`. See README.
