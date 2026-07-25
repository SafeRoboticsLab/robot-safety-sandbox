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
    margin_fn,            # (env) -> (g, l) batched tensors     (None for cumulative)
    mode,                            # REQUIRED: which BACKUP values it (below)
    end_criterion="failure",         # when the episode ends (§4)
    supports_adversary=False,        # can this task take a --adversary run?
    ctrl_dim=12, dstb_dim=3,
    description="",
)
```

- **`cfg_builder`** is plain mjlab — terrain, spawn events, curricula, terminations.
  Algorithm-agnostic.
- **`margin_fn`** composes from `margins.py`. For an **avoid-only** task pass
  `compose(g_fn)` (no `l`) — see §5. It carries `has_target = (l_fn is not None)`.
- **`mode`** is the task's single axis and the **only** thing it says about the
  learner: the `safety_sb3.backups` mode it is trained under. It is the MAP's
  **M** (§3); the **A** (algorithm) and **P** (players) belong to the *run*.

| `mode` | backup | margins | 1P learner | 2P learner |
|---|---|---|---|---|
| `"safety"` | `V = min(g, γV′)` | `margin_fn` required, no `l` | `SafetyPPO1P` / `SafetySAC1P` | `SafetyPPO2P` / `SafetySAC2P` |
| `"reach-avoid"` | `V = min(g, max(l, γV′))` | `margin_fn` required, real `l` | `ReachAvoidPPO1P` / `ReachAvoidSAC1P` | `ReachAvoidPPO2P` / `ReachAvoidSAC2P` |
| `"cumulative"` | `V = r + γ(1−d)V′` | **none** (`margin_fn=None`) | **stock** `PPO` / `SAC` | — |

`mode="cumulative"` is plain reward-maximizing RL — the task policy `π_task` a
safety filter wraps. Its envs are auto-built in **dense-reward** mode (the env's
own reward stack instead of `g`), and it trains with **stock SB3**, keeping the
checkpoint a vanilla SB3 zip that loads without `safety_sb3`.

There is no `default_algo` field and no other way to pin a learner: the name is
computed (§3). There is no `warmstart_from` either — a pipeline's warm-start
lineage is a run-level `--load` choice, recorded in the experiment log.

Register once, and both bridges work:

```python
from robot_safety_sandbox import register, TaskSpec
register(TaskSpec(task_id="my_task", cfg_builder=..., margin_fn=compose(g, l),
                  mode="reach-avoid"))
```

---

## 3. Registry API — and the MAP

> **Here's a MAP to navigate the codebase — Mode. Algorithm. Players.**

    M = Mode       Safety | ReachAvoid | Cumulative    the Bellman operator
    A = Algorithm  PPO | SAC | A2C | DQN               the RL update rule
    P = Players    1P | 2P                             single-player | zero-sum

A learner's name is those three letters concatenated in that order, and each
letter comes from exactly one place:

| letter | source | how it is set |
|---|---|---|
| **M** | the TASK | `TaskSpec(mode=...)` — a property of its margins |
| **A** | the RUN | `train.py --family on_policy` (PPO) / `off_policy` (SAC) |
| **P** | the RUN | `--adversary` |

```python
from robot_safety_sandbox import (
    make_tensor, make_numpy, list_tasks, spec, register, algo_name, TaskSpec,
)

list_tasks(mode=None) -> list[str]        # mode: one of MODES, or None (all)
spec(task_id) -> TaskSpec
register(TaskSpec) -> None
algo_name(task_id, adversary=False, family="on_policy") -> str

make_tensor(task_id, num_envs=2048, device="cuda:0", adversary=False, **kw)  # GPU, PPO family
make_numpy (task_id, num_envs=64,   device="cuda:0", adversary=False, **kw)  # SB3 VecEnv, SAC family
```

`algo_name` is a **formula, not a lookup** — the whole body is

```python
f"{_PREFIX[spec(task_id).mode]}{_ALG[family]}{'2P' if adversary else '1P'}"
```

so nothing in the registry can override it:

| task `mode` | `family="on_policy"` 1P / 2P | `family="off_policy"` 1P / 2P |
|---|---|---|
| `"safety"` (avoid) | `SafetyPPO1P` / `SafetyPPO2P` | `SafetySAC1P` / `SafetySAC2P` |
| `"reach-avoid"` | `ReachAvoidPPO1P` / `ReachAvoidPPO2P` | `ReachAvoidSAC1P` / `ReachAvoidSAC2P` |
| `"cumulative"` | `PPO` (stock SB3) / — | `SAC` (stock SB3) / — |

Cumulative has no **P**: there is no two-player cumulative game, and `algo_name`
raises rather than inventing one. It also **refuses** a reach-avoid learner on an
avoid-only task (no target set) — the guard against the retired `l_neg` pattern.
The registry never imports `safety_sb3` (it re-declares the mode strings as
literals, pinned by a test): `algo_name` returns names only, so the two layers
stay decoupled and a cumulative-only install — dense reward + stock SB3, no
`safety_sb3` — still imports the registry.

### `*PPO2P` and `*SAC2P` are not interchangeable

Same MAP cell, different algorithm. Read this before choosing `--adversary`:

| | `*SAC2P` (off_policy) | `*PPO2P` (on_policy) |
|---|---|---|
| critic | **one shared joint-action critic** `Q(s, [a_ctrl, a_dstb])` | **two independent** `V(s)` nets |
| game | minimax on that single critic — both players read the same value | alternating **best-response approximation** |
| data | one replay buffer | two rollout buffers |
| control flow | a single update | a ctrl/dstb **phase machine** (`ctrl_rollouts_per_cycle`, `dstb_rollouts_per_cycle`, `dstb_pretrain_rollouts`) |

The shared-critic form is the closer approximation of the zero-sum value; the
PPO form trades that for on-policy stability and needs its phase schedule tuned.
The E042 result on `go2_stabilize` (best-ever on that task) is `ReachAvoidSAC2P`.

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
avoid-only task declares no `l` (`compose(g_fn)`) and declares `mode="safety"`,
so the MAP resolves it to a `Safety*` learner, which ignores `l`. See safety_sb3
API §5 for the proof, and `margins.py` for the in-code note.

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

One router, two peer families. `examples/train.py` dispatches to the on-policy or
off-policy trainer by a **required `--family`** — neither is the "main" one; they
are the MAP's two wired-up **A** values, and both resolve the learner the same
way — the task's `mode` × `--adversary`. Note the 2P cells differ structurally
between the families (see §3):

| `--family` | trainer | the MAP's **A** | learners |
|---|---|---|---|
| `on_policy` (alias `ppo`) | `examples/train_on_policy.py` | `PPO` | `{Safety,ReachAvoid}PPO{1P,2P}` |
| `off_policy` (alias `sac`) | `examples/train_off_policy.py` | `SAC` | `{Safety,ReachAvoid}SAC{1P,2P}` |

`train.py` forwards every other flag verbatim to the chosen trainer (run
`--family <f> --help` to see its options). The two trainers are also directly
runnable — neither is subordinate. Set the family on the CLI or via `family:` in
a `--config` YAML.

```bash
# on-policy (PPO family)
python examples/train.py --family on_policy  --task go2_gap_chain --terminal-type all    # ReachAvoidPPO1P
python examples/train.py --family ppo        --task digit_stabilize_avoid --adversary    # SafetyPPO2P
# off-policy (SAC family)
python examples/train.py --family off_policy --task go2_stabilize                        # ReachAvoidSAC1P
python examples/train.py --family sac        --task go2_stabilize --adversary --num-envs 1024  # ReachAvoidSAC2P
python examples/train.py --family sac        --task digit_stabilize_avoid --adversary    # SafetySAC2P
```

- `--terminal-type {all,g}` — forwarded to reach-avoid learners; ignored (with a
  notice) on avoid tasks.
- `--end-criterion {failure,reach-avoid,timeout}` — overrides the task's default
  for this run.
- `--adversary` — two-player run; the learner is resolved by `algo_name`.
- `--config <file.yaml>` — a reusable **recipe** (keys = flag names) that sets
  defaults; explicit CLI flags still override it (`argparse defaults < config <
  CLI`). A recipe carries its own `family:` key, so `--family` isn't needed on the
  CLI. Every run also dumps its fully-resolved config to `<outdir>/config.yaml`
  (re-run with `--config <that file>` to reproduce). Recipes live in `configs/`.

```bash
python examples/train.py --config configs/go2_stabilize_reachavoidsac2p.yaml          # the E042 recipe (family: off_policy)
python examples/train.py --config configs/go2_stabilize_reachavoidsac2p.yaml --seed 3 # override one knob
```
- **Env/task overrides** — a config `env_overrides:` dict (or `--env-override KEY=VAL`,
  repeatable) forwards params to the task's `cfg_builder`, overriding values baked into
  its registration (e.g. `gate_close_rate`, `bar_clearance`) — so a recipe can define the
  *environment* too, no argparse edit. An unaccepted key fails loud.

The off-policy trainer exposes the reference-faithful controls (see safety_sb3
[hyperparameters](https://saferoboticslab.github.io/safety-stable-baselines/hyperparameters/)):
`--gamma-schedule` (discount anneal, default the discrete-jump schedule),
`--min-alpha`, per-agent `--critic-lr/--dstb-lr/--ent-coef-lr/--dstb-ent-coef-lr`,
`--eval-rollouts` (safe/success-rate to wandb), and **throughput** leaderboard
defaults (`--leaderboard-freq 2_000_000 --leaderboard-episodes 3` + an on-device
tensor eval env — ~30× over the old settings at 1024 envs).

PPO recipe that works (hard-won): `normalize_obs=True` (obs only), `ent_coef=1e-4`,
`log_std_init=ln(0.3)`, `adaptive_lr=True`, `n_steps=48`. See README.
