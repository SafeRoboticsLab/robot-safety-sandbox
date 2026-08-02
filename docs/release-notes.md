# Release notes

## v0.4.0 — MAP alignment, composable filters, one eval harness (BREAKING)

v0.4.0 tracks **safety_sb3 v0.4.0** (the Mode·Algorithm·Players rename) and, on
top of it, refactors the zoo's three seams: how a task names its learner, how a
filter is composed, and how evaluation is run. It is a breaking release —
checkpoints and CLI flags change — with no compatibility shims.

**Dependency bump.** This release requires `safety_sb3 @ v0.4.0`; it will not work
against v0.3.x (the learner class names it resolves to no longer exist there). To
stay on the old stack, pin both:

```
robot_safety_sandbox @ ...@v0.3.0   # with  safety_sb3 @ ...@v0.3.0
```

### 1. Checkpoints do not carry over

safety_sb3 renamed every learner class, and a saved `.zip` stores the class path,
so **runs trained on ≤ v0.3.x will not load on v0.4.0** (`ModuleNotFoundError` /
`AttributeError` at unpickle). There is no migration shim — retrain on v0.4.0, or
pin v0.3.0 to load an old model. This also means the `V ≥ 0` certificate from an
old SAC twin is unsound anyway (see safety_sb3 v0.4.0 §7, the entropy-bonus fix),
so retraining is the right move regardless.

### 2. The registry resolves the learner by the MAP formula

A task no longer names a learner. It declares its **`mode`** (via its margins) and
whether it `supports_adversary`; the trainer picks the exact learner with a
formula, not a lookup:

```python
algo_name(task_id, adversary=False, family="on_policy") -> "ReachAvoidSAC2P"  # e.g.
```

`train.py --config … --family {on_policy|off_policy}` selects one MAP cell. The
registry re-declares the mode strings as literals (pinned by a test) and returns
**names only**, so the zoo never imports `safety_sb3`. A task written once trains
under any algorithm/player combination the MAP allows. See the
[API guide](API.md) and the safety_sb3 [MAP convention](https://saferoboticslab.github.io/safety-stable-baselines/map/).

### 3. One `mode` axis for tasks (`nominal/` is gone)

The old `nominal/` package and the `kind=` distinction are collapsed into
`tasks/`, with a single `mode` axis: **`safety`** (a `g` margin only),
**`reach-avoid`** (adds a target margin `l`), **`cumulative`** (ordinary reward,
stock SB3). A cumulative task is a first-class citizen — it is what filters wrap.
The rollout reduction is derived from the task's mode, not passed in.

### 4. Filters are composed, not monolithic

The value/CBF filter was decomposed into three swappable parts —
**fallback · monitor · intervention** — behind one composition API. Five named
compositions ship; `eval.py --filter {…}` picks one. Two intervention specifics:

- The **canonical least-restrictive** switch is separated from the **smoothed
  variant** (`--smoothing`: latch + median + release hysteresis). The smoothed
  variant is *not* Def-2 valid step-by-step and must be reported as a smoothed
  variant of the named filter — it exists to reproduce pre-2026-07-25 gauntlets.
- A **rollout ("Gameplay") monitor** runs on an mjlab **shadow sim** seeded with
  the live env's domain randomization.

### 5. Evaluation: one harness, de-jargoned CLI

The four eval scripts are consolidated into a single modular `examples/eval.py`.
The CLI drops the "twin"/"nominal" jargon:

| v0.3.x flag | v0.4.0 flag |
|---|---|
| `--twin` | `--safety-policy` |
| `--nominal` | `--task-policy` |
| `--nominal-from-twin` | `--safety-only` |
| `--nominal-obs-key` | `--task-obs-key` |
| `--dstb-twin` | `--adversary-policy` |

The results summary field `twin` is likewise renamed `safety_policy`. The harness
also records trajectories and distance traveled, and skips the training league when
loading a twin for inference.

### 6. Training a policy inside a safety filter

New capability: wrap the **training** env in a safety filter so a task policy
learns while a fallback controller keeps it (nearly) failure-free — the setting
from *Provably Optimal RL under Safety Filtering* (Oh, Nguyen, Hu, Fisac;
arXiv:2510.18082). It is **config-driven** — a `safety_filter:` block in the
training YAML, no dedicated flags — and failure counting is now always-on in the
base env so the filtered and unfiltered arms share one accounting path. Full
walkthrough: [Training inside a safety filter](safety-filter-training.md).

The showcase task was renamed **`go2_walker_porl` → `go2_walker_filtered`** (module
`go2_porl.py` → `go2_filtered.py`); the "porl" name is dropped from all task ids
and flags, and the paper is cited in the docs instead.

### 7. Assets consolidated

The vendored robot MJCF/meshes now live under a single `envs/assets/` tree.

### New in v0.4.0

- **`car_goal`** — a differential-drive reach-avoid task and the new-user
  [tutorial](tutorial-car-goal.md): drive to a goal disk while avoiding two
  obstacle cylinders. The mjlab-zoo analog of safety_sb3's `bicycle5d`.
- The [API guide](API.md), [Extending](EXTENDING.md), and [Porting a task](porting.md)
  pages, all rewritten around the one-`mode` task model and the MAP formula.

### Migrating

1. Bump both pins to `@v0.4.0`.
2. Retrain any policy you need (checkpoints do not carry over).
3. Update eval commands to the new flag names (table in §5).
4. Replace any `--porl*` flags with a `safety_filter:` config block.
5. Update dashboards: two-player log keys and league dirs changed upstream
   (safety_sb3 v0.4.0 §4).
