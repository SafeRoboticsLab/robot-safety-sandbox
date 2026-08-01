# The MAP naming convention

> **Here's a MAP to navigate the codebase — Mode. Algorithm. Players.**

    M = Mode       Safety | ReachAvoid | Cumulative    the Bellman operator
    A = Algorithm  PPO | SAC | A2C | DQN               the RL update rule
    P = Players    1P | 2P                             single-player | zero-sum

Every learner's name is those three letters concatenated in that order —
`SafetyPPO1P`, `ReachAvoidSAC2P` — and each letter comes from exactly one place:

| letter | source | how it is set |
|---|---|---|
| **M** | the TASK | `TaskSpec(mode=...)` — a property of its margins |
| **A** | the RUN | `train.py --family on_policy` (PPO) / `off_policy` (SAC) |
| **P** | the RUN | `--adversary` |

You never pick a learner directly. `algo_name(task_id, adversary, family)` is a
**formula, not a lookup**:

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
stay decoupled and a cumulative-only install still imports the registry.

## `*PPO2P` and `*SAC2P` are not interchangeable

Same MAP cell, **different algorithm**. Read this before choosing `--adversary`:

| | `*SAC2P` (off_policy) | `*PPO2P` (on_policy) |
|---|---|---|
| critic | **one shared joint-action critic** `Q(s, [a_ctrl, a_dstb])` | **two independent** `V(s)` nets |
| game | minimax on that single critic — both players read the same value | alternating **best-response approximation** |
| data | one replay buffer | two rollout buffers |
| control flow | a single update | a ctrl/dstb **phase machine** (`ctrl_rollouts_per_cycle`, `dstb_rollouts_per_cycle`, `dstb_pretrain_rollouts`) |

The shared-critic form is the closer approximation of the zero-sum value; the PPO
form trades that for on-policy stability and needs its phase schedule tuned. The
E042 result on `go2_stabilize` (best-ever on that task) is `ReachAvoidSAC2P`.

## See also

- [Margins](margins.md) — what the **M** (mode) actually is.
- [Training](../guide/training.md) — how **A** (`--family`) and **P**
  (`--adversary`) are set at run time.
- [The task registry](../reference/registry.md) — `algo_name`, `spec`,
  `list_tasks`.
- safety_sb3 [MAP convention](https://saferoboticslab.github.io/safety-stable-baselines/map/)
  — the same naming law on the algorithm side.
