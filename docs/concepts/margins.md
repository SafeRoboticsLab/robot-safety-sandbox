# Safety and reach-avoid margins

A task in this sandbox is defined by its **margins**, not by a hand-tuned reward.
The sign of a margin *is* the specification.

## The `g` / `l` contract

A task hands the learner two margins per step (shared with `safety_sb3`):

| symbol | meaning | rides on |
|---|---|---|
| `g(s)` | **safety margin** — `g ≥ 0` ⟺ outside the failure set | the reward channel — **never normalized** |
| `l(s)` | **target margin** — `l ≥ 0` ⟺ inside the target set (zeros for avoid-only) | `info["l_x"]` (numpy) / `step_tensor`'s 5th return (tensor) |

The failure set is *defined* as `{s : g(s) < 0}`; the target set as
`{s : l(s) ≥ 0}`. `min` = AND, `max` = OR. Normalize every margin term to `O(1)`
and clamp to `±CLAMP` (see `robot_safety_sandbox/margins.py`). The environment
**terminates on `g < 0`** — enforced by `end_criterion` (see
[termination](termination.md)).

!!! warning "Never normalize or reshape `g`"
    `g` rides on the reward channel but it is **not** a reward to shape — it is the
    Bellman target that carries the specification. `safety_sb3` refuses reward
    normalization. Normalize *observations* only.

## The reach-avoid value

The learner approximates the reach-avoid value function (a Hamilton–Jacobi
reachability value, learned by RL) with the safety Bellman backup

$$
V(s) \;=\; (1-\gamma)\,\min\!\big(l,\,g\big)\;+\;\gamma\,\min\!\Big(g,\;\max\big(l,\,V(s')\big)\Big).
$$

In words: **you can only ever be as safe as your worst future `g`** (the outer
`min(g, ·)` — one collision anywhere on the trajectory condemns the whole path),
and along a safe path you want to eventually **reach** (`max(l, V′)` lets a future
arrival raise today's value). `V(s) ≥ 0` certifies a control-invariant reach-avoid
state: from here the policy can reach the goal without ever entering the failure
set.

## Composing a margin function

Compose a `margin_fn` from a `g` term and an optional `l` term:

```python
from robot_safety_sandbox.margins import compose, avoid_only

compose(g_fn, l_fn)      # reach-avoid task: has_target=True
compose(g_fn)            # avoid-only task:  l is a zero placeholder, has_target=False
avoid_only(margin_fn)    # strip the target off an existing (g, l) builder
```

Available terms (see `robot_safety_sandbox/margins.py` for the full list):
`g_terrain_relative`, reach terms `l_rest` / `l_gap_foothold` / `l_launch_basin`,
and per-robot terms under `robot_safety_sandbox/envs/*/margins.py`.

## Why normalize and clamp

The margins *are* the Bellman targets. If `g` ranged over meters and `l` over
centimeters, the critic would see wildly different scales and the `min`/`max`
backup would be dominated by whichever term is numerically larger — not the one
physically closer to its boundary. Dividing by an `O(1)` scale and clamping keeps
both terms comparable and the value bounded, which is what keeps the safety
Bellman iteration stable. **The sign carries the specification; the scale must not
distort it.**

## Avoid is not a reach-avoid instance

Do **not** emulate an avoid task by pinning `l` to a constant (`l_neg`/`l_zero`,
both removed). It cannot work: a negative constant empties the safe set, a
non-negative one strips the lookahead. An avoid-only task declares no `l`
(`compose(g_fn)`) and declares `mode="safety"`, so the [MAP](map.md) resolves it
to a `Safety*` learner, which ignores `l`. See the safety_sb3 API §5 for the
proof, and `margins.py` for the in-code note.

## See also

- [Termination](termination.md) — when the episode ends (`end_criterion`).
- [The task registry](../reference/registry.md) — how a `margin_fn` is wired into
  a `TaskSpec`.
- [Code reference](../reference.md#margins) — the generated `margins.py` docs.
- The [car-goal tutorial](../tutorial-car-goal.md) — margins built from scratch.
