# Features and limitations

## What the package provides

- **Margin-defined tasks.** A task is a plain mjlab environment plus a margin
  function `(g, l)` and a one-line `TaskSpec`. The sign of the margins *is* the
  specification — no reward shaping. See [margins](../concepts/margins.md).
- **Three task modes.** `safety` (avoid-only), `reach-avoid` (adds a target
  margin), and `cumulative` (ordinary dense-reward RL — the task policy a filter
  wraps).
- **Single-player and adversarial training.** Any task that declares
  `supports_adversary` can be trained as a zero-sum two-player game with a learned
  worst-case disturbance (`--adversary`).
- **PPO and SAC families.** One trainer routes to on-policy (PPO) or off-policy
  (SAC); the exact learner is resolved by the [MAP](../concepts/map.md).
- **Composable safety filters.** A safety filter is a composition of a fallback
  policy, a monitor, and an intervention rule — five named compositions ship. See
  [safety filters](../concepts/filters.md).
- **One evaluation harness.** `examples/eval.py` chooses environment, task policy,
  safety policy + filter, and attack independently and composes them. See
  [evaluation](../guide/evaluation.md).
- **Filtered training (PORL).** Wrap the *training* environment in a safety filter
  so a task policy learns while a fallback keeps it (nearly) failure-free. See
  [train inside a filter](../safety-filter-training.md).
- **GPU-parallel environments.** Thousands of environments step end-to-end on the
  GPU via the mjlab / MuJoCo-Warp stack.
- **Reproducible recipes.** Runs are small YAML configs; every run dumps its
  fully resolved config for exact reproduction.

## Project status and maturity

This is a research codebase released as a package. Task maturity varies widely,
and `list_tasks()` deliberately exposes internal and experimental variants
alongside the primary public tasks — read the maturity column in the
[environments catalog](../environments/index.md) before choosing a task.

| Maturity | Meaning | Examples |
|---|---|---|
| **Tutorial** | Small, fully documented entry point | `car_goal` |
| **Stable** | Reproduced, recommended | `go2_stabilize`, `go2_walker_flat` |
| **Research** | Shipped result, needs staged training / real scale | `go2_gap_*` family |
| **Experimental** | Shelved or in progress; may not reproduce | `go2_crawl_*`, `go2_low_bar_*`, `go2_tunnel_*` |
| **Requires custom mjlab fork** | Imports on stock mjlab, needs a fork to simulate | `digit_*` |

## Known limitations

- **Linux + NVIDIA only.** There is no CPU or non-NVIDIA path for the GPU sim.
  See [requirements](../getting-started/requirements.md).
- **No PyPI release.** Install from a git checkout; the `safety_sb3` dependency is
  a pinned git tag. See [installation](../installation.md).
- **Pinned sim stack.** mjlab 1.2.0 with MuJoCo 3.6.0 / mujoco-warp 3.6.0 — an
  unpinned install breaks at the first environment build. Newer mjlab (1.3+) is
  not validated.
- **Checkpoints are version-specific.** A saved `.zip` stores the learner class
  path, so models trained on ≤ v0.3.x do not load on v0.4.0 (retrain or pin the
  old stack). See the [release notes](../release-notes.md).
- **Evaluation is not bit-reproducible.** mjlab keeps observation noise on in play
  mode and MuJoCo-Warp is not bit-deterministic — compare configurations over
  several runs, not once.
- **Digit simulation needs a fork.** The vendored Digit assets and cfgs import
  cleanly, but stepping the sim needs the lab's mjlab entity-patch fork.
- **Two-player algorithm choice matters.** `*PPO2P` and `*SAC2P` are *different
  algorithms*, not one game with two optimizers — see [the MAP](../concepts/map.md).

For the per-release breaking changes and the upgrade steps, see the
[release notes](../release-notes.md).
