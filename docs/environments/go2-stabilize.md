# Go2 stabilize and locomotion

A Unitree Go2 quadruped must **stay upright** — and, in the locomotion variant,
**track a velocity command** — against a worst-case base force. It is the
simplest and most-reproduced safety benchmark in the repo and the recommended
entry point after the [car-goal tutorial](../tutorial-car-goal.md). The
`go2_stabilize` reach-avoid task is the one that produced the best-ever result on
this stack (a `ReachAvoidSAC2P` policy).

## Status and requirements

| | |
|---|---|
| Maturity | **Stable** (`go2_stabilize`, `go2_walker_flat`) · Research (`go2_locomote`) |
| Requires | stock mjlab sim stack (no fork); NVIDIA GPU |
| Adversary | supported (`go2_stabilize`, `go2_locomote`) |
| Recommended config | `configs/go2_stabilize_reachavoidsac2p.yaml` (SAC2P) · `configs/go2_stabilize_reachavoidppo1p.yaml` (PPO1P) · `configs/go2_walker_flat.yaml` (task policy) |

## Task IDs

| Task ID | Mode | Objective | Learner (1P) | Adversary |
|---|---|---|---|---|
| `go2_stabilize` | reach-avoid | stand / stay upright at rest vs adversarial base force | `ReachAvoidPPO1P` / `ReachAvoidSAC1P` | ✔ → `*2P` |
| `go2_locomote` | reach-avoid | track a forward velocity command vs adversarial force | `ReachAvoidPPO1P` | ✔ → `*2P` |
| `go2_walker_flat` | cumulative | dense-reward flat walker — the task policy filters wrap | stock `PPO` | — |

## What success and failure mean

- **Failure** (`g < 0`): the robot falls — base height below the terrain-relative
  floor, excessive tilt, or a non-foot body contacts the ground. The episode
  terminates.
- **Success** (`l ≥ 0`, reach-avoid): the robot is in the target stance set
  (upright and, for `go2_locomote`, tracking the commanded velocity).

## Observation and action spaces

- **Observations** are proprioceptive (base orientation / angular velocity, joint
  positions and velocities, previous action, and the command for the locomotion
  variant). The two-player critic uses an asymmetric (larger) observation group.
- **Action**: 12 joint targets (`ctrl_dim=12`).
- **Adversary action** (`--adversary`): a 3-dim base wrench (force), scaled by the
  training-time magnitude; `--dstb-scale` sweeps it at eval.

## Margins

- **`g`** (safety) = upright / stance integrity: terrain-relative base height,
  tilt, and non-foot contact — negative on a fall. Rides on the reward channel;
  the env terminates on `g < 0`.
- **`l`** (target) = in the target stance set (upright, at rest for
  `go2_stabilize`; tracking the command for `go2_locomote`).

See [margins](../concepts/margins.md) and `robot_safety_sandbox/margins.py`.

## Training budget and hardware

Modest — `go2_stabilize` needs no curriculum or staging. The SAC2P recipe fits in
~2 GB of GPU memory (the 1M-transition replay is CPU-side) and trains in roughly
an hour or two on a single modern NVIDIA GPU. `go2_walker_flat` is a larger
dense-reward run (4096 envs, ~150M env-steps).

## Train

```bash
# reach-avoid, single-player (PPO)
python examples/train.py --config configs/go2_stabilize_reachavoidppo1p.yaml

# reach-avoid, two-player (SAC) — the reference recipe
python examples/train.py --config configs/go2_stabilize_reachavoidsac2p.yaml

# the dense-reward task policy the filters wrap
python examples/train.py --config configs/go2_walker_flat.yaml
```

## Evaluate

```bash
# the safety policy's own reach / safe rates, unfiltered
python examples/eval.py --task go2_stabilize \
    --safety-policy runs/go2_stabilize_reachavoidsac2p/final_model.zip --safety-only \
    --no-filter --num-envs 256 --steps 1000

# filter gauntlet: the walker under a swept adversarial attack
python examples/eval.py --task go2_locomote --adversary \
    --task-policy   runs/go2_walker_flat/final_model.zip \
    --safety-policy runs/go2_stabilize_reachavoidsac2p/final_model.zip \
    --filter value --dstb policy --dstb-scale 0.5 --num-envs 256
```

## Visualize

```bash
python examples/play.py --task go2_stabilize --algo ReachAvoidSAC2P \
    --run runs/go2_stabilize_reachavoidsac2p
# add --adversary to drive the learned worst-case force too
```

## Expected result

The `ReachAvoidSAC2P` policy holds a robust upright stance against the learned
adversary and is the strongest safety policy on this stack. As a certificate it
supplies the `V(s)` that the [filter gauntlets](../guide/evaluation.md) use to
protect the flat walker.

## Known limitations

- `go2_locomote` reuses the `go2_stabilize` twin as a fallback; that twin was
  trained at rest (cmd_vx = 0), so it is out-of-distribution when asked to hold a
  1 m/s gait — a locomotion-specific fallback is needed for a strong filter result
  there.
- The `V ≥ 0` certificate from an older SAC twin is unsound (the entropy-bonus fix
  in safety_sb3 v0.4.0) — retrain on v0.4.0.

## Related source files

- `robot_safety_sandbox/tasks/go2_stabilize.py` — the `go2_stabilize` registration.
- `robot_safety_sandbox/tasks/go2_velocity.py` — `go2_locomote` and the
  `go2_walker_flat` cumulative task policy.
- `configs/go2_stabilize_reachavoidsac2p.yaml`, `configs/go2_stabilize_reachavoidppo1p.yaml`,
  `configs/go2_walker_flat.yaml`.
