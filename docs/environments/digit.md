# Digit stabilize

An Agility **Digit** humanoid must stay upright against a **worst-case torso
force** — a two-player *avoid* game (ISAACS proper): the control policy maximizes
the safety value, an adversary applies the most destabilizing admissible force,
and the learned value is the robust-invariance certificate.

![digit stabilize rollout](assets/digit.png){ width="420" }

## Tasks

| task | objective | learner (1P → 2P) |
|---|---|---|
| `digit_stabilize` | stand / track vs adversarial torso force (**reach-avoid**, real target) | `ReachAvoidPPO1P` → `ReachAvoidPPO2P` |
| `digit_stabilize_avoid` | stay upright forever (**avoid**) | `SafetyPPO1P` → `SafetyPPO2P` |
| `digit_stabilize_stay` / `digit_box_stabilize_*` | stance-set viability variants | `SafetyPPO1P` → `SafetyPPO2P` |

(Names are the on-policy family; `--family off_policy` gives the `*SAC*` cells —
note `SafetyPPO2P` and `SafetySAC2P` are structurally different games, see the
[API guide](../API.md#3-registry-api-and-the-map).)

The avoid tasks are genuine two-player **avoid** games — with `--adversary` they
resolve to `SafetyPPO2P` (no target set). They do **not** emulate an
avoid task with a degenerate `l` (the retired `l_neg` pattern): avoid is not a
reach-avoid instance — see the [API guide](../API.md#5-marginspy).

## Margins

- **`g`** (safety) = upright / stance integrity: torso tilt, base height,
  non-foot contact — negative on a fall. Digit *stepping* needs the
  `MjlabSafety_Digit` `entity.py` fork patch.
- **`l`** (target, `digit_stabilize` only) = in the target stance set (upright,
  at rest). The avoid variants declare no `l`.

## Run it

```bash
python examples/train.py --family on_policy --task digit_stabilize_avoid --adversary   # -> SafetyPPO2P (two-player avoid)
python examples/train.py --family on_policy --task digit_stabilize                      # reach-avoid, single-player
```

```python
from robot_safety_sandbox import make_tensor, algo_name
env = make_tensor("digit_stabilize_avoid", num_envs=2048, adversary=True)
# algo_name("digit_stabilize_avoid", adversary=True) -> "SafetyPPO2P"
```
