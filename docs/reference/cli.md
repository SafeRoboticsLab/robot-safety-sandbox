# Command-line reference

The four executable entry points and their most-used flags. Every flag is listed
by `--help`; run `python examples/<script>.py --help` (and, for the trainers,
`python examples/train.py --family <f> --help`) for the complete, authoritative
list. The tables below are curated, not generated.

!!! note "One trainer, two families"
    `examples/train.py` is a router: it needs a `--family` (`on_policy`/`ppo` or
    `off_policy`/`sac`, or a `family:` key in the config) and forwards all other
    flags to the chosen trainer. See [training](../guide/training.md).

## Training — `train.py` / `train_on_policy.py` / `train_off_policy.py`

### Common (both families)

| flag | meaning |
|---|---|
| `--config CONFIG` | YAML recipe (keys = flag names); sets defaults, CLI overrides it |
| `--task TASK` | which registered task to train (required unless in the config) |
| `--env-override KEY=VAL` | override a task `cfg_builder` param (repeatable) |
| `--num-envs N` | parallel environments |
| `--steps N` | total env-steps |
| `--seed N` | random seed |
| `--adversary` | two-player run (the MAP's **P**) |
| `--end-criterion {failure,reach-avoid,timeout}` | override the task default |
| `--terminal-type {all,g}` | reach-avoid learners only |
| `--net NET` | comma-separated hidden dims |
| `--device DEVICE` | e.g. `cuda:0` |
| `--out DIR` | output root (keep under `runs/`) |
| `--wandb-project NAME` / `--no-wandb` | logging |

### On-policy (PPO)

| flag | meaning |
|---|---|
| `--load LOAD` / `--load-tensornorm PATH` | warm-start from a checkpoint |
| `--reset-value` | policy-only warm-start (reinit critic, clear optimizer moments) |
| `--lr` / `--ent-coef` / `--vf-coef` | core PPO knobs |
| `--adaptive-lr` / `--no-adaptive-lr`, `--desired-kl` | KL-adaptive learning rate |
| `--gamma`, `--gamma-anneal` / `--no-gamma-anneal` | discount + anneal (anneal OFF for PPO by default) |
| `--std-floor` / `--std-ceil` / `--reset-log-std` / `--max-std` | action-std control |
| `--l-anneal-steps` / `--l-hold-steps` | reach-set curriculum for warm-started reach-avoid |
| `--norm-freeze-steps` | freeze obs-norm at the start of warm-started runs |
| `--video-interval N` | env-steps between wandb eval videos |

!!! warning "`--smoke` is off-policy only"
    There is **no `--smoke` on the on-policy trainer.** Shrink a PPO run with
    `--num-envs` / `--steps` instead.

### Off-policy (SAC)

| flag | meaning |
|---|---|
| `--smoke` | tiny-budget verification (shrinks learning-starts / eval / leaderboard) |
| `--run-suffix STR` | disambiguate sweep cells in wandb / run dirs |
| `--safety-filter KEY=VAL` | filtered-training block (repeatable) — see [train inside a filter](../safety-filter-training.md) |
| `--lr` / `--critic-lr` / `--dstb-lr` / `--ent-coef-lr` / `--dstb-ent-coef-lr` | per-agent learning rates |
| `--tau`, `--target-update-interval` | target-network update |
| `--buffer-size`, `--batch-size`, `--gradient-steps`, `--learning-starts` | replay / update budget |
| `--gamma-schedule {step,geometric,off}`, `--gamma-init`, `--gamma-end` | discount anneal |
| `--target-entropy`, `--min-alpha`, `--max-alpha` | entropy target and alpha bounds |
| `--eval-rollouts` / `--eval-freq` / `--eval-envs` | on-device eval |
| `--leaderboard-freq` / `--leaderboard-episodes` / `--leaderboard-eval-envs` | throughput league |
| `--force-max` / `--force-ramp-frac` / `--force-floor` / `--force-init` | adversary force schedule |

## Evaluation — `eval.py`

Four axes chosen independently (see [evaluation](../guide/evaluation.md)).

| flag | meaning |
|---|---|
| `--task TASK` / `--preset {gap_gauntlet}` / `--env-override KEY=VAL` | the environment |
| `--task-policy PATH` | the `π_task` checkpoint being filtered (omit for fallback-only) |
| `--safety-only` | use the safety policy's own control actor as the task policy |
| `--safety-policy PATH` | the safety-policy checkpoint (certificate + fallback) |
| `--task-obs-key` / `--safety-obs-key` | override an obs group (auto-detected by default) |
| `--filter {value,critic,qcbf,rollout,gameplay}` / `--no-filter` | the filter composition, or the control arm |
| `--eps` | switching threshold (hand over when the monitored margin ≤ eps) |
| `--smoothing` (+ `--caution` / `--hysteresis` / `--rest-speed`) | smoothed intervention variant |
| `--kappa` | `qcbf` class-K coefficient |
| `--horizon` / `--rollouts` / `--recertify-every` | `rollout` / `gameplay` monitor |
| `--adversary`, `--dstb {none,random,policy}`, `--dstb-scale`, `--adversary-policy` | the attack |
| `--num-envs` / `--steps` / `--episode-s` / `--cmd-vx` | run size |
| `--out PATH` | write metrics JSON |
| `--video PATH` (+ `--video-fps`) | write an mp4 (needs `MUJOCO_GL=egl` headless) |
| `--traj DIR` (+ `--traj-envs` / `--traj-frame {spawn,world}`) | record trajectories |

!!! note "Retired flags"
    `--twin`, `--nominal`, `--nominal-from-twin`, `--nominal-obs-key`, and
    `--dstb-twin` were renamed in v0.4.0. See the
    [rename table](../guide/evaluation.md#flag-renames-from-v03x).

## Rendering — `play.py`

Load a checkpoint and drive it in a viewer.

| flag | meaning |
|---|---|
| `--task TASK` | the task |
| `--algo {SafetyPPO1P,…,ReachAvoidSAC2P,PPO}` | the MAP name the checkpoint was trained as |
| `--run RUN` | run dir holding `final_model.zip` + its obs normalizer — `tensornormalize.pt` (tensor path) or `vecnormalize.pkl` (cumulative on-policy / stock SB3); `play.py` auto-detects whichever is present |
| `--load LOAD` | explicit model `.zip` (overrides `--run`) |
| `--adversary` | two-player checkpoints: also drive the learned disturbance |
| `--viewer {auto,native,viser}` | native (needs `$DISPLAY`) / viser (browser, headless) |
| `--num-envs` / `--steps` / `--env-override KEY=VAL` | herd size, auto-stop, cfg overrides |

## Value probe — `eval_brake_or_jump_value.py`

A gap-specific critic probe (contrasts the reach-avoid twin's value ordering
across a momentum sweep):

| flag | meaning |
|---|---|
| `--ra-model PATH` | corrected-RA checkpoint `.zip` (required) |
| `--task TASK` | split task (sets gap width): `_ra` / `_ra_w20` / `_ra_w30` |
| `--avoid-model DIR` | optional avoid run to contrast against |
| `--buggy-model DIR` | optional (g-anchor) reach-avoid run to contrast against |

!!! info "Generation is a follow-up"
    Generating these tables directly from `argparse` (so defaults and choices
    cannot drift) is a planned improvement. For now, `--help` is the authoritative
    source.
