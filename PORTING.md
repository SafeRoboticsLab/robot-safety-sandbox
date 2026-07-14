# Porting your task into safe_mjlab_zoo + training with safety_sb3

This is the guide for adding a new safety task (new robot, new obstacle, new
objective) and training it.

**Tasks differ structurally.** Spawn distributions, curricula, failure/reach
margins, staging — these are per-task design choices, not zoo requirements.
Calibrate against the three reference families, from simplest to most staged:

| exemplar | spawn | curriculum | pipeline | margins |
|---|---|---|---|---|
| `go2_stabilize` (the ORIGINAL task) | stock resets | none | none — one run from scratch | 8 body-frame inequalities, task-local |
| `go2_crawl` | takeover momentum strata | bar-height + assist | none (single stage) | shared g + rest with exclusion windows |
| `go2_gap` | staged (mid-air -> launch -> takeover) | reverse curricula per stage | 3-stage warm-start chain | shared g + per-stage l |

Start by asking which row your task resembles. If exploration can discover
your target behavior from stock spawns, you may need none of the machinery —
`go2_stabilize` is ~100 lines total.

## The mental model (read this first)

A task is exactly two callables plus a registration:

```
cfg_builder(play: bool) -> ManagerBasedRlEnvCfg    # WHAT the world is + WHERE you spawn
margin_fn(env) -> (g, l)                           # WHAT failure and success mean
register(TaskSpec(...))                            # discoverability + lineage
```

The algorithms never change. The env contract they consume:
- **reward = g(s)**, the physical safety margin (g < 0 == failure). Absolute
  units. **Never normalize, reshape, or add bonuses to it** — safety_sb3 will
  raise if it detects reward normalization, and everything else corrupts the
  Safety Bellman backup silently.
- **l(s)** = target margin (l >= 0 == target reached). Zeros for avoid-only.
- Timeouts are never value-bootstrapped (handled by the algorithms).

## Step 1 — env cfg (`envs/<your_task>/env_cfg.py`)

Plain mjlab `ManagerBasedRlEnvCfg`: scene (robot from `envs/assets_go2` or your
own), terrain, observations, reset events, curricula. Study
`envs/go2_gap/chain.py` (takeover-momentum spawns, stratified across the
decision boundary) and `envs/go2_crawl/env_cfg.py`.

Design-dimension menu — each item says WHEN it applies (none are universal;
`go2_stabilize` uses none of them):

1. **Spawn distribution.** Default: stock resets (stabilize). A deployment
   safety FILTER should instead train on its *takeover* distribution (arrival
   momentum / mid-maneuver states — crawl, gap-chain). If the target behavior
   is a rare win exploration can't find (jump, crawl-through), STAGE it:
   spawn inside/past the maneuver first and extend backward with a reverse
   curriculum (`envs/go2_gap/crossing.py`).
2. **Physical consistency of custom spawns** (only if you write root/joint
   states yourself): spawn at the pose's static equilibrium with zero joint
   velocity. A few cm of ground penetration ejects the robot into the contact
   margin at t=0 and kills every episode before a learning signal exists.
3. **Curricula** (only if difficulty must ramp): promote on the BINDING skill
   event (crossed / passed / carried), read from pre-reset state — promoting
   on generic survival lets an easy behavior drag difficulty past the real
   frontier. If a term mutates `env_origins`, run it AFTER terms that read
   origin-relative positions.
4. **Eval cfgs**: drop per-reset terrain re-rolls (they break row pinning) and
   do controlled eval spawns INSIDE a reset event (out-of-band teleports leave
   stale raycast caches -> false terminations). Applies to any task with
   terrain rows / raycast sensors.

## Step 2 — margins (`margins.py` or your own)

Margins can be composed from the shared library (`compose(g_terrain_relative,
l_rest)`) **or written task-locally** (`envs/go2_stabilize/env_cfg.py` defines
its own 8-inequality stance margins in ~40 lines — often the clearest option).
Conventions: signed distances, O(1) normalized, clamped ~±3.

Choose l to match the DECISION your value function must encode — this is the
most task-specific choice in the whole port:
- stance/stationarity (stabilize) when the target is an equilibrium state;
- command-tracking liveness (locomote) when the target is sustained motion;
- `l_rest` (safe stop) when momentum genuinely commits the robot (gaps);
- rest with EXCLUSION windows (crawl) when only rest in certain regions
  counts — note the failure mode: if the robot can always bail out safely,
  plain rest is satisfiable everywhere and "stop always" wins.
- avoid-only (`l_zero`) when there is no liveness requirement at all.

## Step 3 — register (`tasks/<your_task>.py`)

```python
register(TaskSpec(
  task_id="myrobot_mytask", cfg_builder=my_cfg, margin_fn=compose(g, l),
  default_algo="ReachAvoidPPO", warmstart_from="myrobot_easier_stage",
  supports_adversary=True, description="..."))
```
Import it in `safe_mjlab_zoo/__init__.py`. `warmstart_from` documents the
pipeline lineage — staged warm-starts are how hard skills actually form.

## Step 4 — train

```bash
python examples/train.py --task myrobot_mytask --steps 200000000 --seed 0
# next stage:
python examples/train.py --task myrobot_harder --steps 2000000000 --seed 0 \
    --load runs_zoo/myrobot_mytask/final_model.zip
```

Watch in wandb: `env/Curriculum/*` (THE progress gauge — a stalled curriculum
looks exactly like converged training in the reward curve), `rollout/ep_len_mean`,
and `eval/video` (a clip from step 0, then every 25M steps).

## The knobs (defaults are the validated recipe — change with reason)

| knob | default | when to touch |
|---|---|---|
| `--steps` | 2e8 | **Budget by episode-generations, not habit.** Simple tasks (stabilize) converge in tens of millions; curriculum tasks ratchet once per episode-generation and can need billions (Go2 chain: ~2B). If a curriculum is climbing but slowly, you need more steps, not new ideas. |
| `--num-envs` | 2048 | GPU memory bound (~8 GB at 2048 on the Go2). More envs = more rare-win discovery. |
| `--seed` | 0 | Always set for benchmark runs. |
| `--ent-coef` | 1e-4 | Raise ONLY if exploration provably fails; 5e-3 destroyed precise-control learning. |
| `log_std_init` | ln(0.3) | SB3's default (std 1.0) is far too hot for robot control. |
| `learning_rate` | 5e-4 + adaptive KL (`desired_kl` 0.01) | Leave adaptive on; it replaces schedule tuning. |
| `normalize_obs` | True | Required. clip stays ±10 (SB3 parity — ±100 silently broke warm-starts). |
| `--norm-freeze-steps` | 5M | Warm-started runs: protects the inherited policy while stats re-adapt. |
| `--load` | — | Warm-start the next pipeline stage; loads model + normalizer stats. |
| `--adversary` + `--force-max` | off / 50 N | Robustification channel. Force ramps 8N -> max over 55% of training. |
| n_steps / batch | 48 / (envs*48/4) | rsl_rl-parity minibatch structure; rarely worth touching. |

## Debugging order when training "doesn't work"

1. `env/Curriculum/*` flat at 0? Your rare win never fires — fix spawns/staging
   (rules 1–3 above), not hyperparameters.
2. `ep_len` collapses at t=0? Spawn-state transient (rule 2) or an eval/obs
   staleness artifact.
3. Warm-start behaves worse than its own eval? Normalizer stats/clip mismatch —
   verify with a deterministic A/B rollout on both paths.
4. Reward great but skill absent? Your metric saturates on an easy behavior;
   gate on the skill event and check the videos.
5. Only then touch knobs — and change one at a time.

## SAC / numpy path

`make_numpy(task_id, ...)` exposes the same task as a classic SB3 VecEnv
(`info["l_x"]`, `TimeLimit.truncated`) for `SafetySAC`/`ReachAvoidSAC`/
`IsaacsSAC` and stock SB3 tooling. Prefer the tensor path for on-policy work
(~3x faster).
