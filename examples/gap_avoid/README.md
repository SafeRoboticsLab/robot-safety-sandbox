# Go2 gap: the avoid-only safety policy

This folder holds the **avoid-only** safety policy from the single-gap experiment of the sRAS / Q-CBF paper, frozen as
it was evaluated there, with the walker it protects, the evaluation script that reproduces its row of the results
table, and notes on how it was trained.

It lives on the branch `share/go2-gap-avoid`, which is robot-safety-sandbox `main` plus this folder and a one-line
change (the jump-state file path can be set with `$RSS_JUMP_BANK`). It is a snapshot for sharing; the paper's full code
release will replace it.

## What is here

| path | what |
|---|---|
| `weights/avoid_w30/` | the avoid-only policy: `final_model.zip` (SafetyPPO1P, safety_sb3) + `tensornormalize.pt` (its observation statistics) |
| `weights/go2_walker_flat/` | the task policy it filters: a blind flat-ground walker (stock SB3 PPO) + `vecnormalize.pkl` + its training `config.yaml` |
| `states/init_states.npz` | the 1000 fixed starting states of the evaluation: distance to the gap, lateral offset, heading |
| `states/jump_bank.pt` | 73,654 harvested jump states (gaps 0.12 / 0.20 / 0.30 m), needed only to **train** the gap tasks |
| `eval_avoid_filter.py` | the evaluation: the walker approaches the gap, the filter decides when the avoid policy takes over |

## The task in short

A Go2 walks toward a 0.30 m gap with a blind walker commanded forward at 1.0 m/s. The walker cannot see the gap. The
safety policy can (its observation includes a terrain height scan) and is trained only to avoid failure:

- **safety margin** `g = g_terrain_relative` (`robot_safety_sandbox/margins.py`), the minimum of three terms, each
  positive when fine:
  - base height above the local terrain, minus 0.08 m, divided by 0.25 m;
  - tilt: (sin 70° − |gravity_xy|) / sin 70°;
  - non-foot contact: (10 N − the largest non-foot contact force) / 10 N.

  Failure is `g < 0`. The training episode ends there.
- **no target**: avoid-only (`compose(g)`), so the policy is never rewarded for crossing. Near the gap it brakes and
  stands.

The policy is task `go2_gap_brake_or_jump_avoid_w30`, learner `SafetyPPO1P`, 12 joint-position actions, a 1175-number
observation (proprioception with history plus the height scan, built by the task's environment).

## Setup

You need robot-safety-sandbox (RSS) and safety-stable-baselines (SSB, `safety_sb3`) installed as in RSS's
[INSTALL.md](../../INSTALL.md) (Linux, NVIDIA GPU, conda). Then get this branch:

```bash
git clone -b share/go2-gap-avoid git@github.com:SafeRoboticsLab/robot-safety-sandbox.git rss-gap-avoid
# or, in an existing clone:  git fetch origin share/go2-gap-avoid && git checkout share/go2-gap-avoid
cd rss-gap-avoid
```

The evaluation script puts this checkout first on the Python path, so it uses this branch's code even if another RSS
clone is installed in the environment. Tested with this branch (RSS `main` 8d9bd63 + this folder), safety_sb3 f26fb8e
(SSB `main`), mjlab 1.2.0, MuJoCo 3.6.0, PyTorch 2.10/2.11, stable-baselines3 2.9.0: the full n = 1000 check below
on an RTX 5090, a 16-start run on an RTX 4070.

## Evaluate

```bash
MUJOCO_GL=egl python examples/gap_avoid/eval_avoid_filter.py                    # the paper's setting: n=1000, strict
MUJOCO_GL=egl python examples/gap_avoid/eval_avoid_filter.py --margin belly
MUJOCO_GL=egl python examples/gap_avoid/eval_avoid_filter.py --num-envs 16 --video /tmp/avoid.mp4
```

Results go to `runs/gap_avoid_eval/<filter>_<margin>_n<N>.{json,npz}`: the rates with 95 % intervals, and per start the
base path, whether the filter was engaged, the outcome and where the filter committed.

**What it does.** Each robot starts standing 0.35 to 1.6 m before the gap face, up to 0.35 m to the side, heading
within ±25°; the 1000 starts are fixed. The walker drives. At every step until it commits, the filter copies the
current state into a second simulator and rolls the avoid policy forward 120 steps (2.4 s), twice:

- `J0` = the lowest `g` along the rollout if the avoid policy takes over now;
- `J1` = the same if the walker takes one more step first.

It commits (the avoid policy drives for the rest of the episode) as soon as `J0 ≥ J1 − 0.05` and `J0 ≥ −0.2`. This is
the paper's optimal-stopping Q-CBF: the Q-CBF condition with κ = 1, read as a stopping rule. Each start is scored on its
first episode (20 s; the script runs 700 steps = 14 s):

- **safe**: the episode never ended in a failure (braking and standing counts as safe);
- **crossed**: the base got past the far rest line (gap width + 0.15 m);
- **success**: crossed and safe.

Margins: **strict** (the paper) also ends the episode when any non-foot body touches the ground, so a brake that drops
a knee counts as a failure; **belly** counts only falls.

**What to expect** (n = 1000; GPU simulation is not bit-reproducible, so expect about ±0.02 between runs):

| margin | crossed | safe | success | notes |
|---|---|---|---|---|
| strict (paper) | 0.008 | 0.856 | 0.007 | the filter commits about 1 m before the gap and the policy brakes; most failures are non-foot contacts while braking |
| strict, rerun from a clean clone of this branch | 0.004 | 0.864 | 0.004 | 934 of 1000 committed, median 1.00 m before the gap; 135 of the 136 failures are non-foot contacts |
| belly (paper) | 0.012 | 0.897 | 0.012 | |
| belly, rerun | 0.009 | 0.895 | 0.008 | all 105 failures are falls |

That is the avoid-only row of the paper's table: safe, but it almost never crosses, because nothing in its objective
asks it to. A run takes 35 to 40 min on an RTX 5090 (strict 38 min, belly 33 min). The filter's look-ahead rollouts
dominate, so the time hardly depends on `--num-envs`: 16 starts took 27 min on an RTX 4070.

Other filters, for comparison (`--filter`):
- `value`: switch to the avoid policy while its value `V(x) ≤ --eps`, the classic value filter;
- `none`: the walker alone.

## Use the policy in your own code

```python
from robot_safety_sandbox.eval import load_twin, safety_modules

model, norm = load_twin("examples/gap_avoid/weights/avoid_w30/final_model.zip", "cuda:0")   # also loads tensornormalize.pt
mods = safety_modules(model, num_envs, "cuda:0")
s_obs = norm(obs)                       # obs: the task's safety observation, (num_envs, 1175)
action = mods["fallback_fn"](s_obs=s_obs)   # deterministic action in [-1, 1], 12 joint targets
value = mods["value_fn"](s_obs=s_obs)       # V(x): the policy's estimate of the worst future g; >= 0 means it expects to stay safe
```

Build the observation with the task's own environment rather than by hand: `build_eval_env` (as in
`eval_avoid_filter.py`) gives an env whose `env.safety_obs_key` observation group is exactly what the policy was
trained on.

Things to know before relying on `V`:
- It is the value of this PPO policy (`V^π`), not the optimal safety value. Its sign is the useful part; do not read
  its magnitude as a distance to failure.
- It is trained on the reverse-curriculum starts described below, not on walk-in approaches. That is why the
  evaluation's filter uses simulated rollouts (`J0`, `J1`) rather than `V` directly.

## How it was trained

The policy is the last stage of a warm-started chain; each stage starts from the previous stage's weights (`--load`):

| stage | task (current name) | starts from | envs | steps |
|---|---|---|---|---|
| 1 | `go2_gap_landing`: soft landing from mid-air over the gap | scratch | 6144 | 100M |
| 2 | `go2_gap_crossing`: reverse curriculum, landing → launch | stage 1 | 6144 | 150M |
| 3 | `go2_gap_brake_or_jump_avoid` (gap 0.12 m) | stage 2 | 2048 | 150M |
| 4 | `go2_gap_brake_or_jump_avoid_w20` (gap 0.20 m) | stage 3 | 2048 | 150M |
| 5 | `go2_gap_brake_or_jump_avoid_w30` (gap 0.30 m) | stage 4 | 2048 | 150M → **`weights/avoid_w30`** |

All five stages are avoid-only (`SafetyPPO1P`, margin `g`). Stages 3–5 replay `states/jump_bank.pt`, states recorded
from the stage-2 policy's jumps, on a reverse curriculum. The curriculum has 13 levels:
- levels 0–7 start the robot on the committed jump arc, sliding from the far platform back to the takeoff;
- the next levels start it on the near platform with less and less momentum;
- the last level mixes far-and-slow, near-and-fast and near-and-slow starts.

A level advances when 70 % of episodes end standing on the far side and 15 % of the next level's episodes do (see
`envs/go2_gap/brake_or_jump.py`). The released policy finished at level 9.

Settings of the released policy (read from the checkpoint), shared by stages 3–5:
- network: 512-256-128 for both actor and critic;
- PPO: γ 0.99, GAE λ 0.95, 48 steps per environment per update, batch 24,576, 5 epochs;
- learning rate fixed at 1e-4; entropy coefficient 1e-3; action std capped at 0.4; target KL 0.01.

All five stages ran on one RTX 5090 in July 2026, on an earlier version of this code (the tasks were then called
`go2_gap_split2_avoid*`) and safety_sb3 0.2. The weights were later re-saved so that they load on the current
safety_sb3; the weights themselves are unchanged.

To train it again on the current code:

```bash
export MUJOCO_GL=egl RSS_JUMP_BANK=$PWD/examples/gap_avoid/states/jump_bank.pt
python examples/train.py --task go2_gap_landing  --num-envs 6144 --steps 100000000 --seed 0 --out runs/gap
python examples/train.py --task go2_gap_crossing --num-envs 6144 --steps 150000000 --seed 0 --out runs/gap \
    --load runs/gap/go2_gap_landing/final_model.zip
STAGE="--num-envs 2048 --steps 150000000 --lr 1e-4 --no-adaptive-lr --ent-coef 1e-3 --max-std 0.4 --target-kl 0.01 --out runs/gap"
python examples/train.py --task go2_gap_brake_or_jump_avoid     $STAGE --load runs/gap/go2_gap_crossing/final_model.zip
python examples/train.py --task go2_gap_brake_or_jump_avoid_w20 $STAGE --load runs/gap/go2_gap_brake_or_jump_avoid/final_model.zip
python examples/train.py --task go2_gap_brake_or_jump_avoid_w30 $STAGE --load runs/gap/go2_gap_brake_or_jump_avoid_w20/final_model.zip
```

- **These commands have not been re-run on the current code.** Defaults have changed since July. Compare against the
  settings above, and watch the `Curriculum/*` keys: a stalled curriculum looks like converged training in the reward
  curve.
- **The jump bank came from our stage-2 policy.** If you retrain stage 2, your jumps will differ from the bank's. The
  bank is still valid physics, but it is not the same pipeline.
- **To change only the last stage**, for example the margin or the gap width, start from the released policy:
  `--load examples/gap_avoid/weights/avoid_w30/final_model.zip`. The warm start also restores its observation
  statistics.

The walker is task `go2_walker_flat` (stock PPO, 4096 envs, 150M steps, seed 0); its full training settings are in
`weights/go2_walker_flat/config.yaml`. RSS's `configs/go2_walker_flat.yaml` is the current recipe for the same task.

## Where the code is

| file | what |
|---|---|
| `robot_safety_sandbox/tasks/go2_gap_brake_or_jump.py` | registers the avoid and reach-avoid tasks at gaps 0.12 / 0.20 / 0.30 |
| `robot_safety_sandbox/envs/go2_gap/brake_or_jump.py` | the reverse-curriculum environment: starts from the jump bank, the level gate |
| `robot_safety_sandbox/envs/go2_gap/brake_or_jump_harvest.py`, `gap.py` | the island terrain, observations, terminations |
| `robot_safety_sandbox/margins.py` | `g_terrain_relative` and `compose` |
| `robot_safety_sandbox/tasks/go2_gap.py` | `go2_gap_landing`, `go2_gap_crossing` (stages 1–2) |
| `robot_safety_sandbox/envs/go2_gap/eval_gauntlet.py` | the evaluation world (`gap_cfg_transform`) |
| `robot_safety_sandbox/filters/`, `robot_safety_sandbox/eval/` | the shadow simulator, filter composition and evaluation harness |
| `docs/environments/go2-gap.md` | overview of the whole Go2 gap family |

## Known limitations

- **The jump bank must be supplied to train.** The harvesting script that produced it is not part of the repo; the
  bank file is.
- **Evaluation is not bit-reproducible on GPU.** Compare configurations over n=1000 or several seeds, not single
  episodes.
- **The strict margin is harsh on braking.** Any knee or thigh touching the ground ends the episode, so part of the
  avoid policy's 14 % failures are hard stops rather than falls into the gap.
- **The policy sees the terrain height scan.** That scan exists in simulation; the robot has no such sensor, so this
  policy is a simulation result, not a hardware policy.
