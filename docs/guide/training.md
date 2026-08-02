# Training

One router, two peer RL families. `examples/train.py` dispatches to the on-policy
or off-policy trainer by a **required `--family`** — neither is the "main" one;
they are the [MAP](../concepts/map.md)'s two wired-up **A** values, and both
resolve the learner the same way: the task's `mode` × `--adversary`.

| `--family` | trainer | the MAP's **A** | learners |
|---|---|---|---|
| `on_policy` (alias `ppo`) | `examples/train_on_policy.py` | `PPO` | `{Safety,ReachAvoid}PPO{1P,2P}`, plus **stock SB3 `PPO`** for `cumulative` (numpy bridge, plain SB3 zip) |
| `off_policy` (alias `sac`) | `examples/train_off_policy.py` | `SAC` | `{Safety,ReachAvoid}SAC{1P,2P}`, plus `safety_sb3.CumulativeSAC1P` for `cumulative` (tensor path; SB3-compatible SAC checkpoint) |

!!! note "Cumulative is not the same class in both families"
    A `cumulative` (plain reward-maximizing) task trains under **stock
    `stable_baselines3.PPO`** on the numpy bridge in the on-policy family, but under
    **`safety_sb3.CumulativeSAC1P`** on the tensor path in the off-policy family —
    stock SAC lacks the GPU tensor collector and executed-action readback that
    filtered training needs, so the off-policy cumulative learner is the safety_sb3
    class (its checkpoint stays an SB3-compatible SAC zip). There is no two-player
    cumulative game in either family.

The 2P cells differ **structurally** between the families — `*PPO2P` and `*SAC2P`
are different algorithms (see [the MAP](../concepts/map.md)). `train.py` forwards
every other flag verbatim to the chosen trainer; run `--family <f> --help` to see
that trainer's options, or the [CLI reference](../reference/cli.md).

## Basic runs

```bash
# on-policy (PPO family)
python examples/train.py --family on_policy  --task go2_gap_chain --terminal-type all   # ReachAvoidPPO1P
python examples/train.py --family ppo        --task digit_stabilize_avoid --adversary   # SafetyPPO2P
# off-policy (SAC family)
python examples/train.py --family off_policy --task go2_stabilize                       # ReachAvoidSAC1P
python examples/train.py --family sac        --task go2_stabilize --adversary --num-envs 1024  # ReachAvoidSAC2P
```

- `--adversary` — two-player run; the learner is resolved by `algo_name`.
- `--terminal-type {all,g}` — forwarded to reach-avoid learners; ignored (with a
  notice) on avoid tasks.
- `--end-criterion {failure,reach-avoid,timeout}` — overrides the task's default
  (see [termination](../concepts/termination.md)).

## Config recipes (recommended)

Rather than a wall of flags, a run is a small **recipe** — a YAML file whose keys
are the trainer's flag names, plus a reserved `family:` key so `--family` isn't
needed on the CLI. Precedence is **argparse defaults < config < explicit CLI
flags**.

```bash
python examples/train.py --config configs/go2_stabilize_reachavoidsac2p.yaml           # the E042 recipe (family: off_policy)
python examples/train.py --config configs/go2_stabilize_reachavoidsac2p.yaml --seed 3  # override one knob
```

Every run also dumps its fully resolved config to `<outdir>/config.yaml` — re-run
with `--config <that file>` to reproduce it exactly. The shipped recipes live in
[`configs/`](../reference/registry.md#configuration-recipes):

| file | family | what |
|---|---|---|
| `configs/car_goal.yaml` | `on_policy` | `ReachAvoidPPO1P` — the [car-goal tutorial](../tutorial-car-goal.md) recipe |
| `configs/go2_stabilize_reachavoidppo1p.yaml` | `on_policy` | `ReachAvoidPPO1P` — the safety-PPO recipe |
| `configs/go2_stabilize_reachavoidsac2p.yaml` | `off_policy` | `ReachAvoidSAC2P` — the E042 recipe (best-ever on `go2_stabilize`) |
| `configs/go2_walker_flat.yaml` | `on_policy` | `mode: cumulative` — the dense-reward Go2 walker task policy on stock SB3 PPO |
| `configs/go2_walker_filtered.yaml` | `off_policy` | `mode: cumulative` + a `safety_filter:` block — [filtered training](../safety-filter-training.md) |

## Env / task overrides

A config `env_overrides:` dict (or `--env-override KEY=VAL`, repeatable) forwards
params to the task's `cfg_builder`, overriding values baked into its registration
(e.g. `gate_close_rate`, `bar_clearance`) — so a recipe can define the
*environment* too, with no argparse edit. An unaccepted key fails loudly.

```bash
python examples/train.py --config <recipe>.yaml --env-override gate_close_rate=0.003
```

## Warm starts

Some tasks form their skill only through **staged warm-starts** — a `--load` per
stage seeding a rare-win skill for the next (see [Go2 gap](../environments/go2-gap.md)).
The warm-start *lineage* is a run-level `--load` choice recorded in your
experiment log, not a registry field.

- `--load <run>/final_model.zip` — warm-start from a checkpoint (loads the value
  net too).
- `--reset-value` (on-policy) — policy-only warm-start: reinitialize the critic and
  clear optimizer moments. Needed when the previous objective differs (e.g.
  avoid → reach-avoid), where a critic fit to a different objective would explode
  `value_loss`.

## The PPO recipe that works (hard-won)

`normalize_obs=True` (observations only — **never** the margins, which carry the
specification), `ent_coef=1e-4`, `log_std_init=ln(0.3)`, `adaptive_lr=True`
(`desired_kl=0.01`, lr `5e-4`), `n_steps=48`. Watch the `env/Curriculum/*` logger
keys — a stalled curriculum looks exactly like converged training in the reward
curve.

The off-policy trainer additionally exposes the reference-faithful safety controls
(`--gamma-schedule`, per-agent learning rates, throughput leaderboard defaults);
see the [CLI reference](../reference/cli.md#off-policy-sac) and the safety_sb3
[hyperparameters](https://saferoboticslab.github.io/safety-stable-baselines/hyperparameters/).

## Output

Each run writes to `runs/<task>/` (override with `--out`, but keep runs under
`runs/`): `final_model.zip`, the observation normalizer, `config.yaml` (resolved
recipe), and a TensorBoard directory.

The normalizer file depends on the training path:

| Training path | Normalizer file |
|---|---|
| Safety / reach-avoid PPO or SAC (tensor path) | `tensornormalize.pt` |
| `cumulative` on-policy (stock SB3 PPO, numpy bridge) | `vecnormalize.pkl` |
| `cumulative` off-policy (`CumulativeSAC1P`, tensor path) | `tensornormalize.pt` |

In short: only the on-policy `cumulative` path (stock SB3 on the numpy bridge)
writes `vecnormalize.pkl`; every tensor-path run — including off-policy
`cumulative` — writes `tensornormalize.pt`. `examples/play.py` auto-detects
whichever is present next to the checkpoint.

Base tensor environments also report always-on `safety/*` counters (how often
training fails); see
[filtered training](../safety-filter-training.md#failure-counting-is-always-on).
