# Training configs

Reusable experiment **recipes** for `examples/train.py` (PPO family) and
`examples/train_sac.py` (SAC family). A recipe is a YAML file whose keys are the
trainer's argparse flag names (the `dest`, e.g. `num_envs`, `gamma_schedule`).

## How it works

```bash
python examples/train_sac.py --config configs/go2_stabilize_gameplaysac.yaml
# override any single knob on top of the recipe:
python examples/train_sac.py --config configs/go2_stabilize_gameplaysac.yaml --seed 3 --num-envs 2048
```

Precedence: **argparse defaults  <  `--config` file  <  explicit CLI flags.** So a
recipe is a starting point you can still tweak per run. Config keys are validated
against the trainer's flags (an unknown key is an error).

## Env / task overrides (no trainer flag needed)

A reserved **`env_overrides:`** block (a dict) tunes the *environment/task* itself
— params baked into the task's registration (its `cfg_builder`), e.g.
`gate_close_rate`, `bar_clearance`, spawn/force ranges. It is a **passthrough**
(not validated against the trainer flags) forwarded to `make_tensor`, so a config
can define a full experiment — algorithm **and** env — without editing `train.py`
or adding argparse:

```yaml
task: go2_low_bar_gate_ra
num_envs: 1024
env_overrides:
  gate_close_rate: 0.003   # override the value baked into the task registration
  bar_clearance: 0.30
```

Or per-knob on the CLI (repeatable; overrides the config's `env_overrides` per key):

```bash
python examples/train.py --config <recipe>.yaml --env-override gate_close_rate=0.003
```

An override key the task's `cfg_builder` doesn't accept fails loud. The resolved
`env_overrides` is written into the per-run `config.yaml` (reproducible).

## Reproducibility

Every run writes its **fully-resolved** config to `<outdir>/config.yaml`. To
reproduce a run exactly: `--config <that run's config.yaml>`. This is the durable
record of "what hyperparameters that run used" — pair it with the `docs/log/`
experiment registry for the narrative of what was tried and why.

## Recipes here

| file | trainer | what |
|---|---|---|
| `go2_stabilize_gameplaysac.yaml` | `train_sac.py` | 2-player reach-avoid SAC (GameplaySAC), reference-faithful + fast leaderboard — the E042 config |
| `go2_stabilize_reachavoidppo.yaml` | `train.py` | 1-player reach-avoid PPO (ReachAvoidPPO), the safety-PPO recipe |

Add new recipes freely. Keep **canonical** recipes here (committed); keep ad-hoc
sweeps / scratch experiments out of the repo (they belong with the private
experiment log).
