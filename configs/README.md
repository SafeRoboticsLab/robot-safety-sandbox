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
