# Evaluation

One entry point, `examples/eval.py`, over four axes chosen **independently**.
Nothing about a particular terrain lives in the harness
(`robot_safety_sandbox/eval/`); a task's own configuration is an `EvalPreset`
registered by the task that owns it.

!!! note "One harness, not four scripts (v0.4.0)"
    Earlier versions had separate `eval_*.py` scripts. They are consolidated into
    the single `examples/eval.py`. Commands from older docs that used `--twin`,
    `--nominal`, or `--nominal-from-twin` will raise an argparse error — the flag
    names changed (table below).

## The four axes

| axis | flag | module |
|---|---|---|
| environment | `--task` / `--preset` / `--env-override` | `robot_safety_sandbox/eval/envs.py` |
| task policy `π_task` | `--task-policy` (or `--safety-only`) | `robot_safety_sandbox/eval/policies.py` |
| filter | `--safety-policy` + `--filter {value,critic,qcbf,rollout,gameplay}` | `robot_safety_sandbox/eval/filters.py` |
| intervention | `--eps` (+ `--smoothing`, see below) | `robot_safety_sandbox/filters/intervention.py` |
| attack | `--dstb {none,random,policy}` + `--dstb-scale` | `robot_safety_sandbox/eval/runner.py` |
| metrics | automatic (+ the preset's extras) | `robot_safety_sandbox/eval/metrics.py` |

- `--task-policy` — the `π_task` checkpoint (a stock SB3 zip) being filtered. Omit
  for a zero task policy (a fallback-only run).
- `--safety-only` — use the **safety** policy's own control actor as the task
  policy (the disturbance-effect probe).
- `--safety-policy` — the safety-policy checkpoint supplying the certificate + the
  fallback.
- `--task-obs-key` / `--safety-obs-key` — override an obs group (both are
  auto-detected by default).
- `--no-filter` — the unfiltered control arm.

### Flag renames from v0.3.x

| v0.3.x flag | v0.4.0 flag |
|---|---|
| `--twin` | `--safety-policy` |
| `--nominal` | `--task-policy` |
| `--nominal-from-twin` | `--safety-only` |
| `--nominal-obs-key` | `--task-obs-key` |
| `--dstb-twin` | `--adversary-policy` |

The results summary field `twin` is likewise renamed `safety_policy`.

## Examples

```bash
# filter gauntlet on flat ground under a swept adversarial attack
python examples/eval.py --task go2_locomote --adversary \
    --task-policy   runs/go2_walker_flat/final_model.zip \
    --safety-policy runs/go2_stabilize_sac2p/final_model.zip \
    --filter gameplay --dstb policy --dstb-scale 0.5 --num-envs 256

# the gap gauntlet (E021), a preset rather than its own script
python examples/eval.py --preset gap_gauntlet \
    --task-policy   runs/go2_walker_flat/final_model.zip \
    --safety-policy runs/go2_gap_chain_ra/final_model.zip \
    --filter value --gap-width 0.35 --n-gaps 1 --num-envs 256 --steps 600

# the unfiltered control arm (--safety-policy is still required — it supplies the
# certificate + fallback the harness always loads — but --no-filter never applies it)
python examples/eval.py --task go2_locomote --adversary \
    --task-policy   runs/go2_walker_flat/final_model.zip \
    --safety-policy runs/go2_stabilize_sac2p/final_model.zip \
    --no-filter --num-envs 256
```

The env is always a `MjlabTensorSafetyEnv`, never a raw `ManagerBasedRlEnv`: that
is what supplies the task's `(g, l)` under the training contract, the **live**
`[ctrl, dstb]` action space (so a disturbance is *delivered*, not just predicted
in a shadow rollout), and obs-group auto-detection.

## Attacks

`--dstb-scale` is the attack-strength knob, so a robustness result is a **curve**
rather than a point: run the same command at `0.0 / 0.25 / … / 1.0`. `1.0` is the
task's own training-time magnitude. It reaches the physics on whichever channel
the task uses (a `wrench` disturbance is unit-normalized before scaling, so the
knob is the bridge's `force_scale`). `--dstb policy` plays a trained min-player;
supply a separate `--adversary-policy` when the attacker is not the same safety
policy as the certificate.

## Which switch you get

Every filter but `qcbf` composes `LeastRestrictiveIntervention` — the canonical
rule and nothing more: pass the task policy's action iff the monitored margin
`> eps`, hand over to the fallback entirely otherwise, no state carried between
steps. It is Def-2 valid by construction under an exact monitor.

`--smoothing` substitutes `HeuristicSmoothingIntervention`: the same switch
wrapped in an engagement latch, a 5-step median, release hysteresis, and a
rest-speed release gate (`--caution`, `--hysteresis`, `--rest-speed` apply only
then). Those heuristics are strictly more conservative and **not** Def-2 valid
step-by-step, so a run using them is a *smoothed variant* of the named filter and
must be reported as one.

!!! warning "Reproducing pre-2026-07-25 gauntlets"
    Every gauntlet before 2026-07-25 ran the smoothed variant, which was then the
    only implementation and the default. Reproducing those numbers needs
    `--smoothing`; the default no longer gives them.

## Metrics

The CBF-DDP comparison protocol — `task_success` and `safe_rate` from the task's
own margins, per-actuator `jerk`, `intervention_mass` = `‖π_task − π_filtered‖₁`,
wall-clock per control step split filter/env, engagement/caution rates, and
distance travelled. Write them to JSON with `--out`. A task adds its own reading
by registering an `EvalPreset` with a `metrics` hook (e.g. the gap's
crossing/livelock rates in `robot_safety_sandbox/envs/go2_gap/eval_gauntlet.py`).
Record trajectories with `--traj DIR`.

!!! note "Evaluation is not bit-reproducible"
    mjlab keeps observation noise on in play mode and MuJoCo-Warp is not
    bit-deterministic, so repeat runs of the *same* command differ. Compare
    configurations over several runs, not once.

See the [CLI reference](../reference/cli.md#evaluation-evalpy) for the full flag
list, and [safety filters](../concepts/filters.md) for the filter compositions.
