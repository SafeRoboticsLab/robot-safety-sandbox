# Safety-filter architecture

A **safety filter** watches a task policy and takes over when its proposed action
looks unsafe. In this repo a filter is not a class per recipe — it is a
**composition of three swappable modules** (following Hsu / Hu / Fisac Defs. 1–2):

```text
                 proposed action a_task
                          |
        +-----------------v------------------+
        |   Monitor (Δ):  is a_task safe?    |   value / Q-CBF / shadow-rollout
        +-----------------+------------------+
                          |
        Intervention (φ): Δ > eps ? pass a_task : hand over
                          |
        Fallback (π^<):   drives when the filter engages
                          |
                          v
                    executed action
```

- **Fallback** (`π^<`) — the controller run when the filter engages, usually the
  safety policy's own control actor.
- **Monitor** (`Δ`) — the rule that scores how safe the proposed action is.
- **Intervention** (`φ`) — how control is handed over.

There is one concrete `SafetyFilter(fallback, monitor, intervention)`; it is never
subclassed.

## The shipped compositions

`examples/eval.py --filter {…}` picks one:

| `--filter` | monitor | idea |
|---|---|---|
| `value` | the safety value `V(s)` | pass while `V ≥ eps`; the Safety Value filter |
| `critic` | a safety critic `Q(s, a)` | Gameplay-Filters baseline |
| `qcbf` | a Q-CBF condition `Q(x,u) ≥ κ V(x)` | minimal-modification (least-intrusive) filter |
| `rollout` | an mjlab **shadow-sim** H-step rollout of the fallback | verify the fallback stays safe for H steps |
| `gameplay` | a shadow rollout **against a learned adversary** | worst-case rollout monitor |

## The intervention rule

Every filter but `qcbf` composes the **canonical least-restrictive switch**: pass
the task policy's action iff the monitored margin `> eps`, else hand over entirely
to the fallback, with no state carried between steps. It is Def-2 valid by
construction under an exact monitor.

`--smoothing` substitutes a **smoothed variant** (`HeuristicSmoothingIntervention`:
engagement latch + 5-step median + release hysteresis + rest-speed gate). It is
strictly more conservative and **not** Def-2 valid step-by-step — results using it
must be reported as a *smoothed variant* of the named filter. It exists to
reproduce gauntlets from before 2026-07-25. See
[evaluation](../guide/evaluation.md#which-switch-you-get).

## Where filters are used

- **At evaluation** — wrap a trained task policy and measure how safe it is, under
  an attack. See [evaluation](../guide/evaluation.md).
- **During training** — wrap the *training environment* so a task policy learns
  (nearly) failure-free (the PORL setting). See
  [train inside a filter](../safety-filter-training.md).

The composition API lives in `robot_safety_sandbox/filters/` (modules) and
`robot_safety_sandbox/eval/filters.py` (the named compositions and their wiring).
