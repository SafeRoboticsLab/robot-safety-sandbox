# Episode termination

`end_criterion` is a `TaskSpec` field (with a `--end-criterion` override on
`examples/train.py`) that says **when the episode ends**:

| `end_criterion` | terminates when | use |
|---|---|---|
| `"failure"` (default) | `g < 0` (+ timeout). **Never on reach.** | reach *deeper* — the agent keeps going after reaching, so the reach-avoid value climbs with `l` up to the `g` ceiling |
| `"reach-avoid"` | `g < 0` **or** (`g ≥ 0` **and** `l ≥ 0`) | reach and stop — the episode ends at the target boundary |
| `"timeout"` | only the env timeout | diagnostic / pure value-learning |

This is the **environment half** of a pairing whose algorithm half is the
learner's `terminal_type` (safety_sb3 §4). They are orthogonal; all pairings are
constructible. The pairing that learns to reach *deeper* into the target is
`end_criterion="failure"` + `terminal_type="all"`.

## How it is implemented

Implemented as a mjlab `DoneTerm` (`zoo_reach_success`, fires on `g ≥ 0 ∧ l ≥ 0`)
added only in `reach-avoid` mode — a real termination term, so mjlab auto-resets
on the same step rather than one step late. Default `"failure"` adds no term.

Defaults reproduce prior behavior exactly: an audit of all safety tasks found
**none currently terminates on success**, so every task stays `"failure"` and is
bit-identical. Switch a task to reach-and-stop by setting
`end_criterion="reach-avoid"` on its `TaskSpec`, or per-run with
`--end-criterion`.

## See also

- [Margins](margins.md) — the `g` / `l` the criterion tests.
- [The task registry](../reference/registry.md) — the `TaskSpec` field.
- The [car-goal tutorial](../tutorial-car-goal.md) — `car_goal` uses
  `end_criterion="reach-avoid"` (drive there and stop).
