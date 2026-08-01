# Choose a workflow

The package supports four primary workflows. Decide which one you need, then
follow the linked page.

| I want to… | Workflow | Produces | Page |
|---|---|---|---|
| Synthesize a safety / reach-avoid policy (a certificate + controller) | **Safety-policy training** | a `safety_sb3` checkpoint | [Training](training.md) |
| Train an ordinary dense-reward policy (the thing a filter will wrap) | **Task-policy training** | a stock SB3 checkpoint | [Training](training.md) |
| Test how safe a task policy is under a filter (and an attack) | **Safety-filter evaluation** | evaluation metrics | [Evaluation](evaluation.md) |
| Train a task policy *inside* a filter, failure-free during training | **Filtered task-policy training** | a stock SB3 checkpoint | [Train inside a filter](../safety-filter-training.md) |

## The four workflows

### 1. Safety-policy training

```text
task (+ margins g, l)  ->  safety_sb3 learner  ->  safety-policy checkpoint
                            (Safety* / ReachAvoid*, 1P or 2P)
```

You supply a task whose `mode` is `safety` or `reach-avoid`. The learner is
resolved by the [MAP](../concepts/map.md) (`mode` × `--family` × `--adversary`).
The output is a checkpoint carrying both a controller and a value function
`V(s)` — the `V ≥ 0` reach-avoid / safety certificate.

### 2. Task-policy training

```text
task (mode="cumulative", dense reward)  ->  stock SB3 learner  ->  task-policy checkpoint
```

A `cumulative` task is ordinary reward-maximizing RL. It trains with **stock
SB3**, so the checkpoint is a vanilla SB3 zip that loads without `safety_sb3`.
This is the **task policy** (`π_task`) a safety filter wraps — e.g.
`go2_walker_flat`.

### 3. Safety-filter evaluation

```text
task-policy checkpoint  +  safety-policy checkpoint  +  a filter composition
        (π_task)                (certificate + fallback)
                                        |
                                        v
                        one eval harness  ->  metrics (safe_rate, task_success, ...)
                     (optionally under a learned worst-case attack)
```

A **safety filter** watches the task policy and hands control to a **fallback**
when a **monitor** says the proposed action is unsafe. `examples/eval.py` composes
the environment, the task policy, the filter, and the attack independently. See
[evaluation](evaluation.md) and the [filter concepts](../concepts/filters.md).

### 4. Filtered task-policy training

```text
task policy  +  safety fallback  +  filter-wrapped training env
                                        |
                                        v
                        SAC learner  ->  a task policy trained (nearly) failure-free
```

Here the filter sits **around the environment during training**, so the task
policy learns while a fallback guarantees it (nearly) never fails — the *Provably
Optimal RL under Safety Filtering* (PORL) setting. It is off-policy (SAC) so the
executed action, not the proposed one, enters the replay buffer. See
[train inside a filter](../safety-filter-training.md).

## Glossary of participating pieces

- **Task policy** (`π_task`, "nominal") — an ordinary dense-reward policy; the
  thing being filtered. Trained with stock SB3 (`mode="cumulative"`).
- **Safety policy** ("twin") — a `safety_sb3` checkpoint carrying the value `V(s)`
  and a fallback controller. Certifies states and drives the robot when engaged.
- **Fallback** — the controller the filter switches to when the monitor rejects
  the task policy's action (usually the safety policy's own actor).
- **Monitor** — the rule that decides whether a proposed action is safe (a value
  test, a Q-CBF condition, or a shadow-sim rollout).
- **Intervention** — how control is handed over (the canonical least-restrictive
  switch, or the smoothed variant).
- **Adversary** — a learned worst-case disturbance, used to stress two-player
  policies and filters (`--adversary`).
