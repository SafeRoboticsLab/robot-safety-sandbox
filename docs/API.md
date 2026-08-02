# API guide

This page is the landing map for the **environment layer** — tasks, margins, the
registry, and the bridges to `safety_sb3`. The material that used to live here has
been split by intent into concepts, reference, and user-guide pages; follow the
links below. The algorithm layer (learners, backups, `terminal_type`) is
documented in
[safety-stable-baselines `docs/API.md`](https://github.com/SafeRoboticsLab/safety-stable-baselines/blob/main/docs/API.md).

## Concepts

- [Safety and reach-avoid margins](concepts/margins.md) — the `g` / `l` contract,
  the reach-avoid value, composing a `margin_fn`, and why avoid is not a
  reach-avoid instance.
- [The MAP naming convention](concepts/map.md) — Mode · Algorithm · Players, the
  `algo_name` formula, and why `*PPO2P` and `*SAC2P` are different algorithms.
- [Episode termination](concepts/termination.md) — the `end_criterion` field.
- [Safety-filter architecture](concepts/filters.md) — fallback · monitor ·
  intervention.

## Reference

- [Task registry and Python API](reference/registry.md) — `TaskSpec` fields, the
  registry functions, the two bridges, and configuration recipes.
- [Code reference](reference.md) — the generated docstrings.
- [Command-line reference](reference/cli.md) — every flag of the trainers, the
  evaluator, and the viewer.

## User guide

- [Choose a workflow](guide/workflows.md) — the four primary workflows.
- [Training](guide/training.md) — the trainer router, recipes, warm starts.
- [Evaluation](guide/evaluation.md) — the eval harness and its four axes.

## Add a task or robot

- [Tutorial: build the car-goal task](tutorial-car-goal.md) — the full stack, from
  scratch.
- [Extending](EXTENDING.md) and [Porting a task](porting.md).
- [Installation](installation.md) and [requirements](getting-started/requirements.md).
