# Robot Safety Sandbox

Parallelized **mjlab** environments for safety-policy synthesis, task-policy
training, and safety-filter evaluation — the environment layer for
[safety-stable-baselines](https://github.com/SafeRoboticsLab/safety-stable-baselines).

Reach-avoid / avoid-only × single-player / zero-sum two-player, end to end on
GPU, plus a `filters/` library in which a safety filter is a composition
of three swappable modules: fallback (pi^<) / monitor (Delta) / intervention (phi).

## The MAP

> **Here's a MAP to navigate the codebase — Mode. Algorithm. Players.**

    M = Mode       Safety | ReachAvoid | Cumulative    the Bellman operator
    A = Algorithm  PPO | SAC | A2C | DQN               the RL update rule
    P = Players    1P | 2P                             single-player | zero-sum

Every learner name consists of those three letters concatenated — `SafetyPPO1P`,
`ReachAvoidSAC2P` — and each letter has exactly one source: **M** is the task's
`TaskSpec(mode=...)`, **A** is `train.py --family`, **P** is `--adversary`.
`algo_name(task_id, adversary, family)` is that concatenation and nothing else.

Note `*PPO2P` and `*SAC2P` are different ALGORITHMS, not one game with two
optimizers: `*SAC2P` is minimax on one shared joint-action critic
`Q(s, [a_ctrl, a_dstb])`; `*PPO2P` is an alternating best-response approximation
with two independent `V(s)` nets, two rollout buffers, and a phase machine.
See [the API guide](API.md).

## Start here

- **[Tutorial: your first env](tutorial-car-goal.md)** — new here? Build one complete
  reach-avoid task from scratch (a car that reaches a goal while avoiding obstacles)
  and train it end to end. Theory-grounded, seven steps, one small robot.
- **[Environments](environments/index.md)** — the robot benchmark showreel (Go2 gap, crawl, Digit): task, margins, run-it snippet, figures.

- **[Installation](installation.md)** — the pinned mjlab sim stack.
- **[API guide](API.md)** — the `g`/`l` contract, `TaskSpec`, the registry, `end_criterion`.
- **[Code reference](reference.md)** — auto-generated from source docstrings.
- **[Extending](EXTENDING.md)** / **[Porting a task](porting.md)** — add a margin, sensor, terrain, or robot.

```python
from robot_safety_sandbox import make_tensor, list_tasks, algo_name
from safety_sb3 import ReachAvoidPPO1P

algo_name("go2_gap_chain")                            # -> 'ReachAvoidPPO1P'
env = make_tensor("go2_gap_chain", num_envs=2048)     # ~50k steps/s on 12 GB
model = ReachAvoidPPO1P("MlpPolicy", env, normalize_obs=True, terminal_type="all")
model.learn(2_000_000_000)
```
