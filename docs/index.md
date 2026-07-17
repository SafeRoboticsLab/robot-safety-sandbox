# Robot Safety Sandbox

Parallelized **mjlab** environments for nominal-policy training, safety-policy
synthesis, and safety-filter evaluation — the environment layer for
[safety-stable-baselines](https://github.com/SafeRoboticsLab/safety-stable-baselines).

Reach-avoid / avoid-only × single-agent / adversarial (ISAACS), on GPU end-to-end,
plus a `filters/` library (value shielding, R-CBF/Q-CBF projection, rollout shielding).

## Start here

- **[Environments](environments/index.md)** — the robot benchmark showreel (Go2 gap, crawl, Digit): task, margins, run-it snippet, figures.

- **[Installation](installation.md)** — the pinned mjlab sim stack.
- **[API guide](API.md)** — the `g`/`l` contract, `TaskSpec`, the registry, `end_criterion`.
- **[Code reference](reference.md)** — auto-generated from source docstrings.
- **[Extending](EXTENDING.md)** / **[Porting a task](porting.md)** — add a margin, sensor, terrain, or robot.

```python
from robot_safety_sandbox import make_tensor, list_tasks, algo_name
from safety_sb3 import ReachAvoidPPO

env = make_tensor("go2_gap_chain", num_envs=2048)     # ~50k steps/s on 12 GB
model = ReachAvoidPPO("MlpPolicy", env, normalize_obs=True, terminal_type="all")
model.learn(2_000_000_000)
```
