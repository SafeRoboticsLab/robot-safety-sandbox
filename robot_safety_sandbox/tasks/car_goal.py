"""car_goal: differential-drive reach-avoid (the mjlab-zoo analog of the
``safety_sb3`` bicycle5d validation env). A car drives to a goal disk while
avoiding obstacle cylinders; single-player reach-avoid, 2 wheel-velocity
controls, ended on reach (g >= 0 AND l >= 0) or collision.
"""

from __future__ import annotations

from ..registry import TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.car_goal.env_cfg import (
    car_goal_env_cfg,
    car_margins,
  )
  register(TaskSpec(
    task_id="car_goal", cfg_builder=car_goal_env_cfg, margin_fn=car_margins,
    ctrl_dim=2, default_algo="ReachAvoidPPO", supports_adversary=False,
    end_criterion="reach-avoid",
    kwargs={"ctrl_gain": 1.0, "adversary_body": "agent"},
    description="Differential-drive car reach-avoid: drive to the goal disk "
                "while avoiding obstacle cylinders (bicycle5d analog)."))
