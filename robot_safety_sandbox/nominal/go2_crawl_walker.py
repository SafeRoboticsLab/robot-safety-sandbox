"""Go2 low-crawl walker (nominal pi_task, crawl/bar experiments).

The Stage-1 dense-reward walker from the two-stage crawl decomposition: crawl
terrain + bar perception, body_height target 0.22 (a ducked crawl), trained on
the env's dense reward stack -- the thing reach-avoid l-shaping structurally
cannot produce (gait quality). Nominal controller for the crawl safety filter.
"""

from __future__ import annotations

from ..registry import TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_crawl.env_cfg import (
    unitree_go2_crawl_walk_env_cfg, unitree_go2_crawl_walk_video_env_cfg)

  register(TaskSpec(
    task_id="go2_crawl_walk", cfg_builder=unitree_go2_crawl_walk_env_cfg,
    kind="nominal", default_algo="PPO",
    description="Dense-reward low-crawl WALKER (vanilla SB3 PPO -> a real "
                "gait). Nominal pi_task for the crawl safety filter; train "
                "with train_nominal.py."))
  register(TaskSpec(
    task_id="go2_crawl_walk_video",
    cfg_builder=unitree_go2_crawl_walk_video_env_cfg,
    kind="nominal", default_algo="PPO",
    description="Packed-terrain herd render of go2_crawl_walk (eval video)."))
