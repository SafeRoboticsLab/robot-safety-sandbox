"""Classic-control nominal twins: the gym dense reward, vanilla SB3.

The dense counterpart of tasks/classic_safety.py — same envs, gym's own
reward stack (healthy + forward - ctrl_cost), kind="nominal". Used as the
task policy in classic filter experiments and as the gym-faithful baseline.
"""

from __future__ import annotations

from ..registry import TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.classic.hopper_env_cfg import hopper_env_cfg

  def hopper_dense_cfg(play: bool = False):
    # gym default episode (1000 steps @ dt 0.008) for dense locomotion training
    return hopper_env_cfg(play=play, episode_s=8.0)

  register(TaskSpec(
    task_id="hopper_dense", cfg_builder=hopper_dense_cfg,
    kind="nominal", default_algo="PPO", ctrl_dim=3,
    kwargs={"ctrl_gain": 1.0},
    description="Gym Hopper-v4 dense reward (healthy + forward - ctrl_cost), "
                "vanilla SB3 PPO. Nominal twin of hopper_safety; train with "
                "train_nominal.py."))
