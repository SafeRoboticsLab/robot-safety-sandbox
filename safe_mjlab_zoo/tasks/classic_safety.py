"""Classic-control safety tasks (Robust-Gymnasium lineage), GPU-native.

The safe_adaptation_dev `mujoco-spar` ISAACS experiments (SafetyHopper-v4 & co.
on CPU MuJoCo, 16 subproc envs) re-homed on the mjlab/warp tensor path — same
obs/action/healthy semantics, same action-additive disturbance convention
(dstb = +-25% of the ctrl bound), orders of magnitude faster.

Margins: g = the gym healthy conditions as a min-form margin (survival task,
avoid-only / l = 0). The dense gym reward lives on the kind="nominal" twins
(see nominal/classic_dense.py) for filter experiments and baselines.
"""

from __future__ import annotations

from ..margins import compose, l_zero
from ..registry import TaskSpec, register


def register_all() -> None:
  from safe_mjlab_zoo.envs.classic.hopper_env_cfg import (
    hopper_env_cfg, hopper_health_margin)

  def hopper_margins(env):
    g = hopper_health_margin(env)
    return g, l_zero(env)

  register(TaskSpec(
    task_id="hopper_safety", cfg_builder=hopper_env_cfg,
    margin_fn=hopper_margins,
    ctrl_dim=3, dstb_dim=3, default_algo="SafetyPPO",
    supports_adversary=True,
    kwargs={"ctrl_gain": 1.0, "dstb_mode": "action", "dstb_gain": 0.25},
    description="Gym SafetyHopper survival (avoid-only: g = healthy margin, "
                "l = 0), ISAACS action-additive dstb +-0.25. GPU port of the "
                "safe_adaptation_dev robust_gym_hopper_isaacs setup."))
