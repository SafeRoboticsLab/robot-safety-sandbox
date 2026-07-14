"""Go2 blind flat-terrain velocity walker (nominal pi_task, gap experiments).

Deliberately safety-oblivious: 47-dim proprioception (no scans), walks at the
commanded velocity, knows nothing about gaps. The safety twins
(go2_gap_chain_avoid / go2_gap_chain_ra) supply the obstacle handling at
deploy time via the value filter (examples/eval_filter.py).
"""

from __future__ import annotations

from ..registry import TaskSpec, register


def register_all() -> None:
  from safe_mjlab_zoo.envs.velocity.go2 import unitree_go2_flat_env_cfg

  register(TaskSpec(
    task_id="go2_walker_flat", cfg_builder=unitree_go2_flat_env_cfg,
    kind="nominal", default_algo="PPO",
    description="Blind flat-terrain velocity walker (dense reward, vanilla "
                "SB3 PPO). Nominal pi_task for the gap filter experiments; "
                "train with train_nominal.py."))
