"""Go2 blind flat-terrain velocity walker (mode=CUMULATIVE pi_task, gap
experiments).

Deliberately safety-oblivious: 47-dim proprioception (no scans), walks at the
commanded velocity, knows nothing about gaps. Plain reward-maximizing RL — the
CUMULATIVE backup (reward + gamma*V'), i.e. stock SB3 PPO on the env's dense
reward stack. The safety twins (go2_gap_chain_avoid / go2_gap_chain_ra) supply
the obstacle handling at deploy time via the value filter
(examples/eval_filter.py).

Env cfgs: envs/velocity/go2.py.
"""

from __future__ import annotations

from ..registry import CUMULATIVE, TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.velocity.go2 import unitree_go2_flat_env_cfg

  register(TaskSpec(
    task_id="go2_walker_flat", cfg_builder=unitree_go2_flat_env_cfg,
    mode=CUMULATIVE,
    description="Blind flat-terrain velocity walker (dense reward, stock SB3 "
                "PPO). Task policy pi_task for the gap filter experiments; "
                "train with train.py --family on_policy."))
