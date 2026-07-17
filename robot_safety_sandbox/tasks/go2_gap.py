"""Go2 gap-jumping benchmark family (parkour skill 1).

Pipeline (each stage warm-starts the next; the jump is NOT learnable in one
stage — it forms through the landing -> crossing reverse curriculum):

  go2_gap_landing   avoid-only   SafetyPPO     mid-air spawn -> soft land
  go2_gap_crossing  avoid-only   SafetyPPO     reverse curriculum launch->land
  go2_gap_chain     reach-avoid  ReachAvoidPPO arrival momentum -> safe rest
  go2_gap_chain (+adversary)     GameplayPPO   two-player reach-avoid game

The env cfgs are NATIVE to the zoo (envs/go2_gap/*, migrated phase-2).
"""

from __future__ import annotations

from ..margins import compose, g_terrain_relative, l_gap_foothold, l_rest
from ..registry import TaskSpec, register


def _cfgs():
  from robot_safety_sandbox.envs.go2_gap.landing import unitree_go2_landing_env_cfg
  from robot_safety_sandbox.envs.go2_gap.crossing import unitree_go2_crossing_env_cfg
  from robot_safety_sandbox.envs.go2_gap.chain import (
    unitree_go2_crossing_chain_env_cfg,
    unitree_go2_crossing_chain_isaacs_env_cfg)
  return dict(landing=unitree_go2_landing_env_cfg,
              crossing=unitree_go2_crossing_env_cfg,
              chain=unitree_go2_crossing_chain_env_cfg,
              chain_isaacs=unitree_go2_crossing_chain_isaacs_env_cfg)


def register_all() -> None:
  cfgs = _cfgs()
  g = g_terrain_relative
  register(TaskSpec(
    task_id="go2_gap_landing", cfg_builder=cfgs["landing"],
    margin_fn=compose(g), default_algo="SafetyPPO",
    description="Mid-air over a gap with clearing velocity; learn soft landing."))
  register(TaskSpec(
    task_id="go2_gap_crossing", cfg_builder=cfgs["crossing"],
    margin_fn=compose(g), default_algo="SafetyPPO",
    warmstart_from="go2_gap_landing",
    description="Reverse curriculum from the landed state back to the launch."))
  register(TaskSpec(
    task_id="go2_gap_chain", cfg_builder=cfgs["chain"],
    margin_fn=compose(g, l_rest), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_crossing", supports_adversary=False,
    description="Takeover momentum -> reach safe rest (brake / jump when needed)."))
  register(TaskSpec(
    task_id="go2_gap_chain_isaacs", cfg_builder=cfgs["chain_isaacs"],
    margin_fn=compose(g, l_rest), default_algo="GameplayPPO",
    warmstart_from="go2_gap_chain", supports_adversary=True,
    description="Chain + worst-case base-force adversary (pinned curricula)."))
