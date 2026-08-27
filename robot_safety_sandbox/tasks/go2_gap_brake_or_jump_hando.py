"""handover-range finetune task. A single NEW task variant,
distinct from the `_ra*` / `_ras` arms (which the package + eval depend on):

  go2_gap_brake_or_jump_ra_hando

the naive-RA `_w30` recipe verbatim — brake_or_jump reverse curriculum @ gap 0.30,
`g = g_terrain_relative`, `l = l_stable_far`, mode REACH_AVOID, BARE (no lander /
gate / policy_mask, so `algo_name` picks `ReachAvoidPPO1P`) — EXCEPT the reset
event, which is the 50% harvested-handover / 50% curriculum spawn mixture
(envs/go2_gap/brake_or_jump_hando.py). Warm-start from the trained RAS arm to extend
its origination range from D<=0.35 to D>=0.45. Existing tasks untouched.
"""
from __future__ import annotations

from functools import partial

from ..margins import compose, g_terrain_relative
from ..registry import REACH_AVOID, TaskSpec, register

HANDO_GAP_WIDTH = 0.30


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_gap.brake_or_jump import l_stable_far
  from robot_safety_sandbox.envs.go2_gap.brake_or_jump_hando import (
    unitree_go2_brake_or_jump_hando_env_cfg)

  register(TaskSpec(
    task_id="go2_gap_brake_or_jump_ra_hando",
    cfg_builder=partial(unitree_go2_brake_or_jump_hando_env_cfg,
                        gap_width=HANDO_GAP_WIDTH),
    margin_fn=compose(g_terrain_relative, l_stable_far),
    mode=REACH_AVOID,
    description=(
      "brake_or_jump reverse curriculum @ gap 0.30 with a 50% "
      "harvested-handover (uniform over D 0.30-0.55, real ~0.8 m/s walking "
      "momentum) / 50% existing curriculum spawn mixture. g=g_terrain_relative, "
      "l=l_stable_far, margins/gate UNCHANGED. BARE reach-avoid (ReachAvoidPPO1P, "
      "no lander/mask). Warm-start from the RAS arm to extend origination D<=0.35 -> "
      ">=0.45. One-variable (reset spawn) vs go2_gap_brake_or_jump_ra_w30.")))
