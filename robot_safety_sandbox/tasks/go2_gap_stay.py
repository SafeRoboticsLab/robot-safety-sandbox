"""sRAS Phase II — the STAY (viability) game task. A single NEW
task, distinct from the crossing arms (existing tasks untouched):

  go2_gap_stay

Avoid-only (mode SAFETY -> SafetyPPO1P): g = g_terrain_relative (the standard
legged failure margin: terrain-relative height, tilt, non-foot contact), NO reach
target. Spawn = harvested LANDED far-side states (scratchpad/e100_harvest.py);
5 s episodes; FULL margin (illegal_contact ACTIVE). The trained SafetyPPO critic
is V_stay; its conservatively-calibrated non-negative set is Omega[stay].
"""
from __future__ import annotations

from functools import partial

from ..margins import compose, g_terrain_relative
from ..registry import AVOID, TaskSpec, register

STAY_GAP_WIDTH = 0.30


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_gap.stay import unitree_go2_stay_env_cfg

  register(TaskSpec(
    task_id="go2_gap_stay",
    cfg_builder=partial(unitree_go2_stay_env_cfg, gap_width=STAY_GAP_WIDTH),
    margin_fn=compose(g_terrain_relative),
    mode=AVOID,
    description=(
      "sRAS Phase II: avoid-only STAY/viability game @ gap 0.30. "
      "Spawn = harvested landed far-side states (stay-game harvest); g=g_terrain_"
      "relative, no reach; FULL margin (illegal_contact ACTIVE); 5 s episodes; "
      "NO curriculum. SafetyPPO1P -> V_stay, whose calibrated >=0 set is "
      "Omega[stay]=viab(T\\F).")))
