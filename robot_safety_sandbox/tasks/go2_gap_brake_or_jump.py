"""SPLIT TEST v2 tasks: avoid vs reach-avoid on a reverse curriculum over
HARVESTED real jump states. Phase 1 = width 0.12; Phase 2 widens the gap
(0.20, 0.30) to test whether the commitment-envelope split persists/grows.
Identical env/warm-start/optimizer; the ONLY difference is the reach term l."""

from __future__ import annotations

from functools import partial

from ..margins import compose, g_terrain_relative
from ..registry import TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_gap.brake_or_jump import (
    unitree_go2_brake_or_jump_env_cfg, l_stable_far)
  g = g_terrain_relative

  def _pair(width, suffix, warm_avoid, warm_ra):
    cb = partial(unitree_go2_brake_or_jump_env_cfg, gap_width=width)
    register(TaskSpec(
      task_id=f"go2_gap_brake_or_jump_avoid{suffix}", cfg_builder=cb,
      margin_fn=compose(g), default_algo="SafetyPPO",
      warmstart_from=warm_avoid,
      description=f"brake-or-jump avoid-only @gap {width}: reverse curriculum over "
                  f"harvested jump states; no reach term (compose(g), no l)."))
    register(TaskSpec(
      task_id=f"go2_gap_brake_or_jump_ra{suffix}", cfg_builder=cb,
      margin_fn=compose(g, l_stable_far), default_algo="ReachAvoidPPO",
      warmstart_from=warm_ra,
      description=f"brake-or-jump reach-avoid @gap {width}: RA target = stable far "
                  f"stance. Single-variable contrast vs _avoid."))

  # Phase 1 (0.12) — existing IDs (no suffix), warm-start the crossing jumper.
  _pair(0.12, "", "go2_gap_crossing", "go2_gap_crossing")
  # Phase 2 — widen; each stage warm-starts the previous width's twin.
  _pair(0.20, "_w20", "go2_gap_brake_or_jump_avoid", "go2_gap_brake_or_jump_ra")
  _pair(0.30, "_w30", "go2_gap_brake_or_jump_avoid_w20", "go2_gap_brake_or_jump_ra_w20")
