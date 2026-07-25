"""SPLIT TEST v2 tasks: avoid vs reach-avoid on a reverse curriculum over
HARVESTED real jump states. Phase 1 = width 0.12; Phase 2 widens the gap
(0.20, 0.30) to test whether the commitment-envelope split persists/grows.
Identical env/warm-start/optimizer; the ONLY difference is the reach term l."""

from __future__ import annotations

from functools import partial

from ..margins import compose, g_terrain_relative
from ..registry import AVOID, REACH_AVOID, TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_gap.brake_or_jump import (
    unitree_go2_brake_or_jump_env_cfg, l_stable_far)
  g = g_terrain_relative

  def _pair(width, suffix):
    cb = partial(unitree_go2_brake_or_jump_env_cfg, gap_width=width)
    register(TaskSpec(
      task_id=f"go2_gap_brake_or_jump_avoid{suffix}", cfg_builder=cb,
      margin_fn=compose(g), mode=AVOID,
      description=f"brake-or-jump avoid-only @gap {width}: reverse curriculum over "
                  f"harvested jump states; no reach term (compose(g), no l)."))
    register(TaskSpec(
      task_id=f"go2_gap_brake_or_jump_ra{suffix}", cfg_builder=cb,
      margin_fn=compose(g, l_stable_far), mode=REACH_AVOID,
      description=f"brake-or-jump reach-avoid @gap {width}: RA target = stable far "
                  f"stance. Single-variable contrast vs _avoid."))

  # Phase 1 (0.12) — existing IDs (no suffix); Phase 2 widens the gap. The
  # warm-start lineage (0.12 twins from go2_gap_crossing, then each width from
  # the previous width's SAME twin) is a run-level --load choice, recorded in
  # docs/log/experiments.md.
  _pair(0.12, "")
  _pair(0.20, "_w20")
  _pair(0.30, "_w30")
