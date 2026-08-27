"""Stage-2 RAS v2 launch-quality tasks — the TWIN.

The brake_or_jump reverse curriculum @ gap 0.30 with the AUGMENTED decision-band
spawn mixture (runway / deep-crouch / synthetic cold-edge; see
envs/go2_gap/brake_or_jump_aug.py) that teaches stronger origination launches.
Two arms differing in EXACTLY {gate + lander + policy_mask}:

  go2_gap_brake_or_jump_ra_v2   aug spawns only. ReachAvoidPPO1P, no handover.
  go2_gap_brake_or_jump_ras_v2  aug spawns + a LIVE handover to the weak-launch
                                lander on the calibrated geometric gate
                                (x_rel>=0.22, vx>=1.7 — the one that PASSED the
                                acceptance-style Part A). ReachAvoidMaskedPPO1P
                                (lander steps propagate to the value but are
                                excluded from the policy gradient via policy_mask).

Margins are UNCHANGED from the naive-RA arm: compose(g_terrain_relative,
l_stable_far). ras_v2 MUST be trained with `--algo ReachAvoidMaskedPPO1P`; ra_v2
with the default `ReachAvoidPPO1P`. Warm-start both from the RAS arm
(runs/ras_e082/go2_gap_brake_or_jump_ras/final_model.zip), NO --reset-value.
"""
from __future__ import annotations

import os
from functools import partial

from ..margins import compose, g_terrain_relative
from ..registry import REACH_AVOID, TaskSpec, register
from ..vhat import geom_landable_gate

AUG_GAP_WIDTH = 0.30

# weak-launch lander (the "stay" fallback). Portable via RAAS_LANDER_V2_DIR;
# default = the local lander-v2 run (dir holding final_model.zip + tensornormalize.pt).
_LANDER_V2 = os.environ.get("RAAS_LANDER_V2_DIR",
                            "runs/lander_v2/go2_gap_raas_safelanding_v2")
# Calibrated gate: (x_rel>=0.22, vx>=1.7) PASSED the acceptance-style Part A (no
# decision-band sabotage) in the Stage-2 A/B; dormant at TODAY's weak cold edge (latch
# 0.017) but fires once Stage-2 launches strengthen. geom_landable_gate signature
# is (mj, xr_min, vx_min).
_GATE = dict(xr_min=0.22, vx_min=1.7)


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_gap.brake_or_jump import l_stable_far
  from robot_safety_sandbox.envs.go2_gap.brake_or_jump_aug import (
    unitree_go2_brake_or_jump_aug_env_cfg)
  g = g_terrain_relative
  cb = partial(unitree_go2_brake_or_jump_aug_env_cfg, gap_width=AUG_GAP_WIDTH)

  register(TaskSpec(
    task_id="go2_gap_brake_or_jump_ra_v2", cfg_builder=cb,
    margin_fn=compose(g, l_stable_far), mode=REACH_AVOID,
    description=(
      "RAS v2 Stage-2 aug-only: brake_or_jump reverse curriculum @ "
      "gap 0.30 with the augmented decision-band spawn mixture (runway/deep-crouch/"
      "synthetic cold-edge), g=g_terrain_relative, l=l_stable_far. NO handover. "
      "ReachAvoidPPO1P. One-variable control vs go2_gap_brake_or_jump_ras_v2.")))

  register(TaskSpec(
    task_id="go2_gap_brake_or_jump_ras_v2", cfg_builder=cb,
    margin_fn=compose(g, l_stable_far), mode=REACH_AVOID,
    kwargs=dict(hybrid_skill=_LANDER_V2,
                latch_margin_fn=partial(geom_landable_gate, **_GATE)),
    description=(
      "RAS v2 Stage-2 aug + LIVE handover: same augmented env as "
      "go2_gap_brake_or_jump_ra_v2, plus the weak-launch lander hands over on "
      "the calibrated geometric gate (x_rel>=0.22 AND vx>=1.7); lander steps "
      "propagate to the value but are policy_masked. TRAIN WITH "
      "ReachAvoidMaskedPPO1P (--algo). Differs from _ra_v2 in EXACTLY "
      "{gate + lander + policy_mask}.")))
