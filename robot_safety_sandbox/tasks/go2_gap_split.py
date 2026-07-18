"""SPLIT TEST tasks: avoid-only vs reach-avoid on ONE identical setup.

The clean single-variable A/B for the R-CBF liveness claim: same reverse
curriculum (far stance -> near-edge standstill), same gap ladder (steppable ->
half a Go2), same warm-start (soft-landing), same optimizer. The ONLY
difference is the reach term l. Hypothesis: through the mid-arc levels both
regress to the shared "land softly" policy; as the curriculum reaches the
near-edge standstill, avoid-only (indifferent between standing and crossing,
both safe) regresses to standing, while reach-avoid (l grounds the far stance)
retains a gradient to discover the jump.
"""

from __future__ import annotations

from ..margins import compose, g_terrain_relative, l_zero
from ..registry import TaskSpec, register


def register_all() -> None:
  from safe_mjlab_zoo.envs.go2_gap.split_test import (
    unitree_go2_split_env_cfg, l_stable_far)
  g = g_terrain_relative
  register(TaskSpec(
    task_id="go2_gap_split_avoid", cfg_builder=unitree_go2_split_env_cfg,
    margin_fn=compose(g, l_zero), default_algo="SafetyPPO",
    warmstart_from="go2_gap_landing",
    description="SPLIT TEST avoid-only (SafetyPPO): reverse curriculum "
                "far-stance -> near-edge standstill + gap ladder; identical to "
                "_ra except no reach term."))
  register(TaskSpec(
    task_id="go2_gap_split_ra", cfg_builder=unitree_go2_split_env_cfg,
    margin_fn=compose(g, l_stable_far), default_algo="ReachAvoidPPO",
    warmstart_from="go2_gap_landing",
    description="SPLIT TEST reach-avoid (ReachAvoidPPO): same setup, RA target "
                "= stable stance on the far platform. Single-variable contrast "
                "vs _avoid."))
