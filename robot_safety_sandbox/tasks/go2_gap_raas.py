"""RAAS v2 ("reach-and-always-safe") tasks — a 2-policy hybrid at fixed gap 0.30.

The plan:
  - pi_safelanding: avoid-only airborne lander -> soft-land on the far side.
    The frozen safety fallback, and the policy whose rollouts define the safelanding
    certificate.
  - V_safelanding / S_safelanding={V>=0}: state-only land-value + conformal
    threshold (fit AFTER the lander trains).
  - tube-init + switching-env 3-run ablation (RA / avoid-g_terrain /
    avoid-g=V_safelanding), added once the lander and its certificate land.

This module currently registers Phase 1 (the lander). The old gap-jump tasks
(``go2_gap_raas_step{1,2}``, pure-random 2-step over ``l_stable_far``) are RETIRED
— that was the clean-negative approach superseded by the certified-launch design;
the old env ``envs/go2_gap/raas.py`` stays on disk (unused) until it is rewritten
into the tube-init env.
"""

from __future__ import annotations

import os
from functools import partial

from ..margins import compose, g_terrain_relative
from ..registry import AVOID, REACH_AVOID, TaskSpec, register
from ..vhat import l_vhat, l_vhat_reward

# RAAS v2 pins one gap width across the whole pipeline (lander, certificate, runs).
RAAS_GAP_WIDTH = 0.30


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_gap.landing import unitree_go2_landing_env_cfg

  # pi_safelanding: the avoid-only airborne lander at the fixed gap.
  # Mode=AVOID (g only) -> SafetyPPO1P. Spawns mid-air over the 0.30 gap with
  # clearing momentum and learns a soft landing onto the far platform.
  register(TaskSpec(
    task_id="go2_gap_raas_safelanding",
    cfg_builder=partial(unitree_go2_landing_env_cfg, gap_width=RAAS_GAP_WIDTH),
    margin_fn=compose(g_terrain_relative),
    mode=AVOID,
    description=(
      "RAAS v2 pi_safelanding: avoid-only airborne lander @ gap 0.30. "
      "Soft-land from airborne+momentum onto the far platform; the frozen "
      "safety fallback + source of V_safelanding/S_safelanding.")))

  # pi_safelanding_v2: the lander RETRAINED on the harvested weak-
  # launch / in-gap manifold (frozen RAS reach policy from rest), same avoid-only
  # g_terrain_relative @ gap 0.30. Warm-start from the base lander, NO --reset-value (same
  # margin -> value transfers). Makes the certified handover LIVE at the cold edge.
  from robot_safety_sandbox.envs.go2_gap.safelanding_v2 import (
    unitree_go2_safelanding_v2_env_cfg)
  register(TaskSpec(
    task_id="go2_gap_raas_safelanding_v2",
    cfg_builder=partial(unitree_go2_safelanding_v2_env_cfg, gap_width=RAAS_GAP_WIDTH),
    margin_fn=compose(g_terrain_relative),
    mode=AVOID,
    description=(
      "RAAS v2 pi_safelanding_v2: avoid-only lander @ gap 0.30 RETRAINED "
      "on the harvested weak-launch/in-gap manifold (frozen RAS reach policy from "
      "rest, ras_harvest_weaklaunch.py) instead of the base lander's ballistic S_airborne box. "
      "Same env/terrain/obs/g as the base lander. Warm-start the base lander, SafetyPPO1P, no --reset-value.")))

  # harvest env: SAME lander env but a BROAD airborne reset (S_airborne) — for
  # rolling the FROZEN lander to label the certificate dataset, NOT for training.
  register(TaskSpec(
    task_id="go2_gap_raas_safelanding_harvest",
    cfg_builder=partial(unitree_go2_landing_env_cfg, gap_width=RAAS_GAP_WIDTH, broad=True),
    margin_fn=compose(g_terrain_relative),
    mode=AVOID,
    description=(
      "RAAS v2 harvest: gap 0.30 lander env with a BROAD airborne spawn "
      "(wide x/z/momentum/pose incl. doomed) — roll frozen pi_safelanding to build "
      "the balanced V_safelanding dataset. Not a training task.")))

  # ---- the 3-run ablation (tube init + switching env) --------------------
  # Shared switching env: hybrid_skill = frozen pi_safelanding (handover on
  # s in S_safelanding), latch guard = l_vhat (V_safelanding - p*). Paths portable:
  # RAAS_LANDER_DIR (default = the local safelanding run) + RAAS_VSL_PATH (in vhat.py).
  lander = os.environ.get("RAAS_LANDER_DIR",
                          "runs/raas_safelanding/go2_gap_raas_safelanding")
  switch = {"hybrid_skill": lander, "latch_margin_fn": l_vhat}
  tube = partial(unitree_go2_landing_env_cfg, gap_width=RAAS_GAP_WIDTH, tube=True)

  # Run 1 — Reach-avoid: reach S_safelanding, avoid S_failure.
  register(TaskSpec(
    task_id="go2_gap_raas_run1_ra", cfg_builder=tube,
    margin_fn=compose(g_terrain_relative, l_vhat_reward), mode=REACH_AVOID,
    kwargs=dict(switch),
    description=(
      "RAAS v2 Run 1 (RA): g=g_terrain_relative (avoid S_failure), l=l_vhat_reward "
      "(un-saturated logit reach reward, Youden p*); handover guard stays "
      "l_vhat (sigmoid p*=0.975); tube init.")))

  # Run 2 — Avoidonly1: avoid S_failure only (the naive-avoid control).
  register(TaskSpec(
    task_id="go2_gap_raas_run2_avoid_failure", cfg_builder=tube,
    margin_fn=compose(g_terrain_relative), mode=AVOID,
    kwargs=dict(switch),
    description=(
      "RAAS v2 ablation Run 2 (Avoidonly1): g=g_terrain_relative, no reach; tube init "
      "+ handover. Naive-avoid control (expect loiter / no standstill initiation).")))

  # Run 3 — Avoidonly2: avoid the COMPLEMENT of S_safelanding (the avoid != RA control).
  register(TaskSpec(
    task_id="go2_gap_raas_run3_avoid_notsafeland", cfg_builder=tube,
    margin_fn=compose(l_vhat), mode=AVOID,
    kwargs=dict(switch),
    description=(
      "RAAS v2 ablation Run 3 (Avoidonly2): g=V_safelanding-p* (avoid leaving "
      "S_safelanding), no reach; tube init + handover. The 'avoid is not a "
      "reach-avoid instance' negative control.")))

  # EVAL/PROBE env: S_edge spawn (grounded near edge, rest->slow) + the SAME
  # switching env. Load an ablation checkpoint here and roll it (raas_landable_map /
  # play.py) to measure the STANDSTILL-INITIATION far-reach rate + render an eval
  # video from near-standstill states (not the airborne tube). Not a training task.
  register(TaskSpec(
    task_id="go2_gap_raas_edge",
    cfg_builder=partial(unitree_go2_landing_env_cfg, gap_width=RAAS_GAP_WIDTH, edge=True),
    margin_fn=compose(g_terrain_relative, l_vhat), mode=REACH_AVOID,
    kwargs=dict(switch),
    description=(
      "RAAS v2 standstill-initiation probe: S_edge spawn (grounded, rest->slow "
      "at the near edge) + handover env. Eval a run's policy here for the initiation "
      "rate + edge-state eval video.")))
