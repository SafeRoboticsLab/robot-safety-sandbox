"""CERTIFIED-TRANSITION funnel tasks (T006 / E083): the batch-2 GET-UP and DESCENT reach-avoid problems.

Two transition funnels, each a first-class reach-avoid problem with its OWN value function (fixing the E081
hazard vacuum where transitions were uncertified zero-shot maneuvers):
  * go2_getup   — GET-UP funnel RA_rest→stand: PRONE spawns under load W∈[0,160], reach the FULL stance target
    while avoiding the universal no-slam catastrophe (``getup_margins``). The genuine gap.
  * go2_descend — DESCENT funnel RA_stand→rest: DISTURBED standing spawns under HEAVY load W∈[80,260], reach
    the full rest set (``weight_rest_margins``; E084 compares vs the reused ``go2_weight_rest_hi``).

Plus fixed-W eval variants (``*_at_{W}``) at pinned W for the physics/handoff sweeps. All ADDITIVE — built on
``go2_stabilize`` + the ``go2_weight_ladder`` carried-load channel; no existing task is modified.
"""

from __future__ import annotations

import functools

from ..registry import REACH_AVOID, TaskSpec, register

# Pinned loads (N) for the fixed-W eval tasks.
_GETUP_EVAL_WS = (0, 40, 80, 120, 160, 200)
_DESCEND_EVAL_WS = (80, 140, 200, 260)


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_transitions.env_cfg import (
    go2_getup_env_cfg,
    go2_descend_env_cfg,
    getup_margins,
    GETUP_LO, GETUP_HI, DESCEND_LO, DESCEND_HI,
  )
  from robot_safety_sandbox.envs.go2_weight_ladder.env_cfg import weight_rest_margins, LOAD_HEIGHT

  # ── GET-UP funnel: prone spawns under load W∈[0,160], reach the FULL stance target. ──
  register(TaskSpec(
    task_id="go2_getup", cfg_builder=go2_getup_env_cfg,
    margin_fn=getup_margins, mode=REACH_AVOID, supports_adversary=True,
    description=f"[transition GET-UP RA_rest→stand, HIGH-CoM (h={LOAD_HEIGHT}): PRONE spawns under load "
                f"W∈[{GETUP_LO:.0f},{GETUP_HI:.0f}]N; g=universal no-slam(<80+1.3W N), l=FULL stance target — "
                "get up to a stable stand despite the load + adversary]"))
  # ── DESCENT funnel (dedicated): disturbed standing spawns under HEAVY load W∈[80,260], reach the rest set. ──
  register(TaskSpec(
    task_id="go2_descend", cfg_builder=go2_descend_env_cfg,
    margin_fn=weight_rest_margins, mode=REACH_AVOID, supports_adversary=True,
    description=f"[transition DESCENT RA_stand→rest (dedicated), HIGH-CoM (h={LOAD_HEIGHT}): DISTURBED standing "
                f"spawns (tilt+lateral vel) under HEAVY load W∈[{DESCEND_LO:.0f},{DESCEND_HI:.0f}]N; "
                "g=no-slam(<80+1.3W N), target=low+level+settled rest]"))

  # ── FIXED-W eval variants (W pinned via lo=hi=W): each funnel's training obs surface at a single test load,
  # for the physics gate / handoff sweep (mirrors the weight/leg arms' ``*_at_*``). ──
  for _w in _GETUP_EVAL_WS:
    register(TaskSpec(
      task_id=f"go2_getup_at_{_w}",
      cfg_builder=functools.partial(go2_getup_env_cfg, lo=float(_w), hi=float(_w)),
      margin_fn=getup_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[transition GET-UP eval @ fixed load W={_w}N]"))
  for _w in _DESCEND_EVAL_WS:
    register(TaskSpec(
      task_id=f"go2_descend_at_{_w}",
      cfg_builder=functools.partial(go2_descend_env_cfg, lo=float(_w), hi=float(_w)),
      margin_fn=weight_rest_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[transition DESCENT eval @ fixed load W={_w}N]"))
