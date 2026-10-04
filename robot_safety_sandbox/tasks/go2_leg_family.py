"""LEG-DEGRADATION two-mode ODD tasks: Go2 with a degrading FR leg.

The leg-ODD analog of ``go2_weight_ladder`` — same two-mode spec-family template, but the ODD is
θ = FR-leg allowable-torque fraction (the existing ``randomize_fr_torque`` machinery) instead of a carried
load W. Two conditioned reach-avoid modes, each with a per-episode θ:
  * go2_leg_stand — STAND spec (``stance_margins``): hold a stable stand despite the weak leg + adversary.
    θ ∈ [0.5, 1.0] (the comfortably-feasible stance band).
  * go2_leg_rest  — REST spec (``leg_rest_margins``): come down softly to a LOW LEVEL belly rest WITHOUT
    slamming (constant 80 N no-slam cap — no carried load). θ ∈ [0.0, 1.0] INCLUDING a fully-dead leg
    (θ=0 — lying down needs no FR leg). ``illegal_contact`` raised to 500 N so a belly rest does not
    terminate; only fell_over@70° stays a live failure.

Plus fixed-θ eval variants (``*_at_{pct}``) at pinned θ for the physics/handoff sweeps. All ADDITIVE — built
on ``go2_stabilize`` + the existing FR-torque θ machinery in go2_broken_leg; no existing task is modified.
"""

from __future__ import annotations

import functools

from ..registry import REACH_AVOID, TaskSpec, register

# Pinned FR-torque percentages for the fixed-θ eval tasks (physics gate + handoff sweep). θ = pct/100;
# includes 0 (fully-dead FR leg) — valid for REST, a stress case for STAND.
_EVAL_PCTS = (100, 80, 60, 50, 40, 20, 10, 0)


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import (
    go2_leg_stand_env_cfg,
    go2_leg_rest_env_cfg,
    stance_margins,
    leg_rest_margins,
    LEG_STAND_LO, LEG_STAND_HI, LEG_REST_LO, LEG_REST_HI,
  )
  # ── STAND mode: FR torque θ∈[0.5,1.0] randomized per-episode, θ exposed; stance target. ──
  register(TaskSpec(
    task_id="go2_leg_stand", cfg_builder=go2_leg_stand_env_cfg,
    margin_fn=stance_margins, mode=REACH_AVOID, supports_adversary=True,
    description=f"[leg-family STAND: FR torque θ∈[{LEG_STAND_LO:.1f},{LEG_STAND_HI:.1f}] rand per-episode, "
                "θ exposed to actor+critic; stance target — hold a stand despite the weak leg + adversary]"))
  # ── REST mode: FR torque θ∈[0.0,1.0] randomized per-episode (incl. fully dead), θ exposed; soft-rest. ──
  register(TaskSpec(
    task_id="go2_leg_rest", cfg_builder=go2_leg_rest_env_cfg,
    margin_fn=leg_rest_margins, mode=REACH_AVOID, supports_adversary=True,
    description=f"[leg-family REST: FR torque θ∈[{LEG_REST_LO:.1f},{LEG_REST_HI:.1f}] rand per-episode "
                "(incl. dead), θ exposed; safety=no-slam(<80N), target=low+level+settled. Sit softly?]"))
  # ── FIXED-θ eval tasks (θ pinned via lo=hi=θ): give each conditioned mode its training obs surface at a
  # single test torque, for the physics gate / handoff sweep (mirrors the weight arms' ``*_at_{W}``). ──
  for _pct in _EVAL_PCTS:
    _theta = _pct / 100.0
    register(TaskSpec(
      task_id=f"go2_leg_stand_at_{_pct}",
      cfg_builder=functools.partial(go2_leg_stand_env_cfg, lo=_theta, hi=_theta),
      margin_fn=stance_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[leg-family STAND eval @ fixed FR-torque θ={_theta:.2f} ({_pct}%)]"))
    register(TaskSpec(
      task_id=f"go2_leg_rest_at_{_pct}",
      cfg_builder=functools.partial(go2_leg_rest_env_cfg, lo=_theta, hi=_theta),
      margin_fn=leg_rest_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[leg-family REST eval @ fixed FR-torque θ={_theta:.2f} ({_pct}%)]"))
