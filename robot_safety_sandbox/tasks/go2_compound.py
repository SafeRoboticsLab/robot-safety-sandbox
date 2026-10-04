"""COMPOUND two-mode ODD tasks: Go2 losing its FR leg WHILE carrying a high-CoM load.

The COMPOSITION of the weight-ladder and leg-degradation modes: a CONSTANT high-CoM load W=80 N @ h=0.25 is
carried every episode, and the FR-leg torque fraction θ is the randomized ODD the two modes condition on.
Unlike the UNLOADED leg-at-stance certificate (essentially flat — the matched negative control), standing on
3.5 legs UNDER the pendulum genuinely fails below θ_c≈0.3, so V_compound_stand(x,θ) contracts → a real
certified leg handoff.

  * go2_compound_stand — STAND spec (``stance_margins``): hold a stand despite the weak leg + the load +
    adversary. θ ∈ [0.4, 1.0] (comfortably-feasible band above the gate's θ_c). θ exposed to actor+critic.
  * go2_compound_rest  — REST spec (``weight_rest_margins``): come down softly to a low level belly rest with
    the LOAD-CONDITIONED slam cap SLAM_CAP(80)=184 N. θ ∈ [0.0, 1.0] INCLUDING a fully-dead leg.
    ``illegal_contact`` raised to 500 N so a belly rest under load does not terminate.

Plus fixed-θ eval variants (``*_at_{pct}``) at pinned θ for the handoff sweep. All ADDITIVE — built on
``go2_stabilize`` + the load channel + the FR-torque θ machinery; no existing task is modified.
"""

from __future__ import annotations

import functools

from ..registry import REACH_AVOID, TaskSpec, register

# Pinned FR-torque percentages for the fixed-θ eval tasks (handoff sweep). θ = pct/100; includes 0 (fully-dead
# FR leg) — valid for REST, the deep-failure stress case for STAND under load.
_EVAL_PCTS = (100, 80, 60, 50, 40, 30, 20, 10, 0)


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_compound.env_cfg import (
    go2_compound_stand_env_cfg,
    go2_compound_rest_env_cfg,
    stance_margins,
    weight_rest_margins,
    COMPOUND_W, COMPOUND_H, STAND_THETA_LO, STAND_THETA_HI, REST_THETA_LO, REST_THETA_HI,
  )
  _LOAD = f"W={COMPOUND_W:.0f}N@h={COMPOUND_H}"
  # ── STAND mode: constant load + FR torque θ∈[0.4,1.0] randomized per-episode, θ exposed; stance target. ──
  register(TaskSpec(
    task_id="go2_compound_stand", cfg_builder=go2_compound_stand_env_cfg,
    margin_fn=stance_margins, mode=REACH_AVOID, supports_adversary=True,
    description=f"[compound STAND ({_LOAD}): FR torque θ∈[{STAND_THETA_LO:.1f},{STAND_THETA_HI:.1f}] rand "
                "per-episode, θ exposed; stance target — hold a stand on 3.5 legs under the pendulum load]"))
  # ── REST mode: constant load + FR torque θ∈[0.0,1.0] (incl. dead), θ exposed; load-conditioned soft-rest. ──
  register(TaskSpec(
    task_id="go2_compound_rest", cfg_builder=go2_compound_rest_env_cfg,
    margin_fn=weight_rest_margins, mode=REACH_AVOID, supports_adversary=True,
    description=f"[compound REST ({_LOAD}): FR torque θ∈[{REST_THETA_LO:.1f},{REST_THETA_HI:.1f}] rand "
                "per-episode (incl. dead), θ exposed; safety=no-slam(<80+1.3W N), target=low+level+settled]"))
  # ── FIXED-θ eval tasks (θ pinned via lo=hi=θ): each mode's training obs surface at a single test torque,
  # for the leg-death handoff sweep (mirrors the leg/weight arms' ``*_at_*``). ──
  for _pct in _EVAL_PCTS:
    _theta = _pct / 100.0
    register(TaskSpec(
      task_id=f"go2_compound_stand_at_{_pct}",
      cfg_builder=functools.partial(go2_compound_stand_env_cfg, lo=_theta, hi=_theta),
      margin_fn=stance_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[compound STAND eval @ fixed FR-torque θ={_theta:.2f} ({_pct}%), {_LOAD}]"))
    register(TaskSpec(
      task_id=f"go2_compound_rest_at_{_pct}",
      cfg_builder=functools.partial(go2_compound_rest_env_cfg, lo=_theta, hi=_theta),
      margin_fn=weight_rest_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[compound REST eval @ fixed FR-torque θ={_theta:.2f} ({_pct}%), {_LOAD}]"))
