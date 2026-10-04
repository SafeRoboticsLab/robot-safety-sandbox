"""WEIGHT-LADDER two-mode ODD tasks (HIGH-CoM): Go2 under a growing carried load W at a high centre of mass.

The carried load is a high-CoM inverted pendulum (LOAD_HEIGHT lever), so once the body tilts the load torques
it further over — the destabilizing load physics. Two conditioned reach-avoid modes, each carrying a per-
episode downward load W (Newtons):
  * go2_weight_stand_hi — STAND spec (``stance_margins``): hold a stable stand despite the load + adversary.
    W ∈ [0, 150] (the regime where standing is plausibly feasible).
  * go2_weight_rest_hi  — REST spec (``weight_rest_margins``): come down softly to a low level belly rest
    WITHOUT slamming, with a LOAD-CONDITIONED slam cap. ``illegal_contact`` raised to 500 N so a belly
    rest under load does not terminate. W ∈ [0, 300] (includes loads where only a rest is feasible).

Plus two UNIFIED single-policy baselines (``go2_weight_unified_hi`` / ``_disc_hi``) — one certificate that
must cover BOTH stand and rest across the whole ladder without a handoff — and fixed-W eval variants
(``*_hi_at_{W}``) at pinned W for the handoff sweeps. All ADDITIVE — built on ``go2_stabilize`` + the
``base.py`` load channel; no existing task is modified.
"""

from __future__ import annotations

import functools

from ..registry import REACH_AVOID, TaskSpec, register

# Pinned loads (N) for the fixed-W eval tasks (handoff sweep).
_EVAL_WS = (0, 60, 90, 120, 150, 200, 250)


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_weight_ladder.env_cfg import (
    go2_weight_stand_env_cfg,
    go2_weight_rest_env_cfg,
    go2_weight_unified_env_cfg,
    stance_margins,
    weight_rest_margins,
    weight_unified_margins,
    STAND_LO, STAND_HI, REST_LO, REST_HI, LOAD_HEIGHT,
  )
  # ══════════════════════════════════════════════════════════════════════════════════════════════
  # HIGH-CoM (inverted-pendulum) variants: the LOAD_HEIGHT lever torques the body further over once tilted.
  # Plus the two UNIFIED single-policy baselines the spec-family is compared against.
  # ══════════════════════════════════════════════════════════════════════════════════════════════
  _stand_hi = functools.partial(go2_weight_stand_env_cfg, load_height=LOAD_HEIGHT)
  _rest_hi = functools.partial(go2_weight_rest_env_cfg, load_height=LOAD_HEIGHT)
  _unified_hi = functools.partial(go2_weight_unified_env_cfg, load_height=LOAD_HEIGHT)
  # ── STAND-hi: stance target, W∈[0,150], HIGH-CoM. ──
  register(TaskSpec(
    task_id="go2_weight_stand_hi", cfg_builder=_stand_hi,
    margin_fn=stance_margins, mode=REACH_AVOID, supports_adversary=True,
    description=f"[weight-ladder STAND, HIGH-CoM (h={LOAD_HEIGHT}): carried load W∈[{STAND_LO:.0f},"
                f"{STAND_HI:.0f}]N torques the body once tilted; stance target — hold a stand]"))
  # ── REST-hi: load-conditioned soft-rest, W∈[0,300], HIGH-CoM. ──
  register(TaskSpec(
    task_id="go2_weight_rest_hi", cfg_builder=_rest_hi,
    margin_fn=weight_rest_margins, mode=REACH_AVOID, supports_adversary=True,
    description=f"[weight-ladder REST, HIGH-CoM (h={LOAD_HEIGHT}): carried load W∈[{REST_LO:.0f},"
                f"{REST_HI:.0f}]N; safety=no-slam(<80+1.3W N), target=low+level+settled]"))
  # ── UNIFIED-hi: single-policy baseline, l=max(l_stand,l_rest), W∈[0,300], HIGH-CoM. ──
  register(TaskSpec(
    task_id="go2_weight_unified_hi", cfg_builder=_unified_hi,
    margin_fn=weight_unified_margins, mode=REACH_AVOID, supports_adversary=True,
    description=f"[weight-ladder UNIFIED baseline, HIGH-CoM (h={LOAD_HEIGHT}): W∈[{REST_LO:.0f},"
                f"{REST_HI:.0f}]N; one policy, l=max(l_stand,l_rest) — stand OR rest, no handoff]"))
  # ── UNIFIED-DISC-hi: unified with rest_discount=0.5 (prefer standing while feasible). ──
  register(TaskSpec(
    task_id="go2_weight_unified_disc_hi", cfg_builder=_unified_hi,
    margin_fn=functools.partial(weight_unified_margins, rest_discount=0.5),
    mode=REACH_AVOID, supports_adversary=True,
    description=f"[weight-ladder UNIFIED baseline (rest_discount=0.5), HIGH-CoM (h={LOAD_HEIGHT}): "
                f"l=max(l_stand, l_rest−0.5) — the rest target is discounted so standing is preferred]"))
  # ── FIXED-W eval variants for the hi arms (partial lo=hi; LOAD_HEIGHT lever kept). ──
  for _w in _EVAL_WS:
    register(TaskSpec(
      task_id=f"go2_weight_stand_hi_at_{_w}",
      cfg_builder=functools.partial(go2_weight_stand_env_cfg, lo=float(_w), hi=float(_w), load_height=LOAD_HEIGHT),
      margin_fn=stance_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[weight-ladder STAND-hi eval @ fixed load W={_w}N]"))
    register(TaskSpec(
      task_id=f"go2_weight_rest_hi_at_{_w}",
      cfg_builder=functools.partial(go2_weight_rest_env_cfg, lo=float(_w), hi=float(_w), load_height=LOAD_HEIGHT),
      margin_fn=weight_rest_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[weight-ladder REST-hi eval @ fixed load W={_w}N]"))
    register(TaskSpec(
      task_id=f"go2_weight_unified_hi_at_{_w}",
      cfg_builder=functools.partial(go2_weight_unified_env_cfg, lo=float(_w), hi=float(_w), load_height=LOAD_HEIGHT),
      margin_fn=weight_unified_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[weight-ladder UNIFIED-hi eval @ fixed load W={_w}N]"))
