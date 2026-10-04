"""Go2 carrying a growing downward LOAD W — the WEIGHT-LADDER ODD.

The ODD is a per-episode carried weight W (Newtons), applied as an exogenous DOWNWARD base wrench
[0, 0, -W]. As W grows, the STRONGEST feasible safety specification changes: while W is small the robot
can hold a normal STAND (stance_margins); once W exceeds what the legs can hold, standing is infeasible
and the strongest thing left is to come down SOFTLY to a low level belly rest WITHOUT slamming
(weight_rest_margins, with a LOAD-CONDITIONED slam cap since a heavier payload rests heavier). This module
trains the two conditioned modes V_stand(x, W) and V_rest(x, W); the certified handoff between them
is a later layer. It is ADDITIVE — everything is built on top of ``go2_stabilize_env_cfg`` and the
existing weight-load channel in ``base.py``; no existing task or margin is modified.

LOAD CHANNEL (two sources, eval override taking precedence — see ``base.MjlabTensorSafetyEnv._apply_dstb``):
  * TRAINING: the ``randomize_base_load`` reset event samples a per-env W ~ U[lo, hi] and stores it on the
    INNER mjlab env (``env._weight_W``). ``_apply_dstb`` reads it off ``self.mj`` and adds [0,0,-W] to the
    adversary wrench each step. It is a PLAIN STORE (no compounding — a fresh W is written each reset).
  * EVAL: set ``env.base_load`` ([N,3] tensor) directly on the OUTER bridge env each step;
    it takes precedence over the inner ``_weight_W``.
The W conditioning obs ``weight_theta`` (W / 150, appended to actor+critic) mirrors the leg-ODD
``fr_torque_theta`` — the oracle signal for the conditioned modes.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from robot_safety_sandbox.envs.go2_stabilize.env_cfg import (  # reused verbatim
  go2_stabilize_env_cfg,
  stance_margins,
)

# ── ODD distribution: per-episode carried load W (Newtons), applied as a downward base wrench [0,0,-W] ──
# STAND mode trains over W ∈ [0, 150] (the regime where a normal stand is plausibly feasible); REST mode
# over W ∈ [0, 300] (includes the heavy loads where only a soft belly rest is left). W is normalized by
# WEIGHT_NORM in the conditioning obs (so W ∈ [0, 300] → ~[0, 2]).
STAND_LO, STAND_HI = 0.0, 150.0
REST_LO, REST_HI = 0.0, 300.0
WEIGHT_NORM = 150.0                 # obs normalization: weight_theta = W / WEIGHT_NORM (~[0, 2])

# HIGH-CoM load height (m): the height above the base at which the carried load hangs. A nonzero height turns
# the pure downward wrench into an INVERTED PENDULUM — once tilted, the load torques the body further over
# (see base.py _apply_dstb). A load height of 0.0 is a pure force with no lever (bit-identical to a plain
# downward wrench); the ``*_hi`` tasks use LOAD_HEIGHT to engage the inverted-pendulum torque.
LOAD_HEIGHT = 0.25


# ══════════════════════════════════════════════════════════════════════════════════════════════
# PER-EPISODE LOAD RANDOMIZATION + W CONDITIONING OBS (mirrors go2_broken_leg's θ machinery)
# ══════════════════════════════════════════════════════════════════════════════════════════════
def randomize_base_load(
  env,
  env_ids: torch.Tensor | None,
  lo: float = STAND_LO,
  hi: float = STAND_HI,
  load_height: float = 0.0,
) -> None:
  """Reset event: sample a FRESH per-env carried load W ~ U[lo, hi] (Newtons) for the reset envs and store
  it on the INNER mjlab env as ``env._weight_W`` ([num_envs]). ``base.py``'s ``_apply_dstb`` reads this off
  ``self.mj`` each step and ADDS the downward wrench [0, 0, -W] to the adversary force. A PLAIN STORE — a
  fresh W is written each reset, so there is no compounding across episodes (unlike a multiplicative model
  edit). Lazily allocates the store on first call (mirrors ``randomize_fr_torque``'s lazy θ store).

  ``load_height`` (m) sets the HIGH-CoM lever: it is committed to ``env._weight_h`` ([num_envs]), which
  ``_apply_dstb`` reads to add the inverted-pendulum torque τ = R(quat)·[0,0,h] × [0,0,-W]. ``load_height=0``
  ⇒ ``_weight_h`` = 0 ⇒ zero torque ⇒ a PURE-FORCE load (bit-identical to a plain downward wrench)."""
  if getattr(env, "_weight_W", None) is None:
    env._weight_W = torch.zeros(env.num_envs, device=env.device)
  if getattr(env, "_weight_h", None) is None:
    env._weight_h = torch.zeros(env.num_envs, device=env.device)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)
  W = torch.rand(len(env_ids), device=env.device) * (hi - lo) + lo
  env._weight_W[env_ids] = W
  env._weight_h[env_ids] = float(load_height)


def weight_theta(env) -> torch.Tensor:
  """Per-env ODD θ = carried load W, normalized by ``WEIGHT_NORM``. Shape (num_envs, 1). Read straight back
  from ``env._weight_W`` (the value the reset event committed this episode); zeros before it is allocated.
  Mirrors ``fr_torque_theta``."""
  W = getattr(env, "_weight_W", None)
  if W is None:
    return torch.zeros(env.num_envs, 1, device=env.device)
  return W.reshape(-1, 1) / WEIGHT_NORM


def _add_weight_conditioning_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Expose W to BOTH the actor (adapt strategy per-load) and the critic (W-correct value). Separate
  ObservationTermCfg per group (the manager owns per-term state) — mirrors ``_add_fr_conditioning_obs``."""
  for group in ("actor", "critic"):
    cfg.observations[group].terms["weight_theta"] = ObservationTermCfg(func=weight_theta)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# WEIGHT-CONDITIONED SOFT-REST reach-avoid margins — the strongest spec once standing is infeasible.
# ══════════════════════════════════════════════════════════════════════════════════════════════
# SAFETY g = don't SLAM: max non-foot ground-contact force < SLAM_CAP(W). The cap is LOAD-CONDITIONED —
#   a heavier carried payload rests HEAVIER (the static belly contact ≈ mg_robot + W), so a fixed 80 N cap
#   (as in the leg soft-rest) would flag a gentle rest under load as a slam. SLAM_CAP(W) = 80 N base
#   allowance + 1.3·W static scaling (constants exposed for calibration against the measured static force).
# TARGET l = LOW + LEVEL + SETTLED: base low (belly down, base_z < H_LIE), small roll/pitch (level), small
#   linear & angular velocity (settled). A normal STAND has base_z ≈ 0.32 ≫ H_LIE ⇒ l < 0 (not on-target),
#   so "gracefully come down and settle low" is the target here, distinct from the STAND spec.
SLAM_CAP_BASE, SLAM_CAP_SLOPE = 80.0, 1.3   # SLAM_CAP(W) = 80 + 1.3·W  (N); calibrate against static rest force
SLAM_REST_SCALE = 80.0                      # g normalization scale
H_LIE, HS = 0.15, 0.06                      # target: base LOW (base_z < H_LIE ⇒ (H_LIE-base_z)/HS > 0)
RP_TOL, RP_SCALE = 0.25, 0.25               # target: max(|roll|,|pitch|) sines small — LEVEL
V_TOL, VS = 0.30, 0.30                      # target: |base lin vel| small — SETTLED
W_TOL, WS = 0.50, 0.50                      # target: |base ang vel| small — SETTLED

# REST-mode illegal_contact termination threshold: a HIGH constant above SLAM_CAP(300)=470 N, so a belly
# rest UNDER LOAD does not terminate — the reach-avoid g (load-conditioned cap) does the safety scoring.
REST_CONTACT_N = 500.0


def _raise_contact_termination(cfg: ManagerBasedRlEnvCfg, force_threshold: float) -> ManagerBasedRlEnvCfg:
  """Raise the inherited ``illegal_contact`` termination threshold (default 10 N) to ``force_threshold`` so a
  belly rest below it does NOT end the episode. Generalizes go2_broken_leg's ``_soften_contact_termination``
  (which pins the leg soft-rest's 80 N SLAM_N) to an arbitrary per-mode threshold."""
  t = cfg.terminations.get("illegal_contact")
  if t is not None:
    t.params = {**(dict(t.params) if t.params else {}), "force_threshold": float(force_threshold)}
  return cfg


def weight_rest_margins(env):
  """(g, l): LOAD-CONDITIONED no-slam safety + LOW/level/settled belly-rest target. Reads the
  ``nonfoot_ground_touch`` contact sensor (as ``soft_rest_margins`` does) and the per-env load W from
  ``env._weight_W`` (zeros if unset). g = (SLAM_CAP(W) − force)/scale with SLAM_CAP(W) = 80 + 1.3·W; l is
  the min of (low, level, settled-lin, settled-ang) — satisfiable only by a low still level pose."""
  d = env.scene["robot"].data
  s = env.scene["nonfoot_ground_touch"]
  fh = s.data.force_history if s.data.force_history is not None else s.data.force
  force = torch.norm(fh, dim=-1).flatten(1).amax(1)          # per-env max non-foot ground-contact force
  pg = d.projected_gravity_b                                 # gravity in body frame; (x,y)→0 when level
  tilt = torch.maximum(pg[:, 0].abs(), pg[:, 1].abs())       # ~ max(|pitch|,|roll|) sines
  v = torch.linalg.norm(d.root_link_lin_vel_b, dim=1)        # base linear speed
  w = torch.linalg.norm(d.root_link_ang_vel_b, dim=1)        # base angular speed
  base_z = d.root_link_pos_w[:, 2]
  W = getattr(env, "_weight_W", None)
  if W is None:
    W = torch.zeros(env.num_envs, device=env.device)
  slam_cap = SLAM_CAP_BASE + SLAM_CAP_SLOPE * W              # LOAD-CONDITIONED slam cap (N)
  g = (slam_cap - force) / SLAM_REST_SCALE                   # SAFE: contact < SLAM_CAP(W)
  l = torch.stack([                                          # TARGET: LOW and LEVEL and SETTLED
    (H_LIE - base_z) / HS,                                   #   low (belly down)
    (RP_TOL - tilt) / RP_SCALE,                              #   level
    (V_TOL - v) / VS,                                        #   settled (linear)
    (W_TOL - w) / WS,                                        #   settled (angular)
  ], dim=0).amin(dim=0)
  return g, l


def weight_unified_margins(env, rest_discount: float = 0.0):
  """(g, l) for the SINGLE-POLICY UNIFIED baseline — one certificate that must cover BOTH the strong (stand)
  and weak (rest) specs across the whole ladder, WITHOUT the certified handoff. It is the fair baseline the
  spec-FAMILY is compared against: a monolithic policy can only certify max over the two targets everywhere.

    g = the load-conditioned no-slam SAFETY of ``weight_rest_margins`` (a belly rest under load must not
        slam; the stand spec's corner-height g is subsumed — standing keeps corners well clear of the cap).
    l = max(l_stand, l_rest): ON-TARGET if EITHER a stable stand (``stance_margins``' stance-band l) OR a
        low/level/settled belly rest (``weight_rest_margins``' l) is achieved. The union of the two targets —
        so the unified policy is free to pick whichever the load permits, but from ONE value function.

  ``rest_discount`` (≥0) DISCOUNTS the rest branch only: l = max(l_stand, l_rest − rest_discount). It makes
  the low-rest target strictly less rewarding than a genuine stand, so the unified policy prefers to stand
  while feasible and only concedes to resting when standing is truly infeasible (a soft handoff-preference
  prior baked into the single objective, contrasting the EXPLICIT certified handoff of the spec family)."""
  g, l_rest = weight_rest_margins(env)           # load-conditioned no-slam g + low/level/settled rest l
  _, l_stand = stance_margins(env)               # stance-band l (corners in band, settled) — discard its g
  l = torch.maximum(l_stand, l_rest - rest_discount)
  return g, l


# ══════════════════════════════════════════════════════════════════════════════════════════════
# ENV BUILDERS: the two conditioned modes (STAND and REST), each carrying a per-episode load W.
# ══════════════════════════════════════════════════════════════════════════════════════════════
def go2_weight_stand_env_cfg(
  play: bool = False, lo: float = STAND_LO, hi: float = STAND_HI, load_height: float = 0.0,
) -> ManagerBasedRlEnvCfg:
  """STAND mode: ``go2_stabilize`` + a per-episode carried load W ~ U[lo, hi] (applied downward) + W exposed
  to actor+critic. Margins = ``stance_margins`` (unchanged): return to a stable stand despite the load and
  the adversary. Everything else (adversary, obs/reward/spawn) inherited from ``go2_stabilize_env_cfg``.
  ``load_height`` (m) sets the HIGH-CoM lever (0 = pure force; LOAD_HEIGHT = inverted pendulum)."""
  cfg = go2_stabilize_env_cfg(play=play)
  cfg.events["randomize_base_load"] = EventTermCfg(
    func=randomize_base_load, mode="reset", params={"lo": lo, "hi": hi, "load_height": load_height})
  _add_weight_conditioning_obs(cfg)
  return cfg


def go2_weight_rest_env_cfg(
  play: bool = False, lo: float = REST_LO, hi: float = REST_HI, load_height: float = 0.0,
) -> ManagerBasedRlEnvCfg:
  """REST mode: ``go2_stabilize`` + a per-episode carried load W ~ U[lo, hi] (heavier range) + W exposed to
  actor+critic + the ``illegal_contact`` termination raised to REST_CONTACT_N (so a belly rest under load
  does not terminate). Margins = ``weight_rest_margins`` (LOAD-CONDITIONED no-slam + low/level/settled): come
  down softly to a low level rest when standing is infeasible. ``load_height`` (m) = HIGH-CoM lever."""
  cfg = go2_stabilize_env_cfg(play=play)
  cfg.events["randomize_base_load"] = EventTermCfg(
    func=randomize_base_load, mode="reset", params={"lo": lo, "hi": hi, "load_height": load_height})
  _add_weight_conditioning_obs(cfg)
  _raise_contact_termination(cfg, REST_CONTACT_N)
  return cfg


def go2_weight_unified_env_cfg(
  play: bool = False, lo: float = REST_LO, hi: float = REST_HI, load_height: float = 0.0,
) -> ManagerBasedRlEnvCfg:
  """UNIFIED mode: identical env plumbing to REST (``go2_stabilize`` + per-episode load W ~ U[lo, hi] + W
  exposed + ``illegal_contact`` raised to REST_CONTACT_N so a belly rest under load does not terminate). It
  differs only in the MARGIN it is registered with — ``weight_unified_margins`` (l = max(l_stand, l_rest)),
  the SINGLE-POLICY baseline that must cover BOTH standing and resting across the whole ladder without a
  handoff. ``load_height`` (m) = HIGH-CoM lever."""
  cfg = go2_stabilize_env_cfg(play=play)
  cfg.events["randomize_base_load"] = EventTermCfg(
    func=randomize_base_load, mode="reset", params={"lo": lo, "hi": hi, "load_height": load_height})
  _add_weight_conditioning_obs(cfg)
  _raise_contact_termination(cfg, REST_CONTACT_N)
  return cfg


__all__ = [
  "randomize_base_load", "weight_theta", "weight_rest_margins", "weight_unified_margins", "stance_margins",
  "go2_weight_stand_env_cfg", "go2_weight_rest_env_cfg", "go2_weight_unified_env_cfg",
  "STAND_LO", "STAND_HI", "REST_LO", "REST_HI", "WEIGHT_NORM", "LOAD_HEIGHT",
  "SLAM_CAP_BASE", "SLAM_CAP_SLOPE", "REST_CONTACT_N",
]
