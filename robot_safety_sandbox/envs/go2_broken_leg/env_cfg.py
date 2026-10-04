"""Go2 with a degrading FRONT-RIGHT leg — the FR-torque ODD machinery and the leg-family spec.

The ODD parameter is θ = the FR leg's allowable-torque fraction (1.0 = nominal … 0.0 = dead), applied as a
per-world scaling of the three FR actuators' ``actuator_forcerange`` (the force-limited affine position
servo's saturation limit). ``data.ctrl`` is still written for the FR actuators (the 12-D action space is
preserved) and the FR joints stay readable; only the peak torque the FR leg can deliver is reduced. Only the
robot dynamics change; nothing else in the task moves.

This module provides:

  * the FR-torque machinery — resolve/cache the FR actuators' nominal forcerange, randomize θ per-episode
    (``randomize_fr_torque``), and expose θ to the policy (``fr_torque_theta`` / ``_add_fr_conditioning_obs``);
  * the LEG-FAMILY two-mode spec — as θ falls, the strongest feasible safety spec changes. While θ is
    comfortably high the robot can hold a normal STAND (``stance_margins``, θ ∈ [0.5, 1.0]); once the FR leg
    is too weak, standing is infeasible and the strongest thing left is to come down softly to a LOW LEVEL
    belly rest WITHOUT slamming (``leg_rest_margins``, θ ∈ [0.0, 1.0] including a fully-dead leg — lying down
    needs no FR leg). The certified handoff between the modes is a later layer.

Everything is ADDITIVE on top of ``go2_stabilize`` + the per-episode θ machinery; no existing task or margin
is modified.
"""

from __future__ import annotations

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.observation_manager import ObservationTermCfg

from robot_safety_sandbox.envs.go2_stabilize.env_cfg import (  # reused verbatim
  go2_stabilize_env_cfg,
  stance_margins,
)

# The FR leg's 3 actuators, named by the joints they drive. Degrading these three motors'
# allowable torque => the weak front-right leg. (Actuator names carry the ``robot/`` scene
# prefix, matching the entity name in the mjlab scene.)
FR_ACTUATOR_NAMES = (
  "robot/FR_hip_joint",
  "robot/FR_thigh_joint",
  "robot/FR_calf_joint",
)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# PER-EPISODE θ RANDOMIZATION — the FR-leg-degradation ODD axis.
# ══════════════════════════════════════════════════════════════════════════════════════════════
# The ODD parameter θ = the FR leg's allowable-torque fraction, θ ∈ [FR_TORQUE_LO, FR_TORQUE_HI],
# sampled FRESH PER-EPISODE at reset (a continuum, not a fixed per-env specialist).
FR_TORQUE_LO, FR_TORQUE_HI = 0.2, 1.0   # θ span: 0.2 = badly weakened FR leg … 1.0 = nominal


def _resolve_fr_act_ids(env, actuator_names: tuple[str, ...] = FR_ACTUATOR_NAMES) -> torch.Tensor:
  """Global model column indices of the FR leg's 3 actuators, resolved by NAME (robust to model layout) so
  every FR-torque event scales exactly the same three motors."""
  m = env.sim.mj_model
  raw = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
  ids = torch.tensor([i for i in raw if i >= 0], device=env.device, dtype=torch.long)
  assert len(ids) == len(actuator_names), (
    f"weak-leg(randomized): expected {len(actuator_names)} FR actuators, resolved {len(ids)} "
    f"from {actuator_names} (got ids {raw})")
  return ids


def _ensure_fr_cache(env, actuator_names: tuple[str, ...] = FR_ACTUATOR_NAMES) -> None:
  """Lazily resolve+cache the FR actuator ids, snapshot the NOMINAL forcerange, and allocate the per-env
  θ store. Idempotent — safe to call from the startup event OR lazily from the reset event. The nominal
  snapshot is taken ONCE, before any per-episode scaling, so the reset event can always write ABSOLUTELY
  from nominal (no compounding across resets)."""
  if getattr(env, "_fr_act_ids", None) is None:
    env._fr_act_ids = _resolve_fr_act_ids(env, actuator_names)
    # forcerange is [nworld, nu, 2] = [-effort_limit, +effort_limit]; cache the FR rows [nworld, 3, 2].
    env._fr_nominal_forcerange = env.sim.model.actuator_forcerange[:, env._fr_act_ids, :].clone()
    nworld = env.sim.model.actuator_forcerange.shape[0]
    env._fr_torque_frac = torch.ones(nworld, device=env.device)   # default θ=1.0 until first reset


@requires_model_fields("actuator_forcerange")
def cache_fr_nominal_forcerange(
  env,
  env_ids: torch.Tensor | None = None,
  actuator_names: tuple[str, ...] = FR_ACTUATOR_NAMES,
) -> None:
  """Startup event: resolve the FR actuator ids once, snapshot their nominal ``actuator_forcerange``, and
  allocate ``env._fr_torque_frac`` ([nworld], default 1.0). The ``@requires_model_fields`` decorator makes
  ``sim.expand_model_fields()`` allocate real per-world memory for ``actuator_forcerange`` (needed because
  the randomized env carries no fixed degrade event to trigger that expansion)."""
  _ensure_fr_cache(env, actuator_names)


@requires_model_fields("actuator_forcerange")
def randomize_fr_torque(
  env,
  env_ids: torch.Tensor | None,
  lo: float = FR_TORQUE_LO,
  hi: float = FR_TORQUE_HI,
  actuator_names: tuple[str, ...] = FR_ACTUATOR_NAMES,
) -> None:
  """Reset event: sample a FRESH per-env θ ~ U[lo, hi] for the reset envs, write the FR actuators'
  ``actuator_forcerange`` ABSOLUTELY from the cached nominal (``nominal * θ`` — NEVER a multiply of the
  current value, which would compound across resets and drift the leg toward dead), and store θ on
  ``env._fr_torque_frac`` for the conditioning obs to read back. Triggers the startup cache lazily if the
  reset fires first."""
  _ensure_fr_cache(env, actuator_names)
  ids = env._fr_act_ids
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)
  # θ ~ U[lo, hi] per reset env (plain torch RNG on device — consistent with dr.* which samples the same way).
  frac = torch.rand(len(env_ids), device=env.device) * (hi - lo) + lo
  # ABSOLUTE write from cached nominal ⇒ no compounding across resets.
  env.sim.model.actuator_forcerange[env_ids[:, None], ids, :] = (
    env._fr_nominal_forcerange[env_ids] * frac[:, None, None])
  env._fr_torque_frac[env_ids] = frac


# ── θ CONDITIONING observation (the oracle signal for the conditioned modes) ────────────────────
def fr_torque_theta(env) -> torch.Tensor:
  """Per-env ODD θ = the FR-leg torque fraction ∈ [lo, hi]. Shape (num_envs, 1). Read straight back from
  ``env._fr_torque_frac`` (the value the reset event committed this episode); ones before it is allocated."""
  frac = getattr(env, "_fr_torque_frac", None)
  if frac is None:
    return torch.ones(env.num_envs, 1, device=env.device)
  return frac.reshape(-1, 1)


def _add_fr_conditioning_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Expose θ to BOTH the actor (adapt strategy per-θ) and the critic (θ-correct value). Separate
  ObservationTermCfg per group (the manager owns per-term state)."""
  for group in ("actor", "critic"):
    cfg.observations[group].terms["fr_torque_theta"] = ObservationTermCfg(func=fr_torque_theta)


def go2_weak_leg_randomized_env_cfg(
  play: bool = False, lo: float = FR_TORQUE_LO, hi: float = FR_TORQUE_HI,
) -> ManagerBasedRlEnvCfg:
  """Base randomized-θ env: ``go2_stabilize`` with the FR leg's allowable torque θ ~ U[lo, hi] randomized
  PER-EPISODE (fresh θ each reset), θ NOT exposed. The startup-cache + reset-randomize events provide the θ
  channel. Everything else (margins, adversary, obs/reward/spawn) is inherited unchanged from
  ``go2_stabilize_env_cfg``. The leg-family builders below extend this env."""
  cfg = go2_stabilize_env_cfg(play=play)
  cfg.events["cache_fr_nominal"] = EventTermCfg(func=cache_fr_nominal_forcerange, mode="startup", params={})
  cfg.events["randomize_fr_torque"] = EventTermCfg(
    func=randomize_fr_torque, mode="reset", params={"lo": lo, "hi": hi})
  return cfg


# ══════════════════════════════════════════════════════════════════════════════════════════════
# LEG-FAMILY two-mode spec — the weight-ladder template on the FR-torque θ axis.
# ══════════════════════════════════════════════════════════════════════════════════════════════
# As θ falls the strongest feasible safety spec changes: while θ is comfortably high the robot can hold a
# normal STAND (``stance_margins``, θ~U[0.5,1.0] — the comfortably-feasible band); once the FR leg is too weak,
# standing is infeasible and the strongest thing left is to come down softly to a LOW LEVEL belly rest WITHOUT
# slamming (``leg_rest_margins``, θ~U[0.0,1.0] INCLUDING a fully-dead leg — lying down needs no FR leg). This is
# the leg-ODD analog of go2_weight_{stand,rest}; the certified handoff between the modes is a later layer.
LEG_STAND_LO, LEG_STAND_HI = 0.5, 1.0       # STAND mode θ span: comfortably-feasible stance band
LEG_REST_LO, LEG_REST_HI = 0.0, 1.0         # REST mode θ span: includes a fully-dead FR leg (θ=0)

# No-slam safety + LOW/level/settled rest-target constants (shared by ``leg_rest_margins``).
#   SAFETY g = don't SLAM: max non-foot ground-contact force < SLAM_N. A gentle belly settle measures tens of
#     N; a hard impact is hundreds–thousands of N. 80 N admits a soft belly rest while still flagging a crash.
#   TARGET l = LOW + LEVEL + SETTLED: base low (belly down), small roll/pitch (level), small linear & angular
#     velocity (settled) — satisfiable only by a low still level held pose.
SLAM_N, SLAM_SCALE = 80.0, 80.0     # non-foot contact safe set: < 80 N (soft rest ok) safe, ≥ 80 N (slam) fail
RP_TOL, RP_SCALE = 0.25, 0.25       # target: max(|roll|,|pitch|) small (~14°) — LEVEL
V_TOL, V_SCALE = 0.30, 0.30         # target: |base linear vel| small — SETTLED (not drifting)
W_TOL, W_SCALE = 0.50, 0.50         # target: |base angular vel| small — SETTLED (not tumbling)
LEG_H_LIE, LEG_HS = 0.15, 0.06      # target: base LOW (base_z < H_LIE ⇒ (H_LIE-base_z)/HS > 0)

# REST-mode illegal_contact threshold: a HIGH constant so a belly rest under a dead leg does NOT terminate —
# the reach-avoid g (constant 80 N no-slam cap) does the safety scoring; only fell_over@70° stays a live
# failure. Mirrors go2_weight_ladder.REST_CONTACT_N.
LEG_REST_CONTACT_N = 500.0


def _raise_contact_termination(cfg: ManagerBasedRlEnvCfg, force_threshold: float) -> ManagerBasedRlEnvCfg:
  """Raise the inherited ``illegal_contact`` termination threshold (default 10 N) to ``force_threshold`` so a
  belly rest below it does NOT end the episode — only a slam does. (fell_over@70° still guards against tipping
  fully over.)"""
  t = cfg.terminations.get("illegal_contact")
  if t is not None:
    t.params = {**(dict(t.params) if t.params else {}), "force_threshold": float(force_threshold)}
  return cfg


def leg_rest_margins(env):
  """(g, l): CONSTANT no-slam safety (< SLAM_N = 80 N) + LOW/level/settled belly-rest target. Reads the
  ``nonfoot_ground_touch`` contact sensor. g = (SLAM_N − force)/SLAM_SCALE; l = min of (low, level, settled-lin,
  settled-ang) — satisfiable only by a LOW still level pose (a normal STAND at base_z≈0.32 ≫ H_LIE ⇒ l<0, so
  standing is off-target here). Distinct from the load-conditioned ``weight_rest_margins`` by the CONSTANT cap
  (there is no carried load here, so the belly rests at ~robot weight only)."""
  d = env.scene["robot"].data
  s = env.scene["nonfoot_ground_touch"]
  fh = s.data.force_history if s.data.force_history is not None else s.data.force
  force = torch.norm(fh, dim=-1).flatten(1).amax(1)          # per-env max non-foot ground-contact force
  pg = d.projected_gravity_b                                 # gravity in body frame; (x,y)→0 when level
  tilt = torch.maximum(pg[:, 0].abs(), pg[:, 1].abs())       # ~ max(|pitch|,|roll|) sines
  v = torch.linalg.norm(d.root_link_lin_vel_b, dim=1)        # base linear speed
  w = torch.linalg.norm(d.root_link_ang_vel_b, dim=1)        # base angular speed
  base_z = d.root_link_pos_w[:, 2]
  g = (SLAM_N - force) / SLAM_SCALE                           # SAFE: contact < 80 N (soft belly rest ok)
  l = torch.stack([                                          # TARGET: LOW and LEVEL and SETTLED
    (LEG_H_LIE - base_z) / LEG_HS,                            #   low (belly down)
    (RP_TOL - tilt) / RP_SCALE,                               #   level
    (V_TOL - v) / V_SCALE,                                    #   settled (linear)
    (W_TOL - w) / W_SCALE,                                    #   settled (angular)
  ], dim=0).amin(dim=0)
  return g, l


def go2_leg_stand_env_cfg(
  play: bool = False, lo: float = LEG_STAND_LO, hi: float = LEG_STAND_HI,
) -> ManagerBasedRlEnvCfg:
  """LEG-FAMILY STAND mode: ``go2_stabilize`` + FR-leg torque θ ~ U[lo, hi] randomized per-episode + θ
  exposed to actor+critic. Margins = ``stance_margins`` (unchanged): hold a stable stand despite the weak leg
  and the adversary. θ span defaults to the comfortably-feasible band [0.5, 1.0]. Everything else (adversary,
  obs/reward/spawn, the cache-nominal startup + reset randomize events) inherited from the randomized env."""
  cfg = go2_weak_leg_randomized_env_cfg(play=play, lo=lo, hi=hi)
  _add_fr_conditioning_obs(cfg)
  return cfg


def go2_leg_rest_env_cfg(
  play: bool = False, lo: float = LEG_REST_LO, hi: float = LEG_REST_HI,
) -> ManagerBasedRlEnvCfg:
  """LEG-FAMILY REST mode: ``go2_stabilize`` + FR-leg torque θ ~ U[lo, hi] randomized per-episode (span
  defaults to [0.0, 1.0], INCLUDING a fully-dead leg) + θ exposed to actor+critic + ``illegal_contact`` raised
  to LEG_REST_CONTACT_N so a belly rest under a dead leg does not terminate. Margins = ``leg_rest_margins``
  (constant no-slam + LOW/level/settled): come down softly to a low level rest when standing is infeasible."""
  cfg = go2_weak_leg_randomized_env_cfg(play=play, lo=lo, hi=hi)
  _add_fr_conditioning_obs(cfg)
  _raise_contact_termination(cfg, LEG_REST_CONTACT_N)
  return cfg


__all__ = ["stance_margins", "FR_ACTUATOR_NAMES",
           "go2_weak_leg_randomized_env_cfg", "cache_fr_nominal_forcerange", "randomize_fr_torque",
           "fr_torque_theta", "FR_TORQUE_LO", "FR_TORQUE_HI",
           "leg_rest_margins", "go2_leg_stand_env_cfg", "go2_leg_rest_env_cfg",
           "LEG_STAND_LO", "LEG_STAND_HI", "LEG_REST_LO", "LEG_REST_HI", "LEG_REST_CONTACT_N"]
