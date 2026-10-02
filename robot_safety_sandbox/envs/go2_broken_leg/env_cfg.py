"""Go2 with a WEAK front-right leg — an ODD variant of ``go2_stabilize``.

Identical to ``go2_stabilize`` in every respect — same flat terrain, zero-command
stance target, same ``stance_margins``, same symmetric adversarial base force (the
two-player game), same obs / reward / spawn — EXCEPT the FRONT-RIGHT leg's 3 actuators
have their ALLOWABLE TORQUE degraded to a fraction ``torque_frac`` of nominal (default
0.5 = half torque). A weakened motor, not a removed one:

  * the FR leg is still physically present; its 3 joints (``FR_hip_joint``,
    ``FR_thigh_joint``, ``FR_calf_joint``) are still READABLE in the observation;
  * the action space stays 12-D — the policy still emits FR commands (``data.ctrl`` is
    still written for the FR actuators);
  * the FR actuators keep full stiffness for small corrections but SATURATE at
    ``torque_frac`` of their rated torque — so the leg can still push, just not as hard.

A fully DEAD leg (``torque_frac=0``) makes the "stable stand" target structurally
infeasible: with zero torque under the front-right, that trunk corner sags below the
0.10 m floor even with the adversary off (measured: ~39% of env-steps, plus the limp
limb drags -> illegal_contact). So we degrade rather than kill: ``torque_frac`` is the
continuous physics-limit ODD axis (1.0 = normal ... 0.0 = dead), and a partial value is
winnable + can bifurcate. This also gives the "amount of torque available" axis for the
gradual 4->3-leg dynamic-ODD test directly.

Mechanism (MODEL OVERRIDE — mirrors how the payload env writes per-env model params at
startup): mjlab's ``BuiltinPositionActuator`` is a force-limited affine position servo
(``force = gainprm[0]*ctrl + biasprm[1]*qpos + biasprm[2]*qvel``, clamped to
``actuator_forcerange = [-effort_limit, +effort_limit]``). ``effort_limit`` IS the
allowable torque, so a ``startup`` event scales the FR actuators' ``actuator_forcerange``
rows by ``torque_frac`` in the per-world model. ctrl is still written (12-D preserved),
the joints stay readable, stiffness is unchanged — only the peak torque the FR leg can
deliver is reduced. Only the robot dynamics change; nothing else in the task moves.
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


@requires_model_fields("actuator_forcerange")
def degrade_fr_leg_torque(
  env,
  env_ids: torch.Tensor | None,
  torque_frac: float = 0.5,
  actuator_names: tuple[str, ...] = FR_ACTUATOR_NAMES,
) -> None:
  """Startup event: scale the FR leg actuators' allowable torque (``actuator_forcerange``)
  by ``torque_frac`` in the per-world model. The affine servo law and stiffness are
  unchanged; only the saturation limit — the peak torque the FR motors can deliver — is
  reduced. ``data.ctrl`` is still written for these actuators (12-D action preserved) and
  the FR joints stay readable. Resolves the actuator ids by NAME (robust to model layout)."""
  m = env.sim.mj_model
  raw = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
  ids = torch.tensor([i for i in raw if i >= 0], device=env.device, dtype=torch.long)
  assert len(ids) == len(actuator_names), (
    f"weak-leg: expected {len(actuator_names)} FR actuators, resolved {len(ids)} "
    f"from {actuator_names} (got ids {raw})")
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)
  # forcerange is [nworld, nu, 2] = [-effort_limit, +effort_limit]; scaling both columns
  # by torque_frac degrades the allowable torque symmetrically.
  env.sim.model.actuator_forcerange[env_ids[:, None], ids, :] *= float(torque_frac)


def go2_weak_leg_env_cfg(play: bool = False, torque_frac: float = 0.5) -> ManagerBasedRlEnvCfg:
  """The weak-front-right-leg task: ``go2_stabilize`` with the 3 FR actuators' allowable
  torque scaled to ``torque_frac`` of nominal (default 0.5). Everything else — margins,
  adversary, obs/reward/spawn — is inherited unchanged from ``go2_stabilize_env_cfg``."""
  cfg = go2_stabilize_env_cfg(play=play)
  cfg.events["degrade_fr_leg"] = EventTermCfg(
    func=degrade_fr_leg_torque,
    mode="startup",
    params={"torque_frac": torque_frac},
  )
  return cfg


# ══════════════════════════════════════════════════════════════════════════════════════════════
# PER-EPISODE θ RANDOMIZATION — the leg-degradation ODD distribution (blind/conditioned/history arms)
# ══════════════════════════════════════════════════════════════════════════════════════════════
# The ODD parameter θ = the FR leg's allowable-torque fraction, θ ∈ [FR_TORQUE_LO, FR_TORQUE_HI],
# sampled FRESH PER-EPISODE at reset (a continuum, not a fixed per-env specialist). This mirrors the
# payload phase's `_add_odd_events` (per-episode rigidity × mass) — the three arms below differ ONLY in
# what the policy observes (blind: nothing; conditioned: θ on actor+critic; history: K-frame stack, no θ).
FR_TORQUE_LO, FR_TORQUE_HI = 0.2, 1.0   # θ span: 0.2 = badly weakened FR leg … 1.0 = nominal
HISTORY_K = 16                          # frames of proprio+action history (0.32 s @ 50 Hz) for the history arm


def _resolve_fr_act_ids(env, actuator_names: tuple[str, ...] = FR_ACTUATOR_NAMES) -> torch.Tensor:
  """Global model column indices of the FR leg's 3 actuators, resolved by NAME (robust to model layout).
  Mirrors ``degrade_fr_leg_torque``'s resolution so the randomized arms scale exactly the same motors."""
  m = env.sim.mj_model
  raw = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
  ids = torch.tensor([i for i in raw if i >= 0], device=env.device, dtype=torch.long)
  assert len(ids) == len(actuator_names), (
    f"weak-leg(randomized): expected {len(actuator_names)} FR actuators, resolved {len(ids)} "
    f"from {actuator_names} (got ids {raw})")
  return ids


def _ensure_fr_cache(env, actuator_names: tuple[str, ...] = FR_ACTUATOR_NAMES) -> None:
  """Lazily resolve+cache the FR actuator ids, snapshot the NOMINAL forcerange, and allocate the per-env
  θ store. Idempotent — safe to call from the startup event OR lazily from the reset event (mirrors how
  ``payload_odd`` lazily resolves its index). The nominal snapshot is taken ONCE, before any per-episode
  scaling, so the reset event can always write ABSOLUTELY from nominal (no compounding across resets)."""
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
  the randomized arms carry NO fixed ``degrade_fr_leg`` event to trigger that expansion)."""
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
  reset fires first (mirrors ``payload_odd``'s lazy index resolution)."""
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


# ── θ CONDITIONING observation (the oracle signal for the conditioned arm) ──────────────────────
def fr_torque_theta(env) -> torch.Tensor:
  """Per-env ODD θ = the FR-leg torque fraction ∈ [lo, hi]. Shape (num_envs, 1). Read straight back from
  ``env._fr_torque_frac`` (the value the reset event committed this episode); ones before it is allocated."""
  frac = getattr(env, "_fr_torque_frac", None)
  if frac is None:
    return torch.ones(env.num_envs, 1, device=env.device)
  return frac.reshape(-1, 1)


def _add_fr_conditioning_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Expose θ to BOTH the actor (adapt strategy per-θ) and the critic (θ-correct value). Separate
  ObservationTermCfg per group (the manager owns per-term state) — mirrors ``_add_odd_conditioning_obs``."""
  for group in ("actor", "critic"):
    cfg.observations[group].terms["fr_torque_theta"] = ObservationTermCfg(func=fr_torque_theta)


def _set_fr_obs_history(cfg: ManagerBasedRlEnvCfg, k: int) -> None:
  """Frame-stack the actor+critic obs groups (K frames of proprio+action), no θ — mirrors the payload
  history arm's ``_set_obs_history``. The policy INFERS θ from the felt dynamics (the deployable arm)."""
  for group in ("actor", "critic"):
    g = cfg.observations.get(group)
    if g is not None:
      g.history_length = k


# ── BUILDERS: the three ODD-conditioned arms (θ = FR torque fraction, randomized per-episode) ───
def go2_weak_leg_randomized_env_cfg(
  play: bool = False, lo: float = FR_TORQUE_LO, hi: float = FR_TORQUE_HI,
) -> ManagerBasedRlEnvCfg:
  """BLIND arm: ``go2_stabilize`` with the FR leg's allowable torque θ ~ U[lo, hi] randomized PER-EPISODE
  (fresh θ each reset), θ HIDDEN from the policy. The worst-case baseline. NO fixed ``degrade_fr_leg``
  event — the startup-cache + reset-randomize events replace it. Everything else (margins, adversary,
  obs/reward/spawn) is inherited unchanged from ``go2_stabilize_env_cfg``."""
  cfg = go2_stabilize_env_cfg(play=play)
  cfg.events["cache_fr_nominal"] = EventTermCfg(func=cache_fr_nominal_forcerange, mode="startup", params={})
  cfg.events["randomize_fr_torque"] = EventTermCfg(
    func=randomize_fr_torque, mode="reset", params={"lo": lo, "hi": hi})
  return cfg


def go2_weak_leg_conditioned_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """CONDITIONED arm: the randomized (blind) env AND θ = FR torque fraction exposed to actor+critic — one
  policy that can adapt strategy per-θ (the oracle upper bound of the blind/conditioned/history A/B/C)."""
  cfg = go2_weak_leg_randomized_env_cfg(play=play)
  _add_fr_conditioning_obs(cfg)
  return cfg


def go2_weak_leg_history_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """HISTORY arm: the randomized (blind) env, NO θ, but a K-frame proprio+action history stack — infer θ
  from the felt dynamics (the deployable alternative to a raw-θ input)."""
  cfg = go2_weak_leg_randomized_env_cfg(play=play)
  _set_fr_obs_history(cfg, HISTORY_K)
  return cfg


# ══════════════════════════════════════════════════════════════════════════════════════════════
# SOFT-REST reach-avoid margins — the ODD-AWARE safety objective for the degraded-leg regime.
# ══════════════════════════════════════════════════════════════════════════════════════════════
# A change in the ODD (leg dying) also changes what safety CAN mean: you cannot demand a normal STAND
# from a robot that physically can't stand. So the safety question becomes "does it come down SOFTLY, or
# SLAM trying to fight?" (Buzi's reframing.)
#   SAFETY g = don't SLAM: max non-foot ground-contact force < SLAM_N (10 N). Belly rest is ALLOWED as
#             long as it stays gentle; a hard impact (or fighting-and-crashing) is the violation. This
#             matches the env's built-in illegal_contact termination (also 10 N) — consistent.
#   TARGET l = stay LEVEL: small roll & pitch (base upright-ish). Satisfiable by standing level OR by a
#             level gentle belly-down — so "gracefully sit down" is on-target when standing is infeasible.
# The interesting question this enables: when the leg dies mid-episode, does the policy GRACEFULLY LOWER
# to a soft level rest, or keep fighting the pull it can no longer resist and SLAM?
# SLAM_N = the non-foot contact force separating a GENTLE belly settle from a SLAM. A gentle settle onto
# the belly measured ~60-70 N in the payload descent work; a hard impact is hundreds–thousands of N. 80 N
# ADMITS a soft belly rest (Buzi's intent) while still flagging a crash. (10 N would forbid any real rest —
# even the dead leg dragging in E046 exceeded it.) The env's illegal_contact termination is raised to match.
SLAM_N, SLAM_SCALE = 80.0, 80.0     # non-foot contact safe set: < 80 N (soft rest ok) safe, ≥ 80 N (slam) fail
RP_TOL, RP_SCALE = 0.25, 0.25       # target: max(|roll|,|pitch|) small (~14°) — LEVEL (stand OR belly)
V_TOL, V_SCALE = 0.30, 0.30         # target: |base linear vel| small — SETTLED (not drifting)
W_TOL, W_SCALE = 0.50, 0.50         # target: |base angular vel| small — SETTLED (not tumbling)


def _soften_contact_termination(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Raise the inherited ``illegal_contact`` termination threshold (default 10 N) to SLAM_N so a GENTLE belly
  rest (< SLAM_N) does NOT end the episode — only a slam does. Keeps the termination consistent with the
  soft_rest_margins safety boundary. (fell_over @70° still guards against tipping fully over.)"""
  t = cfg.terminations.get("illegal_contact")
  if t is not None:
    t.params = {**(dict(t.params) if t.params else {}), "force_threshold": SLAM_N}
  return cfg


def soft_rest_margins(env):
  """(g, l): soft-contact safety (no slam, <SLAM_N) + STABLE-LEVEL-POSE target. Safety reads the
  ``nonfoot_ground_touch`` contact sensor (same one illegal_contact uses); target = LEVEL (small roll &
  pitch) AND SETTLED (small linear & angular velocity). The 'settled' terms give the missing stable
  attractor: with a weak leg the only way to be on-target is to hold a still low pose (sit and settle),
  so 'gracefully sit down' is the target while 'topple / thrash' is off-target. The descent motion itself
  is a transient reach phase (reach-avoid tolerates it, as long as safety — no slam — holds throughout)."""
  d = env.scene["robot"].data
  s = env.scene["nonfoot_ground_touch"]
  fh = s.data.force_history if s.data.force_history is not None else s.data.force
  force = torch.norm(fh, dim=-1).flatten(1).amax(1)        # per-env max non-foot ground-contact force
  pg = d.projected_gravity_b                                # gravity in body frame; (x,y)→0 when level
  tilt = torch.maximum(pg[:, 0].abs(), pg[:, 1].abs())      # ~ max(|pitch|,|roll|) sines
  v = torch.linalg.norm(d.root_link_lin_vel_b, dim=1)       # base linear speed
  w = torch.linalg.norm(d.root_link_ang_vel_b, dim=1)       # base angular speed
  g = (SLAM_N - force) / SLAM_SCALE                          # SAFE: contact < SLAM_N (belly rest ok if gentle)
  l = torch.stack([                                          # TARGET: level AND settled (a STABLE held pose)
    (RP_TOL - tilt) / RP_SCALE,
    (V_TOL - v) / V_SCALE,
    (W_TOL - w) / W_SCALE,
  ], dim=0).amin(dim=0)
  return g, l


# ── FIXED-θ EVAL builders (θ pinned via lo=hi=theta): give the conditioned/history policies their
# training obs surface (θ scalar / K-frame stack) at a single test torque, for the survival sweep. ──
def go2_weak_leg_conditioned_at_env_cfg(play: bool = False, theta: float = 1.0) -> ManagerBasedRlEnvCfg:
  """CONDITIONED policy eval @ a FIXED FR-torque θ (lo=hi=theta ⇒ constant θ every episode): forcerange
  degraded to θ AND θ appended to actor+critic (the value the conditioned policy expects)."""
  cfg = go2_weak_leg_randomized_env_cfg(play=play, lo=theta, hi=theta)
  _add_fr_conditioning_obs(cfg)
  return cfg


def go2_weak_leg_history_at_env_cfg(play: bool = False, theta: float = 1.0) -> ManagerBasedRlEnvCfg:
  """HISTORY policy eval @ a FIXED FR-torque θ (lo=hi=theta): forcerange degraded to θ, K-frame stack, no θ."""
  cfg = go2_weak_leg_randomized_env_cfg(play=play, lo=theta, hi=theta)
  _set_fr_obs_history(cfg, HISTORY_K)
  return cfg


# ── SOFT-REST env builders: base leg envs + illegal_contact threshold raised to SLAM_N (paired with
# margin_fn=soft_rest_margins at registration). Belly rest < SLAM_N no longer terminates. ──
def go2_weak_leg_soft_env_cfg(play: bool = False, torque_frac: float = 0.2) -> ManagerBasedRlEnvCfg:
  return _soften_contact_termination(go2_weak_leg_env_cfg(play=play, torque_frac=torque_frac))


def go2_weak_leg_blind_soft_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _soften_contact_termination(go2_weak_leg_randomized_env_cfg(play=play))


def go2_weak_leg_conditioned_soft_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _soften_contact_termination(go2_weak_leg_conditioned_env_cfg(play=play))


def go2_weak_leg_history_soft_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _soften_contact_termination(go2_weak_leg_history_env_cfg(play=play))


def go2_weak_leg_conditioned_soft_at_env_cfg(play: bool = False, theta: float = 1.0) -> ManagerBasedRlEnvCfg:
  return _soften_contact_termination(go2_weak_leg_conditioned_at_env_cfg(play=play, theta=theta))


def go2_weak_leg_history_soft_at_env_cfg(play: bool = False, theta: float = 1.0) -> ManagerBasedRlEnvCfg:
  return _soften_contact_termination(go2_weak_leg_history_at_env_cfg(play=play, theta=theta))


# ══════════════════════════════════════════════════════════════════════════════════════════════
# DYNAMIC-θ SOFT-REST training envs (E064) — θ changes MID-EPISODE during training.
# ══════════════════════════════════════════════════════════════════════════════════════════════
# Same soft-rest envs as the static arms (soft_rest_margins, cache_fr_nominal startup, the
# conditioning-obs / K-frame history) BUT the per-episode ``randomize_fr_torque`` RESET event is
# DROPPED — a per-step ``LegDeathCallback`` (see callbacks.py) drives θ per-env instead, letting the
# leg die/recover/hold WITHIN an episode so the policies practice the transition. The startup cache
# event stays (it allocates the per-world forcerange + the θ store the callback writes into).
def _drop_reset_randomize(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Remove the per-episode reset θ-randomization so the per-step LegDeathCallback is the SOLE θ
  driver (avoids a spurious reset spike). Idempotent; keeps the ``cache_fr_nominal`` startup event."""
  cfg.events.pop("randomize_fr_torque", None)
  return cfg


def go2_weak_leg_blind_soft_dyn_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """BLIND arm, DYNAMIC θ: blind soft-rest env with the reset randomization dropped (θ driven per-step
  mid-episode by LegDeathCallback), θ HIDDEN from the policy."""
  return _soften_contact_termination(_drop_reset_randomize(go2_weak_leg_randomized_env_cfg(play=play)))


def go2_weak_leg_conditioned_soft_dyn_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """CONDITIONED arm, DYNAMIC θ: conditioned soft-rest env (θ exposed to actor+critic), reset
  randomization dropped — the obs term reads the LIVE θ the callback writes each step."""
  return _soften_contact_termination(_drop_reset_randomize(go2_weak_leg_conditioned_env_cfg(play=play)))


def go2_weak_leg_history_soft_dyn_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """HISTORY arm, DYNAMIC θ: history soft-rest env (K-frame proprio+action stack, no θ), reset
  randomization dropped — infer the mid-episode θ change from the felt dynamics."""
  return _soften_contact_termination(_drop_reset_randomize(go2_weak_leg_history_env_cfg(play=play)))


# ══════════════════════════════════════════════════════════════════════════════════════════════
# LEG-DEGRADATION SPEC FAMILY (T004 / E076, DEMO 2) — the weight-ladder template on the FR-torque axis.
# ══════════════════════════════════════════════════════════════════════════════════════════════
# Mirrors go2_weight_ladder's two-mode spec family, but the ODD is θ = FR-leg allowable-torque fraction
# (the existing per-episode ``randomize_fr_torque`` machinery), NOT a carried load. As θ falls the strongest
# feasible safety spec changes: while θ is comfortably high the robot can hold a normal STAND (stance_margins,
# θ~U[0.5,1.0] — the feasible band per E047 weak50≈0.075 fail); once the FR leg is too weak, standing is
# infeasible and the strongest thing left is to come down softly to a LOW LEVEL belly rest WITHOUT slamming
# (``leg_rest_margins``, θ~U[0.0,1.0] INCLUDING a fully-dead leg — lying down needs no FR leg). This is the
# leg-ODD analog of go2_weight_{stand,rest}; the certified handoff between the modes is a later layer (E076
# ramp). ADDITIVE — everything is built on ``go2_weak_leg_randomized_env_cfg`` + the existing θ conditioning
# obs; no existing task or margin is modified.
LEG_STAND_LO, LEG_STAND_HI = 0.5, 1.0       # STAND mode θ span: comfortably-feasible stance band
LEG_REST_LO, LEG_REST_HI = 0.0, 1.0         # REST mode θ span: includes a fully-dead FR leg (θ=0)

# leg_rest_margins constants — MIRROR go2_weight_ladder.weight_rest_margins' l EXACTLY (LOW + level + settled),
# but with a CONSTANT no-slam cap (there is no carried load here, so no load conditioning: the belly rests at
# ~robot weight only). H_LIE/HS add the LOW target that go2_broken_leg's soft_rest_margins lacks.
LEG_H_LIE, LEG_HS = 0.15, 0.06              # target: base LOW (base_z < H_LIE ⇒ (H_LIE-base_z)/HS > 0)
# (RP_TOL/RP_SCALE, V_TOL/V_SCALE, W_TOL/W_SCALE reuse soft_rest_margins' module constants: 0.25/0.30/0.50.)

# REST-mode illegal_contact threshold: a HIGH constant so a belly rest under a dead leg does NOT terminate —
# the reach-avoid g (constant 80 N no-slam cap) does the safety scoring; only fell_over@70° stays a live
# failure. Mirrors go2_weight_ladder.REST_CONTACT_N.
LEG_REST_CONTACT_N = 500.0


def _raise_contact_termination(cfg: ManagerBasedRlEnvCfg, force_threshold: float) -> ManagerBasedRlEnvCfg:
  """Raise the inherited ``illegal_contact`` termination threshold (default 10 N) to ``force_threshold`` so a
  belly rest below it does NOT end the episode. Generalizes ``_soften_contact_termination`` (which pins the
  80 N soft-rest SLAM_N) to an arbitrary per-mode threshold — mirrors go2_weight_ladder's helper."""
  t = cfg.terminations.get("illegal_contact")
  if t is not None:
    t.params = {**(dict(t.params) if t.params else {}), "force_threshold": float(force_threshold)}
  return cfg


def leg_rest_margins(env):
  """(g, l): CONSTANT no-slam safety (< SLAM_N = 80 N) + LOW/level/settled belly-rest target. Reads the
  ``nonfoot_ground_touch`` contact sensor (as ``soft_rest_margins`` does). g = (SLAM_N − force)/SLAM_SCALE;
  l = min of (low, level, settled-lin, settled-ang) — satisfiable only by a LOW still level pose. Distinct
  from ``soft_rest_margins`` by the added LOW target (a normal STAND at base_z≈0.32 ≫ H_LIE ⇒ l<0, so
  standing is off-target here); distinct from ``weight_rest_margins`` by the CONSTANT cap (no carried load)."""
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


__all__ = ["go2_weak_leg_env_cfg", "stance_margins", "degrade_fr_leg_torque",
           "go2_weak_leg_randomized_env_cfg", "go2_weak_leg_conditioned_env_cfg",
           "go2_weak_leg_history_env_cfg", "cache_fr_nominal_forcerange", "randomize_fr_torque",
           "fr_torque_theta", "FR_TORQUE_LO", "FR_TORQUE_HI", "HISTORY_K",
           "go2_weak_leg_conditioned_at_env_cfg", "go2_weak_leg_history_at_env_cfg",
           "soft_rest_margins", "go2_weak_leg_soft_env_cfg", "go2_weak_leg_blind_soft_env_cfg",
           "go2_weak_leg_conditioned_soft_env_cfg", "go2_weak_leg_history_soft_env_cfg",
           "go2_weak_leg_conditioned_soft_at_env_cfg", "go2_weak_leg_history_soft_at_env_cfg",
           "go2_weak_leg_blind_soft_dyn_env_cfg", "go2_weak_leg_conditioned_soft_dyn_env_cfg",
           "go2_weak_leg_history_soft_dyn_env_cfg",
           "leg_rest_margins", "go2_leg_stand_env_cfg", "go2_leg_rest_env_cfg",
           "LEG_STAND_LO", "LEG_STAND_HI", "LEG_REST_LO", "LEG_REST_HI", "LEG_REST_CONTACT_N"]
