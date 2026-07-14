"""Reach-avoid margins for the Digit v3 safety-filter task (zoo convention).

Terms are signed distances normalized to O(1) (``compose`` in the zoo's
top-level ``margins`` clamps to +-3); g < 0 == failure, l >= 0 == in the
stabilized stance.

    g (avoid)  : NOT fallen — torso height, uprightness (<80 deg), and no
                 non-foot body touching the ground. FALL-ONLY: no foot-planted
                 / stance constraint (see DYNAMIC-RECOVERY note below).
    l (reach)  : the "settled in place" target — roughly upright (<20 deg), low
                 planar speed (loose, so stepping to recover is allowed), near
                 the spawn origin.

DYNAMIC-RECOVERY design (2026-07-08). Sim2sim proved the crux: a PD *statue*
topples in ~3.5 s in mjlab (weak toe-pushrod ankle-roll authority, the
calibration residual) while a *walking* policy survives 20 s in BOTH mjlab and
ar-control. Static standing depends on the ankle authority mjlab lacks; dynamic
(stepping) balance does not, and it transfers. Every prior safety policy that
"stood" in mjlab did so by a static tiptoe/hop trick that then failed in
ar-control (avoid-only: 20 s -> 3.75 s). So we STOP forcing a planted stance:
the foot-planted term is removed from ``g`` entirely (a lifted foot during a
recovery step no longer trips the avoid margin), and ``l`` stays loose on
velocity so the policy is free to step in place to balance — the transferable
strategy — rather than manufacture an untransferable static equilibrium.
"""

from __future__ import annotations

import math

import torch

from robot_safety_sandbox.envs.digit_safety.mdp import _FOOT_AND_KNEE_BODIES

# --- g (avoid) thresholds + normalizers ---
GROUND_CLEARANCE = 0.03   # non-foot/knee bodies must clear this (m)
MIN_TORSO_HEIGHT = 0.40   # torso above this or we've fallen (m)
HEIGHT_NORM = 0.50        # normalizer for the height/clearance margins (m)
SIN_TILT_G = math.sin(math.radians(80.0))  # avoid: not past 80 deg

# --- l (reach / STAY-IN-PLACE) thresholds + normalizers ---
# LESSON (see report): the earlier `l_digit_stance` was a 12-term hard min with
# a razor-thin flat-foot/knee/velocity intersection -> effectively UNREACHABLE
# -> the reach-avoid value was pessimistic everywhere and its optimum chased the
# unreachable pose, degrading g-convergence (ep_len plateaued ~90 vs avoid-only
# ~450). The reach-avoid backup is correct; the target must be REACHABLE.
#
# `l_digit_stay` is the simple, reachable target: "stand in place, stopped,
# upright" — the loose set a stabilized standing robot naturally enters (stepping
# to recover is allowed; the target is being settled/at-rest, not a fixed pose).
SIN_STAY = math.sin(math.radians(20.0))  # roughly upright (loose vs g's 80 deg)
# Loosened (v2): the earlier tight motion terms (V 0.5, R 1.5, + yaw) over-
# constrained recovery — with flat feet forced in g, the policy couldn't
# step/sway to catch itself and fell in ~1s. Give it room to recover: allow
# more speed + drift, and drop the yaw term entirely. "In place" -> "roughly
# in place." Robustness is prerequisite to the stay-in-place refinement.
V_STAY = 1.00             # planar base speed target (m/s) — allow recovery motion
R_STAY = 3.00             # stay within this of the spawn origin (m) — loose

# --- REACH-SET CURRICULUM (l-annealing) ---
# Warm-starting reach-avoid from an avoid-only base and then applying the STRICT
# l target immediately yanks the policy toward a tight set far from where the
# base lives -> OOD -> PPO erodes g (ep_len 440 -> 73). Fix: contract the reach
# set gradually. l reads env._l_alpha in [0, 1] (default 1.0 = strict, no
# curriculum); each threshold lerps from a LOOSE endpoint (alpha=0: ~= g's
# non-fallen set, so the warm base already satisfies l and reach-avoid ~=
# avoid-only) to the STRICT target (alpha=1). A callback ramps alpha 0 -> 1, so
# the target tightens slowly and the policy stays in-distribution throughout.
# alpha=0 uprightness == g's 80 deg cone (trivially met by any standing state).
SIN_STAY_LOOSE = SIN_TILT_G          # sin(80 deg): l-upright == g-upright at a=0
SIN_STAY_STRICT = SIN_STAY           # sin(20 deg): the true stay target
V_STAY_LOOSE, V_STAY_STRICT = 3.00, V_STAY   # planar speed ceiling (m/s)
# (drift term dropped — unobservable by the actor; see l_digit_stay. R_STAY* left
# defined above for reference only, no longer used in the margin.)
# Anti-tiptoe (STAY home, 2026-07-09): the first stay policy converged to a
# robust TIPTOE stand (measured min-foot-z 0.104->0.119 ~= tiptoe 0.127) —
# nothing in the margin mentioned feet. The planted term returns HERE, in the
# stance set, where it is finally structurally sound: persistent (stay backup,
# no reach-once masking), non-terminal (fall-only termination — a violation
# dents the margin, recovery steps aren't punished as falls), observable (the
# obs-104 actor sees foot_height/foot_contact), and annealed (loose 0.20 sits
# ABOVE tiptoe so the tiptoe warm-start is in-distribution, then contracts).
# min over feet: a step keeps the stance foot planted; only both-feet-elevated
# (tiptoe/hop) violates. Flat foot-site z ~0.014; strict 0.06 excludes tiptoe.
PLANT_LOOSE, PLANT_STRICT = 0.20, 0.06  # min-foot-site z ceiling (m)


def _lerp(lo: float, hi: float, a: float) -> float:
  return lo + (hi - lo) * a
# Anti-tiptoe: require at least ONE foot planted (min over feet). Measured:
# flat foot-site z ~0.014, tiptoe/jump ~0.127. Using MIN (not max) targets the
# both-feet-up tiptoe/jump while leaving stepping free (a step keeps the stance
# foot planted -> min stays low). Loose threshold -> reachable by flat standing.
PLANT_MAX = 0.06          # min-foot-site height ceiling for "a foot is planted"
PLANT_NORM = 0.06

# --- (deprecated) tight stance thresholds, kept for reference ---
SIN_TILT_L = math.sin(math.radians(15.0))
LIN_TOL = 0.30
ANG_TOL = 0.50
TORSO_LOW = 0.85
TORSO_HIGH = 1.05
TORSO_BAND_NORM = 0.15
MAX_FOOT_HEIGHT = 0.10
KNEE_TARGET = 0.344
KNEE_BAND = 0.15


def _resolve(env) -> dict:
  """Resolve & cache body/site/joint indices on the env (once)."""
  cache = getattr(env, "_digit_zoo_idx", None)
  if cache is not None:
    return cache
  robot = env.scene["robot"]

  all_ids, _ = robot.find_bodies((".*",))
  all_ids = all_ids.tolist() if hasattr(all_ids, "tolist") else list(all_ids)
  allowed: set[int] = set()
  for name in _FOOT_AND_KNEE_BODIES:
    try:
      ids, _ = robot.find_bodies((name,))
      allowed.update(ids.tolist() if hasattr(ids, "tolist") else ids)
    except (ValueError, RuntimeError):
      pass
  failure_ids = [b for b in all_ids if b not in allowed]

  torso_ids, _ = robot.find_bodies(("torso",))
  foot_ids, _ = robot.find_sites(("left_foot", "right_foot"))
  lk, _ = robot.find_joints(("left_knee_joint",))
  rk, _ = robot.find_joints(("right_knee_joint",))

  cache = {
    "failure_ids": failure_ids,
    "torso_id": int(torso_ids[0]),
    "foot_ids": foot_ids,
    "left_knee": int(lk[0]),
    "right_knee": int(rk[0]),
  }
  env._digit_zoo_idx = cache
  return cache


def g_digit_stand(env) -> torch.Tensor:
  """Avoid margin (FALL-ONLY): torso height, uprightness (<80 deg), non-foot
  body ground clearance.

  No foot-planted / stance term (see the DYNAMIC-RECOVERY note in the module
  docstring): forcing a planted stance terminated legitimate recovery steps and
  pushed the policy into an untransferable static tiptoe. Here the robot is
  free to lift a foot and step to catch itself; g < 0 only on an actual fall.
  """
  idx = _resolve(env)
  d = env.scene["robot"].data

  torso_z = d.body_link_pose_w[:, idx["torso_id"], 2]
  torso_margin = (torso_z - MIN_TORSO_HEIGHT) / HEIGHT_NORM

  grav_xy = torch.norm(d.projected_gravity_b[:, :2], dim=1)
  orient = (SIN_TILT_G - grav_xy) / SIN_TILT_G

  failure_z = d.body_link_pose_w[:, idx["failure_ids"], 2]
  body_clear = ((failure_z - GROUND_CLEARANCE) / HEIGHT_NORM).min(dim=1).values

  return torch.stack(
    [torso_margin, orient, body_clear], dim=-1
  ).min(dim=-1).values


def l_digit_stay(env) -> torch.Tensor:
  """Reach margin (REACHABLE): stand in place, stopped, upright.

  ``l >= 0`` iff the robot is roughly upright (<20 deg), at rest (low planar
  speed), and has a foot PLANTED (anti-tiptoe, min over feet — see PLANT_*).
  Loose by design so a stabilized standing robot naturally enters it —
  stepping to recover is allowed; the target is the settled state, not a pose.

  Only OBSERVABLE quantities: uprightness (``projected_gravity``) and planar
  speed (``base_lin_vel``). The earlier DRIFT term (distance from spawn) was
  DROPPED — the actor has no base-position observation and no memory to
  integrate velocity, so it could not sense or correct drift; as an unobservable
  min() term it only injected noise into the reach signal. The low-speed term
  suppresses drift indirectly (a robot that holds ~zero velocity doesn't wander).

  Thresholds are annealed by ``env._l_alpha`` in [0, 1] (default 1.0 = strict):
  the reach-set curriculum contracts them from LOOSE (alpha=0, ~= g's non-fallen
  set) to STRICT (alpha=1, the stay target). See the REACH-SET CURRICULUM note.
  Each term is normalized by its CURRENT (annealed) threshold so it stays O(1)
  and crosses zero exactly at that threshold.
  """
  a = float(getattr(env, "_l_alpha", 1.0))
  sin_stay = _lerp(SIN_STAY_LOOSE, SIN_STAY_STRICT, a)
  v_stay = _lerp(V_STAY_LOOSE, V_STAY_STRICT, a)
  plant_max = _lerp(PLANT_LOOSE, PLANT_STRICT, a)

  idx = _resolve(env)
  d = env.scene["robot"].data

  grav_xy = torch.norm(d.projected_gravity_b[:, :2], dim=1)
  v_xy = torch.norm(d.root_link_lin_vel_b[:, :2], dim=1)
  min_foot_z = d.site_pos_w[:, idx["foot_ids"], 2].min(dim=1).values

  terms = torch.stack(
    [
      (sin_stay - grav_xy) / sin_stay,   # roughly upright (observable)
      (v_stay - v_xy) / v_stay,          # low speed (observable; allows recovery)
      (plant_max - min_foot_z) / plant_max,  # stance foot planted (anti-tiptoe)
    ],
    dim=0,
  )
  return terms.amin(dim=0)


def g_digit_stabilize(env) -> torch.Tensor:
  """STAY margin: remain in the stance set forever (avoid formulation).

  STRUCTURAL LESSON (2026-07-09, the "hidden error" that broke every
  reach-avoid run): stabilization is reach-AND-STAY, but the reach-avoid
  backup ``V = min(g, max(l, gV'))`` is reach-ONCE — and the robot SPAWNS
  inside the stay target (measured l=0.44-0.86 > 0 at every curriculum
  alpha, with l >~ g). At any healthy standing state ``max(l, gV') = l >= g``
  masks the recursion: the value target collapses to the instantaneous
  ``g(s)``, independent of the future. Falling 0.2 s later is invisible to
  the backup; advantages at standing states are ~0 (noise), so PPO erodes a
  warm-started standing policy instead of preserving it. Reach-avoid answers
  "can I touch the target once?" — trivially yes at t=0.

  "Stay upright + settled forever" is a VIABILITY problem, which is exactly
  the avoid backup ``V = min(g', gV')`` on the composite margin

      g' = min(g_fall, l_stance)

  Termination stays FALL-ONLY (the env's ``safety_margin_violated`` uses the
  fall terms): a stance excursion (recovery step, transient 30-deg tilt) is
  non-terminal and recoverable — it dents the min-backup margin, it doesn't
  end the episode. That resolves the old "stance-in-g kills recovery" trap,
  which was a termination problem, not a margin problem. ``l_digit_stay``'s
  ``_l_alpha`` annealing applies unchanged: at alpha=0 g' ~= g_fall (warm
  start from the avoid base is in-distribution), then the stance tightens.

  Train with SafetyPPO (the proven avoid recipe) via compose(g_digit_stabilize,
  l_zero).
  """
  return torch.minimum(g_digit_stand(env), l_digit_stay(env))


# --- BOX task (avoid) thresholds — MIRROR the env termination exactly
# (env_cfgs safety_margin_violated: box_body_name="box_load",
# min_box_height=1.0, box_tilt_limit_rad=1.3963=80deg) so g < 0 == terminate.
BOX_MIN_HEIGHT = 1.0
BOX_TILT_COS = math.cos(math.radians(80.0))
BOX_NORM = 0.5


def _resolve_box(env) -> int:
  bid = getattr(env, "_digit_zoo_box_id", None)
  if bid is None:
    ids, _ = env.scene["robot"].find_bodies(("box_load",))
    bid = int(ids[0])
    env._digit_zoo_box_id = bid
  return bid


def g_box_load(env) -> torch.Tensor:
  """Avoid margin for the free box: not dropped (height) and not spilling
  (tilt). Both are terminal in the env, so they live in g."""
  bid = _resolve_box(env)
  pose = env.scene["robot"].data.body_link_pose_w[:, bid]
  height = (pose[:, 2] - BOX_MIN_HEIGHT) / BOX_NORM
  qx, qy = pose[:, 4], pose[:, 5]
  cos_tilt = 1.0 - 2.0 * (qx * qx + qy * qy)
  tilt = (cos_tilt - BOX_TILT_COS) / (1.0 - BOX_TILT_COS)
  return torch.minimum(height, tilt)


def g_digit_box_stand(env) -> torch.Tensor:
  """Box avoid stage: don't fall AND don't drop/spill the box."""
  return torch.minimum(g_digit_stand(env), g_box_load(env))


def g_digit_box_stabilize(env) -> torch.Tensor:
  """Box STAY margin: remain in the stance set (annealed) AND keep the box
  balanced forever — the viability formulation of the original project task."""
  return torch.minimum(g_digit_stabilize(env), g_box_load(env))


def l_digit_stance(env) -> torch.Tensor:
  """(Deprecated — too tight/unreachable, see report.) Hardware-pose stance."""
  idx = _resolve(env)
  d = env.scene["robot"].data

  grav_xy = torch.norm(d.projected_gravity_b[:, :2], dim=1)
  v_b = d.root_link_lin_vel_b
  w_b = d.root_link_ang_vel_b
  torso_z = d.body_link_pose_w[:, idx["torso_id"], 2]
  foot_z = d.site_pos_w[:, idx["foot_ids"], 2]
  jp = d.joint_pos
  lk = jp[:, idx["left_knee"]]
  rk = jp[:, idx["right_knee"]]

  terms = torch.stack(
    [
      (SIN_TILT_L - grav_xy) / SIN_TILT_L,                 # upright (tight)
      (LIN_TOL - v_b[:, 0].abs()) / LIN_TOL,
      (LIN_TOL - v_b[:, 1].abs()) / LIN_TOL,
      (LIN_TOL - v_b[:, 2].abs()) / LIN_TOL,
      (ANG_TOL - w_b[:, 0].abs()) / ANG_TOL,
      (ANG_TOL - w_b[:, 1].abs()) / ANG_TOL,
      (ANG_TOL - w_b[:, 2].abs()) / ANG_TOL,
      (torso_z - TORSO_LOW) / TORSO_BAND_NORM,             # height band low
      (TORSO_HIGH - torso_z) / TORSO_BAND_NORM,            # height band high
      (MAX_FOOT_HEIGHT - foot_z.max(dim=1).values) / MAX_FOOT_HEIGHT,  # flat foot
      (KNEE_BAND - (lk - KNEE_TARGET).abs()) / KNEE_BAND,  # hardware knee (L +)
      (KNEE_BAND - (rk + KNEE_TARGET).abs()) / KNEE_BAND,  # hardware knee (R -)
    ],
    dim=0,
  )
  return terms.amin(dim=0)
