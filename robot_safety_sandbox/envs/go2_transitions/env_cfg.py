"""Go2 CERTIFIED-TRANSITION funnels — batch 2 of the dynamic-ODD switching demo (T006 / E083).

Batch 1 (go2_weight_ladder) certified the two OPERATING MODES (STAND, REST) and the instantaneous switch,
but left the TRANSITION TRAJECTORIES uncertified — the E081 hazard vacuum (~35%/cycle death from untrained
zero-shot get-up / descent maneuvers). Batch 2 makes each transition its OWN reach-avoid problem, with its own
value function, so the trajectory between modes is certified end-to-end:

  * GET-UP funnel  RA_rest→stand  (``go2_getup``):  from PRONE states UNDER LOAD, REACH the FULL STANCE target
    while avoiding the universal no-slam catastrophe. This is the genuine gap — nothing in the zoo trains a
    loaded get-up. It needs new PRONE-SPAWN infra (``reset_prone``), since ``go2_stabilize`` spawns standing.
  * DESCENT funnel RA_stand→rest (``go2_descend``): from DISTURBED STANDING states UNDER HEAVY LOAD, REACH the
    FULL REST set while avoiding the slam. ``go2_weight_rest`` already approximates this; the dedicated version
    emphasises heavy W and disturbed inits so E084 can compare it against the reused ``go2_weight_rest_hi``.

Both are ADDITIVE: built on ``go2_stabilize`` + the ``go2_weight_ladder`` carried-load channel + margins; no
existing task, margin, or asset is modified. The universal catastrophe g (load-conditioned no-slam) and the W
conditioning obs are REUSED verbatim from ``go2_weight_ladder`` — the funnels differ from the operating modes
only in their SPAWN distribution and (for get-up) which target l the same-g reach-avoid problem drives toward.

SPAWN INFRA (get-up): ``reset_prone`` mirrors mjlab's ``reset_root_state_uniform`` / ``reset_joints_by_offset``
(``mjlab/envs/mdp/events.py``) — it writes the SAME sim buffers (root state via ``write_root_state_to_sim``,
joint state via ``write_joint_state_to_sim``) but to a randomized PRONE/low folded pose instead of the standing
default. It REPLACES the default standing spawn events (``reset_base`` / ``reset_robot_joints`` are popped).
"""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from robot_safety_sandbox.envs.go2_stabilize.env_cfg import (  # reused verbatim
  go2_stabilize_env_cfg,
  stance_margins,
)
from robot_safety_sandbox.envs.go2_weight_ladder.env_cfg import (  # the carried-load channel + rest g
  randomize_base_load,
  weight_rest_margins,
  _add_weight_conditioning_obs,
  _raise_contact_termination,
  go2_weight_rest_env_cfg,
  REST_CONTACT_N,
  LOAD_HEIGHT,
)

# ── GET-UP ODD distribution: carried load W ∈ [0, 160] N (feasible band + a boundary margin, per the E075
# calibration lesson) — heavier than STAND's [0,150] so the funnel sees the get-up boundary, lighter than
# REST's [0,300] since a get-up from the very heaviest loads is infeasible. ──
GETUP_LO, GETUP_HI = 0.0, 160.0
# ── DESCENT ODD distribution: carried load W ∈ [80, 260] N (HEAVY emphasis — the regime where a controlled
# descent is the strongest feasible spec; disturbed standing inits stress the funnel). ──
DESCEND_LO, DESCEND_HI = 80.0, 260.0

# ── PRONE SPAWN pose distribution (get-up). base LOW to the floor, small random tilt, any yaw, folded legs. ──
PRONE_Z = (0.09, 0.13)      # absolute base height (m): belly near the floor (default stand base_z ≈ 0.32)
PRONE_RP = 0.15             # base roll/pitch ~ U[-0.15, 0.15] rad (small — nearly flat on the floor)
PRONE_YAW = math.pi         # base yaw ~ U[-π, π] (any heading)
THIGH_FOLD = (0.3, 1.8)     # covers the MEASURED settled-rest band (E086 probe p2..p98: 0.31..1.67)
CALF_FOLD = (-2.75, -1.85)  # measured settled-rest band: -2.73..-1.90
HIP_FOLD = (-0.9, 0.9)      # rest posture splays hips to ±0.75-0.9 (E086 probe) — old ±0.2 missed it

# ── DISTURBED-STANDING spawn disturbance (descent). ``reset_base`` (reset_root_state_uniform) already
# randomizes x/y/yaw but NOT tilt or velocity (its pose_range has no roll/pitch, its velocity_range is empty),
# so a DEDICATED disturbance is added on top: an initial base TILT + LATERAL velocity that a heavy load then
# torques — the funnel must recover a stand→rest under an already-off-balance start. ──
DESCEND_TILT = 0.2          # extra base roll/pitch ~ U[-0.2, 0.2] rad at spawn
DESCEND_LAT_VEL = 0.4       # extra base lateral velocity ~ U[-0.4, 0.4] m/s at spawn (x and y)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# GET-UP margins: the SAME universal no-slam g as the REST mode, but the FULL STANCE target l.
# ══════════════════════════════════════════════════════════════════════════════════════════════
def getup_margins(env):
  """(g, l) for the GET-UP funnel RA_rest→stand. Reuses the existing computations — no forked constants:

    g = the LOAD-CONDITIONED no-slam safety of ``weight_rest_margins`` (F_universal: max non-foot ground
        contact < 80 + 1.3·W N). This is the universal catastrophe shared by every mode; the ``fell_over``
        termination stays as the tip catastrophe. A prone robot rests below the cap ⇒ g > 0 while folded.
    l = the FULL STANCE target of ``stance_margins`` (corners in [0.10, 0.40], |v| ≤ 0.20, |ω| ≤ 0.174):
        the funnel must REACH AND HOLD a genuine stand, not merely leave the floor. l < 0 while prone.

  So V > 0 == "can get up to a stable stand from here, under this load, without slamming, despite the
  adversary". The episode does NOT end on reach (end_criterion='failure', the zoo default) — the value
  anchors handle the reach-and-hold, exactly as every other reach-avoid task in the zoo."""
  g, _ = weight_rest_margins(env)   # load-conditioned no-slam (F_universal) — reused verbatim
  _, l = stance_margins(env)        # full stance-band target — reach AND hold a stand
  return g, l


# ══════════════════════════════════════════════════════════════════════════════════════════════
# PRONE SPAWN infra (get-up): initialize reset envs in a randomized prone/low folded pose.
# Mirrors mjlab's reset_root_state_uniform / reset_joints_by_offset — writes the SAME sim buffers.
# ══════════════════════════════════════════════════════════════════════════════════════════════
def _prone_joint_ids(env):
  """Resolve + cache the (thigh, calf, hip) joint index tensors on the inner env (find_joints is a regex over
  the 12 joint names; do it once)."""
  cache = getattr(env, "_prone_joint_ids", None)
  if cache is None:
    asset = env.scene["robot"]
    thigh = torch.tensor(asset.find_joints(".*thigh_joint")[0], device=env.device, dtype=torch.long)
    calf = torch.tensor(asset.find_joints(".*calf_joint")[0], device=env.device, dtype=torch.long)
    hip = torch.tensor(asset.find_joints(".*hip_joint")[0], device=env.device, dtype=torch.long)
    cache = env._prone_joint_ids = {"thigh": thigh, "calf": calf, "hip": hip}
  return cache


def reset_prone(env, env_ids: torch.Tensor | None) -> None:
  """Reset event: initialize the reset envs in a randomized PRONE/low folded pose (base z ~ U[0.09, 0.13],
  small roll/pitch ~ U[±0.15], any yaw, thighs/calves/hips near a folded pose, zero velocities). Mirrors
  mjlab's ``reset_root_state_uniform`` (root pose+vel via ``write_root_state_to_sim``) and
  ``reset_joints_by_offset`` (joint state via ``write_joint_state_to_sim``), writing the SAME sim buffers.
  REPLACES the default standing spawn — the getup cfg pops ``reset_base`` / ``reset_robot_joints`` first."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)
  asset = env.scene["robot"]
  n = len(env_ids)
  dev = env.device

  # ── ROOT STATE (13): position (3) + quat wxyz (4) + lin vel (3) + ang vel (3), all world frame. ──
  root = asset.data.default_root_state[env_ids].clone()          # (n, 13) — default STANDING root state
  origins = env.scene.env_origins[env_ids]                       # per-env grid origin (flat: z ≈ 0)
  # position: keep the default x/y (offset into the grid), OVERRIDE z to an absolute low prone height.
  root[:, 0:2] = root[:, 0:2] + origins[:, 0:2]
  root[:, 2] = origins[:, 2] + sample_uniform(PRONE_Z[0], PRONE_Z[1], (n,), dev)
  # orientation: small random roll/pitch (near flat) + any yaw, composed onto the default quat.
  roll = sample_uniform(-PRONE_RP, PRONE_RP, (n,), dev)
  pitch = sample_uniform(-PRONE_RP, PRONE_RP, (n,), dev)
  yaw = sample_uniform(-PRONE_YAW, PRONE_YAW, (n,), dev)
  root[:, 3:7] = quat_mul(root[:, 3:7], quat_from_euler_xyz(roll, pitch, yaw))
  root[:, 7:13] = 0.0                                            # zero linear + angular velocity
  asset.write_root_state_to_sim(root, env_ids=env_ids)

  # ── JOINT STATE: default joints with thighs/calves/hips overwritten to a folded pose; zero velocity. ──
  ids = _prone_joint_ids(env)
  jp = asset.data.default_joint_pos[env_ids].clone()             # (n, njoints)
  jv = torch.zeros_like(asset.data.default_joint_vel[env_ids])
  jp[:, ids["thigh"]] = sample_uniform(THIGH_FOLD[0], THIGH_FOLD[1], (n, len(ids["thigh"])), dev)
  jp[:, ids["calf"]] = sample_uniform(CALF_FOLD[0], CALF_FOLD[1], (n, len(ids["calf"])), dev)
  jp[:, ids["hip"]] = sample_uniform(HIP_FOLD[0], HIP_FOLD[1], (n, len(ids["hip"])), dev)
  lim = asset.data.soft_joint_pos_limits[env_ids]               # (n, njoints, 2) — clamp so no limit violation
  jp = jp.clamp(lim[..., 0], lim[..., 1])
  asset.write_joint_state_to_sim(jp, jv, env_ids=env_ids)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# DISTURBED-STANDING spawn disturbance (descent): add tilt + lateral velocity on top of ``reset_base``.
# ══════════════════════════════════════════════════════════════════════════════════════════════
def reset_standing_disturbance(
  env, env_ids: torch.Tensor | None, tilt: float = DESCEND_TILT, lat_vel: float = DESCEND_LAT_VEL,
) -> None:
  """Reset event that COMPOSES ON TOP of the default standing spawn (``reset_base`` runs first, sets the
  upright root pose+vel; this event then perturbs it): apply a random base TILT (roll/pitch ~ U[±tilt]) and a
  random LATERAL base velocity (x, y ~ U[±lat_vel]). It reads the root state ``reset_base`` just wrote and
  writes it back perturbed — mirroring ``reset_root_state_uniform``'s buffers. Must be inserted AFTER
  ``reset_base`` in the events dict so it composes (mjlab applies reset events in insertion order)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.long)
  asset = env.scene["robot"]
  n = len(env_ids)
  dev = env.device
  # tilt: compose a small random roll/pitch onto the current (standing) orientation.
  pos = asset.data.root_link_pos_w[env_ids].clone()
  quat = asset.data.root_link_quat_w[env_ids].clone()
  roll = sample_uniform(-tilt, tilt, (n,), dev)
  pitch = sample_uniform(-tilt, tilt, (n,), dev)
  zeros = torch.zeros(n, device=dev)
  quat = quat_mul(quat, quat_from_euler_xyz(roll, pitch, zeros))
  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  # lateral velocity: add x/y push to the current (zero) base velocity.
  vel = asset.data.root_link_vel_w[env_ids].clone()             # (n, 6): lin (3) + ang (3)
  vel[:, 0] = vel[:, 0] + sample_uniform(-lat_vel, lat_vel, (n,), dev)
  vel[:, 1] = vel[:, 1] + sample_uniform(-lat_vel, lat_vel, (n,), dev)
  asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# ENV BUILDERS: the two transition funnels.
# ══════════════════════════════════════════════════════════════════════════════════════════════
def go2_getup_env_cfg(
  play: bool = False, lo: float = GETUP_LO, hi: float = GETUP_HI, load_height: float = LOAD_HEIGHT,
) -> ManagerBasedRlEnvCfg:
  """GET-UP funnel RA_rest→stand: ``go2_stabilize`` with the STANDING spawn REPLACED by ``reset_prone`` +
  a per-episode carried load W ~ U[lo, hi] (HIGH-CoM lever) + W exposed to actor+critic + ``illegal_contact``
  raised to REST_CONTACT_N (a prone contact must NOT terminate; the reach-avoid g does the scoring).
  Margins = ``getup_margins`` (universal no-slam g + FULL stance target l). The stance target does NOT end the
  episode (end_criterion='failure', set on the TaskSpec) — the value anchors handle reach-and-hold."""
  cfg = go2_stabilize_env_cfg(play=play)
  # REPLACE the default STANDING spawn with the PRONE spawn (pop the two standing reset events, add prone).
  cfg.events.pop("reset_base", None)
  cfg.events.pop("reset_robot_joints", None)
  cfg.events["reset_prone"] = EventTermCfg(func=reset_prone, mode="reset", params={})
  # per-episode carried load W + W conditioning obs (same channel as the weight ladder).
  cfg.events["randomize_base_load"] = EventTermCfg(
    func=randomize_base_load, mode="reset", params={"lo": lo, "hi": hi, "load_height": load_height})
  _add_weight_conditioning_obs(cfg)
  _raise_contact_termination(cfg, REST_CONTACT_N)   # 500 N: prone/loaded contact must not terminate
  return cfg


def go2_descend_env_cfg(
  play: bool = False, lo: float = DESCEND_LO, hi: float = DESCEND_HI, load_height: float = LOAD_HEIGHT,
) -> ManagerBasedRlEnvCfg:
  """DESCENT funnel RA_stand→rest (dedicated): the ``go2_weight_rest`` pattern (standing spawn, load-
  conditioned soft-rest margins, ``illegal_contact`` raised to REST_CONTACT_N, W exposed) but with a HEAVY
  load range W ~ U[80, 260] and a DISTURBED-STANDING init (``reset_standing_disturbance`` adds base tilt +
  lateral velocity on top of the standing spawn). Margins = ``weight_rest_margins`` (unchanged — E084 compares
  this dedicated descent funnel against the reused ``go2_weight_rest_hi``)."""
  cfg = go2_weight_rest_env_cfg(play=play, lo=lo, hi=hi, load_height=load_height)
  # DISTURBED-STANDING spawn v2 (E083 post-mortem): the custom ``reset_standing_disturbance`` event CLOBBERED
  # ``reset_base``'s freshly-written pose with a STALE ``data.root_link_pos_w`` read (deterministic z=0.445
  # instead of 0.32 -> every episode began with a loaded drop-slam, median death 0.20s -> the descend funnel
  # cold-started to failure=1.0 TWICE). Fix: no custom event — mjlab's ``reset_root_state_uniform`` natively
  # supports roll/pitch pose ranges and a velocity_range, composed correctly from default_root_state. Softened
  # to ±0.15 rad / ±0.3 m/s (spawns must be survivable to train — the E044 lesson).
  rb = cfg.events["reset_base"]
  pr = dict(rb.params.get("pose_range", {}))
  pr["roll"] = (-0.15, 0.15); pr["pitch"] = (-0.15, 0.15)
  rb.params = {**rb.params, "pose_range": pr,
               "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}}
  return cfg


__all__ = [
  "getup_margins", "reset_prone", "reset_standing_disturbance",
  "go2_getup_env_cfg", "go2_descend_env_cfg",
  "GETUP_LO", "GETUP_HI", "DESCEND_LO", "DESCEND_HI",
]
