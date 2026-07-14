"""Crawl safety FILTER (skill 2), momentum-reactive — mirrors the crossing-chain
jumping filter (see chat 2026-07-05).

Deployment model (identical to the gap line): a nominal walker drives the robot
forward; near a bar the crawl filter takes over and executes a SHORT maneuver —
duck-and-coast THROUGH a passable bar, or BRAKE to a stop before an impossible
one — then hands back.  The filter never learns to walk: it is spawned into the
takeover distribution (arrival momentum / real mid-gait walker states) and
learns to react.  This is why the gap line never trained locomotion — landing
spawned mid-air, crossing-chain spawned with arrival momentum, and the HANDOVER
stratum replays real go2_velocity walker states.

Reach objective: rest mode (l = come to a safe stop) + a per-row obstacle
window (``env._rest_obstacle_window_w``, read by the reach-avoid wrapper):
  * PASSABLE bar -> only rest PAST the bar counts (must duck-coast-through);
  * IMPOSSIBLE bar -> rest BEFORE the bar counts (must stop).

Terrain height curriculum (start on a bar the robot coasts under upright,
0.50 m, lower a notch on each crossing): the duck emerges continuously as the
bars descend, seeded by MID-BAR spawns (under the beam, coasting out) exactly
as the gap line's mid-air stratum seeded the landing.

Strata by row feasibility (an upright coast clears clearance >= 0.42):
  PASSABLE:   MUST-CROSS (duck-coast-through, reverse curriculum from the exit)
              + MID-BAR (under the beam, coasting out) + HANDOVER (walker states)
  IMPOSSIBLE: STOPPABLE (brake before) + DOOMED (unstoppable -> teaches V<0)
              + HANDOVER
"""

from __future__ import annotations

import copy
from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

import safe_mjlab_zoo.envs.parkour.mdp as mdp
from safe_mjlab_zoo.envs.terrains.crawl_filter import (
  BAR_DEPTH,
  CRAWL_FILTER_TERRAINS_CFG,
  _BAR_X,
  bar_clearance_for_level,
  is_impossible_level,
)
from safe_mjlab_zoo.envs.go2_gap.chain import (
  _handover_data,
  apply_handover_joints,
  phase_random_offset,
)

# --- momentum-reactive constants (mirror crossing_chain) ----------------------
_A_BRAKE = 3.0
_VX_CAP = 3.4
_NOSE = 0.35
_UPRIGHT_FIT = 0.42          # clearance an upright coast clears (trunk top ~0.38)
_CROSS_LEVELS = 5
_HANDOVER_LEVELS = 5

# Crouch joint poses (thigh, calf) and their standing-equilibrium base height,
# interpolated by alpha = clamp((0.30 - clearance)/0.08, 0, 1). Spawn z is set
# to the equilibrium (feet-borne) so the crouch doesn't penetrate the ground and
# eject the robot into the contact margin (the v1 spawn-transient bug).
_CROUCH_SHALLOW = (1.2, -2.3)   # base ~0.19
_CROUCH_DEEP = (1.35, -2.55)    # base ~0.15
_EQ_Z_SHALLOW = 0.166
_EQ_Z_DEEP = 0.146

# Spawn strata fractions (passable rows).
_FRAC_MUSTCROSS = 0.45
_FRAC_MIDBAR = 0.25
# remainder 0.30 = handover replay
# Impossible rows: STOPPABLE 0.5 / DOOMED 0.2 / HANDOVER 0.3.


def _ensure_crawl_buffers(env):
  if not hasattr(env, "_cross_level"):
    n, dev = env.num_envs, env.device
    env._cross_level = torch.zeros(n, dtype=torch.long, device=dev)
    env._was_mustcross = torch.zeros(n, dtype=torch.bool, device=dev)
    env._mustcross_approach = torch.zeros(n, dtype=torch.bool, device=dev)
    env._crouch_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    env._crouch_alpha = torch.zeros(n, device=dev)
    env._splay_mag = torch.zeros(n, device=dev)  # hip abduction beyond default
    env._duck_approach = torch.zeros(n, dtype=torch.bool, device=dev)  # started
    # before the bar (committed/approach) -> gates the duck height curriculum;
    # assist spawns (under/past the bar) do NOT gate (they'd promote for free).
    env._pit_start_x = torch.zeros(n, device=dev)  # v4 shrinking-island collapse
    # line start-x per env (set at reset ~PIT_BEHIND behind the spawn).
    env._leg_jitter = torch.full((n,), 0.02, device=dev)  # per-env crouch-pose
    # joint-jitter magnitude (reset raises it for diverse init leg configs).
    env._handover_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    env._handover_jpos = torch.zeros(n, 12, device=dev)
    env._handover_jvel = torch.zeros(n, 12, device=dev)
    env._handover_level = torch.zeros(n, dtype=torch.long, device=dev)
    env._rest_obstacle_window_w = torch.zeros(n, 2, device=dev)
    # Phase-1: set each step by crawl_locomote_margins when the base is past
    # the bar exit AND upright; read by the curriculum at reset.
    env._ever_crossed_upright = torch.zeros(n, dtype=torch.bool, device=dev)
    # Reverse spatial curriculum level (force-the-motion): 0 = spawn just short
    # of the exit, higher = progressively further back to the full approach.
    env._reach_level = torch.zeros(n, dtype=torch.long, device=dev)


def set_rest_obstacle_window(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Per-row rest window: PASSABLE -> exclude the whole approach (only past-bar
  rest counts -> must cross); IMPOSSIBLE -> target rest before the bar (stop)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  _ensure_crawl_buffers(env)
  impossible = is_impossible_level(env.scene.terrain.terrain_levels[env_ids])
  ox = env.scene.env_origins[env_ids, 0]
  lo = torch.where(impossible, ox + _BAR_X - 0.35, ox - 100.0)
  env._rest_obstacle_window_w[env_ids, 0] = lo
  env._rest_obstacle_window_w[env_ids, 1] = ox + _BAR_X + BAR_DEPTH + 0.40


def _crouch_z(alpha):
  return _EQ_Z_SHALLOW + (_EQ_Z_DEEP - _EQ_Z_SHALLOW) * alpha


def reset_takeover_crawl(env, env_ids, asset_cfg=SceneEntityCfg("robot"),
                         stop_margin: float = 0.0):
  """Takeover-distribution spawn, stratified by row feasibility."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  _ensure_crawl_buffers(env)
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), device)

  terrain = env.scene.terrain
  levels = terrain.terrain_levels[env_ids]
  clearance = bar_clearance_for_level(levels)
  impossible = is_impossible_level(levels)
  upright_fits = clearance >= _UPRIGHT_FIT
  alpha = ((0.30 - clearance) / 0.08).clamp(0.0, 1.0)  # crouch depth for this bar
  hdata = _handover_data(device)

  r = u(0.0, 1.0)
  # Passable-row strata.
  mustcross = (~impossible) & (r < _FRAC_MUSTCROSS)
  midbar = (~impossible) & (r >= _FRAC_MUSTCROSS) & (r < _FRAC_MUSTCROSS + _FRAC_MIDBAR)
  p_handover = (~impossible) & (r >= _FRAC_MUSTCROSS + _FRAC_MIDBAR)
  # Impossible-row strata.
  stoppable = impossible & (r < 0.5)
  doomed = impossible & (r >= 0.5) & (r < 0.7)
  i_handover = impossible & (r >= 0.7)
  handover = p_handover | i_handover
  if hdata is None:                      # fold handover into brake/cross
    mustcross = mustcross | p_handover
    stoppable = stoppable | i_handover
    handover = torch.zeros_like(handover)

  env._was_mustcross[env_ids] = mustcross
  env._handover_mask[env_ids] = handover
  crouch = torch.zeros(n, dtype=torch.bool, device=device)

  # --- defaults (z is ABSOLUTE base height) ---
  d = u(0.3, 2.2)                        # nose-distance to the bar face
  vx = u(0.0, 1.5)
  x = (_BAR_X - _NOSE - d).clamp(min=0.15)
  z_abs = 0.32 + u(-0.01, 0.03)          # near standing height
  vz = u(-0.05, 0.05)

  # --- STOPPABLE (impossible rows): brakeable arrival -> brake before bar ---
  d_s = u(0.5, 2.2)
  v_brake = torch.sqrt(2.0 * _A_BRAKE * (d_s - stop_margin).clamp_min(0.05))
  vx = torch.where(stoppable, (v_brake * u(0.15, 0.9)).clamp(0.0, _VX_CAP), vx)
  x = torch.where(stoppable, (_BAR_X - _NOSE - d_s).clamp(min=0.15), x)

  # --- DOOMED (impossible rows): unstoppable -> can't win, teaches V<0 ---
  vx_dm = u(2.6, _VX_CAP)
  d_dm = u(0.1, 0.6)
  vx = torch.where(doomed, vx_dm, vx)
  x = torch.where(doomed, (_BAR_X - _NOSE - d_dm).clamp(min=0.15), x)

  # --- MUST-CROSS (passable): reverse curriculum from the exit ---
  # assist=1 -> spawn AT the bar exit, ducked, coasting out (trivial settle);
  # assist->0 -> spawn approaching with momentum (must duck-coast-through).
  assist = (1.0 - env._cross_level[env_ids].float() / _CROSS_LEVELS).clamp(0, 1)
  at_exit = mustcross & (u(0.0, 1.0) < assist)
  approach = mustcross & ~at_exit
  env._mustcross_approach[env_ids] = approach  # only real approach-crossings gate terrain
  # approaching share: fast enough it must cross (unstoppable), varied distance
  d_mc = u(0.3, 1.4)
  vx_mc = torch.maximum(u(1.4, 2.6), torch.sqrt(2.0 * _A_BRAKE * d_mc) * 1.1)
  vx_mc = vx_mc.clamp(0.8, _VX_CAP)
  x = torch.where(approach, (_BAR_X - _NOSE - d_mc).clamp(min=0.15), x)
  vx = torch.where(approach, vx_mc, vx)
  # at-exit share: just past the bar exit, moving out slowly
  x = torch.where(at_exit, _BAR_X + BAR_DEPTH + u(0.0, 0.4), x)
  vx = torch.where(at_exit, u(0.6, 1.4), vx)

  # --- MID-BAR (passable): under the beam, ducked, coasting out ---
  x = torch.where(midbar, _BAR_X + u(0.0, BAR_DEPTH), x)
  vx = torch.where(midbar, u(0.8, 1.8), vx)

  # crouch ONLY the spawns that are UNDER the beam under a LOW bar (midbar and
  # the at-exit share): they'd strike upright. Approaching robots stay upright
  # and must duck DYNAMICALLY as they reach the bar (the learned skill); the
  # exit share under a HIGH bar also fits upright. Crouched spawns sit at the
  # crouch standing-equilibrium so they don't penetrate and eject.
  needs_low = ~upright_fits
  crouch = (midbar | at_exit) & needs_low
  env._crouch_mask[env_ids] = crouch
  env._crouch_alpha[env_ids] = alpha
  z_abs = torch.where(crouch, _crouch_z(alpha) + u(0.005, 0.02), z_abs)

  pose_xy = torch.stack([x, u(-0.06, 0.06)], dim=1)
  euler = torch.stack([u(-0.05, 0.05), u(-0.05, 0.05), u(-0.06, 0.06)], dim=1)
  velocities = root[:, 7:13] + torch.stack(
    [vx, u(-0.10, 0.10), vz, u(-0.10, 0.10), u(-0.10, 0.10), u(-0.10, 0.10)], dim=1)
  origins = env.scene.env_origins[env_ids]
  positions = torch.empty(n, 3, device=device)
  positions[:, 0:2] = root[:, 0:2] + pose_xy + origins[:, 0:2]
  positions[:, 2] = z_abs + origins[:, 2]
  orientations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]))

  # --- HANDOVER replay: real mid-gait walker states on the approach ---
  if hdata is not None and bool(handover.any()):
    h_idx = handover.nonzero().flatten()
    lvl = env._handover_level[env_ids][h_idx].float()
    n_rows = len(hdata["z"])
    lo_f = 0.15 * lvl
    hi_f = (lo_f + 0.30).clamp(max=1.0)
    frac = torch.rand(len(h_idx), device=device)
    rows = ((lo_f + frac * (hi_f - lo_f)) * (n_rows - 1)).long()
    ease = (1.0 - lvl / _HANDOVER_LEVELS)
    d_lo = 0.25 + ease * 0.75
    d_hi = 0.60 + ease * 1.15
    d_h = d_lo + torch.rand(len(h_idx), device=device) * (d_hi - d_lo)
    ox = env.scene.env_origins[env_ids][h_idx]
    positions[h_idx, 0] = ox[:, 0] + _BAR_X - _NOSE - d_h
    positions[h_idx, 1] = ox[:, 1] + sample_uniform(-0.06, 0.06, (len(h_idx),), device)
    positions[h_idx, 2] = hdata["z"][rows]
    orientations[h_idx] = hdata["quat"][rows]
    velocities[h_idx, 0:3] = hdata["lin_vel_w"][rows]
    velocities[h_idx, 3:6] = hdata["ang_vel_w"][rows]
    env._handover_jpos[env_ids[h_idx]] = hdata["joint_pos"][rows]
    env._handover_jvel[env_ids[h_idx]] = hdata["joint_vel"][rows]
    env._crouch_mask[env_ids[h_idx]] = False  # walker states are upright

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def apply_crouch_joints(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Write the clearance-matched crouch pose (after reset_robot_joints).
  Zero joint velocity + tight noise: crouch spawns are transient-fragile."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0 or not hasattr(env, "_crouch_mask"):
    return
  mask = env._crouch_mask[env_ids]
  if not bool(mask.any()):
    return
  asset = env.scene[asset_cfg.name]
  ids = env_ids[mask.nonzero().flatten()]
  m = int(len(ids))
  alpha = env._crouch_alpha[ids].unsqueeze(-1)
  jpos = asset.data.default_joint_pos[ids].clone()
  thigh = _CROUCH_SHALLOW[0] + (_CROUCH_DEEP[0] - _CROUCH_SHALLOW[0]) * alpha
  calf = _CROUCH_SHALLOW[1] + (_CROUCH_DEEP[1] - _CROUCH_SHALLOW[1]) * alpha
  for leg in range(4):
    jpos[:, 3 * leg + 1] = thigh.squeeze(-1)
    jpos[:, 3 * leg + 2] = calf.squeeze(-1)
  # Splayed hips (shoulders open, legs spread OUTWARD -> wide low base). Sign
  # verified from eval video: outward is +hip for the LEFT legs (FL=0, RL=6) and
  # -hip for the RIGHT legs (FR=3, RR=9) -- the opposite of the naive guess. The
  # default +-0.1 is kept so splay_mag = 0 (plain narrow crouch) is unchanged.
  if hasattr(env, "_splay_mag"):
    sm = env._splay_mag[ids]  # (m,)  outward = AWAY from body centerline
    jpos[:, 0] = -0.1 + sm     # FL (left)  -> more positive = outward
    jpos[:, 3] = 0.1 - sm      # FR (right) -> more negative = outward
    jpos[:, 6] = -0.1 + sm     # RL (left)
    jpos[:, 9] = 0.1 - sm      # RR (right)
  # Per-joint leg-config randomization: wider jitter (set by the reset via
  # _leg_jitter) makes the init leg configs DIVERSE so the policy stitches one
  # coherent forward-crawl controller from many states. Default 0.02 (tight).
  jit = (env._leg_jitter[ids].unsqueeze(-1) if hasattr(env, "_leg_jitter")
         else 0.02)
  jpos += sample_uniform(-1.0, 1.0, (m, 12), env.device) * jit
  asset.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), env_ids=ids)


# --- curricula (run at reset, before reset events; read ended-episode state) --

def _crossed_bar(env, env_ids):
  x = (env.scene["robot"].data.root_link_pos_w[env_ids, 0]
       - env.scene.env_origins[env_ids, 0])
  return (x > (_BAR_X + BAR_DEPTH)) & (x < 20.0)  # past exit, guard stale reads


def cross_assist_levels(env, env_ids) -> torch.Tensor:
  """Reverse curriculum on the MUST-CROSS maneuver: crossed past the bar and
  settled (timeout) -> promote (less exit assist, spawn approaching); fall ->
  demote."""
  _ensure_crawl_buffers(env)
  was = env._was_mustcross[env_ids]
  t_o = env.termination_manager.time_outs[env_ids]
  crossed = _crossed_bar(env, env_ids)
  lvl = env._cross_level[env_ids]
  lvl = torch.where(was & t_o & crossed, lvl + 1, lvl)
  lvl = torch.where(was & ~t_o, lvl - 1, lvl)
  env._cross_level[env_ids] = lvl.clamp(0, _CROSS_LEVELS)
  return env._cross_level.float().mean()


def handover_levels_crawl(env, env_ids) -> torch.Tensor:
  _ensure_crawl_buffers(env)
  was = env._handover_mask[env_ids]
  t_o = env.termination_manager.time_outs[env_ids]
  lvl = env._handover_level[env_ids]
  lvl = torch.where(was & t_o, lvl + 1, lvl)
  lvl = torch.where(was & ~t_o, lvl - 1, lvl)
  env._handover_level[env_ids] = lvl.clamp(0, _HANDOVER_LEVELS)
  return env._handover_level.float().mean()


def crawl_height_levels(env, env_ids) -> torch.Tensor | None:
  """Terrain (bar-height) curriculum gated on the BINDING skill: promote only
  when a MUST-CROSS-APPROACH env (started before the bar, not the trivial
  at-exit share) crossed and settled; demote on falls. Gating on real approach
  crossings keeps the terrain coupled to the skill (else the always-succeeding
  at-exit spawns run the bars ahead of what the robot can actually cross).
  Must run LAST among curricula: it mutates env_origins, which the origin-
  reading curricula (cross_assist) must see unchanged."""
  terrain = env.scene.terrain
  if terrain is None or not hasattr(terrain, "update_env_origins"):
    return None
  _ensure_crawl_buffers(env)
  t_o = env.termination_manager.time_outs[env_ids]
  approach = env._mustcross_approach[env_ids]
  crossed = _crossed_bar(env, env_ids)
  terrain.update_env_origins(env_ids, approach & t_o & crossed, ~t_o)
  return terrain.terrain_levels.float().mean()


def pinned_levels_crawl(env, env_ids) -> torch.Tensor:
  _ensure_crawl_buffers(env)
  env._cross_level[env_ids] = 4
  env._handover_level[env_ids] = 3
  return env._cross_level.float().mean()


# --- perception ---------------------------------------------------------------

def _add_bar_perception(cfg: ManagerBasedRlEnvCfg) -> None:
  bar_scan = RayCastSensorCfg(
    name="bar_scan", frame=ObjRef(type="body", name="base_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.0, 0.6), resolution=0.1, direction=(1.0, 0.0, 0.3)),
    max_distance=4.0, exclude_parent_body=True)
  bar_scan_low = RayCastSensorCfg(
    name="bar_scan_low", frame=ObjRef(type="body", name="base_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.0, 0.6), resolution=0.1, direction=(1.0, 0.0, 0.05)),
    max_distance=4.0, exclude_parent_body=True)
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (bar_scan, bar_scan_low)
  for name in ("bar_scan", "bar_scan_low"):
    cfg.observations["proprioception"].terms[name] = ObservationTermCfg(
      func=mdp.ray_distances, params={"sensor_name": name, "max_distance": 4.0})
    cfg.observations["critic"].terms[name] = ObservationTermCfg(
      func=mdp.ray_distances, params={"sensor_name": name, "max_distance": 4.0})
  cfg.observations["critic"].terms["bar_info"] = ObservationTermCfg(
    func=mdp.bar_info, params={})


# --- env cfg builders ---------------------------------------------------------

# =============================================================================
# PHASE 1 -- crouch-crawl LOCOMOTION (the "crossing" analog of gap-jumping).
# Objective is velocity-tracking (ReachAvoidPPO), NOT rest: avoid-only/rest lets
# the robot succeed by stopping, so the crouch-crawl motor skill never forms
# (diagnosed on the single-stage run). Here l rewards SUSTAINED forward motion,
# so the only way to score is to crawl THROUGH. Momentum init + a height
# curriculum from above-robot down bootstrap it; Phase 2 (the existing rest
# env) adds the stop-vs-crawl decision, warm-started from this.
# =============================================================================
V_CMD = 1.0               # target forward crawl speed (m/s), world +x
_MAX_PASSABLE_ROW = 10    # ROW_CLEARANCES index of 0.22 m (last passable row)
_BAR_EXIT = _BAR_X + BAR_DEPTH          # 3.3 m: past this == crossed
_CROSS_CLEAR = 0.5        # base this far past the exit == absorbing success
_FRAC_HANDOVER_LOCO = 0.5  # fraction spawned from real mid-gait walker states
_REACH_LEVELS = 8         # reverse spatial-curriculum levels


def reset_forced_crossing(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """FORCE-THE-MOTION spawn (the gap-crossing recipe). Every env spawns MID-GAIT
  from a real walker state (committed to forward motion -- standing still means
  actively braking against the momentum), at a position set by a REVERSE
  spatial curriculum: reach_level 0 spawns just short of the bar exit (a step or
  two clears), higher levels progressively further back to the full approach.
  The robot learns to keep galloping forward THROUGH the bar from the exit
  backward; braking never clears, so it never promotes. Bar is fixed high (row
  0, ~0.50 m) -- this run isolates the forward gallop, not the crouch."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  _ensure_crawl_buffers(env)
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()
  o = env.scene.env_origins[env_ids]

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), device)

  # reverse spatial curriculum: level 0 -> just short of the exit; max -> approach
  assist = 1.0 - env._reach_level[env_ids].float() / _REACH_LEVELS
  x_near = _BAR_EXIT - 0.1                 # a step short of clearing
  x_far = _BAR_X - _NOSE - 1.0             # full approach
  x_spawn = x_near + (x_far - x_near) * (1.0 - assist) + u(-0.1, 0.1)

  pos = root[:, 0:3] + o
  pos[:, 0] = o[:, 0] + x_spawn
  pos[:, 1] = o[:, 1]
  ori = root[:, 3:7].clone()
  vel = torch.zeros(n, 6, device=device)

  hdata = _handover_data(device)
  if hdata is not None:                    # committed mid-gait walker state
    n_rows = len(hdata["z"])
    rows = (torch.rand(n, device=device) * (n_rows - 1)).long()
    pos[:, 1] = o[:, 1] + sample_uniform(-0.06, 0.06, (n,), device)
    pos[:, 2] = hdata["z"][rows]
    ori = hdata["quat"][rows]
    vel[:, 0:3] = hdata["lin_vel_w"][rows]
    vel[:, 3:6] = hdata["ang_vel_w"][rows]
    env._handover_jpos[env_ids] = hdata["joint_pos"][rows]
    env._handover_jvel[env_ids] = hdata["joint_vel"][rows]
    env._handover_mask[env_ids] = True
  else:
    vel[:, 0] = V_CMD
    env._handover_mask[env_ids] = False
  env._crouch_mask[env_ids] = False
  env._ever_crossed_upright[env_ids] = False

  asset.write_root_link_pose_to_sim(
    torch.cat([pos, ori], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


# =============================================================================
# DUCK sub-task (avoid-only, SafetyPPO): approach a low bar with momentum; some
# spawns are STOPPABLE (brake before the bar) and some are UNSTOPPABLE (fast +
# close, so braking would collide -> the ONLY safe option is to DUCK). The
# robot learns brake-if-stoppable / duck-if-committed purely to stay safe --
# no reach term, no crossing incentive. Bar starts where a slight duck is
# needed and descends as the crouch deepens.
# =============================================================================
_DUCK_MIN_ROW = 3         # 0.39 m: GENTLE start (near standing trunk-top ~0.38 ->
                          # only a slight duck needed). Raised from row 6 (0.30 m,
                          # too hard as a fresh-policy start); the curriculum
                          # descends toward the deep ducks as the crawl firms up.
_DUCK_MAX_ROW = 10        # 0.22 m: near the crouch floor (deepest feasible duck)

# DUCK spawn STRATA (force engagement so rock-in-place isn't viable) + pose mix.
_FRAC_DUCK_COMMITTED = 0.35  # close + fast: braking collides -> MUST duck through
_FRAC_DUCK_APPROACH = 0.40   # farther + wide momentum: approach, then duck
# remainder 0.25 = ASSIST: spawn UNDER the bar, low, coasting out -> an easy
#   crossing that teaches the reach-avoid value crossing is SAFE (breaks the
#   risk-averse rock-in-place). Assist does NOT gate the height curriculum.
# Poses: HANDOVER (walker gait, approach only) / CROUCH (narrow) / SPLAY (hips
#   spread wide -> low, wide, STABLE base -- the posture the deep crouch tipped
#   out of).
_SPLAY_MAG = (0.35, 0.65)    # extra hip abduction beyond the default 0.1 (limit 1.05)


def reset_duck_approach(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Momentum approach with (a) spawn-position STRATA that force engagement and
  (b) a diverse init-pose MIX.

  Strata: COMMITTED (close + fast, braking collides -> must duck THROUGH),
  APPROACH (farther + wide momentum, must cross), ASSIST (spawn UNDER the bar,
  low, coasting out -> an easy crossing that teaches the reach-avoid value that
  crossing is SAFE, breaking the risk-averse rock-in-place). Committed+approach
  start before the bar and GATE the height curriculum (env._duck_approach);
  assist does not (it would promote for free). Poses: HANDOVER (walker gait,
  approach only) / CROUCH (narrow) / SPLAY (hips wide -> stable low base),
  applied by the crouch_joints / handover_joints reset events via the masks."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  _ensure_crawl_buffers(env)
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()
  o = env.scene.env_origins[env_ids]

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), device)

  hdata = _handover_data(device)
  have_h = hdata is not None

  # --- position strata ---
  rs = u(0.0, 1.0)
  committed = rs < _FRAC_DUCK_COMMITTED
  approach = (rs >= _FRAC_DUCK_COMMITTED) & (
    rs < _FRAC_DUCK_COMMITTED + _FRAC_DUCK_APPROACH)
  assist = rs >= (_FRAC_DUCK_COMMITTED + _FRAC_DUCK_APPROACH)

  x_c = (_BAR_X - _NOSE - u(0.1, 0.5)).clamp(min=0.1)   # committed: close
  x_a = (_BAR_X - _NOSE - u(0.5, 2.2)).clamp(min=0.1)   # approach: farther, wide
  x_as = _BAR_X + u(-0.1, BAR_DEPTH - 0.1)              # assist: under the bar
  x_rel = torch.where(committed, x_c, torch.where(approach, x_a, x_as))

  vx_c = u(1.8, _VX_CAP)                                # committed: fast (unstoppable)
  vx_a = u(0.3, 3.2)                                    # approach: WIDE momentum
  vx_as = u(0.5, 1.5)                                   # assist: gentle coast-out
  vx = torch.where(committed, vx_c, torch.where(approach, vx_a, vx_as))

  pos = root[:, 0:3] + o
  pos[:, 0] = o[:, 0] + x_rel
  pos[:, 1] = o[:, 1] + u(-0.06, 0.06)
  ori = root[:, 3:7].clone()
  vel = torch.zeros(n, 6, device=device)
  vel[:, 0] = vx

  # --- pose mix (per stratum) ---
  rp = u(0.0, 1.0)
  handover = torch.zeros(n, dtype=torch.bool, device=device)
  crouch = torch.zeros(n, dtype=torch.bool, device=device)
  splay = torch.zeros(n, dtype=torch.bool, device=device)
  # COMMITTED + ASSIST must be LOW (duck at speed / fit under the bar).
  low_strata = committed | assist
  splay |= low_strata & (rp < 0.5)
  crouch |= low_strata & (rp >= 0.5)
  # APPROACH: upright walker gait / crouch / splay (learn the upright->duck).
  if have_h:
    handover |= approach & (rp < 0.4)
    crouch |= approach & (rp >= 0.4) & (rp < 0.7)
    splay |= approach & (rp >= 0.7)
  else:
    crouch |= approach & (rp < 0.5)
    splay |= approach & (rp >= 0.5)

  low_mask = crouch | splay
  alpha = torch.where(splay, u(0.0, 0.4), u(0.2, 0.9))
  z_low = _crouch_z(alpha) - torch.where(
    splay, torch.full_like(alpha, 0.005), torch.zeros_like(alpha))
  pos[:, 2] = torch.where(low_mask, z_low, pos[:, 2])
  env._crouch_mask[env_ids] = low_mask
  env._crouch_alpha[env_ids] = alpha
  env._splay_mag[env_ids] = torch.where(
    splay, u(*_SPLAY_MAG), torch.zeros(n, device=device))

  # HANDOVER (approach only): real walker state (upright, momentum), full range.
  env._handover_mask[env_ids] = handover
  if have_h and bool(handover.any()):
    h = handover.nonzero().flatten()
    n_rows = len(hdata["z"])
    rows = (torch.rand(len(h), device=device) * (n_rows - 1)).long()
    pos[h, 2] = hdata["z"][rows]
    ori[h] = hdata["quat"][rows]
    vel[h, 0:3] = hdata["lin_vel_w"][rows]
    vel[h, 3:6] = hdata["ang_vel_w"][rows]
    env._handover_jpos[env_ids[h]] = hdata["joint_pos"][rows]
    env._handover_jvel[env_ids[h]] = hdata["joint_vel"][rows]

  # committed + approach start BEFORE the bar -> they gate the height curriculum.
  env._duck_approach[env_ids] = committed | approach

  asset.write_root_link_pose_to_sim(
    torch.cat([pos, ori], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


def crawl_duck_height_levels(env, env_ids):
  """Bar-height curriculum GATED ON CROSSING (not mere survival). Promote (LOWER
  the bar) only when an APPROACH env (started before the bar) actually got its
  base PAST the bar exit AND survived to timeout; demote when an approach env
  failed. Survived-but-didn't-cross (the rock-in-place exploit) neither promotes
  nor demotes -- so standing safely no longer ramps the curriculum for free.
  Assist spawns (start under/past the bar) don't gate. Band [0.39 m, 0.22 m]."""
  terrain = env.scene.terrain
  if terrain is None or not hasattr(terrain, "update_env_origins"):
    return None
  _ensure_crawl_buffers(env)
  approach = env._duck_approach[env_ids]
  t_o = env.termination_manager.time_outs[env_ids]
  crossed = _crossed_bar(env, env_ids)
  promote = approach & t_o & crossed
  demote = approach & ~t_o
  terrain.update_env_origins(env_ids, promote, demote)
  terrain.terrain_levels.clamp_(_DUCK_MIN_ROW, _DUCK_MAX_ROW)
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types]
  # duck_height is now itself an honest crossing signal: with the crossing gate,
  # it only rises when approach envs genuinely got past the bar.
  return terrain.terrain_levels.float().mean()


# =============================================================================
# v4: DENSE safe-trajectory init + shrinking-island temporal forcing.
# Spawn across the WHOLE crawl corridor (high-approach -> duck-approach ->
# duck-under -> duck-exit -> high-exit) so the value function sees every safe
# state and the policy CHAINS them (no need to explore into the ~1-2 cm corridor).
# The shrinking island (per-env collapse line, tasks/go2_crawl.py::_pit_edge)
# makes standing on the approach eventually fatal -> urgency kills the standing
# exploit at any bar height. Under-bar/exit are forever-safe (line capped at bar).
# =============================================================================
_FRAC_TRAJ_HIGH_APP = 0.20    # upright, before bar, forward momentum (TRANSIENT)
_FRAC_TRAJ_DUCK_APP = 0.20    # low duck, just before bar (TRANSIENT)
_FRAC_TRAJ_DUCK_UNDER = 0.25  # low duck, spanning UNDER the bar (FOREVER-SAFE)
_FRAC_TRAJ_DUCK_EXIT = 0.15   # low duck, at the exit (FOREVER-SAFE)
# remainder 0.20 = HIGH_EXIT: upright, past the exit (FOREVER-SAFE; highest safe
#   rate, zero success unless it keeps walking forward -> driven by velocity l)
_TRAJ_LEG_JITTER = 0.12       # wide per-joint jitter -> diverse init leg configs


def reset_crawl_trajectory(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """v4 dense init across the entire crawl trajectory, with diverse leg configs +
  momentum, and the shrinking-island collapse line seeded behind each spawn.

  Strata (position x posture): HIGH_APPROACH / DUCK_APPROACH (transient, before
  the bar -> gate the height curriculum) and DUCK_UNDER / DUCK_EXIT / HIGH_EXIT
  (forever-safe, past the pit cap). Poses applied by crouch_joints (low strata,
  diverse via _leg_jitter) and handover_joints (upright strata)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  from safe_mjlab_zoo.tasks.go2_crawl import _PIT_BEHIND  # lazy: avoid import cycle
  _ensure_crawl_buffers(env)
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()
  o = env.scene.env_origins[env_ids]

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), device)

  hdata = _handover_data(device)
  have_h = hdata is not None
  bar_exit = _BAR_X + BAR_DEPTH

  # --- trajectory strata ---
  rs = u(0.0, 1.0)
  c0 = _FRAC_TRAJ_HIGH_APP
  c1 = c0 + _FRAC_TRAJ_DUCK_APP
  c2 = c1 + _FRAC_TRAJ_DUCK_UNDER
  c3 = c2 + _FRAC_TRAJ_DUCK_EXIT
  high_app = rs < c0
  duck_app = (rs >= c0) & (rs < c1)
  duck_under = (rs >= c1) & (rs < c2)
  duck_exit = (rs >= c2) & (rs < c3)
  high_exit = rs >= c3

  # x_rel per stratum (base-x relative to the origin/bar)
  x_ha = _BAR_X - _NOSE - u(0.4, 2.0)          # upright, well before the bar
  x_da = _BAR_X - _NOSE - u(0.0, 0.6)          # ducked, front near/at the bar face
  x_du = u(_BAR_X - 0.2, bar_exit - 0.2)       # base UNDER the bar (2.3 .. 3.1)
  x_de = bar_exit + u(0.0, 0.3)                # ducked, just past the exit
  x_he = bar_exit + u(0.3, 1.5)               # upright, past the exit
  x_rel = torch.where(high_app, x_ha, torch.where(duck_app, x_da,
          torch.where(duck_under, x_du, torch.where(duck_exit, x_de, x_he))))
  x_rel = x_rel.clamp(min=0.1)

  # forward momentum per stratum (all move FORWARD; the velocity reach needs it)
  vx = torch.where(high_app, u(0.5, 3.0),
       torch.where(high_exit, u(0.3, 2.0), u(0.3, 2.0)))

  pos = root[:, 0:3] + o
  pos[:, 0] = o[:, 0] + x_rel
  pos[:, 1] = o[:, 1] + u(-0.08, 0.08)
  ori = root[:, 3:7].clone()
  vel = torch.zeros(n, 6, device=device)
  vel[:, 0] = vx

  # --- posture: upright strata vs low-duck strata ---
  upright = high_app | high_exit
  low_mask = duck_app | duck_under | duck_exit
  rp = u(0.0, 1.0)
  splay = low_mask & (rp < 0.5)
  crouch = low_mask & (rp >= 0.5)
  # diverse crouch depth + splay magnitude
  alpha = torch.where(splay, u(0.0, 0.5), u(0.2, 1.0))
  z_low = _crouch_z(alpha) - torch.where(
    splay, torch.full_like(alpha, 0.005), torch.zeros_like(alpha))
  pos[:, 2] = torch.where(low_mask, z_low, pos[:, 2])
  env._crouch_mask[env_ids] = low_mask
  env._crouch_alpha[env_ids] = alpha
  env._splay_mag[env_ids] = torch.where(
    splay, u(*_SPLAY_MAG), torch.zeros(n, device=device))
  env._leg_jitter[env_ids] = _TRAJ_LEG_JITTER  # diverse leg configs

  # HANDOVER (upright strata): real walker gait for a fraction (clean fwd prior)
  handover = torch.zeros(n, dtype=torch.bool, device=device)
  if have_h:
    handover = upright & (rp < 0.6)
  env._handover_mask[env_ids] = handover
  if have_h and bool(handover.any()):
    h = handover.nonzero().flatten()
    n_rows = len(hdata["z"])
    rows = (torch.rand(len(h), device=device) * (n_rows - 1)).long()
    pos[h, 2] = hdata["z"][rows]
    ori[h] = hdata["quat"][rows]
    vel[h, 0:3] = hdata["lin_vel_w"][rows]
    vel[h, 3:6] = hdata["ang_vel_w"][rows]
    env._handover_jpos[env_ids[h]] = hdata["joint_pos"][rows]
    env._handover_jvel[env_ids[h]] = hdata["joint_vel"][rows]

  # shrinking-island collapse line starts PIT_BEHIND behind each spawn; the cap
  # in _pit_edge leaves under-bar/exit forever-safe automatically.
  env._pit_start_x[env_ids] = pos[:, 0] - _PIT_BEHIND
  # only the transient approach strata gate the crossing-gated height curriculum
  env._duck_approach[env_ids] = high_app | duck_app

  asset.write_root_link_pose_to_sim(
    torch.cat([pos, ori], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


def caught_by_pit(env, env_ids=None):
  """Termination: the advancing collapse line passed the base (the shrinking
  island crumbled under it). Approach states are transient; under-bar/exit stay
  safe because _pit_edge is capped just before the bar face."""
  from safe_mjlab_zoo.tasks.go2_crawl import _pit_edge  # lazy: avoid import cycle
  edge = _pit_edge(env)
  if edge is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  base_x = env.scene["robot"].data.root_link_pos_w[:, 0]
  return base_x < edge


def unitree_go2_crawl_duck_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """DUCK sub-task (v4): dense safe-trajectory init + shrinking-island temporal
  forcing. Spawn across the whole corridor; standing on the approach eventually
  falls into the (virtual) pit; reach-avoid with forward-velocity reach."""
  cfg = unitree_go2_crawl_env_cfg(play=play)
  cfg.scene.terrain.max_init_terrain_level = _DUCK_MIN_ROW  # start where a duck is needed
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_crawl_trajectory, mode="reset", params={})
  # KEEP crouch_joints (crouch/SPLAY pose for _crouch_mask envs, diverse via
  # _leg_jitter) and handover_joints (walker gait for the upright strata).
  cfg.events.pop("rest_obstacle_window", None)
  # Wider default-joint jitter for the upright non-handover strata (diversity).
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.2, 0.2)
  # Shrinking-island temporal failure: standing on the approach -> caught -> end.
  cfg.terminations["caught_by_pit"] = TerminationTermCfg(func=caught_by_pit)
  cfg.curriculum = {
    "duck_height": CurriculumTermCfg(func=crawl_duck_height_levels)}
  return cfg


def unitree_go2_crawl_duck_video_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Packed-terrain + herd-camera variant of the duck task, for EVAL VIDEO ONLY.
  The normal 12x3 m bar patches spread the envs ~24 m apart -- too far for a herd
  render. Here the patches are narrow (8 x 1.7 m) so ~6-8 robots pack into one
  scene (the unitree-rl-mjlab herd look), viewed from a 3/4 side angle through the
  now near-transparent bars. Same task/margins/reset -- only geometry + camera
  differ (a slightly narrower track is fine for visualization)."""
  cfg = unitree_go2_crawl_duck_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(
    cfg.scene.terrain.terrain_generator, size=(8.0, 1.7), num_cols=8)
  cfg.viewer.body_name = "base_link"
  cfg.viewer.max_extra_envs = 8       # draw the whole packed row as one herd
  cfg.viewer.azimuth = 58.0
  cfg.viewer.elevation = -13.0
  cfg.viewer.distance = 6.8
  return cfg


def unitree_go2_crawl_walk_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """STAGE-1 dense-reward LOCOMOTION: learn a proper low-crawl GAIT via the env's
  own dense reward stack (train with stock PPO in dense mode -- NO reach-avoid
  g/l). track-velocity + feet_gait + feet_clearance + orientation + energy shape
  a real trot; a LOW body-height target makes it a ducked crawl; the bar enforces
  the low posture. This is the nominal controller the Stage-2 RA stop/crawl filter
  rides on (the two-stage decomposition -- l-shaping can't author a gait)."""
  cfg = unitree_go2_crawl_env_cfg(play=play)
  # Simple continuous-walk spawn (upright + forward momentum + walker gait prior);
  # no pit / dense-init strata / rest window (those are Stage-2 safety machinery).
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_momentum_approach, mode="reset", params={})
  cfg.events.pop("rest_obstacle_window", None)
  # LOW body-height target -> a ducked crawl gait (Go2 standing ~0.30).
  if "body_height" in cfg.rewards:
    cfg.rewards["body_height"].params["target_height"] = 0.22
  # Drop the gap-specific rewards (no gap on the bar terrain; they read 0/noise).
  for k in ("gap_crossing", "gap_crossing_bonus"):
    cfg.rewards.pop(k, None)
  # Start on a moderate bar that forces a mild duck; fixed for v1 (add a height
  # curriculum once the base gait is clean).
  cfg.scene.terrain.max_init_terrain_level = 5   # 0.33 m clearance
  cfg.curriculum = {}
  return cfg


def unitree_go2_crawl_walk_video_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Packed-terrain herd render of the Stage-1 walker (eval video only)."""
  cfg = unitree_go2_crawl_walk_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(
    cfg.scene.terrain.terrain_generator, size=(8.0, 1.7), num_cols=8)
  cfg.viewer.body_name = "base_link"
  cfg.viewer.max_extra_envs = 8
  cfg.viewer.azimuth = 58.0
  cfg.viewer.elevation = -13.0
  cfg.viewer.distance = 6.8
  return cfg


def forced_crossing_reach_levels(env, env_ids):
  """Reverse spatial curriculum: promote (spawn further back) when the env
  cleared the bar this episode, demote otherwise. Each env settles at the
  spawn-distance it can gallop across; the mean = how far back the learned
  forward-crossing reaches."""
  _ensure_crawl_buffers(env)
  crossed = env._ever_crossed_upright[env_ids]
  lvl = env._reach_level[env_ids]
  lvl = torch.where(crossed, lvl + 1, lvl - 1)
  env._reach_level[env_ids] = lvl.clamp(0, _REACH_LEVELS)
  return env._reach_level.float().mean()


def reset_momentum_approach(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Phase-1 spawn (passable rows): MIX of upright momentum-approach and real
  mid-gait walker states (handover = gait prior). Crouch is learned via the
  descending height curriculum; the crossed_success termination ends the
  episode on crossing so short episodes count as wins."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  _ensure_crawl_buffers(env)
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()
  o = env.scene.env_origins[env_ids]

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), device)

  pos = root[:, 0:3] + o
  ori = root[:, 3:7].clone()
  vel = torch.zeros(n, 6, device=device)
  # upright momentum approach (default)
  pos[:, 0] = o[:, 0] + (_BAR_X - _NOSE - u(0.5, 1.5))
  pos[:, 1] = o[:, 1]
  vel[:, 0] = V_CMD + u(-0.2, 0.4)

  # handover walker states (gait prior)
  hdata = _handover_data(device)
  handover = torch.zeros(n, dtype=torch.bool, device=device)
  if hdata is not None:
    handover = u(0.0, 1.0) < _FRAC_HANDOVER_LOCO
    h = handover.nonzero().flatten()
    if len(h) > 0:
      n_rows = len(hdata["z"])
      rows = (torch.rand(len(h), device=device) * (n_rows - 1)).long()
      d_h = 0.3 + torch.rand(len(h), device=device) * 1.2
      oh = o[h]
      pos[h, 0] = oh[:, 0] + _BAR_X - _NOSE - d_h
      pos[h, 1] = oh[:, 1] + sample_uniform(-0.06, 0.06, (len(h),), device)
      pos[h, 2] = hdata["z"][rows]
      ori[h] = hdata["quat"][rows]
      vel[h, 0:3] = hdata["lin_vel_w"][rows]
      vel[h, 3:6] = hdata["ang_vel_w"][rows]
      env._handover_jpos[env_ids[h]] = hdata["joint_pos"][rows]
      env._handover_jvel[env_ids[h]] = hdata["joint_vel"][rows]
  env._handover_mask[env_ids] = handover
  env._crouch_mask[env_ids] = False
  env._ever_crossed_upright[env_ids] = False   # fresh per episode

  asset.write_root_link_pose_to_sim(
    torch.cat([pos, ori], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


def crawl_locomote_height_levels(env, env_ids):
  """Height-only curriculum (passable rows): promote an env that crossed past
  the bar AND survived to timeout; demote on falls. Capped at the passable
  floor (0.22 m) -- impossible bars/stop are Phase 2. Start on the highest bar
  (max_init_terrain_level=0) so the upright momentum coast wins from step 0."""
  terrain = env.scene.terrain
  if terrain is None or not hasattr(terrain, "update_env_origins"):
    return None
  _ensure_crawl_buffers(env)
  # Promote on a CLEAN crossing at any point this episode (base past the bar
  # while upright), demote otherwise. Episodes are short (the gallop crosses
  # then may fall) so "crossed & survived-to-timeout" never fires; and a level
  # too high just demotes, so momentum-coasting can't ratchet the bar up.
  crossed = env._ever_crossed_upright[env_ids]
  terrain.update_env_origins(env_ids, crossed, ~crossed)
  terrain.terrain_levels.clamp_(0, _MAX_PASSABLE_ROW)
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types]
  return terrain.terrain_levels.float().mean()


def unitree_go2_crawl_locomote_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Phase 1: crouch-crawl LOCOMOTION on passable bars. Momentum-init approach,
  height curriculum high->low, NO rest window / crouch seed / handover / stop.
  Pair with crawl_locomote_margins (velocity-tracking l) via ReachAvoidPPO."""
  cfg = unitree_go2_crawl_env_cfg(play=play)
  # FORCE THE MOTION: committed mid-gait spawns + a reverse spatial curriculum
  # so "stand still" is never an option (the gap-crossing recipe). The bar
  # stays high (row 0), so this run learns the sustained forward GALLOP; a
  # follow-up warm-starts and lowers the bar to add the crouch.
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_forced_crossing, mode="reset", params={})
  cfg.events.pop("crouch_joints", None)
  # KEEP handover_joints: applies the committed walker joint states.
  cfg.events.pop("rest_obstacle_window", None)  # no rest objective in Phase 1
  cfg.curriculum = {
    "reach_level": CurriculumTermCfg(func=forced_crossing_reach_levels)}
  return cfg


def unitree_go2_crawl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  from safe_mjlab_zoo.envs.go2_gap.gap import (
    unitree_go2_gap_reach_avoid_env_cfg,
  )

  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(CRAWL_FILTER_TERRAINS_CFG)
  cfg.scene.terrain.max_init_terrain_level = 0     # start on the highest bar
  cfg.episode_length_s = 8.0

  # CONTACT FAILURE SET = TRUNK + HIP only (thigh/calf excluded). Diagnosed on a
  # trained checkpoint: 99% of terminations were illegal_contact, but the robot
  # was ducking correctly (base_z 0.20 < the 0.30 m bar) and moving forward -- it
  # was the FRONT THIGHS/KNEES scraping the approach ground (76% of failures had
  # a thigh <0.06 m off the floor, 0.3 m BEFORE the bar). A crawl low enough to
  # clear the bar necessarily drops the thighs to the floor, and full non-foot
  # contact killed that legal crawl as if it were a bar strike. Excluding the
  # lower limbs (thigh + calf) makes the low crawl legal; a torso-into-bar or
  # belly-slam (base geoms) or a hip drop still fails. This does NOT reopen the
  # old avoid-only lean-back exploit: the REACH term (forward velocity) now
  # punishes any non-forward motion, so leaning back onto the knees fails l.
  _CRAWL_CONTACT_OK = tuple(
    f"{leg}_{seg}_collision"
    for leg in ("FL", "FR", "RL", "RR")
    for seg in ("thigh", "calf1", "calf2"))
  _sensors = []
  for _s in (cfg.scene.sensors or ()):
    if getattr(_s, "name", None) == "nonfoot_ground_touch":
      _s = replace(_s, primary=replace(
        _s.primary, exclude=tuple(_s.primary.exclude) + _CRAWL_CONTACT_OK))
    _sensors.append(_s)
  cfg.scene.sensors = tuple(_sensors)

  _add_bar_perception(cfg)

  cfg.events["reset_base"] = EventTermCfg(func=reset_takeover_crawl, mode="reset", params={})
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  cfg.events["crouch_joints"] = EventTermCfg(func=apply_crouch_joints, mode="reset", params={})
  cfg.events["handover_joints"] = EventTermCfg(func=apply_handover_joints, mode="reset", params={})
  cfg.events["rest_obstacle_window"] = EventTermCfg(func=set_rest_obstacle_window, mode="reset", params={})
  cfg.events.pop("push_robot", None)
  cfg.events.pop("randomize_terrain", None)

  # Phase-offset-invariant gait clock (deployment handovers at arbitrary phase).
  for gname in ("proprioception", "critic"):
    term = copy.deepcopy(cfg.observations[gname].terms["phase"])
    period = float(term.params.get("period", 0.5))
    term.func = phase_random_offset
    term.params = {"period": period}
    cfg.observations[gname].terms["phase"] = term

  # Drop height_scan (935-dim downward terrain grid, inherited from the parkour
  # lineage). This task's ground is FLAT; the only obstacle -- the overhead bar
  # -- is sensed by bar_scan/bar_scan_low, not a terrain height grid. Removing it
  # slims the obs 1245 -> ~310 and cuts a near-constant distractor block the
  # policy otherwise has to learn to ignore (2026-07-09, user-directed).
  for gname in ("proprioception", "critic"):
    cfg.observations[gname].terms.pop("height_scan", None)

  # ORDER MATTERS: cross_assist reads env_origins (via _crossed_bar) and
  # crawl_height_levels MUTATES env_origins (update_env_origins) -> terrain
  # MUST run last, else cross_assist reads moved origins and never promotes.
  cfg.curriculum = {
    "cross_assist": CurriculumTermCfg(func=cross_assist_levels),
    "handover_level": CurriculumTermCfg(func=handover_levels_crawl),
    "terrain_levels": CurriculumTermCfg(func=crawl_height_levels),
  }

  # LOW side-profile follow camera: azimuth 90 (robot moving L->R) + a nearly
  # LEVEL elevation (-5) close in (2.6 m) so the camera looks UNDER the beam and
  # the crawling body stays visible. The old -14 deg / 3.3 m looked down onto the
  # bar's wall/pillars and occluded the under-bar crawl (hard to judge learning).
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.6
  cfg.viewer.elevation = -5.0
  cfg.viewer.azimuth = 90.0
  return cfg


def unitree_go2_crawl_isaacs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_crawl_env_cfg(play=play)
  cfg.events["reset_base"].params["stop_margin"] = 0.3
  cfg.curriculum = {"pinned_levels": CurriculumTermCfg(func=pinned_levels_crawl)}
  return cfg
