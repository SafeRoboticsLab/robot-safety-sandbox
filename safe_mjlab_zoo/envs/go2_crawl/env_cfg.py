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
    env._handover_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    env._handover_jpos = torch.zeros(n, 12, device=dev)
    env._handover_jvel = torch.zeros(n, 12, device=dev)
    env._handover_level = torch.zeros(n, dtype=torch.long, device=dev)
    env._rest_obstacle_window_w = torch.zeros(n, 2, device=dev)


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
  jpos += sample_uniform(-0.02, 0.02, (m, 12), env.device)
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
_FRAC_HANDOVER_LOCO = 0.5  # fraction spawned from real mid-gait walker states


def reset_momentum_approach(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Phase-1 spawn: MIX of (a) real mid-gait walker states on the approach
  (handover -- the GAIT PRIOR that lets the robot learn sustained crawling
  instead of the lunge-and-crash local optimum; safety RL strips gait shaping,
  so the walking manifold must be seeded through the state distribution, as the
  gap CHAIN did) and (b) upright momentum-approach. Passable rows; crouch is
  learned via the descending height curriculum."""
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
  # (b) default: upright momentum approach
  d = u(0.5, 1.5)
  pos[:, 0] = o[:, 0] + (_BAR_X - _NOSE - d)
  pos[:, 1] = o[:, 1]
  vel[:, 0] = V_CMD + u(-0.2, 0.4)

  # (a) handover walker states (gait prior)
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
  t_o = env.termination_manager.time_outs[env_ids]
  crossed = _crossed_bar(env, env_ids)
  terrain.update_env_origins(env_ids, crossed & t_o, ~t_o)
  terrain.terrain_levels.clamp_(0, _MAX_PASSABLE_ROW)
  terrain.env_origins[:] = terrain.terrain_origins[
    terrain.terrain_levels, terrain.terrain_types]
  return terrain.terrain_levels.float().mean()


def unitree_go2_crawl_locomote_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Phase 1: crouch-crawl LOCOMOTION on passable bars. Momentum-init approach,
  height curriculum high->low, NO rest window / crouch seed / handover / stop.
  Pair with crawl_locomote_margins (velocity-tracking l) via ReachAvoidPPO."""
  cfg = unitree_go2_crawl_env_cfg(play=play)
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_momentum_approach, mode="reset", params={})
  cfg.events.pop("crouch_joints", None)        # crouch is learned, not seeded
  # KEEP handover_joints: applies the walker joint states for the handover
  # gait-prior fraction (after reset_robot_joints); without it the handover
  # spawns get default joints and lose the gait.
  cfg.events.pop("rest_obstacle_window", None)  # no rest objective in Phase 1
  cfg.curriculum = {
    "crawl_height": CurriculumTermCfg(func=crawl_locomote_height_levels)}
  return cfg


def unitree_go2_crawl_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  from safe_mjlab_zoo.envs.go2_gap.gap import (
    unitree_go2_gap_reach_avoid_env_cfg,
  )

  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(CRAWL_FILTER_TERRAINS_CFG)
  cfg.scene.terrain.max_init_terrain_level = 0     # start on the highest bar
  cfg.episode_length_s = 8.0

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

  # ORDER MATTERS: cross_assist reads env_origins (via _crossed_bar) and
  # crawl_height_levels MUTATES env_origins (update_env_origins) -> terrain
  # MUST run last, else cross_assist reads moved origins and never promotes.
  cfg.curriculum = {
    "cross_assist": CurriculumTermCfg(func=cross_assist_levels),
    "handover_level": CurriculumTermCfg(func=handover_levels_crawl),
    "terrain_levels": CurriculumTermCfg(func=crawl_height_levels),
  }

  # Side-profile follow camera (rsl_rl crawl setting): azimuth 90 with the
  # robot moving L->R into the bar keeps the body unoccluded by the low bar
  # (the inherited gap 3/4 top-down view azimuth 130 / elevation -35 hides it).
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 3.3
  cfg.viewer.elevation = -14.0
  cfg.viewer.azimuth = 90.0
  return cfg


def unitree_go2_crawl_isaacs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_crawl_env_cfg(play=play)
  cfg.events["reset_base"].params["stop_margin"] = 0.3
  cfg.curriculum = {"pinned_levels": CurriculumTermCfg(func=pinned_levels_crawl)}
  return cfg
