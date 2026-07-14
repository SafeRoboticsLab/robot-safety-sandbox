"""Crossing task: REVERSE CURRICULUM that extends backward from the learned
landing to the launch.

Every env starts at ``back_level=0`` = the landing state (apex over the gap,
full clearing velocity — the sub-skill SafetyPPO already learns). An env that
reaches the far platform upright is promoted to spawn *further back* along the
arc (toward a committed launch off the near edge); one that fails is demoted.
So the working landing bootstraps the launch, per-env, like the terrain-level
curriculum. Avoid-only ``g`` (SafetyPPO) still forces it because every level
spawns airborne-committed over the gap (falling in -> g<0), so standing is never
an option. Meant for large ``num_envs`` (mjlab parallelism).

(Generating the jump from a full standstill needs the reach target l or the
adversary; this curriculum covers landing -> launch with g only.)
"""

from __future__ import annotations

from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from robot_safety_sandbox.envs.terrains.island import ISLAND_CROSSING_TERRAINS_CFG
from robot_safety_sandbox.envs.go2_gap.gap import unitree_go2_gap_reach_avoid_env_cfg

L_LEVELS = 5  # 0 = landing (apex) ... L = committed launch (near edge)
_FAR_X = 0.7  # x_rel beyond which the robot is on the far platform (all gaps <=0.6)


def _ensure_back_level(env):
  if not hasattr(env, "_back_level"):
    env._back_level = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def reset_crossing_reverse(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Spawn each env along the arc per its back_level (0 apex-land -> L launch)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  _ensure_back_level(env)
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()
  f = (env._back_level[env_ids].float() / L_LEVELS).clamp(0.0, 1.0)  # (n,)

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), device)

  # Interpolate spawn state: f=0 apex-landing -> f=1 committed launch at near edge.
  x = (0.15 - 0.25 * f) + u(-0.05, 0.05)
  y = u(-0.10, 0.10)
  z = (0.35 - 0.30 * f) + u(-0.05, 0.05)
  pose = torch.stack([x, y, z, u(-0.10, 0.10), u(-0.15, 0.15), u(-0.15, 0.15)], dim=1)
  vx = (3.0 - 0.5 * f) + u(-0.20, 0.20)
  vz = (0.0 + 0.6 * f) + u(-0.20, 0.20)
  vel = torch.stack([vx, u(-0.10, 0.10), vz, u(-0.2, 0.2), u(-0.2, 0.2), u(-0.2, 0.2)], dim=1)

  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
  )
  velocities = root[:, 7:13] + vel
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def crossing_reverse_levels(env, env_ids) -> torch.Tensor:
  """Promote an env's start further back when it reached the far platform
  upright; demote otherwise. Read on reset (ended-episode final state)."""
  _ensure_back_level(env)
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
  up = -robot.data.projected_gravity_b[env_ids, 2]
  # Promote only on a *successful* landing: survived the full episode (timed out,
  # not crash-terminated) AND ended on the far platform upright. Using x_rel alone
  # would promote on the ballistic fly-past even when the landing crashes.
  survived = env.termination_manager.time_outs[env_ids]
  reached = survived & (x_rel > _FAR_X) & (up > 0.7)
  lvl = env._back_level[env_ids]
  lvl = torch.where(reached, lvl + 1, lvl - 1)
  env._back_level[env_ids] = lvl.clamp(0, L_LEVELS)
  return env._back_level.float().mean()


def unitree_go2_crossing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(ISLAND_CROSSING_TERRAINS_CFG)
  cfg.events["reset_base"] = EventTermCfg(func=reset_crossing_reverse, mode="reset", params={})
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  if not play and "push_robot" in cfg.events:
    cfg.events.pop("push_robot", None)
  cfg.curriculum = {"crossing_levels": CurriculumTermCfg(func=crossing_reverse_levels)}
  return cfg
