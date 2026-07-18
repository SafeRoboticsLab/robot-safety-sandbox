"""Harvest env for the split test: a SMALL fixed gap on a long island, with
walk-in spawns spanning a momentum sweep, used to (1) probe which source policy
crosses from low->moderate momentum and (2) harvest successful full-trajectory
sim states for the reverse curriculum.

Reused by the reverse-curriculum trainer later; here only the harvest env +
walk-in reset live.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from robot_safety_sandbox.envs.terrains.island import IslandCrossingTerrainCfg
from robot_safety_sandbox.envs.go2_gap.gap import unitree_go2_gap_reach_avoid_env_cfg


def harvest_terrain(gap_width: float, island_length: float = 3.0):
  return TerrainGeneratorCfg(
    curriculum=False, size=(8.0, 2.0), border_width=5.0,
    num_rows=1, num_cols=10, color_scheme="none",
    sub_terrains={"island": IslandCrossingTerrainCfg(
      proportion=1.0, gap_width_range=(gap_width, gap_width), gap_depth=1.0,
      island_length=island_length, back_pit_length=0.8)},
  )


def reset_harvest_walkin(env, env_ids, asset_cfg=SceneEntityCfg("robot"),
                         x_range=(-1.5, -0.2), vx_range=(0.0, 1.5)):
  """Walk-in spawns on the near platform: standing pose, forward momentum swept
  across vx_range; the source policy (forward cmd) then walks in and (maybe)
  crosses. Records the spawn vx on env._harvest_v0 for momentum binning."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  asset = env.scene[asset_cfg.name]
  dev = env.device
  n = int(len(env_ids))

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), dev)

  root = asset.data.default_root_state[env_ids].clone()
  dx = u(x_range[0], x_range[1])
  dy = u(-0.12, 0.12)
  vx = u(vx_range[0], vx_range[1])
  if not hasattr(env, "_harvest_v0"):
    env._harvest_v0 = torch.zeros(env.num_envs, device=dev)
  env._harvest_v0[env_ids] = vx
  pose = torch.stack(
    [dx, dy, torch.zeros(n, device=dev), u(-0.06, 0.06), u(-0.08, 0.08),
     u(-0.12, 0.12)], dim=1)
  vel = torch.stack(
    [vx, u(-0.10, 0.10), u(-0.10, 0.10), u(-0.2, 0.2), u(-0.2, 0.2),
     u(-0.2, 0.2)], dim=1)
  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]))
  velocities = root[:, 7:13] + vel
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def unitree_go2_harvest_env_cfg(play: bool = False, gap_width: float = 0.06,
                                x_range=(-1.5, -0.2), vx_range=(0.0, 1.5)):
  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = harvest_terrain(gap_width)
  cfg.episode_length_s = 6.0
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_harvest_walkin, mode="reset",
    params={"x_range": x_range, "vx_range": vx_range})
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  if "push_robot" in cfg.events:
    cfg.events.pop("push_robot", None)
  cfg.curriculum = {}
  return cfg
