"""Landing sub-task: spawn mid-air over a gap with enough forward velocity to
reach the far platform, and learn to SOFT-LAND into a safe stance.

Avoid-only SafetyPPO test (uses g(s) only, ignores l): the winning-landing
signal is rare and gets buried/explodes at small env counts, so this task is
meant to be run at very large ``num_envs`` (mjlab parallelism) so enough
successful landings appear per iteration to learn from.

Reuses the gaps env (island terrain + height_scan proprioception, depth
dropped) and replaces the reset with a mid-air-over-gap launch.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from robot_safety_sandbox.envs.terrains.island import ISLAND_CROSSING_TERRAINS_CFG
from robot_safety_sandbox.envs.go2_gap.gap import unitree_go2_gap_reach_avoid_env_cfg

# [x, y, z, roll, pitch, yaw] offset relative to (env_origin + default pose);
# origin is the gap's near edge -> spawn just over the gap, elevated (apex).
_POSE_LOW = [0.05, -0.10, 0.25, -0.10, -0.15, -0.15]
_POSE_HIGH = [0.25, 0.10, 0.45, 0.10, 0.15, 0.15]
# [vx, vy, vz, wx, wy, wz]: strong forward velocity that clears the gap.
_VEL_LOW = [2.50, -0.10, -0.50, -0.20, -0.20, -0.20]
_VEL_HIGH = [3.50, 0.10, 0.30, 0.20, 0.20, 0.20]


def reset_midair_land(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Spawn each env mid-air over the gap with forward velocity that clears it."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()
  pose = sample_uniform(
    torch.tensor(_POSE_LOW, device=device), torch.tensor(_POSE_HIGH, device=device),
    (n, 6), device,
  )
  vel = sample_uniform(
    torch.tensor(_VEL_LOW, device=device), torch.tensor(_VEL_HIGH, device=device),
    (n, 6), device,
  )
  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
  )
  velocities = root[:, 7:13] + vel
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def unitree_go2_landing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(ISLAND_CROSSING_TERRAINS_CFG)
  cfg.events["reset_base"] = EventTermCfg(func=reset_midair_land, mode="reset", params={})
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  if not play and "push_robot" in cfg.events:
    cfg.events.pop("push_robot", None)  # avoid-only landing test: no extra disturbance
  return cfg
