"""Event terms for safety training on parkour terrain.

The key event is :func:`reset_robot_midair_over_gaps`, which spawns a
fraction of the robots elevated above the terrain with a strong forward
velocity — giving the safety policy explicit exposure to mid-air
transitions over gaps.  Without such initialisation, a purely on-ground
spawn + walking command will rarely place the robot mid-flight, so the
policy never learns to recover/land after a jump.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_robot_midair_over_gaps(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  *,
  midair_fraction: float = 0.5,
  ground_pose_range: dict[str, tuple[float, float]] | None = None,
  ground_velocity_range: dict[str, tuple[float, float]] | None = None,
  midair_x_range: tuple[float, float] = (1.0, 6.5),
  midair_y_range: tuple[float, float] = (-0.2, 0.2),
  midair_z_range: tuple[float, float] = (0.35, 0.75),
  midair_vx_range: tuple[float, float] = (1.5, 3.5),
  midair_vy_range: tuple[float, float] = (-0.2, 0.2),
  midair_vz_range: tuple[float, float] = (-1.0, 0.5),
  midair_roll_range: tuple[float, float] = (-0.25, 0.25),
  midair_pitch_range: tuple[float, float] = (-0.25, 0.25),
  midair_yaw_range: tuple[float, float] = (-0.3, 0.3),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset robot with a mix of on-ground and strategic mid-air spawns.

  A ``midair_fraction`` of the resetting envs are spawned at a
  randomised forward offset along the terrain patch, elevated above
  the platform, and given a forward linear velocity.  Because the
  forward offset covers the full patch length, a significant share of
  these spawns lands over a gap — the safety policy is therefore
  forced to learn how to bridge gaps during recovery, even though the
  underlying task/velocity policy only knows how to walk.

  The remaining envs reset normally within ``ground_pose_range`` /
  ``ground_velocity_range`` to keep a healthy baseline of walking
  states in the replay.

  Parameters
  ----------
  midair_fraction:
      Fraction of reset envs that receive a mid-air spawn (0-1).
  ground_pose_range / ground_velocity_range:
      Pose/velocity ranges for normal on-ground resets (same semantics
      as :func:`mjlab.envs.mdp.events.reset_root_state_uniform`).
  midair_x_range:
      Forward offset (m) along the patch.  Should span gap locations.
  midair_y_range:
      Lateral offset (m) relative to the patch origin.
  midair_z_range:
      Additional height (m) above the default spawn height — the robot
      is placed mid-flight above the platform.
  midair_v{x,y,z}_range:
      Linear velocity (m/s) imparted at spawn.  ``vx`` should be
      positive (forward).
  midair_{roll,pitch,yaw}_range:
      Orientation perturbation (rad) about the default upright pose.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  if len(env_ids) == 0:
    return

  if ground_pose_range is None:
    ground_pose_range = {
      "x": (-0.3, 0.3),
      "y": (-0.2, 0.2),
      "z": (0.0, 0.0),
      "yaw": (-0.2, 0.2),
    }
  if ground_velocity_range is None:
    ground_velocity_range = {}

  asset: Entity = env.scene[asset_cfg.name]
  assert not asset.is_fixed_base, "Mid-air reset only supports floating-base robots."

  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  num_resets = int(len(env_ids))
  device = env.device

  # Decide which envs get a mid-air spawn.
  sample = torch.rand(num_resets, device=device)
  is_midair = sample < float(midair_fraction)

  # --- Ground resets (sampled from ground_pose_range / ground_velocity_range) ---
  ground_pose_list = [
    ground_pose_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ground_pose_t = torch.tensor(ground_pose_list, device=device)
  ground_pose_samples = sample_uniform(
    ground_pose_t[:, 0], ground_pose_t[:, 1], (num_resets, 6), device=device
  )

  ground_vel_list = [
    ground_velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ground_vel_t = torch.tensor(ground_vel_list, device=device)
  ground_vel_samples = sample_uniform(
    ground_vel_t[:, 0], ground_vel_t[:, 1], (num_resets, 6), device=device
  )

  # --- Mid-air resets ---
  midair_pose_list = [
    midair_x_range,
    midair_y_range,
    midair_z_range,
    midair_roll_range,
    midair_pitch_range,
    midair_yaw_range,
  ]
  midair_pose_t = torch.tensor(midair_pose_list, device=device)
  midair_pose_samples = sample_uniform(
    midair_pose_t[:, 0], midair_pose_t[:, 1], (num_resets, 6), device=device
  )

  midair_vel_list = [
    midair_vx_range,
    midair_vy_range,
    midair_vz_range,
    (-0.2, 0.2),  # roll rate
    (-0.2, 0.2),  # pitch rate
    (-0.3, 0.3),  # yaw rate
  ]
  midair_vel_t = torch.tensor(midair_vel_list, device=device)
  midair_vel_samples = sample_uniform(
    midair_vel_t[:, 0], midair_vel_t[:, 1], (num_resets, 6), device=device
  )

  # Merge ground and mid-air samples according to is_midair.
  mask = is_midair.unsqueeze(-1).float()
  pose_samples = ground_pose_samples * (1.0 - mask) + midair_pose_samples * mask
  vel_samples = ground_vel_samples * (1.0 - mask) + midair_vel_samples * mask

  # --- Apply to sim ---
  positions = (
    root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
  )
  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  velocities = root_states[:, 7:13] + vel_samples

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)
