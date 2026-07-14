"""Parkour-specific observation terms.

Includes depth camera observations, forward progress tracking, and gait phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor, ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def depth_image(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  far_clip: float = 3.0,
  near_clip: float = 0.01,
) -> torch.Tensor:
  """Get depth image from camera sensor, normalized and clipped.

  Returns:
    Tensor of shape (num_envs, 1, height, width) suitable for CNN input.
  """
  sensor: CameraSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.depth is not None, "Depth not enabled on camera sensor."
  depth = data.depth  # (num_envs, H, W, 1)

  # MuJoCo returns 0 for rays that hit no geometry (open air, gap void, sky).
  # Without this fix, gaps appear black (≈ wall at 1 cm) — indistinguishable
  # from a close obstacle and impossible for the CNN to learn from.
  # Remap no-hit pixels to far_clip so open space appears bright (far), not dark (near).
  depth = torch.where(depth < near_clip, torch.full_like(depth, far_clip), depth)
  depth = torch.clamp(depth, near_clip, far_clip)
  depth = (depth - near_clip) / (far_clip - near_clip)

  # Rearrange from (B, H, W, 1) to (B, 1, H, W) for CNN.
  depth = depth.permute(0, 3, 1, 2)

  return depth


def base_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Robot base height above ground."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def forward_distance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Robot forward (x) position — used by critic for progress awareness."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 0:1]


def body_pitch(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Robot body pitch angle, useful for crawl posture awareness."""
  asset: Entity = env.scene[asset_cfg.name]
  # Extract pitch from projected gravity.
  proj_grav = asset.data.projected_gravity_b  # (B, 3)
  # pitch ≈ arcsin(gx / g), but projected gravity provides gx directly
  return proj_grav[:, 0:1]


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(
  env: ManagerBasedRlEnv,
  period: float,
  command_name: str,
) -> torch.Tensor:
  """Sinusoidal gait clock for coordinated locomotion.

  Returns (sin, cos) of the gait phase. Zeroed when command is near zero.
  """
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  out = torch.zeros(env.num_envs, 2, device=env.device)
  out[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  out[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  command = env.command_manager.get_command(command_name)
  stand_mask = torch.linalg.norm(command, dim=1) < 0.1
  out = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(out), out)
  return out


def ray_distances(
  env: ManagerBasedRlEnv, sensor_name: str, max_distance: float = 4.0
) -> torch.Tensor:
  """Normalized raycast distances for non-down sensors (e.g. forward bar scan).

  The stock ``height_scan`` semantics (base height minus hit z) only make
  sense for downward grids; forward/up rays report the raw distance instead.
  Misses (distance < 0) map to 1.0 = "nothing within range".
  """
  scan = env.scene[sensor_name]
  d = scan.data.distances
  return torch.where(d < 0, torch.ones_like(d), (d / max_distance).clamp(0.0, 1.0))


def bar_info(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Privileged analytic bar observation: [dist_to_bar_face/4 (clamped ±1),
  under-beam clearance].  Requires the crawl-filter terrain (bar face at
  ``_BAR_X`` from each patch origin; clearance from the difficulty row)."""
  from robot_safety_sandbox.envs.terrains.crawl_filter import _BAR_X, bar_clearance_for_level

  terrain = env.scene.terrain
  robot: Entity = env.scene["robot"]
  bar_x = env.scene.env_origins[:, 0] + _BAR_X
  dist = ((bar_x - robot.data.root_link_pos_w[:, 0]) / 4.0).clamp(-1.0, 1.0)
  clearance = bar_clearance_for_level(terrain.terrain_levels)
  return torch.stack([dist, clearance], dim=1)
