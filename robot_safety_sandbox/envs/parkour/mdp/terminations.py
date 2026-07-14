"""Parkour-specific termination conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Terminate when non-foot body parts contact obstacles with force above threshold."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)
  assert data.found is not None
  force_mag = torch.norm(data.force, dim=-1) if data.force is not None else data.found.float()
  return (force_mag > force_threshold).any(dim=-1)


def base_too_low(
  env: ManagerBasedRlEnv,
  min_height: float = 0.12,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Terminate when robot base drops below minimum height (fell into gap)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] < min_height


def base_too_high(
  env: ManagerBasedRlEnv,
  max_height: float = 1.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Terminate when robot base is too high (launched/glitched)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] > max_height


def base_below_local_terrain(
  env: ManagerBasedRlEnv,
  min_clearance: float = 0.08,
  sensor_name: str = "terrain_scan",
  footprint_radius: float = 0.5,
  obstacle_margin: float = 0.12,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Terminate when the base sinks below the *local* platform.

  Unlike :func:`base_too_low` (absolute world-z), this is terrain-relative so it
  detects falling into a gap or off a cliff onto lower ground regardless of the
  platform's absolute elevation.  It mirrors the height component of the
  reach-avoid safety margin (see ``ParkourReachAvoidVecEnvWrapper``): the
  reference ground is the highest raycast hit within ``footprint_radius`` that
  is still well below the base (excluding walls/bars/step risers within
  ``obstacle_margin`` of the base).  Termination ≈ ``g_height < 0``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  scan: RayCastSensor = env.scene[sensor_name]
  hit = scan.data.hit_pos_w  # (B, N, 3)
  dist = scan.data.distances  # (B, N)

  base_z = asset.data.root_link_pos_w[:, 2]
  base_xy = asset.data.root_link_pos_w[:, None, :2]
  planar = torch.norm(hit[..., :2] - base_xy, dim=-1)
  in_footprint = (dist >= 0) & (planar <= footprint_radius)

  hit_z = hit[..., 2]
  below = in_footprint & (hit_z < base_z[:, None] - obstacle_margin)
  neg_inf = torch.full_like(hit_z, -1.0e9)
  pos_inf = torch.full_like(hit_z, 1.0e9)
  ground_ref = torch.where(below, hit_z, neg_inf).max(dim=1).values
  lowest = torch.where(in_footprint, hit_z, pos_inf).min(dim=1).values
  lowest = torch.where(in_footprint.any(dim=1), lowest, base_z)
  ground_ref = torch.where(below.any(dim=1), ground_ref, lowest)

  return (base_z - ground_ref) < min_clearance
