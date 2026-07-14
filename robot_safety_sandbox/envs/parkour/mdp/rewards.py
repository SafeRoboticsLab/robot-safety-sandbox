"""Parkour-specific reward terms.

Rewards for forward progress, obstacle negotiation, body posture,
and energy efficiency during parkour tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor, ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ---------------------------------------------------------------------------
# Forward progress
# ---------------------------------------------------------------------------


def forward_velocity(
  env: ManagerBasedRlEnv,
  target_vel: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward forward (body-frame x) velocity, with Gaussian tracking."""
  asset: Entity = env.scene[asset_cfg.name]
  vel_b = asset.data.root_link_lin_vel_b
  fwd_vel = vel_b[:, 0]
  error = torch.square(fwd_vel - target_vel)
  return torch.exp(-error / 0.25)


def forward_progress(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward forward displacement per step (world x direction)."""
  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_lin_vel_w
  return torch.clamp(vel_w[:, 0], min=0.0) * env.step_dt


def _detect_gap(
  scan: RayCastSensor,
  robot_z: torch.Tensor,
  drop_threshold: float,
  min_gap_rays: int,
) -> torch.Tensor:
  """Shared gap detection from a raycast sensor.

  Returns a boolean tensor (num_envs,) that is True when a gap is detected.
  Requires at least ``min_gap_rays`` rays to show a significant drop or miss,
  which suppresses single-ray false positives from terrain roughness noise
  while still catching real gaps that span multiple grid cells.
  """
  hit_z = scan.data.hit_pos_w[..., 2]          # (num_envs, num_rays)
  drop = robot_z.unsqueeze(1) - hit_z           # positive = terrain below robot
  miss_mask = scan.data.distances < 0           # ray missed (no geometry hit)
  gap_ray = (drop > drop_threshold) | miss_mask # per-ray gap flag
  return gap_ray.sum(dim=-1) >= min_gap_rays    # (num_envs,) bool


def gap_crossing_reward(
  env: ManagerBasedRlEnv,
  scan_sensor_name: str,
  drop_threshold: float = 0.15,
  min_gap_rays: int = 2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense reward: forward progress while a gap is detected nearby.

  Uses ``min_gap_rays`` to require at least that many scan rays to report a
  significant terrain drop before declaring a gap present.  This eliminates
  false positives caused by single-ray noise or small terrain bumps.
  """
  asset: Entity = env.scene[asset_cfg.name]
  scan: RayCastSensor = env.scene[scan_sensor_name]

  robot_z = asset.data.root_link_pos_w[:, 2]  # (num_envs,)
  gap_detected = _detect_gap(scan, robot_z, drop_threshold, min_gap_rays).float()

  vel_w = asset.data.root_link_lin_vel_w
  fwd_progress = torch.clamp(vel_w[:, 0], min=0.0) * env.step_dt

  return gap_detected * fwd_progress


class gap_crossing_bonus:
  """Sparse bonus fired once when the robot genuinely crosses a gap forward.

  Records the world-x position when the robot first enters a gap.  The bonus
  fires only when the gap disappears AND the robot is now ahead of where it
  entered — ruling out turn-arounds, lateral escapes, or backing away.

  The framework instantiates this class with ``cls(cfg=..., env=...)`` then
  calls ``instance(env, **params)`` each step.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._scan_sensor_name = cfg.params["scan_sensor_name"]
    self._drop_threshold = cfg.params.get("drop_threshold", 0.7)
    self._min_gap_rays = cfg.params.get("min_gap_rays", 2)
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._asset_name = asset_cfg.name
    self._prev_gap = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # World-x position at which each env first detected a gap this episode.
    self._gap_entry_x = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    scan_sensor_name: str,
    drop_threshold: float = 0.7,
    min_gap_rays: int = 2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[self._asset_name]
    scan: RayCastSensor = env.scene[self._scan_sensor_name]
    robot_x = asset.data.root_link_pos_w[:, 0]
    robot_z = asset.data.root_link_pos_w[:, 2]

    gap_now = _detect_gap(scan, robot_z, self._drop_threshold, self._min_gap_rays)

    # On episode reset, re-sync state without firing a bonus.
    just_reset = env.episode_length_buf <= 1
    self._prev_gap = torch.where(just_reset, gap_now, self._prev_gap)
    self._gap_entry_x = torch.where(just_reset, robot_x, self._gap_entry_x)

    # Record entry x the first step a gap is seen (prev=False → now=True).
    gap_entered = (~self._prev_gap) & gap_now
    self._gap_entry_x = torch.where(gap_entered, robot_x, self._gap_entry_x)

    # Bonus fires when: gap disappears AND robot is forward of the entry point.
    # The 0.1 m margin ensures the robot clearly moved past, not just scan noise.
    gap_cleared = self._prev_gap & ~gap_now
    moved_forward = robot_x > self._gap_entry_x + 0.1
    crossed = (gap_cleared & moved_forward).float()

    self._prev_gap = gap_now.clone()
    return crossed


# ---------------------------------------------------------------------------
# Lateral drift and yaw
# ---------------------------------------------------------------------------


def lateral_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize lateral (y) velocity to keep robot on track."""
  asset: Entity = env.scene[asset_cfg.name]
  vel_b = asset.data.root_link_lin_vel_b
  return torch.square(vel_b[:, 1])


def yaw_rate_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive yaw rotation."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.root_link_ang_vel_b
  return torch.square(ang_vel[:, 2])


# ---------------------------------------------------------------------------
# Body posture
# ---------------------------------------------------------------------------


def body_orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-upright body orientation (roll and pitch via projected gravity)."""
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    gravity_w = asset.data.gravity_vec_w
    proj_grav = quat_apply_inverse(body_quat_w, gravity_w)
    return torch.sum(torch.square(proj_grav[:, :2]), dim=1)
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def body_height_penalty(
  env: ManagerBasedRlEnv,
  target_height: float = 0.30,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target body height."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]
  return torch.square(height - target_height)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities (xy)."""
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  else:
    ang_vel = asset.data.root_link_ang_vel_b
  return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


# ---------------------------------------------------------------------------
# Collision penalties
# ---------------------------------------------------------------------------


def body_collision(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize non-foot contacts (body hitting obstacles)."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.squeeze(-1)


# ---------------------------------------------------------------------------
# Gait and foot rewards
# ---------------------------------------------------------------------------


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float = 0.06,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize feet dragging below target clearance, weighted by foot velocity.

  One-sided: only penalizes feet that are *below* the target height during swing.
  Feet lifted above the target are not penalized, allowing high stepping for
  gap crossing and obstacle negotiation.
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)
  # Only penalize feet below target (dragging), not above (high stepping).
  delta = torch.clamp(target_height - foot_z, min=0.0)
  return torch.sum(delta * vel_norm, dim=1)


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding during contact."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)
  return torch.sum(vel_sq * in_contact, dim=1)


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize high impact forces at landing."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  data = contact_sensor.data
  assert data.force is not None
  force_mag = torch.norm(data.force, dim=-1)
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
  impact = force_mag * first_contact.float()
  return torch.sum(impact, dim=1)


# ---------------------------------------------------------------------------
# Action regularization
# ---------------------------------------------------------------------------


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize change in actions between steps."""
  return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_acc_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize joint acceleration for smooth motion."""
  asset: Entity = env.scene["robot"]
  return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def joint_pos_limits(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joints approaching their limits."""
  asset: Entity = env.scene[asset_cfg.name]
  pos = asset.data.joint_pos
  lower = asset.data.soft_joint_pos_limits[..., 0]
  upper = asset.data.soft_joint_pos_limits[..., 1]
  below = torch.clamp(lower - pos, min=0.0)
  above = torch.clamp(pos - upper, min=0.0)
  return torch.sum(below + above, dim=1)


def energy_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize energy consumption (action magnitude * joint velocity)."""
  asset: Entity = env.scene[asset_cfg.name]
  action = env.action_manager.action
  joint_vel = asset.data.joint_vel
  return torch.sum(torch.abs(action * joint_vel), dim=1)


def feet_gait(
  env: ManagerBasedRlEnv,
  period: float,
  offset: list[float],
  threshold: float,
  command_threshold: float,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  """Reward matching a trot gait pattern (diagonal legs in sync).

  Compares desired stance/swing phase with actual contact state.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  is_contact = sensor.data.current_contact_time > 0
  global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
  leg_phase = (global_phase + offsets) % 1.0
  is_stance = leg_phase < threshold
  reward = (is_stance == is_contact).float().mean(dim=1)
  command = env.command_manager.get_command(command_name)
  if command is not None:
    total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    reward = reward * (total_command > command_threshold).float()
  return reward


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking commanded linear velocity (Gaussian kernel)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  return torch.exp(-(xy_error + 2 * z_error) / std**2)
