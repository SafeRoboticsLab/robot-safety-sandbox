"""Vendored mdp functions for the Digit safety task (fork-independence).

The Digit env builders (``builders.py``) and margins (``margins.py``) were
written against the SafeRoboticsLab mjlab fork (MjlabSafety_Digit). This module
vendors every mdp symbol they need that is NOT part of stock mjlab 1.2, so the
zoo's Digit tasks run against a stock mjlab install. Function bodies are copied
faithfully from the fork; only imports were adjusted.

Provenance (fork = MjlabSafety_Digit @ 28b7ed9):

* fork ``src/mjlab/envs/mdp/terminations.py``:
    - ``safety_margin_violated``  (fork-only)
    - ``nan_detection``           (byte-identical copy also exists in stock
      mjlab 1.2.0 ``envs/mdp/terminations.py``; vendored anyway so the Digit
      task is pinned to the exact semantics it was trained with)
* fork ``src/mjlab/envs/mdp/observations.py``:
    - ``box_pose_relative``, ``box_lin_vel_relative``  (fork-only)
* fork ``src/mjlab/envs/mdp/events.py``:
    - ``randomize_terrain``  (byte-identical copy also exists in stock mjlab
      1.2.0; vendored anyway, same rationale as ``nan_detection``)
    - ``apply_body_impulse``  (stock mjlab 1.2.0 has a same-named class but
      WITHOUT the fork's ``random_body_selection`` parameter, which the Digit
      box builders use — vendoring this class is REQUIRED, not just defensive)
* fork ``src/mjlab/tasks/velocity/mdp/rewards.py`` (all fork-only):
    - ``undesired_contacts``, ``joint_deviation_l1``, ``lin_vel_z_l2``,
      ``stand_still_joint_deviation_l1``, ``stand_still_flat_orientation_l2``
* fork ``src/mjlab/rl/safety_vecenv_wrapper.py``:
    - ``_FOOT_AND_KNEE_BODIES``  (constant only; the wrapper itself is not
      vendored — the zoo has its own bridge)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  sample_uniform,
)
from mjlab.utils.nan_guard import NanGuard

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.sensor import ContactSensor
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Body-name constant (from the fork's rl/safety_vecenv_wrapper.py).
# Bodies allowed to touch the ground: feet, knees, and the toe linkage.
##

_FOOT_AND_KNEE_BODIES = (
  "left_knee",
  "right_knee",
  "left_heel_spring",
  "right_heel_spring",
  "left_toe_A",
  "right_toe_A",
  "left_toe_B",
  "right_toe_B",
  "left_toe_pitch",
  "right_toe_pitch",
  "left_toe_roll",
  "right_toe_roll",
)


##
# Terminations.
##


def nan_detection(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate environments that have NaN/Inf values in their physics state."""
  return NanGuard.detect_nans(env.sim.data)


def safety_margin_violated(
  env: ManagerBasedRlEnv,
  ground_clearance: float = 0.03,
  min_torso_height: float = 0.3,
  tilt_limit_rad: float = 1.3963,
  failure_body_names: tuple[str, ...] = (
    "torso",
    "left_hip_roll",
    "right_hip_roll",
    "left_hip_yaw",
    "right_hip_yaw",
    "left_hip_pitch",
    "right_hip_pitch",
    "left_achillies_rod",
    "right_achillies_rod",
    "left_shin",
    "right_shin",
    "left_tarsus",
    "right_tarsus",
    "left_toe_A_rod",
    "right_toe_A_rod",
    "left_toe_B_rod",
    "right_toe_B_rod",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_elbow",
    "right_elbow",
  ),
  torso_body_name: str = "torso",
  box_body_name: str | None = None,
  min_box_height: float = 1.0,
  box_tilt_limit_rad: float = 1.3963,
  foot_site_names: tuple[str, ...] | None = None,
  max_foot_height: float = 0.10,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the safety margin g(s) < 0.

  Checks failure conditions (matching SafetyVecEnvWrapper):
  1. Any non-foot/knee body below ground_clearance.
  2. Torso below min_torso_height.
  3. Orientation exceeding tilt_limit_rad.
  4. (If box) Box height below min_box_height.
  5. (If box) Box tilted more than box_tilt_limit_rad.
  6. (If foot sites) Any foot site above max_foot_height.

  This runs INSIDE the env step (before auto-reset), so the terminal
  state is correctly recorded.
  """
  import math

  asset: Entity = env.scene[asset_cfg.name]

  # Cache body indices on first call.
  cache_key = "_safety_term_cache"
  if not hasattr(env, cache_key):
    failure_ids, _ = asset.find_bodies(failure_body_names)
    torso_ids, _ = asset.find_bodies((torso_body_name,))
    cache = {
      "failure_ids": failure_ids,
      "torso_id": torso_ids[0] if len(torso_ids) > 0 else 0,
      "sin_tilt_limit": math.sin(tilt_limit_rad),
      "has_box": False,
    }
    if box_body_name is not None:
      try:
        box_ids, _ = asset.find_bodies((box_body_name,))
        if len(box_ids) > 0:
          cache["has_box"] = True
          cache["box_id"] = box_ids[0]
          cache["sin_box_tilt_limit"] = math.sin(box_tilt_limit_rad)
      except (ValueError, RuntimeError):
        pass
    cache["has_foot_sites"] = False
    if foot_site_names is not None:
      try:
        foot_site_ids, _ = asset.find_sites(foot_site_names)
        if len(foot_site_ids) > 0:
          cache["has_foot_sites"] = True
          cache["foot_site_ids"] = foot_site_ids
      except (ValueError, RuntimeError):
        pass
    setattr(env, cache_key, cache)
  cache = getattr(env, cache_key)

  # 1. Non-foot body height check.
  failure_heights = asset.data.body_link_pose_w[
    :, cache["failure_ids"], 2
  ]
  body_violated = failure_heights.min(dim=1).values < ground_clearance

  # 2. Torso height check.
  torso_height = asset.data.body_link_pose_w[:, cache["torso_id"], 2]
  torso_violated = torso_height < min_torso_height

  # 3. Orientation check.
  proj_grav = asset.data.projected_gravity_b
  grav_xy_norm = torch.norm(proj_grav[:, :2], dim=1)
  orient_violated = grav_xy_norm > cache["sin_tilt_limit"]

  violated = body_violated | torso_violated | orient_violated

  # 4 & 5. Box checks (only when box body is present).
  if cache["has_box"]:
    box_pose = asset.data.body_link_pose_w[:, cache["box_id"]]
    # Height check.
    box_height_violated = box_pose[:, 2] < min_box_height
    # Tilt check: local +Z projected onto world +Z.
    # For quaternion [w,x,y,z]: R[2,2] = 1 - 2*(qx² + qy²).
    # sin(tilt) = sqrt(qx² + qy²) * 2  (small angle approx irrelevant,
    # we compare grav_xy_norm style).  Instead, use the same approach:
    # the box's up-vector z-component = 1 - 2*(qx²+qy²).
    # tilt > limit  ⟺  cos(tilt) < cos(limit)  ⟺  up_z < cos(limit)
    # But we want consistency with sin-based check. Use:
    # sin(tilt) = sqrt(1 - up_z²), violated when sin(tilt) > sin(limit).
    qx = box_pose[:, 4]
    qy = box_pose[:, 5]
    box_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    box_sin_tilt = torch.sqrt(
      torch.clamp(1.0 - box_up_z * box_up_z, min=0.0)
    )
    box_tilt_violated = box_sin_tilt > cache["sin_box_tilt_limit"]
    violated = violated | box_height_violated | box_tilt_violated

  # 6. Foot height check (anti-tiptoeing).
  if cache["has_foot_sites"]:
    foot_heights = asset.data.site_pos_w[
      :, cache["foot_site_ids"], 2
    ]  # (num_envs, num_foot_sites)
    foot_too_high = foot_heights.max(dim=1).values > max_foot_height
    violated = violated | foot_too_high

  return violated


##
# Observations (box state relative to robot root).
##


def box_pose_relative(
  env: ManagerBasedRlEnv,
  box_body_name: str = "box_load",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Box position and orientation relative to the robot root frame.

  Returns a 7-dim vector per env: [rel_pos (3), rel_quat (4)].
  Position is in the robot's body frame so the observation is
  invariant to the robot's world-frame heading and position.
  """
  asset: Entity = env.scene[asset_cfg.name]

  cache_key = "_box_obs_cache"
  if not hasattr(env, cache_key):
    box_ids, _ = asset.find_bodies((box_body_name,))
    setattr(env, cache_key, {"box_id": box_ids[0]})
  cache = getattr(env, cache_key)

  # Root pose.
  root_pos = asset.data.root_link_pos_w  # (N, 3)
  root_quat = asset.data.root_link_quat_w  # (N, 4)

  # Box world pose.
  box_pose = asset.data.body_link_pose_w[:, cache["box_id"]]
  box_pos = box_pose[:, 0:3]  # (N, 3)
  box_quat = box_pose[:, 3:7]  # (N, 4)

  # Relative position in root frame.
  rel_pos = quat_apply_inverse(root_quat, box_pos - root_pos)

  # Relative orientation.
  rel_quat = quat_mul(quat_conjugate(root_quat), box_quat)

  return torch.cat([rel_pos, rel_quat], dim=-1)


def box_lin_vel_relative(
  env: ManagerBasedRlEnv,
  box_body_name: str = "box_load",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Box linear velocity in the robot root frame.  Shape (N, 3).

  Computed as the velocity of the box body minus the root body
  velocity, rotated into the root frame.
  """
  asset: Entity = env.scene[asset_cfg.name]

  cache_key = "_box_vel_cache"
  if not hasattr(env, cache_key):
    box_ids, _ = asset.find_bodies((box_body_name,))
    setattr(env, cache_key, {"box_id": box_ids[0]})
  cache = getattr(env, cache_key)

  root_quat = asset.data.root_link_quat_w
  root_vel = asset.data.root_link_lin_vel_w  # (N, 3)

  # Box linear velocity from cvel (angular, linear) -> indices 3:6.
  box_vel = asset.data.data.cvel[:, cache["box_id"], 3:6]

  rel_vel = quat_apply_inverse(root_quat, box_vel - root_vel)
  return rel_vel


##
# Events.
##


def randomize_terrain(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Randomize the sub-terrain for each environment on reset.

  This picks a random terrain type (column) and difficulty level (row) for each
  environment. Useful for play/evaluation mode to test on varied terrains.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  terrain = env.scene.terrain
  if terrain is not None:
    terrain.randomize_env_origins(env_ids)


class apply_body_impulse:
  """Apply random impulses to bodies for a sampled duration.

  Simulates transient external disturbances such as bumps, wind gusts, or
  collisions with unseen objects. A constant force/torque wrench is applied
  to one or more bodies for a randomly sampled duration, followed by a
  cooldown period of silence before the next impulse.

  **Lifecycle of a single impulse:**

  1. **Cooldown.** The event is idle for a random duration sampled from ``cooldown_s``.
    No force is applied.
  2. **Trigger.** A force vector is sampled uniformly per component from ``force_range``
    and written to ``xfrc_applied`` on the selected bodies.
  3. **Sustain.** The force is held constant for a random duration sampled from
    ``duration_s``.
  4. **Expire.** The force is zeroed and the cooldown restarts at step 1.

  Each environment runs its own independent timer so impulses are decorrelated across
  the batch.

  **Application point.** By default, forces act at each body's center of mass.
  ``body_point_offset`` shifts the application point in the body's local frame, for
  example ``(0, 0, 0.1)`` for 10 cm above the CoM. The offset produces additional
  torque via the cross product ``offset x force``, causing the body to tip rather than
  just translate. This is analogous to choosing where on the body an external push is
  applied.

  Use with ``mode="step"``.
  """

  @dataclass
  class VizCfg:
    """Arrow visualization settings for active impulse forces."""

    rgba: tuple[float, float, float, float] = (0.9, 0.2, 0.8, 0.9)
    """Arrow color (RGBA)."""
    scale: float = 0.005
    """Arrow length in meters per Newton of force."""
    width: float = 0.015
    """Arrow shaft width in meters."""
    min_force: float = 1.0
    """Minimum force magnitude (N) below which arrows are hidden."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt
    self._viz_cfg: apply_body_impulse.VizCfg = cfg.params.get(
      "viz_cfg", apply_body_impulse.VizCfg()
    )
    offset = cfg.params.get("body_point_offset", None)
    self._body_point_offset: torch.Tensor | None = (
      torch.tensor(offset, device=self._device, dtype=torch.float32)
      if offset is not None
      else None
    )

    self._num_bodies = (
      len(self._body_ids)
      if isinstance(self._body_ids, list)
      else self._asset.num_bodies
    )

    self._time_remaining = torch.zeros(self._num_envs, device=self._device)
    self._interval_time_left = torch.zeros(self._num_envs, device=self._device)
    self._active = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    body_point_offset: tuple[float, float, float] | None = None,
    random_body_selection: bool = False,
  ) -> None:
    """Tick impulse state: expire old impulses, trigger new ones.

    Args:
      env: The environment instance.
      env_ids: Unused (step events always operate on all envs).
      force_range: ``(min, max)`` uniform range for each force component (N).
      torque_range: ``(min, max)`` uniform range for each torque component (Nm).
      duration_s: ``(min, max)`` uniform range for impulse duration in seconds.
      cooldown_s: ``(min, max)`` uniform range for the cooldown between consecutive
        impulses in seconds.
      asset_cfg: Entity and body selection. ``body_ids`` on the config selects which
        bodies receive forces.
      body_point_offset: Optional ``(x, y, z)`` offset in the body frame where the
        force is applied. Generates additional torque via ``cross(offset, force)``.
      random_body_selection: If True, each trigger picks exactly ONE random
        body from ``asset_cfg.body_ids`` to receive the impulse (all other
        bodies get zero force).  This mimics a localized push rather than
        a whole-body shake.  Default False (legacy behaviour: all bodies).
    """
    del env, env_ids, asset_cfg  # Unused.
    dt = self._step_dt

    # Decrement timers for active envs.
    self._time_remaining[self._active] -= dt

    # Clear expired impulses and resample their interval timers.
    expired = self._active & (self._time_remaining <= 0)
    if expired.any():
      expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
      zeros = torch.zeros((len(expired_ids), self._num_bodies, 3), device=self._device)
      self._asset.write_external_wrench_to_sim(
        zeros, zeros, env_ids=expired_ids, body_ids=self._body_ids
      )
      self._active[expired_ids] = False
      self._time_remaining[expired_ids] = 0.0
      int_low, int_high = cooldown_s
      self._interval_time_left[expired_ids] = (
        torch.rand(len(expired_ids), device=self._device) * (int_high - int_low)
        + int_low
      )

    # Decrement interval timers.
    self._interval_time_left -= dt

    # Trigger new impulses for eligible envs.
    eligible = (~self._active) & (self._interval_time_left <= 0)
    if not eligible.any():
      return

    trigger_ids = eligible.nonzero(as_tuple=False).squeeze(-1)
    n = len(trigger_ids)

    # Sample forces and torques.
    size = (n, self._num_bodies, 3)
    forces = sample_uniform(*force_range, size, self._device)
    torques = sample_uniform(*torque_range, size, self._device)

    # Optionally keep only ONE random body per trigger, zero the rest.
    if random_body_selection and self._num_bodies > 1:
      chosen = torch.randint(
        0, self._num_bodies, (n,), device=self._device
      )  # (n,)
      mask = torch.zeros(
        (n, self._num_bodies, 1), device=self._device
      )
      mask[torch.arange(n, device=self._device), chosen, 0] = 1.0
      forces = forces * mask
      torques = torques * mask

    # Adjust torque for off-CoM application point.
    if body_point_offset is not None:
      offset_local = torch.tensor(
        body_point_offset, device=self._device, dtype=torch.float32
      )
      body_quat = self._asset.data.body_com_quat_w[trigger_ids][:, self._body_ids]
      # Rotate offset into world frame: (n, num_bodies, 3).
      offset_w = quat_apply(
        body_quat.reshape(-1, 4), offset_local.expand(n * self._num_bodies, 3)
      ).reshape(n, self._num_bodies, 3)
      torques = torques + torch.cross(offset_w, forces, dim=-1)

    self._asset.write_external_wrench_to_sim(
      forces, torques, env_ids=trigger_ids, body_ids=self._body_ids
    )

    # Sample duration and set timers.
    dur_low, dur_high = duration_s
    self._time_remaining[trigger_ids] = (
      torch.rand(n, device=self._device) * (dur_high - dur_low) + dur_low
    )
    self._active[trigger_ids] = True

    # Resample interval timers.
    int_low, int_high = cooldown_s
    self._interval_time_left[trigger_ids] = (
      torch.rand(n, device=self._device) * (int_high - int_low) + int_low
    )

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Draw arrows for active impulse forces."""
    if not self._active.any():
      return
    viz = self._viz_cfg
    min_sq = viz.min_force * viz.min_force
    wrench = self._asset.data.body_external_wrench  # (nworld, nbody, 6)
    com_pos = self._asset.data.body_com_pos_w  # (nworld, nbody, 3)
    offset = self._body_point_offset
    com_quat = self._asset.data.body_com_quat_w if offset is not None else None
    for env_idx in visualizer.get_env_indices(self._num_envs):
      if not self._active[env_idx]:
        continue
      for i in range(wrench.shape[1]):
        force = wrench[env_idx, i, :3]
        if (force * force).sum().item() < min_sq:
          continue
        force_np = force.cpu().numpy()
        start_np = com_pos[env_idx, i].cpu().numpy()
        if offset is not None and com_quat is not None:
          offset_w = quat_apply(com_quat[env_idx, i], offset)
          start_np = start_np + offset_w.cpu().numpy()
        end_np = start_np + force_np * viz.scale
        visualizer.add_arrow(
          start=start_np,
          end=end_np,
          color=viz.rgba,
          width=viz.width,
        )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)

    # Clear forces for reset envs.
    if isinstance(env_ids, slice):
      reset_ids = env_ids
    else:
      reset_ids = env_ids

    if self._active[reset_ids].any():
      if isinstance(env_ids, slice):
        active_ids = self._active.nonzero(as_tuple=False).squeeze(-1)
      else:
        active_ids = env_ids[self._active[env_ids]]
      if len(active_ids) > 0:
        zeros = torch.zeros(
          (len(active_ids), self._num_bodies, 3),
          device=self._device,
        )
        self._asset.write_external_wrench_to_sim(
          zeros, zeros, env_ids=active_ids, body_ids=self._body_ids
        )

    self._time_remaining[reset_ids] = 0.0
    self._interval_time_left[reset_ids] = 0.0
    self._active[reset_ids] = False


##
# Rewards (fork-only additions to tasks/velocity/mdp/rewards.py).
##


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalise any contact on monitored bodies exceeding the force threshold.

  Returns the count of monitored bodies currently in contact, aggregated
  across all matched slots so the output is always shape [B].
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).sum(dim=-1).float()  # [B]
  assert data.force is not None
  force_mag = torch.norm(data.force, dim=-1)  # [B, N]
  return (force_mag > force_threshold).sum(dim=-1).float()  # [B]


def lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize vertical (z) linear velocity of the base in world frame."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_w[:, 2])


def joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize L1 deviation of selected joints from their default positions."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.abs(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=1,
  )


def stand_still_joint_deviation_l1(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize joint deviation from default when the velocity command is near zero.

  Encourages the robot to hold its default pose when asked to stand still.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  is_standing = (torch.norm(command[:, :3], dim=1) < command_threshold).float()
  joint_dev = torch.sum(
    torch.abs(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=1,
  )
  return is_standing * joint_dev


def stand_still_flat_orientation_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize non-flat orientation when the velocity command is near zero.

  Directly penalizes torso tilt during standing, complementing the
  joint-deviation-based stand_still penalty.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  is_standing = (torch.norm(command[:, :3], dim=1) < command_threshold).float()

  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    body_quat_w = body_quat_w.squeeze(1)
    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

  return is_standing * xy_squared
