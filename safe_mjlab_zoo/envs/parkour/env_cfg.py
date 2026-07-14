"""Parkour task configuration factory.

Creates a base parkour environment config with depth camera + proprioception,
parkour-specific rewards, and custom terrain. Robot-specific configs call
this factory and customize as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import safe_mjlab_zoo.envs.parkour.mdp as mdp
from safe_mjlab_zoo.envs.parkour.terrains import PARKOUR_TERRAINS_CFG


def make_parkour_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base parkour task configuration with depth vision."""

  ##
  # Sensors
  ##

  # Forward-facing depth camera mounted on robot head/front.
  front_depth_camera = CameraSensorCfg(
    name="front_depth",
    parent_body="",  # Set per-robot (e.g., "robot/base_link").
    pos=(0.342, 0.0, 0.03),  # Just past front collision geom (base2 ends ~0.335m).
    # Camera looks forward and ~20° downward so the robot can see the ground
    # 0.5–1.5 m ahead (critical for gap detection). Base orientation rotated
    # from (0.5, 0.5, -0.5, -0.5) by an additional 20° pitch-down.
    quat=(0.406, 0.579, -0.579, -0.406),  # w, x, y, z — 20° downward tilt
    fovy=87.0,  # Wide FOV for obstacle detection.
    width=64,
    height=64,
    data_types=("depth",),
    use_textures=False,
    use_shadows=False,
  )

  # Terrain raycast for height scan (complements depth camera).
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=False,
  )

  ##
  # Observations
  ##

  # Proprioceptive observations (1D, concatenated).
  proprio_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.5, "command_name": "twist"},
    ),
    "base_height": ObservationTermCfg(
      func=mdp.base_height,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.03, n_max=0.03),  # Reduced noise to preserve gap signal.
      scale=1 / terrain_scan.max_distance,
    ),
  }

  # Depth image observation (2D, kept as separate tensor for CNN).
  depth_terms = {
    "depth_image": ObservationTermCfg(
      func=mdp.depth_image,
      params={"sensor_name": "front_depth", "far_clip": 3.0, "near_clip": 0.01},
    ),
  }

  # Critic gets privileged information (no depth needed).
  critic_terms = {
    **proprio_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "forward_distance": ObservationTermCfg(
      func=mdp.forward_distance,
    ),
  }

  observations = {
    "proprioception": ObservationGroupCfg(
      terms=proprio_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=5,  # Temporal context for gap approach planning.
    ),
    "depth": ObservationGroupCfg(
      terms=depth_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=5,  # Match actor history for consistent temporal context.
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    ),
  }

  ##
  # Commands (parkour uses forward velocity command only)
  ##

  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(5.0, 10.0),
      rel_standing_envs=0.0,  # No standing — always moving forward.
      rel_heading_envs=0.0,
      heading_command=False,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.3, 1.5),  # Allow slower approach for gap negotiation.
        lin_vel_y=(-0.1, 0.1),  # Minimal lateral.
        ang_vel_z=(-0.2, 0.2),  # Minimal yaw.
      ),
    ),
  }

  ##
  # Events (domain randomization)
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.3, 0.3),
          "y": (-0.2, 0.2),
          "z": (0.0, 0.0),
          "yaw": (-0.2, 0.2),  # Mostly forward-facing.
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 5.0),
      params={
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "z": (-0.2, 0.2),
          "roll": (-0.3, 0.3),
          "pitch": (-0.3, 0.3),
          "yaw": (-0.3, 0.3),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.5),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.02, 0.02),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.03, 0.03),
          1: (-0.03, 0.03),
          2: (-0.04, 0.04),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    # Primary: track commanded velocity (same as velocity tracker).
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.5,
      params={"command_name": "twist", "std": 0.5},
    ),

    # Forward progress bonus (parkour-specific).
    "forward_progress": RewardTermCfg(
      func=mdp.forward_progress,
      weight=1.0,  # Reduced from 2.0 — less rush, more room for cautious gap approach.
    ),

    # Gait pattern: trot with 0.5s period. Reduced weight to allow the robot
    # to break gait when needed for strategic foot placement at gaps.
    "feet_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.25,  # Reduced from 0.5 — advisory, not mandatory.
      params={
        "period": 0.5,
        "offset": [0.0, 0.5, 0.5, 0.0],  # Set per-robot (trot pattern).
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
      },
    ),

    # Stay on track.
    "lateral_velocity": RewardTermCfg(
      func=mdp.lateral_velocity_penalty,
      weight=-0.5,
    ),
    "yaw_rate": RewardTermCfg(
      func=mdp.yaw_rate_penalty,
      weight=-0.1,
    ),

    # Body posture.
    "body_orientation": RewardTermCfg(
      func=mdp.body_orientation,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "body_height": RewardTermCfg(
      func=mdp.body_height_penalty,
      weight=-0.5,  # Reduced from -1.0 — allow body to dip when bridging gaps.
      params={"target_height": 0.30},
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),

    # Collision penalty.
    "body_collision": RewardTermCfg(
      func=mdp.body_collision,
      weight=-1.0,
      params={"sensor_name": "nonfoot_ground_touch", "force_threshold": 10.0},
    ),

    # Foot rewards. One-sided clearance: only penalizes feet dragging below
    # target, not high stepping (needed for gap crossing).
    "feet_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-0.25,  # Reduced from -0.5 — less restrictive for obstacle negotiation.
      params={
        "target_height": 0.06,  # Reduced from 0.08 — only penalize actual dragging.
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.2,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-5e-4,
      params={"sensor_name": "feet_ground_contact"},
    ),

    # Action regularization.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
    "energy_penalty": RewardTermCfg(
      func=mdp.energy_penalty,
      weight=-1e-5,
    ),

    # Gap crossing — dense: forward progress while a gap is detected.
    # drop_threshold=0.7: Go2 at 0.30m → flat terrain drop ~0.30m (< 0.7, no gap),
    # rough terrain drop ~0.42m (< 0.7, no gap), real gap drop ~2.3m (> 0.7, gap).
    # min_gap_rays=2 requires at least 2 scan rays to report a drop, eliminating
    # single-ray noise from rough terrain.
    "gap_crossing": RewardTermCfg(
      func=mdp.gap_crossing_reward,
      weight=5.0,
      params={
        "scan_sensor_name": "terrain_scan",
        "drop_threshold": 0.7,
        "min_gap_rays": 1,
      },
    ),

    # Gap crossing — sparse: one-time bonus when the robot clears a gap forward.
    # Only fires if the robot moved past its gap-entry x position, ruling out
    # turn-arounds or backing away.
    "gap_crossing_bonus": RewardTermCfg(
      func=mdp.gap_crossing_bonus,
      weight=10.0,
      params={
        "scan_sensor_name": "terrain_scan",
        "drop_threshold": 0.7,
        "min_gap_rays": 1,
      },
    ),

    # Termination penalty.
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-50.0),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "base_too_low": TerminationTermCfg(
      func=mdp.base_too_low,
      params={"min_height": 0.05},  # Lowered from 0.10 — allow body dip during gap bridging.
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_parkour,
    ),
  }

  ##
  # Assemble
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(PARKOUR_TERRAINS_CFG),
        max_init_terrain_level=3,
      ),
      sensors=(terrain_scan, front_depth_camera),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=2.0,
      elevation=-15.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=35.0,
  )
