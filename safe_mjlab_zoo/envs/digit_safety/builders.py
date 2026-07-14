"""Digit v3 velocity/safety environment builders (vendored, fork-independent).

Vendored from the mjlab fork (MjlabSafety_Digit @ 28b7ed9):

* ``make_velocity_env_cfg`` — the fork's base velocity cfg
  (``src/mjlab/tasks/velocity/velocity_env_cfg.py``). The zoo's own
  ``envs/velocity/env_cfg.py`` factory is NOT compatible (different reward /
  observation term names: no ``upright``/``air_time``/``foot_swing_height``
  keys, no actor ``base_lin_vel``), so the fork's base is vendored here rather
  than merged — correctness over deduplication. The fork's base is itself
  stock mjlab 1.2's ``velocity_env_cfg.py`` with a ``test1`` action-rate
  metric instead of ``mean_action_acc``.
* the Digit builders — ``src/mjlab/tasks/velocity/config/digit_v3/env_cfgs.py``
  verbatim, with imports repointed: robot cfgs come from
  ``safe_mjlab_zoo.envs.assets_digit``; fork-only mdp functions come from the
  vendored ``safe_mjlab_zoo.envs.digit_safety.mdp`` (as ``zoo_mdp``);
  everything else resolves against stock mjlab modules.

Builders kept (all of them work with what is vendored):
  rough / flat locomotion, flat+load, flat+box (welded), and the safety family
  (flat safety, calibrated, rigidtoe, box, box-calibrated, box-rigidtoe).
The zoo tasks use ``digit_v3_flat_safety_rigidtoe_env_cfg`` and
``digit_v3_flat_safety_box_rigidtoe_env_cfg``.

KNOWN RESIDUAL COUPLING (documented 2026-07, not import-level): every import
in this module resolves against stock mjlab 1.2.0, and cfg CONSTRUCTION works
under a pure stock install. But actually SIMULATING Digit under stock mjlab
1.2.0 fails at env construction: the Digit MJCF keeps free BALL joints on the
achilles-rod closed-loop linkage, and stock ``mjlab/entity/entity.py`` includes
ball joints in the observable joint state (``joint_vel`` is DOF-sized, 36,
while ``default_joint_vel`` is joint-sized, 32 -> shape mismatch in
``joint_pos_rel``/``joint_vel_rel``). The fork carries a 6-line patch that
excludes ball joints from the observable joint state (fork ``entity.py``,
"Ball joints are multi-DOF passive constraint joints..."). Until upstream
mjlab handles ball joints, running Digit requires an mjlab with that patch
(the lab's installed mjlab has it).
"""

import math
from dataclasses import replace

import torch

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from safe_mjlab_zoo.envs.assets_digit.digit_constants import (
  DIGIT_ACTION_SCALE,
  DIGIT_CALIBRATED_ARTICULATION,
  DIGIT_RIGIDTOE_ARTICULATION,
  get_digit_robot_cfg,
  get_spec_upright_calibrated,
  get_spec_upright_calibrated_rigidtoe,
)
from safe_mjlab_zoo.envs.assets_digit.digit_with_box import (
  get_spec_upright_calibrated_rigidtoe_with_box_on_arms,
  get_spec_upright_calibrated_with_box_on_arms,
  get_spec_upright_with_box,
  get_spec_upright_with_box_on_arms,
)
from safe_mjlab_zoo.envs.assets_digit.digit_with_load import get_spec_upright_with_load
from safe_mjlab_zoo.envs.digit_safety import mdp as zoo_mdp


##
# Base velocity cfg (fork version, see module docstring).
##


def action_rate_l2(env) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )

  ##
  # Observations
  ##

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
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
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1 / terrain_scan.max_distance,
    ),
  }

  critic_terms = {
    **actor_terms,
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  metrics = {
    "test1": MetricsTermCfg(
      func=action_rate_l2,
    )
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=0.0,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=0.0,  # Override per-robot
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,  # Override per-robot.
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.5,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
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
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
          {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
          {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan,),
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
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
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
    episode_length_s=20.0,
  )


##
# Digit v3 builders (fork env_cfgs.py, imports repointed).
##


def digit_v3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Digit v3 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_digit_robot_cfg()}
  cfg.sim.mujoco.ccd_iterations = 500

  # Use a focused terrain set: flat, random bumps, and sinusoidal waves.
  assert (
    cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
  )
  cfg.scene.terrain.terrain_generator = replace(
    cfg.scene.terrain.terrain_generator,
    sub_terrains={
      "flat": terrain_gen.BoxFlatTerrainCfg(proportion=1.0),
      "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0,
        noise_range=(0.02, 0.10),
        noise_step=0.02,
        border_width=0.25,
      ),
      "wave_terrain": terrain_gen.HfWaveTerrainCfg(
        proportion=1.0,
        amplitude_range=(0.0, 0.2),
        num_waves=4,
        border_width=0.25,
      ),
    },
  )

  # Tighten solver settings for closed-loop linkages.
  cfg.sim.mujoco.timestep = 0.005
  cfg.sim.mujoco.iterations = 50
  cfg.sim.mujoco.ls_iterations = 50

  # Set raycast sensor frame to Digit torso.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "torso"

  site_names = ("left_foot", "right_foot")
  geom_names = ("left_toe_roll_collision", "right_toe_roll_collision")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_toe_roll|right_toe_roll)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = DIGIT_ACTION_SCALE

  cfg.viewer.body_name = "torso"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  # Actor observation noise (additive Gaussian) per Digit hardware spec.
  actor_terms = cfg.observations["actor"].terms
  actor_terms["base_lin_vel"].noise = Gnoise(mean=0.0, std=0.15)
  actor_terms["base_ang_vel"].noise = Gnoise(mean=0.0, std=0.15)
  actor_terms["projected_gravity"].noise = Gnoise(mean=0.0, std=0.075)
  actor_terms["joint_pos"].noise = Gnoise(mean=0.0, std=0.0875)
  actor_terms["joint_vel"].noise = Gnoise(mean=0.0, std=0.075)
  actor_terms["height_scan"].noise = Unoise(n_min=-0.05, n_max=0.05)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.05, 0.05),
    1: (-0.05, 0.05),
    2: (-0.05, 0.05),
  }
  cfg.events["encoder_bias"].params["bias_range"] = (-0.03, 0.03)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body (actuated).
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*toe_A.*": 0.2,
    r".*toe_B.*": 0.2,
    # Lower body (passive, linkage-driven — permissive std).
    r".*tarsus.*": 0.5,
    r".*toe_pitch.*": 0.5,
    r".*toe_roll.*": 0.5,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r"shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body (actuated).
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*toe_A.*": 0.3,
    r".*toe_B.*": 0.3,
    # Lower body (passive, linkage-driven — permissive std).
    r".*tarsus.*": 0.8,
    r".*toe_pitch.*": 0.8,
    r".*toe_roll.*": 0.8,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r"shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["foot_clearance"].weight = -1.0

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # 1. Termination penalty: strongly penalise non-timeout episode endings
  #    (e.g. falling over). This stabilises early training significantly.
  cfg.rewards["termination_penalty"] = RewardTermCfg(
    func=envs_mdp.is_terminated,
    weight=-100.0,
  )

  # 2. Enable biped air-time reward (disabled at 0.0 in the base config).
  #    Encourages alternating foot lifts and a natural gait rhythm.
  #    Weight kept low (0.25) to avoid single-foot pivot exploitation.
  cfg.rewards["air_time"].weight = 0.25

  # 3. Reduce action-rate penalty to match V4 tuning (-0.1 → -0.008).
  #    The base value is far too aggressive and suppresses leg swing.
  cfg.rewards["action_rate_l2"].weight = -0.008

  # 4. Torque penalty: discourages energy-wasteful high-torque strategies.
  # Hip-pitch/knee effort limits were reduced from spec (216.9 / 231.3 Nm)
  # down to 150 Nm to tame motor torque.  Weight -3e-7 gives bite at peak
  # ≈ 6.75e-3, ~equivalent to the original 80 Nm × -1e-6 baseline (6.4e-3),
  # i.e. same torque-suppression intensity, just at the new 150 Nm peak.
  cfg.rewards["dof_torques_l2"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2,
    weight=-3.0e-7,
  )

  # 5. Joint acceleration penalty: encourages smooth, non-jerky motion.
  #    Applied only to actuated leg and arm joints (skip passive linkages).
  cfg.rewards["dof_acc_l2"] = RewardTermCfg(
    func=envs_mdp.joint_acc_l2,
    weight=-2.0e-7,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          ".*_hip_roll_joint",
          ".*_hip_yaw_joint",
          ".*_hip_pitch_joint",
          ".*_knee_joint",
          ".*_toe_A_joint",
          ".*_toe_B_joint",
          ".*_shoulder_roll_joint",
          ".*_shoulder_pitch_joint",
          "shoulder_yaw_joint_.*",
          ".*_elbow_joint",
        ),
      )
    },
  )

  # 6. Undesired contacts: penalise rod/tarsus/shin bodies touching the
  #    terrain (mimics V4's undesired_contacts term, weight -0.1).
  rod_tarsus_contact_cfg = ContactSensorCfg(
    name="rod_tarsus_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left|right)_(achillies_rod|toe_A_rod|toe_B_rod|tarsus|shin)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (rod_tarsus_contact_cfg,)
  cfg.rewards["undesired_contacts"] = RewardTermCfg(
    func=zoo_mdp.undesired_contacts,
    weight=-0.1,
    params={
      "sensor_name": rod_tarsus_contact_cfg.name,
      "force_threshold": 1.0,
    },
  )

  # 7. Keep fall termination at default 70°.  The V4 value of 40° is too
  #    strict for early training — short episodes give no useful gradient
  #    signal.  Tighten only after the policy can reliably stand and walk.

  # 8. Strong flat-orientation penalty.
  #    Much stronger than the positive `upright` reward already present;
  #    directly penalises projected-gravity xy deviation.
  cfg.rewards["flat_orientation_l2"] = RewardTermCfg(
    func=envs_mdp.flat_orientation_l2,
    weight=-2.5,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso",))},
  )

  # 9. Vertical velocity penalty.
  #    Prevents the robot from bouncing or hopping.
  cfg.rewards["lin_vel_z_l2"] = RewardTermCfg(
    func=zoo_mdp.lin_vel_z_l2,
    weight=-2.0,
  )

  # 10. Angular velocity penalty: increase weight to match V4 (-0.1).
  cfg.rewards["body_ang_vel"].weight = -0.1

  # 11. Joint deviation penalties keep specific joints near zero (V4 values).
  cfg.rewards["joint_deviation_hip_roll"] = RewardTermCfg(
    func=zoo_mdp.joint_deviation_l1,
    weight=-0.1,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_roll_joint",))},
  )
  cfg.rewards["joint_deviation_hip_yaw"] = RewardTermCfg(
    func=zoo_mdp.joint_deviation_l1,
    weight=-0.2,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_yaw_joint",))},
  )
  cfg.rewards["joint_deviation_hip_pitch"] = RewardTermCfg(
    func=zoo_mdp.joint_deviation_l1,
    weight=-0.15,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_pitch_joint",))},
  )
  cfg.rewards["joint_deviation_arms"] = RewardTermCfg(
    func=zoo_mdp.joint_deviation_l1,
    weight=-0.2,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          ".*_shoulder_roll_joint",
          ".*_shoulder_pitch_joint",
          "shoulder_yaw_joint_.*",
          ".*_elbow_joint",
        ),
      )
    },
  )

  # 12. Stand-still penalty: penalise joint deviation when command is zero.
  #     Forces the robot to hold its default upright pose when not moving.
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=zoo_mdp.stand_still_joint_deviation_l1,
    weight=-0.8,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          ".*_hip_roll_joint",
          ".*_hip_yaw_joint",
          ".*_hip_pitch_joint",
          ".*_knee_joint",
          ".*_toe_A_joint",
          ".*_toe_B_joint",
        ),
      ),
    },
  )
  cfg.rewards["stand_still_upright"] = RewardTermCfg(
    func=zoo_mdp.stand_still_flat_orientation_l2,
    weight=-15.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
    },
  )

  # 13. Simplify the command space so the robot must learn to walk forward
  #     before being asked to turn.  Lateral velocity and heading control are
  #     disabled entirely; angular velocity is reintroduced by the curriculum
  #     at step 6000*24.  Without this, the robot exploits spinning on one
  #     foot to satisfy angular commands without ever walking.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_standing_envs = 0.2
  # Angular tracking is irrelevant while ang_vel_z=0; reduce its weight to
  # avoid it dominating once turning is re-introduced by the curriculum.
  cfg.rewards["track_angular_velocity"].weight = 0.5

  # 14. Override velocity curriculum (train from scratch, 20k iters):
  #   0-2k:   stand stable before introducing any walking signal
  #   2k-6k:  forward only to build a clean walking gait
  #   6k-10k: add backward to remove directional bias
  #   10k+:   add turning once both directions are solid
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (0.0, 0.0)},
    {"step": 3000 * 24, "lin_vel_x": (0.0, 1.0)},
    {"step": 6000 * 24, "lin_vel_x": (-0.5, 1.0)},
    {"step": 10000 * 24, "lin_vel_x": (-0.5, 1.0), "ang_vel_z": (-0.5, 0.5)},
  ]

  # Randomize initial joint positions ±0.1 rad around default pose so the
  # policy learns to recover from non-ideal starting configurations.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.2, 0.2)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-1.0, 1.0)
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.03, 0.10)

  # 15. Override push_robot with reduced rotational perturbations.
  cfg.events["push_robot"].interval_range_s = (2.0, 5.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.4, 0.4),
    "roll": (-0.4, 0.4),
    "pitch": (-0.4, 0.4),
    "yaw": (-0.2, 0.2),
  }

  # 16. Sim-to-real: randomize PD gains ±20% so the policy can't rely on
  #     exact actuator dynamics (real motors differ due to friction, gear
  #     efficiency, temperature, and different motor driver behavior).
  cfg.events["pd_gain_randomization"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
    },
  )

  # 17. Sim-to-real: randomize effort limits 70-100% so the policy learns
  #     to work under tighter torque budgets (Agility sim limits differ
  #     per joint and real hardware varies with temperature/battery).
  cfg.events["effort_randomization"] = EventTermCfg(
    mode="startup",
    func=dr.effort_limits,
    params={
      "effort_limit_range": (0.7, 1.0),
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
    },
  )

  # 18. Randomize joint frictionloss to model harmonic-drive stiction.
  #     XML defaults to 0; ranges scale with each joint's effort budget.
  cfg.events["joint_frictionloss_randomization"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "ranges": {
        # Hips + knee: harmonic-drive output, high stiction.
        r".*_hip_roll_joint": (0.5, 3.0),
        r".*_hip_yaw_joint": (0.3, 2.0),
        r".*_hip_pitch_joint": (0.5, 3.0),
        r".*_knee_joint": (0.5, 3.0),
        # Toe motors: smaller gear ratio, lower reflected friction.
        r".*_toe_A_joint": (0.1, 0.8),
        r".*_toe_B_joint": (0.1, 0.8),
        # Arms: similar gearing to hips but lighter loaded.
        r".*_shoulder_roll_joint": (0.3, 2.0),
        r".*_shoulder_pitch_joint": (0.3, 2.0),
        r"shoulder_yaw_joint_.*": (0.3, 2.0),
        r".*_elbow_joint": (0.3, 2.0),
      },
      "operation": "abs",
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=zoo_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  return cfg


def digit_v3_flat_safety_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat terrain config for safety training.

  Key differences from the locomotion config:
  - Zero velocity commands (the robot learns to survive, not walk).
  - Broader initial state distribution: larger joint perturbations and
    non-zero base velocity at reset so the robot starts from diverse,
    potentially unstable states.
  - Stronger and more frequent disturbance pushes so the robot faces
    meaningful perturbations during training.

  The goal is for the safety policy to learn a large safe set — if the
  robot is pushed hard, it should recover (e.g. by stepping, using
  arms) rather than only staying safe from already-stable states.
  """
  cfg = digit_v3_flat_env_cfg(play=play)

  # Zero out velocity curriculum — always command zero velocity.
  # The agent learns to stay safe while standing, not walking.
  # In play mode, curriculum is already cleared, so guard the access.
  if "command_vel" in cfg.curriculum:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (0.0, 0.0)},
    ]

  # --- Termination: safety margin violation ---
  # Replace the arbitrary fell_over (70°) with physics-based failure
  # detection that matches the safety wrapper's g(s) definition.
  # This runs INSIDE the env step (pre-reset) so the terminal reward
  # is computed from the correct (failing) state.
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=zoo_mdp.safety_margin_violated,
    params={
      "ground_clearance": 0.03,
      "min_torso_height": 0.3,
      "tilt_limit_rad": 1.3963,  # 80 degrees
    },
  )

  # --- Broader initial states ---
  # The robot should start from diverse configurations, not just a
  # perfect upright stance.  This forces the safety policy to learn
  # recovery from a wider basin of attraction.
  #
  # Start moderate: ±0.25 rad joint position, ±1.5 rad/s joint velocity.
  # These values keep ~70-80% of initial states safe (g >= 0) so the
  # policy has enough positive signal to learn from.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.25, 0.25)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-1.5, 1.5)

  # Base velocity at reset: give the robot a moderate initial push so
  # it doesn't always start stationary.
  cfg.events["reset_base"].params["velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.3, 0.3),
    "pitch": (-0.3, 0.3),
    "yaw": (-0.2, 0.2),
  }

  # --- Stronger, more frequent disturbances ---
  # Impulse pushes: ±1.0 m/s linear (was ±0.5), ±0.8 rad/s angular
  # (was ±0.4).  Strong enough to challenge, not so strong that the
  # robot can never survive.
  cfg.events["push_robot"].interval_range_s = (1.5, 4.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-1.0, 1.0),
    "y": (-1.0, 1.0),
    "z": (-0.5, 0.5),
    "roll": (-0.8, 0.8),
    "pitch": (-0.8, 0.8),
    "yaw": (-0.4, 0.4),
  }

  if play:
    # Disable automatic pushes so the user can apply forces manually.
    cfg.events.pop("push_robot", None)

    # Zero velocity commands to match training distribution.
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (0.0, 0.0)

  return cfg


def digit_v3_flat_safety_calibrated_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """No-box safety config on the sim2sim-calibrated Digit model.

  Same task as ``digit_v3_flat_safety_env_cfg`` (survive pushes, zero
  velocity command) but with the ar-control-calibrated dynamics: shin/heel
  leaf springs + fitted armature/frictionloss/damping.

  Purpose: a clean sim2sim gap test.  With no box, mjlab and ar-control
  share the *identical* mechanical system, so any behavior gap isolates the
  calibrated dynamics transfer (no box confound).

  ``toe_roll_stiffness`` is left at 0 (free ankle): the modeled pushrod
  connect-constraints + leaf springs are the closest structural match to
  ar-control's rigid linkage.  Adding artificial ankle stiffness would
  diverge from ar-control and defeat the test; the RL policy balances the
  compliant ankle actively.
  """
  cfg = digit_v3_flat_safety_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_spec_upright_calibrated
  cfg.scene.entities["robot"].articulation = DIGIT_CALIBRATED_ARTICULATION

  # Keep the stiff leaf springs out of reset joint randomization and the
  # pose reward (the pose std dicts don't cover the 4 new spring joints).
  cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=("(?!.*shin_joint|.*heel_spring).*",),
  )
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=("(?!.*shin_joint|.*heel_spring).*",),
  )
  return cfg


def digit_v3_flat_safety_rigidtoe_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """No-box safety config on the `_rigidtoe` model with fitted holding gains.

  Same task as the *Calibrated* safety cfg, but the robot uses (a) the rigid
  toe transmission (pushrod rod DOFs removed; measured motor->toe map as
  fixed-tendon equalities) and (b) TRAINING-ONLY effective kp/kd scales
  (DIGIT_RIGIDTOE_GAIN_SCALES) fitted so loaded joint holding matches
  ar-control (the mirror survives ar-control's full 28 s hard sway with
  these; the nominal-PD model creeps and falls in 1.5 s). Observation dims
  are unchanged vs the calibrated cfg (checkpoints obs-compatible).
  DEPLOYMENT: export policy metadata via the *Calibrated* task so the
  deployed command stream carries NOMINAL gains.
  """
  cfg = digit_v3_flat_safety_calibrated_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_spec_upright_calibrated_rigidtoe
  cfg.scene.entities["robot"].articulation = DIGIT_RIGIDTOE_ARTICULATION
  return cfg


def digit_v3_flat_safety_box_rigidtoe_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Box safety config on the `_rigidtoe + fitted gains` plant.

  Same task as the Box-Safety-Calibrated cfg (balance the free box, survive
  pushes) on the creep-corrected model. See
  ``digit_v3_flat_safety_rigidtoe_env_cfg`` for the plant rationale and the
  deployment-parity rule (export metadata via a *Calibrated* task).
  """
  cfg = digit_v3_flat_safety_with_box_calibrated_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = (
    get_spec_upright_calibrated_rigidtoe_with_box_on_arms
  )
  cfg.scene.entities["robot"].articulation = DIGIT_RIGIDTOE_ARTICULATION
  return cfg


def digit_v3_flat_with_load_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat terrain config with a bar fixed between the robot's hands."""
  cfg = digit_v3_flat_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_spec_upright_with_load
  return cfg


def digit_v3_flat_with_box_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat terrain config with an open-top box held by both hands."""
  cfg = digit_v3_flat_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_spec_upright_with_box
  return cfg


def reset_box_state_to_arms(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
  from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
  )

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  mj_model = env.unwrapped.sim.mj_model
  try:
    box_jnt = mj_model.joint(f"{asset_cfg.name}/box_freejoint")
    root_jnt = mj_model.joint(f"{asset_cfg.name}/floating_base_joint")
  except KeyError:
    return

  box_qadr = int(box_jnt.qposadr[0])
  box_vadr = int(box_jnt.dofadr[0])
  root_qadr = int(root_jnt.qposadr[0])

  key_qpos = mj_model.key("init_state").qpos
  spec_box_pos = torch.tensor(
    key_qpos[box_qadr : box_qadr + 3], dtype=torch.float, device=env.device
  ).unsqueeze(0)
  spec_box_quat = torch.tensor(
    key_qpos[box_qadr + 3 : box_qadr + 7], dtype=torch.float, device=env.device
  ).unsqueeze(0)

  spec_root_pos = torch.tensor(
    key_qpos[root_qadr : root_qadr + 3], dtype=torch.float, device=env.device
  ).unsqueeze(0)
  spec_root_quat = torch.tensor(
    key_qpos[root_qadr + 3 : root_qadr + 7], dtype=torch.float, device=env.device
  ).unsqueeze(0)

  local_box_pos = quat_apply_inverse(spec_root_quat, spec_box_pos - spec_root_pos)
  local_box_quat = quat_mul(quat_conjugate(spec_root_quat), spec_box_quat)

  current_root_pos = env.sim.data.qpos[env_ids, root_qadr : root_qadr + 3]
  current_root_quat = env.sim.data.qpos[env_ids, root_qadr + 3 : root_qadr + 7]

  new_box_pos = current_root_pos + quat_apply(
    current_root_quat, local_box_pos.expand(len(env_ids), -1)
  )
  new_box_quat = quat_mul(current_root_quat, local_box_quat.expand(len(env_ids), -1))

  env.sim.data.qpos[env_ids, box_qadr : box_qadr + 3] = new_box_pos
  env.sim.data.qpos[env_ids, box_qadr + 3 : box_qadr + 7] = new_box_quat
  env.sim.data.qvel[env_ids, box_vadr : box_vadr + 6] = 0.0


def digit_v3_flat_safety_with_box_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Flat terrain safety config with free box on forearms.

  The box is a free body (freejoint on worldbody) resting on the
  robot's extended forearms, supported purely by contact with
  invisible platform geoms.  Under strong disturbance the box can
  slide, tilt, and fall off the arms.

  Safety margin includes both robot failure (falling, torso too low,
  excessive tilt) and box failure (box height < 0.4 m, box tilt > 45°).

  The arm pose is a "tray" configuration (forearms extended forward
  and level).  Initial perturbations are reduced because the box makes
  balance significantly harder.
  """
  cfg = digit_v3_flat_safety_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_spec_upright_with_box_on_arms

  # ~35 collision geoms (6 hip_pitch + 4 shin + 6 torso +
  # 2 shoulder_pitch + 4 shoulder_yaw + 4 elbow + 2 platform +
  # 2 feet + 5 box).
  cfg.sim.nconmax = 80

  # The spec_fn names ALL geoms (via _name_all_geoms) so that the
  # entity system's CollisionCfg can properly enable/disable them.
  # Replace the default FEET_COLLISION with a single config that
  # enables feet + torso + shoulder + forearm + platform + box geoms.
  # Everything else is disabled (disable_other_geoms=True).
  #
  # Arm-torso penetration is prevented by restricting shoulder_pitch
  # to ±90° in the box spec (see digit_with_box.py).  Torso geoms
  # are included so the box and shoulders collide with the torso
  # rather than passing through it.
  box_platform_collision = CollisionCfg(
    geom_names_expr=(
      r"^(left|right)_toe_roll_collision$",  # feet
      r"^(left|right)_hip_pitch_geom_[123]$",  # upper leg collision primitives
      r"^(left|right)_shin_geom_[12]$",  # shin collision cylinders
      r"^torso_geom_\d+$",  # torso collision body
      r"^(left|right)_shoulder_pitch_geom_1$",  # shoulder pitch cylinders
      r"^(left|right)_shoulder_yaw_geom_[12]$",  # shoulder yaw cylinders
      r"^(left|right)_elbow_geom_[12]$",  # forearm cylinders + spheres
      r"^(left|right)_elbow_platform$",  # forearm platforms
      r"^box_",  # box panels
    ),
    condim=4,
    priority=1,
    friction=(1.0, 0.005, 0.0001),
    disable_other_geoms=True,
  )
  cfg.scene.entities["robot"].collisions = (box_platform_collision,)

  # Override termination with box-aware safety margin check.
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=zoo_mdp.safety_margin_violated,
    params={
      "ground_clearance": 0.03,
      "min_torso_height": 0.3,
      "tilt_limit_rad": 1.3963,  # 80 degrees
      "box_body_name": "box_load",
      # Box starts at ~1.14 m.  Terminate if it drops more than
      # ~14 cm (slid off arms) or tilts beyond 80° (spilling).
      "min_box_height": 1.0,
      "box_tilt_limit_rad": 1.3963,  # 80 degrees
      # Anti-tiptoeing: foot sites must stay below 10 cm.
      # In standing pose foot sites are at ~1.8 cm.  Tiptoeing
      # raises them above 10 cm; this forces flat-footed stance.
      "foot_site_names": ("left_foot", "right_foot"),
      "max_foot_height": 0.10,
    },
  )

  # ── Box observations ──────────────────────────────────────────────
  # The policy needs to see the box state to learn to keep it stable.
  # We add relative pose (7D) and relative velocity (3D) to both
  # actor and critic observation groups.
  from mjlab.managers.observation_manager import ObservationTermCfg
  from mjlab.utils.noise import UniformNoiseCfg as Unoise

  cfg.observations["actor"].terms["box_pose"] = ObservationTermCfg(
    func=zoo_mdp.box_pose_relative,
    noise=Unoise(n_min=-0.01, n_max=0.01),
  )
  cfg.observations["actor"].terms["box_vel"] = ObservationTermCfg(
    func=zoo_mdp.box_lin_vel_relative,
    noise=Unoise(n_min=-0.1, n_max=0.1),
  )
  cfg.observations["critic"].terms["box_pose"] = ObservationTermCfg(
    func=zoo_mdp.box_pose_relative,
  )
  cfg.observations["critic"].terms["box_vel"] = ObservationTermCfg(
    func=zoo_mdp.box_lin_vel_relative,
  )

  # Reduce initial perturbations slightly — the box makes balance
  # harder, so the same perturbation level would cause too many
  # immediate failures.
  # Exclude box_freejoint and arm joints from joint offset randomization
  # so the box starts exactly on the arms at every reset.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.15, 0.15)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-1.0, 1.0)
  cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=("(?!box_|.*shoulder|.*elbow).*",)
  )
  cfg.events["reset_box_state"] = EventTermCfg(
    func=reset_box_state_to_arms,
    mode="reset",
  )
  cfg.events["reset_base"].params["velocity_range"] = {
    "x": (-0.4, 0.4),
    "y": (-0.4, 0.4),
    "z": (-0.15, 0.15),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.15, 0.15),
  }

  # Strongly penalise arm deviation from the tray pose.  The base
  # config uses -0.2 which is too weak — the policy exploits wide
  # shoulder ranges to shove arms into the torso.  Combined with
  # torso self-collision above, this keeps arms in a physically
  # valid configuration.
  cfg.rewards["joint_deviation_arms"].weight = -2.0

  # ── Localized body impulses ───────────────────────────────────────
  # Replace the root-only velocity push with force impulses applied
  # to ONE randomly selected body per trigger (legs, torso, arms).
  # This mimics real-world localized disturbances (a bump on a leg,
  # a shove on the torso) instead of applying forces to every body
  # at once — which is unphysical and far too destabilizing.
  cfg.events.pop("push_robot", None)
  if not play:
    cfg.events["body_impulse"] = EventTermCfg(
      func=zoo_mdp.apply_body_impulse,
      mode="step",
      params={
        "force_range": (-80.0, 80.0),
        "torque_range": (-8.0, 8.0),
        "duration_s": (0.1, 0.3),
        "cooldown_s": (1.0, 3.0),
        "random_body_selection": True,
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=(
            "torso",
            "left_hip_pitch",
            "right_hip_pitch",
            "left_shin",
            "right_shin",
            "left_shoulder_roll",
            "right_shoulder_roll",
            "left_elbow",
            "right_elbow",
          ),
        ),
      },
    )

  return cfg


def digit_v3_flat_safety_with_box_calibrated_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Box safety config on the sim2sim-calibrated Digit model.

  Same task as ``digit_v3_flat_safety_with_box_env_cfg`` but with the
  ar-control-calibrated dynamics (see sim2sim/README.md):

  - shin/heel leaf springs (k=6000/4375) + fitted joint damping,
    toe_roll stiffness 300 Nm/rad (standing-stable ankle);
  - fitted armature and frictionloss on all actuated joints.

  The 4 spring DOFs enlarge joint-enumerating observations, so
  policies trained on the uncalibrated model are incompatible.
  """
  cfg = digit_v3_flat_safety_with_box_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_spec_upright_calibrated_with_box_on_arms
  cfg.scene.entities["robot"].articulation = DIGIT_CALIBRATED_ARTICULATION

  # Keep the stiff leaf springs out of reset joint randomization —
  # a ±0.15 rad offset on a 6000 Nm/rad spring is a violent impulse.
  cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=("(?!box_|.*shoulder|.*elbow|.*shin_joint|.*heel_spring).*",),
  )

  # Exclude the passive spring joints from the pose reward: they're
  # unactuated (no point shaping them), and the std_walking/std_running
  # pattern dicts don't cover them (std tensor sizes would mismatch).
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=("(?!.*shin_joint|.*heel_spring).*",),
  )
  return cfg


def digit_v3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Digit v3 flat terrain velocity configuration."""
  cfg = digit_v3_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum.
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
