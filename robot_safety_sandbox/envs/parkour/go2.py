"""Unitree Go2 parkour environment configurations."""

import math

from robot_safety_sandbox.envs.assets_go2.go2_constants import get_go2_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import (
  CameraSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  RayCastSensorCfg,
)

import robot_safety_sandbox.envs.parkour.mdp as mdp
from robot_safety_sandbox.envs.parkour.env_cfg import make_parkour_env_cfg


def unitree_go2_parkour_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 parkour training configuration."""
  cfg = make_parkour_env_cfg()

  # Tune contact/collision limits for parkour terrain.
  # Keep these moderate — the terrain grid size (below) is the main lever for
  # controlling total geom count and CCD load.  Go2 contacts at most ~10-15
  # geom pairs per step; 80 gives headroom for multi-body obstacle contacts.
  cfg.sim.nconmax = 80
  cfg.sim.njmax = 800
  cfg.sim.mujoco.ccd_iterations = 200
  cfg.sim.contact_sensor_maxmatch = 200

  # Set robot.
  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Configure sensors for Go2 body names.
  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  for sensor in cfg.scene.sensors or ():
    if isinstance(sensor, RayCastSensorCfg) and sensor.name == "terrain_scan":
      sensor.frame.name = "base_link"
    if isinstance(sensor, CameraSensorCfg) and sensor.name == "front_depth":
      sensor.parent_body = "robot/base_link"

  # Contact sensors.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  # Enable terrain curriculum.
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # Viewer.
  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -15.0

  # Set per-robot observation params.
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  # Set per-robot event params.
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # Set per-robot reward params.
  cfg.rewards["body_orientation"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_height"].params["target_height"] = 0.30  # Go2 standing height ~0.30m.
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["feet_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]  # Trot: FR/RL, FL/RR.
  cfg.rewards["feet_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["feet_slip"].params["asset_cfg"].site_names = site_names

  # Add illegal contact termination.
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["proprioception"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
