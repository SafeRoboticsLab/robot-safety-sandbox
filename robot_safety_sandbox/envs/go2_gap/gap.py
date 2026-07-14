"""Minimal gaps-only environment for the reach-avoid gap experiment.

A single gap per patch (flat approach -> gap -> flat landing), gap width
spanning clearly-jumpable to clearly-uncrossable.  Goal behaviors:
  * large gap, not yet jumped -> stop before the edge,
  * small gap -> jump across,
  * already crossed -> stop or continue forward.

This is the **privileged-actor** first pass: the actor sees proprioception
*including* the raycast ``height_scan`` (so the gap geometry is directly
observable), and the depth camera is dropped for speed.  A later pass can swap
to a depth-only actor for deployability.

Key choices (see chat rationale):
  * Reach margin ``l`` = *foothold support* (handled by the gap wrapper), so
    "safely stopped before an uncrossable gap" is itself a target state and the
    policy learns to stop rather than launch.
  * No distance-curriculum; ``randomize_terrain`` re-rolls the gap width each
    reset so the policy faces the full width distribution from the start.
  * Init sampling places the robot before / over (slightly elevated) / after the
    gap with varied pose + velocity spanning recoverable and unrecoverable
    flight states.
"""

from __future__ import annotations

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import robot_safety_sandbox.envs.parkour.mdp as mdp
from robot_safety_sandbox.envs.parkour.go2 import unitree_go2_parkour_env_cfg
from robot_safety_sandbox.envs.parkour.terrains import GAP_EDGE_TERRAINS_CFG


def _drop_depth(cfg: ManagerBasedRlEnvCfg) -> None:
  """Remove the front depth camera + depth obs group (privileged actor)."""
  cfg.scene.sensors = tuple(
    s
    for s in (cfg.scene.sensors or ())
    if not (isinstance(s, CameraSensorCfg) and s.name == "front_depth")
  )
  if "depth" in cfg.observations:
    cfg.observations.pop("depth", None)


def unitree_go2_gap_reach_avoid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_parkour_env_cfg(play=play)

  # --- Fixed-edge gap terrain (near edge == patch origin, no curriculum). ---
  cfg.scene.terrain.terrain_generator = replace(GAP_EDGE_TERRAINS_CFG)

  # --- Short, jump-scoped episodes. ---
  # Policy 2 only handles the jump maneuver, so episodes end shortly after a
  # jump resolves (lands / falls); this also keeps the policy out of flat-margin
  # "just standing" states that drive the entropy/std runaway.
  cfg.episode_length_s = 3.0

  # --- Follow camera: up + tilted down + 3/4 angle so the robot and the gap
  #     are both visible over the low curbs (for train/video + eval clips). ---
  cfg.viewer.distance = 2.6
  cfg.viewer.elevation = -35.0
  cfg.viewer.azimuth = 130.0

  # --- Privileged actor: keep height_scan in proprioception, drop depth. ---
  _drop_depth(cfg)

  # --- Constant forward drive (safety overrides it to stop at a big gap). ---
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (1.0, 1.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  if hasattr(twist, "rel_standing_envs"):
    twist.rel_standing_envs = 0.0
  if hasattr(twist, "heading_command"):
    twist.heading_command = False

  # --- Terrain-relative failure terminations (match g). ---
  cfg.terminations.pop("base_too_low", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )
  cfg.terminations["base_below_local_terrain"] = TerminationTermCfg(
    func=mdp.base_below_local_terrain,
    params={
      "min_clearance": 0.08,
      "sensor_name": "terrain_scan",
      "footprint_radius": 0.5,
      "obstacle_margin": 0.12,
    },
  )

  # NOTE: no per-reset randomize_terrain — with curriculum=False the 256 envs
  # are already spread across the 10x10 patch grid (all gap widths represented),
  # and re-rolling on every reset is very expensive under the short 3 s episodes.

  # --- Pure-jump init: edge (commit/refuse) + mid-air (finish jump) only. ---
  # The patch origin is AT the near edge, so offsets are relative to the edge.
  # No far-back (walk-the-approach) or far-platform (standing) spawns -> Policy 2
  # only ever acts in jump-relevant states.
  cfg.events["reset_base"] = EventTermCfg(
    func=mdp.reset_robot_midair_over_gaps,
    mode="reset",
    params={
      "midair_fraction": 0.5,
      # On the approach just before the edge, then pushed forward (commit/refuse).
      "ground_pose_range": {"x": (-0.5, -0.1), "y": (-0.15, 0.15), "yaw": (-0.15, 0.15)},
      "ground_velocity_range": {"x": (0.5, 2.0), "y": (-0.2, 0.2)},
      # Mid-air just past the edge, over the gap, mid-arc (finish the jump).
      "midair_x_range": (0.0, 0.5),
      "midair_y_range": (-0.15, 0.15),
      "midair_z_range": (0.10, 0.45),
      # Forward velocity spanning unrecoverable (slow) -> clears (fast).
      "midair_vx_range": (1.0, 2.5),
      "midair_vy_range": (-0.2, 0.2),
      "midair_vz_range": (-1.0, 0.3),
      "midair_roll_range": (-math.radians(10.0), math.radians(10.0)),
      "midair_pitch_range": (-math.radians(10.0), math.radians(10.0)),
      "midair_yaw_range": (-0.2, 0.2),
    },
  )
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.5, 0.5)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.5, 0.5)

  # --- No terrain curriculum (widths exposed via randomize_terrain). ---
  cfg.curriculum = {}

  # Lighter, less frequent pushes than full parkour-safety.
  if not play and "push_robot" in cfg.events:
    cfg.events["push_robot"].interval_range_s = (3.0, 6.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.2, 0.2),
      "roll": (-0.3, 0.3),
      "pitch": (-0.3, 0.3),
      "yaw": (-0.3, 0.3),
    }

  if play:
    cfg.events.pop("push_robot", None)
    cfg.events["reset_base"].params["midair_fraction"] = 0.25

  return cfg
