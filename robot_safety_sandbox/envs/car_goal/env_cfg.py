"""car_goal: a differential-drive car that must REACH a goal disk while AVOIDING
obstacle cylinders -- the mjlab-zoo, reach-avoid analog of ``bicycle5d.py``.

Single-player reach-avoid (ReachAvoidPPO, 2 wheel-velocity controls):

  g(s) = signed distance from the car to the nearest obstacle, normalized.
         g >= 0 == not in collision.  Rides on ``reward`` (the safety hook).
  l(s) = goal_radius - distance-to-goal, normalized.
         l >= 0 == the car is inside the goal disk.  Rides on ``info["l_x"]``.

The discriminating property (mirrors bicycle5d): an AVOID policy sits still (g is
already > 0 at spawn), while a REACH-AVOID policy must drive to the goal without
hitting anything -- so ``reach_rate(reach-avoid) >> reach_rate(avoid)``.

A FRESH minimal ``ManagerBasedRlEnvCfg`` (NOT the heavy velocity locomotion base):
flat arena terrain, 2-D wheel-velocity action, a handful of car-frame observation
terms, a randomized spawn, and time-out + collision terminations. ``base.py``
injects the g-margin reward hook and (for end_criterion="reach-avoid") the
success termination; ``rewards`` is therefore intentionally EMPTY here.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.viewer import ViewerConfig

from robot_safety_sandbox.envs.assets_car.car_constants import get_car_robot_cfg
from robot_safety_sandbox.envs.terrains.car_arena import (
  CAR_ARENA_CFG,
  GOAL_RADIUS,
  OBSTACLES,
  START_TO_GOAL,
)

# --- margin normalization (kept O(1), clamped +/-3 per the base contract) ------
_G_SCALE = 0.5             # obstacle-distance normalization
_L_SCALE = 3.0             # goal-distance normalization (l stays graded out to the
                          # spawn distance ~3 m, so the reach gradient never dies)
_CAR_RADIUS = 0.15         # car footprint radius (chassis half-diagonal ~0.14)
_CLAMP = 3.0

_WHEEL_SPEED = 20.0        # action [-1, 1] -> wheel velocity target (rad/s);
                          # wheel radius 0.05 m -> ~1 m/s straight-line top speed
                          # (measured ~0.97 m/s) -- the 2 m goal / 12 s horizon
                          # is comfortably reachable at this speed


def _obstacles(device) -> torch.Tensor:
  return torch.tensor(OBSTACLES, device=device, dtype=torch.float32)  # (M, 3)


def _car_local_xy(env) -> torch.Tensor:
  """Car planar position RELATIVE TO ITS SPAWN ORIGIN (the frame the fixed
  OBSTACLES / goal offsets live in)."""
  d = env.scene["robot"].data
  return d.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]


# --- margins (the task contract: (g, l), each (num_envs,), g/l >= 0 good) ------

def car_margins(env):
  p = _car_local_xy(env)                       # (N, 2)
  obs = _obstacles(p.device)                    # (M, 3): dx, dy, r
  diff = p[:, None, :] - obs[None, :, :2]       # (N, M, 2)
  clearance = torch.linalg.norm(diff, dim=-1) - obs[None, :, 2] - _CAR_RADIUS
  g = (clearance.amin(dim=1) / _G_SCALE).clamp(-_CLAMP, _CLAMP)

  goal = torch.tensor([START_TO_GOAL, 0.0], device=p.device)
  dist = torch.linalg.norm(p - goal, dim=-1)
  l = ((GOAL_RADIUS - dist) / _L_SCALE).clamp(-_CLAMP, _CLAMP)
  return g, l


def car_collision(env, margin_fn=None) -> torch.Tensor:
  """Absorbing-failure termination: the car overlapped an obstacle (g < 0).
  Makes collision terminal so base.py anchors g to the failure value there
  (mirrors bicycle5d's ``terminated = g < 0``)."""
  g, _ = car_margins(env)
  return g < 0.0


# --- observations (car-frame, mirroring bicycle5d's obs vector) ----------------

def obs_root_vel(env) -> torch.Tensor:
  """Body-frame planar velocity + yaw rate: (vx, vy, wz)."""
  d = env.scene["robot"].data
  return torch.cat([d.root_link_lin_vel_b[:, :2], d.root_link_ang_vel_b[:, 2:3]],
                   dim=-1)


def _vec_to_car(env, targets_local: torch.Tensor) -> torch.Tensor:
  """Rotate world offsets (target - car) into the car frame. ``targets_local`` is
  (M, 2): target positions relative to the spawn origin."""
  d = env.scene["robot"].data
  quat = d.root_link_quat_w                                   # (N, 4)
  n, m = quat.shape[0], targets_local.shape[0]
  car_local = _car_local_xy(env)                              # (N, 2)
  off = torch.zeros(n, m, 3, device=quat.device)
  off[..., :2] = targets_local[None, :, :] - car_local[:, None, :]
  q = quat[:, None, :].expand(n, m, 4).reshape(-1, 4)
  return quat_apply_inverse(q, off.reshape(-1, 3)).reshape(n, m, 3)[..., :2]


def obs_goal_car(env) -> torch.Tensor:
  goal = torch.tensor([[START_TO_GOAL, 0.0]], device=env.device)
  return _vec_to_car(env, goal).reshape(env.num_envs, 2)


def obs_obstacles_car(env) -> torch.Tensor:
  obs = _obstacles(env.device)                                # (M, 3)
  rel = _vec_to_car(env, obs[:, :2])                          # (N, M, 2)
  rad = obs[:, 2][None, :, None].expand(rel.shape[0], -1, 1)
  return torch.cat([rel, rad], dim=-1).reshape(env.num_envs, -1)  # (N, 3M)


def _fix_wheel_actuators(spec) -> None:
  """The built-in <velocity> servo inherits its ctrlrange from the wheel joint,
  but the wheels spin freely (no joint range) -> "invalid control range" at
  compile. The policy already bounds the command ([-1, 1] * scale), so drop the
  ctrl limit on the two wheel actuators."""
  for a in spec.actuators:
    if a.name.split("/")[-1] in ("left", "right"):
      a.inheritrange = 0.0
      a.ctrllimited = False


# --- env cfg builder -----------------------------------------------------------

def car_goal_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  obs_terms = {
    "root_vel": ObservationTermCfg(func=obs_root_vel),
    "goal": ObservationTermCfg(func=obs_goal_car),
    "obstacles": ObservationTermCfg(func=obs_obstacles_car),
  }
  observations = {
    "actor": ObservationGroupCfg(terms=dict(obs_terms), concatenate_terms=True,
                                 history_length=1),
    "critic": ObservationGroupCfg(terms=dict(obs_terms), concatenate_terms=True,
                                  history_length=1),
  }

  actions = {
    "wheels": JointVelocityActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=_WHEEL_SPEED,
      use_default_offset=True),
  }

  events = {
    # Randomized spawn: small position + heading jitter (per-episode variety, the
    # anti-memorization role bicycle5d gets from its randomize=True). Zero initial
    # velocity -> episodes start from a STANDSTILL, so the reach-avoid-vs-avoid
    # "does it initiate from rest?" contrast is exercised.
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform, mode="reset",
      params={
        "pose_range": {"x": (-0.2, 0.2), "y": (-0.3, 0.3), "yaw": (-0.4, 0.4)},
        "velocity_range": {},
      }),
    "reset_wheels": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset, mode="reset",
      params={
        "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("left", "right")),
      }),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "collision": TerminationTermCfg(func=car_collision),
  }

  return ManagerBasedRlEnvCfg(
    decimation=4,
    episode_length_s=12.0,   # 600 control steps @ dt 0.02: ample time to weave
                            # to the 2 m goal at ~1.5 m/s (goal was unreachable at
                            # 3 m / 8 s -- see START_TO_GOAL / _WHEEL_SPEED)
    scene=SceneCfg(
      num_envs=1,
      env_spacing=2.0,
      extent=5.0,
      terrain=TerrainEntityCfg(
        terrain_type="generator", terrain_generator=CAR_ARENA_CFG),
      entities={"robot": get_car_robot_cfg()},
      spec_fn=_fix_wheel_actuators,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards={},                     # base.py injects the g-margin reward hook
    terminations=terminations,
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot", body_name="agent",
      distance=3.5, elevation=-35.0, azimuth=90.0,
      # Render ONLY the car's own arena (mjlab renders `max_extra_envs`
      # neighbor tiles by default -> the corridor of other patches receding
      # into the distance). Zero neighbors = one clean showcase arena.
      max_extra_envs=0,
      width=960, height=720),
  )
