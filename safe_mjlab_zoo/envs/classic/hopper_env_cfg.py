"""Hopper (gym Hopper-v4 / Robust-Gymnasium SafetyHopper-v4), mjlab-native.

THE PORTING TEMPLATE for classic-control envs. Recipe (repeat per robot):
  1. assets_classic/xmls/<robot>.xml — gym XML minus floor/light/actuators/
     <option> (mjlab owns terrain, motors, integrator/timestep).
  2. assets_classic/<robot>_constants.py — EntityCfg + BuiltinMotorActuatorCfg
     replicating the gym motors (gear, ctrl range).
  3. This file: gym-EXACT obs terms, gym reset noise, healthy-state margins
     (min-form), gym dense reward (for the kind="nominal" twin), timeout.
  4. tasks/classic_safety.py: kind="safety" spec with dstb_mode="action"
     (the Robust-Gymnasium ISAACS convention) + a kind="nominal" dense twin.

Gym reference semantics (SafetyHopper-v4):
  obs  = [qpos[1:], clip(qvel, +-10)]  (11,)   qpos=[rootx,rootz,rooty,thigh,leg,foot]
  act  = 3 motor ctrls in [-1, 1] (gear 200)
  dt   = 0.008  (timestep 0.002 x frame_skip 4)  [deviation: implicit, not RK4]
  healthy (all must hold):  z in (0.7, inf);  |angle| < 0.2;
                            obs[1:] in (-100, 100)
  reset noise: uniform(+-5e-3) on all qpos and qvel
  reward = 1.0 (healthy) + vx - 1e-3 * |ctrl|^2
  ISAACS episode: timeout 200 steps (1.6 s), dstb action-additive +-0.25.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from safe_mjlab_zoo.envs.assets_classic.hopper_constants import get_hopper_robot_cfg

# gym healthy ranges (SafetyHopper-v4 defaults)
HEALTHY_Z_MIN = 0.7
HEALTHY_ANGLE = 0.2
HEALTHY_STATE = 100.0
QVEL_CLIP = 10.0
RESET_NOISE = 5e-3
# margin normalizations (O(1) per typical excursion)
Z_NORM = 0.3
ANGLE_NORM = 0.2
STATE_NORM = 10.0


# --- gym-exact reset ------------------------------------------------------------

def hopper_reset(env, env_ids, asset_cfg=SceneEntityCfg("robot")) -> None:
  """gym reset: qpos = init_qpos + U(+-5e-3), qvel = U(+-5e-3), ALL joints.

  Custom (not mdp.reset_joints_by_offset) because that helper clamps to the
  soft joint limits, and the hopper's UNLIMITED root joints read range (0,0)
  -> rootz's 1.25 default would be clamped to 0 (torso inside the floor).
  """
  from mjlab.utils.lab_api.math import sample_uniform
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  asset = env.scene[asset_cfg.name]
  n = len(env_ids)
  qpos = asset.data.default_joint_pos[env_ids].clone()
  qpos += sample_uniform(-RESET_NOISE, RESET_NOISE, qpos.shape, env.device)
  qvel = sample_uniform(-RESET_NOISE, RESET_NOISE, (n, qpos.shape[1]), env.device)
  asset.write_joint_state_to_sim(qpos, qvel, env_ids=env_ids)


# --- gym-exact observations ---------------------------------------------------

def hopper_qpos_tail(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
  """qpos[1:] = [z, angle, thigh, leg, foot] (drops rootx, as gym does)."""
  return env.scene[asset_cfg.name].data.joint_pos[:, 1:]


def hopper_qvel_clipped(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
  """clip(qvel, +-10) — all 6 joint velocities."""
  return env.scene[asset_cfg.name].data.joint_vel.clamp(-QVEL_CLIP, QVEL_CLIP)


# --- healthy margins / termination ---------------------------------------------

def hopper_health_margin(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
  """Min-form healthy margin: > 0 iff the gym hopper is 'healthy'.

  g = min((z - 0.7)/0.3, (0.2 - |angle|)/0.2, (100 - max|state|)/10) where
  state = [qpos[2:], clip(qvel, +-10)] (gym checks obs[1:]).
  """
  d = env.scene[asset_cfg.name].data
  z = d.joint_pos[:, 1]
  angle = d.joint_pos[:, 2]
  state = torch.cat(
    [d.joint_pos[:, 2:], d.joint_vel.clamp(-QVEL_CLIP, QVEL_CLIP)], dim=1)
  return torch.minimum(
    torch.minimum((z - HEALTHY_Z_MIN) / Z_NORM,
                  (HEALTHY_ANGLE - angle.abs()) / ANGLE_NORM),
    (HEALTHY_STATE - state.abs().amax(dim=1)) / STATE_NORM)


def hopper_unhealthy(env) -> torch.Tensor:
  return hopper_health_margin(env) < 0.0


# --- gym dense reward (the kind="nominal" twin) ---------------------------------

def hopper_forward_vel(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
  """gym forward_reward: rootx velocity (dx/dt)."""
  return env.scene[asset_cfg.name].data.joint_vel[:, 0]


def hopper_healthy_reward(env) -> torch.Tensor:
  return (hopper_health_margin(env) >= 0.0).float()


def hopper_ctrl_cost(env) -> torch.Tensor:
  return env.action_manager.action.square().sum(dim=1)


# --- env cfg --------------------------------------------------------------------

def hopper_env_cfg(play: bool = False, episode_s: float = 1.6
                   ) -> ManagerBasedRlEnvCfg:
  """episode_s=1.6 == the 200-step ISAACS timeout; gym default would be 8.0."""
  from mjlab.envs.mdp.actions import JointEffortActionCfg

  obs_terms = {
    "qpos_tail": ObservationTermCfg(func=hopper_qpos_tail),
    "qvel": ObservationTermCfg(func=hopper_qvel_clipped),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=dict(obs_terms), concatenate_terms=True,
      enable_corruption=False, history_length=1),
    "critic": ObservationGroupCfg(
      terms=dict(obs_terms), concatenate_terms=True,
      enable_corruption=False, history_length=1),
  }

  actions = {
    "joint_effort": JointEffortActionCfg(
      entity_name="robot",
      actuator_names=("thigh_joint", "leg_joint", "foot_joint"),
      scale=1.0),  # policy action == gym ctrl in [-1, 1]
  }

  events = {
    # gym reset: uniform(+-5e-3) around defaults on ALL qpos & qvel (root incl.)
    "reset_robot_joints": EventTermCfg(func=hopper_reset, mode="reset",
                                       params={}),
  }

  rewards = {
    "healthy": RewardTermCfg(func=hopper_healthy_reward, weight=1.0),
    "forward": RewardTermCfg(func=hopper_forward_vel, weight=1.0),
    "ctrl_cost": RewardTermCfg(func=hopper_ctrl_cost, weight=-1e-3),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "unhealthy": TerminationTermCfg(func=hopper_unhealthy),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      entities={"robot": get_hopper_robot_cfg()},
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=1.0,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot", body_name="torso",
      distance=3.0, elevation=-10.0, azimuth=90.0),
    sim=SimulationCfg(
      nconmax=10, njmax=50,
      mujoco=MujocoCfg(timestep=0.002, iterations=10, ls_iterations=20),
    ),
    decimation=4,          # dt = 0.008, gym frame_skip
    episode_length_s=episode_s,
  )
