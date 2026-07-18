"""SPLIT TEST: reverse curriculum (far-stance -> near-edge standstill) coupled
to a gap-width ladder (steppable -> half-a-robot), to A/B avoid-only vs
reach-avoid under an IDENTICAL setup.

ONE per-env difficulty = the mjlab terrain level (native rows):
  row 0   : gap 0.05, spawn = STABLE STANCE on the far platform (trivial)
  ...     : gap widens, spawn walks backward along the ballistic arc
  row N-1 : gap ~0.22 (~half a Go2), spawn = STANDING at the near edge with
            NO momentum and no nudges   <-- the decision point

Momentum is randomized at every level EXCEPT the top (near-edge standstill).
Promotion (move up a row = wider gap + spawn further back) fires when the env
reaches a stable far stance and survives; demotion otherwise. Each env settles
at its competence frontier, so the avoid/RA split appears in the terrain-level
distribution and the near-edge behavior.

RA target l = a stable stance on the far platform (`l_stable_far`); the avoid
twin uses `l_zero`; g and everything else are identical. Intended warm-start =
the soft-landing skill (go2_gap_landing): both twins start knowing "land
softly" (the shared mid-arc behavior) and NEITHER starts able to launch from a
standstill, so whether the near-edge jump gets discovered is the single thing
the reach term decides.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from safe_mjlab_zoo.envs.terrains.island import IslandCrossingTerrainCfg
from safe_mjlab_zoo.envs.go2_gap.gap import unitree_go2_gap_reach_avoid_env_cfg

NUM_ROWS = 8
_TOP = NUM_ROWS - 1
_X_FAR = 0.30  # x_rel past the widest gap (0.22) -> on the far platform

SPLIT_TERRAINS_CFG = TerrainGeneratorCfg(
  curriculum=True,
  size=(6.0, 2.0),
  border_width=5.0,
  num_rows=NUM_ROWS,
  num_cols=10,
  color_scheme="none",
  sub_terrains={
    "island": IslandCrossingTerrainCfg(
      proportion=1.0,
      gap_width_range=(0.05, 0.22),  # steppable -> ~half a Go2
      gap_depth=1.0,
      island_length=2.5,             # long near platform (stable stance / stand)
      back_pit_length=0.8,
    ),
  },
)

# Arc anchors [dx, dz, vx, vz], relative to (env_origin + default root state);
# the origin is the gap NEAR edge. f=0 far stance -> 0.5 apex -> 1 near stand.
_A_FAR = torch.tensor([0.90, 0.00, 0.30, 0.0])
_A_APEX = torch.tensor([0.20, 0.12, 2.20, 0.0])
_A_NEAR = torch.tensor([-0.12, 0.00, 0.00, 0.0])


def _arc(f: torch.Tensor) -> torch.Tensor:
  """f: (n,) in [0,1] -> (n,4) [dx, dz, vx, vz] along the reverse arc."""
  far, apex, near = _A_FAR.to(f), _A_APEX.to(f), _A_NEAR.to(f)
  t1 = (f / 0.5).clamp(0, 1).unsqueeze(-1)
  seg1 = far + t1 * (apex - far)
  t2 = ((f - 0.5) / 0.5).clamp(0, 1).unsqueeze(-1)
  seg2 = apex + t2 * (near - apex)
  return torch.where((f <= 0.5).unsqueeze(-1), seg1, seg2)


def reset_split_reverse(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Spawn each env along the arc per its terrain level; momentum randomized
  at all levels except the top (near-edge standstill = zero velocity)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  asset = env.scene[asset_cfg.name]
  dev = env.device
  n = int(len(env_ids))
  lvl = env.scene.terrain.terrain_levels[env_ids].float()
  f = (lvl / _TOP).clamp(0.0, 1.0)
  arc = _arc(f)
  mom = (lvl < _TOP).float()  # 0 at the top level -> exact standstill

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), dev)

  root = asset.data.default_root_state[env_ids].clone()
  dx = arc[:, 0] + u(-0.05, 0.05)
  dy = u(-0.10, 0.10)
  dz = arc[:, 1]
  roll, pitch, yaw = u(-0.08, 0.08), u(-0.10, 0.10), u(-0.12, 0.12)
  vx = arc[:, 2] * mom + mom * u(-0.8, 0.8)
  vz = arc[:, 3] * mom + mom * u(-0.4, 0.4)

  pose = torch.stack([dx, dy, dz, roll, pitch, yaw], dim=1)
  vel = torch.stack(
    [vx, u(-0.10, 0.10) * mom, vz,
     u(-0.2, 0.2) * mom, u(-0.2, 0.2) * mom, u(-0.2, 0.2) * mom], dim=1)
  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]))
  velocities = root[:, 7:13] + vel
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def split_levels(env, env_ids) -> torch.Tensor:
  """Single-axis reverse+width curriculum: move UP (wider gap + spawn further
  back toward the near edge) when the env reached a stable far stance and
  survived; move DOWN otherwise. Read on reset (ended-episode final state)."""
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
  up = -robot.data.projected_gravity_b[env_ids, 2]
  survived = env.termination_manager.time_outs[env_ids]
  reached = survived & (x_rel > _X_FAR) & (up > 0.7)
  env.scene.terrain.update_env_origins(env_ids, reached, ~reached)
  return torch.mean(env.scene.terrain.terrain_levels.float())


def l_stable_far(env, x_far=_X_FAR, pos_norm=0.20, v_rest=0.6, v_norm=0.5,
                 up_min=0.85, up_norm=0.15):
  """RA target: a STABLE STANCE on the far platform (past the gap, upright,
  nearly at rest). min-form intersection target."""
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
  up = -robot.data.projected_gravity_b[:, 2]
  past = (x_rel - x_far) / pos_norm
  rest = (v_rest - speed) / v_norm
  upright = (up - up_min) / up_norm
  return torch.minimum(torch.minimum(past, rest), upright)


def unitree_go2_split_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(SPLIT_TERRAINS_CFG)
  cfg.scene.terrain.max_init_terrain_level = 0  # everyone starts easiest
  cfg.episode_length_s = 4.0
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_split_reverse, mode="reset", params={})
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  if "push_robot" in cfg.events:
    cfg.events.pop("push_robot", None)
  cfg.curriculum = {"split_levels": CurriculumTermCfg(func=split_levels)}
  return cfg
