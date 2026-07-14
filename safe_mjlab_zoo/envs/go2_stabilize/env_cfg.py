"""Go2 flat-ground stabilization / locomotion vs an adversarial force — the
ORIGINAL safety task of this line of work (ISAACS Tier 2), and deliberately
the SIMPLEST task in the zoo:

  * flat terrain, stock velocity-task spawns — no custom spawn distribution
  * NO curriculum of any kind
  * no staged pipeline — trainable from scratch in one run
  * margins are a handful of body-frame inequalities

Contrast with go2_gap (staged pipeline + reverse curricula) and go2_crawl
(momentum-filter spawns + rest windows): every design dimension is a per-task
choice, and this task chooses "none of the above". Use it as the porting
starting point when your task doesn't need machinery.

Tasks:
  stabilize: g = min trunk-corner height (don't hit the ground);
             l = stance band (corners in [0.10, 0.40], |v| and |w| small)
             -> V > 0 == "can return to a stable stand despite the adversary"
  locomote:  g = min(base-height, tilt); l = tracking the velocity command
             -> V > 0 == "can keep moving at the command despite the adversary"
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.utils.lab_api.math import quat_apply

from safe_mjlab_zoo.envs.velocity.go2 import unitree_go2_flat_env_cfg

# Go2 trunk bounding-box half-extents (m): length, width, height.
TRUNK_HALF = (0.18, 0.045, 0.05)
CORNER_LOW = 0.10
CORNER_HIGH = 0.40
LIN_TOL = 0.20
ANG_TOL = 0.174

Z_MIN = 0.18
Z_SCALE = 0.15
TILT_COS = 0.64
TILT_NORM = 1.0 - TILT_COS
LOCO_VEL_TOL = 0.5


def _corner_offsets(device, dtype):
  hl, hw, hh = TRUNK_HALF
  pts = [[sx * hl, sy * hw, sz * hh]
         for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]
  return torch.tensor(pts, device=device, dtype=dtype)


def stance_margins(env):
  """(g, l): corner-height safety + stabilize-to-stance target."""
  d = env.scene["robot"].data
  pos, quat = d.root_link_pos_w, d.root_link_quat_w
  v_b, w_b = d.root_link_lin_vel_b, d.root_link_ang_vel_b
  n = pos.shape[0]
  offs = _corner_offsets(pos.device, pos.dtype)
  q = quat[:, None, :].expand(n, 8, 4).reshape(-1, 4)
  o = offs[None].expand(n, 8, 3).reshape(-1, 3)
  corners_w = pos[:, None, :].expand(n, 8, 3).reshape(-1, 3) + quat_apply(q, o)
  cz = corners_w.reshape(n, 8, 3)[..., 2]
  min_c, max_c = cz.amin(dim=1), cz.amax(dim=1)
  g = min_c - CORNER_LOW
  l = torch.stack([
    min_c - CORNER_LOW, CORNER_HIGH - max_c,
    LIN_TOL - v_b[:, 0].abs(), LIN_TOL - v_b[:, 1].abs(),
    LIN_TOL - v_b[:, 2].abs(),
    ANG_TOL - w_b[:, 0].abs(), ANG_TOL - w_b[:, 1].abs(),
    ANG_TOL - w_b[:, 2].abs(),
  ], dim=0).amin(dim=0)
  return g, l


def locomote_margins(env):
  """(g, l): upright/off-floor safety + velocity-command-tracking target."""
  d = env.scene["robot"].data
  base_z = d.root_link_pos_w[:, 2]
  up = -d.projected_gravity_b[:, 2]
  v_b = d.root_link_lin_vel_b[:, :2]
  cmd = env.command_manager.get_command("twist")[:, :2]
  g = torch.minimum((base_z - Z_MIN) / Z_SCALE, (up - TILT_COS) / TILT_NORM)
  l = LOCO_VEL_TOL - torch.linalg.norm(v_b - cmd, dim=1)
  return g, l


def _pin_twist(cfg, vx: float) -> None:
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (vx, vx)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  if hasattr(twist, "rel_standing_envs"):
    twist.rel_standing_envs = 0.0
  if hasattr(twist, "heading_command"):
    twist.heading_command = False
    if getattr(twist.ranges, "heading", None) is not None:
      twist.ranges.heading = None


def go2_stabilize_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_flat_env_cfg(play=play)
  _pin_twist(cfg, 0.0)     # zero command: the target is a stable stand
  return cfg


def go2_locomote_env_cfg(play: bool = False, cmd_vx: float = 1.0) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_flat_env_cfg(play=play)
  _pin_twist(cfg, cmd_vx)  # constant forward drive: the target is tracking it
  return cfg
