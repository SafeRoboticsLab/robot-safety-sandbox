"""Go2 low-bar crawl benchmark family (parkour skill 2) — momentum-reactive
filter: duck-coast-through a passable bar or brake before an impossible one.

Two-phase decomposition (mirrors gap-jumping crossing -> chain):
  go2_crawl_locomote  Phase 1: learn the crouch-crawl LOCOMOTION (velocity l)
  go2_crawl           Phase 2: decide crawl vs stop (rest l + window), warm from P1
  go2_crawl_isaacs    Phase 3: + worst-case force adversary

Env cfgs are NATIVE to the zoo (envs/go2_crawl/*).
"""

from __future__ import annotations

import torch

from ..margins import CLAMP, g_terrain_relative, l_rest
from ..registry import TaskSpec, register

_V_CMD = 1.0   # forward crawl target (match envs/go2_crawl V_CMD)
_V_TOL = 0.7   # tracking tolerance (l >= 0 within this of the command)


def crawl_locomote_margins(env):
  """Phase 1: g = crawl safety (bar strike / fall / off-ground); l = forward
  velocity tracking toward (V_CMD, 0) in world frame. l FORCES sustained
  forward motion THROUGH the bar -- the crouch-crawl motor skill that the rest
  objective let the robot avoid by stopping. No stop decision here (Phase 2)."""
  g = g_terrain_relative(env).clamp(-CLAMP, CLAMP)
  v = env.scene["robot"].data.root_link_lin_vel_w[:, :2]
  cmd = torch.tensor([_V_CMD, 0.0], device=env.device)
  l = _V_TOL - torch.linalg.norm(v - cmd, dim=1)
  return g, l.clamp(-CLAMP, CLAMP)


def crawl_margins(env):
  """g = terrain-relative failure margin; l = safe rest RESTRICTED by the
  per-row obstacle window the env publishes (env._rest_obstacle_window_w):
  passable rows exclude the whole approach (only rest PAST the bar counts ->
  crawl-through is the target), impossible rows target rest BEFORE the bar.
  Without the window, plain rest is satisfiable everywhere and 'stop always'
  wins — the documented failure mode of this task."""
  g = g_terrain_relative(env).clamp(-CLAMP, CLAMP)
  l = l_rest(env)
  win = getattr(env, "_rest_obstacle_window_w", None)
  if win is not None:
    x = env.scene["robot"].data.root_link_pos_w[:, 0]
    d_out = torch.maximum(win[:, 0] - x, x - win[:, 1])
    l = torch.minimum(l, d_out / 0.3)
  return g, l.clamp(-CLAMP, CLAMP)


def register_all() -> None:
  from safe_mjlab_zoo.envs.go2_crawl.env_cfg import (
    unitree_go2_crawl_env_cfg, unitree_go2_crawl_isaacs_env_cfg,
    unitree_go2_crawl_locomote_env_cfg)
  register(TaskSpec(
    task_id="go2_crawl_locomote", cfg_builder=unitree_go2_crawl_locomote_env_cfg,
    margin_fn=crawl_locomote_margins, default_algo="ReachAvoidPPO",
    description="Phase 1: crouch-crawl LOCOMOTION under a descending bar "
                "(velocity-tracking reach, momentum init, passable only)."))
  register(TaskSpec(
    task_id="go2_crawl", cfg_builder=unitree_go2_crawl_env_cfg,
    margin_fn=crawl_margins,
    default_algo="ReachAvoidPPO", warmstart_from="go2_crawl_locomote",
    description="Phase 2: decide crawl vs stop (rest + window), warm from P1."))
  register(TaskSpec(
    task_id="go2_crawl_isaacs", cfg_builder=unitree_go2_crawl_isaacs_env_cfg,
    margin_fn=crawl_margins,
    default_algo="IsaacsPPO", warmstart_from="go2_crawl",
    supports_adversary=True,
    description="Crawl + worst-case base-force adversary."))
