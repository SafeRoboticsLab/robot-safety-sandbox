"""Reach-avoid vs avoid-only SAFETY FILTER on a walk-in approach to a gap.

Start the robot STANDING ~1.5 m back from the gap edge, drive it forward with
the blind flat walker, and shield with a safety twin's V(s) + fallback (the
library ValueShield). Question: does the RA certificate let the robot walk
freely and engage only to JUMP at the edge, where avoid-only brakes early?

Split_v2 removes the joint-reset event, so we set the FULL standing pose (root at
default height, spawn-x back, + default joints) or the robot folds at reset.

  # avoid twin (SafetyPPO): the certificate brakes early -> robot livelocks short of the gap
  python examples/eval_split_filter.py --safety runs/<avoid_run>/final_model.zip \
      --spawn-x-rel -0.8 --cmd-vx 0.85 --label avoid --out avoid.mp4
  # reach-avoid twin (ReachAvoidPPO): defers to the walker, engages only to JUMP at the edge
  python examples/eval_split_filter.py --safety runs/<ra_run>/final_model.zip \
      --spawn-x-rel -0.8 --cmd-vx 0.85 --label RA --out ra.mp4

The walker is the blind flat velocity policy (runs_dense/go2_walker_flat). Border
tint in the video = fraction of the herd under filter control (green walker / red safety).
"""
from __future__ import annotations

import argparse
import os
import sys

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)
sys.path.insert(0, os.path.join(_ZOO, "examples"))

import imageio
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from robot_safety_sandbox.filters import ValueShield
from eval_filter import build_filter_env_cfg, load_walker, load_safety, CTRL_GAIN

SPAWN_X_REL = -1.5   # metres back from the gap edge (origin is AT the edge)


def reset_standing_full(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Valid standing spawn: default root height, SPAWN_X_REL back, + DEFAULT
  joints (brake_or_jump dropped the joint reset -> must set it or the robot folds)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  asset = env.scene[asset_cfg.name]
  n = len(env_ids)
  root = asset.data.default_root_state[env_ids].clone()
  origins = env.scene.env_origins[env_ids]
  pos = root[:, 0:3] + origins
  pos[:, 0] = origins[:, 0] + SPAWN_X_REL
  asset.write_root_link_pose_to_sim(torch.cat([pos, root[:, 3:7]], dim=-1),
                                    env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device),
                                        env_ids=env_ids)
  jp = asset.data.default_joint_pos[env_ids].clone()
  asset.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)


def _border(frame, frac, bw=14):
  c = np.array([int(255 * frac), int(200 * (1 - frac)), 30], dtype=np.uint8)
  f = frame.copy()
  f[:bw, :] = c; f[-bw:, :] = c; f[:, :bw] = c; f[:, -bw:] = c
  return f


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--safety", required=True)
  p.add_argument("--spawn-x-rel", type=float, default=-1.5,
                 help="metres back from the gap edge to spawn (edge = 0)")
  p.add_argument("--walker", default="runs_dense/go2_walker_flat/final_model.zip")
  p.add_argument("--task", default="go2_gap_brake_or_jump_ra_w30")
  p.add_argument("--gap-width", type=float, default=0.30)
  p.add_argument("--island-length", type=float, default=3.0)
  p.add_argument("--num-envs", type=int, default=6)
  p.add_argument("--steps", type=int, default=350)
  p.add_argument("--cmd-vx", type=float, default=1.0)
  p.add_argument("--eps", type=float, default=0.0)
  p.add_argument("--caution", type=float, default=0.45)
  p.add_argument("--hysteresis", type=float, default=0.15)
  p.add_argument("--no-filter", action="store_true")
  p.add_argument("--full-horizon", action="store_true",
                 help="disable early termination (fell/contact/below-terrain) so "
                      "each robot plays one continuous take to the horizon")
  p.add_argument("--out", default=None, help="mp4 path (omit -> no video)")
  p.add_argument("--label", default="")
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()
  dev = args.device
  render = args.out is not None
  global SPAWN_X_REL
  SPAWN_X_REL = args.spawn_x_rel

  cfg = build_filter_env_cfg(args.task, args.num_envs, args.gap_width, 1, 20.0,
                             args.cmd_vx, spawn_x=(SPAWN_X_REL, SPAWN_X_REL),
                             island_length=args.island_length)
  cfg.events["reset_base"] = EventTermCfg(func=reset_standing_full, mode="reset",
                                          params={})
  if args.full_horizon:
    for k in list(cfg.terminations):
      if k != "time_out":
        cfg.terminations.pop(k)
    print(f"[full-horizon] terminations kept: {list(cfg.terminations)}")
  env = ManagerBasedRlEnv(cfg=cfg, device=dev,
                          render_mode="rgb_array" if render else None)
  walker, wvn = load_walker(args.walker, dev)
  safety, snorm = load_safety(args.safety, dev)
  robot = env.scene["robot"]; ox = env.scene.env_origins[:, 0]
  n = args.num_envs

  def value_fn(s_obs):
    with torch.no_grad():
      return safety.policy.predict_values(s_obs).squeeze(-1)

  def fallback_fn(s_obs):
    with torch.no_grad():
      return torch.clamp(safety.policy._predict(s_obs, deterministic=True), -1, 1)

  filt = ValueShield(n, dev, value_fn, fallback_fn, eps=args.eps,
                     caution=args.caution, hysteresis=args.hysteresis)

  obs, _ = env.reset()
  prev_done = torch.ones(n, dtype=torch.bool, device=dev)
  frames = []
  reached = torch.zeros(n, dtype=torch.bool, device=dev)
  fell = torch.zeros(n, dtype=torch.bool, device=dev)
  print(f"# label={args.label} safety={os.path.basename(os.path.dirname(args.safety))} "
        f"filter={'OFF' if args.no_filter else 'ON'} spawn_x={SPAWN_X_REL} gap={args.gap_width}")
  print(f"#{'t':>4} {'x_rel':>7} {'V':>7} {'eng%':>5} {'spd':>5} {'alive%':>6}")
  for t in range(args.steps):
    w = obs["actor"].detach().cpu().numpy()
    if wvn is not None: w = wvn.normalize_obs(w)
    aw, _ = walker.predict(w, deterministic=True)
    aw = torch.as_tensor(np.clip(aw, -1, 1), dtype=torch.float32, device=dev)
    s_obs = snorm(obs["proprioception"].float())
    speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
    action, finfo = filt.act(aw, speed=speed, fresh=prev_done, s_obs=s_obs)
    eng, cau, V = finfo.engaged, finfo.caution, finfo.value
    if args.no_filter:
      action = aw; eng = torch.zeros_like(eng); cau = torch.zeros_like(cau)
    cmd = env.command_manager.get_command("twist")
    cmd[:, 0] = torch.where(eng, torch.ones_like(speed),
                torch.where(cau, torch.zeros_like(speed),
                            torch.full_like(speed, args.cmd_vx)))
    obs, _r, term, trunc, _e = env.step(action * CTRL_GAIN)
    x = robot.data.root_link_pos_w[:, 0] - ox
    reached |= x > (args.gap_width + 0.15)
    fell |= term & ~reached
    done = term | trunc; prev_done = done.clone()
    if bool(done.any()):
      filt.reset(done); reached &= ~done; fell &= ~done
    if render:
      frames.append(_border(np.asarray(env.render()), float(eng.float().mean())))
    if t % 25 == 0 or t == args.steps - 1:
      print(f" {t:>4} {x.mean().item():>7.2f} {V.mean().item():>7.3f} "
            f"{eng.float().mean().item()*100:>4.0f}% {speed.mean().item():>5.2f} "
            f"{(~term).float().mean().item()*100:>5.0f}%")

  print(f"[result] {args.label}: reached_far={float(reached.float().mean()):.2f}  "
        f"fell={float(fell.float().mean()):.2f}  "
        f"max_x_rel={float((robot.data.root_link_pos_w[:,0]-ox).max()):.2f}  "
        f"intervention_rate={filt.intervention_rate(args.steps):.2f}")
  if render:
    imageio.mimwrite(args.out, frames, fps=30, macro_block_size=1)
    print(f"[video] {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
  main()
