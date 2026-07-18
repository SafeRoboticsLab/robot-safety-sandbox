"""Filter-claim DEMO video: nominal walker + value-shield safety filter, rendered.

Reuses examples/eval_filter.py's env/policy plumbing (ValueShield + the chain/
island walk-in approach) but runs a small herd with render_mode and writes an
mp4, tinting the frame border by how much of the herd is under FILTER control
(green = walker driving, red = safety fallback engaged). Run once per twin:

  avoid twin -> herd walks up, filter engages at the braking boundary, robots
                STOP short of the gap (livelock).
  RA twin    -> filter lets them reach the edge, engages to JUMP, releases, they
                walk on.

  python tools_e040/filter_demo_video.py \
    --walker runs_dense/go2_walker_flat/final_model.zip \
    --safety runs_e025_ref/avoid_w30/final_model.zip \
    --task go2_gap_split2_ra_w30 --gap-width 0.30 --island-length 3.0 \
    --num-envs 6 --steps 400 --spawn-x -1.6 -1.4 --gap-x 0.0 --rest-x 1.2 \
    --out tools_e040/filter_avoid.mp4 --label "avoid filter"
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
from robot_safety_sandbox.filters import ValueShield
from eval_filter import build_filter_env_cfg, load_walker, load_safety, CTRL_GAIN


def _border(frame, frac_engaged, bw=14):
  """Tint a border: green (walker) -> red (safety fallback) by frac_engaged."""
  c = np.array([int(255 * frac_engaged), int(200 * (1 - frac_engaged)), 30],
               dtype=np.uint8)
  f = frame.copy()
  f[:bw, :] = c; f[-bw:, :] = c; f[:, :bw] = c; f[:, -bw:] = c
  return f


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--walker", required=True)
  p.add_argument("--safety", required=True)
  p.add_argument("--task", default="go2_gap_split2_ra_w30")
  p.add_argument("--gap-width", type=float, default=0.30)
  p.add_argument("--island-length", type=float, default=3.0)
  p.add_argument("--num-envs", type=int, default=6)
  p.add_argument("--steps", type=int, default=400)
  p.add_argument("--episode-s", type=float, default=20.0)
  p.add_argument("--cmd-vx", type=float, default=1.0)
  p.add_argument("--spawn-x", type=float, nargs=2, default=(-1.6, -1.4))
  p.add_argument("--gap-x", type=float, default=0.0)
  p.add_argument("--rest-x", type=float, default=1.2)
  p.add_argument("--eps", type=float, default=0.0)
  p.add_argument("--caution", type=float, default=0.45)
  p.add_argument("--hysteresis", type=float, default=0.15)
  p.add_argument("--fps", type=int, default=30)
  p.add_argument("--out", required=True)
  p.add_argument("--label", default="")
  p.add_argument("--device", default="cuda:0")
  args = p.parse_args()
  dev = args.device

  cfg = build_filter_env_cfg(args.task, args.num_envs, args.gap_width, 1,
                             args.episode_s, args.cmd_vx,
                             spawn_x=tuple(args.spawn_x),
                             island_length=args.island_length)
  env = ManagerBasedRlEnv(cfg=cfg, device=dev, render_mode="rgb_array")
  walker, wvn = load_walker(args.walker, dev)
  safety, snorm = load_safety(args.safety, dev)
  robot = env.scene["robot"]
  ox = env.scene.env_origins[:, 0]
  n = args.num_envs

  def value_fn(s_obs):
    with torch.no_grad():
      return safety.policy.predict_values(s_obs).squeeze(-1)

  def fallback_fn(s_obs):
    with torch.no_grad():
      return torch.clamp(safety.policy._predict(s_obs, deterministic=True), -1, 1)

  filt = ValueShield(n, dev, value_fn, fallback_fn, eps=args.eps,
                     caution=args.caution, hysteresis=args.hysteresis)

  obs_dict, _ = env.reset()
  prev_done = torch.ones(n, dtype=torch.bool, device=dev)
  frames, crossed_ever = [], torch.zeros(n, dtype=torch.bool, device=dev)
  for t in range(args.steps):
    w_obs = obs_dict["actor"].detach().cpu().numpy()
    if wvn is not None:
      w_obs = wvn.normalize_obs(w_obs)
    a_walk, _ = walker.predict(w_obs, deterministic=True)
    a_walk = torch.as_tensor(np.clip(a_walk, -1, 1), dtype=torch.float32, device=dev)
    s_obs = snorm(obs_dict["proprioception"].float())
    speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
    action, finfo = filt.act(a_walk, speed=speed, fresh=prev_done, s_obs=s_obs)
    engaged, caution = finfo.engaged, finfo.caution
    cmd = env.command_manager.get_command("twist")
    cmd[:, 0] = torch.where(
      engaged, torch.ones_like(speed),
      torch.where(caution, torch.zeros_like(speed),
                  torch.full_like(speed, args.cmd_vx)))
    obs_dict, _r, term, trunc, _e = env.step(action * CTRL_GAIN)
    x_rel = robot.data.root_link_pos_w[:, 0] - ox
    crossed_ever |= x_rel > args.rest_x
    done = term | trunc
    prev_done = done.clone()
    if bool(done.any()):
      filt.reset(done)
      crossed_ever &= ~done
    frames.append(_border(np.asarray(env.render()),
                          float(engaged.float().mean())))

  imageio.mimwrite(args.out, frames, fps=args.fps, macro_block_size=1)
  eng = filt.intervention_rate(args.steps)
  print(f"[demo] {args.label or args.out}: {len(frames)} frames -> {args.out}")
  print(f"[demo] intervention_rate={eng:.2f}  crossed(any-alive)={float(crossed_ever.float().mean()):.2f}"
        f"  max_x_rel={float((robot.data.root_link_pos_w[:,0]-ox).max()):.2f}  (rest_x={args.rest_x})")


if __name__ == "__main__":
  main()
