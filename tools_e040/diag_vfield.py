"""Diagnostic: walk the nominal walker toward the gap (NO filter) and log the
safety value V(x_rel) so we can see where each twin's certificate is valid /
where it drops below 0 (engagement boundary). Tells us the spawn range + eps."""
import argparse, os, sys
_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO); sys.path.insert(0, os.path.join(_ZOO, "examples"))
import numpy as np, torch
from mjlab.envs import ManagerBasedRlEnv
from eval_filter import build_filter_env_cfg, load_walker, load_safety, CTRL_GAIN

p = argparse.ArgumentParser()
p.add_argument("--walker", required=True)
p.add_argument("--safety", required=True)
p.add_argument("--task", default="go2_gap_split2_ra_w30")
p.add_argument("--gap-width", type=float, default=0.30)
p.add_argument("--island-length", type=float, default=3.0)
p.add_argument("--num-envs", type=int, default=64)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--spawn-x", type=float, nargs=2, default=(-1.6, -1.4))
p.add_argument("--cmd-vx", type=float, default=1.0)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()
dev = a.device

cfg = build_filter_env_cfg(a.task, a.num_envs, a.gap_width, 1, 20.0, a.cmd_vx,
                           spawn_x=tuple(a.spawn_x), island_length=a.island_length)
env = ManagerBasedRlEnv(cfg=cfg, device=dev)
walker, wvn = load_walker(a.walker, dev)
safety, snorm = load_safety(a.safety, dev)
robot = env.scene["robot"]; ox = env.scene.env_origins[:, 0]

obs, _ = env.reset()
print(f"# task={a.task} safety={os.path.basename(os.path.dirname(a.safety))}")
print(f"#{'step':>4} {'x_rel':>7} {'V_mean':>8} {'V_min':>8} {'V>0%':>6} {'speed':>6} {'alive%':>6}")
for t in range(a.steps):
  w = obs["actor"].detach().cpu().numpy()
  if wvn is not None: w = wvn.normalize_obs(w)
  aw, _ = walker.predict(w, deterministic=True)
  aw = torch.as_tensor(np.clip(aw, -1, 1), dtype=torch.float32, device=dev)
  with torch.no_grad():
    V = safety.policy.predict_values(snorm(obs["proprioception"].float())).squeeze(-1)
  obs, _r, term, trunc, _e = env.step(aw * CTRL_GAIN)
  x = robot.data.root_link_pos_w[:, 0] - ox
  sp = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
  alive = (~term).float().mean()
  if t % 10 == 0 or t == a.steps - 1:
    print(f" {t:>4} {x.mean().item():>7.2f} {V.mean().item():>8.3f} {V.min().item():>8.3f} "
          f"{(V>0).float().mean().item()*100:>5.0f}% {sp.mean().item():>6.2f} {alive.item()*100:>5.0f}%")
