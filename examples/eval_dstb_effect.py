"""Disturbance-effect probe: is the ISAACS adversary doing anything?

Evaluates the trained CTRL policy under three disturbance conditions —
  none    : dstb = 0                       (the deployable-policy baseline)
  random  : dstb ~ U[-1, 1]                (naive robustness)
  trained : dstb = the trained min-player  (worst-case, the ISAACS game)
— and reports survival, episode length and worst-case margin. This is the
honest "adversarial effect" readout for survival tasks whose videos look like
nothing (a hopper standing still): the game lives in the margin statistics,
not the behavior.

  python examples/eval_dstb_effect.py \
      --ckpt runs/hopper_safety_adv/checkpoints/model_50003968_steps.zip \
      --task hopper_safety --episodes 512
"""

from __future__ import annotations

import argparse
import os
import sys

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get(
    "SAFETY_SB3_PATH",
    os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

import torch as th  # noqa: E402

import safety_sb3  # noqa: E402
from safety_sb3.tensor_env import TensorVecNormalize  # noqa: E402
from robot_safety_sandbox import algo_name, make_tensor, spec  # noqa: E402


def run_condition(env, norm, ctrl_policy, dstb_fn, n_episodes, ctrl_dim, device):
  obs = env.reset()
  n = env.num_envs
  ep_len = th.zeros(n, device=device)
  ep_min_g = th.full((n,), 1e9, device=device)
  done_lens, done_min_g, timeouts_hit = [], [], 0
  finished = 0
  while finished < n_episodes:
    o = norm.normalize_obs(obs) if norm is not None else obs
    with th.no_grad():
      a_ctrl = th.clamp(ctrl_policy._predict(o, deterministic=True), -1, 1)
    a_dstb = dstb_fn(o)
    obs, g, dones, touts, _l = env.step_tensor(th.cat([a_ctrl, a_dstb], dim=1))
    ep_len += 1
    ep_min_g = th.minimum(ep_min_g, g)
    if bool(dones.any()):
      d = dones.bool()
      done_lens.append(ep_len[d].clone())
      done_min_g.append(ep_min_g[d].clone())
      timeouts_hit += int((touts & d).sum())
      finished += int(d.sum())
      ep_len[d] = 0
      ep_min_g[d] = 1e9
  lens = th.cat(done_lens)[:n_episodes]
  min_gs = th.cat(done_min_g)[:n_episodes]
  return dict(
    ep_len=float(lens.mean()),
    survival=timeouts_hit / max(finished, 1),  # episodes reaching timeout
    min_g_mean=float(min_gs.mean()),
    min_g_p10=float(min_gs.quantile(0.10)),
  )


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--ckpt", required=True)
  p.add_argument("--task", default="hopper_safety")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--episodes", type=int, default=512)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--end-criterion",
                 choices=["failure", "reach-avoid", "timeout"], default=None,
                 help="WHEN the episode ends from (g,l); default = the task's "
                      "TaskSpec value. Match whatever the checkpoint trained "
                      "with. (terminal_type is restored from the loaded model.)")
  args = p.parse_args()

  s = spec(args.task)
  ctrl_dim = s.ctrl_dim
  env = make_tensor(args.task, args.num_envs, args.device, adversary=True,
                    end_criterion=args.end_criterion)
  # The two-player learner for THIS task's problem: IsaacsPPO (avoid) or
  # GameplayPPO (reach-avoid). Resolved rather than hardcoded — safety_sb3
  # v0.2.0 reused the name IsaacsPPO for the avoid game and renamed the
  # reach-avoid game to GameplayPPO, so a hardcoded IsaacsPPO.load would
  # deserialize a reach-avoid checkpoint into the wrong class.
  algo = algo_name(args.task, adversary=True)
  Algo = getattr(safety_sb3, algo, None)
  if Algo is None:
    raise SystemExit(
      f"'{args.task}' needs the '{algo}' learner, which this safety_sb3 does "
      f"not export (two-player learners need safety_sb3 >= v0.2.0).")
  # custom_objects neutralizes machine-specific state baked into the
  # checkpoint (absolute leaderboard/tensorboard paths from the training box).
  model = Algo.load(args.ckpt, device=args.device, custom_objects={
    "_use_lb": False, "_lb_dir": "/tmp/isaacs_lb_probe",
    "tensorboard_log": None})
  # obs normalizer (tensornorm saved next to checkpoints)
  d = os.path.dirname(args.ckpt)
  cand = sorted([f for f in os.listdir(d) if f.startswith("tensornorm")])
  norm = None
  if cand:
    norm = TensorVecNormalize.load(os.path.join(d, cand[-1]), env)
    norm.training = False
    print(f"[probe] obs norm: {cand[-1]}")

  dstb_dim = s.dstb_dim
  n = args.num_envs

  def dstb_none(o):
    return th.zeros(n, dstb_dim, device=args.device)

  def dstb_random(o):
    return th.rand(n, dstb_dim, device=args.device) * 2 - 1

  def dstb_trained(o):
    with th.no_grad():
      return th.clamp(model.dstb_policy._predict(o, deterministic=True), -1, 1)

  print(f"\n=== dstb-effect probe: {os.path.basename(args.ckpt)} on {args.task} "
        f"({args.episodes} episodes/condition) ===")
  print(f"{'condition':10s} {'survival':>9s} {'ep_len':>8s} {'min_g mean':>11s} {'min_g p10':>10s}")
  for name, fn in (("none", dstb_none), ("random", dstb_random),
                   ("trained", dstb_trained)):
    r = run_condition(env, norm, model.policy, fn, args.episodes, ctrl_dim,
                      args.device)
    print(f"{name:10s} {r['survival']:9.1%} {r['ep_len']:8.1f} "
          f"{r['min_g_mean']:11.3f} {r['min_g_p10']:10.3f}")
  print("\nReading: none==trained everywhere -> the adversary found nothing "
        "(ctrl robust or dstb weak). trained << none -> the game is live; "
        "survival@trained is the ROBUST survival rate (the number that matters).")
  env.close()


if __name__ == "__main__":
  main()
