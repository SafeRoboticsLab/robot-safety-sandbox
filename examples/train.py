"""Zoo trainer: any registered task on the GPU-resident path, full telemetry.

  python examples/train.py --task go2_gap_landing --steps 200000000 --seed 0
  python examples/train.py --task go2_gap_chain --steps 2000000000 --seed 0 \
      --load runs_zoo/go2_gap_crossing/final_model.zip

Everything a benchmark run needs is on by default: wandb (metrics + eval
videos from step 0), periodic checkpoints (+ normalizer stats), curriculum
telemetry (env/Curriculum/* — WATCH THESE: a stalled curriculum looks exactly
like converged training in the reward curve), seeding, and warm-start with
normalizer transfer + freeze. See PORTING.md for the knobs.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)
# Dev fallback: safety_sb3 is a pip dependency in release; when running from a
# source checkout, look for the sibling repo or $SAFETY_SB3_PATH.
try:
  import safety_sb3  # noqa: F401
except ImportError:
  _cand = os.environ.get(
    "SAFETY_SB3_PATH",
    os.path.join(os.path.dirname(_ZOO), "safety-stable-baselines"))
  if os.path.isdir(_cand):
    sys.path.insert(0, _cand)

import torch as th  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback  # noqa: E402

from safety_sb3 import IsaacsPPO, ReachAvoidPPO, SafetyPPO  # noqa: E402
from safe_mjlab_zoo import list_tasks, make_tensor, spec  # noqa: E402
from safe_mjlab_zoo.callbacks import (  # noqa: E402
  ForceRampCallback,
  PerEnvForceScaleCallback,
  NormFreezeCallback,
  TensorNormSaveCallback,
  VideoWandbCallback,
)

ALGOS = {"SafetyPPO": SafetyPPO, "ReachAvoidPPO": ReachAvoidPPO}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--task", required=True, help=f"one of {list_tasks()}")
  p.add_argument("--num-envs", type=int, default=2048)
  p.add_argument("--steps", type=int, default=200_000_000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--load", default=None, help="warm-start model .zip (previous stage)")
  p.add_argument("--adversary", action="store_true")
  p.add_argument("--force-max", type=float, default=50.0)
  p.add_argument("--dstb-pretrain", type=int, default=20,
                 help="IsaacsPPO dstb-pretrain ROLLOUTS (keep << total rollouts)")
  p.add_argument("--lr", type=float, default=5e-4)
  p.add_argument("--ent-coef", type=float, default=1e-4)
  p.add_argument("--video-interval", type=int, default=25_000_000)
  p.add_argument("--norm-freeze-steps", type=int, default=5_000_000,
                 help="freeze obs-norm updates at the start of WARM-STARTED runs")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=os.path.join(_ZOO, "runs_zoo"))
  p.add_argument("--wandb-project", default="safe_mjlab_zoo")
  p.add_argument("--no-wandb", action="store_true")
  args = p.parse_args()

  s = spec(args.task)
  tag = args.task + ("_adv" if args.adversary else "")
  outdir = os.path.join(args.out, tag)
  os.makedirs(outdir, exist_ok=True)

  env = make_tensor(args.task, args.num_envs, args.device, adversary=args.adversary)
  akw = {}
  if args.adversary:
    Algo = IsaacsPPO  # full two-player game, tensor path
    akw = dict(ctrl_action_dim=s.ctrl_dim,
               dstb_learning_rate=3e-4, dstb_ent_coef=2e-3,  # reference values
               dstb_pretrain_rollouts=args.dstb_pretrain,
               ctrl_rollouts_per_cycle=12, dstb_rollouts_per_cycle=3,
               use_leaderboard=True,
               leaderboard_dir=os.path.join(outdir, "leaderboard"))
  else:
    Algo = ALGOS[s.default_algo if s.default_algo in ALGOS else "ReachAvoidPPO"]

  model = Algo(
    "MlpPolicy", env, **akw,
    n_steps=48, batch_size=args.num_envs * 48 // 4, n_epochs=5,
    gamma=0.99, gae_lambda=0.95, learning_rate=args.lr,
    ent_coef=args.ent_coef, clip_range=0.2, max_grad_norm=1.0,
    normalize_obs=True, adaptive_lr=True, desired_kl=0.01,
    policy_kwargs=dict(log_std_init=math.log(0.3),
                       net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])),
    seed=args.seed, verbose=1, device=args.device,
    tensorboard_log=outdir)

  cbs = [
    CheckpointCallback(save_freq=max(1, 25_000_000 // args.num_envs),
                       save_path=os.path.join(outdir, "checkpoints"),
                       name_prefix="model"),
    TensorNormSaveCallback(os.path.join(outdir, "checkpoints")),
  ]

  if args.load:
    model.set_parameters(args.load, exact_match=False, device=args.device)
    tvn = model.env
    pt = args.load.replace("final_model.zip", "tensornormalize.pt")
    pkl = args.load.replace("final_model.zip", "vecnormalize.pkl")
    if os.path.exists(pt):
      st = th.load(pt, map_location=args.device, weights_only=True)
      tvn.obs_mean, tvn.obs_var, tvn.count = (
        st["obs_mean"].to(tvn.device), st["obs_var"].to(tvn.device),
        st["count"].to(tvn.device))
      print(f"[warm-start] {args.load} + tensor obs stats")
    elif os.path.exists(pkl):
      import pickle
      with open(pkl, "rb") as f:
        vn = pickle.load(f)
      tvn.obs_mean = th.as_tensor(vn.obs_rms.mean, dtype=th.float32, device=tvn.device)
      tvn.obs_var = th.as_tensor(vn.obs_rms.var, dtype=th.float32, device=tvn.device)
      tvn.count = th.tensor(float(vn.obs_rms.count), device=tvn.device)
      print(f"[warm-start] {args.load} + converted numpy VecNormalize stats")
    cbs.append(NormFreezeCallback(args.norm_freeze_steps))

  if args.adversary:
    cbs.append(ForceRampCallback(args.force_max, int(0.55 * args.steps)))
    cbs.append(PerEnvForceScaleCallback())

  if not args.no_wandb:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    wandb.init(project=args.wandb_project, name=tag, config=vars(args),
               sync_tensorboard=True, save_code=False, reinit=True)
    cbs.append(WandbCallback(verbose=0))
    cbs.append(VideoWandbCallback(
      lambda: make_tensor(args.task, 2, args.device, adversary=False,
                          render_mode="rgb_array"),
      interval=args.video_interval))

  model.learn(total_timesteps=args.steps, callback=CallbackList(cbs))
  model.save(os.path.join(outdir, "final_model.zip"))
  if hasattr(model.env, "save"):
    model.env.save(os.path.join(outdir, "tensornormalize.pt"))
  print(f"[done] {outdir}")


if __name__ == "__main__":
  main()
