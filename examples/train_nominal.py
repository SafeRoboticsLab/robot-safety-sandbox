"""NOMINAL task-policy trainer (vanilla SB3 PPO, standard GAE, dense reward).

Trains kind="nominal" tasks — the task policies safety filters wrap. Nothing
safety-related happens here: the env's own dense reward stack via the numpy
bridge (auto-dense for nominal tasks), vanilla stable_baselines3. Safety
training (margins + safety_sb3 learners) lives in train.py; the composition
is examples/eval_filter.py.

  python examples/train_nominal.py --task go2_walker_flat --num-envs 4096
  python examples/train_nominal.py --task go2_crawl_walk --num-envs 2560
"""

from __future__ import annotations

import argparse
import math
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

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize  # noqa: E402

from robot_safety_sandbox import list_tasks, make_numpy, make_tensor, spec  # noqa: E402
from robot_safety_sandbox.callbacks import (  # noqa: E402
  DenseMetricsCallback, DenseVideoWandbCallback, VecNormSaveCallback)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--task", required=True)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--steps", type=int, default=150_000_000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--lr", type=float, default=3e-4)
  p.add_argument("--ent-coef", type=float, default=5e-3)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=os.path.join(_ZOO, "runs"),
                 help="output root; ALWAYS keep runs under runs/ (git-ignored) — "
                      "never invent runs_<suffix> siblings, they escape .gitignore")
  p.add_argument("--wandb-project", default="robot_safety_sandbox")
  p.add_argument("--video-interval", type=int, default=10_000_000,
                 help="env-steps between wandb eval clips of the unaided gait")
  p.add_argument("--no-wandb", action="store_true")
  p.add_argument("--load", default=None,
                 help="warm-start: PPO.load this zip (safety_sb3 zips OK)")
  p.add_argument("--load-tensornorm", default=None,
                 help="transplant obs normalizer from a tensornormalize.pt")
  p.add_argument("--reset-log-std", type=float, default=None)
  args = p.parse_args()

  s = spec(args.task)
  if s.kind != "nominal":
    raise SystemExit(
      f"'{args.task}' is a {s.kind} task — train it with train.py "
      f"(this trainer is for kind='nominal': {list_tasks(kind='nominal')})")
  outdir = os.path.join(args.out, args.task)
  os.makedirs(outdir, exist_ok=True)

  # numpy bridge, auto-dense for nominal tasks (reward = env dense reward
  # stack); vanilla PPO gets standard GAE + timeout bootstrap.
  env = make_numpy(args.task, args.num_envs, args.device)
  # VecMonitor (UNDER VecNormalize) tracks per-env return/length and injects
  # info["episode"] -> SB3 logs rollout/ep_rew_mean + ep_len_mean on the RAW
  # (un-normalized) dense reward. The bare bridge emits no episode infos, so
  # without this the whole rollout/ section is missing.
  env = VecMonitor(env)
  # dense reward CAN be normalized (unlike the safety margin g).
  env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

  if args.load:
    from stable_baselines3.common.buffers import RolloutBuffer as _RB
    model = PPO.load(
      args.load, env=env, device=args.device,
      custom_objects={"tensorboard_log": outdir, "learning_rate": args.lr,
                      "ent_coef": args.ent_coef,
                      "n_steps": 24, "batch_size": args.num_envs * 24 // 4,
                      # safety_sb3 zips carry the tensor rollout buffer; the
                      # numpy bridge needs the stock numpy buffer.
                      "rollout_buffer_class": _RB,
                      "rollout_buffer_kwargs": {}})

    print(f"[warm-start] loaded {args.load}")
  else:
    model = PPO(
      "MlpPolicy", env, n_steps=24, batch_size=args.num_envs * 24 // 4, n_epochs=5,
      gamma=0.99, gae_lambda=0.95, learning_rate=args.lr, ent_coef=args.ent_coef,
      clip_range=0.2, max_grad_norm=1.0,
      policy_kwargs=dict(log_std_init=math.log(0.5),
                         net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])),
      seed=args.seed, verbose=1, device=args.device, tensorboard_log=outdir)
  if args.load_tensornorm:
    import torch as _th
    _st = _th.load(args.load_tensornorm, map_location="cpu", weights_only=True)
    env.obs_rms.mean = _st["obs_mean"].numpy().astype("float64")
    env.obs_rms.var = _st["obs_var"].numpy().astype("float64")
    env.obs_rms.count = 1.0e6
    print(f"[warm-start] obs normalizer transplanted from {args.load_tensornorm}")
  if args.reset_log_std is not None:
    import torch as _th
    with _th.no_grad():
      model.policy.log_std.fill_(math.log(args.reset_log_std))
    print(f"[warm-start] log_std reset -> {args.reset_log_std}")

  ckpt_dir = os.path.join(outdir, "checkpoints")
  cbs = [
    CheckpointCallback(save_freq=max(1, 10_000_000 // args.num_envs),
                       save_path=ckpt_dir, name_prefix="model"),
    # policy checkpoints alone can't be evaluated -> save the obs normalizer too.
    VecNormSaveCallback(ckpt_dir, save_freq_steps=10_000_000),
    # per-term reward / termination breakdown (env/*) from the bridge metrics().
    DenseMetricsCallback(),
  ]

  if not args.no_wandb:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    wandb.init(project=args.wandb_project, name=args.task + "_nominal",
               config=vars(args), sync_tensorboard=True, save_code=False,
               reinit=True)
    cbs.append(WandbCallback(verbose=0))
    # Eval clips: a "<task>_video" packed-terrain herd variant if registered
    # (auto-dense: no force / no safety hook -> the unaided policy).
    _vtask = args.task + "_video"
    _vtask = _vtask if _vtask in list_tasks() else args.task
    cbs.append(DenseVideoWandbCallback(
      lambda: make_tensor(_vtask, 8, args.device, render_mode="rgb_array"),
      interval=args.video_interval))

  model.learn(total_timesteps=args.steps, callback=cbs)
  model.save(os.path.join(outdir, "final_model.zip"))
  env.save(os.path.join(outdir, "vecnormalize.pkl"))
  print(f"[done] {outdir}")


if __name__ == "__main__":
  main()
