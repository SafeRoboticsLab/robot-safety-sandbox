"""Two-player reach-avoid SAC trainer (GameplaySAC / ISAACS-SAC family).

A SEPARATE entrypoint from `train.py` (which is PPO-only: its ALGOS dict and
n_steps/n_epochs/clip_range/two-player-PPO-cycle knobs are PPO-specific). This
script runs the SAC-side of the 2x2 — the two-player REACH-AVOID game
`GameplaySAC` (Gameplay Filters eq. 6a, Hsu et al. 2024) — on the GPU-resident
tensor path, retraining a task's ctrl policy against a learned worst-case
disturbance actor.

  python examples/train_gameplay_sac.py --smoke --no-wandb --num-envs 64 --steps 300000
  python examples/train_gameplay_sac.py --num-envs 1024 --steps 100000000 --seed 0

Feature under test: discount (gamma) ANNEALING, now ON by default in safety_sb3
(gamma 0.99 -> 0.9999 over the first 50% of training). We pass gamma=0.99 and
leave the anneal on; watch `train/gamma` climb.

SAC hypers mirror the reference `safe_adaptation_dev/config/go2_pybullet_isaacs_br.yaml`
(critic_0 / actor_0 / actor_1): lr 1e-4, tau 0.01, target_update_interval 2,
entropy auto-tuned from 0.1, actor net 256x3 / critic net 128x3.
"""

from __future__ import annotations

import argparse
import os
import sys

_ZOO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ZOO)
# Dev fallback: safety_sb3 is a pip dependency in release; when running from a
# source checkout, look for the sibling repo or $SAFETY_SB3_PATH (mirrors train.py).
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

from safety_sb3 import GameplaySAC  # noqa: E402
from robot_safety_sandbox import algo_name, list_tasks, make_numpy, make_tensor, spec  # noqa: E402
from robot_safety_sandbox.callbacks import (  # noqa: E402
  ForceRampCallback,
  PerEnvForceScaleCallback,
  TensorNormSaveCallback,
  VideoWandbCallback,
)

TASK = "go2_stabilize"
TAG = "go2_stabilize_gameplaysac"


class _NormEvalVecEnv:
  """Wrap the numpy leaderboard-eval VecEnv so its obs are normalized with the
  training TensorVecNormalize running stats before reaching the actors.

  WHY a numpy env (not the prompt's suggested make_tensor): the leaderboard's
  `GameplaySAC._eval_pair` routes any env exposing `num_envs` + `step_async` to
  the vec loop `_eval_pair_vec`, which drives it with the CLASSIC numpy VecEnv
  API (`step_async` / `step_wait` returning `(obs, g, dones, infos)` with
  `info["l_x"]`). A `TensorVecEnv` deliberately RAISES on `step_async`/`step_wait`
  (tensor path only), so a tensor eval env would crash the first leaderboard
  step. `make_numpy(..., adversary=True)` gives the exact surface it needs.

  WHY normalize: the policy trains on normalized obs (normalize_obs=True wraps
  the train env in TensorVecNormalize), but the numpy eval env returns RAW obs.
  Feeding raw obs to the actors would make every leaderboard score meaningless.
  We normalize with the SAME running stats (rsl_rl train/eval parity). The norm
  fn is injected AFTER model construction (the normalizer doesn't exist before).
  """

  def __init__(self, inner):
    self.inner = inner
    self.num_envs = inner.num_envs
    self._norm = None  # callable(np.ndarray) -> np.ndarray, set post-construction

  def set_normalizer(self, fn):
    self._norm = fn

  def _n(self, obs):
    return obs if self._norm is None else self._norm(obs)

  def reset(self):
    return self._n(self.inner.reset())

  def step_async(self, actions):
    self.inner.step_async(actions)

  def step_wait(self):
    obs, g, dones, infos = self.inner.step_wait()
    return self._n(obs), g, dones, infos

  def close(self):
    self.inner.close()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--steps", type=int, default=100_000_000)
  p.add_argument("--lr", type=float, default=1e-4)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=os.path.join(_ZOO, "runs"),
                 help="output root; ALWAYS keep runs under runs/ (git-ignored) — "
                      "never invent runs_<suffix> siblings, they escape .gitignore")
  p.add_argument("--wandb-project", default="robot_safety_sandbox")
  p.add_argument("--no-wandb", action="store_true")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--gradient-steps", type=int, default=4,
                 help="SGD updates per collect (tensor path: NEVER -1, which "
                      "means num_envs updates/step). Small int, 2-4.")
  p.add_argument("--batch-size", type=int, default=4096)
  p.add_argument("--buffer-size", type=int, default=1_000_000)
  p.add_argument("--force-max", type=float, default=50.0)
  p.add_argument("--force-ramp-frac", type=float, default=0.55)
  p.add_argument("--force-floor", type=float, default=0.3)
  p.add_argument("--force-init", type=float, default=0.5)
  p.add_argument("--end-criterion", choices=["failure", "reach-avoid", "timeout"],
                 default=None, help="WHEN the episode ends from (g,l); default = "
                 "the task's TaskSpec value (go2_stabilize -> failure).")
  p.add_argument("--video-interval", type=int, default=5_000_000)
  p.add_argument("--smoke", action="store_true",
                 help="tiny-budget verification: shrink learning_starts / "
                      "leaderboard cadence / eval sizes so a short run exercises "
                      "every code path (compose, gamma anneal, leaderboard).")
  # --- gamma annealing (reference-faithful discrete jumps by default) ---
  p.add_argument("--gamma-schedule", choices=["step", "geometric", "off"],
                 default="step", help="discount anneal shape: 'step' = REFERENCE "
                 "discrete jumps (0.99->0.999@20%%->0.9999@40%%, hold; resets "
                 "alpha on each jump) [DEFAULT]; 'geometric' = smooth to end by "
                 "--gamma-anneal-frac; 'off' = constant gamma.")
  p.add_argument("--gamma-init", type=float, default=0.99, help="starting gamma")
  p.add_argument("--gamma-end", type=float, default=0.9999, help="final gamma held after annealing")
  p.add_argument("--gamma-period-frac", type=float, default=0.20,
                 help="step schedule: horizon fraction between jumps (~10-20%%)")
  p.add_argument("--gamma-ratio", type=float, default=0.1,
                 help="step schedule: gap (1-gamma) multiplier per jump")
  p.add_argument("--gamma-anneal-frac", type=float, default=0.5,
                 help="geometric schedule: horizon fraction to reach --gamma-end")
  # --- entropy-temperature (alpha) floor/ceiling ---
  p.add_argument("--min-alpha", type=float, default=1e-3,
                 help="floor on the learned entropy temperature (reference 1e-3)")
  p.add_argument("--max-alpha", type=float, default=None,
                 help="optional ceiling on the entropy temperature (default none)")
  # --- safe/success-rate evaluation (logged to wandb) ---
  p.add_argument("--eval-rollouts", type=int, default=100,
                 help="episodes per safe/success-rate eval (reference ~100)")
  p.add_argument("--eval-freq", type=int, default=2_000_000,
                 help="env-steps between safe/success-rate evals (0 = off)")
  p.add_argument("--eval-envs", type=int, default=128,
                 help="parallel envs in the (separate) eval env")
  # --- per-agent learning rates (None -> fall back to --lr; reference go2 uses
  # a uniform 1e-4 for the nets, with a smaller dstb entropy lr ~5e-5) ---
  p.add_argument("--critic-lr", type=float, default=None)
  p.add_argument("--dstb-lr", type=float, default=None, help="dstb ACTOR lr")
  p.add_argument("--ent-coef-lr", type=float, default=None, help="ctrl entropy(alpha) lr")
  p.add_argument("--dstb-ent-coef-lr", type=float, default=None, help="dstb entropy(alpha) lr")
  p.add_argument("--lr-schedule", action="store_true",
                 help="enable StepLR decay of the ctrl/dstb/critic lrs")
  p.add_argument("--lr-period", type=int, default=1_000_000)
  p.add_argument("--lr-decay", type=float, default=0.1)
  p.add_argument("--lr-end", type=float, default=0.0)
  args = p.parse_args()

  # --- resolve task + learner (must be the two-player reach-avoid game) ---
  s = spec(TASK)
  if s.kind != "safety":
    raise SystemExit(f"'{TASK}' is a {s.kind} task, not a safety task.")
  algo = algo_name(TASK, adversary=True)
  if algo != "GameplayPPO":
    # algo_name returns the PPO-family NAME for the (reach-avoid, 2-player) cell;
    # the SAC analog we instantiate is GameplaySAC (same problem, off-policy).
    raise SystemExit(
      f"'{TASK}' + --adversary resolves to '{algo}', not the two-player "
      f"reach-avoid game — this script only trains GameplaySAC.")
  ctrl_dim = s.ctrl_dim  # 12 leading control action dims; env action = [ctrl, dstb]
  print(f"[algo] {TASK} adversary=True -> GameplaySAC (ctrl_action_dim={ctrl_dim})")

  tag = TAG + ("_smoke" if args.smoke else "")
  outdir = os.path.join(args.out, tag)
  os.makedirs(outdir, exist_ok=True)

  # --- two-player tensor env (GPU-resident training env) ---
  env = make_tensor(TASK, args.num_envs, args.device,
                    adversary=True, end_criterion=args.end_criterion)
  eff_ec = args.end_criterion if args.end_criterion is not None else s.end_criterion
  print(f"[end-criterion] {TASK} -> {eff_ec}")

  # --- leaderboard eval env: numpy VecEnv (adversary on), obs-normalized ---
  # The leaderboard is ON by default for 2-agent training (user preference); a
  # real eval env makes it actually evaluate (else `_lb_eval_env is None` and it
  # never scores). See _NormEvalVecEnv for why numpy + normalization.
  lb_eval_n = 8 if args.smoke else 64
  n_eval_episodes = 2 if args.smoke else 10
  leaderboard_freq = 5_000 if args.smoke else 100_000
  lb_eval = _NormEvalVecEnv(
    make_numpy(TASK, lb_eval_n, args.device, adversary=True,
               end_criterion=args.end_criterion))

  learning_starts = 5 * args.num_envs
  # Build the gamma-anneal schedule from the CLI (default = reference discrete jumps).
  from safety_sb3 import GeometricGammaAnneal, StepGammaAnneal
  if args.gamma_schedule == "step":
    gamma_anneal = StepGammaAnneal(init=args.gamma_init, end=args.gamma_end,
                                   ratio=args.gamma_ratio,
                                   period_frac=args.gamma_period_frac)
  elif args.gamma_schedule == "geometric":
    gamma_anneal = GeometricGammaAnneal(init=args.gamma_init, end=args.gamma_end,
                                        anneal_frac=args.gamma_anneal_frac)
  else:
    gamma_anneal = False
  # entropy auto-tuned starting ~0.1 (reference alpha=0.1, learn_alpha=true).
  # net_arch matches the reference: actor 256x3 (actor_0/actor_1), critic 128x3
  # (critic_0). Reference activation is Sin (custom to safe_adaptation_dev); SB3
  # has no Sin, so we keep the SAC-standard ReLU (documented deviation).
  model = GameplaySAC(
    "MlpPolicy", env,
    ctrl_action_dim=ctrl_dim,
    normalize_obs=True,
    gamma=args.gamma_init,         # anneals per --gamma-schedule (default step jumps)
    gamma_anneal=gamma_anneal,
    min_alpha=args.min_alpha, max_alpha=args.max_alpha,
    critic_learning_rate=args.critic_lr, dstb_learning_rate=args.dstb_lr,
    ent_coef_lr=args.ent_coef_lr, dstb_ent_coef_lr=args.dstb_ent_coef_lr,
    lr_schedule=args.lr_schedule, lr_period=args.lr_period,
    lr_decay=args.lr_decay, lr_end=args.lr_end,
    learning_rate=args.lr,         # shared / ctrl-actor lr (reference 1e-4)
    tau=0.01,                      # reference tau
    target_update_interval=2,      # reference update_target_period 2
    ent_coef="auto_0.1",           # reference alpha 0.1, learn_alpha true
    buffer_size=args.buffer_size,
    batch_size=args.batch_size,
    train_freq=1,
    gradient_steps=args.gradient_steps,
    learning_starts=learning_starts,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256], qf=[128, 128, 128])),
    use_leaderboard=True,
    leaderboard_dir=os.path.join(outdir, "leaderboard"),
    leaderboard_eval_env=lb_eval,
    n_eval_episodes=n_eval_episodes,
    leaderboard_freq=leaderboard_freq,
    seed=args.seed,
    verbose=1,
    device=args.device,
    tensorboard_log=outdir)
  # Inject the training normalizer's stats into the leaderboard eval env now that
  # the TensorVecNormalize exists (see _NormEvalVecEnv).
  norm_fn = getattr(model.env, "normalize_obs_np", None)
  if norm_fn is not None:
    lb_eval.set_normalizer(norm_fn)
    print("[leaderboard] eval env obs-normalized with training running stats")
  print(f"[recipe] net pi=256x3 qf=128x3 (ReLU) lr={args.lr} tau=0.01 "
        f"tgt_upd=2 ent=auto_0.1 gamma={args.gamma_init}->{args.gamma_end} "
        f"({args.gamma_schedule}) min_alpha={args.min_alpha} max_alpha={args.max_alpha} "
        f"batch={args.batch_size} grad_steps={args.gradient_steps} "
        f"learn_starts={learning_starts} buffer={args.buffer_size} "
        f"lb_eval={lb_eval_n}x{n_eval_episodes}")

  # --- callbacks ---
  cbs = [
    CheckpointCallback(save_freq=max(1, 25_000_000 // args.num_envs),
                       save_path=os.path.join(outdir, "checkpoints"),
                       name_prefix="model"),
    TensorNormSaveCallback(os.path.join(outdir, "checkpoints")),
    # adversary force curriculum (same as train.py's --adversary path)
    ForceRampCallback(args.force_max, int(args.force_ramp_frac * args.steps)),
    PerEnvForceScaleCallback(lo=args.force_floor, init=args.force_init),
  ]
  print(f"[adversary] force ramp 8->{args.force_max}N over "
        f"{args.force_ramp_frac:.0%}; per-env scale floor={args.force_floor} "
        f"init={args.force_init}")

  # --- safe/success-rate eval (logged to wandb): a SEPARATE tensor eval env
  # whose obs-normalizer stats are synced from the training env at each eval so
  # the metric sees the same normalization the policy trains on. go2_stabilize
  # is a reach-avoid task (has a target l), so reach_avoid=True. ---
  if args.eval_freq > 0:
    from safety_sb3 import SafeSuccessRateEvalCallback
    from safety_sb3.tensor_env import TensorVecNormalize
    eval_n = 8 if args.smoke else args.eval_envs
    eval_metric_env = TensorVecNormalize(
      make_tensor(TASK, eval_n, args.device, adversary=True,
                  end_criterion=args.end_criterion))

    class _SyncedSafeSuccessEval(SafeSuccessRateEvalCallback):
      """Push the training normalizer stats into the eval env right before an
      eval fires (the eval env is frozen during eval, so it won't self-update)."""
      def _on_step(self):
        if self.eval_freq > 0 and self.num_timesteps >= self._next_eval:
          tenv, ev = self.model.env, self.eval_env
          if hasattr(tenv, "obs_mean") and hasattr(ev, "obs_mean"):
            ev.obs_mean = tenv.obs_mean.clone()
            ev.obs_var = tenv.obs_var.clone()
            ev.count = (tenv.count.clone() if th.is_tensor(tenv.count)
                        else tenv.count)
        return super()._on_step()

    ef = 20_000 if args.smoke else args.eval_freq
    cbs.append(_SyncedSafeSuccessEval(
      eval_metric_env, n_rollouts=(16 if args.smoke else args.eval_rollouts),
      eval_freq=ef, reach_avoid=True, verbose=1))
    print(f"[eval] safe/success-rate every {ef} steps over "
          f"{16 if args.smoke else args.eval_rollouts} rollouts ({eval_n} envs)")

  if not args.no_wandb:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    wandb.init(project=args.wandb_project, name=tag, config=vars(args),
               sync_tensorboard=True, save_code=False, reinit=True)
    cbs.append(WandbCallback(verbose=0))
    _vtask = TASK + "_video"
    _vtask = _vtask if _vtask in list_tasks() else TASK
    cbs.append(VideoWandbCallback(
      lambda: make_tensor(_vtask, 8, args.device, adversary=False,
                          render_mode="rgb_array",
                          end_criterion=args.end_criterion),
      interval=args.video_interval))

  model.learn(total_timesteps=args.steps, callback=CallbackList(cbs))
  model.save(os.path.join(outdir, "final_model.zip"))
  if hasattr(model.env, "save"):
    model.env.save(os.path.join(outdir, "tensornormalize.pt"))
  print(f"[done] {outdir}")

  # --- smoke self-checks: prove the key contracts the feature test relies on ---
  if args.smoke:
    sample = model.replay_buffer.sample(8)
    adim = int(sample.actions.shape[1])
    exp = ctrl_dim + s.dstb_dim
    print(f"[smoke] replay action dim = {adim} (expect ctrl {ctrl_dim} + dstb "
          f"{s.dstb_dim} = {exp}) -> {'OK' if adim == exp else 'MISMATCH'}")
    print(f"[smoke] gamma (final, post-anneal) = {model.gamma:.6f} "
          f"(started 0.99; should have climbed)")
    lb = getattr(model, "_leaderboard", None)
    if lb is not None:
      lbdir = os.path.join(outdir, "leaderboard")
      nfiles = len(os.listdir(lbdir)) if os.path.isdir(lbdir) else 0
      print(f"[smoke] leaderboard: {len(lb.ctrl_steps)} ctrl / "
            f"{len(lb.dstb_steps)} dstb archived; {nfiles} files in {lbdir}")
    ok = (os.path.exists(os.path.join(outdir, "final_model.zip"))
          and os.path.exists(os.path.join(outdir, "tensornormalize.pt")))
    print(f"[smoke] saved final_model.zip + tensornormalize.pt -> "
          f"{'OK' if ok else 'MISSING'}")


if __name__ == "__main__":
  main()
