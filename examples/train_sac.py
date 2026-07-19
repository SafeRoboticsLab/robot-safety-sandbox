"""Unified SAC trainer: ALL FOUR SAC-family safety learners (the SAC analog of
`train.py`, which is PPO-only).

The 2x2 SAC taxonomy (problem x players), resolved from the task's margins
(avoid vs reach-avoid) x the run's player count (--adversary):

                 1-player            2-player (--adversary)
    avoid        SafetySAC           IsaacsSAC
    reach-avoid  ReachAvoidSAC       GameplaySAC

  # 1-player reach-avoid (ReachAvoidSAC)
  python examples/train_sac.py --task go2_stabilize --steps 100000000 --seed 0
  # 2-player reach-avoid (GameplaySAC) -- the E042 config
  python examples/train_sac.py --task go2_stabilize --adversary --num-envs 1024
  # 1-player avoid (SafetySAC) / 2-player avoid (IsaacsSAC)
  python examples/train_sac.py --task digit_stabilize_avoid [--adversary]

This generalizes `train_gameplay_sac.py` (GameplaySAC-only). The crux vs that
script is VARIANT-CONDITIONAL construction: the two-player classes take
`ctrl_action_dim`, per-agent LRs, and the numpy leaderboard eval env + adversary
force curriculum; the single-player classes take none of those (they'd TypeError)
and there is no adversary. Everything else (SAC hypers, gamma anneal, alpha
floor/ceil, the SafeSuccessRateEvalCallback + train->eval normalizer sync) is
common to all four and copied verbatim from `train_gameplay_sac.py`.

SAC hypers mirror `safe_adaptation_dev/config/go2_pybullet_isaacs_br.yaml`
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

from robot_safety_sandbox import (  # noqa: E402
  algo_name, list_tasks, make_numpy, make_tensor, spec)
from robot_safety_sandbox.callbacks import (  # noqa: E402
  ForceRampCallback,
  PerEnvForceScaleCallback,
  TensorNormSaveCallback,
  VideoWandbCallback,
)

# PPO learner NAME (from algo_name, the shared 2x2 resolver) -> SAC class NAME.
# Same problem x player cell, off-policy analog. We resolve to a name first and
# import lazily so a missing SAC class fails with a clear message, not ImportError
# at module load (mirrors train.py's fail-closed ALGOS gate).
PPO_TO_SAC = {
  "SafetyPPO": "SafetySAC",
  "ReachAvoidPPO": "ReachAvoidSAC",
  "IsaacsPPO": "IsaacsSAC",
  "GameplayPPO": "GameplaySAC",
}
REACH_AVOID_ALGOS = {"ReachAvoidPPO", "GameplayPPO"}  # -> eval reach_avoid flag


class _NormEvalVecEnv:
  """Wrap the numpy leaderboard-eval VecEnv so its obs are normalized with the
  training TensorVecNormalize running stats before reaching the actors.
  (Two-player only -- the leaderboard is a 2-agent construct.)

  WHY a numpy env (not make_tensor): the leaderboard's `_eval_pair` routes any
  env exposing `num_envs` + `step_async` to `_eval_pair_vec`, which drives it
  with the CLASSIC numpy VecEnv API (`step_async` / `step_wait` returning
  `(obs, g, dones, infos)` with `info["l_x"]`). A `TensorVecEnv` deliberately
  RAISES on `step_async`/`step_wait` (tensor path only), so a tensor eval env
  would crash the first leaderboard step. `make_numpy(..., adversary=True)`
  gives the exact surface it needs.

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
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--task", required=True, help=f"one of {list_tasks()}")
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--steps", type=int, default=100_000_000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--adversary", action="store_true",
                 help="run the TWO-PLAYER game (IsaacsSAC for avoid tasks, "
                      "GameplaySAC for reach-avoid). Off = single-player "
                      "(SafetySAC / ReachAvoidSAC).")
  p.add_argument("--out", default=os.path.join(_ZOO, "runs"),
                 help="output root; ALWAYS keep runs under runs/ (git-ignored) — "
                      "never invent runs_<suffix> siblings, they escape .gitignore")
  p.add_argument("--wandb-project", default="robot_safety_sandbox")
  p.add_argument("--no-wandb", action="store_true")
  p.add_argument("--end-criterion", choices=["failure", "reach-avoid", "timeout"],
                 default=None, help="WHEN the episode ends from (g,l); default = "
                 "the task's TaskSpec value.")
  p.add_argument("--smoke", action="store_true",
                 help="tiny-budget verification: shrink learning_starts / eval "
                      "cadence / leaderboard sizes so a short run exercises every "
                      "code path (compose, gamma anneal, eval, leaderboard).")
  # --- net arch (mirror train.py's --net; qf kept on its reference default) ---
  p.add_argument("--net", default="256,256,256",
                 help="comma-separated hidden dims for the pi (actor) net "
                      "(reference 256x3)")
  p.add_argument("--qf-net", default="128,128,128",
                 help="comma-separated hidden dims for the qf (critic) net "
                      "(reference 128x3)")
  # --- core SAC knobs (common to all four) ---
  p.add_argument("--lr", type=float, default=1e-4,
                 help="shared / ctrl-actor learning rate (reference 1e-4)")
  p.add_argument("--tau", type=float, default=0.01, help="target soft-update rate")
  p.add_argument("--target-update-interval", type=int, default=2)
  p.add_argument("--ent-coef", default="auto_0.1",
                 help="SAC entropy temperature (reference alpha 0.1, learned)")
  p.add_argument("--buffer-size", type=int, default=1_000_000)
  p.add_argument("--batch-size", type=int, default=4096)
  p.add_argument("--gradient-steps", type=int, default=4,
                 help="SGD updates per collect (tensor path: NEVER -1, which "
                      "means num_envs updates/step). Small int, 2-4.")
  p.add_argument("--learning-starts", type=int, default=None,
                 help="warmup transitions before learning (default 5*num_envs)")
  # --- gamma annealing (reference-faithful discrete jumps by default) ---
  p.add_argument("--gamma-schedule", choices=["step", "geometric", "off"],
                 default="step", help="discount anneal shape: 'step' = REFERENCE "
                 "discrete jumps (0.99->0.999@20%%->0.9999@40%%, hold; resets "
                 "alpha on each jump) [DEFAULT]; 'geometric' = smooth to end by "
                 "--gamma-anneal-frac; 'off' = constant gamma.")
  p.add_argument("--gamma-init", type=float, default=0.99, help="starting gamma")
  p.add_argument("--gamma-end", type=float, default=0.9999,
                 help="final gamma held after annealing")
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
  # --- reach-avoid terminal valuation (reach-avoid learners only) ---
  p.add_argument("--terminal-type", choices=["all", "g"], default="all",
                 help="reach-avoid learners only: value a terminal step as "
                      "min(l,g) ('all', default) or g ('g'). Ignored on avoid "
                      "tasks (SafetySAC/IsaacsSAC have no reach margin l).")
  # --- safe/success-rate evaluation (logged to wandb; all four) ---
  p.add_argument("--eval-rollouts", type=int, default=100,
                 help="episodes per safe/success-rate eval (reference ~100)")
  p.add_argument("--eval-freq", type=int, default=2_000_000,
                 help="env-steps between safe/success-rate evals (0 = off)")
  p.add_argument("--eval-envs", type=int, default=128,
                 help="parallel envs in the (separate) eval env")
  p.add_argument("--video-interval", type=int, default=5_000_000)
  # --- adversary force curriculum (two-player only) ---
  p.add_argument("--force-max", type=float, default=50.0)
  p.add_argument("--force-ramp-frac", type=float, default=0.55)
  p.add_argument("--force-floor", type=float, default=0.3)
  p.add_argument("--force-init", type=float, default=0.5)
  # --- per-agent learning rates (two-player only; None -> fall back to --lr) ---
  p.add_argument("--critic-lr", type=float, default=None)
  p.add_argument("--dstb-lr", type=float, default=None, help="dstb ACTOR lr")
  p.add_argument("--ent-coef-lr", type=float, default=None, help="ctrl entropy(alpha) lr")
  p.add_argument("--dstb-ent-coef-lr", type=float, default=None, help="dstb entropy(alpha) lr")
  p.add_argument("--lr-schedule", action="store_true",
                 help="enable StepLR decay of the ctrl/dstb/critic lrs (2P)")
  p.add_argument("--lr-period", type=int, default=1_000_000)
  p.add_argument("--lr-decay", type=float, default=0.1)
  p.add_argument("--lr-end", type=float, default=0.0)
  # --- leaderboard knobs (two-player only) ---
  p.add_argument("--leaderboard-eval-envs", type=int, default=64)
  p.add_argument("--leaderboard-episodes", type=int, default=10)
  p.add_argument("--leaderboard-freq", type=int, default=100_000)
  args = p.parse_args()

  # --- resolve task + learner (2x2: problem from margins, players from --adversary) ---
  s = spec(args.task)
  if s.kind != "safety":
    raise SystemExit(
      f"'{args.task}' is a {s.kind} task (dense reward, no margins) — train it "
      f"with train_nominal.py; this trainer is for the safety layer.")
  algo = algo_name(args.task, adversary=args.adversary)  # PPO-family name
  if algo not in PPO_TO_SAC:
    raise SystemExit(f"'{args.task}' resolves to '{algo}', which has no SAC analog.")
  sac_name = PPO_TO_SAC[algo]
  try:
    import safety_sb3 as _sb3
    Algo = getattr(_sb3, sac_name)
  except (ImportError, AttributeError):
    raise SystemExit(
      f"'{args.task}'{' +--adversary' if args.adversary else ''} needs the "
      f"'{sac_name}' learner, which this safety_sb3 does not export. The SAC "
      f"family (all four cells) requires safety_sb3 >= v0.2.0.")
  reach_avoid = algo in REACH_AVOID_ALGOS
  two_player = args.adversary
  print(f"[algo] {args.task} adversary={two_player} -> {sac_name} "
        f"(reach_avoid={reach_avoid})")

  tag = f"{args.task}_{sac_name.lower()}" + ("_smoke" if args.smoke else "")
  outdir = os.path.join(args.out, tag)
  os.makedirs(outdir, exist_ok=True)

  # --- training env (GPU-resident tensor path; adversary iff two-player) ---
  env = make_tensor(args.task, args.num_envs, args.device,
                    adversary=two_player, end_criterion=args.end_criterion)
  eff_ec = args.end_criterion if args.end_criterion is not None else s.end_criterion
  print(f"[end-criterion] {args.task} -> {eff_ec}"
        f"{' (override)' if args.end_criterion is not None else ' (task default)'}")

  # --- gamma-anneal schedule from the CLI (default = reference discrete jumps) ---
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

  learning_starts = (args.learning_starts if args.learning_starts is not None
                     else 5 * args.num_envs)
  pi_net = [int(x) for x in args.net.split(",") if x.strip()]
  qf_net = [int(x) for x in args.qf_net.split(",") if x.strip()]

  # kwargs COMMON to all four (on the SafetySAC base + SAC).
  akw = dict(
    normalize_obs=True,
    gamma=args.gamma_init,          # anneals per --gamma-schedule
    gamma_anneal=gamma_anneal,
    min_alpha=args.min_alpha, max_alpha=args.max_alpha,
    learning_rate=args.lr,
    tau=args.tau,
    target_update_interval=args.target_update_interval,
    ent_coef=args.ent_coef,
    buffer_size=args.buffer_size,
    batch_size=args.batch_size,
    train_freq=1,
    gradient_steps=args.gradient_steps,
    learning_starts=learning_starts,
    policy_kwargs=dict(net_arch=dict(pi=pi_net, qf=qf_net)),
    seed=args.seed,
    verbose=1,
    device=args.device,
    tensorboard_log=outdir,
  )
  # terminal_type is a REACH-AVOID knob (min(l,g) valuation of a terminal step);
  # the avoid classes have no l, so pass it only for the reach-avoid cells.
  if reach_avoid:
    akw["terminal_type"] = args.terminal_type
    print(f"[terminal-type] {sac_name} -> {args.terminal_type}")

  # --- VARIANT-CONDITIONAL: two-player-only construction ---
  lb_eval = None
  if two_player:
    lb_eval_n = 8 if args.smoke else args.leaderboard_eval_envs
    n_lb_episodes = 2 if args.smoke else args.leaderboard_episodes
    leaderboard_freq = 5_000 if args.smoke else args.leaderboard_freq
    lb_eval = _NormEvalVecEnv(
      make_numpy(args.task, lb_eval_n, args.device, adversary=True,
                 end_criterion=args.end_criterion))
    akw.update(dict(
      ctrl_action_dim=s.ctrl_dim,   # env action = [ctrl, dstb]
      critic_learning_rate=args.critic_lr, dstb_learning_rate=args.dstb_lr,
      ent_coef_lr=args.ent_coef_lr, dstb_ent_coef_lr=args.dstb_ent_coef_lr,
      lr_schedule=args.lr_schedule, lr_period=args.lr_period,
      lr_decay=args.lr_decay, lr_end=args.lr_end,
      use_leaderboard=True,
      leaderboard_dir=os.path.join(outdir, "leaderboard"),
      leaderboard_eval_env=lb_eval,
      n_eval_episodes=n_lb_episodes,
      leaderboard_freq=leaderboard_freq,
    ))
    print(f"[two-player] ctrl_action_dim={s.ctrl_dim} dstb_dim={s.dstb_dim}; "
          f"leaderboard {lb_eval_n}x{n_lb_episodes} every {leaderboard_freq}")

  model = Algo("MlpPolicy", env, **akw)

  # Two-player: inject the training normalizer's stats into the leaderboard eval
  # env now that the TensorVecNormalize exists (see _NormEvalVecEnv).
  if two_player and lb_eval is not None:
    norm_fn = getattr(model.env, "normalize_obs_np", None)
    if norm_fn is not None:
      lb_eval.set_normalizer(norm_fn)
      print("[leaderboard] eval env obs-normalized with training running stats")
  print(f"[recipe] net pi={pi_net} qf={qf_net} (ReLU) lr={args.lr} tau={args.tau} "
        f"tgt_upd={args.target_update_interval} ent={args.ent_coef} "
        f"gamma={args.gamma_init}->{args.gamma_end} ({args.gamma_schedule}) "
        f"min_alpha={args.min_alpha} max_alpha={args.max_alpha} "
        f"batch={args.batch_size} grad_steps={args.gradient_steps} "
        f"learn_starts={learning_starts} buffer={args.buffer_size}")

  # --- callbacks ---
  cbs = [
    CheckpointCallback(save_freq=max(1, 25_000_000 // args.num_envs),
                       save_path=os.path.join(outdir, "checkpoints"),
                       name_prefix="model"),
    TensorNormSaveCallback(os.path.join(outdir, "checkpoints")),
  ]
  # VARIANT-CONDITIONAL: adversary force curriculum (two-player only).
  if two_player:
    cbs.append(ForceRampCallback(args.force_max,
                                 int(args.force_ramp_frac * args.steps)))
    cbs.append(PerEnvForceScaleCallback(lo=args.force_floor, init=args.force_init))
    print(f"[adversary] force ramp 8->{args.force_max}N over "
          f"{args.force_ramp_frac:.0%}; per-env scale floor={args.force_floor} "
          f"init={args.force_init}")

  # --- safe/success-rate eval (all four): a SEPARATE tensor eval env whose
  # obs-normalizer stats are synced from the training env at each eval so the
  # metric sees the same normalization the policy trains on. reach_avoid flag
  # comes from the resolved learner; eval env carries the adversary iff 2P. ---
  if args.eval_freq > 0:
    from safety_sb3 import SafeSuccessRateEvalCallback
    from safety_sb3.tensor_env import TensorVecNormalize
    eval_n = 8 if args.smoke else args.eval_envs
    eval_metric_env = TensorVecNormalize(
      make_tensor(args.task, eval_n, args.device, adversary=two_player,
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
    n_roll = 16 if args.smoke else args.eval_rollouts
    cbs.append(_SyncedSafeSuccessEval(
      eval_metric_env, n_rollouts=n_roll, eval_freq=ef,
      reach_avoid=reach_avoid, verbose=1))
    print(f"[eval] safe/success-rate every {ef} steps over {n_roll} rollouts "
          f"({eval_n} envs, reach_avoid={reach_avoid})")

  if not args.no_wandb:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    wandb.init(project=args.wandb_project, name=tag, config=vars(args),
               sync_tensorboard=True, save_code=False, reinit=True)
    cbs.append(WandbCallback(verbose=0))
    _vtask = args.task + "_video"
    _vtask = _vtask if _vtask in list_tasks() else args.task
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

  # --- smoke self-checks: prove the key contracts the variant relies on ---
  if args.smoke:
    sample = model.replay_buffer.sample(8)
    adim = int(sample.actions.shape[1])
    exp = (s.ctrl_dim + s.dstb_dim) if two_player else s.ctrl_dim
    kind = f"ctrl {s.ctrl_dim} + dstb {s.dstb_dim}" if two_player else f"ctrl {s.ctrl_dim}"
    print(f"[smoke] replay action dim = {adim} (expect {kind} = {exp}) -> "
          f"{'OK' if adim == exp else 'MISMATCH'}")
    print(f"[smoke] gamma (final, post-anneal) = {model.gamma:.6f} "
          f"(started {args.gamma_init}; should have climbed if schedule on)")
    if two_player:
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
