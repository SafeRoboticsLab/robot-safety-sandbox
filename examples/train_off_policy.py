"""Unified SAC trainer: ALL FOUR SAC-family safety learners — the off-policy
half of the MAP (Mode. Algorithm. Players.), A = SAC.

The task's mode supplies M, `--adversary` supplies P; the four cells are:

                 1-player            2-player (--adversary)
    avoid        SafetySAC1P         SafetySAC2P
    reach-avoid  ReachAvoidSAC1P     ReachAvoidSAC2P

  # 1-player reach-avoid (ReachAvoidSAC1P)
  python examples/train_off_policy.py --task go2_stabilize --steps 100000000 --seed 0
  # 2-player reach-avoid (ReachAvoidSAC2P) -- the reference-faithful stabilize config
  python examples/train_off_policy.py --task go2_stabilize --adversary --num-envs 1024
  # 1-player avoid (SafetySAC1P) / 2-player avoid (SafetySAC2P)
  python examples/train_off_policy.py --task digit_stabilize_avoid [--adversary]

The 2P SAC game is NOT the 2P PPO game with a different optimizer: *SAC2P is the
minimax game on ONE shared joint-action critic Q(s, [a_ctrl, a_dstb]), while
*PPO2P is an alternating best-response approximation with two independent V(s)
nets and a phase machine. Picking `--adversary` here gets you the former.

The crux of this script is VARIANT-CONDITIONAL construction: the two-player
classes take `ctrl_action_dim`, per-agent LRs, and the leaderboard eval env +
adversary force curriculum; the single-player classes take none of those (they'd
TypeError) and there is no adversary. Everything else (SAC hypers, gamma anneal,
alpha floor/ceil, the SafeSuccessRateEvalCallback + train->eval normalizer sync)
is common to all four.

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

from _run_config import dump_config, merge_config  # noqa: E402  (examples/ sibling)
from robot_safety_sandbox import (  # noqa: E402
  CUMULATIVE, algo_name, list_tasks, make_tensor, spec)
from robot_safety_sandbox.callbacks import (  # noqa: E402
  ForceRampCallback,
  PerEnvForceScaleCallback,
  TensorNormSaveCallback,
  VideoWandbCallback,
)

def _wrap_safety_filter(env, args):
  """Wrap the training env in a safety filter, per the ``safety_filter:`` config.

  Presence of ``safety_policy`` (the fallback twin's checkpoint) selects the
  FILTERED (treatment) arm: every proposed action is certified by the filter and
  the EXECUTED action is what enters the replay buffer. Absence returns the bare
  env -- the unfiltered (control) arm -- which already counts failures on its
  own, so both arms share one accounting path. See docs/safety-filter-training.md.
  """
  sf = args.safety_filter or {}
  policy = sf.get("safety_policy")
  if not policy:
    return env
  from robot_safety_sandbox.eval.filters import SwitchCfg, build_filter
  from robot_safety_sandbox.eval.policies import load_twin, safety_modules
  from robot_safety_sandbox.filtered_env import FilteredTensorEnv

  model, norm = load_twin(policy, args.device)
  mods = safety_modules(model, env.num_envs, args.device)
  mods["norm"] = norm
  bundle = build_filter(
    sf.get("filter", "critic"), mods, _FilterEnvView(env, args.task),
    switch=SwitchCfg(eps=float(sf.get("eps", 0.0)),
                     smoothing=bool(sf.get("smoothing", False))))
  return FilteredTensorEnv(env, bundle.filt, norm=norm)


class _FilterEnvView:
  """The handful of attributes ``build_filter`` reads off an eval env.

  The value/critic recipes need only num_envs/device/task; the rollout ones
  would also need cfg_builder, and are refused above rather than half-supported
  (a shadow sim inside a training loop is a separate design question).
  """

  def __init__(self, env, task: str):
    self.num_envs, self.device, self.task = env.num_envs, str(env.mj.device), task


# NOTE: the leaderboard eval env is a RAW TensorVecEnv (the 2P learner dispatches
# to `_eval_pair_tensor`, on-device, normalizing obs via the live training
# normalizer). Profiling showed the league eval (not the numpy transfer) was the
# throughput bottleneck, fixed by fewer eval episodes + a higher leaderboard_freq
# (see the --leaderboard-* defaults).


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--config", default=None,
                 help="YAML recipe of args (keys = flag dest names). Sets defaults; "
                      "explicit CLI flags override it. See configs/.")
  p.add_argument("--env-override", action="append", metavar="KEY=VAL", default=None,
                 help="override an env/task cfg_builder param (repeatable), e.g. "
                      "--env-override gate_close_rate=0.003. Also settable as a "
                      "config `env_overrides:` dict. Forwarded to make_tensor.")
  p.add_argument("--task", required=True, help=f"one of {list_tasks()}")
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--steps", type=int, default=100_000_000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--adversary", action="store_true",
                 help="run the TWO-PLAYER minimax game on a shared joint-action "
                      "critic (SafetySAC2P for avoid tasks, ReachAvoidSAC2P for "
                      "reach-avoid). Off = single-player (*SAC1P).")
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
  # --- train the task policy INSIDE a safety filter -------------------------
  # This is config-driven, not argparse: a `safety_filter:` block in the YAML
  # (schema in examples/_run_config.py, worked example in
  # configs/go2_walker_filtered.yaml, background in docs/safety-filter-training.md)
  # selects the FILTERED (treatment) arm when it carries a `safety_policy`; its
  # absence is the unfiltered (control) arm. Both arms count failures via the
  # base env's always-on safety/* counters, one shared accounting path -- so if
  # the arms counted differently the headline plot would be measuring the
  # instrumentation. `--safety-filter KEY=VAL` (repeatable) overrides the block
  # one key at a time, mirroring `--env-override`.
  p.add_argument("--run-suffix", default=None, metavar="STR",
                 help="appended to the run tag, which names BOTH the output "
                      "directory and the wandb run. Use it for every cell of a "
                      "sweep: without it, cells sharing task+learner+arm "
                      "overwrite each other's run dir and appear in wandb under "
                      "one indistinguishable name.")
  p.add_argument("--safety-filter", dest="safety_filter_override",
                 action="append", metavar="KEY=VAL", default=None,
                 help="override one key of the `safety_filter:` config block "
                      "(repeatable), e.g. --safety-filter eps=0.1 "
                      "--safety-filter safety_policy=runs/twin/final_model.zip. "
                      "Keys: safety_policy (twin checkpoint; present -> filtered "
                      "arm, absent -> control), filter {value,critic,qcbf}, eps, "
                      "smoothing. The rollout/gameplay monitors are deliberately "
                      "not offered: a shadow sim inside the training loop is a "
                      "separate design question.")
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
  p.add_argument("--target-entropy", default=None,
                 help="SAC's entropy TARGET; 'auto' (SB3 default) means "
                      "-dim(A), i.e. -12 for the go2's 12 joints. That asks a "
                      "locomotion policy to be near-deterministic, so alpha is "
                      "driven down and exploration dies early (alpha "
                      "collapses to the floor in every cell and the policy "
                      "settles into a standstill). A LESS negative value "
                      "(e.g. -4, -6) holds exploration open.")
  p.add_argument("--min-alpha", type=float, default=1e-3,
                 help="floor on the learned entropy temperature (reference "
                      "1e-3). A clamp, not an objective — it fights the "
                      "entropy loss rather than changing what it asks for; "
                      "prefer --target-entropy to keep exploration alive. "
                      "0 (or negative) disables the floor.")
  p.add_argument("--max-alpha", type=float, default=None,
                 help="optional ceiling on the entropy temperature (default none)")
  # --- reach-avoid terminal valuation (reach-avoid learners only) ---
  p.add_argument("--terminal-type", choices=["all", "g"], default="all",
                 help="reach-avoid learners only: value a terminal step as "
                      "min(l,g) ('all', default) or g ('g'). Ignored on avoid "
                      "tasks (the Safety* cells have no reach margin l).")
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
  # --- leaderboard knobs (two-player only). Defaults tuned for THROUGHPUT: the
  # league eval cost scales with (episodes x pairings x episode_len x freq), and
  # profiling showed it was ~90% of wall-clock at the old 100k/10 settings (one
  # _leaderboard_step ~100s). The league is a relative ranking, so a few episodes
  # every ~2M steps suffices; the eval env is a RAW tensor env (on-device). ---
  p.add_argument("--leaderboard-eval-envs", type=int, default=64)
  p.add_argument("--leaderboard-episodes", type=int, default=3,
                 help="eval batches per pairing (was 10; league is a ranking)")
  p.add_argument("--leaderboard-freq", type=int, default=2_000_000,
                 help="env-steps between league evals (was 100k = every ~98 "
                      "vec-steps at 1024 envs -- absurdly frequent)")
  args = merge_config(p)   # parse args; an optional --config sets defaults, CLI overrides

  # --- resolve task + learner (2x2: problem from margins, players from --adversary) ---
  s = spec(args.task)
  cumulative = s.mode == CUMULATIVE
  if cumulative and args.adversary:
    raise SystemExit(
      f"'{args.task}' is mode={CUMULATIVE!r}: there is no two-player "
      "cumulative game (see registry.algo_name), so --adversary has no "
      "meaning here.")
  # The MAP: M from the task's mode, A = SAC (this is the off-policy family),
  # P from --adversary. Resolve the NAME first, import lazily, so a missing cell
  # fails with a clear message rather than an ImportError at module load.
  # mode="cumulative" resolves to the bare "SAC" under the MAP (no M prefix, no
  # P suffix -- there is no two-player cumulative game). On the TENSOR path that
  # is safety_sb3's CumulativeSAC1P rather than stock SB3 SAC: same backup
  # (r + gamma*V'), but stock SB3 has no GPU-resident collect loop, and no
  # executed-action readback -- which filtered training depends on.
  sac_name = ("CumulativeSAC1P" if cumulative else
              algo_name(args.task, adversary=args.adversary,
                        family="off_policy"))
  try:
    import safety_sb3 as _sb3
    Algo = getattr(_sb3, sac_name)
  except (ImportError, AttributeError):
    raise SystemExit(
      f"'{args.task}'{' +--adversary' if args.adversary else ''} needs the "
      f"'{sac_name}' learner, which this safety_sb3 does not export.")
  reach_avoid = sac_name.startswith("ReachAvoid")
  two_player = args.adversary
  print(f"[algo] {args.task} adversary={two_player} -> {sac_name} "
        f"(reach_avoid={reach_avoid})")

  # The filtered (treatment) and unfiltered (control) arms share task AND
  # learner, so the arm has to be in the tag or they would overwrite each
  # other's run directory and wandb run. The filtered arm gets a "_filtered"
  # marker; the control arm is a plain run (distinguish A/B cells with
  # --run-suffix).
  filtered_arm = bool((args.safety_filter or {}).get("safety_policy"))
  arm = "_filtered" if filtered_arm else ""
  # A hyperparameter SWEEP runs the same task+learner+arm many times, so without
  # --run-suffix every cell lands on the same tag: same run directory (they
  # clobber) and, worse, the same wandb run NAME, since both trainers hardcode
  # wandb.init(name=tag) and ignore WANDB_NAME. Distinguishing cells by config
  # alone makes a sweep unreadable in the UI.
  tag = (f"{args.task}_{sac_name.lower()}{arm}"
         + (f"_{args.run_suffix}" if args.run_suffix else "")
         + ("_smoke" if args.smoke else ""))
  outdir = os.path.join(args.out, tag)
  os.makedirs(outdir, exist_ok=True)
  dump_config(outdir, args)   # reproducible: re-run with --config <outdir>/config.yaml

  # --- training env (GPU-resident tensor path; adversary iff two-player) ---
  env = make_tensor(args.task, args.num_envs, args.device,
                    adversary=two_player, end_criterion=args.end_criterion,
                    cfg_overrides=args.env_overrides)
  env = _wrap_safety_filter(env, args)
  eff_ec = args.end_criterion if args.end_criterion is not None else s.end_criterion
  print(f"[end-criterion] {args.task} -> {eff_ec}"
        f"{' (override)' if args.end_criterion is not None else ' (task default)'}")

  # --- gamma-anneal schedule from the CLI (default = reference discrete jumps) ---
  from safety_sb3 import GeometricGammaAnneal, StepGammaAnneal
  if cumulative:
    # The gamma anneal is a SAFETY-backup device: it walks the discount toward 1
    # so the avoid/reach-avoid value approaches the infinite-horizon one. A
    # cumulative return has no such limit to approach (gamma -> 1 diverges), and
    # the anneal's alpha resets would wreck a reward-maximizing run. Force it
    # off regardless of the CLI default.
    gamma_anneal = False
    if args.gamma_schedule != "off":
      print(f"[gamma] --gamma-schedule {args.gamma_schedule} IGNORED on a "
            f"mode={CUMULATIVE!r} task; gamma held at {args.gamma_init}")
  elif args.gamma_schedule == "step":
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

  # kwargs COMMON to all four cells (on the shared safety-SAC base + SAC).
  akw = dict(
    normalize_obs=True,
    gamma=args.gamma_init,          # anneals per --gamma-schedule
    gamma_anneal=gamma_anneal,
    min_alpha=args.min_alpha, max_alpha=args.max_alpha,
    learning_rate=args.lr,
    tau=args.tau,
    target_update_interval=args.target_update_interval,
    ent_coef=args.ent_coef,
    **({} if args.target_entropy is None else
       {"target_entropy": (args.target_entropy if args.target_entropy == "auto"
                           else float(args.target_entropy))}),
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
    # RAW tensor eval env -> the 2P learner's _eval_pair_tensor (on-device, no numpy
    # VecEnv, no per-step host<->device sync; obs normalized via the live
    # training normalizer inside _eval_pair_tensor -- no stats to inject).
    lb_eval = make_tensor(args.task, lb_eval_n, args.device, adversary=True,
                          end_criterion=args.end_criterion,
                          cfg_overrides=args.env_overrides)
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

  print(f"[recipe] net pi={pi_net} qf={qf_net} (ReLU) lr={args.lr} tau={args.tau} "
        f"tgt_upd={args.target_update_interval} ent={args.ent_coef} "
        f"gamma={args.gamma_init}->{args.gamma_end} ({args.gamma_schedule}) "
        f"min_alpha={args.min_alpha} max_alpha={args.max_alpha} "
        f"batch={args.batch_size} grad_steps={args.gradient_steps} "
        f"learn_starts={learning_starts} buffer={args.buffer_size}")
  # Provenance for the timeout-terminal fix: prove the flag reached the buffer.
  _boot = getattr(getattr(model, "replay_buffer", None), "bootstrap_on_timeout", "n/a")
  print(f"[timeout] bootstrap_on_timeout={_boot} "
        f"(False => a timeout is TERMINAL min(l,g); expected for safety/reach-avoid)")

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
  if args.eval_freq > 0 and cumulative:
    # SafeSuccessRateEvalCallback reads safe/success off (g, l). On a cumulative
    # task the learner's g slot IS the dense reward, so those rates would be
    # arithmetic on rewards wearing safety names. The run's quality signal is
    # rollout/ep_rew_mean, and its SAFETY signal is the base env's safety/*
    # counters (dense_margins tasks report a real margin there).
    print("[eval] safe/success-rate eval SKIPPED: meaningless on a "
          f"mode={CUMULATIVE!r} task (g is the dense reward, not a margin)")
  elif args.eval_freq > 0:
    from safety_sb3 import SafeSuccessRateEvalCallback
    from safety_sb3.tensor_env import TensorVecNormalize
    eval_n = 8 if args.smoke else args.eval_envs
    eval_metric_env = TensorVecNormalize(
      make_tensor(args.task, eval_n, args.device, adversary=two_player,
                  end_criterion=args.end_criterion,
                  cfg_overrides=args.env_overrides))

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
    # The always-on training-safety counters (every run reports these).
    m = env.metrics()
    print("[smoke] safety counters: "
          + str({k: round(v, 4) for k, v in m.items()
                 if k.startswith("safety/")}))
    if filtered_arm:
      # The contract the filtered arm rests on: the learner must be able to
      # SEE the executed action through the normalizer wrapper. If this reports
      # None, the filter is running but the buffer is storing proposals.
      seen = getattr(model.env, "executed_action", None)
      verdict = ("OK" if seen is not None else
                 "NOT VISIBLE -- buffer would store PROPOSED actions")
      print(f"[smoke] executed-action readback through "
            f"{type(model.env).__name__}: {verdict}")
    ok = (os.path.exists(os.path.join(outdir, "final_model.zip"))
          and os.path.exists(os.path.join(outdir, "tensornormalize.pt")))
    print(f"[smoke] saved final_model.zip + tensornormalize.pt -> "
          f"{'OK' if ok else 'MISSING'}")


if __name__ == "__main__":
  main()
