"""Zoo trainer (PPO family): any registered task, full telemetry.

  python examples/train.py --family on_policy --task go2_gap_landing --steps 200000000
  python examples/train.py --family on_policy --task go2_gap_chain --steps 2000000000 \
      --load runs/go2_gap_crossing/final_model.zip
  python examples/train.py --family on_policy --task go2_walker_flat --num-envs 4096

It dispatches on the TASK's ``mode`` (registry.MODES), the two branches sharing
one argparse / config / wandb / checkpoint plumbing:

  safety / reach-avoid  the GPU-resident tensor bridge + a safety_sb3 learner
                        (Safety/ReachAvoid/Isaacs/Gameplay PPO, resolved from
                        the mode x --adversary).
  cumulative            plain reward-maximizing RL: the numpy bridge in dense
                        mode + STOCK stable_baselines3 PPO (see
                        :func:`_train_cumulative` — the checkpoint stays a
                        vanilla SB3 zip on purpose). Safety-only knobs are
                        refused in this mode, never silently applied.

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

from _run_config import dump_config, merge_config  # noqa: E402  (examples/ sibling)
from robot_safety_sandbox import (  # noqa: E402
  CUMULATIVE, algo_name, list_tasks, make_tensor, spec)
from robot_safety_sandbox.callbacks import (  # noqa: E402
  ForceRampCallback,
  FwdForceAnnealCallback,
  GaitThreshRampCallback,
  LAnnealCallback,
  PerEnvForceScaleCallback,
  NormFreezeCallback,
  StdFloorCallback,
  TensorNormSaveCallback,
  VideoWandbCallback,
)

# safety_sb3 supplies the SAFETY learners, and is OPTIONAL at import time: a
# CUMULATIVE-only install (dense-reward task policies on stock SB3, see base.py)
# trains through this same trainer, so a missing safety_sb3 must not break the
# import — ALGOS just stays empty and the safety branch fails loud below.
ALGOS = {}
try:
  from safety_sb3 import IsaacsPPO, ReachAvoidPPO, SafetyPPO  # noqa: E402

  ALGOS.update({"SafetyPPO": SafetyPPO, "ReachAvoidPPO": ReachAvoidPPO})
  # The two-player learners need safety_sb3 >= v0.2.0, where the 2x2 (avoid /
  # reach-avoid) x (1P / 2P) is complete. v0.2.0 also RENAMED the two-player
  # reach-avoid game IsaacsPPO -> GameplayPPO and reused the name IsaacsPPO for
  # the two-player AVOID game (ISAACS eq. 7). So on v0.1.0 `IsaacsPPO` still
  # imports and still trains — as the WRONG problem. Gate on GameplayPPO's
  # presence (the v0.2.0 tell) rather than trusting the name.
  try:
    from safety_sb3 import GameplayPPO  # noqa: E402

    ALGOS["GameplayPPO"] = GameplayPPO
    ALGOS["IsaacsPPO"] = IsaacsPPO
  except ImportError:
    pass  # v0.1.0: leave both two-player learners UNAVAILABLE (fail closed)
except ImportError:
  pass  # no safety_sb3: only mode="cumulative" tasks are trainable here

# --- mode="cumulative" (plain reward-maximizing RL) -------------------------
# This trainer's own defaults are the SAFETY recipe; a cumulative task keeps the
# vanilla-PPO recipe (what examples/train_nominal.py used before it became a
# shim onto this trainer). Applied as argparse DEFAULTS, so a --config file or
# an explicit CLI flag still wins.
_CUMULATIVE_DEFAULTS = dict(
  num_envs=1024, steps=150_000_000, lr=3e-4, ent_coef=5e-3,
  video_interval=10_000_000, adaptive_lr=False)  # stock PPO: no KL-adaptive LR

# Knobs that only mean something for the SAFETY backups. In cumulative mode they
# are REJECTED rather than silently ignored — several would quietly change the
# objective (a gamma anneal toward 0.9999 is a safety-value trick, an l-anneal
# reshapes a target set this mode does not have). NB there is no gamma knob in
# this trainer at all (gamma is fixed 0.99 in both branches); the SAC trainer's
# --gamma-* flags don't exist here, so a config carrying them is already
# rejected by merge_config's key validation.
_SAFETY_ONLY = (
  "adversary", "terminal_type", "end_criterion", "l_anneal_steps",
  "l_hold_steps", "adaptive_lr", "desired_kl", "max_std", "std_floor",
  "std_ceil", "target_kl", "reset_value", "fwd_force", "force_max",
  "force_ramp_frac", "force_floor", "force_init", "dstb_pretrain",
  "norm_freeze_steps",
)


def _train_cumulative(args, outdir):
  """mode="cumulative": plain reward-maximizing RL on the env's dense reward —
  the TASK policy a safety filter wraps.

  Deliberately STOCK ``stable_baselines3.PPO`` on the NUMPY bridge, not
  ``SafetyPPO(mode="cumulative")``. The two are numerically identical
  (safety_sb3's cumulative buffer reproduces SB3's GAE bit-for-bit), so the
  class choice is free — and stock PPO keeps the checkpoint a plain SB3 zip,
  which (a) examples/eval_filter.py reads with ``PPO.load`` and (b) stays
  loadable in an install without safety_sb3. Do NOT "upgrade" this to a safety
  learner: it would change the saved artifact format.
  """
  from stable_baselines3 import PPO
  from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

  from robot_safety_sandbox import make_numpy
  from robot_safety_sandbox.callbacks import (
    DenseMetricsCallback, DenseVideoWandbCallback, VecNormSaveCallback)

  print(f"[algo] {args.task} mode={CUMULATIVE} -> PPO (stock stable_baselines3)")
  # numpy bridge, auto-dense for cumulative tasks (reward = the env's dense
  # reward stack); stock PPO gets standard GAE + timeout bootstrap.
  env = make_numpy(args.task, args.num_envs, args.device,
                   cfg_overrides=args.env_overrides)
  # VecMonitor (UNDER VecNormalize) tracks per-env return/length and injects
  # info["episode"] -> SB3 logs rollout/ep_rew_mean + ep_len_mean on the RAW
  # (un-normalized) dense reward. The bare bridge emits no episode infos, so
  # without this the whole rollout/ section is missing.
  env = VecMonitor(env)
  # dense reward CAN be normalized (unlike the safety margin g).
  env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

  net = [int(x) for x in args.net.split(",") if x.strip()]
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
      vf_coef=args.vf_coef, clip_range=0.2, max_grad_norm=1.0,
      policy_kwargs=dict(log_std_init=math.log(0.5),
                         net_arch=dict(pi=net, vf=net)),
      seed=args.seed, verbose=1, device=args.device, tensorboard_log=outdir)
  print(f"[recipe] net={net} ent_coef={args.ent_coef} lr={args.lr} "
        f"vf_coef={args.vf_coef} (stock PPO: no adaptive LR, gamma=0.99)")
  if args.load_tensornorm:
    st = th.load(args.load_tensornorm, map_location="cpu", weights_only=True)
    env.obs_rms.mean = st["obs_mean"].numpy().astype("float64")
    env.obs_rms.var = st["obs_var"].numpy().astype("float64")
    env.obs_rms.count = 1.0e6
    print(f"[warm-start] obs normalizer transplanted from {args.load_tensornorm}")
  if args.reset_log_std is not None:
    with th.no_grad():
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
    wandb.init(project=args.wandb_project, name=args.task, config=vars(args),
               sync_tensorboard=True, save_code=False, reinit=True)
    cbs.append(WandbCallback(verbose=0))
    # Eval clips: a "<task>_video" packed-terrain herd variant if registered
    # (auto-dense: no force / no safety hook -> the unaided policy).
    _vtask = args.task + "_video"
    _vtask = _vtask if _vtask in list_tasks() else args.task
    cbs.append(DenseVideoWandbCallback(
      lambda: make_tensor(_vtask, 8, args.device, render_mode="rgb_array"),
      interval=args.video_interval))

  model.learn(total_timesteps=args.steps, callback=CallbackList(cbs))
  model.save(os.path.join(outdir, "final_model.zip"))   # plain SB3 zip
  env.save(os.path.join(outdir, "vecnormalize.pkl"))
  print(f"[done] {outdir}")


def _peek_mode():
  """The mode of the task named on the CLI (or in the --config), resolved BEFORE
  the strict parse so a cumulative run can swap in the plain-RL defaults.
  Returns None when the task can't be resolved yet (e.g. a bare --help)."""
  pre = argparse.ArgumentParser(add_help=False)
  pre.add_argument("--task")
  pre.add_argument("--config")
  known, _ = pre.parse_known_args()
  task = known.task
  if task is None and known.config and os.path.exists(known.config):
    import yaml
    with open(known.config) as f:
      task = (yaml.safe_load(f) or {}).get("task")
  try:
    return spec(task).mode if task else None
  except KeyError:
    return None


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--config", default=None,
                 help="YAML recipe of args (keys = flag dest names). Sets defaults; "
                      "explicit CLI flags override it. See configs/.")
  p.add_argument("--env-override", action="append", metavar="KEY=VAL", default=None,
                 help="override an env/task cfg_builder param (repeatable), e.g. "
                      "--env-override gate_close_rate=0.003. Also settable as a "
                      "config `env_overrides:` dict. Forwarded to make_tensor.")
  p.add_argument("--task", required=True, help=f"one of {list_tasks()}")
  p.add_argument("--num-envs", type=int, default=2048)
  p.add_argument("--steps", type=int, default=200_000_000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--load", default=None, help="warm-start model .zip (previous stage)")
  p.add_argument("--load-tensornorm", default=None,
                 help="mode=cumulative only: transplant the obs normalizer from "
                      "a safety run's tensornormalize.pt. The safety branch "
                      "auto-detects the normalizer next to --load instead.")
  p.add_argument("--reset-value", action="store_true",
                 help="on warm-start, re-initialize the VALUE (critic) net so it "
                      "learns the new objective's returns from scratch (POLICY-only "
                      "warm-start). Use when the source stage optimized a DIFFERENT "
                      "margin (e.g. avoid g -> reach-avoid): a value net fit to the "
                      "old returns blows up (value_loss explosion -> policy collapse) "
                      "when the new returns diverge. Also clears the optimizer moments "
                      "(stale Adam v on the reset params would give a huge first step).")
  p.add_argument("--reset-log-std", type=float, default=None,
                 help="on warm-start, re-inflate the action std to this value "
                      "(escape a CONVERGED/collapsed source policy while keeping "
                      "its learned features); leave unset to preserve the std")
  p.add_argument("--max-std", type=float, default=None,
                 help="RA-stable: hard cap on the action std (clamped after "
                      "every update; prevents the organic std inflation that "
                      "eroded motor skill in the synthesis runs)")
  p.add_argument("--target-kl", type=float, default=None,
                 help="RA-stable: SB3 target_kl early-stop for actor epochs")
  # The two orthogonal reach-avoid termination knobs (see registry.END_CRITERIA
  # and safety_sb3 backups.TERMINAL_TYPES). terminal_type is ALGORITHM-side (how
  # a terminal step is VALUED); end_criterion is ENV-side (WHEN the episode ends
  # from g, l). The pairing the campaign wants — reach-DEEPER — is the default
  # terminal_type="all" with end_criterion="failure".
  p.add_argument("--terminal-type", choices=["all", "g"], default="all",
                 help="reach-avoid learners only: value a terminal step as "
                      "min(l,g) ('all', default) or g ('g'). Ignored (with a "
                      "notice) on avoid tasks — SafetyPPO/IsaacsPPO have no l.")
  p.add_argument("--end-criterion", choices=["failure", "reach-avoid", "timeout"],
                 default=None,
                 help="WHEN the episode ends from (g,l); default = the task's "
                      "TaskSpec value. 'failure' (all tasks today): failure set "
                      "+ timeout, never on reach. 'reach-avoid': also end on "
                      "success (g>=0 & l>=0). 'timeout': only the env timeout.")
  p.add_argument("--adversary", action="store_true")
  p.add_argument("--force-max", type=float, default=50.0)
  # ISAACS game-balance knobs. The adversary force is SUSTAINED (every step),
  # so calibrate force_max to the ctrl's physical envelope (a biped's static
  # ankle authority caps sustained lateral force ~40N; go2-style 0.3x-bodyweight
  # targets are quadruped-only). If force_scale pins at the floor, the game is
  # frozen in an unwinnable state (zero gradient) -> lower force_max / floor.
  p.add_argument("--force-ramp-frac", type=float, default=0.55,
                 help="fraction of the run over which the global ramp reaches "
                      "force-max")
  p.add_argument("--force-floor", type=float, default=0.3,
                 help="per-env survival-curriculum minimum force scale")
  p.add_argument("--force-init", type=float, default=0.5,
                 help="per-env survival-curriculum initial force scale")
  p.add_argument("--fwd-force", type=float, default=15.0,
                 help="crawl: start magnitude (N) of the forward-current force, "
                      "annealed to 0 over the run")
  p.add_argument("--dstb-pretrain", type=int, default=20,
                 help="IsaacsPPO dstb-pretrain ROLLOUTS (keep << total rollouts)")
  p.add_argument("--lr", type=float, default=5e-4)
  p.add_argument("--ent-coef", type=float, default=1e-4)
  # Safety-RL recipe knobs. Defaults keep the locomotion recipe (go2); for a
  # SAFETY task match the proven rsl_rl config: --ent-coef 0 --no-adaptive-lr
  # --net 128,128,128. The safety margin g(s) is smooth with tiny advantages, so
  # (a) any positive entropy bonus makes log-std run away, and (b) the KL-adaptive
  # LR death-spirals (can't improve -> LR crashes -> updates freeze). See
  # digit_v3_flat_safety_ppo_runner_cfg in MjlabSafety_Digit/.../rl_cfg.py.
  p.add_argument("--adaptive-lr", action=argparse.BooleanOptionalAction,
                 default=True, help="KL-adaptive LR (locomotion); use "
                 "--no-adaptive-lr for a fixed LR on safety tasks")
  p.add_argument("--desired-kl", type=float, default=0.01,
                 help="KL target for the adaptive LR. Tight values (0.01) can "
                      "death-spiral the LR to 0 (frozen policy); loosen to "
                      "0.02-0.03 to keep learning alive for exploration")
  p.add_argument("--std-floor", type=float, default=None,
                 help="floor the action std at this value (prevents premature "
                      "std collapse -> more sustained exploration)")
  p.add_argument("--std-ceil", type=float, default=None,
                 help="ceiling the action std (prevents entropy-bonus log-std "
                      "runaway -> aggressive thrashing / ep_len collapse)")
  p.add_argument("--l-anneal-steps", type=int, default=0,
                 help="reach-set curriculum: linearly contract the reach target "
                      "l from LOOSE (~=g) to STRICT over this many steps (0=off, "
                      "l stays strict). Use on WARM-STARTED reach-avoid to avoid "
                      "the OOD collapse when a tight l yanks the policy off the "
                      "avoid base")
  p.add_argument("--l-hold-steps", type=int, default=20_000_000,
                 help="hold l at LOOSE (alpha=0) this long before contracting, "
                      "so a re-inflated warm-start policy re-settles on the base")
  p.add_argument("--net", default="512,256,128",
                 help="comma-separated hidden dims for pi & vf "
                      "(safety: 128,128,128)")
  p.add_argument("--vf-coef", type=float, default=0.5,
                 help="value loss coefficient (rsl_rl safety uses 1.0)")
  p.add_argument("--video-interval", type=int, default=5_000_000,
                 help="env-steps between wandb eval videos (~2k rollouts at "
                      "2560 envs); frequent clips make regressions visible early")
  p.add_argument("--norm-freeze-steps", type=int, default=5_000_000,
                 help="freeze obs-norm updates at the start of WARM-STARTED runs")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=os.path.join(_ZOO, "runs"),
                 help="output root; ALWAYS keep runs under runs/ (git-ignored) — "
                      "never invent runs_<suffix> siblings, they escape .gitignore")
  p.add_argument("--wandb-project", default="robot_safety_sandbox")
  p.add_argument("--no-wandb", action="store_true")
  if _peek_mode() == CUMULATIVE:
    p.set_defaults(**_CUMULATIVE_DEFAULTS)
  args = merge_config(p)   # parse args; an optional --config sets defaults, CLI overrides

  s = spec(args.task)
  if s.mode == CUMULATIVE:
    # Safety knobs are refused here, never silently applied: this mode
    # maximizes the env's dense reward and has neither margins nor a target set.
    bad = sorted(d for d in _SAFETY_ONLY
                 if getattr(args, d) != p.get_default(d))
    if bad:
      raise SystemExit(
        f"'{args.task}' is a mode={CUMULATIVE!r} task (plain reward-maximizing "
        f"RL on the env's dense reward, stock stable_baselines3 PPO) — these "
        f"SAFETY-only knobs have no meaning for it and are refused rather than "
        f"silently applied: {bad}.")
    tag = args.task
  else:
    if args.load_tensornorm:
      raise SystemExit(
        "--load-tensornorm is a mode='cumulative' knob; the safety branch "
        "auto-detects tensornormalize.pt / vecnormalize.pkl next to --load.")
    tag = args.task + ("_adv" if args.adversary else "")
  outdir = os.path.join(args.out, tag)
  os.makedirs(outdir, exist_ok=True)
  dump_config(outdir, args)   # reproducible: re-run with --config <outdir>/config.yaml
  if s.mode == CUMULATIVE:
    return _train_cumulative(args, outdir)

  # Resolve the learner from the task's PROBLEM (avoid vs reach-avoid, set by
  # its margins) x the RUN's player count (--adversary). algo_name() also
  # refuses an avoid-only task on a reach-avoid learner, which has no valid
  # formulation for any constant l (see margins.py).
  algo = algo_name(args.task, adversary=args.adversary)
  if not ALGOS:
    raise SystemExit(
      f"'{args.task}' is a mode={s.mode!r} task and needs the '{algo}' learner, "
      f"but safety_sb3 is not installed (only mode='cumulative' tasks train "
      f"without it). Install safety-stable-baselines or set $SAFETY_SB3_PATH.")
  if algo not in ALGOS:
    raise SystemExit(
      f"'{args.task}'{' +--adversary' if args.adversary else ''} needs the "
      f"'{algo}' learner, which this safety_sb3 does not export. The "
      f"two-player learners require safety_sb3 >= v0.2.0 (pyproject still "
      f"pins v0.1.0, where 'IsaacsPPO' silently means the reach-avoid game "
      f"and there is no avoid game at all). Bump the pin before training this.")
  print(f"[algo] {args.task} adversary={args.adversary} -> {algo}")

  env = make_tensor(args.task, args.num_envs, args.device,
                    adversary=args.adversary, end_criterion=args.end_criterion,
                    cfg_overrides=args.env_overrides)
  eff_ec = args.end_criterion if args.end_criterion is not None else s.end_criterion
  print(f"[end-criterion] {args.task} -> {eff_ec}"
        f"{' (override)' if args.end_criterion is not None else ' (task default)'}")
  Algo = ALGOS[algo]
  akw = {}
  # terminal_type is a REACH-AVOID learner knob only; passing it to an avoid
  # learner (SafetyPPO/IsaacsPPO) would be a TypeError, and it is meaningless
  # there anyway (no l). Pass it only when the resolved algo is reach-avoid.
  if algo in ("ReachAvoidPPO", "GameplayPPO"):
    akw["terminal_type"] = args.terminal_type
    print(f"[terminal-type] {algo} -> {args.terminal_type}")
  elif args.terminal_type != "all":
    print(f"[terminal-type] ignored: {algo} is an avoid learner (no target "
          f"set), so --terminal-type {args.terminal_type} has no effect.")
  if args.adversary:  # full two-player game, tensor path
    akw.update(dict(ctrl_action_dim=s.ctrl_dim,
               dstb_learning_rate=3e-4, dstb_ent_coef=2e-3,  # reference values
               dstb_pretrain_rollouts=args.dstb_pretrain,
               ctrl_rollouts_per_cycle=12, dstb_rollouts_per_cycle=3,
               use_leaderboard=True,
               leaderboard_dir=os.path.join(outdir, "leaderboard")))

  net = [int(x) for x in args.net.split(",") if x.strip()]
  model = Algo(
    "MlpPolicy", env, **akw,
    n_steps=48, batch_size=args.num_envs * 48 // 4, n_epochs=5,
    gamma=0.99, gae_lambda=0.95, learning_rate=args.lr,
    ent_coef=args.ent_coef, vf_coef=args.vf_coef, clip_range=0.2,
    max_grad_norm=1.0, normalize_obs=True,
    adaptive_lr=args.adaptive_lr,
    desired_kl=args.desired_kl if args.adaptive_lr else None,
    policy_kwargs=dict(log_std_init=math.log(0.3),
                       net_arch=dict(pi=net, vf=net)),
    seed=args.seed, verbose=1, device=args.device,
    tensorboard_log=outdir)
  print(f"[recipe] net={net} ent_coef={args.ent_coef} "
        f"adaptive_lr={args.adaptive_lr} vf_coef={args.vf_coef} lr={args.lr}")

  cbs = [
    CheckpointCallback(save_freq=max(1, 25_000_000 // args.num_envs),
                       save_path=os.path.join(outdir, "checkpoints"),
                       name_prefix="model"),
    TensorNormSaveCallback(os.path.join(outdir, "checkpoints")),
  ]
  if args.std_floor is not None or args.std_ceil is not None:
    cbs.append(StdFloorCallback(args.std_floor if args.std_floor is not None
                                else 1e-6, args.std_ceil))
  if args.l_anneal_steps > 0:
    cbs.append(LAnnealCallback(anneal_steps=args.l_anneal_steps,
                               hold_steps=args.l_hold_steps))
    print(f"[curriculum] reach-set l-anneal: hold {args.l_hold_steps} then "
          f"contract over {args.l_anneal_steps} steps")

  # Crawl: anneal the forward-current force 15 N -> 0 (hold 40M, decay over 160M)
  # so it bootstraps the gait then weans off -> final policy crawls unaided.
  if "crawl" in args.task and not args.adversary:
    cbs.append(FwdForceAnnealCallback(
      start=args.fwd_force, hold_steps=int(0.13 * args.steps),
      anneal_steps=int(0.54 * args.steps)))
    cbs.append(GaitThreshRampCallback())  # trot-threshold curriculum

  if args.target_kl is not None:
    model.target_kl = args.target_kl
    print(f"[ra-stable] target_kl={args.target_kl} (early-stop actor epochs)")
  if args.max_std is not None:
    import math as _math
    import torch as _th
    from stable_baselines3.common.callbacks import BaseCallback as _BC

    class _StdCap(_BC):
      def _on_rollout_start(self):
        with _th.no_grad():
          if hasattr(self.model.policy, "log_std"):
            self.model.policy.log_std.clamp_(max=_math.log(args.max_std))
      def _on_step(self):
        return True

    cbs.append(_StdCap())
    print(f"[ra-stable] action std capped at {args.max_std}")
  if args.load:
    model.set_parameters(args.load, exact_match=False, device=args.device)
    if args.reset_value:
      from functools import partial
      pol = model.policy
      # Re-init the value pathway only (keep the warm-started policy pathway):
      # the value MLP branch + the value head. SB3 inits the value net with
      # orthogonal gain sqrt(2) (extractor) / 1.0 (head).
      if hasattr(pol.mlp_extractor, "value_net"):
        pol.mlp_extractor.value_net.apply(partial(pol.init_weights, gain=math.sqrt(2)))
      pol.value_net.apply(partial(pol.init_weights, gain=1.0))
      # Fresh optimizer moments: stale Adam second-moments on the just-reset value
      # params would produce a huge effective first step and re-explode the value.
      # Policy WEIGHTS stay warm-started; only the optimizer state resets.
      pol.optimizer = pol.optimizer_class(
        pol.parameters(), lr=args.lr, **(pol.optimizer_kwargs or {}))
      print("[warm-start] value net RESET (policy-only warm-start) + optimizer "
            "moments cleared")
    if args.reset_log_std is not None:
      with th.no_grad():
        model.policy.log_std.fill_(math.log(args.reset_log_std))
      print(f"[warm-start] re-inflated action std -> {args.reset_log_std} "
            f"(re-exploring around the loaded features)")
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
    cbs.append(ForceRampCallback(args.force_max,
                                 int(args.force_ramp_frac * args.steps)))
    cbs.append(PerEnvForceScaleCallback(lo=args.force_floor,
                                        init=args.force_init))
    print(f"[adversary] force ramp 8->{args.force_max}N over "
          f"{args.force_ramp_frac:.0%}; per-env scale floor={args.force_floor} "
          f"init={args.force_init}")

  if not args.no_wandb:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    wandb.init(project=args.wandb_project, name=tag, config=vars(args),
               sync_tensorboard=True, save_code=False, reinit=True)
    cbs.append(WandbCallback(verbose=0))
    # Use a "<task>_video" packed-terrain herd variant for the eval clip if one
    # is registered (crawl spreads envs ~24 m -> a herd needs packed patches).
    _vtask = args.task + "_video"
    _vtask = _vtask if _vtask in list_tasks() else args.task
    cbs.append(VideoWandbCallback(
      lambda: make_tensor(_vtask, 8, args.device, adversary=False,
                          render_mode="rgb_array",  # native herd, one scene
                          end_criterion=args.end_criterion),
      interval=args.video_interval))

  model.learn(total_timesteps=args.steps, callback=CallbackList(cbs))
  model.save(os.path.join(outdir, "final_model.zip"))
  if hasattr(model.env, "save"):
    model.env.save(os.path.join(outdir, "tensornormalize.pt"))
  print(f"[done] {outdir}")


if __name__ == "__main__":
  main()
