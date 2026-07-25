"""Evaluation entry point: pick an env, a filter, a nominal, an attack.

One script for every filter evaluation in this repo. The four axes are chosen
independently on the command line and the harness composes them; nothing about
any particular terrain lives here (see robot_safety_sandbox/eval/).

    --task / --preset / --env-override   WHICH WORLD
    --nominal                            the pi_task being filtered
    --twin  / --filter                   the certificate + which composition
    --dstb  / --dstb-scale               WHO IS ATTACKING, and how hard

  # filter gauntlet on flat ground under a swept adversarial attack
  python examples/eval.py --task go2_locomote --adversary \
      --nominal runs/go2_walker_flat/final_model.zip \
      --twin runs/go2_stabilize_sac2p/final_model.zip \
      --filter gameplay --dstb policy --dstb-scale 0.5 --num-envs 256

  # the gap gauntlet (E021), now a preset rather than its own script
  python examples/eval.py --preset gap_gauntlet \
      --nominal runs/go2_walker_flat/final_model.zip \
      --twin runs/go2_gap_chain_ra/final_model.zip \
      --filter value --gap-width 0.35 --n-gaps 1 --num-envs 256 --steps 600

  # the unfiltered control arm
  python examples/eval.py ... --no-filter

An attack STRENGTH sweep is the point of --dstb-scale: run the same command at
0.0 / 0.25 / ... / 1.0 and the result is a curve, not a point.
"""

from __future__ import annotations

import argparse
import json
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

from robot_safety_sandbox import list_tasks, spec  # noqa: E402
from robot_safety_sandbox.eval import (  # noqa: E402
  FILTERS, NominalPolicy, RolloutCfg, SwitchCfg, TwinNominal,
  TwistCommandSurgery, VideoRecorder, ZeroNominal, build_eval_env, build_filter,
  list_presets, load_nominal, load_twin, make_dstb, preset, protocol_metrics,
  run_eval, safety_modules)
from robot_safety_sandbox.eval.metrics import MetricSet  # noqa: E402


def _parse_overrides(pairs):
  """``KEY=VAL`` strings -> a cfg_builder kwargs dict (VAL parsed as YAML)."""
  import yaml
  out = {}
  for item in pairs or []:
    if "=" not in item:
      raise SystemExit(f"--env-override expects KEY=VAL, got {item!r}")
    k, v = item.split("=", 1)
    out[k.strip()] = yaml.safe_load(v)
  return out


def build_parser(pre_args):
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  # --- which world
  p.add_argument("--task", default=None, help=f"one of {list_tasks()}")
  p.add_argument("--preset", default=None, choices=list_presets(),
                 help="a named evaluation configuration (env surgery + the "
                      "task's own success metrics)")
  p.add_argument("--env-override", action="append", metavar="KEY=VAL",
                 default=None,
                 help="override a param of the task's cfg_builder (repeatable)")
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--episode-s", type=float, default=None,
                 help="override the env's episode length (s)")
  p.add_argument("--cmd-vx", type=float, default=1.0,
                 help="forward velocity command given to the nominal")
  p.add_argument("--engaged-cmd-vx", type=float, default=1.0,
                 help="command held while the FALLBACK drives -- the value the "
                      "twin was trained under (feeding it 0 is OOD)")
  p.add_argument("--no-command-surgery", action="store_true",
                 help="do not drive the velocity command from the filter's "
                      "verdict (see eval/envs.py TwistCommandSurgery)")
  p.add_argument("--end-criterion", default=None,
                 choices=["failure", "reach-avoid", "timeout"])
  p.add_argument("--safety-obs-key", default=None,
                 help="override the twin's obs group (default: auto-detected)")
  p.add_argument("--nominal-obs-key", default=None,
                 help="override the nominal's obs group (default: auto-detected)")
  # --- the policies
  p.add_argument("--nominal", default=None,
                 help="pi_task checkpoint (stock SB3 zip); omit for a zero "
                      "nominal (fallback-only run)")
  p.add_argument("--nominal-from-twin", action="store_true",
                 help="use the TWIN's own control policy as the nominal -- the "
                      "disturbance-effect probe: evaluate the deployable "
                      "policy the two-player game produced against its own "
                      "adversary (pair with --no-filter and a --dstb sweep)")
  p.add_argument("--twin", default=None,
                 help="safety twin checkpoint (any of the eight MAP cells)")
  # --- the filter
  p.add_argument("--filter", default="value", choices=sorted(FILTERS),
                 help="which COMPOSITION to deploy; they differ in one module "
                      "each: " + " | ".join(f"{k}: {v}"
                                            for k, v in FILTERS.items()))
  p.add_argument("--no-filter", action="store_true",
                 help="CONTROL arm: run the nominal unfiltered")
  p.add_argument("--eps", type=float, default=0.0)
  p.add_argument("--caution", type=float, default=0.45)
  p.add_argument("--hysteresis", type=float, default=0.15)
  p.add_argument("--kappa", type=float, default=0.8,
                 help="--filter qcbf: class-K coefficient, Q(x,u) >= kappa V(x)")
  p.add_argument("--horizon", type=int, default=20,
                 help="--filter rollout/gameplay: H sim steps per certification")
  p.add_argument("--rollouts", type=int, default=1,
                 help="--filter rollout/gameplay: parallel rollouts per env; "
                      "the verdict is their MIN (one failure condemns)")
  p.add_argument("--recertify-every", type=int, default=1,
                 help="--filter rollout/gameplay: re-certify every k steps")
  # NOTE: there is deliberately no --rollout-reach-avoid. Which reduction
  # certifies the rollout is DERIVED from the task's mode (see
  # eval/filters.py::reach_avoid_reduction); it is not the operator's to pick.
  p.add_argument("--contact-history", default="sync",
                 choices=("sync", "instantaneous", "ignore"))
  # --- the attack
  p.add_argument("--adversary", action="store_true",
                 help="give the env a live disturbance channel (the task must "
                      "declare supports_adversary)")
  p.add_argument("--dstb", default="none", choices=["none", "random", "policy"],
                 help="who plays the disturbance: nobody, uniform noise, or a "
                      "trained min-player (--dstb-twin, else --twin)")
  p.add_argument("--dstb-scale", type=float, default=1.0,
                 help="ATTACK STRENGTH, continuous. 1.0 = the task's own "
                      "training-time magnitude; sweep it for a curve")
  p.add_argument("--dstb-twin", default=None,
                 help="checkpoint supplying the disturbance actor, when it is "
                      "not the same twin as the certificate")
  # --- run
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--seed", type=int, default=None)
  p.add_argument("--progress-every", type=int, default=0)
  p.add_argument("--out", default=None, help="write metrics JSON here")
  p.add_argument("--video", default=None,
                 help="write an mp4 here; the border tint is the fraction of "
                      "the herd the filter is driving. Needs an offscreen GL "
                      "context — use MUJOCO_GL=egl on a headless box "
                      "(otherwise mujoco raises 'gladLoadGL error')")
  p.add_argument("--video-fps", type=int, default=30)
  p.add_argument("--tag", default=None, help="label copied into the summary")

  if pre_args.preset:
    ps = preset(pre_args.preset)
    if ps.add_args:
      ps.add_args(p)
    if ps.defaults:
      p.set_defaults(**ps.defaults)
    if ps.task:
      p.set_defaults(task=ps.task)
  return p


def main():
  pre = argparse.ArgumentParser(add_help=False)
  pre.add_argument("--preset", default=None)
  pre_args, _ = pre.parse_known_args()
  args = build_parser(pre_args).parse_args()

  if not args.task:
    raise SystemExit("--task is required (or use a --preset that names one). "
                     f"Registered: {list_tasks()}")
  if not args.twin:
    raise SystemExit("--twin is required: the filter's certificate and its "
                     "fallback both come from a safety twin.")
  ps = preset(args.preset) if args.preset else None
  device = args.device

  # 1. THE ENVIRONMENT ---------------------------------------------------------
  overrides = _parse_overrides(args.env_override)
  env = build_eval_env(
    args.task, args.num_envs, device, adversary=args.adversary,
    env_overrides=overrides, end_criterion=args.end_criterion,
    cfg_transform=ps.cfg_transform(args) if (ps and ps.cfg_transform) else None,
    safety_obs_key=args.safety_obs_key, nominal_obs_key=args.nominal_obs_key,
    render_mode="rgb_array" if args.video else None)
  if args.episode_s is not None:
    env.mj.cfg.episode_length_s = args.episode_s
  print(f"[env] {args.task} n={env.num_envs} adversary={env.adversary} "
        f"obs: nominal='{env.nominal_obs_key}' safety='{env.safety_obs_key}'"
        + (f" preset={args.preset}" if ps else ""))

  # 2. THE SAFETY FILTER -------------------------------------------------------
  twin, norm = load_twin(args.twin, device)
  mods = safety_modules(twin, env.num_envs, device)
  mods["norm"] = norm
  bundle = build_filter(
    args.filter, mods, env,
    switch=SwitchCfg(eps=args.eps, caution=args.caution,
                     hysteresis=args.hysteresis),
    rollout=RolloutCfg(horizon=args.horizon, rollouts=args.rollouts,
                       recertify_every=args.recertify_every,
                       contact_history=args.contact_history),
    kappa=args.kappa)
  print(f"[filter] {args.filter}: {FILTERS[args.filter]}"
        + (" (NOT APPLIED: --no-filter control arm)" if args.no_filter else ""))

  # 3. THE NOMINAL POLICY ------------------------------------------------------
  if args.nominal and args.nominal_from_twin:
    raise SystemExit("--nominal and --nominal-from-twin are alternatives: the "
                     "nominal is either an external pi_task or the twin's own "
                     "control policy, not both.")
  if args.nominal_from_twin:
    print(f"[nominal] the twin's own control policy ({mods['twin']})")
    nominal = TwinNominal(mods["fallback_fn"], norm)
  elif args.nominal:
    nominal = NominalPolicy(*load_nominal(args.nominal, device), device=device)
  else:
    print("[nominal] none: zero action (fallback-only run)")
    nominal = ZeroNominal(env.num_envs, env.ctrl_dim, device)

  # 4. THE ATTACK --------------------------------------------------------------
  dstb_fn = None
  if args.dstb != "none":
    if not env.adversary:
      raise SystemExit(f"--dstb {args.dstb} needs a live disturbance channel; "
                       "pass --adversary (the task must support one).")
    src = mods
    if args.dstb_twin:
      d_twin, d_norm = load_twin(args.dstb_twin, device)
      src = safety_modules(d_twin, env.num_envs, device) | {"norm": d_norm}
    dstb_fn = make_dstb(args.dstb, env, src)
    print(f"[dstb] {args.dstb} @ scale={args.dstb_scale} "
          f"(dim={env.dstb_dim}, mode={env.bridge.dstb_mode})")

  # 5. THE METRICS -------------------------------------------------------------
  metrics = protocol_metrics(env, bundle.filt)
  if ps and ps.metrics:
    extra = ps.metrics(args, env)
    if extra is not None:
      metrics = MetricSet(metrics, extra)

  surgery = None
  if not args.no_command_surgery:
    surgery = TwistCommandSurgery(cmd_vx=args.cmd_vx,
                                  engaged_cmd_vx=args.engaged_cmd_vx).bind(env)
    if not surgery.available:
      surgery = None

  video = VideoRecorder(args.video, args.video_fps) if args.video else None
  summary = run_eval(
    env, nominal, bundle.filt, metrics, steps=args.steps, norm=norm,
    dstb_fn=dstb_fn, dstb_scale=args.dstb_scale, no_filter=args.no_filter,
    command_surgery=surgery, video=video, seed=args.seed,
    progress_every=args.progress_every)
  if video is not None:
    video.save()

  summary.update(
    task=args.task, preset=args.preset, composition=args.filter,
    filter="off" if args.no_filter else os.path.basename(
      os.path.dirname(os.path.abspath(args.twin))),
    twin=mods["twin"], steps=args.steps, num_envs=args.num_envs,
    dstb=args.dstb, dstb_scale=args.dstb_scale, seed=args.seed, tag=args.tag)
  if bundle.shadow is not None:
    summary["certifications"] = bundle.shadow.seeds

  print("\n=== EVAL ===")
  for k, v in summary.items():
    print(f"  {k:28s} {v}")
  if args.out:
    with open(args.out, "w") as f:
      json.dump(summary, f, indent=2)
    print(f"[saved] {args.out}")
  bundle.close()
  env.close()


if __name__ == "__main__":
  main()
