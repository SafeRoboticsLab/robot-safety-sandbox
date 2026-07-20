"""Interactively play a trained safety checkpoint in mjlab's real-time viewer.

The zoo trains raw safety_sb3 models on the GPU-resident tensor bridge
(``MjlabTensorSafetyEnv.step_tensor``), while mjlab's interactive viewers
(``NativeMujocoViewer`` / ``ViserPlayViewer``) expect the rsl-rl-style
``EnvProtocol`` (``get_observations`` / ``step`` / ``unwrapped`` ...). This script
is the ~30-line adapter between the two, plus checkpoint loading — so any trained
policy opens in the same pause / single-step / speed / env-cycle viewer the
unitree_rl_mjlab ``scripts/play.py`` uses.

  # PPO single-player (loads + plays on the same adversary-off env)
  python examples/play.py --task car_goal --algo ReachAvoidPPO --run runs/car_goal

  # two-player SAC: load with the adversary on (to match the saved [ctrl,dstb]
  # action space), then play the DEPLOYABLE ctrl policy with the adversary off
  python examples/play.py --task go2_stabilize --algo GameplaySAC \
      --run runs/go2_stabilize_gameplaysac

  # ... or drive the LEARNED disturbance too -> watch it resist the worst-case force
  python examples/play.py --task go2_stabilize --algo GameplaySAC \
      --run runs/go2_stabilize_gameplaysac --adversary

  --viewer native  (a MuJoCo window; needs $DISPLAY). Also lets you SHOVE the robot
                   by hand: double-click a body, then Ctrl+drag (right = force,
                   left = torque) — standard MuJoCo mouse perturbations.
  --viewer viser   (a browser viewer; prints a URL — works headless; no mouse-force)
  --viewer auto    (native if a display exists, else viser; default)
"""

from __future__ import annotations

import argparse
import os
import sys

import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import safety_sb3  # noqa: E402
from robot_safety_sandbox import make_tensor  # noqa: E402

# Two-player learners were trained on a [ctrl, dstb] action space; the checkpoint
# must be loaded against an adversary=True env so the saved shapes line up. The
# PLAY env is always adversary-off -- we show the deployable ctrl policy (exactly
# what train_gameplay_sac's VideoWandbCallback renders).
_TWO_PLAYER = ("IsaacsPPO", "IsaacsSAC", "GameplayPPO", "GameplaySAC")


class _PlayEnv:
  """Present the tensor safety env through mjlab's viewer ``EnvProtocol``."""

  def __init__(self, safety_env):
    self._e = safety_env
    self._obs = None

  @property
  def num_envs(self):
    return self._e.num_envs

  @property
  def device(self):
    return self._e.mj.device

  @property
  def cfg(self):
    return self._e.mj.cfg

  @property
  def unwrapped(self):
    return self._e.mj  # the ManagerBasedRlEnv the viewer reads sim/scene from

  def get_observations(self):
    if self._obs is None:
      self._obs = self._e.reset()
    return self._obs

  def step(self, actions):
    self._obs, *_ = self._e.step_tensor(th.clamp(actions, -1.0, 1.0))
    return self._obs

  def reset(self):
    self._obs = self._e.reset()
    return self._obs

  def close(self):
    fn = getattr(self._e, "close", None)
    if callable(fn):
      fn()


class _Policy:
  """obs -> deterministic action, with the training obs-normalizer folded in.

  ``adversary=False`` returns just the ctrl action (the deployable policy).
  ``adversary=True`` also runs the learned disturbance actor and returns the
  concatenated ``[ctrl, dstb]`` the two-player env expects -- so you watch the
  robot hold up against the WORST-CASE force the game trained against (mirrors
  isaacs.py's tensor rollout: ``cat([ctrl_actor(o), dstb_actor(o)])``, actions
  already in [-1, 1], the env clamps).

  ``stock=True`` is a vanilla SB3 PPO checkpoint (a kind="nominal" bridge/walker,
  trained on the numpy path): obs stats come from ``vecnormalize.pkl`` and actions
  from ``model.predict`` (numpy), not the safety_sb3 tensor policy."""

  def __init__(self, model, run_dir, device, adversary=False, stock=False):
    self._m = model
    self._adv = adversary
    self._stock = stock
    self._dstb = None if stock else getattr(model.policy, "dstb_actor", None)
    if adversary and self._dstb is None:
      raise RuntimeError(
        "--adversary needs a two-player checkpoint exposing policy.dstb_actor")
    self._mean = self._var = None
    tp = os.path.join(run_dir, "tensornormalize.pt")
    vp = os.path.join(run_dir, "vecnormalize.pkl")
    if os.path.exists(tp):
      vn = th.load(tp, map_location=device, weights_only=False)
      self._mean = vn["obs_mean"].to(device)
      self._var = vn["obs_var"].to(device)
    elif os.path.exists(vp):                       # stock SB3 VecNormalize stats
      import pickle
      with open(vp, "rb") as f:
        vecnorm = pickle.load(f)
      self._mean = th.as_tensor(vecnorm.obs_rms.mean, dtype=th.float32, device=device)
      self._var = th.as_tensor(vecnorm.obs_rms.var, dtype=th.float32, device=device)
    else:
      print(f"[play] no obs normalizer in {run_dir} -> raw observations")

  def _norm(self, o):
    if self._mean is None:
      return o
    return th.clamp((o - self._mean) / th.sqrt(self._var + 1e-8), -10.0, 10.0)

  def __call__(self, obs):
    with th.no_grad():
      o = self._norm(obs)
      if self._stock:                              # vanilla SB3 PPO: numpy predict
        act, _ = self._m.predict(o.detach().cpu().numpy(), deterministic=True)
        return th.as_tensor(act, device=obs.device, dtype=obs.dtype)
      ctrl = self._m.policy._predict(o, deterministic=True)
      if not self._adv:
        return ctrl
      dstb = self._dstb(o, deterministic=True)
      return th.cat([ctrl, dstb], dim=1)


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--task", required=True)
  ap.add_argument("--algo", required=True, choices=[
      "SafetyPPO", "ReachAvoidPPO", "IsaacsPPO", "GameplayPPO",
      "SafetySAC", "ReachAvoidSAC", "IsaacsSAC", "GameplaySAC", "PPO"],
      help="the learner CLASS the checkpoint was trained as ('PPO' = a vanilla "
           "SB3 nominal bridge/walker on the numpy path)")
  ap.add_argument("--run", help="run dir holding final_model.zip + tensornormalize.pt")
  ap.add_argument("--load", help="explicit path to a model .zip (overrides --run)")
  ap.add_argument("--num-envs", type=int, default=4, help="herd size to render")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--viewer", choices=["auto", "native", "viser"], default="auto")
  ap.add_argument("--steps", type=int, default=None, help="auto-stop after N steps")
  ap.add_argument("--adversary", action="store_true",
                  help="two-player checkpoints: also drive the learned disturbance "
                       "actor, so the robot holds up against the worst-case force")
  ap.add_argument("--env-override", action="append", default=[], metavar="KEY=VAL",
                  help="override a cfg_builder param (repeatable), e.g. "
                       "--env-override fixed_level=11. Same generic passthrough as "
                       "train.py; values are parsed as int/float/bool else string.")
  args = ap.parse_args()

  run_dir = args.run or (os.path.dirname(args.load) if args.load else None)
  model_path = args.load or (os.path.join(args.run, "final_model.zip") if args.run else None)
  if model_path is None or not os.path.exists(model_path):
    ap.error(f"model not found: pass --run <dir> or --load <zip> (got {model_path})")

  # generic env/task param passthrough to the cfg_builder (e.g. fixed_level=11),
  # values coerced via YAML -- mirrors train.py's --env-override.
  import yaml
  cfg_overrides = {}
  for item in args.env_override:
    if "=" not in item:
      ap.error(f"--env-override expected KEY=VAL, got {item!r}")
    k, v = item.split("=", 1)
    cfg_overrides[k.strip()] = yaml.safe_load(v)
  cfg_overrides = cfg_overrides or None
  if cfg_overrides:
    print(f"[play] env overrides: {cfg_overrides}")

  stock = args.algo == "PPO"
  load_adv = args.algo in _TWO_PLAYER
  if args.adversary and not load_adv:
    ap.error(f"--adversary needs a two-player checkpoint; {args.algo} is single-player")
  show_adv = bool(args.adversary)
  mode = ("nominal walker (stock PPO)" if stock else
          "ctrl + learned worst-case disturbance" if show_adv else
          "deployable ctrl policy (adversary off)")
  print(f"[play] {args.task} | {args.algo} | play mode: {mode}")

  def _mk(adversary):
    return make_tensor(args.task, args.num_envs, args.device, adversary=adversary,
                       cfg_overrides=cfg_overrides)

  if stock:
    from stable_baselines3 import PPO as _PPO
    model = _PPO.load(model_path, device=args.device)  # spaces restored from zip
    play_env = _mk(False)
  else:
    load_env = _mk(load_adv)                 # env matching the SAVED action space
    Cls = getattr(safety_sb3, args.algo)
    model = Cls.load(model_path, env=load_env, device=args.device)
    # play env: adversary-on iff we're driving the disturbance actor (reuse load_env
    # when the flag matches, else build the other-player-count env)
    play_env = load_env if show_adv == load_adv else _mk(show_adv)
  print(f"[play] loaded {model_path}")

  env = _PlayEnv(play_env)
  # draw the whole herd in one scene (like the unitree videos)
  try:
    env.cfg.viewer.max_extra_envs = max(1, args.num_envs - 1)
  except Exception:
    pass
  policy = _Policy(model, run_dir, args.device, adversary=show_adv, stock=stock)

  backend = args.viewer
  if backend == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    backend = "native" if has_display else "viser"
  print(f"[play] viewer backend: {backend}")

  if backend == "native":
    from mjlab.viewer import NativeMujocoViewer
    viewer = NativeMujocoViewer(env, policy)
  else:
    from mjlab.viewer import ViserPlayViewer
    viewer = ViserPlayViewer(env, policy)

  viewer.run(num_steps=args.steps)
  env.close()


if __name__ == "__main__":
  main()
