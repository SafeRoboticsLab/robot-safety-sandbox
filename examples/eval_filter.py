"""R-CBF claim gauntlet: nominal walker + value-based safety filter, batched.

Composes the blind dense-reward walker (stock SB3 PPO, numpy VecNormalize) with
a safety twin's certificate V(s) + fallback policy (safety_sb3, tensor norm):

  run the WALKER;  when V(s) <= eps  ->  execute the twin's own fallback
  (latched; release on V > eps+hyst AND near rest).

The SAME protocol wraps both twins -- only the certificate/fallback pair
changes (avoid-only vs reach-avoid completion). Predictions:

  avoid twin  : vetoes at the braking boundary, fallback stops -> livelock
                before gap 1 (safe, zero crossings).
  RA twin     : fallback carries the crossing when feasible; degrades to stop
                on uncrossable widths. Safe AND task flows.

  python examples/eval_filter.py \
      --walker runs/go2_walker_flat/final_model.zip \
      --safety runs/go2_gap_chain_avoid/final_model.zip \
      --gap-width 0.35 --n-gaps 1 --num-envs 256 --steps 600
  # add --no-filter for the walker-only baseline
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dataclasses import replace  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402

from robot_safety_sandbox import spec  # noqa: E402

CTRL_GAIN = 3.0        # bridge convention: policy action * gain -> env action
# terrain geometry is task-dependent -> CLI args (--gap-x/--rest-x/--spawn-x):
#   chain terrain: gap face 2.5, rest zone past 5.0, spawn 0.15..0.45
#   single-gap  : gap face 0.0, crossed past ~1.2, spawn -1.9..-1.5
REST_X = 5.0           # overwritten from args in main()
GAP_X = 2.5


# --- env surgery -------------------------------------------------------------

def _reset_standing(env, env_ids, x_lo=0.15, x_hi=0.45,
                    asset_cfg=SceneEntityCfg("robot")):
  """Standing spawn on the approach: the WALKER walks in naturally."""
  from mjlab.utils.lab_api.math import sample_uniform
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  asset = env.scene[asset_cfg.name]
  n = len(env_ids)
  root = asset.data.default_root_state[env_ids].clone()
  pos = root[:, 0:3] + env.scene.env_origins[env_ids]
  pos[:, 0] = env.scene.env_origins[env_ids, 0] + sample_uniform(
    x_lo, x_hi, (n,), env.device)
  asset.write_root_link_pose_to_sim(
    torch.cat([pos, root[:, 3:7]], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(
    torch.zeros(n, 6, device=env.device), env_ids=env_ids)


def build_filter_env_cfg(task: str, num_envs: int, gap_width: float,
                         n_gaps: int, episode_s: float, cmd_vx: float,
                         spawn_x=(0.15, 0.45), island_length=None):
  """Chain env emitting BOTH obs groups: safety 'proprioception' + walker
  'actor' (grafted from the flat walker cfg), with a standing spawn, a fixed
  forward command, and a PINNED gap width (no curriculum)."""
  cfg = spec(task).cfg_builder(play=True)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = episode_s
  cfg.curriculum = {}
  cfg.events["reset_base"] = EventTermCfg(
    func=_reset_standing, mode="reset",
    params={"x_lo": spawn_x[0], "x_hi": spawn_x[1]})
  cfg.events.pop("handover_joints", None)
  cfg.events.pop("randomize_terrain", None)
  cfg.events.pop("push_robot", None)

  # fixed forward command (both policies see the same, in-distribution value)
  twist = cfg.commands["twist"]
  twist.resampling_time_range = (1.0e9, 1.0e9)
  twist.ranges.lin_vel_x = (cmd_vx, cmd_vx)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  if hasattr(twist, "rel_standing_envs"):
    twist.rel_standing_envs = 0.0
  if hasattr(twist, "heading_command"):
    twist.heading_command = False

  # pin the gap width (feasible vs infeasible is a CLI knob, not a curriculum).
  # Terrain cfgs differ across task families -> only set fields that exist.
  import dataclasses
  gen = cfg.scene.terrain.terrain_generator
  subs = {}
  for k, v in gen.sub_terrains.items():
    names = {f.name for f in dataclasses.fields(v)}
    kw = {}
    if "gap_width_range" in names:
      kw["gap_width_range"] = (gap_width, gap_width)
    if "n_gaps_max" in names:
      kw["n_gaps_max"] = n_gaps
    if island_length is not None and "island_length" in names:
      # walk-in-viable eval variant: the native 0.7 m island cannot host a
      # walking approach; extending it keeps the SAME gap geometry (origin at
      # the gap face) while giving the walker a real approach corridor.
      kw["island_length"] = island_length
    subs[k] = replace(v, **kw) if kw else v
  cfg.scene.terrain.terrain_generator = replace(gen, sub_terrains=subs,
                                                curriculum=False)

  # graft the walker's exact blind actor group as 'actor'
  from robot_safety_sandbox.envs.velocity.go2 import unitree_go2_flat_env_cfg
  walk_cfg = unitree_go2_flat_env_cfg(play=True)
  walk_group = copy.deepcopy(walk_cfg.observations["actor"])
  assert "height_scan" not in walk_group.terms
  cfg.observations["actor"] = walk_group
  return cfg


# --- policy loading ----------------------------------------------------------

def load_walker(zip_path: str, device: str):
  from stable_baselines3 import PPO
  model = PPO.load(zip_path, device=device)
  vn_path = os.path.join(os.path.dirname(zip_path), "vecnormalize.pkl")
  if not os.path.exists(vn_path):
    cand = sorted([f for f in os.listdir(os.path.dirname(zip_path))
                   if f.startswith("vecnormalize")])
    vn_path = os.path.join(os.path.dirname(zip_path), cand[-1]) if cand else None
  vn = None
  if vn_path and os.path.exists(vn_path):
    with open(vn_path, "rb") as f:
      vn = pickle.load(f)
    vn.training = False
  print(f"[walker] {zip_path} (+ {os.path.basename(vn_path) if vn_path else 'NO NORM'})")
  return model, vn


def _serialized_class_name(entry) -> str:
  """Class NAME out of one SB3-serialized ``data`` entry (base64 cloudpickle)."""
  if not isinstance(entry, dict) or ":serialized:" not in entry:
    return ""
  try:
    import base64
    import cloudpickle
    obj = cloudpickle.loads(base64.b64decode(entry[":serialized:"]))
    return getattr(obj, "__name__", "")
  except Exception:
    return str(entry.get("__module__", ""))


def _zip_data(zip_path: str) -> dict:
  import zipfile
  with zipfile.ZipFile(zip_path) as z:
    return json.loads(z.read("data"))


def twin_class_name(zip_path: str) -> str:
  """Read the MAP cell (Mode + Algorithm + Players) out of the checkpoint.

  The learner class is not stored by SB3, but each of the three letters leaves a
  fingerprint that is, so the name is RECONSTRUCTED rather than guessed:

    A  SAC saves ``actor.optimizer.pth`` (PPO saves ``policy.optimizer.pth``)
    P  a two-player twin saves the disturbance actor's optimizer, and its data
       carries ``ctrl_action_dim`` (where the joint action splits)
    M  the buffer class: {ReachAvoid,Safety}{ReplayBuffer,RolloutBuffer}

  Returns "" when nothing safety_sb3-shaped is found (e.g. a stock SB3 zip), so
  the caller can fall back to trying candidates in order.
  """
  import zipfile
  with zipfile.ZipFile(zip_path) as z:
    names = set(z.namelist())
  data = _zip_data(zip_path)
  sac = any(n.startswith("actor.optimizer") for n in names)
  alg = "SAC" if sac else "PPO"
  two = ("ctrl_action_dim" in data
         or any(n.startswith("dstb") for n in names))
  buf = _serialized_class_name(
    data.get("replay_buffer_class" if sac else "rollout_buffer_class"))
  if not sac and not buf.startswith(("ReachAvoid", "Safety", "Tensor")):
    return ""                        # stock stable_baselines3 checkpoint
  mode = "ReachAvoid" if "ReachAvoid" in buf else "Safety"
  return f"{mode}{alg}{'2P' if two else '1P'}"


def load_safety(zip_path: str, device: str):
  """Load a safety twin — any of the eight MAP cells — plus its obs normalizer.

  Was PPO-1P only; the critic filters (Safety Critic, R-CBF) need a Q(s, a),
  which only the off-policy twins have, and the gameplay filter needs a two-
  player twin's disturbance actor. The class is resolved from the checkpoint
  (:func:`twin_class_name`) with the old try-in-order chain kept as a fallback.
  """
  import safety_sb3
  guess = twin_class_name(zip_path)
  data = _zip_data(zip_path)
  candidates = ([guess] if guess else []) + [
    "ReachAvoidPPO1P", "SafetyPPO1P", "ReachAvoidSAC1P", "SafetySAC1P",
    "ReachAvoidPPO2P", "SafetyPPO2P", "ReachAvoidSAC2P", "SafetySAC2P"]
  # Two-player learners take the joint-action split as a CONSTRUCTOR argument
  # (SB3 restores it into __dict__ too late for _setup_model), and the SAC
  # learners' tensor path builds a device-resident replay buffer sized off
  # self.env — which a checkpoint loaded for INFERENCE does not have. Override
  # both: no env, no buffer, actors and critic only.
  ctrl_dim = data.get("ctrl_action_dim")
  model, errors = None, []
  for name in dict.fromkeys(candidates):
    cls = getattr(safety_sb3, name, None)
    if cls is None:
      continue
    kw = {"ctrl_action_dim": int(ctrl_dim)} if (
      "2P" in name and ctrl_dim is not None) else {}
    custom = ({"_tensor_path": False, "buffer_size": 1} if "SAC" in name
              else None)
    try:
      model = cls.load(zip_path, device=device, custom_objects=custom, **kw)
      break
    except Exception as e:                       # noqa: BLE001 - report them all
      errors.append(f"{name}: {type(e).__name__}: {e}")
  if model is None:
    raise SystemExit(f"could not load safety twin {zip_path}:\n  "
                     + "\n  ".join(errors))
  pt = zip_path.replace("final_model.zip", "tensornormalize.pt")
  if not os.path.exists(pt):
    d = os.path.dirname(zip_path)
    cand = sorted([f for f in os.listdir(d) if f.startswith("tensornorm")])
    pt = os.path.join(d, cand[-1]) if cand else None
  assert pt and os.path.exists(pt), "safety obs-norm stats (.pt) not found"
  st = torch.load(pt, map_location=device, weights_only=True)
  mean, var = st["obs_mean"].to(device), st["obs_var"].to(device)
  print(f"[safety] {zip_path} ({type(model).__name__}) + {os.path.basename(pt)}")

  def norm(obs):
    return torch.clamp((obs - mean) / torch.sqrt(var + 1e-8), -10.0, 10.0)
  return model, norm


def safety_modules(model, num_envs: int, device: str):
  """Wire a loaded twin into the callables/modules the filter recipes take.

  Every key is keyed off what the twin ACTUALLY has, so an on-policy twin
  simply has no ``q_fn`` and a single-player one no ``dstb_fn`` — a caller that
  needs one gets a KeyError naming the twin, not a silent wrong number.

    fallback   PolicyFallback over pi^<        every twin
    value_fn   V(s)                            *PPO* directly; *SAC* as
                                               Q(s, pi^<(s)) — safe iff >= 0
    q_fn       Q(s, a), DIFFERENTIABLE in a    *SAC* only (CriticMonitor,
                                               QCBFIntervention)
    dstb_fn    pi_dstb(s)                      *2P* only (AdversarialRollout-
                                               Monitor's disturbance player)

  All of them take the NORMALIZED observation as ``s_obs`` and tolerate extra
  ctx kwargs, matching the ``**ctx`` convention in robot_safety_sandbox.filters.
  The twin's own bounds are respected: actions are clamped to [-1, 1], and the
  twin critic is reduced by MIN across the ensemble, which is the conservative
  reading under the zoo's "safe iff >= 0" convention (and the same reduction
  the SAC learners use to form their targets).
  """
  from robot_safety_sandbox.filters import PolicyFallback
  policy = model.policy
  policy.set_training_mode(False)
  out = {}

  if hasattr(policy, "predict_values"):                     # on-policy twin
    def value_fn(s_obs, **_):
      with torch.no_grad():
        return policy.predict_values(s_obs).squeeze(-1)
    out["value_fn"] = value_fn

    def fallback_fn(s_obs, **_):
      with torch.no_grad():
        return torch.clamp(policy._predict(s_obs, deterministic=True), -1., 1.)
  else:                                                     # off-policy twin
    def fallback_fn(s_obs, **_):
      with torch.no_grad():
        return torch.clamp(policy.actor(s_obs, deterministic=True), -1., 1.)

    dstb_actor = getattr(policy, "dstb_actor", None)

    def _joint(action, s_obs):
      """The critic's action argument. A 2P twin's critic is over the FULL
      concatenated [ctrl, dstb] action (TwoPlayerSACPolicy), so the ctrl action
      alone does not index it: append the disturbance the adversary would play.
      pi_dstb depends on s only, so d(Q)/d(a_ctrl) is unaffected."""
      if dstb_actor is None:
        return action
      return torch.cat([action, dstb_actor(s_obs, deterministic=True)], dim=-1)

    def q_fn(action, s_obs, **_):
      qs = policy.critic(s_obs, _joint(action, s_obs))
      return torch.cat(qs, dim=1).min(dim=1).values
    out["q_fn"] = q_fn

    def value_fn(s_obs, **_):
      with torch.no_grad():
        return q_fn(action=fallback_fn(s_obs), s_obs=s_obs)
    out["value_fn"] = value_fn

    if dstb_actor is not None:
      def dstb_fn(s_obs, **_):
        with torch.no_grad():
          return torch.clamp(dstb_actor(s_obs, deterministic=True), -1., 1.)
      out["dstb_fn"] = dstb_fn

  out["fallback_fn"] = fallback_fn
  out["fallback"] = PolicyFallback(num_envs, device, fallback_fn)
  out["twin"] = type(model).__name__
  return out


# --- filter ------------------------------------------------------------------
# The latched eps-switch + caution band lives in the library now, as the
# composition PolicyFallback + ValueMonitor + LeastRestrictiveIntervention
# (robot_safety_sandbox.filters.safety_value_filter); this script wires the
# twin's value head and fallback policy into it and keeps the walker-command
# surgery (caution -> zero command) at the call site, where the env lives.

#: --filter -> (recipe name, what the twin must supply). The five recipes differ
#: in ONE module each; nothing below branches on the filter beyond this table.
FILTERS = {
  "value":    "V(s) from an on-policy twin, latched switch",
  "critic":   "Q(s, u_nom) from a SAC twin, same latched switch",
  "qcbf":     "Q(s, u) from a SAC twin, minimal modification (R-CBF)",
  "rollout":  "simulate pi^< for H steps in a shadow env, latched switch",
  "gameplay": "the same rollout, played against the twin's dstb actor",
}


def build_shadow(args, live_env, num_envs: int, adversary: bool, obs_adapter):
  """A shadow sim over THIS run's env cfg (not the task's stock one).

  The gauntlet env is surgically modified (pinned gap width, standing spawn,
  grafted walker obs group — see build_filter_env_cfg), so the shadow has to be
  built from the SAME cfg builder rather than via ``make_tensor(task)``, or the
  rollout would certify a different world than the one being filtered.
  """
  from robot_safety_sandbox import MjlabTensorSafetyEnv, spec
  from robot_safety_sandbox.filters import MjlabShadowSim
  s = spec(args.task)

  def cfg_builder(play: bool = False, **_):
    return build_filter_env_cfg(args.task, num_envs, args.gap_width,
                                args.n_gaps, args.episode_s, args.cmd_vx,
                                spawn_x=tuple(args.spawn_x),
                                island_length=args.island_length)

  kw = dict(s.kwargs)
  kw.setdefault("ctrl_gain", CTRL_GAIN)
  bridge = MjlabTensorSafetyEnv(
    num_envs, args.device, cfg_builder=cfg_builder, margin_fn=s.margin_fn,
    ctrl_dim=s.ctrl_dim, dstb_dim=s.dstb_dim, adversary=adversary,
    end_criterion=s.end_criterion, obs_key="proprioception", **kw)
  return bridge, MjlabShadowSim(
    live_env, bridge, num_envs=args.num_envs, rollouts_per_env=args.rollouts,
    obs_adapter=obs_adapter, contact_history=args.contact_history)


def build_filter(args, mods, device, live_env):
  """Assemble the requested composition out of the twin's modules."""
  from robot_safety_sandbox import filters as F
  n = args.num_envs
  switch = dict(eps=args.eps, caution=args.caution, hysteresis=args.hysteresis)

  def need(key):
    if key not in mods:
      raise SystemExit(
        f"--filter {args.filter} needs '{key}', which the twin at --safety "
        f"({mods['twin']}) does not have. Q(s, a) comes from an off-policy "
        "twin ({Safety,ReachAvoid}SAC{1P,2P}); a disturbance actor, from a 2P "
        f"one. This twin supplies: {sorted(k for k in mods if k.endswith('_fn'))}.")
    return mods[key]

  if args.filter == "value":
    return F.safety_value_filter(n, device, need("value_fn"),
                                 mods["fallback_fn"], **switch), None
  if args.filter == "critic":
    return F.safety_critic_filter(n, device, need("q_fn"),
                                  mods["fallback_fn"], **switch), None
  if args.filter == "qcbf":
    return F.qcbf_filter(n, device, need("q_fn"), mods["fallback_fn"],
                         kappa=args.kappa), None
  adversarial = args.filter == "gameplay"
  bridge, shadow = build_shadow(
    args, live_env, args.num_envs * args.rollouts, adversarial,
    obs_adapter=lambda obs: {"s_obs": mods["norm"](obs.float())})
  if adversarial:
    filt = F.gameplay_filter(n, device, mods["fallback_fn"], shadow,
                             args.horizon, need("dstb_fn"),
                             reach_avoid=args.rollout_reach_avoid,
                             recertify_every=args.recertify_every, **switch)
  else:
    filt = F.rollout_filter(n, device, mods["fallback_fn"], shadow,
                            args.horizon,
                            reach_avoid=args.rollout_reach_avoid,
                            recertify_every=args.recertify_every, **switch)
  return filt, (bridge, shadow)


# --- main --------------------------------------------------------------------

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--walker", required=True)
  p.add_argument("--safety", required=True)
  p.add_argument("--task", default="go2_gap_chain_ra",
                 help="env-cfg source (twins share the env; either id works)")
  p.add_argument("--gap-width", type=float, default=0.35)
  p.add_argument("--n-gaps", type=int, default=1)
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--steps", type=int, default=600)
  p.add_argument("--episode-s", type=float, default=20.0)
  p.add_argument("--cmd-vx", type=float, default=1.0)
  p.add_argument("--gap-x", type=float, default=2.5,
                 help="x_rel of the gap face (chain: 2.5; single-gap: 0.0)")
  p.add_argument("--rest-x", type=float, default=5.0,
                 help="x_rel counted as CROSSED (chain: 5.0; single-gap: ~1.2)")
  p.add_argument("--spawn-x", type=float, nargs=2, default=(0.15, 0.45),
                 help="standing-spawn x_rel range (single-gap: -1.9 -1.5)")
  p.add_argument("--eps", type=float, default=0.0)
  p.add_argument("--caution", type=float, default=0.45,
                 help="V band (eps, caution]: zero the walker command "
                      "(walker-driven deceleration before any handover)")
  p.add_argument("--hysteresis", type=float, default=0.15)
  p.add_argument("--no-filter", action="store_true")
  p.add_argument("--island-length", type=float, default=None,
                 help="override island_length on island-type terrains "
                      "(walk-in-viable eval: 3.0)")
  p.add_argument("--hybrid-skill", default=None,
                 help="dir with a FROZEN skill (final_model.zip + "
                      "tensornormalize.pt): while the safety fallback is "
                      "engaged, envs whose certificate margin l_vhat>0 latch "
                      "to the frozen skill until episode end (funnel filter "
                      "second stage). Requires $VHAT_PATH (vhat_cross.pt).")
  p.add_argument("--filter", default="value", choices=sorted(FILTERS),
                 help="which COMPOSITION to deploy; they differ in one module "
                      "each: " + " | ".join(f"{k}: {v}"
                                            for k, v in FILTERS.items()))
  p.add_argument("--kappa", type=float, default=0.8,
                 help="--filter qcbf: class-K coefficient, Q(x,u) >= kappa V(x)")
  p.add_argument("--horizon", type=int, default=20,
                 help="--filter rollout/gameplay: H sim steps per certification "
                      "(1 nominal + H-1 fallback)")
  p.add_argument("--rollouts", type=int, default=1,
                 help="--filter rollout/gameplay: parallel rollouts per env; "
                      "the verdict is their MIN (one failure condemns)")
  p.add_argument("--recertify-every", type=int, default=1,
                 help="--filter rollout/gameplay: re-certify every k steps and "
                      "latch in between (amortizes the H sim steps)")
  p.add_argument("--rollout-reach-avoid", action="store_true",
                 help="--filter rollout/gameplay: score the rollout with the "
                      "reach-avoid reduction max_t min(l, min_s<=t g) instead "
                      "of avoid's min_t g")
  p.add_argument("--contact-history", default="sync",
                 choices=("sync", "instantaneous", "ignore"),
                 help="--filter rollout/gameplay: how the shadow sim gets the "
                      "contact-force history the margin reads "
                      "(see filters/rollout.py)")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=None, help="write metrics JSON here")
  args = p.parse_args()

  device = args.device
  global GAP_X, REST_X
  GAP_X, REST_X = args.gap_x, args.rest_x
  env_cfg = build_filter_env_cfg(args.task, args.num_envs, args.gap_width,
                                 args.n_gaps, args.episode_s, args.cmd_vx,
                                 spawn_x=tuple(args.spawn_x),
                                 island_length=args.island_length)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  walker, wvn = load_walker(args.walker, device)
  safety, snorm = load_safety(args.safety, device)

  # funnel second stage: frozen skill + calibrated certificate (l_vhat)
  hyb = None
  if args.hybrid_skill:
    from robot_safety_sandbox.tasks.go2_gap import _load_vhat
    hpol, hnorm = load_safety(
      os.path.join(args.hybrid_skill, "final_model.zip"), device)
    vmlp, vmean, vvar, p_star = _load_vhat(device)
    hyb = dict(pol=hpol, norm=hnorm, vmlp=vmlp, vmean=vmean, vvar=vvar,
               p_star=p_star,
               latch=torch.zeros(args.num_envs, dtype=torch.bool, device=device))
    hyb_steps = 0
  mods = safety_modules(safety, args.num_envs, device)
  mods["norm"] = snorm
  filt, shadow_pair = build_filter(args, mods, device, env)
  print(f"[filter] {args.filter}: {FILTERS[args.filter]}")

  robot = env.scene["robot"]
  origin_x = env.scene.env_origins[:, 0]
  n = args.num_envs

  # per-episode bookkeeping (aggregate over all episodes seen in the run)
  ep_crossed = torch.zeros(n, dtype=torch.bool, device=device)
  ep_engaged = torch.zeros(n, dtype=torch.bool, device=device)
  ep_steps = torch.zeros(n, device=device)
  ep_max_x = torch.zeros(n, device=device)
  tot = dict(episodes=0, violations=0, crossings=0, timeouts_before_gap=0,
             timeouts_past_gap=0, engaged_episodes=0)
  fin_len, fin_maxx = [], []
  v_min_trace = []

  obs_dict, _ = env.reset()
  prev_done = torch.ones(n, dtype=torch.bool, device=device)  # t=0 is fresh
  for t in range(args.steps):
    # walker action (numpy path)
    w_obs = obs_dict["actor"].detach().cpu().numpy()
    if wvn is not None:
      w_obs = wvn.normalize_obs(w_obs)
    a_walk, _ = walker.predict(w_obs, deterministic=True)
    a_walk = torch.as_tensor(np.clip(a_walk, -1, 1), dtype=torch.float32,
                             device=device)
    # safety value + fallback (tensor path) via the library filter
    s_obs = snorm(obs_dict["proprioception"].float())
    speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
    action, finfo = filt(a_walk, speed=speed, fresh=prev_done, s_obs=s_obs)
    engaged, v = finfo.engaged, finfo.value
    # QCBFIntervention modifies rather than switches, so it reports no caution
    # band; the walker-command surgery below then simply never fires.
    caution = (finfo.caution if finfo.caution is not None
               else torch.zeros_like(engaged))
    if args.no_filter:
      if hasattr(filt.intervention, "engaged"):
        filt.intervention.engaged.zero_()
      filt.telemetry.reset()
      action = a_walk
      engaged = torch.zeros_like(engaged)
      caution = torch.zeros_like(caution)
    # caution band: zero the walker's COMMAND (walker-driven deceleration);
    # restored to cmd_vx otherwise. Written into the live command buffer so
    # both policies' command obs stay consistent with what the walker does.
    cmd = env.command_manager.get_command("twist")
    # caution: zero the walker's command (slow down, walker still acting).
    # engaged: the SAFETY policy acts and was trained under the env's
    # constant forward command (1.0) — feeding it cmd=0 is OOD and turns the
    # traverse fallback into a braker (2026-07-11: hybrid_rate 0.0 cell).
    cmd[:, 0] = torch.where(
      engaged, torch.ones_like(speed),
      torch.where(caution, torch.zeros_like(speed),
                  torch.full_like(speed, args.cmd_vx)))
    if hyb is not None:
      # certificate on the CURRENT obs (frozen normalizer baked into vhat)
      raw = obs_dict["proprioception"].float()
      with torch.no_grad():
        nv = torch.clamp((raw - hyb["vmean"]) / torch.sqrt(hyb["vvar"] + 1e-8),
                         -10.0, 10.0)
        p_hat = torch.sigmoid(hyb["vmlp"](nv).squeeze(-1))
        a_frozen = torch.clamp(
          hyb["pol"].policy._predict(hyb["norm"](raw), deterministic=True),
          -1.0, 1.0)
      # v9 semantics: the latch condition is the TASK's reach margin
      # (state-only certified-launch: airborne AND momentum AND V_hat_land),
      # NOT the funnel-era obs-feature certificate above (proven OOD at
      # handover states; kept only for logging).
      from robot_safety_sandbox.tasks.go2_gap import l_certified_launch
      l_v = l_certified_launch(env.unwrapped)
      # second-stage switch: fallback engaged AND certified -> frozen skill,
      # latched to episode end (the maneuver is committed; no mid-flight
      # handbacks).
      hyb["latch"] |= engaged & (l_v > 0.0)
      action = torch.where(hyb["latch"].unsqueeze(-1), a_frozen, action)
      hyb_steps += int(hyb["latch"].sum())

    obs_dict, _r, terminated, truncated, _extras = env.step(action * CTRL_GAIN)

    x_rel = robot.data.root_link_pos_w[:, 0] - origin_x
    ep_crossed |= x_rel > REST_X
    ep_engaged |= engaged
    ep_steps += 1
    ep_max_x = torch.maximum(ep_max_x, x_rel)
    v_min_trace.append(float(v.min()))

    done = terminated | truncated
    prev_done = done.clone()
    if hyb is not None:
      # a latch is an episode-scoped commitment — MUST clear on reset (the
      # 2026-07-11 gauntlet bug: uncleared latches left the lander driving
      # walker episodes forever -> hybrid_rate 0.87, livelock 77%).
      hyb["latch"] &= ~done
    if bool(done.any()):
      d = done
      tot["episodes"] += int(d.sum())
      tot["violations"] += int((terminated & d).sum())
      tot["crossings"] += int((ep_crossed & d).sum())
      before = truncated & ~ep_crossed & (x_rel < GAP_X + 0.2)
      tot["timeouts_before_gap"] += int(before.sum())      # livelock signature
      tot["timeouts_past_gap"] += int((truncated & ep_crossed).sum())
      tot["engaged_episodes"] += int((ep_engaged & d).sum())
      fin_len.append(ep_steps[d].clone())
      fin_maxx.append(ep_max_x[d].clone())
      ep_crossed &= ~d
      ep_engaged &= ~d
      ep_steps[d] = 0
      ep_max_x[d] = 0
      filt.reset(d)
      if hyb is not None:
        hyb["latch"] &= ~d

  # censored survivors: alive at eval end without completing an episode in the
  # window — with a filter engaged these are LIVELOCKED-SAFE robots, and
  # omitting them was silently inflating violation_rate.
  alive = ep_steps > 0
  tot["censored_alive"] = int(alive.sum())
  tot["censored_alive_before_gap"] = int(
    (alive & ((robot.data.root_link_pos_w[:, 0] - origin_x) < GAP_X + 0.2)).sum())
  ep = max(tot["episodes"] + tot["censored_alive"], 1)
  lens = torch.cat(fin_len) if fin_len else torch.zeros(1)
  maxx = torch.cat(fin_maxx) if fin_maxx else torch.zeros(1)
  summary = dict(
    **tot,
    ep_len_mean=float(lens.mean()),
    max_x_mean=float(maxx.mean()),      # how far robots GET (gap face at 2.5)
    max_x_p90=float(maxx.quantile(0.9)),
    v_mean_overall=float(sum(v_min_trace) / max(len(v_min_trace), 1)),
    violation_rate=tot["violations"] / ep,
    crossing_rate=tot["crossings"] / ep,
    livelock_rate=(tot["timeouts_before_gap"]
                   + tot["censored_alive_before_gap"]) / ep,
    intervention_rate=filt.telemetry.intervention_rate(args.steps),
    caution_rate=filt.telemetry.caution_rate(args.steps),
    hybrid_rate=(hyb_steps / (n * args.steps)) if hyb is not None else 0.0,
    gap_width=args.gap_width, n_gaps=args.n_gaps, eps=args.eps,
    composition=args.filter,
    rollouts=(args.rollouts if args.filter in ("rollout", "gameplay") else 0),
    filter="off" if args.no_filter else (os.path.basename(
      os.path.dirname(args.safety)) + ("+funnel" if hyb is not None else "")),
  )
  if shadow_pair is not None:
    summary["certifications"] = shadow_pair[1].seeds
  print("\n=== FILTER GAUNTLET ===")
  for k, v in summary.items():
    print(f"  {k:22s} {v}")
  if args.out:
    with open(args.out, "w") as f:
      json.dump(summary, f, indent=2)
    print(f"[saved] {args.out}")
  if shadow_pair is not None:
    shadow_pair[0].close()


if __name__ == "__main__":
  main()
