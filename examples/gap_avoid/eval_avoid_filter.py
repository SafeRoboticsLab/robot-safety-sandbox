#!/usr/bin/env python
"""Evaluate the avoid-only gap policy as a safety filter over the flat walker.

Setup (the single-gap venue of the sRAS / Q-CBF paper, avoid-only row):

* World: a 0.30 m gap on a long island, 20 s episodes. Each robot starts
  standing on the near platform at a distance d in [0.35, 1.6] m from the gap
  face, a lateral offset y in [-0.35, 0.35] m and a heading in [-25, 25] deg.
  The 1000 starts are fixed (``states/init_states.npz``), so every run, and
  every method, uses the same starting states.
* Task policy: the blind flat walker (``go2_walker_flat``), commanded straight
  ahead at 1.0 m/s. It cannot see the gap and walks into it if nothing stops it.
* Safety policy: the avoid-only policy (``go2_gap_brake_or_jump_avoid_w30``,
  SafetyPPO1P, safety margin g only, no reach target).
* Filter (default ``qcbf``): the optimal-stopping Q-CBF of the paper. At every
  step before it commits, it rolls the safety policy forward in a copy of the
  simulator ("shadow") to resolution, twice:

      J0 = min_t g   if the safety policy takes over NOW
      J1 = min_t g   if the walker takes one more step, then the safety policy

  and commits (hands control to the safety policy for the rest of the episode)
  as soon as  J0 >= J1 - delta_c  and  J0 >= theta_min. Until then the walker
  drives. For the avoid-only policy, committing means braking and standing.

Margins: ``strict`` keeps the illegal-contact termination (any non-foot body
touching the ground ends the episode as a failure); ``belly`` removes it (only
falls and dropping into the gap count). The paper reports strict.

Outcome per start (first episode only):
  safe     = the episode never ended in a failure (standing still is safe)
  crossed  = the base got past the far rest line (gap width + 0.15 m)
  success  = crossed and safe

Expected for the released avoid policy, strict, n=1000 (paper):
  cross 0.008 / safe 0.856 / success 0.007   (the filter brakes; it almost never crosses)
GPU simulation is not bit-reproducible; expect about +-0.02 between runs.

  MUJOCO_GL=egl python examples/gap_avoid/eval_avoid_filter.py              # paper setting, ~15 min
  MUJOCO_GL=egl python examples/gap_avoid/eval_avoid_filter.py --num-envs 32 --video /tmp/avoid.mp4
  MUJOCO_GL=egl python examples/gap_avoid/eval_avoid_filter.py --filter value --eps 0.0
  MUJOCO_GL=egl python examples/gap_avoid/eval_avoid_filter.py --filter none   # the walker alone
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
  sys.path.insert(0, REPO)

AVOID_TASK = "go2_gap_brake_or_jump_avoid_w30"
DEFAULT_POLICY = os.path.join(HERE, "weights", "avoid_w30", "final_model.zip")
DEFAULT_WALKER = os.path.join(HERE, "weights", "go2_walker_flat", "final_model.zip")
DEFAULT_INITS = os.path.join(HERE, "states", "init_states.npz")

GAP_X = 0.0            # x of the gap face, relative to each env's origin
ISLAND_LENGTH = 3.0    # long enough for a 1.6 m approach
EPISODE_S = 20.0
CMD_VX = 1.0           # the walker's forward command


# --- starts: fixed distance / lateral offset / heading per env ----------------

def make_reset(inits: np.ndarray, gap_x: float = GAP_X):
  """Reset event: env i starts standing at ``inits[i] = (d, y, yaw)``."""
  import torch
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

  def reset_fixed_starts(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    if len(env_ids) == 0:
      return
    asset = env.scene[asset_cfg.name]
    dev = env.device
    n = int(len(env_ids))
    root = asset.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]
    idx = env_ids.detach().cpu().numpy().astype(int)
    ds = torch.as_tensor(inits[idx], device=dev, dtype=torch.float32)
    d, y, yaw = ds[:, 0], ds[:, 1], ds[:, 2]
    pos = torch.empty(n, 3, device=dev)
    pos[:, 0] = origins[:, 0] + (gap_x - d)
    pos[:, 1] = origins[:, 1] + y
    pos[:, 2] = origins[:, 2] + root[:, 2]
    z0 = torch.zeros(n, device=dev)
    quat = quat_mul(root[:, 3:7], quat_from_euler_xyz(z0, z0, yaw))
    asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
    jp = asset.data.default_joint_pos[env_ids].clone()
    asset.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)

  return reset_fixed_starts


def make_cfg_transform(inits, gap_width: float, belly: bool):
  """The gap_gauntlet world (one gap, no curriculum) with the fixed starts."""
  from mjlab.managers.event_manager import EventTermCfg
  from robot_safety_sandbox.envs.go2_gap.eval_gauntlet import gap_cfg_transform
  base = gap_cfg_transform(gap_width=gap_width, n_gaps=1, episode_s=EPISODE_S,
                           cmd_vx=CMD_VX, island_length=ISLAND_LENGTH, full_pose=True)
  reset_fn = make_reset(inits)

  def transform(cfg):
    cfg = base(cfg)
    cfg.events["reset_base"] = EventTermCfg(func=reset_fn, mode="reset", params={})
    if belly:
      cfg.terminations.pop("illegal_contact", None)
    return cfg

  return transform


# --- first-episode recorder ----------------------------------------------------

class FirstEpisodeRecorder:
  """Each env's FIRST episode: base x/y path, engaged flag, and its outcome.

  mjlab resets inside ``step``, so on a done step the robot already holds the
  next start. The recorder therefore appends the previous step's position and
  closes an env at its first done; no post-reset state enters a trajectory.
  Positions are relative to the env origin, so x = 0 is the gap face.
  """

  def __init__(self, env, *, rest_x: float, gap_width: float, filter_active: bool):
    self.n = int(env.num_envs)
    self.rest_x, self.gap_width = float(rest_x), float(gap_width)
    self.filter_active = bool(filter_active)
    self.buf_x = [[] for _ in range(self.n)]
    self.buf_y = [[] for _ in range(self.n)]
    self.buf_e = [[] for _ in range(self.n)]
    self.closed = np.zeros(self.n, dtype=bool)
    self.terminated = np.zeros(self.n, dtype=bool)
    self.max_x = np.full(self.n, -1e9, dtype=np.float32)
    self._pending = None

  @staticmethod
  def _read(env):
    d = env.robot.data
    ox = env.mj.scene.env_origins
    x = (d.root_link_pos_w[:, 0] - ox[:, 0]).detach().cpu().numpy()
    y = (d.root_link_pos_w[:, 1] - ox[:, 1]).detach().cpu().numpy()
    return x, y

  def update(self, rec):
    env = rec.env
    x, y = self._read(env)
    eng = rec.info.engaged
    if eng is None or not self.filter_active:
      eng = np.zeros(self.n, dtype=bool)
    else:
      eng = eng.detach().cpu().numpy().astype(bool)
    done = rec.out.done.detach().cpu().numpy().astype(bool)
    term = rec.out.terminated.detach().cpu().numpy().astype(bool)
    if self._pending is not None:
      px, py = self._pending
      openm = ~self.closed
      for j in np.nonzero(openm)[0]:
        self.buf_x[j].append(px[j])
        self.buf_y[j].append(py[j])
        self.buf_e[j].append(eng[j])
      self.max_x = np.where(openm, np.maximum(self.max_x, px), self.max_x)
      d_open = done & openm
      self.terminated |= term & d_open
      self.closed |= d_open
    self._pending = (x, y)

  def result(self) -> dict:      # run_eval calls this; the data comes from arrays()
    return {}

  def arrays(self) -> dict:
    alive_len = np.array([len(b) for b in self.buf_x], dtype=np.int64)
    T = max(int(alive_len.max()) if self.n else 1, 1)
    X = np.full((self.n, T), np.nan, dtype=np.float32)
    Y = np.full((self.n, T), np.nan, dtype=np.float32)
    E = np.zeros((self.n, T), dtype=bool)
    for j in range(self.n):
      L = alive_len[j]
      if L:
        X[j, :L], Y[j, :L], E[j, :L] = self.buf_x[j], self.buf_y[j], self.buf_e[j]
    crossed = self.max_x > self.rest_x
    safe = ~self.terminated
    last_x = np.array([b[-1] if b else np.nan for b in self.buf_x], dtype=np.float32)
    at_gap = (last_x > GAP_X - 0.15) & (last_x < GAP_X + self.gap_width + 0.15)
    cause = np.where(~self.terminated, "none", np.where(at_gap, "gapfall", "elsewhere"))
    return dict(x=X, y=Y, engaged=E, alive_len=alive_len, crossed=crossed, safe=safe,
                success=crossed & safe, terminated=self.terminated, max_x=self.max_x,
                last_x=last_x, cause=cause)


# --- the optimal-stopping Q-CBF ----------------------------------------------------

def build_shadow(env, mods, task: str):
  """A shadow copy of the live world that scores rollouts with the POLICY's own
  training margins (``task``), so g in the shadow is the g the policy learned."""
  from robot_safety_sandbox.base import MjlabTensorSafetyEnv
  from robot_safety_sandbox.filters import MjlabShadowSim
  from robot_safety_sandbox.registry import spec
  s = spec(task)
  bridge = MjlabTensorSafetyEnv(
    env.num_envs, env.device, cfg_builder=env.cfg_builder, margin_fn=s.margin_fn,
    ctrl_dim=s.ctrl_dim, dstb_dim=s.dstb_dim, adversary=False,
    end_criterion=s.end_criterion, obs_key=env.safety_obs_key, **s.kwargs)
  shadow = MjlabShadowSim(
    env.mj, bridge, num_envs=env.num_envs, rollouts_per_env=1,
    obs_adapter=lambda obs: {"s_obs": mods["norm"](obs.float())},
    contact_history="sync")
  return bridge, shadow


def roll_to_resolution(shadow, first_action, policy_fn, margin_fn, reach_avoid, cap=120):
  """Copy the live state into the shadow, apply ``first_action`` for one step,
  then let the safety policy drive for up to ``cap`` steps. Returns
  J = min_t g (avoid), or min(min_t g, l at the end) (reach-avoid).
  The live env is not advanced."""
  import torch
  shadow_env = shadow._shadow
  with torch.no_grad():
    shadow.seed()
    g0, _ = margin_fn(shadow_env)
    gmin = g0.float()
    step = shadow.step(first_action)
    gmin = torch.minimum(gmin, step.g.float())
    live = ~step.done.bool()
    last_l = step.l.float()
    a = policy_fn(**step.ctx)
    for _ in range(cap - 1):
      if not bool(live.any()):
        break
      step = shadow.step(a)
      gmin = torch.where(live, torch.minimum(gmin, step.g.float()), gmin)
      new_live = live & ~step.done.bool()
      last_l = torch.where(new_live, step.l.float(), last_l)
      live = new_live
      a = policy_fn(**step.ctx)
    return torch.minimum(gmin, last_l) if reach_avoid else gmin


def make_stopping_qcbf(env, mods, shadow, task: str, delta_c: float, theta_min: float,
                       cap: int):
  """SafetyFilter = safety-policy fallback + optimal-stopping Q-CBF latch."""
  import torch
  from robot_safety_sandbox.filters.core import SafetyFilter
  from robot_safety_sandbox.filters.fallback import PolicyFallback
  from robot_safety_sandbox.filters.intervention import Intervention
  from robot_safety_sandbox.filters.monitor import CriticMonitor
  from robot_safety_sandbox.filters.telemetry import FilterInfo
  from robot_safety_sandbox.registry import REACH_AVOID, spec

  s = spec(task)
  n, device = env.num_envs, env.device

  class StoppingQCBF(Intervention):
    """Commit (latch the safety policy) iff J0 >= J1 - delta_c and J0 >= theta_min.
    This is the Q-CBF condition with kappa = 1, Q(x, u_walker) = J1 and
    Q(x, u_safe) = J0, read as an optimal-stopping rule."""

    def __init__(self):
      super().__init__(n, device, None)
      self.latched = torch.zeros(n, dtype=torch.bool, device=device)
      self.last_trigger = torch.zeros(n, dtype=torch.bool, device=device)
      self.j0_at_commit = torch.full((n,), float("nan"), device=device)

    def __call__(self, a_nom, fallback, monitor, *, speed=None, fresh=None, **ctx):
      del speed, fresh, monitor
      a_safe = fallback(**ctx)
      if bool(self.latched.all()):
        self.last_trigger = torch.zeros_like(self.latched)
        return a_safe, FilterInfo(engaged=self.latched, value=torch.zeros(n, device=device))
      ra = s.mode == REACH_AVOID
      J0 = roll_to_resolution(shadow, a_safe, mods["fallback_fn"], s.margin_fn, ra, cap)
      J1 = roll_to_resolution(shadow, a_nom, mods["fallback_fn"], s.margin_fn, ra, cap)
      commit = (~self.latched) & (J0 >= J1 - delta_c) & (J0 >= theta_min)
      self.last_trigger = commit
      self.j0_at_commit = torch.where(commit, J0, self.j0_at_commit)
      self.latched = self.latched | commit
      action = torch.where(self.latched.unsqueeze(-1), a_safe, a_nom)
      return action, FilterInfo(engaged=self.latched, value=J0 - J1)

    def reset(self, done):
      self.latched &= ~done
      self.last_trigger &= ~done

  interv = StoppingQCBF()
  monitor = CriticMonitor(n, device, lambda action, **_: torch.zeros(n, device=device))
  return SafetyFilter(PolicyFallback(n, device, mods["fallback_fn"]), monitor, interv), interv


# --- one evaluation ----------------------------------------------------------------

def evaluate(args) -> dict:
  import torch
  from robot_safety_sandbox.eval import (NominalPolicy, SwitchCfg, TwistCommandSurgery,
                                         VideoRecorder, build_eval_env, build_filter,
                                         load_nominal, load_twin, run_eval, safety_modules)

  inits = np.load(args.inits)["inits"].astype(np.float32)
  if args.num_envs > len(inits):
    raise SystemExit(f"--num-envs {args.num_envs} > {len(inits)} starts in {args.inits}")
  inits = inits[:args.num_envs]
  rest_x = args.gap_width + 0.15
  dev = args.device

  env = build_eval_env(args.task, args.num_envs, dev,
                       cfg_transform=make_cfg_transform(inits, args.gap_width, args.margin == "belly"),
                       render_mode="rgb_array" if args.video else None)
  env.mj.cfg.episode_length_s = EPISODE_S
  terms = tuple(getattr(env.mj.termination_manager, "active_terms", ()))
  assert ("illegal_contact" in terms) == (args.margin == "strict"), terms
  print(f"[eval] task={args.task} n={args.num_envs} margin={args.margin} "
        f"filter={args.filter} terminations={terms}", flush=True)

  model, norm = load_twin(args.policy, dev)
  mods = safety_modules(model, args.num_envs, dev)
  mods["norm"] = norm

  interv = None
  if args.filter == "qcbf":
    _, shadow = build_shadow(env, mods, args.task)
    filt, interv = make_stopping_qcbf(env, mods, shadow, args.task,
                                      args.delta_c, args.theta_min, args.cap)
  else:  # "value" (switch to the safety policy while V(x) <= eps) or "none"
    filt = build_filter("value", mods, env, switch=SwitchCfg(eps=args.eps)).filt

  walker = NominalPolicy(*load_nominal(args.walker, dev), device=dev)
  recorder = FirstEpisodeRecorder(env, rest_x=rest_x, gap_width=args.gap_width,
                                  filter_active=args.filter != "none")
  surgery = TwistCommandSurgery(cmd_vx=CMD_VX, engaged_cmd_vx=CMD_VX).bind(env)
  if not surgery.available:
    surgery = None

  # per-env first-episode termination cause + where the filter committed
  tmgr = env.mj.termination_manager
  term_names = [t for t in tmgr.active_terms if t != "time_out"]
  cause = {t: torch.zeros(args.num_envs, dtype=torch.bool, device=dev) for t in term_names}
  closed = torch.zeros(args.num_envs, dtype=torch.bool, device=dev)
  commit_x = torch.full((args.num_envs,), float("nan"), device=dev)
  base_update = recorder.update

  def update(rec):
    nonlocal closed, commit_x
    openm = ~closed
    if interv is not None:
      new = interv.last_trigger & openm & torch.isnan(commit_x)
      if bool(new.any()):
        xr = env.robot.data.root_link_pos_w[:, 0] - env.mj.scene.env_origins[:, 0]
        commit_x = torch.where(new, xr, commit_x)
    done = rec.out.done.bool()
    open_done = done & ~closed
    if bool(open_done.any()):
      for t in term_names:
        try:
          cause[t] |= open_done & tmgr.get_term(t).bool()
        except Exception:
          pass
    closed |= done
    base_update(rec)

  recorder.update = update
  video = VideoRecorder(args.video) if args.video else None
  run_eval(env, walker, filt, recorder, steps=args.steps, norm=norm,
           no_filter=args.filter == "none", command_surgery=surgery, video=video,
           seed=args.seed, progress_every=max(1, args.steps // 5))
  if video is not None:
    video.save()

  data = recorder.arrays()
  data["commit_x"] = commit_x.detach().cpu().numpy()
  data["term_cause"] = {t: int(v.sum()) for t, v in cause.items()}
  env.close()
  del env, filt, model, walker, recorder, mods
  gc.collect()
  torch.cuda.empty_cache()
  return data


def summarize(d: dict) -> dict:
  n = int(len(d["safe"]))
  E, L = d["engaged"], d["alive_len"]
  interv = float(np.mean([E[i, :int(L[i])].mean() if int(L[i]) > 0 else 0.0
                          for i in range(n)]))
  cx = d["commit_x"]
  out = dict(
    n=n,
    cross_rate=float(d["crossed"].mean()),
    safe_rate=float(d["safe"].mean()),
    success_rate=float(d["success"].mean()),
    intervention_rate=interv,
    n_fail=int(d["terminated"].sum()),
    n_fail_at_gap=int((d["cause"] == "gapfall").sum()),
    n_fail_elsewhere=int((d["cause"] == "elsewhere").sum()),
    termination_causes=d["term_cause"],
  )
  if np.isfinite(cx).any():
    v = cx[np.isfinite(cx)]
    out["committed"] = int(len(v))
    out["commit_x_p10_p50_p90"] = [float(np.percentile(v, q)) for q in (10, 50, 90)]
  return out


def wilson(p: float, n: int, z: float = 1.96):
  if n == 0:
    return (float("nan"), float("nan"))
  den = 1 + z * z / n
  c = (p + z * z / (2 * n)) / den
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return (c - h, c + h)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--policy", default=DEFAULT_POLICY, help="safety policy (SB3 .zip; "
                  "tensornormalize.pt next to it)")
  ap.add_argument("--task", default=AVOID_TASK, help="the policy's training task (its margins "
                  "are used inside the filter)")
  ap.add_argument("--walker", default=DEFAULT_WALKER, help="task policy (stock SB3 PPO .zip; "
                  "vecnormalize.pkl next to it)")
  ap.add_argument("--inits", default=DEFAULT_INITS)
  ap.add_argument("--num-envs", type=int, default=1000)
  ap.add_argument("--steps", type=int, default=700, help="control steps (50 Hz); every start "
                  "is scored on its first episode")
  ap.add_argument("--margin", choices=["strict", "belly"], default="strict")
  ap.add_argument("--filter", choices=["qcbf", "value", "none"], default="qcbf",
                  help="qcbf = optimal-stopping Q-CBF (paper); value = switch to the safety "
                       "policy while V(x) <= eps; none = the walker alone")
  ap.add_argument("--delta-c", type=float, default=0.05, help="qcbf: commit tolerance")
  ap.add_argument("--theta-min", type=float, default=-0.2, help="qcbf: minimum J0 to commit")
  ap.add_argument("--cap", type=int, default=120, help="qcbf: shadow rollout length (steps)")
  ap.add_argument("--eps", type=float, default=0.0, help="value: switching threshold")
  ap.add_argument("--gap-width", type=float, default=0.30)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--video", default=None, help="write an mp4 of the batch (use a small "
                  "--num-envs)")
  ap.add_argument("--out", default="runs/gap_avoid_eval", help="output directory")
  args = ap.parse_args()

  os.makedirs(args.out, exist_ok=True)
  tag = f"{args.filter}_{args.margin}_n{args.num_envs}"
  d = evaluate(args)
  r = summarize(d)
  r["ci95"] = {k: wilson(r[k], r["n"]) for k in ("cross_rate", "safe_rate", "success_rate")}
  r["args"] = vars(args)
  causes = d.pop("term_cause")
  np.savez_compressed(os.path.join(args.out, f"{tag}.npz"), **d)
  with open(os.path.join(args.out, f"{tag}.json"), "w") as f:
    json.dump(r, f, indent=2)
  print(f"\n[result] {tag}: cross {r['cross_rate']:.3f}  safe {r['safe_rate']:.3f}  "
        f"success {r['success_rate']:.3f}  intervention {r['intervention_rate']:.3f}")
  print(f"[result] failures {r['n_fail']} (at the gap {r['n_fail_at_gap']}, elsewhere "
        f"{r['n_fail_elsewhere']}); termination causes {causes}")
  if "commit_x_p10_p50_p90" in r:
    p = r["commit_x_p10_p50_p90"]
    print(f"[result] committed {r['committed']}/{r['n']}; commit x p10/p50/p90 = "
          f"{p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f} m (gap face at 0)")
  print(f"[saved] {os.path.join(args.out, tag)}.{{json,npz}}")


if __name__ == "__main__":
  main()
