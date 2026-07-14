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
      --walker runs_dense/go2_walker_flat/final_model.zip \
      --safety runs_zoo/go2_gap_chain_avoid/final_model.zip \
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

from safety_sb3 import ReachAvoidPPO, SafetyPPO  # noqa: E402
from safe_mjlab_zoo import spec  # noqa: E402

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
  from safe_mjlab_zoo.envs.velocity.go2 import unitree_go2_flat_env_cfg
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


def load_safety(zip_path: str, device: str):
  try:
    model = ReachAvoidPPO.load(zip_path, device=device)
  except Exception:
    model = SafetyPPO.load(zip_path, device=device)
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


# --- filter ------------------------------------------------------------------

class BatchValueFilter:
  """Latched eps-switch with median-smoothed V + a CAUTION band (identical
  protocol for both twins).

  Caution band (the old play_filtered mechanism, batched): while
  eps < V <= caution, the WALKER'S COMMAND is zeroed — the walker decelerates
  itself (an in-distribution slowdown), so if V keeps dropping the handover
  happens from a braking stance instead of mid-trot, and after a save the
  released walker doesn't sprint straight back into re-engagement chatter."""

  def __init__(self, n, device, eps, caution=0.45, hysteresis=0.15,
               rest_speed=0.4, disabled=False):
    self.eps, self.caution = eps, max(caution, eps)
    self.hys, self.rest = hysteresis, rest_speed
    self.disabled = disabled
    self.engaged = torch.zeros(n, dtype=torch.bool, device=device)
    self.v_hist = None
    self.overrides = torch.zeros(n, device=device)   # engaged-step counter
    self.cautions = torch.zeros(n, device=device)

  def update(self, v_raw, speed, fresh):
    if self.v_hist is None:
      self.v_hist = v_raw.unsqueeze(0).repeat(5, 1)
    self.v_hist = torch.cat([self.v_hist[1:], v_raw.unsqueeze(0)], dim=0)
    if bool(fresh.any()):
      self.v_hist[:, fresh] = v_raw[fresh].unsqueeze(0)
      self.engaged &= ~fresh
    v_med = self.v_hist.median(dim=0).values
    engage = (v_med <= self.eps) | (v_raw <= self.eps - 0.15)
    release = (v_med > self.eps + self.hys) & (speed < self.rest)
    self.engaged = (self.engaged | engage) & ~release
    caution = (v_med <= self.caution) & ~self.engaged
    if self.disabled:
      self.engaged = torch.zeros_like(self.engaged)
      caution = torch.zeros_like(caution)
    self.overrides += self.engaged.float()
    self.cautions += caution.float()
    return self.engaged, caution


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
    from safe_mjlab_zoo.tasks.go2_gap import _load_vhat
    hpol, hnorm = load_safety(
      os.path.join(args.hybrid_skill, "final_model.zip"), device)
    vmlp, vmean, vvar, p_star = _load_vhat(device)
    hyb = dict(pol=hpol, norm=hnorm, vmlp=vmlp, vmean=vmean, vvar=vvar,
               p_star=p_star,
               latch=torch.zeros(args.num_envs, dtype=torch.bool, device=device))
    hyb_steps = 0
  filt = BatchValueFilter(args.num_envs, device, args.eps, caution=args.caution,
                          hysteresis=args.hysteresis, disabled=args.no_filter)

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
    # safety value + fallback (tensor path)
    s_obs = snorm(obs_dict["proprioception"].float())
    with torch.no_grad():
      v = safety.policy.predict_values(s_obs).squeeze(-1)
      a_safe = torch.clamp(safety.policy._predict(s_obs, deterministic=True),
                           -1.0, 1.0)
    speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
    engaged, caution = filt.update(v, speed, fresh=prev_done)
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
    action = torch.where(engaged.unsqueeze(-1), a_safe, a_walk)
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
      from safe_mjlab_zoo.tasks.go2_gap import l_certified_launch
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
      filt.engaged &= ~d
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
    intervention_rate=float(filt.overrides.sum() / (n * args.steps)),
    caution_rate=float(filt.cautions.sum() / (n * args.steps)),
    hybrid_rate=(hyb_steps / (n * args.steps)) if hyb is not None else 0.0,
    gap_width=args.gap_width, n_gaps=args.n_gaps, eps=args.eps,
    filter="off" if args.no_filter else (os.path.basename(
      os.path.dirname(args.safety)) + ("+funnel" if hyb is not None else "")),
  )
  print("\n=== FILTER GAUNTLET ===")
  for k, v in summary.items():
    print(f"  {k:22s} {v}")
  if args.out:
    with open(args.out, "w") as f:
      json.dump(summary, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
  main()
