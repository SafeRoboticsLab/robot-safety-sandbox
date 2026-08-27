"""V_safelanding margin (the safelanding certificate) for RAAS v2.

Loads the FROZEN state-only land-value net + conformal p* and exposes a margin

    l_vhat(env) = V_safelanding(state_features(env)) - p*      (>=0  <=>  s in S_safelanding)

used three ways in the ablation (all sharing this one function):
  - Run 1 (RA)     : the REACH target l   -> compose(g_terrain_relative, l_vhat)
  - Run 3 (avoid2) : the AVOID margin g   -> compose(l_vhat)  (avoid leaving S_safelanding)
  - all runs       : the handover GUARD   -> latch_margin_fn=l_vhat (latch to pi_safelanding when >0)

Net path resolves from $RAAS_VSL_PATH (default: the vsafelanding artifact). Loaded once,
cached per (path, device); evaluated every step on the reward hook (small MLP, GPU).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from .features import state_features

_DEFAULT_PATH = os.path.expanduser(
  "~/artifacts/robot-safety-sandbox/vsafelanding/v_safelanding.pt")
_CACHE: dict = {}


def _load(device):
  path = os.environ.get("RAAS_VSL_PATH", _DEFAULT_PATH)
  key = (path, str(device))
  if key in _CACHE:
    return _CACHE[key]
  ck = torch.load(path, map_location=device, weights_only=False)
  d = ck["arch"]["d_in"]
  layers = []
  for h in ck["arch"]["hidden"]:
    layers += [nn.Linear(d, h), nn.ReLU()]
    d = h
  layers += [nn.Linear(d, 1)]
  net = nn.Sequential(*layers)
  # saved under the MLP wrapper (keys 'net.0.*'); strip the one-level prefix.
  net.load_state_dict({k.replace("net.", "", 1): v for k, v in ck["state_dict"].items()})
  net.to(device).eval()
  for p in net.parameters():
    p.requires_grad_(False)
  out = {
    "net": net, "mean": ck["feat_mean"].to(device), "std": ck["feat_std"].to(device),
    "pstar": float(ck["p_star"]),                                  # GUARD threshold (sigmoid)
    "logit_pstar_reward": float(ck.get("logit_pstar_reward", 0.0)),# REWARD zero-set (logit)
    "reward_scale": float(ck.get("reward_scale", 1.0)),
  }
  _CACHE[key] = out
  print(f"[l_vhat] V_safelanding <- {path} (guard p*={out['pstar']:.3f}, "
        f"reward p*={ck.get('p_star_reward', float('nan'))}, scale={out['reward_scale']:.4f}, "
        f"AUC~{float(ck.get('test_auc', float('nan'))):.3f})")
  return out


def geom_landable_gate(mj, xr_min: float = -0.06, vx_min: float = 1.23):
  """GEOMETRIC handover gate (RAS phase-2 B+) — the calibrated replacement
  for the OOD-dead V_safelanding guard as the hybrid ``latch_margin_fn``.

  Fires (>=0) iff the base is past the near edge AND carries forward momentum::

      geom = min(x_rel - xr_min, vx - vx_min)   >=0  <=>  x_rel>=xr_min AND vx>=vx_min

  ``x_rel`` is the near-edge-origin frame used everywhere on the gap tasks
  (``root_link_pos_w[:,0] - env_origins[:,0]``, matching ``l_stable_far``/``far_x``
  in ``envs/go2_gap/brake_or_jump.py``). Calibrated offline on the brake_or_jump
  transfer set (held-out land-coverage 1.00, precision 0.93; fires 1.00 on
  airborne committed jumps, 0.00 on slow-grounded pre-launch). This is a
  geometric PROPOSAL margin for the latch only — NEVER a reward term."""
  d = mj.scene["robot"].data
  x_rel = d.root_link_pos_w[:, 0] - mj.scene.env_origins[:, 0]
  vx = d.root_link_lin_vel_w[:, 0]
  return torch.minimum(x_rel - xr_min, vx - vx_min)


def l_vhat(env, clamp: float = 3.0):
  """GUARD margin (`sigmoid(net) − p*`) whose >=0 set is S_safelanding — the crisp
  handover trigger (latch_margin_fn) and Run-3 avoid-g. NOT the reach reward (flat/
  saturated off the set — see the reach-reward note); use `l_vhat_reward` for the RA reach `l`."""
  c = _load(env.device)
  with torch.no_grad():
    x = (state_features(env) - c["mean"]) / c["std"]
    v = torch.sigmoid(c["net"](x).squeeze(-1))
  return (v - c["pstar"]).clamp(-clamp, clamp)


def l_vhat_reward(env, clamp: float = 3.0):
  """REACH REWARD margin (un-saturated fix): the UN-SATURATED logit form
      l = (net(state_features) − logit(p*_reward)) · reward_scale,  clipped ±clamp
  Same-signed as `sigmoid(net) − p*_reward` (identical zero level set) but with a real
  gradient everywhere, and p*_reward recalibrated (Youden-J) so genuinely-landable
  states are IN the set (E-α: 0.906 vs 0.33 at the old p*=0.975). Anti-Goodhart split:
  this reshapes the REWARD only; the handover set stays `l_vhat` (guard)."""
  c = _load(env.device)
  with torch.no_grad():
    x = (state_features(env) - c["mean"]) / c["std"]
    logit = c["net"](x).squeeze(-1)
  return ((logit - c["logit_pstar_reward"]) * c["reward_scale"]).clamp(-clamp, clamp)
