"""V_stay / Omega[stay] margin (the sRAS Phase-II certificate).

The STAY game (task ``go2_gap_stay``) trains an avoid-only SafetyPPO policy whose
critic V_stay is the viability value of a landed far-side state. Its conservatively
calibrated non-negative super-level set is Omega[stay] = viab(T\\F): the landed
states from which the robot can persist without failing.

Phase I needs Omega[stay] inside its margin / termination functions (reach
target = Omega[stay]; failure = F ∪ (T\\Omega[stay])). The SafetyPPO critic operates
on the env's 5-frame history-stacked policy observation, which a per-step margin_fn
cannot cheaply reconstruct -> so this module exposes a DISTILLED compact-state form:
a small MLP on the CURRENT-FRAME dynamic features (base height, orientation, root
velocities, joint state), fit against the true V_stay critic and validated to agree
on held-out landed states (see scratchpad/e100_vstay.py + vstay_validation.json).
Same precedent as vhat.py (a cached small net + a threshold, evaluated on the GPU
every step).

    stay_kernel_margin(env) = distilled_score(features(env)) - tau   (>=0 <=> Omega[stay])

Artifact path resolves from $SRAS_VSTAY_PATH (default: the stay-game artifact).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

_DEFAULT_PATH = os.path.expanduser(
  "~/artifacts/robot-safety-sandbox/stay-game/vstay_distilled.pt")
_CACHE: dict = {}


def stay_features(env) -> torch.Tensor:
  """Current-frame dynamic features [N, 35] of the robot, IDENTICAL in layout to
  the bank-row slice used to fit the distilled net (see e100_vstay.stay_features_
  from_rows): base_z (origin-relative), root_link_quat_w (4), root_link_lin_vel_w
  (3), root_link_ang_vel_w (3), joint_pos (12), joint_vel (12). x_rel is
  deliberately EXCLUDED — viability of a landed state is about dynamic
  recoverability, not absolute position on the far platform."""
  d = env.scene["robot"].data if hasattr(env, "scene") else env.robot.data
  origins = (env.scene.env_origins if hasattr(env, "scene")
             else env.mj.scene.env_origins)
  z = (d.root_link_pos_w[:, 2] - origins[:, 2])[:, None]
  return torch.cat([z, d.root_link_quat_w, d.root_link_lin_vel_w,
                    d.root_link_ang_vel_w, d.joint_pos, d.joint_vel], dim=1)


def _load(device):
  path = os.environ.get("SRAS_VSTAY_PATH", _DEFAULT_PATH)
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
  net.load_state_dict(ck["state_dict"])
  net.to(device).eval()
  for p in net.parameters():
    p.requires_grad_(False)
  out = {"net": net, "mean": ck["feat_mean"].to(device),
         "std": ck["feat_std"].to(device), "tau": float(ck["tau"])}
  _CACHE[key] = out
  print(f"[vstay] Omega[stay] distilled net <- {path} (tau={out['tau']:.4f}, "
        f"AUROC={ck.get('auroc_distilled', float('nan')):.3f})")
  return out


def v_stay(env) -> torch.Tensor:
  """Raw distilled V_stay score (higher = more viable). Diagnostics only."""
  c = _load(env.device)
  with torch.no_grad():
    x = (stay_features(env) - c["mean"]) / c["std"]
    return c["net"](x).squeeze(-1)


def stay_kernel_margin(env, clamp: float = 3.0) -> torch.Tensor:
  """Omega[stay] membership margin: >=0 iff the landed state is in the calibrated
  (conservative, precision>=0.98) stay kernel. For Phase-I reach target / failure
  augmentation."""
  c = _load(env.device)
  with torch.no_grad():
    x = (stay_features(env) - c["mean"]) / c["std"]
    score = c["net"](x).squeeze(-1)
  return (score - c["tau"]).clamp(-clamp, clamp)
