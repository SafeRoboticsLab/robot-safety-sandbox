"""State-only feature extractor for outcome-regression certificates.

THE MARKOV REQUIREMENT (funnel lesson, 2026-07-10): certificate features must
be functions of the physical STATE only — no action history, no commands, no
policy-normalized observations. Controller-dependent features make the
certificate valid only on the distribution of the controller that generated
them; at a filter handover the incoming controller differs and the certificate
is silently OOD (measured: 44% violations at the funnel handover).

Shared by the certificate collection scripts (label time) and the certified
reach margins in tasks/*.py (query time) — one definition, zero train/deploy
feature skew.
"""

from __future__ import annotations

import torch

SCAN_REL_CLAMP = 2.0


def state_features(env, scan_name: str = "terrain_scan") -> torch.Tensor:
  """(N, D) state-only features: root pose/velocity, posture, joints, and
  terrain-scan relative heights (state-only geometry — per-env gap width is
  NOT recoverable from terrain rows; the scan sees the actual patch)."""
  robot = env.scene["robot"]
  origins = env.scene.env_origins
  pos = robot.data.root_link_pos_w - origins                    # (N, 3)
  linv = robot.data.root_link_lin_vel_w                         # (N, 3)
  angv = robot.data.root_link_ang_vel_w                         # (N, 3)
  grav = robot.data.projected_gravity_b                         # (N, 3)
  jpos = robot.data.joint_pos                                   # (N, 12)
  jvel = robot.data.joint_vel                                   # (N, 12)
  scan = env.scene[scan_name]
  hit_z = scan.data.hit_pos_w[..., 2]                           # (N, R)
  dist = scan.data.distances
  base_z = robot.data.root_link_pos_w[:, 2:3]
  rel_h = torch.where(dist >= 0, hit_z - base_z,
                      torch.full_like(hit_z, -SCAN_REL_CLAMP))
  rel_h = rel_h.clamp(-SCAN_REL_CLAMP, SCAN_REL_CLAMP)          # (N, R)
  return torch.cat([pos, linv, angv, grav, jpos, jvel, rel_h], dim=-1)
