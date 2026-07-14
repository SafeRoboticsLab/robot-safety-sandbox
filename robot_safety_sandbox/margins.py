"""Composable reach-avoid margin library (batched torch, mjlab scene API).

A task's ``margin_fn(env) -> (g, l)`` is composed from these terms:

    g (avoid)  : stay out of the failure set    — g < 0 == failure
    l (reach)  : arrive in the target set       — l >= 0 == reached

Conventions: margins are signed distances normalized to O(1); the g terminal
anchor and clamping live in :mod:`base` / the buffers, not here. All terms are
validated on the Go2 parkour tasks (ported from the reference wrappers).
"""

from __future__ import annotations

import math

import torch

# Shared defaults (Go2-scale; override per task).
MIN_CLEARANCE = 0.08
HEIGHT_NORM = 0.25
SIN_TILT_LIMIT = math.sin(math.radians(70.0))
OBSTACLE_MARGIN = 0.12
FOOTPRINT_RADIUS = 0.5
CONTACT_FORCE_THRESHOLD = 10.0
CENTRAL_RADIUS = 0.20
SUPPORT_THRESHOLD = 0.45
SUPPORT_NORM = 0.30
CLAMP = 3.0


# --- g terms -------------------------------------------------------------------

def ground_reference(env, scan_name="terrain_scan"):
  """(base_z, ground_ref): raycast ground reference within the footprint —
  the supporting-surface height under the trunk (gap/edge aware). Shared by
  g_terrain_relative and the airborne term of certified-launch l margins."""
  robot = env.scene["robot"]
  scan = env.scene[scan_name]
  hit = scan.data.hit_pos_w
  dist = scan.data.distances
  base_z = robot.data.root_link_pos_w[:, 2]
  base_xy = robot.data.root_link_pos_w[:, None, :2]
  planar = torch.norm(hit[..., :2] - base_xy, dim=-1)
  hit_z = hit[..., 2]
  in_fp = (dist >= 0) & (planar <= FOOTPRINT_RADIUS)

  below = in_fp & (hit_z < base_z[:, None] - OBSTACLE_MARGIN)
  neg_inf = torch.full_like(hit_z, -1.0e9)
  pos_inf = torch.full_like(hit_z, 1.0e9)
  ground_ref = torch.where(below, hit_z, neg_inf).max(dim=1).values
  lowest = torch.where(in_fp, hit_z, pos_inf).min(dim=1).values
  lowest = torch.where(in_fp.any(dim=1), lowest, base_z)
  ground_ref = torch.where(below.any(dim=1), ground_ref, lowest)
  return base_z, ground_ref


def g_terrain_relative(env, scan_name="terrain_scan",
                       nonfoot_name="nonfoot_ground_touch"):
  """min(local-terrain base height, tilt, non-foot contact) — the standard
  legged failure margin. Terrain-relative (raycast ground reference within the
  footprint), so gaps/platform edges are handled correctly."""
  robot = env.scene["robot"]
  base_z, ground_ref = ground_reference(env, scan_name)
  height = (base_z - ground_ref - MIN_CLEARANCE) / HEIGHT_NORM

  grav_xy = torch.norm(robot.data.projected_gravity_b[:, :2], dim=1)
  tilt = (SIN_TILT_LIMIT - grav_xy) / SIN_TILT_LIMIT

  terms = [height, tilt]
  try:
    sensor = env.scene[nonfoot_name]
    force = (sensor.data.force_history
             if sensor.data.force_history is not None else sensor.data.force)
    if force is not None:
      mag = torch.norm(force, dim=-1)
      while mag.dim() > 1:
        mag = mag.amax(dim=-1)
      terms.append((CONTACT_FORCE_THRESHOLD - mag) / CONTACT_FORCE_THRESHOLD)
  except (KeyError, AttributeError):
    pass
  return torch.stack(terms, dim=-1).min(dim=-1).values


# --- l terms -------------------------------------------------------------------

def l_gap_foothold(env, scan_name="terrain_scan", patch_length=8.0,
                   progress_weight=0.5):
  """Foothold support under the body + forward progress (gap tasks: standing
  on solid ground counts; over a gap the support vanishes -> l < 0)."""
  robot = env.scene["robot"]
  scan = env.scene[scan_name]
  hit = scan.data.hit_pos_w
  dist = scan.data.distances
  base_z = robot.data.root_link_pos_w[:, 2]
  base_xy = robot.data.root_link_pos_w[:, None, :2]
  planar = torch.norm(hit[..., :2] - base_xy, dim=-1)
  central = (dist >= 0) & (planar <= CENTRAL_RADIUS)
  neg_inf = torch.full_like(hit[..., 2], -1.0e9)
  ground_under = torch.where(central, hit[..., 2], neg_inf).max(dim=1).values
  ground_under = torch.where(~central.any(dim=1), base_z - 5.0, ground_under)
  foothold = (SUPPORT_THRESHOLD - (base_z - ground_under)) / SUPPORT_NORM
  origin_x = env.scene.env_origins[:, 0]
  progress = ((robot.data.root_link_pos_w[:, 0] - origin_x)
              / patch_length).clamp(0.0, 1.5)
  return foothold + progress_weight * progress


def l_rest(env, v_rest=0.3, v_rest_norm=0.5, cross_bias_weight=0.3,
           cross_bias_scale=3.0):
  """Safe-stop target (deployment safety-filter objective): l >= 0 iff nearly
  at rest, plus a mild bias for resting further along (cross when safe)."""
  robot = env.scene["robot"]
  speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
  rest = (v_rest - speed) / v_rest_norm
  origin_x = env.scene.env_origins[:, 0]
  prog = ((robot.data.root_link_pos_w[:, 0] - origin_x)
          / cross_bias_scale).clamp(0.0, 1.0)
  return rest + cross_bias_weight * prog


def l_gap_completion(env, rest_x=5.0, pos_norm=0.5, v_rest=0.3, v_rest_norm=0.5):
  """COMPLETION target for the gap decision unit, min-form (conventional RA:
  intersection target set): l >= 0 iff nearly at rest AND past the gap cluster.

  Unlike ``l_rest`` (rest-ANYWHERE + additive cross bias), resting BEFORE the
  gap can never satisfy this l -- the RA fallback engaged by a filter must
  carry the crossing to reach its target, and on uncrossable gaps the target
  is unreachable so the policy degrades to pure-avoid (stop). This asymmetry
  vs the avoid-only twin is the R-CBF claim's mechanism.

  ``rest_x=5.0`` sits on the rest zone for every difficulty row of
  SAFETY_FILTER_TERRAINS_CFG (worst-case cluster end: 2.5 approach + 3x0.5
  gaps + 2x0.35 separators = 4.7 m; separators are deliberately too short to
  stop on, so the cluster is ONE committed maneuver and completion = past the
  whole cluster)."""
  robot = env.scene["robot"]
  speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
  rest = (v_rest - speed) / v_rest_norm
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  crossed = (x_rel - rest_x) / pos_norm
  return torch.minimum(rest, crossed)


def l_launch_basin(env, gap_x=2.5, band=0.35, v_launch=2.2, v_norm=0.5,
                   pos_norm=0.3):
  """LAUNCH-BASIN reach target, min-form: l >= 0 iff the robot is CLOSE to the
  gap (within ``band`` of the face or beyond) AND carries LAUNCH momentum
  (forward vx >= v_launch) — i.e. it is inside the commitment basin from which
  the (warm-started) jump succeeds ballistically.

  Why this instead of a completion target: reach-avoid banks its value AT the
  reach event (post-reach g does not enter V), so making the target the
  commitment point renders the reach itself risk-free — accelerating on the
  approach violates nothing — and the maneuver's residual risk (measured
  96.8% success from this basin) can no longer outvote the objective. Standing
  far from the gap yields V -> 0 < V(accelerate), so the momentum-buildup
  gradient always exists. Post-reach states have l < 0 again (proximity term),
  so flight/landing keep receiving pure-avoid gradients through g. The
  certificate reads: "can reach the validated launch basin safely"."""
  robot = env.scene["robot"]
  vx = robot.data.root_link_lin_vel_w[:, 0]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  momentum = (vx - v_launch) / v_norm
  proximity = (x_rel - (gap_x - band)) / pos_norm
  return torch.minimum(momentum, proximity)


def l_zero(env):
  """Avoid-only tasks: no reach target (l always 'unreached')."""
  n = env.scene["robot"].data.root_link_pos_w.shape[0]
  return torch.zeros(n, device=env.device)


def l_neg(env):
  """Avoid-only tasks under a REACH-AVOID learner (ReachAvoidPPO/IsaacsPPO).

  Those learners use ``v_to_go = min(g, max(l, gV'))``. With ``l_zero`` the
  target set is the WHOLE space (l >= 0 everywhere): ``max(0, gV')`` clips all
  negative futures, so coming failures never propagate — a degenerate backup
  that destroyed a warm-started policy within <25M steps (digit_stabilize_stay
  + IsaacsPPO, 2026-07-10). Pinning ``l = -CLAMP`` (below any reachable value)
  makes ``max(l, gV') = gV'`` — the backup reduces EXACTLY to the avoid /
  viability backup ``min(g, gV')`` — turning IsaacsPPO into a two-player game
  over the robust-invariance value. SafetyPPO ignores ``l``, so avoid-only
  tasks that may also be trained adversarially should use ``l_neg``.
  """
  n = env.scene["robot"].data.root_link_pos_w.shape[0]
  return torch.full((n,), -CLAMP, device=env.device)


# --- composition ---------------------------------------------------------------

def compose(g_fn, l_fn, clamp: float = CLAMP):
  """margin_fn from a g term and an l term (clamped for value regression)."""
  def margin_fn(env):
    g = g_fn(env).clamp(-clamp, clamp)
    l = l_fn(env).clamp(-clamp, clamp)
    return g, l
  return margin_fn
