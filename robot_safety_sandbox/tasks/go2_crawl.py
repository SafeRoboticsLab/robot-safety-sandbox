"""Go2 low-bar crawl benchmark family (parkour skill 2) — momentum-reactive
filter: duck-coast-through a passable bar or brake before an impossible one.

Two-phase decomposition (mirrors gap-jumping crossing -> chain):
  go2_crawl_locomote  Phase 1: learn the crouch-crawl LOCOMOTION (velocity l)
  go2_crawl           Phase 2: decide crawl vs stop (rest l + window), warm from P1
  go2_crawl_isaacs    Phase 3: + worst-case force adversary

Env cfgs are NATIVE to the zoo (envs/go2_crawl/*).
"""

from __future__ import annotations

import torch

from ..envs.terrains.crawl_filter import BAR_DEPTH, _BAR_X
from ..margins import CLAMP, g_terrain_relative, l_rest
from ..registry import TaskSpec, register

_V_CMD = 1.0   # forward crawl target (m/s), world +x
_V_TOL = 0.7   # tracking tolerance (l >= 0 within this of the command)
_BAR_EXIT = _BAR_X + BAR_DEPTH   # past this (upright) == a clean crossing

# Gait-shaping sub-margin for l (v4.3): shape a TROT rather than the degenerate
# stomp, the CONVENTIONAL reach-avoid way -- l = min(velocity, gait) so the value
# is pinned to the WORST sub-margin and shaping pressure concentrates on the gait
# (additive would let velocity mask a bad gait). Phase-free trot score: diagonal
# pairs (FL-RR, FR-RL) synced AND the two diagonals in OPPOSITE contact phase.
# GATED OFF within _GAIT_OFF_DIST of the bar (a deep duck is not a trot -> the
# target set switches to velocity-only there, a valid region-dependent target).
_GAIT_THRESH = 0.8    # (fallback) trot-match fraction for the gait sub-margin >= 0
_GAIT_NORM = 0.2
_GAIT_OFF_DIST = 0.6  # drop the gait requirement within this dist (m) of the bar
# Trot-threshold CURRICULUM (v4.3c): a FIXED 0.8 threshold is unreachable from the
# 0.55 stomp -> gait margin -1.25 masked by gamma*V' -> no gradient (why the trot
# never climbed). Give it the same "start-satisfied, then climb" mechanism that
# made HEADING work: env._gait_thresh starts just above the current trot (0.56)
# and creeps UP toward (trot_ema + gap) as the policy improves -> the target stays
# just-reachable, a mild-negative margin that keeps a gradient (GaitThreshRampCallback).
_GAIT_THRESH_START = 0.56
_GAIT_THRESH_MAX = 0.88

# Heading sub-margin (v4.3b): velocity rewards world +x and the trot pattern is
# heading-agnostic, so the robot discovered it can TURN 90deg and trot sideways
# (crab-walk) to satisfy both. Require the body to FACE +x. Robots spawn facing
# +x, so this is climbable-from-the-start (penalizes turning away) -- and a robot
# must face +x to fit through the bar opening anyway, so it applies EVERYWHERE
# (not gated off at the bar like the trot term).
_HEADING_THRESH = 0.8   # body +x axis world-x component (cos ~37deg) for margin>=0
_HEADING_NORM = 0.2

# Nose-down pitch guard for the duck: projected_gravity_b[:, 0] == sin(pitch),
# > 0 when pitched nose-down. The standard g tilt term is non-directional and
# only bites at ~70deg (already face-planting); this is a DIRECTIONAL, early
# anti-dive signal so the robot learns to lower LEVEL instead of pitching the
# head down under forward momentum.
#
# TIGHTENED to ~15deg (was 37deg). Diagnosed on a trained checkpoint: 99% of
# terminations were illegal_contact and 92.8% of the strong contacts were at
# GROUND height (z<0.08 m) -- the FRONT TRUNK planting into the floor. Because
# the crouched base is so low (~0.20 m), only ~18deg of nose-down already drives
# the front trunk to the ground, well inside the old 37deg guard (which never
# fired). At 15deg the guard marks that dive unsafe, forcing a LEVEL duck -- and
# a level crawl at base_z 0.20 m clears cleanly (trunk-top 0.26 < 0.30 m bar,
# trunk-bottom 0.14 > ground).
_PITCH_DOWN_LIMIT = 0.26   # sin(~15deg): nose-down beyond this == g < 0
_PITCH_DOWN_NORM = 0.20     # O(1) normalization for the value regression

# --- shrinking-island collapse line (v4 temporal forcing) --------------------
# A per-env "collapse line" advances forward with episode time from behind each
# spawn. Standing still on the approach -> the line catches up -> g<0 (the island
# crumbled under the robot), creating URGENCY that kills the rock-in-place exploit
# at ANY bar height. The line is CAPPED just before the bar face, so everywhere
# under the bar and the far platform are FOREVER-safe (the value structure the
# gap gets from its wide far platform). See reset_crawl_trajectory + caught_by_pit.
_PIT_RATE = 0.006    # m/step the collapse line advances (~2.3 s grace before it
                     # catches a motionless robot on the approach; tune up later)
_PIT_BEHIND = 0.7    # line starts this far behind the spawn (initial grace)
_PIT_CAP = 0.35      # line caps this far before the bar face (= nose reach): a
                     # robot whose base reaches bar_face-0.35 has its front at the
                     # bar -> committed under the beam -> forever-safe past here.
_PIT_NORM = 0.30     # O(1) normalization for the value regression

# Constant forward "current" (v4.1): a small persistent +x force on the base every
# step. Unlike an initial momentum burst (which the policy learned to BRAKE away
# -> 0.07 m/s), a constant force cannot be braked off -- standing still means being
# dragged, so bracing is unstable and going WITH it (crawling forward) is the
# low-effort default. Also nudges the exit-standers to keep moving. ~1.5 kg-force.
# v4.2: the force is applied at the base (~CoM, base_z above the feet) which makes
# a forward-TIPPING torque -> the robot plants its nose (94% illegal_contact
# before the bar, worse when galloping). We cancel it with a pitch torque
# tau_y = -F*base_z, moving the effective push to the FEET (no forced pitch). And
# the magnitude is annealable via env._fwd_force_scale (15 -> 0 over training,
# FwdForceAnnealCallback) so the final policy crawls unaided.
_FWD_FORCE = 15.0   # N default / start magnitude (world +x on base_link)
_TIP_SIGN = -1.0    # sign of the tipping-compensation torque (verified in smoke)


def _pit_edge(env):
  """Per-env world-x of the advancing collapse line (shrinking-island edge), or
  None if _pit_start_x is unset. Advances with episode time, capped just before
  the bar face so under-bar/far states never get pit-g<0."""
  start = getattr(env, "_pit_start_x", None)
  if start is None:
    return None
  origin_x = env.scene.env_origins[:, 0]
  cap = origin_x + _BAR_X - _PIT_CAP
  edge = start + _PIT_RATE * env.episode_length_buf.to(start.dtype)
  return torch.minimum(edge, cap)


def crawl_locomote_margins(env):
  """Phase 1: g = crawl safety (bar strike / fall / off-ground); l = forward
  velocity tracking toward (V_CMD, 0). The dense forward signal produces the
  sustained gallop (episodes run full -- no success termination).

  Side effect (runs every step): latch env._ever_crossed_upright when the base
  clears the bar exit while upright. The height curriculum reads this at reset
  -- a clean crossing at ANY point counts as a win even if the robot falls
  after (episodes are short), while a stop or a face-slam never latches it and
  demotes. The upright gate is the avoid half: a dive-crossing is tilted, so it
  doesn't count. (Phase 2 re-introduces the stop decision via rest.)"""
  d = env.scene["robot"].data
  g = g_terrain_relative(env).clamp(-CLAMP, CLAMP)
  v = d.root_link_lin_vel_w[:, :2]
  cmd = torch.tensor([_V_CMD, 0.0], device=env.device)
  l = _V_TOL - torch.linalg.norm(v - cmd, dim=1)
  if hasattr(env, "_ever_crossed_upright"):
    x_rel = d.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    upright = d.projected_gravity_b[:, 2] < -0.6
    env._ever_crossed_upright |= (x_rel > _BAR_EXIT) & upright
  return g, l.clamp(-CLAMP, CLAMP)


def crawl_margins(env):
  """g = terrain-relative failure margin; l = safe rest RESTRICTED by the
  per-row obstacle window the env publishes (env._rest_obstacle_window_w):
  passable rows exclude the whole approach (only rest PAST the bar counts ->
  crawl-through is the target), impossible rows target rest BEFORE the bar.
  Without the window, plain rest is satisfiable everywhere and 'stop always'
  wins — the documented failure mode of this task."""
  g = g_terrain_relative(env).clamp(-CLAMP, CLAMP)
  l = l_rest(env)
  win = getattr(env, "_rest_obstacle_window_w", None)
  if win is not None:
    x = env.scene["robot"].data.root_link_pos_w[:, 0]
    d_out = torch.maximum(win[:, 0] - x, x - win[:, 1])
    l = torch.minimum(l, d_out / 0.3)
  return g, l.clamp(-CLAMP, CLAMP)


def crawl_duck_margins(env):
  """Duck sub-task, REACH-AVOID (ReachAvoidPPO). g = crawl safety with FULL
  non-foot contact (any leg/torso geom striking the bar or ground -> g<0), so
  the avoid-only exploit -- ducking by leaning back onto knees/forearms to
  retreat from the bar -- is unsafe. l = forward-velocity tracking, so the robot
  must keep moving FORWARD, not retreat. Duck-and-crawl-forward is then the only
  high-value behavior: retreating fails l, torso/knee contact fails g, and a low
  bar forces the duck.

  Anti-dive: g also carries a directional nose-down pitch guard. The base g's
  tilt term is non-directional and only fires at ~70deg (post-crash), so the
  face-plant -- ducking correctly but letting forward momentum pitch the head
  down into the ground -- was invisible to the avoid channel until impact. The
  guard makes excessive nose-down pitch g<0 EARLY, giving a dense gradient for a
  level duck (crouch all legs) while leaving a lean-back (nose_down<0) untouched
  (the reach l already punishes retreat)."""
  robot = env.scene["robot"]
  g = g_terrain_relative(env)
  nose_down = robot.data.projected_gravity_b[:, 0]  # sin(pitch); >0 == nose-down
  pitch_guard = (_PITCH_DOWN_LIMIT - nose_down) / _PITCH_DOWN_NORM
  g = torch.minimum(g, pitch_guard)
  # shrinking-island collapse line: g<0 once the advancing line passes the base
  # (approach states are transient; the capped line leaves under-bar/exit safe).
  edge = _pit_edge(env)
  if edge is not None:
    base_x = robot.data.root_link_pos_w[:, 0]
    g = torch.minimum(g, (base_x - edge) / _PIT_NORM)
  g = g.clamp(-CLAMP, CLAMP)
  v = robot.data.root_link_lin_vel_w[:, :2]
  cmd = torch.tensor([_V_CMD, 0.0], device=env.device)
  l = _V_TOL - torch.linalg.norm(v - cmd, dim=1)   # velocity-match sub-margin
  # gait-rhythm sub-margin (min-form, region-switched): reward a TROT on the
  # APPROACH so the value is pinned to the gait when it's the worst term; relax
  # to velocity-only near the bar so the robot can break rhythm to duck.
  try:
    contact = (env.scene["feet_ground_contact"].data.current_contact_time > 0
               ).float()  # (n, 4) == [FL, FR, RL, RR]
    fl, fr, rl, rr = contact[:, 0], contact[:, 1], contact[:, 2], contact[:, 3]
    trot = ((fl == rr).float() + (fr == rl).float() + (fl != fr).float()) / 3.0
    env._gait_trot_last = trot.mean().detach()   # read by the ramp callback
    thresh = getattr(env, "_gait_thresh", _GAIT_THRESH_START)
    gait = (trot - thresh) / _GAIT_NORM
    # NOTE (2026-07-09, user-directed): heading/yaw margin REMOVED. l is now the
    # cleanest reach-avoid form -- min(forward-velocity, gait) only, no discrete
    # face-+x bolt-on. Trades crab-walk risk for theory purity; the bet is that
    # ample training + wide parallelism finds a forward trot without the hack.
    dist_bar = (env.scene.env_origins[:, 0] + _BAR_X) - robot.data.root_link_pos_w[:, 0]
    approach = dist_bar > _GAIT_OFF_DIST
    l = torch.where(approach, torch.minimum(l, gait), l)  # trot on the approach
  except (KeyError, AttributeError):
    pass
  # Constant forward current (annealable, tipping-compensated): re-applied every
  # step so standing still means being dragged. tau_y = -F*base_z cancels the
  # nose-down tip so the push acts at the feet, not the CoM.
  scale = getattr(env, "_fwd_force_scale", None)
  scale = _FWD_FORCE if scale is None else float(scale)
  if scale > 0.0:
    if not hasattr(env, "_fwd_bids"):
      bids, _ = robot.find_bodies("base_link")
      env._fwd_bids = list(bids)
      env._fwd_all = torch.arange(env.num_envs, device=env.device)
    nb = len(env._fwd_bids)
    base_z = robot.data.root_link_pos_w[:, 2]
    forces = torch.zeros((env.num_envs, nb, 3), device=env.device)
    forces[:, :, 0] = scale
    torques = torch.zeros_like(forces)
    torques[:, :, 1] = _TIP_SIGN * scale * base_z.unsqueeze(-1)
    robot.write_external_wrench_to_sim(
      forces, torques, body_ids=env._fwd_bids, env_ids=env._fwd_all)
  return g, l.clamp(-CLAMP, CLAMP)


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_crawl.env_cfg import (
    unitree_go2_crawl_duck_env_cfg, unitree_go2_crawl_duck_video_env_cfg,
    unitree_go2_crawl_env_cfg, unitree_go2_crawl_isaacs_env_cfg,
    unitree_go2_crawl_locomote_env_cfg)
  register(TaskSpec(
    task_id="go2_crawl_duck", cfg_builder=unitree_go2_crawl_duck_env_cfg,
    margin_fn=crawl_duck_margins, default_algo="ReachAvoidPPO",
    description="Momentum approach at a low bar (forces a duck) + forward "
                "velocity reach + FULL non-foot contact -> learn to duck AND "
                "crawl forward (not retreat). Reach-avoid."))
  register(TaskSpec(
    task_id="go2_crawl_duck_video", cfg_builder=unitree_go2_crawl_duck_video_env_cfg,
    margin_fn=crawl_duck_margins, default_algo="ReachAvoidPPO",
    description="Packed-terrain herd render of go2_crawl_duck (eval video only)."))
  register(TaskSpec(
    task_id="go2_crawl_locomote", cfg_builder=unitree_go2_crawl_locomote_env_cfg,
    margin_fn=crawl_locomote_margins, default_algo="ReachAvoidPPO",
    description="Phase 1: crouch-crawl LOCOMOTION under a descending bar "
                "(velocity-tracking reach, momentum init, passable only)."))
  register(TaskSpec(
    task_id="go2_crawl", cfg_builder=unitree_go2_crawl_env_cfg,
    margin_fn=crawl_margins,
    default_algo="ReachAvoidPPO", warmstart_from="go2_crawl_locomote",
    description="Phase 2: decide crawl vs stop (rest + window), warm from P1."))
  register(TaskSpec(
    task_id="go2_crawl_isaacs", cfg_builder=unitree_go2_crawl_isaacs_env_cfg,
    margin_fn=crawl_margins,
    default_algo="IsaacsPPO", warmstart_from="go2_crawl",
    supports_adversary=True,
    description="Crawl + worst-case base-force adversary."))
