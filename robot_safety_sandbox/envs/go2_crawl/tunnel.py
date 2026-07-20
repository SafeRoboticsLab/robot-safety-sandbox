"""TUNNEL crawl -- the crawl campaign's NEW formulation.  A Go2 crouches through
a low overhead beam ("the bar"); the low-clearance region under it (length =
``bar_depth``) is "the tunnel".  Built ON TOP of the low-bar env
(``unitree_go2_low_bar_env_cfg``): it inherits the VIRTUAL-bar terrain, the
analytic ``bar_strike`` termination, and the analytic bar perception swap, then
overrides (1) the spawn, (2) the margins, and (3) the ground-contact failure.

Travel axis is +x.  ``x_rel`` is the base position relative to the bar's near
face (the terrain patch origin): ``x_rel = 0`` at the ENTRANCE (bar near face),
``x_rel = depth`` at the EXIT (bar far face).  ``x_rel < 0`` = approach;
``x_rel > depth`` = far side.

Difference from ``low_bar`` (which was a reverse-curriculum, momentum-secondary
benchmark): the tunnel spawns UNIFORMLY over the whole ~1 m training set with a
FULLY RANDOMIZED pose AND velocity (no curriculum) -- an off-distribution robust
formulation.  Both twins share this env; only the reach term l differs (RA
``tunnel_margins`` vs ``avoid_only(tunnel_margins)``).

Margins:
  g = min( g_bar, g_ground ), clamped +-CLAMP.
    * g_bar    -- IDENTICAL to low_bar: (clearance - trunk_top)/BAR_NORM while
                  the trunk overlaps the bar span [0, depth]; +CLAMP elsewhere.
                  > 0 iff the trunk fits under the beam.
    * g_ground -- "no illegal ground contact": FAILURE iff any body OTHER THAN
                  {*_calf, *_foot} (i.e. base / hip / thigh) touches the ground.
                  +CLAMP when clean, -CLAMP on a disallowed contact (binary).
                  NO base-height / tilt terms -- g is contact + bar only.
  l = (x_rel - (depth + L_MARGIN)) / POS_NORM, clamped +-CLAMP -- position
      completion in the immediate region JUST PAST the exit.  POS_NORM is small
      so l has a steep, informative gradient across the ~1 m training set (the
      low_bar POS_NORM=3.5 flat-gradient trap is avoided).
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from robot_safety_sandbox.margins import CLAMP, CONTACT_FORCE_THRESHOLD
from robot_safety_sandbox.envs.go2_crawl.env_cfg import (
  _crouch_z,
  _ensure_crawl_buffers,
)
from robot_safety_sandbox.envs.go2_crawl.low_bar import (
  BAR_NORM,
  HALF_TRUNK,
  _bar_params,
  _trunk_over_bar,
  _x_rel,
  unitree_go2_low_bar_env_cfg,
)

# --- l (reach) constants ------------------------------------------------------
# l >= 0 only just past the exit; POS_NORM small so l is steeply graded across
# the ~1 m training set (span [-0.3, depth+0.3]) instead of clamped flat.
L_MARGIN = 0.05           # completion x = depth + this (just past the far face)
POS_NORM = 0.4            # steep gradient (~2.5 / m); NOT low_bar's 3.5

# --- spawn constants ----------------------------------------------------------
_X_PAD = 0.3              # spawn x_rel in [-_X_PAD, depth + _X_PAD]
_STAND_FRAC = 0.35        # fraction spawned STANDING; the rest crouched (mixed
                          # across the range -- brake_or_jump lesson: a low base
                          # gets the MATCHING bent-leg crouch pose, never a
                          # root-only fold).
_JOINT_VEL_MAG = 1.5      # per-joint velocity randomization (+-, rad/s)
# base velocity ranges (mostly-forward vx, small vy/vz, small angular)
_VX_LO, _VX_HI = -0.2, 1.2
_VLAT = 0.15              # |vy|, |vz|
_WANG = 0.3              # |angular velocity| each axis


# --- disallowed-contact helper (shared by g_ground and the termination) -------

def _disallowed_ground_contact(env, nonfoot_name="nonfoot_ground_touch"):
  """True where a DISALLOWED body (base / hip / thigh -- anything but calf/foot)
  is touching the ground.  Reads the ``nonfoot_ground_touch`` contact sensor,
  which the tunnel cfg reconfigures so its primary geom set excludes ONLY the
  calf + foot geoms (the allowed load-bearing set).  Mirrors the force-reading
  machinery of ``g_terrain_relative``."""
  false = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  try:
    sensor = env.scene[nonfoot_name]
  except KeyError:
    return false
  force = (sensor.data.force_history
           if sensor.data.force_history is not None else sensor.data.force)
  if force is None:
    return false
  mag = torch.norm(force, dim=-1)
  while mag.dim() > 1:
    mag = mag.amax(dim=-1)
  return mag > CONTACT_FORCE_THRESHOLD


# --- margins ------------------------------------------------------------------

def l_tunnel(env) -> torch.Tensor:
  """Position completion in the immediate region just past the EXIT."""
  _clearance, depth = _bar_params(env)
  return (_x_rel(env) - (depth + L_MARGIN)) / POS_NORM


def g_ground(env) -> torch.Tensor:
  """Binary 'no illegal ground contact' term: +CLAMP clean, -CLAMP on a
  base/hip/thigh ground contact."""
  bad = _disallowed_ground_contact(env)
  return torch.where(bad, torch.full_like(bad, -CLAMP, dtype=torch.float),
                     torch.full_like(bad, CLAMP, dtype=torch.float))


def tunnel_margins(env):
  """g = min(g_bar, g_ground); l = completion past the exit.  The avoid twin
  strips l via ``avoid_only()``."""
  clearance, depth = _bar_params(env)
  robot = env.scene["robot"]

  # VIRTUAL-BAR term -- IDENTICAL to low_bar: threatens only while the trunk
  # (base +- TRUNK_HALF_LEN) overlaps the bar span [0, depth].
  base_z = robot.data.root_link_pos_w[:, 2]
  trunk_top = base_z + HALF_TRUNK
  x_rel = _x_rel(env)
  g_bar = (clearance - trunk_top) / BAR_NORM
  g_bar = torch.where(_trunk_over_bar(x_rel, depth), g_bar,
                      torch.full_like(g_bar, CLAMP))

  g = torch.minimum(g_bar, g_ground(env))
  return g.clamp(-CLAMP, CLAMP), l_tunnel(env).clamp(-CLAMP, CLAMP)


def tunnel_ground_contact(env, margin_fn=None) -> torch.Tensor:
  """Absorbing FAILURE: a disallowed body (base / hip / thigh) plants on the
  ground.  Alongside the inherited analytic ``bar_strike``; base.py's g-anchor
  then grades the terminal state as failure."""
  return _disallowed_ground_contact(env)


# --- spawn: uniform over the training set, randomized pose AND velocity --------

def reset_tunnel(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Uniform spawn over the whole tunnel training set (NO curriculum), with a
  fully randomized pose AND velocity.

  x_rel ~ U[-_X_PAD, depth + _X_PAD] (approach / under-tunnel / just-past-exit).
  Pose: a mix of STANDING (default joints, z ~ 0.34) and CROUCHED (crouch pose
  at depth alpha, z = _crouch_z(alpha) ~ 0.15) so the base height + joint config
  are always physically consistent (the crouch machinery -- crouch_joints /
  _crouch_z -- supplies the matching bent-leg pose).  Velocity: base lin/ang
  velocity here, joint velocities added by the ``tunnel_joint_vel`` reset event
  (runs AFTER reset_robot_joints / crouch_joints have set the joint positions).
  """
  _ensure_crawl_buffers(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  dev = env.device
  asset = env.scene[asset_cfg.name]
  _clearance, depth = _bar_params(env)
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()
  origins = env.scene.env_origins[env_ids]

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), dev)

  # --- position: uniform over approach + under-tunnel (NOT past the exit) ---
  # Upper bound = depth (the exit face), NOT depth+pad: spawning past the exit
  # lands the robot ALREADY in the reach set (l>=0) -> a free 1-step success that
  # gives a "do-nothing wins ~25%" attractor the sparse RA reward can't escape
  # (E045 v1: trained success == the zero-action free baseline). Every spawn now
  # has l<0, so reaching the exit ALWAYS requires crawling. The policy still
  # visits past-exit states by reaching them (value coverage for the filter).
  x_rel = u(-_X_PAD, depth)
  y = u(-0.06, 0.06)

  # --- pose: mix STANDING and CROUCHED (matching joint config via masks) ---
  stand_z = root[:, 2]                        # nominal standing height (~0.32)
  stand = u(0.0, 1.0) < _STAND_FRAC
  crouch = ~stand
  alpha = torch.where(crouch, u(0.0, 1.0), torch.zeros(n, device=dev))
  z = torch.where(crouch, _crouch_z(alpha), stand_z) + u(0.003, 0.015)

  pos = torch.stack([origins[:, 0] + x_rel, origins[:, 1] + y,
                     origins[:, 2] + z], dim=-1)
  euler = torch.stack([u(-0.04, 0.04), u(-0.04, 0.04), u(-0.15, 0.15)], dim=1)
  quat = quat_mul(root[:, 3:7],
                  quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]))

  # --- velocity: base lin (mostly-forward vx) + small angular, all randomized ---
  vel = torch.zeros(n, 6, device=dev)
  vel[:, 0] = u(_VX_LO, _VX_HI)
  vel[:, 1] = u(-_VLAT, _VLAT)
  vel[:, 2] = u(-_VLAT, _VLAT)
  vel[:, 3] = u(-_WANG, _WANG)
  vel[:, 4] = u(-_WANG, _WANG)
  vel[:, 5] = u(-_WANG, _WANG)

  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1),
                                    env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)

  # masks consumed by the crouch_joints event (apply_crouch_joints).
  env._crouch_mask[env_ids] = crouch
  env._crouch_alpha[env_ids] = alpha
  env._splay_mag[env_ids] = torch.zeros(n, device=dev)
  env._leg_jitter[env_ids] = torch.full((n,), 0.02, device=dev)


def apply_tunnel_joint_vel(env, env_ids, asset_cfg=SceneEntityCfg("robot"),
                           vel_mag: float = _JOINT_VEL_MAG):
  """Randomize per-joint velocities AFTER the pose events (reset_robot_joints /
  crouch_joints) have written the joint POSITIONS.  Keeps those positions and
  overwrites the joint velocities with U[-vel_mag, vel_mag] -- the tunnel's
  velocity-randomization requirement (crouch_joints writes zero joint vel)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  asset = env.scene[asset_cfg.name]
  jpos = asset.data.joint_pos[env_ids].clone()
  jvel = sample_uniform(-vel_mag, vel_mag, jpos.shape, env.device)
  asset.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)


# --- contact set reconfiguration ---------------------------------------------
# ALLOWED (load-bearing, no failure) = {calf, foot}; the disallowed FAILURE set
# is therefore base + hip + thigh.  The inherited ``nonfoot_ground_touch`` sensor
# (crawl base) excludes foot + thigh + calf; the tunnel drops thigh from the
# exclude so a thigh plant is a failure again.
_TUNNEL_ALLOWED = tuple(
  f"{leg}_{seg}_collision"
  for leg in ("FL", "FR", "RL", "RR")
  for seg in ("foot", "calf1", "calf2"))


def _set_tunnel_contact_set(cfg: ManagerBasedRlEnvCfg) -> None:
  """Reconfigure the ``nonfoot_ground_touch`` sensor so its primary geom set is
  base + hip + thigh (exclude only the allowed calf + foot geoms)."""
  from dataclasses import replace
  sensors = []
  for s in (cfg.scene.sensors or ()):
    if getattr(s, "name", None) == "nonfoot_ground_touch":
      s = replace(s, primary=replace(s.primary, exclude=_TUNNEL_ALLOWED))
    sensors.append(s)
  cfg.scene.sensors = tuple(sensors)


# --- env cfg builder ----------------------------------------------------------

def unitree_go2_tunnel_env_cfg(play: bool = False, bar_clearance: float = 0.30,
                               bar_depth: float = 0.4) -> ManagerBasedRlEnvCfg:
  """TUNNEL crawl env.  Builds on the low-bar env (virtual bar terrain +
  bar_strike + bar perception), then swaps in the uniform randomized-pose/-vel
  spawn (no curriculum), the base+hip+thigh contact-failure set, and the
  tunnel margins' ground-contact termination.

  ``bar_clearance`` default 0.30 < standing trunk-top (~0.38): a standing robot
  under the tunnel STRIKES (g_bar < 0), so crouching is required to pass."""
  cfg = unitree_go2_low_bar_env_cfg(play=play, bar_clearance=bar_clearance,
                                    bar_depth=bar_depth)

  # Uniform randomized spawn (no reverse curriculum). KEEP reset_robot_joints
  # (default joints) + crouch_joints (crouch pose) so every spawn gets a full
  # joint state; add tunnel_joint_vel LAST so joint velocities are randomized on
  # top of the pose the earlier events set.
  cfg.events["reset_base"] = EventTermCfg(func=reset_tunnel, mode="reset",
                                          params={})
  cfg.events["tunnel_joint_vel"] = EventTermCfg(func=apply_tunnel_joint_vel,
                                                mode="reset", params={})
  cfg.curriculum = {}                            # no curriculum

  # Contact-failure set = base + hip + thigh (allowed = calf + foot).
  _set_tunnel_contact_set(cfg)

  # Ground-plant = absorbing FAILURE, alongside the inherited analytic
  # bar_strike. base.py's g-anchor grades the terminal state as failure. The
  # generic inherited fell_over / base_below_local_terrain terms are LEFT as-is
  # (they don't conflict with the contact set; deepest legal crouch base ~0.15 >
  # the 0.08 clearance limit, so a legal crawl never trips them).
  cfg.terminations["tunnel_ground_contact"] = TerminationTermCfg(
    func=tunnel_ground_contact)
  return cfg
