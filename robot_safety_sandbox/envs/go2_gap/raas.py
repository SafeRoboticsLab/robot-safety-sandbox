"""RAAS -- "reach-and-always-safe": a 2-step gap-jump recipe over PURE RANDOM
init that REPLACES the hand-built landing -> crossing -> reverse-curriculum
pipeline (``brake_or_jump``) with the simplest possible bootstrap.

Both steps share THIS env: the island gap terrain at ``gap_width`` (from
``unitree_go2_harvest_env_cfg``, NO curriculum) with a deliberately UNSHAPED,
fully-uniform-random spawn over the whole gap region.  The recipe is:

  Step 1  SafetyPPO1P (avoid-only), g = the Stage-3 TARGET ``l_stable_far`` used
          AS the safety margin.  No warm-start.  Learns a value/policy that
          keeps the robot in the stable far-stance set from random states.
  Step 2  ReachAvoidPPO1P, the proper Stage-3 margins (g = ``g_terrain_relative``,
          l = ``l_stable_far``), warm-started from Step 1's checkpoint.

The spawn is intentionally the CLEANEST pure-random version -- no curriculum, no
structured sampling -- doomed/unrecoverable airborne spawns (over the gap void)
INCLUDED.  Travel axis is +x; ``x_rel`` is the base position relative to the gap
near-edge origin (the patch origin), exactly as the other gap resets:
``x_rel = 0`` at the near edge, ``0 <= x_rel <= gap_width`` over the void (no
ground -> airborne), ``x_rel > gap_width`` on the far platform.

Pose consistency: a mix of STANDING (default joints, base ~0.32) and CROUCHED
(bent-leg crouch pose at depth alpha, base = ``_crouch_z(alpha)`` ~0.15-0.17) so
base height + joint config are always physically consistent -- reusing the crawl
crouch machinery (``crouch_joints`` / ``_crouch_z``), the same fix that keeps the
tunnel spawns from folding.  A uniform height offset above that grounded
equilibrium then lifts the robot from grounded to airborne.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from robot_safety_sandbox.envs.go2_crawl.env_cfg import (
  _crouch_z,
  _ensure_crawl_buffers,
  apply_crouch_joints,
)
from robot_safety_sandbox.envs.go2_gap.brake_or_jump import _gap_width, l_stable_far
from robot_safety_sandbox.envs.go2_gap.brake_or_jump_harvest import (
  unitree_go2_harvest_env_cfg)

# --- spawn constants ----------------------------------------------------------
_X_PAD = 0.6              # spawn x_rel in [-_X_PAD, gap_width + _X_PAD]: BOTH
                         # platforms + the gap void, uniform.
_STAND_FRAC = 0.5        # fraction spawned STANDING (default joints); the rest
                         # crouched (matching bent-leg pose via the masks).
_Z_AIR_MAX = 0.45        # max height offset above the grounded equilibrium
                         # (0 = grounded, up to airborne).
_Z_LIFT = 0.01           # tiny always-lift so a grounded spawn's feet don't
                         # penetrate under the joint jitter.
# base velocity ranges: forward-biased vx, small lateral/vertical + angular --
# "varying momentum", random/unshaped.
_VX_LO, _VX_HI = -0.5, 1.5
_VLAT = 0.4              # |vy|, |vz|
_WANG = 0.5             # |angular velocity| each axis
# base orientation jitter (small roll/pitch, wider yaw).
_RP = 0.2               # |roll|, |pitch|
_YAW = 0.3              # |yaw|

# --- recoverable-airborne shaping constants -----------------------------------
_GRAV = 9.81            # gravity for the ballistic solve
_AIRBORNE_LIFT = 0.08   # a platform spawn counts as AIRBORNE (-> shaped onto a
                        # landing arc) once its base sits > this above the
                        # STANDING landing height (i.e. it has real airtime to
                        # cross); lower/crouched spawns keep their random momentum
                        # and just settle onto their own platform (recoverable).
                        # Over-void spawns are airborne regardless (no ground).
_VZ_ARC_LO, _VZ_ARC_HI = -1.0, 2.0    # vertical launch vel of the ballistic arc
_X_LAND_LO, _X_LAND_HI = 0.05, 0.40   # landing target past the far edge (x_rel)
_T_MIN = 0.2            # min flight time (clamp; avoids blow-up when h~0)
_VX_ARC_MAX = 4.0       # max |vx| of the shaped arc (never explode)


def reset_gap_raas(env, env_ids, asset_cfg=SceneEntityCfg("robot"),
                   recoverable_airborne: bool = True):
  """Uniform-random spawn over the whole gap region (NO curriculum, NO
  structured sampling).

  x_rel ~ U[-_X_PAD, gap_width + _X_PAD] (near platform / gap void / far
  platform).  Pose: a mix of STANDING and CROUCHED with a matching joint config
  (the crouch machinery supplies the bent-leg pose), lifted by a uniform height
  offset from grounded to airborne.  Joint positions for crouched envs are
  written by the ``crouch_joints`` event (reads the masks set here); standing
  envs keep the default joints from ``reset_robot_joints``.

  Velocity: GROUNDED-on-platform spawns keep the varied random momentum (they're
  on solid ground, recoverable).  AIRBORNE spawns (over the gap void, or lifted
  meaningfully above the platform top) get their vx/vz shaped IFF
  ``recoverable_airborne`` -- each becomes a point on a landable ballistic arc
  toward the far platform (removes the guaranteed-dead spawns without any
  curriculum/ordering; pure per-spawn physics, mirrors landing.py's intent).
  ``recoverable_airborne=False`` reproduces the original unshaped spawn EXACTLY
  (the shaping only appends extra RNG draws and overwrites the airborne subset).
  """
  _ensure_crawl_buffers(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  dev = env.device
  asset = env.scene[asset_cfg.name]
  n = int(len(env_ids))
  gap_width = _gap_width(env)
  root = asset.data.default_root_state[env_ids].clone()
  origins = env.scene.env_origins[env_ids]

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), dev)

  # --- position: uniform across both platforms + the gap void ---
  x_rel = u(-_X_PAD, gap_width + _X_PAD)
  y = u(-0.15, 0.15)

  # --- pose: mix STANDING and CROUCHED (matching joint config via masks) ---
  stand_z = root[:, 2]                        # nominal standing height (~0.32)
  stand = u(0.0, 1.0) < _STAND_FRAC
  crouch = ~stand
  alpha = torch.where(crouch, u(0.0, 1.0), torch.zeros(n, device=dev))
  base_ref = torch.where(crouch, _crouch_z(alpha), stand_z)  # grounded equilib.
  z_air = u(0.0, _Z_AIR_MAX)                                  # lift off the feet
  z = base_ref + z_air + _Z_LIFT                              # grounded -> airborne

  pos = torch.stack([origins[:, 0] + x_rel, origins[:, 1] + y,
                     origins[:, 2] + z], dim=-1)
  euler = torch.stack([u(-_RP, _RP), u(-_RP, _RP), u(-_YAW, _YAW)], dim=1)
  quat = quat_mul(root[:, 3:7],
                  quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]))

  # --- velocity: all 6 base DOFs randomized (forward-biased vx) ---
  vel = torch.zeros(n, 6, device=dev)
  vel[:, 0] = u(_VX_LO, _VX_HI)
  vel[:, 1] = u(-_VLAT, _VLAT)
  vel[:, 2] = u(-_VLAT, _VLAT)
  vel[:, 3] = u(-_WANG, _WANG)
  vel[:, 4] = u(-_WANG, _WANG)
  vel[:, 5] = u(-_WANG, _WANG)

  # --- recoverable-airborne shaping: replace vx/vz of AIRBORNE spawns with a
  # landable ballistic arc onto the far platform (grounded momentum untouched;
  # small vy + angular + orientation noise kept). Extra RNG draws happen HERE,
  # AFTER the unshaped block above, so recoverable_airborne=False is byte-exact.
  if recoverable_airborne:
    over_void = (x_rel >= 0.0) & (x_rel <= gap_width)
    # Ballistic: base falls from z to the standing landing height z_plat=stand_z.
    h = z - stand_z                                          # drop to landing plane
    airborne = over_void | (h > _AIRBORNE_LIFT)             # no ground OR real airtime
    x_land = u(gap_width + _X_LAND_LO, gap_width + _X_LAND_HI)  # far-platform target
    vz_arc = u(_VZ_ARC_LO, _VZ_ARC_HI)                       # mid-arc up or down
    # Solve 0.5 g T^2 - vz T - h = 0 for the (descending) positive root, then
    # vx = horizontal_distance / T. Clamp T (avoid h~0 blow-up) and |vx|.
    disc = (vz_arc * vz_arc + 2.0 * _GRAV * h).clamp(min=0.0)
    T = ((vz_arc + torch.sqrt(disc)) / _GRAV).clamp(min=_T_MIN)
    vx_arc = ((x_land - x_rel) / T).clamp(-_VX_ARC_MAX, _VX_ARC_MAX)
    vel[:, 0] = torch.where(airborne, vx_arc, vel[:, 0])
    vel[:, 2] = torch.where(airborne, vz_arc, vel[:, 2])

  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1),
                                    env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)

  # masks consumed by the crouch_joints event (apply_crouch_joints).
  env._crouch_mask[env_ids] = crouch
  env._crouch_alpha[env_ids] = alpha
  env._splay_mag[env_ids] = torch.zeros(n, device=dev)   # plain narrow crouch
  env._leg_jitter[env_ids] = torch.full((n,), 0.02, device=dev)


def unitree_go2_gap_raas_env_cfg(play: bool = False, gap_width: float = 0.20,
                                 recoverable_airborne: bool = True
                                 ) -> ManagerBasedRlEnvCfg:
  """RAAS gap env: the island gap terrain at ``gap_width`` (no curriculum) with
  the uniform-random RAAS spawn.  Shared by both recipe steps; only the
  margin_fn / learner differ (registered in tasks/go2_gap_raas.py).

  ``recoverable_airborne`` (default True) shapes the AIRBORNE spawns' velocity
  onto a landable ballistic arc toward the far platform -- separating the
  no-curriculum question from doomed-spawn pollution.  Pass
  ``recoverable_airborne=False`` (e.g. ``--env-override recoverable_airborne=False``)
  to reproduce the original fully-unshaped spawn exactly."""
  cfg = unitree_go2_harvest_env_cfg(play=play, gap_width=gap_width)
  cfg.episode_length_s = 4.0

  # Uniform-random spawn (no curriculum, no structured sampling). KEEP
  # reset_robot_joints (default joints for STANDING envs) and add crouch_joints
  # LAST so it overwrites the joint pose for CROUCHED envs (reads the masks that
  # reset_base sets).  Ordering: reset_base -> reset_robot_joints -> crouch_joints.
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_gap_raas, mode="reset",
    params={"recoverable_airborne": recoverable_airborne})
  cfg.events["crouch_joints"] = EventTermCfg(func=apply_crouch_joints,
                                             mode="reset", params={})
  if "push_robot" in cfg.events:
    cfg.events.pop("push_robot", None)

  cfg.curriculum = {}                            # no curriculum term
  return cfg
