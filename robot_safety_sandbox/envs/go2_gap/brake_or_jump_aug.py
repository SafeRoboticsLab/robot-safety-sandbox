"""Stage-2 RAS v2 — brake_or_jump reverse curriculum @ gap 0.30 with
an AUGMENTED decision-band spawn mixture that teaches STRONGER origination launches
(so the launches land in the weak-launch lander's competent deep-in-gap envelope; Stage-2's
Stage-1 finding was that weak launches from rest are near-uncatchable).

Everything is the naive-RA `_w30` recipe verbatim — same island terrain @0.30, same
1175-d obs, margins UNCHANGED (`compose(g_terrain_relative, l_stable_far)`), same
reverse-curriculum machinery and `brake_or_jump_levels` gate — EXCEPT the reset
event. At the near-edge (non-committed) curriculum levels L>K_COMMIT, a fraction of
resets is replaced by three origination-shaping spawn kinds:

  runway (30%)  standing default-pose spawn on the near platform at x_rel in
                [-R_MAX, -0.35], ZERO momentum, small yaw/lateral jitter — makes
                retreat / run-up discoverable. R_MAX is set to the MEASURED usable
                runway (scratchpad/ras_runway_probe.py): the near platform is solid
                ground for x_rel in [-2.8, 0] and the BACK PIT (plunge) begins at
                x_rel <= -3.0, so R_MAX=2.5 leaves a 0.5 m pit-edge margin.
  deep-crouch (15%)  the deepest-base_z decile of the GROUNDED pre-launch frames in
                the weak-launch harvest manifold (the deepest crouches the RAS arm exhibits),
                over-sampled and restored via the bank `_restore`.
  synthetic cold-edge (15%)  arbitrary-pose near-edge states (x_rel in
                [-0.35, 0], grounded, varied pose, vx in [0, 2]) — puts the
                deployment distribution IN training so the critic is calibrated on
                the cold edge (replaces a separate Stage-3 calibration pass).
  curriculum (40%)  the existing `_spawn_level` behaviour (also ALL committed L<=K).

Both arms (ra_v2 / ras_v2) share this env; only the {gate + lander + policy_mask}
bundle differs, wired at task registration (tasks/go2_gap_brake_or_jump_aug.py).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from robot_safety_sandbox.envs.go2_gap.brake_or_jump import (
  D,
  K_COMMIT,
  _ensure,
  _restore,
  _spawn_level,
  unitree_go2_brake_or_jump_env_cfg,
)

# --- measured runway (scratchpad/ras_runway_probe.py, 2026-08-26) --------------
# Standing spawns settle (z~0.245) for x_rel in [-2.8, 0]; plunge into the back pit
# (z~-0.9) at x_rel <= -3.0. R_MAX=2.5 -> spawn range [-2.5, -0.35], 0.5 m margin.
R_MAX = 2.5
_RUNWAY_X = (-R_MAX, -0.35)
_SYNTH_X = (-0.35, 0.0)
_SYNTH_VX = (0.0, 2.0)

# spawn-mix fractions OF THE DECISION-BAND (near-edge, L>K_COMMIT) resets. The
# remaining 1-(sum) keeps the existing curriculum behaviour; committed L<=K_COMMIT
# is ALWAYS pure curriculum (kept as-is per the directive).
_F_RUNWAY, _F_CROUCH, _F_SYNTH = 0.30, 0.15, 0.15

_MANIFOLD_PATH = os.environ.get(
  "RAAS_MANIFOLD_PATH",
  os.path.expanduser(
    "~/artifacts/robot-safety-sandbox/lander-v2/weaklaunch_manifold.npz"))
_KIND_LAUNCH_GROUND = 1        # harvest kind code for grounded pre-launch frames
_CROUCH_DECILE = 0.10          # lowest-base_z decile = deepest crouches
_DEEPCROUCH: dict = {}


def _deepcrouch_bank(env) -> torch.Tensor:
  """The deepest-base_z decile of the grounded pre-launch (kind==1) frames from
  the weak-launch harvest manifold, as a [N, 37] bank-format tensor (cached)."""
  key = _MANIFOLD_PATH
  if key not in _DEEPCROUCH:
    d = np.load(_MANIFOLD_PATH)
    rows = d["rows"]
    grounded = rows[d["kind"] == _KIND_LAUNCH_GROUND]
    z = grounded[:, 1]                                   # col 1 = base height
    thr = float(np.quantile(z, _CROUCH_DECILE))
    deep = grounded[z <= thr]
    _DEEPCROUCH[key] = torch.tensor(np.ascontiguousarray(deep), dtype=torch.float32,
                                    device=env.device)
    print(f"[brake_or_jump_aug] deep-crouch bank: {deep.shape[0]} rows "
          f"(base_z<= {thr:.3f} of {grounded.shape[0]} grounded) <- {_MANIFOLD_PATH}")
  return _DEEPCROUCH[key]


def _spawn_standing(env, ids, x_lo, x_hi, vx_lo, vx_hi, rp_j, yaw_j, lat_j,
                    zero_vel):
  """Standing DEFAULT-pose spawn at x_rel~U[x_lo,x_hi] with default joints (+ tiny
  jitter). ``zero_vel`` -> at rest (runway); else forward vx~U[vx_lo,vx_hi] plus the
  small lateral/angular noise of the cold-edge generator. brake_or_jump popped
  ``reset_robot_joints``, so the joint pose is written here explicitly."""
  asset = env.scene[SceneEntityCfg("robot").name]
  dev = env.device
  n = int(len(ids))

  def u(lo, hi, shape=(n,)):
    return sample_uniform(lo, hi, shape, dev)

  root = asset.data.default_root_state[ids].clone()
  origins = env.scene.env_origins[ids]
  x_rel = u(x_lo, x_hi)
  y = u(-lat_j, lat_j)
  pos = torch.stack([origins[:, 0] + x_rel, origins[:, 1] + y,
                     origins[:, 2] + root[:, 2]], dim=-1)
  euler = torch.stack([u(-rp_j, rp_j), u(-rp_j, rp_j), u(-yaw_j, yaw_j)], dim=1)
  quat = quat_mul(root[:, 3:7],
                  quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]))
  vel = torch.zeros(n, 6, device=dev)
  if not zero_vel:
    vel[:, 0] = u(vx_lo, vx_hi)
    vel[:, 1] = u(-0.15, 0.15)
    vel[:, 3] = u(-0.20, 0.20)
    vel[:, 4] = u(-0.20, 0.20)
    vel[:, 5] = u(-0.20, 0.20)
  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
  asset.write_root_link_velocity_to_sim(vel, env_ids=ids)
  jp = asset.data.default_joint_pos[ids].clone()
  jp = jp + u(-0.05, 0.05, jp.shape)
  jv = torch.zeros_like(asset.data.default_joint_vel[ids])
  asset.write_joint_state_to_sim(jp, jv, env_ids=ids)


def _spawn_deepcrouch(env, ids):
  """Restore a random deep-crouch grounded pre-launch frame at its own harvested
  x_rel, no momentum scaling (via the bank `_restore`)."""
  bank = _deepcrouch_bank(env)
  dev = env.device
  n = int(len(ids))
  pick = torch.randint(0, bank.shape[0], (n,), device=bank.device)
  rows = bank[pick].to(dev)
  _restore(env, ids, rows, rows[:, 0], torch.ones(n, device=dev))


def reset_brake_or_jump_aug(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Augmented reset: committed levels (L<=K_COMMIT) and 40% of the near-edge
  (decision-band + final) levels keep the existing curriculum spawn; the other 60%
  of near-edge resets draw runway / deep-crouch / synthetic-cold-edge origination
  spawns. Curriculum gate is unchanged (`brake_or_jump_levels`)."""
  _ensure(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  dev = env.device
  # peek/level logic mirrors reset_brake_or_jump exactly.
  peek = torch.rand(len(env_ids), device=dev) < 0.20
  env._peek[env_ids] = peek
  lvl = torch.where(peek, min(env._L + 1, D), env._L).long()

  aug = lvl > K_COMMIT                                   # near-edge origination band
  r = torch.rand(len(env_ids), device=dev)
  runway = aug & (r < _F_RUNWAY)
  crouch = aug & (r >= _F_RUNWAY) & (r < _F_RUNWAY + _F_CROUCH)
  synth = aug & (r >= _F_RUNWAY + _F_CROUCH) & (r < _F_RUNWAY + _F_CROUCH + _F_SYNTH)
  curric = ~(runway | crouch | synth)                   # committed + 40% decision

  # per-env spawn-kind record (0 curriculum, 1 runway, 2 deep-crouch, 3 synth) --
  # for the smoke check + any per-kind outcome analysis; harmless in training.
  if not hasattr(env, "_aug_kind"):
    env._aug_kind = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
  kind = torch.zeros(len(env_ids), dtype=torch.long, device=dev)
  kind[runway] = 1
  kind[crouch] = 2
  kind[synth] = 3
  env._aug_kind[env_ids.long()] = kind

  if bool(curric.any()):
    _spawn_level(env, env_ids[curric], lvl[curric])
  if bool(runway.any()):
    _spawn_standing(env, env_ids[runway], *_RUNWAY_X, 0.0, 0.0,
                    rp_j=0.0, yaw_j=0.2, lat_j=0.10, zero_vel=True)
  if bool(synth.any()):
    _spawn_standing(env, env_ids[synth], *_SYNTH_X, *_SYNTH_VX,
                    rp_j=0.12, yaw_j=0.25, lat_j=0.12, zero_vel=False)
  if bool(crouch.any()):
    _spawn_deepcrouch(env, env_ids[crouch])

  # airborne-clean latches: fresh per episode (mirrors reset_brake_or_jump).
  env._td_og[env_ids.long()] = False
  env._bodyplant[env_ids.long()] = False


def unitree_go2_brake_or_jump_aug_env_cfg(play: bool = False,
                                          gap_width: float = 0.30
                                          ) -> ManagerBasedRlEnvCfg:
  """The naive-RA `_w30` env verbatim (curriculum `brake_or_jump_levels`, margins
  set at the task), with the reset event swapped for the augmented spawn mixture."""
  cfg = unitree_go2_brake_or_jump_env_cfg(play=play, gap_width=gap_width,
                                          clean=False)
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_brake_or_jump_aug, mode="reset", params={})
  return cfg
