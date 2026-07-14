"""Safety-filter crossing task: arrive with random momentum, reach a SAFE STOP.

The reach target ``l`` is "come to a safe stop" (rest mode, in the reach-avoid
wrapper), NOT forward motion.  So braking / a single jump / chaining a cluster
all emerge as instrumental ways to reach safe rest given the arrival momentum —
the deployment safety-filter objective.  A mild bias credits resting further
along, so the robot crosses when a crossing is safely available.

Spawn = the takeover distribution: on the approach at a random position with a
random forward momentum (0 .. ~3 m/s), the states a nominal policy would hand to
the filter.  No launch assist.  Warm-starts from the single-gap crossing
(model_4000), whose stop-after-landing is now the desired target behavior.

Curriculum: SURVIVE-based (promote an env when it reaches safe rest = survives to
timeout; demote when it falls).  Distance-independent, so correctly stopping
short (slow arrival) is not penalized the way a distance curriculum would.
"""

from __future__ import annotations

import os
from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from robot_safety_sandbox.envs.terrains.safety_filter import SAFETY_FILTER_TERRAINS_CFG

# Arrival-momentum range (m/s) = the nominal-policy takeover distribution.
_VX_MIN = 0.0
_VX_MAX = 3.0
_GAP_X = 2.5     # gap-cluster start = SafetyFilterTerrainCfg.approach_length (jitter 0)
_A_BRAKE = 3.0   # measured effective braking decel (m/s^2): 2.0 m/s stop in ~0.65 m
_VX_CAP = 3.4
# Spawn mixture fractions: stoppable ground / UNSTOPPABLE ground / mid-air /
# HANDOVER-REPLAY (real mid-gait walker states — the filter takeover
# distribution; active only when the harvested dataset exists).
_FRAC_STOPPABLE = 0.3
_FRAC_UNSTOPPABLE = 0.3
_FRAC_MIDAIR = 0.2  # remainder (0.2) = handover replay
_HANDOVER_DATASET = os.environ.get(
  "SAFE_ZOO_HANDOVER_DATASET",
  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
               "data", "walker_handover_states.pt"),
)
_handover_cache: dict = {}


def _handover_data(device):
  """Lazily load the harvested walker-state dataset onto the env device.

  Sorted by speed ascending so the handover reverse curriculum can sample
  difficulty bands by index range.
  """
  if "data" not in _handover_cache:
    if os.path.exists(_HANDOVER_DATASET):
      raw = torch.load(_HANDOVER_DATASET, map_location=device, weights_only=False)
      order = torch.argsort(raw["speed"])
      _handover_cache["data"] = {k: v[order].to(device) for k, v in raw.items()}
      print(f"[crossing_chain] handover dataset loaded (speed-sorted): "
            f"{len(raw['z'])} states from {_HANDOVER_DATASET}")
    else:
      _handover_cache["data"] = None
  return _handover_cache["data"]


# Handover reverse curriculum: level 0 = SLOW walker states spawned FAR from
# the gap (brake-and-settle, easy) ... _HANDOVER_LEVELS = FAST states at the
# edge (the filter's actual engagement slice: measured 5% survival untrained).
_HANDOVER_LEVELS = 5


def _handover_band(level, n_rows):
  """Speed-index window (into the speed-sorted dataset) for a difficulty level."""
  lo_f = 0.15 * level          # L0: 0.00-0.30 (slowest) ... L5: 0.75-1.00 (fastest)
  hi_f = min(1.0, lo_f + 0.30)
  return int(lo_f * n_rows), max(int(hi_f * n_rows) - 1, 1)
# Per-env reverse curriculum on the TAKEOFF (the proven landing->launch trick):
# level 0 = full assist (spawn at the edge with launch-like vz, a state the
# warm-start already converts) ... _JUMP_LEVELS = no assist (run-up jump from
# flat ground, far from the edge).  Promoted per env on unstoppable successes.
_JUMP_LEVELS = 5


def _ensure_jump_buffers(env):
  if not hasattr(env, "_jump_level"):
    env._jump_level = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._was_unstoppable = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
  if not hasattr(env, "_handover_mask"):
    env._handover_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._handover_jpos = torch.zeros(env.num_envs, 12, device=env.device)
    env._handover_jvel = torch.zeros(env.num_envs, 12, device=env.device)
    env._handover_level = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def reset_takeover(env, env_ids, asset_cfg=SceneEntityCfg("robot"),
                   edge_margin: float = 0.0):
  """Takeover-distribution spawn, stratified across the decision boundary.

  Three strata (the buffer must contain the jump-or-die states, or the policy
  converges to always-brake and never learns to generate a jump):

  * STOPPABLE ground arrival: distance-to-gap d and speed with braking distance
    ``v^2/(2a) < d`` -> the correct behavior is to brake to rest (maintains the
    braking skill).
  * UNSTOPPABLE ground arrival: ``v^2/(2a) > d`` by construction -> braking runs
    into the gap (g<0); the ONLY V>0 path is to jump.  A small upward vz
    (gait-bounce, up to ~0.5 m/s -- mid-stride handover states) bridges toward
    the launch states the warm-start already converts into crossings, giving the
    rare-win gradient a foothold.
  * MID-AIR over the first gap: finish-the-jump / landing maintenance (the
    landing task's key trick).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()

  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), device)

  _ensure_jump_buffers(env)
  hdata = _handover_data(device)
  r = u(0.0, 1.0)
  stoppable = r < _FRAC_STOPPABLE
  unstoppable = (r >= _FRAC_STOPPABLE) & (r < _FRAC_STOPPABLE + _FRAC_UNSTOPPABLE)
  midair_hi = _FRAC_STOPPABLE + _FRAC_UNSTOPPABLE + _FRAC_MIDAIR
  if hdata is None:
    # No dataset: fold the handover share into mid-air (previous behavior).
    midair_hi = 1.0
  midair = (r >= _FRAC_STOPPABLE + _FRAC_UNSTOPPABLE) & (r < midair_hi)
  handover = ~stoppable & ~unstoppable & ~midair
  env._was_unstoppable[env_ids] = unstoppable
  env._handover_mask[env_ids] = handover

  # --- ground strata: sample distance-to-gap, derive speed from the boundary.
  d = u(0.3, 2.2)                       # distance to the gap cluster (m)
  # Unstoppable stratum: reverse curriculum on the takeoff.  assist=1 -> spawn
  # AT the edge with launch-like vz (the state the warm-start converts);
  # assist->0 -> spawn progressively further back with no vz help.
  assist = (1.0 - env._jump_level[env_ids].float() / _JUMP_LEVELS).clamp(0.0, 1.0)
  d_unstoppable = u(0.25, 0.55) + (1.0 - assist) * u(0.0, 1.2)
  d = torch.where(unstoppable, d_unstoppable, d)
  # With a robustified rest set (edge_margin > 0), "stoppable" means brakeable
  # to rest AT LEAST edge_margin before the gap — shifts the decision boundary.
  v_boundary = torch.sqrt(2.0 * _A_BRAKE * (d - edge_margin).clamp_min(0.05))
  vx = torch.where(
    stoppable,
    (v_boundary * u(0.15, 0.90)).clamp(_VX_MIN, _VX_MAX),
    torch.maximum(u(2.2, 3.2), v_boundary * 1.1).clamp(0.8, _VX_CAP),
  )
  x = (_GAP_X - d).clamp(min=0.15)
  z = 0.05 + u(-0.02, 0.02)
  vz = torch.where(
    unstoppable,
    assist * u(0.35, 0.55) + u(0.0, 0.15),  # launch-like at full assist -> ~0
    u(-0.05, 0.05),
  )

  # --- mid-air stratum: over the first gap, mid-arc.
  x = torch.where(midair, _GAP_X + u(0.0, 0.45), x)
  z = torch.where(midair, u(0.15, 0.40), z)
  vx = torch.where(midair, u(1.8, 2.8), vx)
  vz = torch.where(midair, u(-0.5, 0.3), vz)

  pose = torch.stack(
    [x, u(-0.06, 0.06), z,
     u(-0.05, 0.05), u(-0.05, 0.05), u(-0.06, 0.06)], dim=1)
  vel = torch.stack(
    [vx, u(-0.10, 0.10), vz, u(-0.10, 0.10), u(-0.10, 0.10), u(-0.10, 0.10)],
    dim=1)

  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
  )
  velocities = root[:, 7:13] + vel

  # --- HANDOVER-REPLAY stratum: real mid-gait walker states, placed on the
  # approach at a random distance to the gap.  Root pose/vel/orientation and
  # joint states all come from the dataset (joints applied by the
  # ``handover_joints`` event which runs after ``reset_robot_joints``).
  if hdata is not None and bool(handover.any()):
    h_idx = handover.nonzero().flatten()
    # Reverse curriculum: sample speed band + gap distance by per-env level.
    lvl = env._handover_level[env_ids][h_idx].float()
    n_rows = len(hdata["z"])
    lo_f = 0.15 * lvl
    hi_f = (lo_f + 0.30).clamp(max=1.0)
    frac = torch.rand(len(h_idx), device=device)
    rows = ((lo_f + frac * (hi_f - lo_f)) * (n_rows - 1)).long()
    # distance to gap: far at level 0 -> at the edge at level 5
    ease = (1.0 - lvl / _HANDOVER_LEVELS)
    d_lo = 0.25 + ease * 0.75          # L0: 1.00 ... L5: 0.25
    d_hi = 0.60 + ease * 1.15          # L0: 1.75 ... L5: 0.60
    d_h = d_lo + torch.rand(len(h_idx), device=device) * (d_hi - d_lo)
    positions[h_idx, 0] = env.scene.env_origins[env_ids][h_idx, 0] + _GAP_X - d_h
    positions[h_idx, 1] = (env.scene.env_origins[env_ids][h_idx, 1]
                           + sample_uniform(-0.06, 0.06, (len(h_idx),), device))
    positions[h_idx, 2] = hdata["z"][rows]
    orientations[h_idx] = hdata["quat"][rows]
    velocities[h_idx, 0:3] = hdata["lin_vel_w"][rows]
    velocities[h_idx, 3:6] = hdata["ang_vel_w"][rows]
    env._handover_jpos[env_ids[h_idx]] = hdata["joint_pos"][rows]
    env._handover_jvel[env_ids[h_idx]] = hdata["joint_vel"][rows]

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def apply_handover_joints(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Write the replayed joint states for handover-stratum envs.  Registered
  AFTER ``reset_robot_joints`` so the default-pose randomization doesn't
  clobber the mid-gait joint configuration."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0 or not hasattr(env, "_handover_mask"):
    return
  mask = env._handover_mask[env_ids]
  if not bool(mask.any()):
    return
  asset = env.scene[asset_cfg.name]
  ids = env_ids[mask.nonzero().flatten()]
  asset.write_joint_state_to_sim(
    env._handover_jpos[ids], env._handover_jvel[ids], env_ids=ids
  )


def phase_random_offset(env, period: float = 0.5, command_name: str = "twist"):
  """Gait clock with a PER-EPISODE random phase offset (no stand-mask).

  Training episodes always reset the clock to zero, so the policy learns
  states paired with phase~0 at spawn — but at filter-handover time the clock
  offset is arbitrary.  Randomizing the offset per episode makes the policy
  phase-offset invariant, matching deployment.  (The stand-mask is dropped for
  the same reason as in the play script: the command is a constant 1.0 in this
  task, so the mask never fired in training.)
  """
  if not hasattr(env, "_phase_offset"):
    env._phase_offset = torch.rand(env.num_envs, device=env.device)
  fresh = env.episode_length_buf == 0
  if bool(fresh.any()):
    env._phase_offset[fresh] = torch.rand(int(fresh.sum()), device=env.device)
  global_phase = ((env.episode_length_buf * env.step_dt) / period
                  + env._phase_offset) % 1.0
  out = torch.zeros(env.num_envs, 2, device=env.device)
  out[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  out[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  return out


def jump_assist_levels(env, env_ids) -> torch.Tensor:
  """Per-env reverse curriculum on the takeoff assist: an env that survived an
  UNSTOPPABLE episode (jumped and settled -> timeout) is promoted (less assist,
  spawn further back); one that fell is demoted.  Stoppable/mid-air episodes
  leave the level unchanged.  Runs at reset, BEFORE reset events re-roll the
  stratum, so ``_was_unstoppable`` still reflects the ended episode."""
  _ensure_jump_buffers(env)
  was_un = env._was_unstoppable[env_ids]
  time_outs = env.termination_manager.time_outs[env_ids]
  lvl = env._jump_level[env_ids]
  lvl = torch.where(was_un & time_outs, lvl + 1, lvl)
  lvl = torch.where(was_un & ~time_outs, lvl - 1, lvl)
  env._jump_level[env_ids] = lvl.clamp(0, _JUMP_LEVELS)
  return env._jump_level.float().mean()


def handover_levels(env, env_ids) -> torch.Tensor:
  """Per-env reverse curriculum for the handover-replay stratum: survive a
  handover episode (reach rest -> timeout) => promote to faster states spawned
  closer to the gap; fall => demote.  Same machinery as jump_assist."""
  _ensure_jump_buffers(env)
  was_h = env._handover_mask[env_ids]
  time_outs = env.termination_manager.time_outs[env_ids]
  lvl = env._handover_level[env_ids]
  lvl = torch.where(was_h & time_outs, lvl + 1, lvl)
  lvl = torch.where(was_h & ~time_outs, lvl - 1, lvl)
  env._handover_level[env_ids] = lvl.clamp(0, _HANDOVER_LEVELS)
  return env._handover_level.float().mean()


def safety_filter_levels(env, env_ids) -> torch.Tensor | None:
  """Terrain curriculum gated on the BINDING skill: promote only when an
  UNSTOPPABLE episode succeeded (jumped the gap and settled); demote on any
  fall.  Braking / mid-air successes neither promote nor demote — they work at
  any gap width, and letting them promote runs the terrain ahead of the jump
  frontier (observed: terrain level ~5 while jump_assist pinned at ~0.7)."""
  terrain = env.scene.terrain
  if terrain is None or not hasattr(terrain, "update_env_origins"):
    return None
  _ensure_jump_buffers(env)
  time_outs = env.termination_manager.time_outs[env_ids]
  was_un = env._was_unstoppable[env_ids]
  move_up = was_un & time_outs
  move_down = ~time_outs
  terrain.update_env_origins(env_ids, move_up, move_down)
  return terrain.terrain_levels.float().mean()


def unitree_go2_crossing_chain_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  from robot_safety_sandbox.envs.go2_gap.gap import (
    unitree_go2_gap_reach_avoid_env_cfg,
  )

  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  cfg.scene.terrain.terrain_generator = replace(SAFETY_FILTER_TERRAINS_CFG)

  # Long enough to brake to rest, or to chain a cluster and settle after it.
  cfg.episode_length_s = 8.0

  # Spawn = takeover momentum distribution (no launch assist).
  cfg.events["reset_base"] = EventTermCfg(func=reset_takeover, mode="reset", params={})
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  # Runs AFTER reset_robot_joints (dict append order): replayed mid-gait joint
  # states for the handover stratum survive the default-pose randomization.
  cfg.events["handover_joints"] = EventTermCfg(
    func=apply_handover_joints, mode="reset", params={}
  )
  cfg.events.pop("push_robot", None)

  # Phase-offset-invariant gait clock (see phase_random_offset): deployment
  # handovers occur at arbitrary clock offsets, training resets at phase 0.
  import copy as _copy
  for gname in ("proprioception", "critic"):
    term = _copy.deepcopy(cfg.observations[gname].terms["phase"])
    period = float(term.params.get("period", 0.5))
    term.func = phase_random_offset
    term.params = {"period": period}
    cfg.observations[gname].terms["phase"] = term

  # Survive-based difficulty curriculum (reach-safe-rest, not distance) +
  # per-env takeoff-assist reverse curriculum for the unstoppable stratum.
  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(func=safety_filter_levels),
    "jump_assist": CurriculumTermCfg(func=jump_assist_levels),
    "handover_level": CurriculumTermCfg(func=handover_levels),
  }
  return cfg


# --- ISAACS adversarial phase: pinned curricula -----------------------------
# Under adversarial pressure the survival-gated curricula demote (the task
# silently gets easier while the adversary strengthens — a treadmill). During
# the two-player game the task distribution is PINNED at the levels the
# warm-start policy mastered; robustification, not skill acquisition, is the
# objective here.

def pinned_levels(env, env_ids) -> torch.Tensor:
  _ensure_jump_buffers(env)
  env._jump_level[env_ids] = 4
  env._handover_level[env_ids] = 3
  terrain = env.scene.terrain
  if terrain is not None and hasattr(terrain, "update_env_origins"):
    # keep terrain rows where they are (no promote/demote); rows were assigned
    # at startup and randomize_terrain reshuffles columns/rows per reset.
    pass
  return env._jump_level.float().mean()


def unitree_go2_crossing_chain_isaacs_env_cfg(play: bool = False):
  cfg = unitree_go2_crossing_chain_env_cfg(play=play)
  cfg.events["reset_base"].params["edge_margin"] = 0.3
  cfg.curriculum = {"pinned_levels": CurriculumTermCfg(func=pinned_levels)}
  return cfg
