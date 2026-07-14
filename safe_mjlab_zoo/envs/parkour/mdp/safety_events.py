"""Event terms for safety training on parkour terrain.

The key event is :func:`reset_robot_midair_over_gaps`, which spawns a
fraction of the robots elevated above the terrain with a strong forward
velocity — giving the safety policy explicit exposure to mid-air
transitions over gaps.  Without such initialisation, a purely on-ground
spawn + walking command will rarely place the robot mid-flight, so the
policy never learns to recover/land after a jump.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


_GAIT_BANK_CACHE: dict = {}


def _load_gait_bank(path, device):
  import os
  key = (path, str(device))
  if key not in _GAIT_BANK_CACHE:
    d = torch.load(os.path.expanduser(path), map_location=device,
                   weights_only=True)
    _GAIT_BANK_CACHE[key] = d["bank"].float()
    print(f"[gait-bank] {d['bank'].shape[0]} states from {path}")
  return _GAIT_BANK_CACHE[key]


V9_LEVELS = 6


def _ensure_v9_level(env):
  if not hasattr(env, "_v9_level"):
    env._v9_level = torch.zeros(env.num_envs, dtype=torch.long,
                                device=env.device)


def reset_v9_reverse(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  *,
  arrival_bank_path: str,
  retention_fraction: float = 0.15,
  ground_pose_range: dict | None = None,
  ground_velocity_range: dict | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """v9 reverse-curriculum reset (plan of record 2026-07-11): per-env level
  0..5, spawns drawn from REAL-state banks — never hand-interpolated poses,
  never bare teleports (the teleport≠gait lesson).

    A reverse curriculum needs its bottom rung INSIDE the initial policy's
    competence (flat-ladder v1 put the campaign's hardest cell at the floor:
    levels collapsed to 0, ep_len 9). Success-gradient ladder:
    L0     certified midair over the gap (latch -> lander, ~98%): the anchor.
    L1     teleport-committed lip, vx 2.8-3.5 (launcher home, ~60%).
    L2-L5  arrival bank (real mid-stride states) windowed by harvested x:
           (-0.15,-0.05) / (-0.35,-0.15) / (-0.6,-0.35) / (-0.8,-0.6) —
           the realism+distance wall, approached with training momentum.
    L6     natural walk-in spawn (ground_pose/velocity ranges).

  retention_fraction of resets sample a uniformly lower level (stage mixture:
  don't forget takeoff while learning approach). Promotion/demotion happens
  in v9_reverse_levels (curriculum term)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  _ensure_v9_level(env)
  device = env.device
  n = len(env_ids)
  lvl = env._v9_level[env_ids].clone()
  # stage-mixture retention: some envs replay a lower level
  retain = torch.rand(n, device=device) < float(retention_fraction)
  lower = (torch.rand(n, device=device) * lvl.float()).long()
  lvl = torch.where(retain & (lvl > 0), lower, lvl)
  # v10.3: fixed sampling mass on the slow-delivery rungs regardless of the
  # population's level (colleague directive: the frontier must not starve
  # the states the benchmark actually measures).
  if delivery_floor_fraction > 0.0:
    floor = torch.rand(n, device=device) < float(delivery_floor_fraction)
    dlvl = 4 + (torch.rand(n, device=device) * 3).long().clamp(0, 2)
    lvl = torch.where(floor, dlvl, lvl)

  # L5 (and fallback): natural walk-in via the standard ground reset
  reset_robot_midair_over_gaps(
    env, env_ids, midair_fraction=0.0,
    ground_pose_range=ground_pose_range,
    ground_velocity_range=ground_velocity_range, asset_cfg=asset_cfg)

  asset: Entity = env.scene[asset_cfg.name]
  ab = _load_gait_bank(arrival_bank_path, device)   # x,z,quat,linv,angv,jp,jv

  def _spawn_rows(ids, rows, x_override=None):
    if len(ids) == 0:
      return
    x = rows[:, 0] if x_override is None else x_override
    x = x + sample_uniform(-0.02, 0.02, (len(ids),), device)
    z, quat = rows[:, 1:2], rows[:, 2:6]
    linv, angv = rows[:, 6:9], rows[:, 9:12]
    jp, jv = rows[:, 12:24], rows[:, 24:36]
    origins = env.scene.env_origins[ids]
    pos = torch.stack([origins[:, 0] + x, origins[:, 1],
                       origins[:, 2] + z.squeeze(-1)], dim=-1)
    asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1),
                                      env_ids=ids)
    asset.write_root_link_velocity_to_sim(torch.cat([linv, angv], dim=-1),
                                          env_ids=ids)
    asset.write_joint_state_to_sim(jp, jv, env_ids=ids)

  # L0: certified midair (immediate/near-immediate latch -> lander finishes).
  ids0 = env_ids[lvl == 0]
  if len(ids0) > 0:
    reset_robot_midair_over_gaps(
      env, ids0, midair_fraction=1.0,
      ground_pose_range=ground_pose_range,
      ground_velocity_range=ground_velocity_range,
      midair_x_range=(0.0, 0.35), midair_y_range=(-0.1, 0.1),
      midair_z_range=(0.15, 0.45),
      midair_vx_range=(2.0, 3.0), midair_vy_range=(-0.1, 0.1),
      midair_vz_range=(-0.3, 1.0),
      midair_roll_range=(-0.15, 0.15), midair_pitch_range=(-0.2, 0.2),
      midair_yaw_range=(-0.1, 0.1), asset_cfg=asset_cfg)

  # L1: teleport-committed lip (the launcher prior's home distribution).
  ids1 = env_ids[lvl == 1]
  if len(ids1) > 0:
    reset_robot_midair_over_gaps(
      env, ids1, midair_fraction=0.0,
      ground_pose_range=ground_pose_range,
      ground_velocity_range=ground_velocity_range,
      ground_committed_fraction=1.0,
      ground_committed_pose_range={"x": (-0.25, -0.05), "y": (-0.1, 0.1),
                                   "yaw": (-0.1, 0.1)},
      ground_committed_velocity_range={"x": (2.8, 3.5), "y": (-0.1, 0.1)},
      asset_cfg=asset_cfg)

  x_windows = {2: (-0.15, -0.05), 3: (-0.35, -0.15), 4: (-0.6, -0.35),
               5: (-0.8, -0.6)}
  for L, (xlo, xhi) in x_windows.items():
    m = lvl == L
    ids = env_ids[m]
    if len(ids) == 0 or ab.shape[0] == 0:
      continue
    cand = (ab[:, 0] >= xlo) & (ab[:, 0] <= xhi)
    pool = ab[cand] if bool(cand.any()) else ab
    rows = pool[torch.randint(0, pool.shape[0], (len(ids),), device=device)]
    _spawn_rows(ids, rows)
  # L5 envs keep the walk-in reset already applied above.


V10_LEVELS = 7


def _ensure_v10(env):
  if not hasattr(env, "_v10_level"):
    z = lambda: torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._v10_level, env._v10_succ, env._v10_fail = z(), z(), z()


def reset_v10_bridge(
  env, env_ids, *,
  arrival_bank_path: str,
  delivery_bank_path: str,
  retention_fraction: float = 0.15,
  delivery_floor_fraction: float = 0.0,
  ground_pose_range: dict | None = None,
  ground_velocity_range: dict | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """v10 bridge ladder (colleague directive 2026-07-12): continuous rungs
  down to the TRUE delivery distribution.
    L0 certified midair (anchor) | L1 teleport lip | L2-L3 arrival bank
    (fast real gait, x(-0.15,-0.05)/(-0.35,-0.15)) | L4-L6 DELIVERY bank
    (slow walker states: x(-0.4,-0.15)/(-0.8,-0.4)/(-1.2,-0.8)) | L7 walk-in.
  Promotion uses hysteresis (v10_bridge_levels): 2 consecutive successes to
  advance, 2 consecutive failures to demote — kills the +-1 oscillation that
  inflated v9's mean level."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  _ensure_v10(env)
  device = env.device
  n = len(env_ids)
  lvl = env._v10_level[env_ids].clone()
  retain = torch.rand(n, device=device) < float(retention_fraction)
  lower = (torch.rand(n, device=device) * lvl.float()).long()
  lvl = torch.where(retain & (lvl > 0), lower, lvl)
  # v10.3: fixed sampling mass on the slow-delivery rungs regardless of the
  # population's level (colleague directive: the frontier must not starve
  # the states the benchmark actually measures).
  if delivery_floor_fraction > 0.0:
    floor = torch.rand(n, device=device) < float(delivery_floor_fraction)
    dlvl = 4 + (torch.rand(n, device=device) * 3).long().clamp(0, 2)
    lvl = torch.where(floor, dlvl, lvl)

  reset_robot_midair_over_gaps(
    env, env_ids, midair_fraction=0.0,
    ground_pose_range=ground_pose_range,
    ground_velocity_range=ground_velocity_range, asset_cfg=asset_cfg)
  asset: Entity = env.scene[asset_cfg.name]
  ab = _load_gait_bank(arrival_bank_path, device)
  db = _load_gait_bank(delivery_bank_path, device)

  def _spawn_rows(ids, rows):
    if len(ids) == 0:
      return
    x = rows[:, 0] + sample_uniform(-0.02, 0.02, (len(ids),), device)
    z, quat = rows[:, 1:2], rows[:, 2:6]
    linv, angv = rows[:, 6:9], rows[:, 9:12]
    jp, jv = rows[:, 12:24], rows[:, 24:36]
    origins = env.scene.env_origins[ids]
    pos = torch.stack([origins[:, 0] + x, origins[:, 1],
                       origins[:, 2] + z.squeeze(-1)], dim=-1)
    asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1),
                                      env_ids=ids)
    asset.write_root_link_velocity_to_sim(torch.cat([linv, angv], dim=-1),
                                          env_ids=ids)
    asset.write_joint_state_to_sim(jp, jv, env_ids=ids)

  ids0 = env_ids[lvl == 0]
  if len(ids0) > 0:
    reset_robot_midair_over_gaps(
      env, ids0, midair_fraction=1.0,
      ground_pose_range=ground_pose_range,
      ground_velocity_range=ground_velocity_range,
      midair_x_range=(0.0, 0.35), midair_y_range=(-0.1, 0.1),
      midair_z_range=(0.15, 0.45), midair_vx_range=(2.0, 3.0),
      midair_vy_range=(-0.1, 0.1), midair_vz_range=(-0.3, 1.0),
      midair_roll_range=(-0.15, 0.15), midair_pitch_range=(-0.2, 0.2),
      midair_yaw_range=(-0.1, 0.1), asset_cfg=asset_cfg)
  ids1 = env_ids[lvl == 1]
  if len(ids1) > 0:
    reset_robot_midair_over_gaps(
      env, ids1, midair_fraction=0.0,
      ground_pose_range=ground_pose_range,
      ground_velocity_range=ground_velocity_range,
      ground_committed_fraction=1.0,
      ground_committed_pose_range={"x": (-0.25, -0.05), "y": (-0.1, 0.1),
                                   "yaw": (-0.1, 0.1)},
      ground_committed_velocity_range={"x": (2.8, 3.5), "y": (-0.1, 0.1)},
      asset_cfg=asset_cfg)
  for L, (bank, xlo, xhi) in {2: (ab, -0.15, -0.05), 3: (ab, -0.35, -0.15),
                              4: (db, -0.4, -0.15), 5: (db, -0.8, -0.4),
                              6: (db, -1.2, -0.8)}.items():
    ids = env_ids[lvl == L]
    if len(ids) == 0 or bank.shape[0] == 0:
      continue
    cand = (bank[:, 0] >= xlo) & (bank[:, 0] <= xhi)
    pool = bank[cand] if bool(cand.any()) else bank
    rows = pool[torch.randint(0, pool.shape[0], (len(ids),), device=device)]
    _spawn_rows(ids, rows)
  # L7: walk-in reset already applied.


def v10_bridge_levels(env, env_ids) -> torch.Tensor:
  """Hysteresis promotion: 2 consecutive successes -> +1; 2 consecutive
  failures -> -1. Success = time-out truncation WITH gap progress (x>0.1:
  certified entry fires airborne past the face; far-side stabilization is
  x>0.9). v10.1 bug: any-timeout counted as success, so 10s of STANDING
  promoted — the ladder climbed to 6.0 on idling while the composed
  benchmark measured 5%. Predicates must be composed task success."""
  _ensure_v10(env)
  robot = env.scene["robot"]
  x_rel = (robot.data.root_link_pos_w[env_ids, 0]
           - env.scene.env_origins[env_ids, 0])
  ok = env.termination_manager.time_outs[env_ids] & (x_rel > 0.1)
  succ, fail = env._v10_succ[env_ids], env._v10_fail[env_ids]
  succ = torch.where(ok, succ + 1, torch.zeros_like(succ))
  fail = torch.where(ok, torch.zeros_like(fail), fail + 1)
  lvl = env._v10_level[env_ids]
  lvl = torch.where(succ >= 2, lvl + 1, lvl)
  succ = torch.where(succ >= 2, torch.zeros_like(succ), succ)
  lvl = torch.where(fail >= 2, lvl - 1, lvl)
  fail = torch.where(fail >= 2, torch.zeros_like(fail), fail)
  env._v10_level[env_ids] = lvl.clamp(0, V10_LEVELS)
  env._v10_succ[env_ids], env._v10_fail[env_ids] = succ, fail
  return env._v10_level.float().mean()


# --- Target-homotopy curriculum (professor's xi-anneal, dilation form) ------

HOMOTOPY_R_SCHEDULE = [34.0, 30.0, 26.0, 22.0, 20.0, 16.0, 10.0, 6.0, 0.0]


def _ensure_homotopy(env):
  if not hasattr(env, "_homo_stage"):
    env._homo_stage = 0
    env._homo_entered = torch.zeros(env.num_envs, dtype=torch.bool,
                                    device=env.device)
    env._homo_window = []          # rolling per-episode entry outcomes


def homotopy_r(env) -> float:
  _ensure_homotopy(env)
  i = min(env._homo_stage, len(HOMOTOPY_R_SCHEDULE) - 1)
  return HOMOTOPY_R_SCHEDULE[i]


def homotopy_anneal(env, env_ids) -> torch.Tensor:
  """Gated r-anneal: advance to the next (smaller) r once the rolling
  target-entry rate at the CURRENT r exceeds the gate over a full window.
  Entry is recorded by the margin fn (l_hat >= 0 at any step). The final
  r=0 stage is the true task target; earlier stages are scaffolding."""
  _ensure_homotopy(env)
  if env_ids is None or len(env_ids) == 0:
    return torch.tensor(float(env._homo_stage))
  ent = env._homo_entered[env_ids]
  env._homo_window.extend(ent.cpu().tolist())
  env._homo_entered[env_ids] = False
  W, GATE = 4000, 0.6
  if len(env._homo_window) >= W:
    rate = sum(env._homo_window[-W:]) / W
    if rate >= GATE and env._homo_stage < len(HOMOTOPY_R_SCHEDULE) - 1:
      env._homo_stage += 1
      env._homo_window = []
      print(f"[homotopy] ADVANCE -> stage {env._homo_stage} "
            f"(r={homotopy_r(env)}), entry rate was {rate:.2f}", flush=True)
    elif len(env._homo_window) > 3 * W:
      env._homo_window = env._homo_window[-W:]
  return torch.tensor(float(env._homo_stage))


def reset_homotopy_mix(
  env, env_ids, *,
  arrival_bank_path: str,
  delivery_bank_path: str,
  ground_pose_range: dict | None = None,
  ground_velocity_range: dict | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """FIXED spawn mixture for the homotopy run (the xi-anneal is the ONLY
  curriculum): 50% natural walk-in, 20% delivery bank, 15% arrival bank,
  15% midair ballistic (landing retention)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  device = env.device
  n = len(env_ids)
  u = torch.rand(n, device=device)
  # base: walk-in ground reset for everyone; midair for the last 15%
  midair = u >= 0.85
  reset_robot_midair_over_gaps(
    env, env_ids, midair_fraction=0.0,
    ground_pose_range=ground_pose_range,
    ground_velocity_range=ground_velocity_range, asset_cfg=asset_cfg)
  ids_mid = env_ids[midair]
  if len(ids_mid) > 0:
    reset_robot_midair_over_gaps(
      env, ids_mid, midair_fraction=1.0,
      ground_pose_range=ground_pose_range,
      ground_velocity_range=ground_velocity_range,
      midair_x_range=(-0.1, 0.5), midair_y_range=(-0.1, 0.1),
      midair_z_range=(0.05, 0.45), midair_vx_range=(1.5, 3.2),
      midair_vy_range=(-0.1, 0.1), midair_vz_range=(-1.0, 1.5),
      midair_roll_range=(-0.15, 0.15), midair_pitch_range=(-0.2, 0.2),
      midair_yaw_range=(-0.1, 0.1), asset_cfg=asset_cfg)
  asset: Entity = env.scene[asset_cfg.name]
  db = _load_gait_bank(delivery_bank_path, device)
  ab = _load_gait_bank(arrival_bank_path, device)

  def _spawn_rows(ids, rows):
    if len(ids) == 0:
      return
    x = rows[:, 0] + sample_uniform(-0.02, 0.02, (len(ids),), device)
    z, quat = rows[:, 1:2], rows[:, 2:6]
    linv, angv = rows[:, 6:9], rows[:, 9:12]
    jp, jv = rows[:, 12:24], rows[:, 24:36]
    origins = env.scene.env_origins[ids]
    pos = torch.stack([origins[:, 0] + x, origins[:, 1],
                       origins[:, 2] + z.squeeze(-1)], dim=-1)
    asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1),
                                      env_ids=ids)
    asset.write_root_link_velocity_to_sim(torch.cat([linv, angv], dim=-1),
                                          env_ids=ids)
    asset.write_joint_state_to_sim(jp, jv, env_ids=ids)

  deliv = (u >= 0.50) & (u < 0.70)
  arriv = (u >= 0.70) & (u < 0.85)
  for mask, bank in ((deliv, db), (arriv, ab)):
    ids = env_ids[mask]
    if len(ids) == 0 or bank.shape[0] == 0:
      continue
    rows = bank[torch.randint(0, bank.shape[0], (len(ids),), device=device)]
    _spawn_rows(ids, rows)


def reached_far_side(env, x_thresh: float = 0.9, up_thresh: float = 0.7):
  """SUCCESS termination (time_out=True — truncation, NOT failure: the RA
  backup must bootstrap, not fire the terminal anchor, and the curriculum's
  time_outs check must count it as success). Ends the episode once the
  maneuver is genuinely complete: past the gap, upright, back on the ground.
  Without this, the latched lander drifts forward for the remaining ~7 s of
  a 10 s episode on TILING terrain, reaches the NEXT patch's gap, dies, and
  the curriculum demotes an env whose jump was flawless (2026-07-11: a
  plausible contributor to the L3-L4 frontier stall)."""
  from safe_mjlab_zoo.margins import ground_reference
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  up = -robot.data.projected_gravity_b[:, 2]
  base_z, gref = ground_reference(env)
  grounded = (base_z - gref) < 0.42
  return (x_rel > x_thresh) & (up > up_thresh) & grounded


def v9_reverse_levels(env, env_ids) -> torch.Tensor:
  """Promote on composed success (survived to timeout AND far side AND
  upright — the lander carried the flight after the latch); demote on
  failure. Same structure as the proven crossing_reverse_levels."""
  _ensure_v9_level(env)
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
  up = -robot.data.projected_gravity_b[env_ids, 2]
  survived = env.termination_manager.time_outs[env_ids]
  reached = survived & (x_rel > 0.7) & (up > 0.7)
  lvl = env._v9_level[env_ids]
  lvl = torch.where(reached, lvl + 1, lvl - 1)
  env._v9_level[env_ids] = lvl.clamp(0, V9_LEVELS)
  return env._v9_level.float().mean()


def reset_robot_gait_bank_over_gaps(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  *,
  bank_path: str,
  bank_fraction: float = 0.5,
  bank_x_range: tuple[float, float] = (-0.3, -0.05),
  bank_y_range: tuple[float, float] = (-0.1, 0.1),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  **midair_kwargs,
) -> None:
  """Midair spawns + GAIT-BANK committed spawns (no default-pose ground).

  Teleport-commitment (default pose at speed) taught a launch anchored to
  states no running robot occupies (composed success 0.4%, 2026-07-11).
  Bank rows are full sim states harvested mid-stride at speed; placed at the
  lip they are committed by position+momentum AND physically real. Base
  reset runs with midair_fraction as given; bank_fraction of ALL resetting
  envs is then overwritten with bank states (so midair_fraction=1.0 +
  bank_fraction=0.5 -> 50/50 midair/bank, zero default-pose spawns)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  reset_robot_midair_over_gaps(env, env_ids, asset_cfg=asset_cfg,
                               **midair_kwargs)
  device = env.device
  sel = torch.rand(len(env_ids), device=device) < float(bank_fraction)
  ids = env_ids[sel]
  if len(ids) == 0:
    return
  bank = _load_gait_bank(bank_path, device)
  rows = bank[torch.randint(0, bank.shape[0], (len(ids),), device=device)]
  if rows.shape[1] == 36:
    # arrival-bank layout: x(1) z(1) quat(4) ... — spawn AT the harvested x
    # (roll-in: no translation; deeper states physically evolve to the edge).
    x = rows[:, 0] + sample_uniform(-0.03, 0.03, (len(ids),), device)
    rows = rows[:, 1:]
  else:
    x = sample_uniform(bank_x_range[0], bank_x_range[1], (len(ids),), device)
  z, quat = rows[:, 0:1], rows[:, 1:5]
  linv, angv = rows[:, 5:8], rows[:, 8:11]
  jp, jv = rows[:, 11:23], rows[:, 23:35]
  y = sample_uniform(bank_y_range[0], bank_y_range[1], (len(ids),), device)
  origins = env.scene.env_origins[ids]
  pos = torch.stack([origins[:, 0] + x, origins[:, 1] + y,
                     origins[:, 2] + z.squeeze(-1)], dim=-1)
  asset: Entity = env.scene[asset_cfg.name]
  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1),
                                    env_ids=ids)
  asset.write_root_link_velocity_to_sim(torch.cat([linv, angv], dim=-1),
                                        env_ids=ids)
  asset.write_joint_state_to_sim(jp, jv, env_ids=ids)


def reset_robot_midair_over_gaps(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  *,
  midair_fraction: float = 0.5,
  ground_pose_range: dict[str, tuple[float, float]] | None = None,
  ground_velocity_range: dict[str, tuple[float, float]] | None = None,
  midair_x_range: tuple[float, float] = (1.0, 6.5),
  midair_y_range: tuple[float, float] = (-0.2, 0.2),
  midair_z_range: tuple[float, float] = (0.35, 0.75),
  midair_vx_range: tuple[float, float] = (1.5, 3.5),
  midair_vy_range: tuple[float, float] = (-0.2, 0.2),
  midair_vz_range: tuple[float, float] = (-1.0, 0.5),
  midair_roll_range: tuple[float, float] = (-0.25, 0.25),
  midair_pitch_range: tuple[float, float] = (-0.25, 0.25),
  midair_yaw_range: tuple[float, float] = (-0.3, 0.3),
  ground_committed_fraction: float = 0.0,
  ground_committed_pose_range: dict[str, tuple[float, float]] | None = None,
  ground_committed_velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset robot with a mix of on-ground and strategic mid-air spawns.

  A ``midair_fraction`` of the resetting envs are spawned at a
  randomised forward offset along the terrain patch, elevated above
  the platform, and given a forward linear velocity.  Because the
  forward offset covers the full patch length, a significant share of
  these spawns lands over a gap — the safety policy is therefore
  forced to learn how to bridge gaps during recovery, even though the
  underlying task/velocity policy only knows how to walk.

  The remaining envs reset normally within ``ground_pose_range`` /
  ``ground_velocity_range`` to keep a healthy baseline of walking
  states in the replay.

  Parameters
  ----------
  midair_fraction:
      Fraction of reset envs that receive a mid-air spawn (0-1).
  ground_pose_range / ground_velocity_range:
      Pose/velocity ranges for normal on-ground resets (same semantics
      as :func:`mjlab.envs.mdp.events.reset_root_state_uniform`).
  midair_x_range:
      Forward offset (m) along the patch.  Should span gap locations.
  midair_y_range:
      Lateral offset (m) relative to the patch origin.
  midair_z_range:
      Additional height (m) above the default spawn height — the robot
      is placed mid-flight above the platform.
  midair_v{x,y,z}_range:
      Linear velocity (m/s) imparted at spawn.  ``vx`` should be
      positive (forward).
  midair_{roll,pitch,yaw}_range:
      Orientation perturbation (rad) about the default upright pose.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  if len(env_ids) == 0:
    return

  if ground_pose_range is None:
    ground_pose_range = {
      "x": (-0.3, 0.3),
      "y": (-0.2, 0.2),
      "z": (0.0, 0.0),
      "yaw": (-0.2, 0.2),
    }
  if ground_velocity_range is None:
    ground_velocity_range = {}

  asset: Entity = env.scene[asset_cfg.name]
  assert not asset.is_fixed_base, "Mid-air reset only supports floating-base robots."

  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  num_resets = int(len(env_ids))
  device = env.device

  # Decide which envs get a mid-air spawn / a committed-ground spawn (the
  # launch-arc stratum: trainee-EXECUTED launches are the only way the reach
  # value bridges the ground->air boundary — midair seeds latch at t=0 and
  # never exercise the launch action).
  sample = torch.rand(num_resets, device=device)
  is_midair = sample < float(midair_fraction)
  is_committed = (~is_midair) & (
    sample < float(midair_fraction) + float(ground_committed_fraction))

  # --- Ground resets (sampled from ground_pose_range / ground_velocity_range) ---
  ground_pose_list = [
    ground_pose_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ground_pose_t = torch.tensor(ground_pose_list, device=device)
  ground_pose_samples = sample_uniform(
    ground_pose_t[:, 0], ground_pose_t[:, 1], (num_resets, 6), device=device
  )

  ground_vel_list = [
    ground_velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ground_vel_t = torch.tensor(ground_vel_list, device=device)
  ground_vel_samples = sample_uniform(
    ground_vel_t[:, 0], ground_vel_t[:, 1], (num_resets, 6), device=device
  )

  # --- Mid-air resets ---
  midair_pose_list = [
    midair_x_range,
    midair_y_range,
    midair_z_range,
    midair_roll_range,
    midair_pitch_range,
    midair_yaw_range,
  ]
  midair_pose_t = torch.tensor(midair_pose_list, device=device)
  midair_pose_samples = sample_uniform(
    midair_pose_t[:, 0], midair_pose_t[:, 1], (num_resets, 6), device=device
  )

  midair_vel_list = [
    midair_vx_range,
    midair_vy_range,
    midair_vz_range,
    (-0.2, 0.2),  # roll rate
    (-0.2, 0.2),  # pitch rate
    (-0.3, 0.3),  # yaw rate
  ]
  midair_vel_t = torch.tensor(midair_vel_list, device=device)
  midair_vel_samples = sample_uniform(
    midair_vel_t[:, 0], midair_vel_t[:, 1], (num_resets, 6), device=device
  )

  # --- Committed-ground resets (launch-arc stratum) ---
  if ground_committed_pose_range is None:
    ground_committed_pose_range = ground_pose_range
  if ground_committed_velocity_range is None:
    ground_committed_velocity_range = ground_velocity_range
  committed_pose_list = [
    ground_committed_pose_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  committed_pose_t = torch.tensor(committed_pose_list, device=device)
  committed_pose_samples = sample_uniform(
    committed_pose_t[:, 0], committed_pose_t[:, 1], (num_resets, 6),
    device=device
  )
  committed_vel_list = [
    ground_committed_velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  committed_vel_t = torch.tensor(committed_vel_list, device=device)
  committed_vel_samples = sample_uniform(
    committed_vel_t[:, 0], committed_vel_t[:, 1], (num_resets, 6),
    device=device
  )

  # Merge ground / committed-ground / mid-air samples (disjoint strata).
  mask = is_midair.unsqueeze(-1).float()
  cmask = is_committed.unsqueeze(-1).float()
  gmask = 1.0 - mask - cmask
  pose_samples = (ground_pose_samples * gmask
                  + committed_pose_samples * cmask
                  + midair_pose_samples * mask)
  vel_samples = (ground_vel_samples * gmask
                 + committed_vel_samples * cmask
                 + midair_vel_samples * mask)

  # --- Apply to sim ---
  positions = (
    root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
  )
  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  velocities = root_states[:, 7:13] + vel_samples

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)
