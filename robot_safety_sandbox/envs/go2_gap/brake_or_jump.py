"""SPLIT TEST v2 — reverse curriculum over HARVESTED real jump states, per the
colleague's revised spec. Fixed target G = stable far-side stance; the start
distribution moves backward along the jump arc, ending in a decision mixture.

ONE global curriculum level L (all envs at L, a peek fraction at L+1 for the
look-ahead gate). Per-env spawn by level:

  COMMITTED band (0..K):  real harvested states, x-window sliding FAR -> LAUNCH
                          (far stance -> land -> airborne -> takeoff), real vx.
  DECISION band (K+1..D-1): real GROUNDED harvested poses relocated toward the
                          near edge with momentum scaled DOWN gradually (linv &
                          jv scaled together = a coherent slowed gait) -- the
                          'does RA initiate?' region, no momentum cliff.
  FINAL level D: A/B/C mixture --
      A far-enough + slow (runway): the contrast (RA accelerates, avoid may stop)
      B near + fast: committed (both jump)
      C near + slow: stoppable (both stop)

Both twins share this env; only the reach term differs (avoid l_zero /
RA l_stable_far). No synthetic pose+random-velocity: every spawn is a real
harvested pose (committed as-is, or proportionally slowed for the decision band).
"""

from __future__ import annotations

from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

from robot_safety_sandbox.envs.go2_gap.brake_or_jump_harvest import (
  unitree_go2_harvest_env_cfg)

_BANK = {}            # cache per gap width
_BANK_PATH = "~/SAFE/jump_bank.pt"
DEFAULT_WIDTH = 0.12

K_COMMIT = 7          # committed bands 0..7
D = 12                # final level

# --- airborne-clean finetune -------------------------------------
# A "clean" crossing = a genuine AIRBORNE CLEAR: while the base is over the gap
# (x_rel in (0, gap_width)), NO foot is loaded, AND no body-plant occurs. The
# discriminating physics: the gap is a pit, so a loaded foot while the base is
# over the gap means a foot is bridging back to the near edge (or reaching the
# far edge) -- a leg-dragging BRIDGE. A true flight has all feet unloaded through
# the traversal. Formulated as the NEGATION (latch touch-down + body-plant, clean
# = neither) so that committed-band spawns -- which start already PAST the gap
# and never traverse it -- are clean by construction and the clean-gated
# curriculum advances through them instead of stalling at L0.
#
# NB: the first-pass spec latched "went airborne (any single all-feet-off step)
# over the gap", but a touch-and-go bridge also has a brief all-feet-off instant,
# so that flag fired ~0.94 for bridges too (measured: scratchpad/airborne_clean_probe.py,
# 2026-08-21) and did NOT discriminate. The touch-down formulation below gives
# 0.30 (airborne) vs 0.70 (bridge) at the decision band -- see the airborne-clean detail.
_TOUCHDOWN_FORCE = 5.0    # N per foot: a foot is "loaded" above this
_BODYPLANT_FORCE = 10.0   # N: trunk/limb ground contact (matches nonfoot g term)


def _gap_width(env):
  g = env.scene.terrain.cfg.terrain_generator
  return round(float(g.sub_terrains["island"].gap_width_range[0]), 3)


def far_x(env):
  """Success / target x: past the (width-dependent) gap onto the far platform."""
  return _gap_width(env) + 0.15


def _bank(env):
  w = _gap_width(env)
  if w not in _BANK:
    import os
    d = torch.load(os.path.expanduser(_BANK_PATH), map_location=env.device,
                   weights_only=True)
    b = d["bank"].float()
    _BANK[w] = b[torch.abs(b[:, 36] - w) < 1e-3]       # this width's jump states
    print(f"[brake_or_jump] {_BANK[w].shape[0]} harvested states @w={w}")
  return _BANK[w]


def _restore(env, ids, rows, x_target, mom_scale):
  """Write full sim state from bank rows, relocated to x_target with linv&jv
  scaled by mom_scale (per-env tensors)."""
  asset = env.scene[SceneEntityCfg("robot").name]
  dev = env.device
  z = rows[:, 1:2]
  quat = rows[:, 2:6]
  linv = rows[:, 6:9] * mom_scale[:, None]
  angv = rows[:, 9:12] * mom_scale[:, None]
  jp = rows[:, 12:24]
  jv = rows[:, 24:36] * mom_scale[:, None]
  y = sample_uniform(-0.10, 0.10, (len(ids),), dev)
  origins = env.scene.env_origins[ids]
  pos = torch.stack([origins[:, 0] + x_target, origins[:, 1] + y,
                     origins[:, 2] + z.squeeze(-1)], dim=-1)
  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
  asset.write_root_link_velocity_to_sim(torch.cat([linv, angv], dim=-1),
                                        env_ids=ids)
  asset.write_joint_state_to_sim(jp, jv, env_ids=ids)


def _sample_rows(bank, mask_fn, n, dev):
  idx = torch.nonzero(mask_fn(bank), as_tuple=False).flatten()
  if len(idx) == 0:
    idx = torch.arange(bank.shape[0], device=bank.device)
  pick = idx[torch.randint(0, len(idx), (n,), device=idx.device)]
  return bank[pick].to(dev)


def _spawn_level(env, ids, level_vec):
  """Place envs `ids` per their integer level in level_vec (same length)."""
  bank = _bank(env)
  dev = env.device
  committed = bank[:, 0] >= 0.0                       # all rows (committed arc)
  grounded = lambda b: (b[:, 0] > 0.35)               # landed/far grounded poses
  launch = lambda b: (b[:, 0] >= 0.0) & (b[:, 0] < 0.10)

  for L in torch.unique(level_vec).tolist():
    sel = (level_vec == L).nonzero(as_tuple=False).flatten()
    if len(sel) == 0:
      continue
    gids = ids[sel]
    n = len(gids)
    if L <= K_COMMIT:
      # committed: x-window slides far (0.62) -> launch (0.02)
      f = L / K_COMMIT
      xc = 0.62 - f * 0.60
      rows = _sample_rows(bank, lambda b: (b[:, 0] > xc - 0.12) & (b[:, 0] < xc + 0.12),
                          n, dev)
      xt = rows[:, 0] + sample_uniform(-0.03, 0.03, (n,), dev)  # real position
      ms = torch.ones(n, device=dev)
    elif L < D:
      # decision ramp: grounded pose relocated near edge, momentum scaled down
      f = (L - K_COMMIT) / (D - K_COMMIT)              # 0..1
      xt = -0.05 - f * 0.30 + sample_uniform(-0.04, 0.04, (n,), dev)
      s = 0.7 - f * 0.5                                 # 0.7 -> 0.2
      rows = _sample_rows(bank, grounded, n, dev)
      ms = torch.full((n,), 0.0, device=dev) + s
    else:
      # FINAL mixture A/B/C (thirds)
      r = torch.rand(n, device=dev)
      xt = torch.empty(n, device=dev); ms = torch.empty(n, device=dev)
      A, B = r < 1 / 3, (r >= 1 / 3) & (r < 2 / 3)
      C = r >= 2 / 3
      xt[A] = sample_uniform(-0.35, -0.25, (int(A.sum()),), dev)
      ms[A] = sample_uniform(0.15, 0.35, (int(A.sum()),), dev)
      xt[C] = sample_uniform(-0.15, -0.05, (int(C.sum()),), dev)
      ms[C] = sample_uniform(0.05, 0.20, (int(C.sum()),), dev)
      xt[B] = sample_uniform(0.00, 0.06, (int(B.sum()),), dev)
      ms[B] = 1.0
      rows = torch.empty(n, bank.shape[1], device=dev)
      if A.any(): rows[A] = _sample_rows(bank, grounded, int(A.sum()), dev)
      if C.any(): rows[C] = _sample_rows(bank, grounded, int(C.sum()), dev)
      if B.any(): rows[B] = _sample_rows(bank, launch, int(B.sum()), dev)
    _restore(env, gids, rows, xt, ms)


def _ensure(env):
  if not hasattr(env, "_L"):
    env._L = 0
    env._peek = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._win = {"cur": [], "peek": []}
  if not hasattr(env, "_td_og"):
    # airborne-clean latches: per-env, reset in reset_brake_or_jump.
    # _td_og: a foot was loaded while the base was over the gap (bridge).
    # _bodyplant: trunk/limb hit the ground.  clean = ~_td_og & ~_bodyplant.
    env._td_og = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._bodyplant = torch.zeros(env.num_envs, dtype=torch.bool,
                                 device=env.device)


def reset_brake_or_jump(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  _ensure(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  # 20% of resetting envs PEEK at L+1 (look-ahead), rest at L
  peek = torch.rand(len(env_ids), device=env.device) < 0.20
  env._peek[env_ids] = peek
  lvl = torch.where(peek, min(env._L + 1, D), env._L).long()
  _spawn_level(env, env_ids, lvl)
  # airborne-clean latches: fresh per episode.
  env._td_og[env_ids.long()] = False
  env._bodyplant[env_ids.long()] = False


def _levels(env, env_ids, require_clean: bool) -> torch.Tensor:
  """Look-ahead gate: advance L when current-level success >= 0.70 AND the
  L+1 peek success >= 0.15, each over a full rolling window. When
  ``require_clean`` the success predicate additionally demands an airborne-clean
  crossing (airborne-clean finetune)."""
  _ensure(env)
  robot = env.scene["robot"]
  x = robot.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
  up = -robot.data.projected_gravity_b[env_ids, 2]
  ok = (env.termination_manager.time_outs[env_ids] & (x > far_x(env)) & (up > 0.7))
  if require_clean:
    ok = ok & (~env._td_og & ~env._bodyplant)[env_ids]
  pk = env._peek[env_ids]
  env._win["cur"].extend(ok[~pk].cpu().tolist())
  env._win["peek"].extend(ok[pk].cpu().tolist())
  W = 2000
  cur, pw = env._win["cur"], env._win["peek"]
  if len(cur) >= W and len(pw) >= W // 4:
    sc = sum(cur[-W:]) / W
    sp = sum(pw[-W // 2:]) / (W // 2)
    if sc >= 0.70 and sp >= 0.15 and env._L < D:
      env._L += 1
      env._win = {"cur": [], "peek": []}
      print(f"[brake_or_jump] ADVANCE -> L={env._L}  (cur {sc:.2f}, peek {sp:.2f})",
            flush=True)
    elif len(cur) > 3 * W:
      env._win["cur"] = cur[-W:]; env._win["peek"] = pw[-W:]
  return torch.tensor(float(env._L))


def brake_or_jump_levels(env, env_ids) -> torch.Tensor:
  """Outcome-gated reverse curriculum (the naive-RA tasks)."""
  return _levels(env, env_ids, require_clean=False)


def brake_or_jump_levels_clean(env, env_ids) -> torch.Tensor:
  """Clean-gated reverse curriculum: advance only on airborne-clean
  crossings."""
  return _levels(env, env_ids, require_clean=True)


def l_stable_far(env, pos_norm=0.20, v_rest=0.6, v_norm=0.5,
                 up_min=0.85, up_norm=0.15):
  x_far = far_x(env)
  robot = env.scene["robot"]
  x = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
  up = -robot.data.projected_gravity_b[:, 2]
  return torch.minimum(torch.minimum((x - x_far) / pos_norm,
                                     (v_rest - speed) / v_norm),
                       (up - up_min) / up_norm)


def _clean_flag(env) -> torch.Tensor:
  """Per-step latch update; returns ``clean = ~_td_og & ~_bodyplant``.

  Called once per env step from :func:`l_stable_far_clean` (``base._cached_margins``
  memoizes ``margin_fn`` per ``common_step_counter``, so the latch advances
  exactly once per step). Latches a BRIDGE (a foot loaded while the base is over
  the gap) and a body-plant; clean is the complement. Guards the contact sensors:
  on a shadow/eval env that lacks them, degrades to ``clean = True`` (recovers
  the plain l_stable_far behavior) rather than crashing."""
  _ensure(env)
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  og = (x_rel > 0.0) & (x_rel < _gap_width(env))     # base strictly over the pit
  try:
    feet = env.scene["feet_ground_contact"]
    fdata = feet.data.force
    if fdata is None:
      raise AttributeError
    fmag = torch.norm(fdata, dim=-1)                 # [B, F] (per-foot netforce)
    foot_down = fmag > _TOUCHDOWN_FORCE
    while foot_down.dim() > 2:
      foot_down = foot_down.any(dim=-1)              # collapse any slot dim
    feet_down = foot_down.sum(dim=1)                 # [B] number of feet loaded
    non = env.scene["nonfoot_ground_touch"]
    nf = non.data.force_history
    if nf is None:
      nf = non.data.force
    if nf is None:
      raise AttributeError
    mag = torch.norm(nf, dim=-1)
    while mag.dim() > 1:
      mag = mag.amax(dim=-1)                         # [B] peak trunk contact force
    env._td_og |= og & (feet_down > 0)               # a foot bridged over the gap
    env._bodyplant |= (mag > _BODYPLANT_FORCE)
  except (KeyError, AttributeError):
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  return ~env._td_og & ~env._bodyplant


def l_stable_far_clean(env, pos_norm=0.20, v_rest=0.6, v_norm=0.5,
                       up_min=0.85, up_norm=0.15):
  """l_stable_far min'd with a clean-crossing gate: reaching the target now also
  requires a genuine airborne clear (all feet off over the gap, no body-plant).
  A dirty leg-dragging "bridge" crossing gets l <= -1 (alive-but-unsuccessful).
  g is untouched, so safety/liveness are preserved."""
  base = l_stable_far(env, pos_norm, v_rest, v_norm, up_min, up_norm)
  gate = (_clean_flag(env).float() - 0.5) * 2.0     # clean -> +1, dirty -> -1
  return torch.minimum(base, gate)


def unitree_go2_brake_or_jump_env_cfg(play: bool = False,
                                 gap_width: float = DEFAULT_WIDTH,
                                 clean: bool = False
                                 ) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_harvest_env_cfg(play=play, gap_width=gap_width)
  cfg.episode_length_s = 4.0
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_brake_or_jump, mode="reset", params={})
  # bank restore writes the full joint state -> drop the default-pose joint reset
  cfg.events.pop("reset_robot_joints", None)
  if "push_robot" in cfg.events:
    cfg.events.pop("push_robot", None)
  # clean=True: gate curriculum advancement on airborne-clean crossings.
  if clean:
    cfg.curriculum = {
      "brake_or_jump_levels_clean": CurriculumTermCfg(func=brake_or_jump_levels_clean)}
  else:
    cfg.curriculum = {
      "brake_or_jump_levels": CurriculumTermCfg(func=brake_or_jump_levels)}
  return cfg
