"""LOW-BAR crawl -- the 2nd reach-avoid-liveness benchmark, structural twin of
the gap ``brake_or_jump`` split test.  Deployment model (identical to the gap
line): a nominal walker drives the robot forward; near a low bar the crawl
filter takes over -- DUCK-and-coast THROUGH a passable bar, or BRAKE to a stop
before an impossible one -- then hands back.

Reverse-curriculum SPAWN (one global level ``env._L`` in 0..11, a 20 % peek at
L+1 for the look-ahead gate), keyed on an integer level, mirroring
``brake_or_jump``.  The reset ALWAYS writes the full joint state (default OR
crouch) -- the brake_or_jump lesson: a root-only spawn folds the robot.  Bands:

  COMMITTED (L 0..4):  spawn CROUCHED at / just-under / just-past the bar, near
                       static (coasting out slowly).  Near-equilibria that
                       scaffold the crawl skill -- the crawl's scriptable analog
                       of the gap's harvested jump states.
  DECISION  (L 5..8):  at/near the bar face, semi-crouched, LOW forward momentum
                       -- the region where the twins differ (RA initiates, an
                       avoid twin's low-momentum refusal is the isolated var).
  APPROACH  (L 9..10): on the approach (x_rel -0.5..-1.5), STANDING with forward
                       momentum toward the bar; momentum anneals high -> 0 as L
                       rises (secondary axis).
  FINAL     (L 11):    far standstill (x_rel -2.0..-1.5), zero momentum, standing
                       -- must initiate the duck from a standstill.

Both twins share this env (identical spawn distribution); only the reach term
differs (RA ``low_bar_margins`` vs ``avoid_only(low_bar_margins)``).

Margins:
  g = min( crawl integrity (g_terrain_relative: terrain-relative base height,
           tilt, non-foot ground contact EXCLUDING thigh/calf),
           nose-down pitch guard,
           VIRTUAL-BAR term: while x_rel in [0, bar_depth],
             (bar_clearance - trunk_top)/BAR_NORM  -- > 0 iff the trunk fits
             under the beam; < 0 = strike / too-tall / jumping-over. Outside the
             span the bar cannot threaten -> +large. )
  l = min( (x_rel - (bar_depth + 0.5))/POS_NORM , (up - 0.85)/UP_NORM )
      -- completion PAST the bar, position-based, NO rest term (E030 braking
      game).  POS_NORM SPANS the spawn range so l is a real reach gradient (not
      clamped flat at the campaign clamp) even at the far standstill spawn.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from robot_safety_sandbox.margins import CLAMP, g_terrain_relative
from robot_safety_sandbox.envs.terrains.low_bar import LOW_BAR_TERRAINS_CFG
from robot_safety_sandbox.envs.go2_crawl.env_cfg import (
  _crouch_z,
  _ensure_crawl_buffers,
  unitree_go2_crawl_env_cfg,
)

# --- geometry / margin constants ----------------------------------------------
# Trunk half-height, MEASURED from the Go2 base collision box (base1_collision
# half-size z = 0.057 in envs/assets_go2/xmls/go2.xml, centred on base_link);
# at the 0.32 m standing height this puts the trunk top at ~0.377 m, matching the
# crawl lineage's "trunk-top ~0.38". (The directive estimated 0.10-0.12; the
# measured box is smaller, and the crawl code base is consistent with ~0.06.)
HALF_TRUNK = 0.06
# Trunk LONGITUDINAL half-length (base1_collision x half-size = 0.1881 in go2.xml).
# The bar hazard must be keyed on the trunk's FORWARD extent, not the base centre:
# a standing robot strikes the (overhead) bar face when the FRONT of its trunk
# reaches it, while the base is still ~0.19 m short. Keying "under the bar" on the
# base alone leaves a ~0.19 m blind zone in front of the bar where a tall robot
# overlaps the beam with no g penalty (the approach zone the twins loiter in).
TRUNK_HALF_LEN = 0.1881
BAR_NORM = 0.15            # O(1) normalization of the (clearance - trunk_top) gap
POS_NORM = 3.5            # SPANS the spawn range: far spawn x_rel ~ -2.0 with the
                          # target at bar_depth + 0.5 gives l ~ -0.83 (> -1) so a
                          # real reach gradient exists over the whole approach.
L_TARGET_MARGIN = 0.5     # completion x = bar_depth + this (past the bar)
UP_MIN = 0.85
UP_NORM = 0.15
# Directional nose-down pitch guard (same values as the crawl duck task): the
# base g tilt term is non-directional and only fires at ~70 deg (post-crash);
# this makes an early anti-dive gradient so the robot ducks LEVEL, not nose-first.
_PITCH_DOWN_LIMIT = 0.26   # sin(~15 deg)
_PITCH_DOWN_NORM = 0.20

# --- curriculum band boundaries (levels 0..D) ---------------------------------
K_COMMIT = 4              # COMMITTED band  L 0..4
DECISION_END = 8         # DECISION  band  L 5..8
D = 11                   # APPROACH  band  L 9..10 ; FINAL L 11


# --- bar params (fixed per env, read off the terrain cfg) ----------------------

def _bar_params(env) -> tuple[float, float]:
  sub = env.scene.terrain.cfg.terrain_generator.sub_terrains["low_bar"]
  return float(sub.bar_clearance), float(sub.bar_depth)


def _x_rel(env) -> torch.Tensor:
  robot = env.scene["robot"]
  return robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]


# --- perception: analytic bar obs (replaces the crawl-filter bar_info, which
#     hardcodes _BAR_X=2.5 + the crawl clearance table) --------------------------

def low_bar_info(env) -> torch.Tensor:
  """Privileged analytic bar obs: [signed_dist_to_bar_face / 4 (clamped +-1),
  under-beam clearance].  Positive distance = bar is ahead (x_rel < 0)."""
  clearance, _depth = _bar_params(env)
  x_rel = _x_rel(env)
  dist = ((-x_rel) / 4.0).clamp(-1.0, 1.0)
  clr = torch.full_like(dist, clearance)
  return torch.stack([dist, clr], dim=1)


# --- reverse-curriculum spawn --------------------------------------------------

def _ensure_low_bar(env):
  if not hasattr(env, "_L"):
    env._L = 0
    env._peek = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._win = {"cur": [], "peek": []}


def _level_spawn(env, L, n, dev, clearance, depth, root):
  """Per-level spawn arrays: (x_rel, vx, crouch_mask, alpha, base_z)."""
  def u(lo, hi):
    return sample_uniform(lo, hi, (n,), dev)

  stand_z = root[:, 2]                       # nominal standing height (~0.32)
  crouch = torch.zeros(n, dtype=torch.bool, device=dev)
  alpha = torch.zeros(n, device=dev)

  need = clearance < 0.37   # standing trunk-top ~0.38 does NOT fit -> must crouch
  if L <= K_COMMIT:
    # COMMITTED: reverse curriculum from AT-THE-TARGET (L0 deep in the rest zone,
    # x_rel > depth+0.5 so l > 0 -- the RA value bootstraps from actually seeing the
    # reward) sliding back to the bar face as L rises. Crouch ONLY where the robot is
    # under the bar AND the bar is too low to clear standing (the 0.39 bootstrap fits
    # standing, so it is WALKED through, not crouched). Forward momentum toward rest.
    f = L / max(K_COMMIT, 1)                  # 0..1
    xc = (depth + 1.0) * (1.0 - f)            # ~depth+1.0 (rest zone, l>0) -> 0.0 (face)
    x = xc + u(-0.1, 0.1)
    vx = u(0.4, 1.0)
    crouch = (x < depth) & need
    alpha = torch.where(crouch, torch.full((n,), 0.5, device=dev),
                        torch.zeros(n, device=dev))
    z = torch.where(crouch, _crouch_z(alpha), stand_z) + u(0.003, 0.015)
  elif L <= DECISION_END:
    # DECISION: at/near the bar face (strictly before it), low fwd momentum -- the
    # RA-initiates-vs-avoid-refuses region. Semi-crouched only if the bar is low.
    x = u(-0.5, -0.05)
    vx = u(0.3, 0.9)
    crouch = need & torch.ones(n, dtype=torch.bool, device=dev)
    alpha = torch.where(crouch, u(0.3, 0.6), torch.zeros(n, device=dev))
    z = torch.where(crouch, _crouch_z(alpha), stand_z) + u(0.003, 0.015)
  elif L < D:
    # APPROACH: standing on the approach, momentum anneals high -> low as L rises.
    fA = (L - (DECISION_END + 1)) / max(D - 1 - (DECISION_END + 1), 1)  # 0 @9, 1 @10
    x = (-0.5 - fA * 1.0) + u(-0.25, 0.25)     # -0.5..-1.5 back
    vlo = 1.2 - fA * 0.9                        # 1.2 -> 0.3
    vhi = 2.2 - fA * 1.2                        # 2.2 -> 1.0
    vx = u(0.0, 1.0) * (vhi - vlo) + vlo
    z = stand_z + u(-0.01, 0.02)
  else:
    # FINAL: far standstill, standing -- must initiate the duck from a standstill.
    x = u(-2.0, -1.5)
    vx = u(0.0, 0.15)
    z = stand_z + u(-0.01, 0.02)
  return x, vx, crouch, alpha, z


def reset_low_bar(env, env_ids, asset_cfg=SceneEntityCfg("robot"),
                  fixed_level=None):
  """Reverse-curriculum spawn.  ALWAYS writes the full joint state: standing
  spawns get the default joints (reset_robot_joints), crouched spawns get the
  crouch pose (crouch_joints reads the masks set here) -- never a root-only
  spawn (which folds the robot: the brake_or_jump lesson).

  ``fixed_level`` (set by the cfg_builder for play/eval) pins the spawn band and
  disables the look-ahead peek, so every reset spawns from exactly that level."""
  _ensure_low_bar(env)
  if fixed_level is not None:
    env._L = int(fixed_level)
  _ensure_crawl_buffers(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  dev = env.device
  asset = env.scene[asset_cfg.name]
  clearance, depth = _bar_params(env)

  # 20 % of resetting envs PEEK at L+1 (look-ahead gate), rest at L. A pinned
  # spawn (fixed_level) disables the peek so every env sits on exactly that band.
  if fixed_level is None:
    peek = torch.rand(len(env_ids), device=dev) < 0.20
  else:
    peek = torch.zeros(len(env_ids), dtype=torch.bool, device=dev)
  env._peek[env_ids] = peek
  lvl = torch.where(peek, min(env._L + 1, D), env._L).long()

  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()
  origins = env.scene.env_origins[env_ids]

  x = torch.empty(n, device=dev)
  vx = torch.empty(n, device=dev)
  crouch = torch.zeros(n, dtype=torch.bool, device=dev)
  alpha = torch.zeros(n, device=dev)
  z = torch.empty(n, device=dev)
  for L in torch.unique(lvl).tolist():
    sel = (lvl == L).nonzero(as_tuple=False).flatten()
    xs, vs, cs, as_, zs = _level_spawn(env, int(L), len(sel), dev, clearance,
                                       depth, root[sel])
    x[sel], vx[sel], crouch[sel], alpha[sel], z[sel] = xs, vs, cs, as_, zs

  y = sample_uniform(-0.06, 0.06, (n,), dev)
  pos = torch.stack([origins[:, 0] + x, origins[:, 1] + y, origins[:, 2] + z],
                    dim=-1)
  euler = torch.stack([sample_uniform(-0.04, 0.04, (n,), dev),
                       sample_uniform(-0.04, 0.04, (n,), dev),
                       sample_uniform(-0.05, 0.05, (n,), dev)], dim=1)
  quat = quat_mul(root[:, 3:7],
                  quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]))
  vel = torch.zeros(n, 6, device=dev)
  vel[:, 0] = vx
  vel[:, 1] = sample_uniform(-0.05, 0.05, (n,), dev)

  asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)

  # masks consumed by the crouch_joints event (apply_crouch_joints).
  env._crouch_mask[env_ids] = crouch
  env._crouch_alpha[env_ids] = alpha
  env._splay_mag[env_ids] = torch.zeros(n, device=dev)
  env._leg_jitter[env_ids] = torch.full((n,), 0.02, device=dev)


def _success(env, env_ids, depth) -> torch.Tensor:
  """Crossed PAST the bar and upright at timeout."""
  robot = env.scene["robot"]
  x_rel = (robot.data.root_link_pos_w[env_ids, 0]
           - env.scene.env_origins[env_ids, 0])
  up = -robot.data.projected_gravity_b[env_ids, 2]
  t_o = env.termination_manager.time_outs[env_ids]
  return t_o & (x_rel > depth + 0.5) & (up > 0.7)


def low_bar_levels(env, env_ids) -> torch.Tensor:
  """Look-ahead gate (mirror brake_or_jump): advance L when current-level
  success >= 0.70 AND the L+1 peek success >= 0.15, each over a full window."""
  _ensure_low_bar(env)
  _clearance, depth = _bar_params(env)
  ok = _success(env, env_ids, depth)
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
      print(f"[low_bar] ADVANCE -> L={env._L}  (cur {sc:.2f}, peek {sp:.2f})",
            flush=True)
    elif len(cur) > 3 * W:
      env._win["cur"] = cur[-W:]
      env._win["peek"] = pw[-W:]
  return torch.tensor(float(env._L))


# --- margins -------------------------------------------------------------------

def l_low_bar(env) -> torch.Tensor:
  """Completion PAST the bar (position-based; NO rest term)."""
  _clearance, depth = _bar_params(env)
  robot = env.scene["robot"]
  x_rel = _x_rel(env)
  up = -robot.data.projected_gravity_b[:, 2]
  crossed = (x_rel - (depth + L_TARGET_MARGIN)) / POS_NORM
  upright = (up - UP_MIN) / UP_NORM
  return torch.minimum(crossed, upright)


def low_bar_margins(env):
  """g = min(crawl integrity, nose-down pitch guard, virtual-bar term);
  l = completion past the bar.  The avoid twin strips l via avoid_only()."""
  clearance, depth = _bar_params(env)
  robot = env.scene["robot"]

  g = g_terrain_relative(env)
  nose_down = robot.data.projected_gravity_b[:, 0]        # sin(pitch); >0 nose-down
  pitch_guard = (_PITCH_DOWN_LIMIT - nose_down) / _PITCH_DOWN_NORM
  g = torch.minimum(g, pitch_guard)

  # VIRTUAL-BAR term: threatens while the TRUNK overlaps the bar span (keyed on
  # the trunk's forward extent, not the base centre -- see TRUNK_HALF_LEN). A tall
  # robot whose front has reached the bar face is already in the hazard, so the
  # approach zone is no longer a blind spot.
  base_z = robot.data.root_link_pos_w[:, 2]
  trunk_top = base_z + HALF_TRUNK
  x_rel = _x_rel(env)
  bar = (clearance - trunk_top) / BAR_NORM
  bar = torch.where(_trunk_over_bar(x_rel, depth), bar, torch.full_like(bar, CLAMP))
  g = torch.minimum(g, bar)

  return g.clamp(-CLAMP, CLAMP), l_low_bar(env).clamp(-CLAMP, CLAMP)


def _trunk_over_bar(x_rel, depth):
  """True where the trunk (base +/- TRUNK_HALF_LEN) overlaps the bar span [0, depth]
  -- the region a too-tall trunk collides with the overhead beam."""
  return (x_rel + TRUNK_HALF_LEN >= 0.0) & (x_rel - TRUNK_HALF_LEN <= depth)


def low_bar_strike(env, margin_fn=None) -> torch.Tensor:
  """Absorbing bar-strike failure: the trunk overlaps the bar span while too tall
  to clear it.  The beam is VIRTUAL (no physics), so this analytic term is the
  ONLY failure signal for hitting it -- the static-bar analog of the gate's
  ``crushed_by_gate`` and the car's ``car_collision`` (both terminate on g<0).
  Without it a strike is a recoverable reward dip, never an absorbing failure."""
  clearance, depth = _bar_params(env)
  d = env.scene["robot"].data
  trunk_top = d.root_link_pos_w[:, 2] + HALF_TRUNK
  return _trunk_over_bar(_x_rel(env), depth) & (trunk_top > clearance)


# --- env cfg builder -----------------------------------------------------------

def _swap_bar_perception(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace the crawl-filter ``bar_info`` critic obs (hardcodes _BAR_X + the
  crawl clearance table) with the low-bar analytic obs.  The forward ray scans
  (bar_scan / bar_scan_low) are geometry-agnostic and kept as-is."""
  crit = cfg.observations["critic"].terms
  if "bar_info" in crit:
    crit["bar_info"] = ObservationTermCfg(func=low_bar_info, params={})


def unitree_go2_low_bar_env_cfg(play: bool = False, bar_clearance: float = 0.39,
                                bar_depth: float = 0.4,
                                fixed_level: int | None = None
                                ) -> ManagerBasedRlEnvCfg:
  """LOW-BAR crawl env (2nd RA-liveness benchmark).  Builds on the crawl base
  env (Go2 robot + obs groups + rewards + thigh/calf-excluded contact set),
  swaps in the VIRTUAL low-bar terrain, the reverse-curriculum spawn, and the
  spawn-level curriculum.

  ``fixed_level`` (an env_overrides / play-eval knob, e.g. ``--env-override
  fixed_level=11``) PINS the reverse-curriculum spawn band and drops the
  curriculum, so every reset spawns from that level (0 = crouched past the bar,
  11 = far back at a standstill).  Training leaves it None (live curriculum)."""
  cfg = unitree_go2_crawl_env_cfg(play=play)

  # Swap terrain -> VIRTUAL low bar at the requested clearance/depth.
  cfg.scene.terrain.terrain_generator = replace(
    LOW_BAR_TERRAINS_CFG,
    sub_terrains={"low_bar": replace(
      LOW_BAR_TERRAINS_CFG.sub_terrains["low_bar"],
      bar_clearance=bar_clearance, bar_depth=bar_depth)})
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.episode_length_s = 6.0

  # Reverse-curriculum spawn; KEEP reset_robot_joints (default joints) +
  # crouch_joints (crouch pose) so every spawn gets a full joint state.
  reset_params = {} if fixed_level is None else {"fixed_level": int(fixed_level)}
  cfg.events["reset_base"] = EventTermCfg(func=reset_low_bar, mode="reset",
                                          params=reset_params)
  cfg.events.pop("handover_joints", None)      # no walker-state replay here
  cfg.events.pop("rest_obstacle_window", None)  # no rest objective
  cfg.events.pop("push_robot", None)

  _swap_bar_perception(cfg)

  # Bar-strike = absorbing FAILURE. The beam is virtual (no contact), so the
  # physical illegal_contact term can never fire on a strike; this analytic term
  # is the only failure signal for hitting the bar (mirrors crushed_by_gate /
  # car_collision). base.py's g-anchor then grades the terminal state as failure.
  cfg.terminations["bar_strike"] = TerminationTermCfg(func=low_bar_strike)

  # Live curriculum for training; a pinned spawn band means no advancement.
  cfg.curriculum = ({} if fixed_level is not None
                    else {"low_bar_levels": CurriculumTermCfg(func=low_bar_levels)})
  return cfg
