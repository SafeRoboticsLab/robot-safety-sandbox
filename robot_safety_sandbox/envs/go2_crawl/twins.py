"""Crawl R-CBF twins: clean avoid/RA pair on the low-bar task, +/- closing gate.

The low-bar counterpart of the gap twins (tasks/go2_gap.py chain twins). Two
terrain conditions x two certificates:

  STATIC bar   — the NEGATIVE CONTROL. A crawl under a passable static bar is
                 stoppable at every instant (crouched under the beam is
                 statically safe), so there is NO committed region and the
                 theory PREDICTS avoid-filter == RA-filter here.
  CLOSING gate — the committed-region variant. Once the robot's base crosses
                 the bar face, a VIRTUAL ceiling descends (island-style: the
                 physical beam never moves; the hazard lives in g and in the
                 `crushed_by_gate` termination). Lingering under the gate ->
                 crushed; stopping BEFORE the gate is always safe. Crossing
                 requires commitment -> avoid-filter livelocks, RA-filter
                 drives through while the window is open.

Margins are deliberately minimal (no forward force, no gait-l, no shrinking
island, no velocity reach): g = crawl integrity (terrain-relative height/tilt
+ nose-down pitch guard [+ gate ceiling]), l = COMPLETION (at rest past the
bar), mirroring l_gap_completion. Spawns reuse the duck approach strata
(committed / approach / assist) — the same takeover distribution the filter
hands over from.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from robot_safety_sandbox.envs.terrains.crawl_filter import (
  BAR_DEPTH, _BAR_X, bar_clearance_for_level)

# --- gate parameters -----------------------------------------------------------
GATE_CLOSE_RATE = 0.0015   # m per env step (~0.075 m/s at dt 0.02): a committed
                           # crawl (~0.6 m at >=0.5 m/s) clears with margin; a
                           # hesitating one is caught
GATE_MIN = 0.10            # fully closed clearance (uncrossable)
_TRUNK_TOP_OFF = 0.05      # trunk top ~= base_z + half trunk height
_GATE_NORM = 0.10
# --- margin constants (mirrors the duck task's guards) --------------------------
_PITCH_DOWN_LIMIT = 0.26   # sin(pitch) ~ 15 deg nose-down
_PITCH_DOWN_NORM = 0.20
# --- completion target (mirrors l_gap_completion) --------------------------------
REST_X = _BAR_X + BAR_DEPTH + 0.7   # at rest safely PAST the bar (x >= 4.0)
_POS_NORM = 0.5
_V_REST = 0.3
_V_REST_NORM = 0.5


# --- gate machinery --------------------------------------------------------------

def _ensure_gate_buffers(env) -> None:
  if not hasattr(env, "_gate_timer"):
    n = env.num_envs
    env._gate_timer = torch.zeros(n, device=env.device)
    env._gate_entered = torch.zeros(n, dtype=torch.bool, device=env.device)
    env._gate_last_step = torch.full((n,), -1.0, device=env.device)


def gate_clearance(env) -> torch.Tensor:
  """Per-env VIRTUAL ceiling height: open clearance until the base crosses the
  bar face, then descending at GATE_CLOSE_RATE (one-shot; reopened by the
  reset event). Called from BOTH the margin hook and the termination each
  step, so the timer advance is guarded to once per env step."""
  _ensure_gate_buffers(env)
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  env._gate_entered |= x_rel > _BAR_X
  step = env.episode_length_buf.float()
  fresh = step != env._gate_last_step
  env._gate_timer = torch.where(env._gate_entered & fresh,
                                env._gate_timer + 1.0, env._gate_timer)
  env._gate_last_step = torch.where(fresh, step, env._gate_last_step)
  open_c = bar_clearance_for_level(env.scene.terrain.terrain_levels)
  return (open_c - GATE_CLOSE_RATE * env._gate_timer).clamp(min=GATE_MIN)


def reset_gate(env, env_ids, **_) -> None:
  """Reset event: reopen the gate for fresh episodes."""
  _ensure_gate_buffers(env)
  env._gate_timer[env_ids] = 0.0
  env._gate_entered[env_ids] = False


def gate_margin(env) -> torch.Tensor:
  """> 0 while the trunk fits under the (virtual) ceiling INSIDE the gate span;
  large positive outside the span (the gate only threatens under itself)."""
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  under = (x_rel > _BAR_X) & (x_rel < _BAR_X + BAR_DEPTH)
  trunk_top = robot.data.root_link_pos_w[:, 2] + _TRUNK_TOP_OFF
  m = (gate_clearance(env) - trunk_top) / _GATE_NORM
  return torch.where(under, m, torch.ones_like(m))


def crushed_by_gate(env) -> torch.Tensor:
  return gate_margin(env) < 0.0


# --- margins ---------------------------------------------------------------------

def _g_crawl(env) -> torch.Tensor:
  """Crawl integrity: terrain-relative base height/tilt + nose-down guard."""
  from robot_safety_sandbox.margins import g_terrain_relative
  robot = env.scene["robot"]
  g = g_terrain_relative(env)
  nose_down = robot.data.projected_gravity_b[:, 0]
  return torch.minimum(g, (_PITCH_DOWN_LIMIT - nose_down) / _PITCH_DOWN_NORM)


def _l_crawl_completion(env) -> torch.Tensor:
  """min-form completion: at rest PAST the bar (rest-before never satisfies)."""
  robot = env.scene["robot"]
  speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  return torch.minimum((_V_REST - speed) / _V_REST_NORM,
                       (x_rel - REST_X) / _POS_NORM)


def crawl_twin_margins(env):
  """(g, l) for the STATIC-bar twins."""
  return _g_crawl(env), _l_crawl_completion(env)


def crawl_gate_twin_margins(env):
  """(g, l) for the GATE twins: g additionally carries the descending ceiling."""
  g = torch.minimum(_g_crawl(env), gate_margin(env))
  return g, _l_crawl_completion(env)


# --- env cfgs ---------------------------------------------------------------------

def unitree_go2_crawl_twin_env_cfg(play: bool = False, gate: bool = False,
                                   bar_row: int = 5) -> ManagerBasedRlEnvCfg:
  """Clean twin env: duck base MINUS the single-policy machinery (no forward
  force, no rest windows, no curricula), takeover spawns, fixed bar row
  (row 5 = 0.33 m, comfortably passable; eval pins its own)."""
  from robot_safety_sandbox.envs.go2_crawl.env_cfg import (
    reset_duck_approach, unitree_go2_crawl_env_cfg)

  cfg = unitree_go2_crawl_env_cfg(play=play)
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_duck_approach, mode="reset", params={})
  cfg.events.pop("rest_obstacle_window", None)
  cfg.curriculum = {}
  cfg.scene.terrain.max_init_terrain_level = bar_row
  if cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator = replace(
      cfg.scene.terrain.terrain_generator, curriculum=True)
  cfg.episode_length_s = 8.0   # matches the gap twins' horizon

  if gate:
    cfg.events["reset_gate"] = EventTermCfg(func=reset_gate, mode="reset",
                                            params={})
    cfg.terminations["crushed_by_gate"] = TerminationTermCfg(
      func=crushed_by_gate)
  return cfg


def unitree_go2_crawl_gate_twin_env_cfg(play: bool = False
                                        ) -> ManagerBasedRlEnvCfg:
  return unitree_go2_crawl_twin_env_cfg(play=play, gate=True)
