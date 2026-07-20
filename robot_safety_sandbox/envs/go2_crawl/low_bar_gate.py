"""LOW-BAR crawl, CLOSING-GATE variant -- the COMMITTED-region twin of the
static low-bar benchmark (``low_bar.py``).

Everything is reused wholesale from ``low_bar``: the same Go2 crawl base env,
the same VIRTUAL low-bar terrain, the same reverse-curriculum spawn
(``reset_low_bar``), the same look-ahead curriculum (``low_bar_levels``), the
same completion reach term (``l_low_bar``).  The ONE change: the static
virtual-bar g-term is replaced by a **closing gate** and a ``crushed_by_gate``
termination is added.

This is where the twins DIFFER structurally.  Under the STATIC bar a crouch is
statically safe at every instant (stoppable), so avoid and reach-avoid agree.
Here the ceiling DESCENDS once the base crosses the bar face: lingering under
it -> trunk_top eventually exceeds the clearance -> crush.  Stopping BEFORE the
gate is always safe; crossing requires commitment.  So the avoid filter
livelocks (there is no *stay*-safe action under a closing gate that also makes
progress) while the reach-avoid filter drives through while the window is open
-- the split the static bar cannot produce.

Island-style hazard (mirrors ``twins.py`` E035): the physical/visual beam never
moves (it renders at the OPEN clearance); the descending ceiling lives ONLY in
the g margin and in the ``crushed_by_gate`` termination.

Gate mechanism (adapted from ``twins.py`` to this env's origin-at-bar-face
convention, bar face = x_rel 0):
  * clearance starts at ``bar_clearance`` and, once the base crosses the bar
    face (x_rel > 0), DESCENDS at ``gate_close_rate`` m per env step (one-shot;
    tracked by ``_gate_timer``, step-deduped by ``_gate_last_step`` since the
    hook is called from both the margin and the termination each step), clamped
    to a ``GATE_MIN`` floor.
  * gate_g = (gate_clearance - trunk_top)/BAR_NORM while x_rel in [0, bar_depth]
    (under the gate span), large-positive OUTSIDE the span (the gate only
    threatens under itself).  g = min(g_terrain_relative, pitch_guard, gate_g).
  * ``crushed_by_gate`` termination = gate_g < 0 (trunk_top exceeds the
    descending clearance while under the span).
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from robot_safety_sandbox.margins import CLAMP, g_terrain_relative
from robot_safety_sandbox.envs.go2_crawl.low_bar import (
  BAR_NORM,
  HALF_TRUNK,
  _PITCH_DOWN_LIMIT,
  _PITCH_DOWN_NORM,
  _bar_params,
  _x_rel,
  l_low_bar,
  unitree_go2_low_bar_env_cfg,
)

# --- gate parameters -----------------------------------------------------------
GATE_CLOSE_RATE = 0.0015   # m per env step (~0.075 m/s at dt 0.02): a committed
                           # crawl clears the ~0.4 m span with margin; a
                           # hesitating one is caught. Overridable per env cfg.
GATE_MIN = 0.10            # fully-closed clearance (uncrossable floor)


# --- gate machinery ------------------------------------------------------------

def _ensure_gate_buffers(env) -> None:
  if not hasattr(env, "_gate_timer"):
    n = env.num_envs
    env._gate_timer = torch.zeros(n, device=env.device)
    env._gate_entered = torch.zeros(n, dtype=torch.bool, device=env.device)
    env._gate_last_step = torch.full((n,), -1.0, device=env.device)


def gate_clearance(env) -> torch.Tensor:
  """Per-env VIRTUAL ceiling height: the OPEN clearance (``bar_clearance``)
  until the base crosses the bar face (x_rel > 0), then descending at
  ``gate_close_rate`` (one-shot; reopened by ``reset_gate``).  Called from BOTH
  the margin hook and the termination each step, so the timer advance is
  guarded to once per env step (``_gate_last_step``)."""
  _ensure_gate_buffers(env)
  clearance, _depth = _bar_params(env)
  x_rel = _x_rel(env)
  env._gate_entered |= x_rel > 0.0            # bar face = x_rel 0
  step = env.episode_length_buf.float()
  fresh = step != env._gate_last_step
  env._gate_timer = torch.where(env._gate_entered & fresh,
                                env._gate_timer + 1.0, env._gate_timer)
  env._gate_last_step = torch.where(fresh, step, env._gate_last_step)
  rate = float(getattr(env, "_gate_close_rate", GATE_CLOSE_RATE))
  open_c = torch.full_like(env._gate_timer, clearance)
  return (open_c - rate * env._gate_timer).clamp(min=GATE_MIN)


def reset_gate(env, env_ids, gate_close_rate: float = GATE_CLOSE_RATE,
               **_) -> None:
  """Reset event (mode="reset"): reopen the gate for fresh episodes (timer 0,
  not entered).  ADDED alongside ``reset_low_bar`` -- does not replace it.
  Also stashes the per-env close rate read by ``gate_clearance``."""
  _ensure_gate_buffers(env)
  env._gate_close_rate = float(gate_close_rate)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env._gate_timer[env_ids] = 0.0
  env._gate_entered[env_ids] = False


def gate_g_term(env) -> torch.Tensor:
  """> 0 while the trunk fits under the descending ceiling INSIDE the gate span
  [0, bar_depth]; large-positive OUTSIDE the span (the gate only threatens
  under itself, so a robot past the gate is never crushed)."""
  _clearance, depth = _bar_params(env)
  robot = env.scene["robot"]
  trunk_top = robot.data.root_link_pos_w[:, 2] + HALF_TRUNK
  x_rel = _x_rel(env)
  m = (gate_clearance(env) - trunk_top) / BAR_NORM
  under = (x_rel >= 0.0) & (x_rel <= depth)
  return torch.where(under, m, torch.full_like(m, CLAMP))


def crushed_by_gate(env) -> torch.Tensor:
  """Termination: the descending ceiling caught the trunk (gate_g < 0)."""
  return gate_g_term(env) < 0.0


# --- margins -------------------------------------------------------------------

def low_bar_gate_margins(env):
  """Same as ``low_bar_margins`` but the static virtual-bar term is replaced by
  the descending gate term: g = min(g_terrain_relative, nose-down pitch guard,
  gate_g); l = completion past the bar (unchanged from ``low_bar``)."""
  robot = env.scene["robot"]

  g = g_terrain_relative(env)
  nose_down = robot.data.projected_gravity_b[:, 0]        # sin(pitch); >0 nose-down
  pitch_guard = (_PITCH_DOWN_LIMIT - nose_down) / _PITCH_DOWN_NORM
  g = torch.minimum(g, pitch_guard)
  g = torch.minimum(g, gate_g_term(env))

  return g.clamp(-CLAMP, CLAMP), l_low_bar(env).clamp(-CLAMP, CLAMP)


# --- env cfg builder -----------------------------------------------------------

def unitree_go2_low_bar_gate_env_cfg(play: bool = False,
                                     bar_clearance: float = 0.39,
                                     bar_depth: float = 0.4,
                                     gate_close_rate: float = GATE_CLOSE_RATE
                                     ) -> ManagerBasedRlEnvCfg:
  """LOW-BAR crawl CLOSING-GATE env: identical to ``unitree_go2_low_bar_env_cfg``
  (same reverse-curriculum spawn / terrain / obs / l) EXCEPT the virtual bar
  becomes a descending gate and a ``crushed_by_gate`` termination is added.
  The rendered beam stays at the OPEN clearance; the hazard is analytic."""
  cfg = unitree_go2_low_bar_env_cfg(play=play, bar_clearance=bar_clearance,
                                    bar_depth=bar_depth)
  # Reopen the gate on episode reset (alongside the reverse-curriculum spawn).
  cfg.events["reset_gate"] = EventTermCfg(
    func=reset_gate, mode="reset",
    params={"gate_close_rate": gate_close_rate})
  # The gate REPLACES the static bar, so drop the inherited static bar-strike
  # term; the descending ceiling's crushed_by_gate is this env's bar failure.
  cfg.terminations.pop("bar_strike", None)
  # End the episode when the descending ceiling crushes the trunk.
  cfg.terminations["crushed_by_gate"] = TerminationTermCfg(func=crushed_by_gate)
  return cfg
