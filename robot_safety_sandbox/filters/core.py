"""The one concrete safety filter: a composition of fallback/monitor/intervention.

Following Hsu, Hu & Fisac, "The Safety Filter: A Unified View"
(arXiv:2309.05837), Definitions 1-2, a safety filter is not a family of classes
but a COMPOSITION of three swappable modules. There is exactly one concrete
``SafetyFilter`` here and it is never subclassed: a new filter is a new triple,
not a new type.

  fallback      pi^<(x)        the backup controller (fallback.py)
  monitor       Delta(x, u)    how safe is applying u at x (monitor.py)
  intervention  phi(x, u_nom)  what to do about it (intervention.py)

``__call__`` forwards to the intervention and returns (action, FilterInfo). It
contains NO monitor/fallback branching: the intervention consults both itself,
which is exactly what lets the same LeastRestrictiveIntervention serve the
value, critic and gameplay filters unchanged.

Episode boundaries matter. Modules may hold per-env state (the switch latch,
the margin history), and an uncleared latch is a real bug class — a stale latch
once left a fallback driving fresh episodes for a 77% livelock — so callers
MUST call ``reset(done)`` every step with the env's done mask. ``reset`` routes
to all three modules; whichever ones hold state clear it.
"""

from __future__ import annotations

import torch

from .fallback import Fallback
from .intervention import Intervention
from .monitor import Monitor
from .telemetry import EngagementLog, FilterInfo


class SafetyFilter:
  """Compose (fallback, monitor, intervention) into a runtime safety filter.

  :param fallback: a Fallback (pi^<).
  :param monitor: a Monitor (Delta).
  :param intervention: an Intervention (phi) — the module that decides.
  :param telemetry: an EngagementLog, or None to get a default one. Telemetry
      only observes; it never affects the returned action.
  """

  def __init__(self, fallback: Fallback, monitor: Monitor,
               intervention: Intervention,
               telemetry: EngagementLog | None = None):
    self.fallback, self.monitor, self.intervention = (
      fallback, monitor, intervention)
    self._check_consistent()
    self.num_envs, self.device = intervention.num_envs, intervention.device
    self.telemetry = (EngagementLog(self.num_envs, self.device)
                      if telemetry is None else telemetry)

  def _check_consistent(self) -> None:
    """The three modules must agree on num_envs, device and action dim."""
    mods = {"fallback": self.fallback, "monitor": self.monitor,
            "intervention": self.intervention}
    base = {"fallback": Fallback, "monitor": Monitor,
            "intervention": Intervention}
    for name, mod in mods.items():
      if not isinstance(mod, base[name]):
        raise TypeError(f"{name} must be a {base[name].__name__}, "
                        f"got {type(mod).__name__}")
    for attr in ("num_envs", "device", "action_dim"):
      seen = {name: getattr(mod, attr) for name, mod in mods.items()}
      vals = {v for v in seen.values() if v is not None}
      if len(vals) > 1:
        raise ValueError(f"filter modules disagree on {attr}: {seen}")

  def __call__(self, a_nom: torch.Tensor,
               **ctx) -> tuple[torch.Tensor, FilterInfo]:
    """Filter the nominal action; return (action, FilterInfo)."""
    action, info = self.intervention(a_nom, self.fallback, self.monitor, **ctx)
    self.telemetry.update(info)
    return action, info

  # Back-compat spelling of __call__ (the monolithic filters exposed .act).
  act = __call__

  def reset(self, done: torch.Tensor) -> None:
    """Clear per-env state for envs that just finished. Call EVERY step."""
    self.fallback.reset(done)
    self.monitor.reset(done)
    self.intervention.reset(done)
