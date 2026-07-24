"""Runtime safety filters, as compositions rather than a class per recipe.

Following Hsu, Hu & Fisac, "The Safety Filter: A Unified View"
(arXiv:2309.05837), Defs. 1-2, a filter is a triple of swappable modules and
there is exactly ONE concrete filter class:

    SafetyFilter(fallback, monitor, intervention)

      fallback      pi^<(x)        PolicyFallback | ZeroFallback
      monitor       Delta(x, u)    ValueMonitor | CriticMonitor |
                                   RolloutMonitor | AdversarialRolloutMonitor
      intervention  phi(x, u_nom)  LeastRestrictiveIntervention |
                                   QCBFIntervention

The standard recipes are just triples, and the builders below assemble them:

  safety_value_filter    PolicyFallback + ValueMonitor  + LeastRestrictive
                         — the workhorse; what examples/eval_filter.py deploys.
  safety_critic_filter   PolicyFallback + CriticMonitor + LeastRestrictive
                         — the switching filter on an off-policy Q.
  qcbf_filter            PolicyFallback + CriticMonitor + QCBFIntervention
                         — minimal modification under a class-K Q barrier.
  gameplay_filter        PolicyFallback + AdversarialRolloutMonitor +
                         LeastRestrictive — same intervention as the value
                         filter; gated on the shadow-sim path (see monitor.py).

Margin convention throughout the zoo: safe iff >= 0 (g, l, V, Q alike).
Everything is batched over N parallel envs, torch end-to-end, with per-step
context forwarded as ``**ctx``; callers MUST call ``filter.reset(done)`` every
step (see core.py on the latch contract).
"""

from __future__ import annotations

import warnings

from .core import SafetyFilter
from .fallback import Fallback, PolicyFallback, ZeroFallback
from .intervention import (
  Intervention, LeastRestrictiveIntervention, OptIntervention, QCBFIntervention)
from .monitor import (
  AdversarialRolloutMonitor, CriticMonitor, Monitor, RolloutMonitor,
  ValueMonitor)
from .telemetry import EngagementLog, FilterInfo

__all__ = [
  "SafetyFilter", "FilterInfo", "EngagementLog",
  "Fallback", "PolicyFallback", "ZeroFallback",
  "Monitor", "ValueMonitor", "CriticMonitor", "RolloutMonitor",
  "AdversarialRolloutMonitor",
  "Intervention", "LeastRestrictiveIntervention", "OptIntervention",
  "QCBFIntervention",
  "safety_value_filter", "safety_critic_filter", "qcbf_filter",
  "gameplay_filter",
  "ValueShield", "QCBFFilter", "RolloutShield",
]


# --- builders: the standard compositions -------------------------------------

def safety_value_filter(num_envs: int, device: str, value_fn, fallback_fn,
                        *, action_dim: int | None = None, telemetry=None,
                        **switch) -> SafetyFilter:
  """Safety Value Filter: latched eps-switch on an on-policy V(x).

  ``switch`` takes LeastRestrictiveIntervention's options (eps, caution,
  hysteresis, rest_speed, median_window, dip_margin).
  """
  return SafetyFilter(
    PolicyFallback(num_envs, device, fallback_fn, action_dim),
    ValueMonitor(num_envs, device, value_fn, action_dim),
    LeastRestrictiveIntervention(num_envs, device, action_dim=action_dim,
                                 **switch),
    telemetry=telemetry)


def safety_critic_filter(num_envs: int, device: str, q_fn, fallback_fn,
                         *, action_dim: int | None = None, telemetry=None,
                         **switch) -> SafetyFilter:
  """Safety Critic Filter: the same latched switch, monitored by Q(x, u_nom).

  The published Gameplay-Filters baseline: least-restrictive switching that
  scores the NOMINAL ACTION rather than only the state, so an unsafe action at
  a safe state is caught. Free from the decomposition — no new filter class.
  """
  return SafetyFilter(
    PolicyFallback(num_envs, device, fallback_fn, action_dim),
    CriticMonitor(num_envs, device, q_fn, action_dim),
    LeastRestrictiveIntervention(num_envs, device, action_dim=action_dim,
                                 **switch),
    telemetry=telemetry)


def qcbf_filter(num_envs: int, device: str, q_fn, fallback_fn, *,
                action_dim: int | None = None, telemetry=None,
                **opt) -> SafetyFilter:
  """Robust Q-CBF Filter: minimal modification under Q(x, u) >= kappa * V(x).

  ``opt`` takes QCBFIntervention's options (kappa, lr, n_iter, n_backtrack,
  action_low, action_high).
  """
  return SafetyFilter(
    PolicyFallback(num_envs, device, fallback_fn, action_dim),
    CriticMonitor(num_envs, device, q_fn, action_dim),
    QCBFIntervention(num_envs, device, action_dim=action_dim, **opt),
    telemetry=telemetry)


def gameplay_filter(num_envs: int, device: str, fallback_fn, *,
                    action_dim: int | None = None, telemetry=None,
                    monitor_kwargs: dict | None = None,
                    **switch) -> SafetyFilter:
  """Gameplay Filter: least-restrictive switching on an adversarial rollout.

  Note it reuses the SAME intervention type as the value filter — only the
  monitor changes. Construction currently raises: the adversarial rollout
  monitor is gated on the shadow-simulation path (see monitor.py).
  """
  return SafetyFilter(
    PolicyFallback(num_envs, device, fallback_fn, action_dim),
    AdversarialRolloutMonitor(num_envs, device, **(monitor_kwargs or {})),
    LeastRestrictiveIntervention(num_envs, device, action_dim=action_dim,
                                 **switch),
    telemetry=telemetry)


# --- deprecated aliases (one release) ----------------------------------------

def _deprecated(old: str, new: str) -> None:
  warnings.warn(
    f"{old} is deprecated and will be removed in the next release; use "
    f"{new} (a SafetyFilter composition) instead.",
    DeprecationWarning, stacklevel=3)


def ValueShield(num_envs, device, value_fn, fallback_fn, **kw):  # noqa: N802
  """Deprecated alias for :func:`safety_value_filter`."""
  _deprecated("ValueShield", "safety_value_filter")
  return safety_value_filter(num_envs, device, value_fn, fallback_fn, **kw)


def QCBFFilter(num_envs, device, q_fn, fallback_fn, **kw):  # noqa: N802
  """Deprecated alias for :func:`qcbf_filter`."""
  _deprecated("QCBFFilter", "qcbf_filter")
  return qcbf_filter(num_envs, device, q_fn, fallback_fn, **kw)


def RolloutShield(*args, **kw):  # noqa: N802
  """Deprecated alias for :func:`gameplay_filter` (still raises: see above)."""
  _deprecated("RolloutShield", "gameplay_filter")
  return gameplay_filter(*args, **kw)
