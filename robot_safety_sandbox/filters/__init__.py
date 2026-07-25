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
                         — the workhorse; examples/eval.py --filter value.
  safety_critic_filter   PolicyFallback + CriticMonitor + LeastRestrictive
                         — the switching filter on an off-policy Q.
  qcbf_filter            PolicyFallback + CriticMonitor + QCBFIntervention
                         — minimal modification under a class-K Q barrier.
  rollout_filter         PolicyFallback + RolloutMonitor + LeastRestrictive
                         — certify by simulating the fallback, not by trusting
                         a learned scalar.
  gameplay_filter        PolicyFallback + AdversarialRolloutMonitor +
                         LeastRestrictive — the same, with the rollout played
                         against a learned adversary.

Note what the last two do NOT change: the intervention. All four recipes share
one ``LeastRestrictiveIntervention`` / ``QCBFIntervention``; going from a value
monitor to a full adversarial rollout swaps exactly one of the three modules.
That is the whole point of the decomposition, and it is load-bearing — if a new
monitor ever needs the intervention edited, the interface is wrong.

Margin convention throughout the zoo: safe iff >= 0 (g, l, V, Q alike).
Everything is batched over N parallel envs, torch end-to-end, with per-step
context forwarded as ``**ctx``; callers MUST call ``filter.reset(done)`` every
step (see core.py on the latch contract).
"""

from __future__ import annotations

from .core import SafetyFilter
from .fallback import Fallback, PolicyFallback, ZeroFallback
from .intervention import (
  Intervention, LeastRestrictiveIntervention, OptIntervention, QCBFIntervention)
from .monitor import (
  AdversarialRolloutMonitor, CriticMonitor, Monitor, RolloutMonitor,
  ValueMonitor)
from .rollout import MjlabShadowSim, RolloutStep, ShadowSim, sync_env_state
from .telemetry import EngagementLog, FilterInfo

__all__ = [
  "SafetyFilter", "FilterInfo", "EngagementLog",
  "Fallback", "PolicyFallback", "ZeroFallback",
  "Monitor", "ValueMonitor", "CriticMonitor", "RolloutMonitor",
  "AdversarialRolloutMonitor",
  "Intervention", "LeastRestrictiveIntervention", "OptIntervention",
  "QCBFIntervention",
  "ShadowSim", "MjlabShadowSim", "RolloutStep", "sync_env_state",
  "safety_value_filter", "safety_critic_filter", "qcbf_filter",
  "rollout_filter", "gameplay_filter",
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


def rollout_filter(num_envs: int, device: str, fallback_fn, shadow, horizon: int,
                   *, reach_avoid: bool = False, recertify_every: int = 1,
                   action_dim: int | None = None, telemetry=None,
                   **switch) -> SafetyFilter:
  """Rollout Filter: least-restrictive switching on a simulated H-step future.

  ``shadow`` is a ShadowSim (see rollout.py) with ``num_envs * R`` envs; the
  verdict is the min over the R rollouts of each env.
  """
  return SafetyFilter(
    PolicyFallback(num_envs, device, fallback_fn, action_dim),
    RolloutMonitor(num_envs, device, shadow, horizon, reach_avoid=reach_avoid,
                   recertify_every=recertify_every, action_dim=action_dim),
    LeastRestrictiveIntervention(num_envs, device, action_dim=action_dim,
                                 **switch),
    telemetry=telemetry)


def gameplay_filter(num_envs: int, device: str, fallback_fn, shadow,
                    horizon: int, adversary_fn, *, reach_avoid: bool = False,
                    recertify_every: int = 1, action_dim: int | None = None,
                    telemetry=None, **switch) -> SafetyFilter:
  """Gameplay Filter: least-restrictive switching on an ADVERSARIAL rollout.

  Identical to :func:`rollout_filter` but for the disturbance policy driving the
  dstb sub-space of the shadow sim's action — which is what makes the certified
  future a game rather than a nominal-dynamics guess. Note it reuses the SAME
  intervention type (and the same fallback) as the value filter: only the
  monitor changes.

  ``shadow`` must have the concatenated ``[ctrl, dstb]`` action space
  (``MjlabShadowSim.from_task(..., adversary=True)``).
  """
  return SafetyFilter(
    PolicyFallback(num_envs, device, fallback_fn, action_dim),
    AdversarialRolloutMonitor(num_envs, device, shadow, horizon, adversary_fn,
                              reach_avoid=reach_avoid,
                              recertify_every=recertify_every,
                              action_dim=action_dim),
    LeastRestrictiveIntervention(num_envs, device, action_dim=action_dim,
                                 **switch),
    telemetry=telemetry)
