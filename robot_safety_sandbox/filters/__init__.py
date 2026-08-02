"""Runtime safety filters, as compositions rather than a class per recipe.

Following Hsu, Hu & Fisac, "The Safety Filter: A Unified View"
(arXiv:2309.05837), Defs. 1-2, a filter is a triple of swappable modules and
there is exactly ONE concrete filter class:

    SafetyFilter(fallback, monitor, intervention)

      fallback      pi^<(x)        PolicyFallback | ZeroFallback
      monitor       Delta(x, u)    ValueMonitor | CriticMonitor |
                                   RolloutMonitor | AdversarialRolloutMonitor
      intervention  phi(x, u_nom)  LeastRestrictiveIntervention |
                                   QCBFIntervention |
                                   HeuristicSmoothingIntervention (non-canonical)

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

``LeastRestrictive`` above means the CANONICAL rule and nothing else: pass the
nominal iff ``Delta(x, u) > eps``, hand over entirely otherwise, with no state
carried between steps. Every switching builder takes ``smoothing=True`` to
substitute :class:`~.intervention.HeuristicSmoothingIntervention` instead —
the latched, median-smoothed, hysteresis-released variant that the gap/crawl
gauntlets were tuned with. It is a genuinely different filter (strictly more
conservative, and not Def-2 valid instant-by-instant), so a run that sets it
must be reported as a smoothed variant of the named filter, never as the
published one. ⚠ Every evaluation before 2026-07-25 (E051/E054 included) used
the smoothed variant, because it was then the only implementation and it was
the default; those numbers do not carry over to the canonical switch.

Margin convention throughout the zoo: safe iff >= 0 (g, l, V, Q alike).
Everything is batched over N parallel envs, torch end-to-end, with per-step
context forwarded as ``**ctx``; callers MUST call ``filter.reset(done)`` every
step (see core.py on the latch contract).
"""

from __future__ import annotations

from .core import SafetyFilter
from .fallback import Fallback, PolicyFallback, ZeroFallback
from .intervention import (
  HeuristicSmoothingIntervention, Intervention, LeastRestrictiveIntervention,
  OptIntervention, QCBFIntervention)
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
  "Intervention", "LeastRestrictiveIntervention",
  "HeuristicSmoothingIntervention", "OptIntervention", "QCBFIntervention",
  "ShadowSim", "MjlabShadowSim", "RolloutStep", "sync_env_state",
  "safety_value_filter", "safety_critic_filter", "qcbf_filter",
  "rollout_filter", "gameplay_filter",
]


# --- builders: the standard compositions -------------------------------------

#: knobs that exist ONLY on HeuristicSmoothingIntervention. Passing any of them
#: is a request for the smoothed variant, and asking for one without
#: ``smoothing=True`` is a naming error worth failing on rather than ignoring.
_SMOOTHING_ONLY = ("caution", "hysteresis", "rest_speed", "median_window",
                   "dip_margin")


def _switch(num_envs: int, device: str, action_dim, smoothing: bool, **switch):
  """Build the switching intervention the recipe asked for.

  Default is the CANONICAL :class:`LeastRestrictiveIntervention` (one memoryless
  comparison), because the builders below are named after published filters and
  those names denote that rule. ``smoothing=True`` substitutes
  :class:`HeuristicSmoothingIntervention`, which is a different, non-canonical
  object — the composition is then a smoothed VARIANT of the named filter.
  """
  if smoothing:
    return HeuristicSmoothingIntervention(num_envs, device,
                                          action_dim=action_dim, **switch)
  extra = sorted(k for k in switch if k in _SMOOTHING_ONLY)
  if extra:
    raise TypeError(
      f"{extra} are HeuristicSmoothingIntervention knobs, but this filter is "
      "being built with the canonical LeastRestrictiveIntervention, whose only "
      "knob is eps (the rule is a single memoryless comparison: pass the "
      "nominal iff Delta > eps). Pass smoothing=True to get the smoothed "
      "variant these knobs belong to — and then report it as a smoothed "
      "variant, not as the published filter.")
  return LeastRestrictiveIntervention(num_envs, device, action_dim=action_dim,
                                      **switch)


def safety_value_filter(num_envs: int, device: str, value_fn, fallback_fn,
                        *, action_dim: int | None = None, telemetry=None,
                        smoothing: bool = False, **switch) -> SafetyFilter:
  """Safety Value Filter: least-restrictive switching on an on-policy V(x).

  ``switch`` takes the chosen intervention's options: ``eps`` for the canonical
  switch, plus (caution, hysteresis, rest_speed, median_window, dip_margin) when
  ``smoothing=True``.
  """
  return SafetyFilter(
    PolicyFallback(num_envs, device, fallback_fn, action_dim),
    ValueMonitor(num_envs, device, value_fn, action_dim),
    _switch(num_envs, device, action_dim, smoothing, **switch),
    telemetry=telemetry)


def safety_critic_filter(num_envs: int, device: str, q_fn, fallback_fn,
                         *, action_dim: int | None = None, telemetry=None,
                         smoothing: bool = False, **switch) -> SafetyFilter:
  """Safety Critic Filter: the same switch, monitored by Q(x, u_nom).

  The published Gameplay-Filters baseline: least-restrictive switching that
  scores the NOMINAL ACTION rather than only the state, so an unsafe action at
  a safe state is caught. Free from the decomposition — no new filter class.
  """
  return SafetyFilter(
    PolicyFallback(num_envs, device, fallback_fn, action_dim),
    CriticMonitor(num_envs, device, q_fn, action_dim),
    _switch(num_envs, device, action_dim, smoothing, **switch),
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
                   smoothing: bool = False, **switch) -> SafetyFilter:
  """Rollout Filter: least-restrictive switching on a simulated H-step future.

  ``shadow`` is a ShadowSim (see rollout.py) with ``num_envs * R`` envs; the
  verdict is the min over the R rollouts of each env.
  """
  return SafetyFilter(
    PolicyFallback(num_envs, device, fallback_fn, action_dim),
    RolloutMonitor(num_envs, device, shadow, horizon, reach_avoid=reach_avoid,
                   recertify_every=recertify_every, action_dim=action_dim),
    _switch(num_envs, device, action_dim, smoothing, **switch),
    telemetry=telemetry)


def gameplay_filter(num_envs: int, device: str, fallback_fn, shadow,
                    horizon: int, adversary_fn, *, reach_avoid: bool = False,
                    recertify_every: int = 1, action_dim: int | None = None,
                    telemetry=None, smoothing: bool = False,
                    **switch) -> SafetyFilter:
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
    _switch(num_envs, device, action_dim, smoothing, **switch),
    telemetry=telemetry)
