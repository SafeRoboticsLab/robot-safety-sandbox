"""Monitors Delta(x, u) : how safe is it to apply u at x?

Second of the three modules a safety filter composes (Hsu, Hu & Fisac, "The
Safety Filter: A Unified View", Def. 1: Delta : H x U -> R). The monitor ALWAYS
receives the action, one interface for all styles — ``ValueMonitor`` simply
ignores it, because a state value V(x) is a monitor that happens not to depend
on u. Sign convention throughout the zoo: safe iff Delta >= 0.

Three styles, in increasing order of cost and fidelity:

  ValueMonitor    V(x) from an on-policy twin (SafetyPPO1P / ReachAvoidPPO1P,
                  or their 2P counterparts).
  CriticMonitor   Q(x, u) from an off-policy twin (any of the four SAC cells:
                  {Safety,ReachAvoid}SAC{1P,2P}). Also exposes the raw differentiable
                  ``q_fn`` so an optimization-based intervention can take
                  dQ/du through it.
  RolloutMonitor  simulate the fallback for H steps from the successor state
                  and take the min margin over the horizon; the adversarial
                  variant additionally minimizes over a disturbance policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Monitor(ABC):
  """Delta(x, u) -> (N,). Safe iff >= 0."""

  def __init__(self, num_envs: int, device: str, action_dim: int | None = None):
    self.num_envs, self.device = num_envs, device
    self.action_dim = action_dim

  @abstractmethod
  def __call__(self, a: torch.Tensor, fallback=None, **ctx) -> torch.Tensor:
    """Score the action ``a`` (N, A) at the current state; return (N,).

    ``fallback`` is the composition's Fallback, passed for the monitors that
    need pi^< to define their margin (the rollout monitors simulate it; the
    value/critic monitors ignore it).
    """

  def reset(self, done: torch.Tensor) -> None:
    """Clear per-env state for envs that just finished. Default: none held."""


class ValueMonitor(Monitor):
  """V(x) from an on-policy safety twin — action-independent by construction.

  :param value_fn: callable(**ctx) -> (N,) tensor of V(x) (safe iff >= 0).
      Typically wraps ``policy.predict_values`` on the twin's normalized obs.
  """

  def __init__(self, num_envs: int, device: str, value_fn,
               action_dim: int | None = None):
    super().__init__(num_envs, device, action_dim)
    self.value_fn = value_fn

  def __call__(self, a: torch.Tensor | None = None, fallback=None,
               **ctx) -> torch.Tensor:
    return self.value_fn(**ctx)


class CriticMonitor(Monitor):
  """Q(x, u) from an off-policy safety twin.

  :param q_fn: callable(action=(N, A), **ctx) -> (N,) Q values (safe iff >= 0),
      DIFFERENTIABLE w.r.t. the action. Kept as a public attribute because
      optimization-based interventions (QCBFIntervention) need the gradient,
      while ``__call__`` itself is a no-grad convenience evaluation.
  """

  def __init__(self, num_envs: int, device: str, q_fn,
               action_dim: int | None = None):
    super().__init__(num_envs, device, action_dim)
    self.q_fn = q_fn

  def __call__(self, a: torch.Tensor, fallback=None, **ctx) -> torch.Tensor:
    with torch.no_grad():
      return self.q_fn(action=a, **ctx)


class RolloutMonitor(Monitor):
  """Certify u by an imagined H-step fallback rollout (NOT YET IMPLEMENTED).

  Instead of trusting a learned scalar (V or Q), simulate the fallback policy
  from the successor state of u and return the MIN margin over the horizon, so
  the action is certified only if the fallback can carry the system safely from
  wherever u leaves it.

  Planned signature: RolloutMonitor(num_envs, device, shadow_env, horizon,
  margin_fn, recertify_every=1).

  Design intent for the mjlab implementation (why this is a stub):
  - Needs a *shadow simulation*: either a second batched mjlab env stepped with
    saved states (mjlab state save/restore cannot yet round-trip observation-
    history buffers — the same limitation that broke respawn-based commitment
    analysis; live-switch semantics avoided it, a rollout monitor cannot), or a
    learned dynamics model, or a paired env that accepts explicit state setting.
  - Cost: one fallback rollout of horizon H per env per step (amortizable by
    only re-certifying every k steps and latching in between).

  The interface is fixed now so filter-comparison code can be written against
  all three monitor styles; construction raises until the shadow-sim path
  exists. Note that only the MONITOR is missing — the interventions compose
  with it unchanged.
  """

  def __init__(self, *args, **kwargs):
    raise NotImplementedError(
      "RolloutMonitor needs a shadow-simulation path (state save/restore "
      "including observation-history buffers, or a learned model). The "
      "interface is reserved; see the class docstring for the design.")

  def __call__(self, a: torch.Tensor, fallback=None,
               **ctx) -> torch.Tensor:  # pragma: no cover - unreachable
    raise NotImplementedError


class AdversarialRolloutMonitor(RolloutMonitor):
  """Rollout monitor with a min over a disturbance policy (NOT IMPLEMENTED).

  The gameplay variant: the imagined rollout is played against a trained
  adversary, so the margin is a worst-case (over the disturbance class) rather
  than a nominal-dynamics estimate.

  Planned signature: AdversarialRolloutMonitor(num_envs, device, shadow_env,
  horizon, margin_fn, adversary_fn, recertify_every=1).
  """
