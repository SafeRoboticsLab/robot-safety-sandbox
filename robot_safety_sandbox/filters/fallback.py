"""Fallback policies pi^< : the backup controller a filter can hand over to.

First of the three modules a safety filter composes (Hsu, Hu & Fisac, "The
Safety Filter: A Unified View", Def. 1): the fallback proposes the action that
the monitor certifies against and the intervention falls back to.

Batched over N parallel envs, torch end-to-end; the per-step context (typically
the safety twin's normalized observation) arrives as ``**ctx`` so a fallback
never needs to know how the caller obtained it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Fallback(ABC):
  """pi^<(x) -> (N, A). Stateless unless a subclass says otherwise."""

  def __init__(self, num_envs: int, device: str, action_dim: int | None = None):
    self.num_envs, self.device = num_envs, device
    self.action_dim = action_dim

  @abstractmethod
  def __call__(self, **ctx) -> torch.Tensor:
    """Return the fallback action, (N, A)."""

  def reset(self, done: torch.Tensor) -> None:
    """Clear per-env state for envs that just finished. Default: none held."""


class PolicyFallback(Fallback):
  """A trained safety policy's own action.

  :param policy_fn: callable(**ctx) -> (N, A). Typically wraps the safety
      twin's ``policy._predict(obs, deterministic=True)``; the caller owns any
      clipping/normalization so this stays a pure adapter.
  """

  def __init__(self, num_envs: int, device: str, policy_fn,
               action_dim: int | None = None):
    super().__init__(num_envs, device, action_dim)
    self.policy_fn = policy_fn

  def __call__(self, **ctx) -> torch.Tensor:
    return self.policy_fn(**ctx)


class ZeroFallback(Fallback):
  """Null action — the do-nothing backup, for tests and ablations."""

  def __init__(self, num_envs: int, device: str, action_dim: int):
    super().__init__(num_envs, device, action_dim)

  def __call__(self, **ctx) -> torch.Tensor:
    return torch.zeros(self.num_envs, self.action_dim, device=self.device)
