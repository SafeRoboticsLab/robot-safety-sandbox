"""Shared filter contract: propose -> monitor -> override, batched.

The structure mirrors safe_adaptation_dev's SafetyFilter/Monitor split (the
lab's prior filter codebase) with two changes for this repo: everything is
batched over N parallel envs as torch tensors, and the sign convention is the
zoo's margin convention (safe iff value >= 0; safe_adaptation_dev's cost-to-go
critics use the opposite sign).

A filter does not own policies. It receives the nominal action (and whatever
per-step context its monitor needs) and returns the filtered action plus a
FilterInfo. Episode boundaries matter — every filter carries per-env latches
or histories, and an uncleared latch is a real bug class (a stale latch left
a fallback driving fresh episodes: 77% livelock) — so callers MUST call
``reset(done)`` every step with the env's done mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FilterInfo:
  """Per-step filter telemetry (all tensors are (N,) on the filter device)."""

  engaged: torch.Tensor          # bool: fallback/override active this step
  value: torch.Tensor            # the monitored quantity (V, Q, or rollout g)
  caution: torch.Tensor | None = None   # bool: pre-engagement caution band
  intervention: torch.Tensor | None = None  # ||u_filt - u_nom|| per env


class SafetyFilter:
  """Base class: latch bookkeeping + the reset contract."""

  def __init__(self, num_envs: int, device: str):
    self.num_envs = num_envs
    self.device = device
    self.engaged = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.engaged_steps = torch.zeros(num_envs, device=device)

  def act(self, a_nom: torch.Tensor, **ctx) -> tuple[torch.Tensor, FilterInfo]:
    """Return (filtered action, info). Subclasses implement."""
    raise NotImplementedError

  def reset(self, done: torch.Tensor) -> None:
    """Clear per-env state for envs that just finished. Call EVERY step."""
    self.engaged &= ~done

  def intervention_rate(self, steps: int) -> float:
    """Fraction of env-steps spent overriding the nominal so far."""
    return float(self.engaged_steps.sum() / (self.num_envs * max(steps, 1)))
