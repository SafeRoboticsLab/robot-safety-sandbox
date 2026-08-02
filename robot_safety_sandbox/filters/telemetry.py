"""Filter telemetry: the per-step info record and the cumulative counters.

Telemetry is deliberately OUTSIDE the filter modules. A latch, a value history
or a release hysteresis changes what the robot does and therefore lives in the
module that owns it; counters only describe what happened and live here, so
that swapping a monitor or an intervention never silently changes the numbers
a paper reports.
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


class EngagementLog:
  """Cumulative engagement counters over a run, batched over N envs."""

  def __init__(self, num_envs: int, device: str):
    self.num_envs, self.device = num_envs, device
    self.engaged_steps = torch.zeros(num_envs, device=device)
    self.caution_steps = torch.zeros(num_envs, device=device)

  def update(self, info: FilterInfo) -> None:
    """Accumulate one step's FilterInfo."""
    self.engaged_steps += info.engaged.float()
    if info.caution is not None:
      self.caution_steps += info.caution.float()

  def intervention_rate(self, steps: int) -> float:
    """Fraction of env-steps spent overriding the nominal so far."""
    return float(self.engaged_steps.sum() / (self.num_envs * max(steps, 1)))

  def caution_rate(self, steps: int) -> float:
    """Fraction of env-steps spent in the caution band so far."""
    return float(self.caution_steps.sum() / (self.num_envs * max(steps, 1)))

  def reset(self) -> None:
    """Zero the counters (start a fresh accounting window)."""
    self.engaged_steps.zero_()
    self.caution_steps.zero_()
