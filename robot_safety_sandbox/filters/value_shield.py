"""Value-based shielding: latched V(s)-threshold switch with a caution band.

The protocol proven out by the gap/crawl filter gauntlets (this is the
library form of examples/eval_filter.py's BatchValueFilter):

  engage   when median-smoothed V <= eps  (or raw V clearly below)
  release  when V > eps + hysteresis AND the robot is near rest
  caution  while eps < V <= caution and not engaged: the caller should slow
           the nominal down (e.g. zero its velocity command) so a later
           handover happens from a braking stance instead of mid-trot, and a
           released nominal doesn't sprint straight back into re-engagement
           chatter.

Field rules baked in:
- The V history is a 5-step median (single-step V dips at contact events
  otherwise cause spurious engagements).
- Fresh episodes reseed the history and drop the latch (the reset contract).
- Release requires NEAR REST, not just recovered V: releasing at speed hands
  the nominal a state it never visits in training.
"""

from __future__ import annotations

import torch

from .base import FilterInfo, SafetyFilter


class ValueShield(SafetyFilter):
  """Latched eps-switch on a learned safety value.

  :param value_fn: callable(**ctx) -> (N,) tensor of V(s) (zoo convention:
      safe iff >= 0). Typically wraps ``policy.predict_values`` on the safety
      twin's normalized observation.
  :param fallback_fn: callable(**ctx) -> (N, A) fallback action (the safety
      policy's own action).
  :param eps: engagement threshold on V.
  :param caution: upper edge of the caution band (>= eps).
  :param hysteresis: release requires V > eps + hysteresis.
  :param rest_speed: release additionally requires speed < rest_speed.
  """

  def __init__(self, num_envs: int, device: str, value_fn, fallback_fn,
               eps: float = 0.0, caution: float = 0.45,
               hysteresis: float = 0.15, rest_speed: float = 0.4,
               median_window: int = 5):
    super().__init__(num_envs, device)
    self.value_fn, self.fallback_fn = value_fn, fallback_fn
    self.eps, self.caution = eps, max(caution, eps)
    self.hys, self.rest = hysteresis, rest_speed
    self.window = median_window
    self.v_hist: torch.Tensor | None = None
    self.caution_steps = torch.zeros(num_envs, device=device)

  def act(self, a_nom: torch.Tensor, *, speed: torch.Tensor,
          fresh: torch.Tensor, **ctx) -> tuple[torch.Tensor, FilterInfo]:
    """``speed``: (N,) planar base speed; ``fresh``: (N,) bool, True on the
    first step of an episode. Extra kwargs are forwarded to value_fn and
    fallback_fn."""
    v_raw = self.value_fn(**ctx)
    if self.v_hist is None:
      self.v_hist = v_raw.unsqueeze(0).repeat(self.window, 1)
    self.v_hist = torch.cat([self.v_hist[1:], v_raw.unsqueeze(0)], dim=0)
    if bool(fresh.any()):
      self.v_hist[:, fresh] = v_raw[fresh].unsqueeze(0)
      self.engaged &= ~fresh
    v_med = self.v_hist.median(dim=0).values

    engage = (v_med <= self.eps) | (v_raw <= self.eps - 0.15)
    release = (v_med > self.eps + self.hys) & (speed < self.rest)
    self.engaged = (self.engaged | engage) & ~release
    caution = (v_med <= self.caution) & ~self.engaged

    self.engaged_steps += self.engaged.float()
    self.caution_steps += caution.float()

    a_safe = self.fallback_fn(**ctx)
    action = torch.where(self.engaged.unsqueeze(-1), a_safe, a_nom)
    return action, FilterInfo(engaged=self.engaged, value=v_raw,
                              caution=caution)
