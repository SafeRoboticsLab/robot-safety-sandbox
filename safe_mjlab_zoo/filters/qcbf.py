"""Q-CBF / R-CBF projection filter: minimally modify the nominal action.

Where the ValueShield swaps the whole action for the fallback's, this filter
keeps the nominal in charge and corrects its action just enough to satisfy a
discrete-time barrier condition on a learned action-value:

  find u = argmin ||u - u_nom||^2   s.t.   Q(s, u) >= (1 - gamma_cbf) Q_max*0
                                           (zoo convention: Q >= eps is safe)

with Q the safety twin's critic Q(s, u) (safe iff >= 0, like every margin in
the zoo). The constraint is nonlinear in u, so we solve the standard
linearized single-constraint QP in closed form and re-linearize:

  while Q(s, u) < eps:   u <- clip( u + (eps - Q) * dQ/du / ||dQ/du||^2 )

(each step is the exact minimal-norm correction of the linearized
constraint — the same algebra as the analytic CBF-QP solution for one
constraint, no solver needed). If after ``n_iter`` re-linearizations the
constraint still fails by more than ``slack`` (flat/adversarial Q landscape,
or the safe action set is empty at s), the filter declares the projection
infeasible and, when a fallback policy is provided, hands over exactly like
the ValueShield (latched until value recovery); without a fallback it returns
the best projected action found.

Relation to safe_adaptation_dev: that codebase's "Q-CBF" is a threshold
*switching* filter (ValueMonitor on Q(s,u) + full-policy override) — the
architecture our ValueShield already matches. The projection here is the
genuinely CBF-style least-restrictive variant, new in this repo, sharing the
same Q interface so the two are directly comparable.

Practical notes:
- Q must be a differentiable torch callable; gradients are taken w.r.t. the
  action only (one autograd call per re-linearization, batched over envs).
- On-policy twins (SafetyPPO / ReachAvoidPPO) learn V(s), not Q(s, u). Use
  this filter with the off-policy twins (SafetySAC / ReachAvoidSAC critics)
  or a distilled Q head. A V-only workaround (finite-difference through the
  dynamics) is deliberately not provided — it needs a model.
- A learned Q under optimization pressure is a certificate under attack (the
  Goodhart lesson): prefer an ensemble-LCB Q for q_fn, and validate the
  filter against witness rollouts before trusting the table it produces.
"""

from __future__ import annotations

import torch

from .base import FilterInfo, SafetyFilter


class QCBFFilter(SafetyFilter):
  """Least-restrictive action projection through a learned Q(s, a).

  :param q_fn: callable(action=(N,A) requires_grad, **ctx) -> (N,) Q values,
      differentiable w.r.t. the action (zoo convention: safe iff >= 0).
  :param eps: barrier threshold (project until Q >= eps).
  :param n_iter: max re-linearizations per step.
  :param slack: infeasibility margin — Q < eps - slack after n_iter triggers
      the fallback handover.
  :param fallback_fn: optional callable(**ctx) -> (N, A) safety action used
      on infeasibility (latched; released by Q recovery at rest, mirroring
      the ValueShield's release rule).
  :param action_low/high: action-box bounds for the clip step.
  """

  def __init__(self, num_envs: int, device: str, q_fn, eps: float = 0.0,
               n_iter: int = 3, slack: float = 0.05, fallback_fn=None,
               action_low: float = -1.0, action_high: float = 1.0,
               hysteresis: float = 0.15, rest_speed: float = 0.4):
    super().__init__(num_envs, device)
    self.q_fn, self.fallback_fn = q_fn, fallback_fn
    self.eps, self.n_iter, self.slack = eps, n_iter, slack
    self.lo, self.hi = action_low, action_high
    self.hys, self.rest = hysteresis, rest_speed

  def _project(self, a_nom: torch.Tensor, **ctx):
    u = a_nom.clone()
    q = None
    for _ in range(self.n_iter):
      u_g = u.detach().requires_grad_(True)
      q = self.q_fn(action=u_g, **ctx)
      violating = q < self.eps
      if not bool(violating.any()):
        return u.detach(), q.detach()
      grad = torch.autograd.grad(q.sum(), u_g)[0]
      gnorm2 = (grad * grad).sum(dim=-1, keepdim=True).clamp_min(1e-8)
      du = (self.eps - q).clamp_min(0.0).unsqueeze(-1) * grad / gnorm2
      u = torch.where(violating.unsqueeze(-1),
                      (u + du).clamp(self.lo, self.hi), u)
    with torch.no_grad():
      q = self.q_fn(action=u, **ctx)
    return u.detach(), q.detach()

  def act(self, a_nom: torch.Tensor, *, speed: torch.Tensor | None = None,
          **ctx) -> tuple[torch.Tensor, FilterInfo]:
    u_proj, q_proj = self._project(a_nom, **ctx)

    if self.fallback_fn is not None:
      infeasible = q_proj < self.eps - self.slack
      if speed is not None:
        release = (q_proj > self.eps + self.hys) & (speed < self.rest)
      else:
        release = q_proj > self.eps + self.hys
      self.engaged = (self.engaged | infeasible) & ~release
      a_safe = self.fallback_fn(**ctx)
      action = torch.where(self.engaged.unsqueeze(-1), a_safe, u_proj)
    else:
      action = u_proj

    self.engaged_steps += self.engaged.float()
    dev = torch.norm(action - a_nom, dim=-1)
    return action, FilterInfo(engaged=self.engaged, value=q_proj,
                              intervention=dev)
