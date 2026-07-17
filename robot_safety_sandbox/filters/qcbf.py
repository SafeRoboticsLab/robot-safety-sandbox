"""Q-CBF / R-CBF projection filter: minimally modify the nominal action.

Faithful (batched) port of safe_adaptation_dev's projected-gradient R-CBF
(script/eval_safety_filter.py::rcbf_projected_gradient, derivation in
research_notes/rcbf_gradient_method.md). The optimization per env:

  u*(x) = argmin_u ||u_task - u||^2   s.t.   Q(x, u) >= kappa * V(x)

with Q the learned (robust) state-action safety value (zoo convention: safe
iff >= 0) and V(x) = Q(x, u_safe(x)) the safety value under the safety
policy's own action — i.e. the barrier is CLASS-K (preserve a kappa-fraction
of the current safety level), not a fixed floor. Algorithm, per the note:

  1. tau <- kappa * V(x);  if Q(x, u_task) >= tau: return u_task
  2. if V(x) < tau (kappa > 1 with V > 0, or extreme states): return u_safe
     (best-effort — even the safety action cannot meet the threshold)
  3. normalized gradient ascent u <- Proj_U(u + lr * dQ/du / ||dQ/du||) until
     Q(x, u) >= tau (gradients via autograd through the critic — and, for a
     robust Q(x, u, pi_dstb(x, u)), through the disturbance actor: pass a
     q_fn that composes them and the total derivative comes for free)
  4. backtracking binary search on the segment [u_task, u_feas] for the
     closest-to-u_task action that is still feasible
  5. never feasible within n_iter: return the best candidate seen,
     initialized to u_safe (stateless best-effort, matching the reference)

Differences from the reference (deliberate, both mechanical):
- batched over N envs (per-env masks replace the scalar early returns);
- Q convention is the zoo's "safe iff >= 0" (the reference's critics are
  trained the same way in ISAACS, so no sign flip is actually involved).

The legacy 1-D variant (line search on the blend (1-a) u_task + a u_safe)
is dominated by this method (see the note's comparison) and is not ported;
ValueShield covers the a in {0,1} switching case.

Practical notes:
- q_fn must be differentiable w.r.t. the action; one autograd call per
  ascent step, batched over envs.
- On-policy twins (SafetyPPO / ReachAvoidPPO) learn V(s), not Q(s, a): use
  this filter with off-policy twins (SafetySAC / ReachAvoidSAC / IsaacsSAC / GameplaySAC
  critics) or a distilled Q head.
- A learned Q under optimization pressure is a certificate under attack
  (the Goodhart lesson): prefer an ensemble-LCB q_fn and validate against
  witness rollouts before trusting the numbers.
"""

from __future__ import annotations

import torch

from .base import FilterInfo, SafetyFilter


class QCBFFilter(SafetyFilter):
  """Least-restrictive action projection through a learned Q(s, a).

  :param q_fn: callable(action=(N,A), **ctx) -> (N,) Q values, differentiable
      w.r.t. the action (zoo convention: safe iff >= 0).
  :param fallback_fn: callable(**ctx) -> (N, A) the safety policy's action
      u_safe. Required: it defines V(x) = Q(x, u_safe) and the best-effort
      action.
  :param kappa: barrier coefficient in [0, 1]; the constraint is
      Q(x, u) >= kappa * V(x).
  :param lr: normalized-gradient-ascent step size in action units.
  :param n_iter: max ascent steps.
  :param n_backtrack: binary-search refinements toward u_task.
  :param action_low/high: action-box bounds for the projection clip.
  """

  def __init__(self, num_envs: int, device: str, q_fn, fallback_fn,
               kappa: float = 0.8, lr: float = 0.05, n_iter: int = 10,
               n_backtrack: int = 10, action_low: float = -1.0,
               action_high: float = 1.0):
    super().__init__(num_envs, device)
    self.q_fn, self.fallback_fn = q_fn, fallback_fn
    self.kappa, self.lr = kappa, lr
    self.n_iter, self.n_backtrack = n_iter, n_backtrack
    self.lo, self.hi = action_low, action_high

  def _q(self, action, ctx, grad: bool = False):
    if grad:
      return self.q_fn(action=action, **ctx)
    with torch.no_grad():
      return self.q_fn(action=action, **ctx)

  def act(self, a_nom: torch.Tensor, **ctx) -> tuple[torch.Tensor, FilterInfo]:
    a_safe = self.fallback_fn(**ctx)
    with torch.no_grad():
      v_hat = self.q_fn(action=a_safe, **ctx)          # V(x) = Q(x, u_safe)
      tau = self.kappa * v_hat
      q_task = self.q_fn(action=a_nom, **ctx)

    pass_through = q_task >= tau                        # step 1
    best_effort = v_hat < tau                           # step 2
    need = ~pass_through & ~best_effort

    # step 3: normalized gradient ascent from u_task until feasible
    u = a_nom.clone()
    u_feas = a_safe.clone()                             # default if never found
    found = torch.zeros_like(need)
    if bool(need.any()):
      for _ in range(self.n_iter):
        u_g = u.detach().requires_grad_(True)
        q = self.q_fn(action=u_g, **ctx)
        newly = need & ~found & (q.detach() >= tau)
        u_feas = torch.where(newly.unsqueeze(-1), u.detach(), u_feas)
        found |= newly
        active = need & ~found
        if not bool(active.any()):
          break
        grad = torch.autograd.grad(q.sum(), u_g)[0]
        gnorm = grad.norm(dim=-1, keepdim=True)
        step = self.lr * grad / gnorm.clamp_min(1e-8)
        stalled = (gnorm.squeeze(-1) < 1e-8) & active
        active = active & ~stalled
        u = torch.where(active.unsqueeze(-1),
                        (u.detach() + step).clamp(self.lo, self.hi),
                        u.detach())
      # final feasibility check for the last iterate
      with torch.no_grad():
        q = self.q_fn(action=u, **ctx)
      newly = need & ~found & (q >= tau)
      u_feas = torch.where(newly.unsqueeze(-1), u, u_feas)
      found |= newly

      # step 4: backtrack — binary search [u_task, u_feas] per found env
      if bool(found.any()):
        u_lo, u_hi = a_nom.clone(), u_feas.clone()
        for _ in range(self.n_backtrack):
          u_mid = 0.5 * (u_lo + u_hi)
          with torch.no_grad():
            q_mid = self.q_fn(action=u_mid, **ctx)
          ok = (q_mid >= tau).unsqueeze(-1)
          u_hi = torch.where(ok, u_mid, u_hi)
          u_lo = torch.where(ok, u_lo, u_mid)
        u_feas = torch.where(found.unsqueeze(-1), u_hi, u_feas)

    # compose: pass-through | projected | best-effort u_safe (step 5 folds
    # never-feasible envs into u_safe via u_feas's initialization)
    action = torch.where(pass_through.unsqueeze(-1), a_nom,
                         torch.where(best_effort.unsqueeze(-1), a_safe,
                                     u_feas))
    with torch.no_grad():
      q_final = self.q_fn(action=action, **ctx)

    intervened = ~pass_through
    self.engaged = intervened          # stateless (telemetry only, no latch)
    self.engaged_steps += intervened.float()
    dev = torch.norm(action - a_nom, dim=-1)
    return action, FilterInfo(engaged=intervened, value=q_final,
                              intervention=dev)
