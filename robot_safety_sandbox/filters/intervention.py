"""Interventions phi(x, u) : how to modify the nominal when it isn't certified.

Third of the three modules a safety filter composes (Hsu, Hu & Fisac, "The
Safety Filter: A Unified View", Def. 2). The intervention is the ONLY module
with authority over the returned action; it consults the monitor and the
fallback itself, which is why ``SafetyFilter`` carries no branching logic.

Two families:

  LeastRestrictiveIntervention   hard binary switch — pass the nominal through
                                 or hand over to the fallback entirely.
  OptIntervention                minimal modification: solve for the closest
                                 action to the nominal that the monitor still
                                 certifies. ``QCBFIntervention`` is the
                                 projected-gradient solver for the class-K
                                 barrier constraint Q(x, u) >= kappa * V(x).

Margin convention throughout: safe iff >= 0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .telemetry import FilterInfo


class Intervention(ABC):
  """phi(x, u_nom) -> (N, A), returned alongside the step's FilterInfo."""

  def __init__(self, num_envs: int, device: str, action_dim: int | None = None):
    self.num_envs, self.device = num_envs, device
    self.action_dim = action_dim

  @abstractmethod
  def __call__(self, a_nom: torch.Tensor, fallback, monitor,
               **ctx) -> tuple[torch.Tensor, FilterInfo]:
    """Return (filtered action (N, A), FilterInfo)."""

  def reset(self, done: torch.Tensor) -> None:
    """Clear per-env state for envs that just finished. Default: none held."""


class LeastRestrictiveIntervention(Intervention):
  """Latched eps-switch on the monitored margin, with a caution band.

  The protocol proven out by the gap/crawl filter gauntlets (the library form
  of the original eval_filter.py BatchValueFilter):

    engage   when median-smoothed Delta <= eps  (or raw Delta clearly below)
    release  when Delta > eps + hysteresis AND the robot is near rest
    caution  while eps < Delta <= caution and not engaged: the caller should
             slow the nominal down (e.g. zero its velocity command) so a later
             handover happens from a braking stance instead of mid-trot, and a
             released nominal doesn't sprint straight back into re-engagement
             chatter.

  Field rules baked in (FUNCTIONAL state, not telemetry — they change what the
  robot does, so they live here rather than in EngagementLog):
  - The margin history is a 5-step median (single-step dips at contact events
    otherwise cause spurious engagements).
  - Fresh episodes reseed the history and drop the latch (the reset contract).
  - Release requires NEAR REST, not just a recovered margin: releasing at speed
    hands the nominal a state it never visits in training.

  Composed with a ValueMonitor this is the Safety Value Filter; with a
  CriticMonitor, the Safety Critic Filter; with an AdversarialRolloutMonitor,
  the Gameplay Filter — the same object in all three.

  :param eps: engagement threshold on the margin.
  :param caution: upper edge of the caution band (>= eps).
  :param hysteresis: release requires margin > eps + hysteresis.
  :param rest_speed: release additionally requires speed < rest_speed.
  :param median_window: length of the median-smoothing history.
  :param dip_margin: raw margin below eps - dip_margin engages immediately,
      bypassing the median (a deep dip is not smoothing noise).
  """

  def __init__(self, num_envs: int, device: str, eps: float = 0.0,
               caution: float = 0.45, hysteresis: float = 0.15,
               rest_speed: float = 0.4, median_window: int = 5,
               dip_margin: float = 0.15, action_dim: int | None = None):
    super().__init__(num_envs, device, action_dim)
    self.eps, self.caution = eps, max(caution, eps)
    self.hys, self.rest = hysteresis, rest_speed
    self.window, self.dip = median_window, dip_margin
    self.engaged = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.v_hist: torch.Tensor | None = None

  def __call__(self, a_nom: torch.Tensor, fallback, monitor, *,
               speed: torch.Tensor, fresh: torch.Tensor,
               **ctx) -> tuple[torch.Tensor, FilterInfo]:
    """``speed``: (N,) planar base speed; ``fresh``: (N,) bool, True on the
    first step of an episode. Extra kwargs are forwarded to the monitor and
    the fallback."""
    v_raw = monitor(a_nom, fallback=fallback, **ctx)
    if self.v_hist is None:
      self.v_hist = v_raw.unsqueeze(0).repeat(self.window, 1)
    self.v_hist = torch.cat([self.v_hist[1:], v_raw.unsqueeze(0)], dim=0)
    if bool(fresh.any()):
      self.v_hist[:, fresh] = v_raw[fresh].unsqueeze(0)
      self.engaged &= ~fresh
    v_med = self.v_hist.median(dim=0).values

    engage = (v_med <= self.eps) | (v_raw <= self.eps - self.dip)
    release = (v_med > self.eps + self.hys) & (speed < self.rest)
    self.engaged = (self.engaged | engage) & ~release
    caution = (v_med <= self.caution) & ~self.engaged

    a_safe = fallback(**ctx)
    action = torch.where(self.engaged.unsqueeze(-1), a_safe, a_nom)
    return action, FilterInfo(engaged=self.engaged, value=v_raw,
                              caution=caution)

  def reset(self, done: torch.Tensor) -> None:
    self.engaged &= ~done


class OptIntervention(Intervention):
  """Minimal modification: argmin_u ||u_nom - u||^2 s.t. the monitor certifies.

  Subclasses supply the constraint and the solver; they share the contract that
  an env whose nominal already satisfies the constraint passes through
  untouched, and that an env for which no feasible action is found falls back
  to pi^<(x) (best effort).
  """

  @abstractmethod
  def __call__(self, a_nom: torch.Tensor, fallback, monitor,
               **ctx) -> tuple[torch.Tensor, FilterInfo]:
    """Return (projected action, FilterInfo)."""


class QCBFIntervention(OptIntervention):
  """Q-CBF / R-CBF projected-gradient projection through a learned Q(x, u).

  Faithful (batched) port of safe_adaptation_dev's projected-gradient R-CBF
  (script/eval_safety_filter.py::rcbf_projected_gradient, derivation in
  research_notes/rcbf_gradient_method.md). The optimization per env:

    u*(x) = argmin_u ||u_task - u||^2   s.t.   Q(x, u) >= kappa * V(x)

  with Q the learned (robust) state-action safety value (safe iff >= 0) and
  V(x) = Q(x, u_safe(x)) the safety value under the fallback's own action —
  i.e. the barrier is CLASS-K (preserve a kappa-fraction of the current safety
  level), not a fixed floor. Algorithm, per the note:

    1. tau <- kappa * V(x);  if Q(x, u_task) >= tau: return u_task
    2. if V(x) < tau (kappa > 1 with V > 0, or extreme states): return u_safe
       (best-effort — even the fallback action cannot meet the threshold)
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
    trained the same way in the two-player game, so no sign flip is actually
    involved).

  The legacy 1-D variant (line search on the blend (1-a) u_task + a u_safe) is
  dominated by this method (see the note's comparison) and is not ported;
  LeastRestrictiveIntervention covers the a in {0, 1} switching case.

  Practical notes:
  - Requires a CriticMonitor: the raw ``monitor.q_fn`` must be differentiable
    w.r.t. the action; one autograd call per ascent step, batched over envs.
  - On-policy twins (``*PPO*``) learn V(s), not Q(s, a): use this intervention
    with an off-policy twin's critic (any of {Safety,ReachAvoid}SAC{1P,2P}) or a
    distilled Q head.
  - A learned Q under optimization pressure is a certificate under attack (the
    Goodhart lesson): prefer an ensemble-LCB q_fn and validate against witness
    rollouts before trusting the numbers.

  :param kappa: barrier coefficient in [0, 1]; the constraint is
      Q(x, u) >= kappa * V(x).
  :param lr: normalized-gradient-ascent step size in action units.
  :param n_iter: max ascent steps.
  :param n_backtrack: binary-search refinements toward u_task.
  :param action_low/high: action-box bounds for the projection clip.
  """

  def __init__(self, num_envs: int, device: str, kappa: float = 0.8,
               lr: float = 0.05, n_iter: int = 10, n_backtrack: int = 10,
               action_low: float = -1.0, action_high: float = 1.0,
               action_dim: int | None = None):
    super().__init__(num_envs, device, action_dim)
    self.kappa, self.lr = kappa, lr
    self.n_iter, self.n_backtrack = n_iter, n_backtrack
    self.lo, self.hi = action_low, action_high

  def __call__(self, a_nom: torch.Tensor, fallback, monitor,
               **ctx) -> tuple[torch.Tensor, FilterInfo]:
    q_fn = getattr(monitor, "q_fn", None)
    if q_fn is None:
      raise TypeError(
        "QCBFIntervention needs a monitor exposing a differentiable q_fn "
        f"(a CriticMonitor); got {type(monitor).__name__}.")

    a_safe = fallback(**ctx)
    with torch.no_grad():
      v_hat = q_fn(action=a_safe, **ctx)               # V(x) = Q(x, u_safe)
      tau = self.kappa * v_hat
      q_task = q_fn(action=a_nom, **ctx)

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
        q = q_fn(action=u_g, **ctx)
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
        q = q_fn(action=u, **ctx)
      newly = need & ~found & (q >= tau)
      u_feas = torch.where(newly.unsqueeze(-1), u, u_feas)
      found |= newly

      # step 4: backtrack — binary search [u_task, u_feas] per found env
      if bool(found.any()):
        u_lo, u_hi = a_nom.clone(), u_feas.clone()
        for _ in range(self.n_backtrack):
          u_mid = 0.5 * (u_lo + u_hi)
          with torch.no_grad():
            q_mid = q_fn(action=u_mid, **ctx)
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
      q_final = q_fn(action=action, **ctx)

    intervened = ~pass_through          # stateless (no latch)
    dev = torch.norm(action - a_nom, dim=-1)
    return action, FilterInfo(engaged=intervened, value=q_final,
                              intervention=dev)
