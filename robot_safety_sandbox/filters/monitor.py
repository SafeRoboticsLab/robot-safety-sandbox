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
  RolloutMonitor  simulate: one step of the proposed action, then H-1 steps of
                  the fallback, in a shadow sim seeded from the live state, and
                  reduce the (g, l) trace to one margin. The adversarial variant
                  plays that future against a learned disturbance policy — the
                  GAMEPLAY filter. Costs H sim steps per control step; see
                  ``rollout.py`` for the shadow sim and its state round-trip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .rollout import ShadowSim


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
  """Certify u by an imagined H-step fallback rollout in a shadow sim.

  Instead of trusting a learned scalar (V or Q), SIMULATE. Per control step,
  in a :class:`~.rollout.ShadowSim` seeded from the live state:

    1. seed every shadow env from the live env (all robots, exact state);
    2. take ONE step with the proposed action ``u``;
    3. take H-1 steps under the fallback pi^< (the learned safety policy;
       :class:`AdversarialRolloutMonitor` plays it against a learned adversary);
    4. reduce the (g, l) trace to one scalar per env.

  So ``u`` is certified only if the fallback can carry the system safely from
  wherever ``u`` leaves it -- the rollout analogue of the value monitors, with
  the same "safe iff >= 0" convention.

  **The reduction** reuses the zoo's margin semantics rather than inventing a
  new one. Over the horizon, with the running ``G_t = min_{s<=t} g_s``:

    avoid        Delta = min_t g_t                       = G_H
    reach-avoid  Delta = max_t min(l_t, min_{s<=t} g_s)   (``reach_avoid=True``)

  which are exactly the avoid and reach-avoid outcome functionals the safety_sb3
  backups value; the rollout is their Monte-Carlo evaluation under (pi^<, pi_d)
  rather than their dynamic-programming fixed point. Steps after an env's
  episode ends are EXCLUDED (mjlab auto-resets in place, so a post-done margin
  belongs to a different episode); the failure itself is not excluded, because
  the bridge already anchors g to the terminal failure value on the done step.

  **Parallel rollouts, min semantics.** The classical filter has one rollout;
  mjlab lets us run R of them per live env (shadow env ``j`` mirrors live
  ``j // R``). The verdict is the **MIN over the R rollouts** -- one failing
  rollout condemns the action. With R = 1 (the default, and the right thing to
  get correct first) this reduces to the classical form; R > 1 is only
  informative when something makes the rollouts differ (a stochastic fallback,
  a sampled adversary, per-rollout disturbance draws).

  **Cost and amortization.** H shadow sim steps per control step. Set
  ``recertify_every=k`` to run the rollout every k steps and LATCH the verdict
  in between. The latch is safety-relevant state, so it obeys the same reset
  contract as the switch: :meth:`reset` marks the finished envs due, and a due
  env forces the next call to re-certify (the rollout is batched, so this
  refreshes every env at once). Note that with many envs on staggered episode
  boundaries something resets almost every step, which collapses the
  amortization back to k=1 -- ``recertify_every`` earns its keep in the
  few-env deployment setting, or with synchronized episodes.

  :param num_envs: N, the LIVE env count (the filter's batch).
  :param device: torch device.
  :param shadow: a :class:`~.rollout.ShadowSim` with ``N * R`` envs.
  :param horizon: H, total steps per rollout INCLUDING the nominal-action step.
  :param reach_avoid: use the reach-avoid reduction instead of avoid.
  :param recertify_every: k; 1 (default) re-certifies every control step.
  """

  def __init__(self, num_envs: int, device: str, shadow: ShadowSim,
               horizon: int, *, reach_avoid: bool = False,
               recertify_every: int = 1, action_dim: int | None = None):
    super().__init__(num_envs, device, action_dim)
    if int(horizon) < 1:
      raise ValueError(f"horizon must be >= 1, got {horizon}")
    self.shadow, self.horizon = shadow, int(horizon)
    self.reach_avoid = bool(reach_avoid)
    self.recertify_every = max(1, int(recertify_every))
    self.rollouts_per_env = int(getattr(shadow, "rollouts_per_env", 1))
    if shadow.num_envs != num_envs * self.rollouts_per_env:
      raise ValueError(
        f"shadow sim has {shadow.num_envs} envs; a monitor over {num_envs} "
        f"envs with rollouts_per_env={self.rollouts_per_env} needs "
        f"{num_envs * self.rollouts_per_env}")
    self._latched: torch.Tensor | None = None
    self._age = 0
    self._due = torch.ones(num_envs, dtype=torch.bool, device=device)

  # -- the game: what drives the shadow sim on one rollout step ---------------

  def _play(self, a_ctrl: torch.Tensor, ctx: dict) -> torch.Tensor:
    """Fallback-only rollout: the control action IS the whole action."""
    del ctx
    return a_ctrl

  # -- the rollout ------------------------------------------------------------

  def _rollout(self, a_nom: torch.Tensor, fallback, **ctx) -> torch.Tensor:
    r, n = self.rollouts_per_env, self.shadow.num_envs
    self.shadow.seed()
    # Step 1 has no shadow observation yet (obs come out of a step), but it does
    # not need one: the seeded shadow state IS the live state, so the live ctx
    # is the shadow's t=0 ctx. Repeated per rollout, it is what an adversary
    # acting on the nominal step sees.
    ctx0 = {k: (v.repeat_interleave(r, dim=0)
                if torch.is_tensor(v) and v.shape[:1] == (self.num_envs,) else v)
            for k, v in ctx.items()}
    step = self.shadow.step(self._play(a_nom.repeat_interleave(r, dim=0), ctx0))

    alive = torch.ones(n, dtype=torch.bool, device=step.g.device)
    g_run = torch.full_like(step.g, float("inf"))
    ra_run = torch.full_like(step.g, float("-inf"))
    for t in range(self.horizon):
      if t:
        step = self.shadow.step(
          self._play(fallback(**step.ctx), step.ctx))
      g_run = torch.minimum(
        g_run, torch.where(alive, step.g, torch.full_like(step.g, float("inf"))))
      if self.reach_avoid:
        ra_t = torch.minimum(step.l, g_run)
        ra_run = torch.maximum(
          ra_run,
          torch.where(alive, ra_t, torch.full_like(ra_t, float("-inf"))))
      alive = alive & ~step.done
      if not bool(alive.any()):
        break
    verdict = ra_run if self.reach_avoid else g_run
    return verdict.view(self.num_envs, r).amin(dim=1)

  def __call__(self, a: torch.Tensor, fallback=None, **ctx) -> torch.Tensor:
    if fallback is None:
      raise ValueError(
        f"{type(self).__name__} needs the composition's fallback to simulate "
        "pi^< over the horizon; it is passed automatically by the "
        "intervention, so a direct call must pass fallback= itself.")
    if self._latched is None or self._age <= 0 or bool(self._due.any()):
      self._latched = self._rollout(a, fallback, **ctx)
      self._age = self.recertify_every
      self._due.zero_()
    self._age -= 1
    return self._latched

  def reset(self, done: torch.Tensor) -> None:
    """A finished env's latched verdict is stale -> force a re-certification."""
    self._due |= done


class AdversarialRolloutMonitor(RolloutMonitor):
  """Rollout monitor whose imagined future is played against an adversary.

  The GAMEPLAY variant, and the reason the composition is called a gameplay
  filter: the H-step future is not a nominal-dynamics guess but a game, with the
  learned disturbance policy driving the dstb sub-space against the fallback.
  The returned margin is therefore a worst-case-over-the-disturbance-class
  estimate rather than an expectation under undisturbed dynamics.

  The shadow sim must be built with the concatenated ``[a_ctrl, a_dstb]`` action
  space (``MjlabShadowSim.from_task(..., adversary=True)``), which is the same
  action layout the two-player learners train under -- so the dstb actor of a
  ``*SAC2P`` / ``*PPO2P`` twin plugs straight in.

  The adversary acts on EVERY step of the rollout, the nominal step included: a
  disturbance the filter is supposed to be robust to does not politely wait one
  step. On the nominal step it sees the live context (the seeded shadow state is
  the live state), thereafter the shadow's own observations.

  :param adversary_fn: callable(**ctx) -> (n, dstb_dim), batched over the SHADOW
      env count. Typically the twin's ``policy.dstb_actor``.
  """

  def __init__(self, num_envs: int, device: str, shadow: ShadowSim,
               horizon: int, adversary_fn, *, reach_avoid: bool = False,
               recertify_every: int = 1, action_dim: int | None = None):
    super().__init__(num_envs, device, shadow, horizon,
                     reach_avoid=reach_avoid, recertify_every=recertify_every,
                     action_dim=action_dim)
    self.adversary_fn = adversary_fn

  def _play(self, a_ctrl: torch.Tensor, ctx: dict) -> torch.Tensor:
    return torch.cat([a_ctrl, self.adversary_fn(**ctx)], dim=-1)
