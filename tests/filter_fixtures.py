"""Deterministic, simulator-free fixtures for filter characterization tests.

Everything here is seeded and CPU-only so the same (action, FilterInfo) trace
is reproducible across machines: a synthetic observation stream stands in for
the env, and small fixed-weight MLPs stand in for the safety twin's value head,
critic and fallback actor.

Used by ``characterize_filters.py`` (writes the pre-refactor snapshot) and by
``test_filter_equivalence.py`` (replays it through the composed filters).

``Fixture``'s RNG draw order is FROZEN: ``tests/fixtures/filter_characterization
.pt`` is a bitwise anchor against it. Add new fixtures below it, never inside
it.
"""

from __future__ import annotations

import torch

from robot_safety_sandbox.filters.rollout import RolloutStep, ShadowSim

NUM_ENVS = 16
ACT_DIM = 6
OBS_DIM = 12
STEPS = 300
SEED = 20260724


def _mlp(gen: torch.Generator, sizes: list[int]) -> torch.nn.Sequential:
  layers: list[torch.nn.Module] = []
  for i in range(len(sizes) - 1):
    lin = torch.nn.Linear(sizes[i], sizes[i + 1])
    with torch.no_grad():
      lin.weight.copy_(torch.randn(sizes[i + 1], sizes[i], generator=gen) * 0.5)
      lin.bias.copy_(torch.randn(sizes[i + 1], generator=gen) * 0.1)
    layers.append(lin)
    if i < len(sizes) - 2:
      layers.append(torch.nn.Tanh())
  net = torch.nn.Sequential(*layers)
  net.eval()
  return net


class Fixture:
  """A frozen synthetic safety twin + the observation stream it sees."""

  def __init__(self, device: str = "cpu"):
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    self.device = device
    self.num_envs, self.act_dim = NUM_ENVS, ACT_DIM

    self.v_net = _mlp(gen, [OBS_DIM, 32, 1]).to(device)
    self.q_net = _mlp(gen, [OBS_DIM + ACT_DIM, 32, 1]).to(device)
    self.pi_net = _mlp(gen, [OBS_DIM, 32, ACT_DIM]).to(device)

    # observation stream: a slow random walk, so V(s) wanders across the
    # engage/release thresholds instead of sitting on one side of them.
    obs = torch.zeros(STEPS, NUM_ENVS, OBS_DIM)
    noise = torch.randn(STEPS, NUM_ENVS, OBS_DIM, generator=gen) * 0.35
    x = torch.randn(NUM_ENVS, OBS_DIM, generator=gen)
    for t in range(STEPS):
      x = 0.9 * x + noise[t]
      obs[t] = x
    self.obs = obs.to(device)
    self.a_nom = (torch.randn(STEPS, NUM_ENVS, ACT_DIM, generator=gen)
                  .clamp(-1, 1).to(device))
    self.speed = (torch.rand(STEPS, NUM_ENVS, generator=gen) * 1.2).to(device)
    # episode boundaries: every env resets on t=0, then on a fixed schedule
    fresh = torch.zeros(STEPS, NUM_ENVS, dtype=torch.bool)
    fresh[0] = True
    for i in range(NUM_ENVS):
      fresh[:, i][(37 + 5 * i)::(60 + i)] = True
    self.fresh = fresh.to(device)

  # -- the callables a filter is wired with -----------------------------------
  def value_fn(self, s_obs, **_):
    with torch.no_grad():
      return self.v_net(s_obs).squeeze(-1)

  def fallback_fn(self, s_obs, **_):
    with torch.no_grad():
      return self.pi_net(s_obs).clamp(-1.0, 1.0)

  def q_fn(self, action, s_obs, **_):
    """Differentiable w.r.t. ``action`` (QCBF needs the gradient)."""
    return self.q_net(torch.cat([s_obs, action], dim=-1)).squeeze(-1)


class ToyShadowSim(ShadowSim):
  """A 1-D shadow sim for testing rollout-monitor SEMANTICS without a simulator.

  A point on a line. The action's first component is a velocity command, the
  safety margin IS the position (safe iff x >= 0, i.e. the failure set is the
  negative half-line), the target set sits at x = 1, and falling below -0.5 ends
  the episode. Everything a rollout monitor has to get right — horizon min,
  min over parallel rollouts, post-termination masking, recertification latching
  — is visible here in closed form.

  :param state_fn: callable() -> (N,) the LIVE x. Called by every ``seed()``, so
      a test can move the live state between control steps (and, crucially, NOT
      move it, which is what makes seeding idempotent).
  :param drift: (R,) per-rollout constant added to the commanded velocity — the
      only reason R > 1 rollouts of a deterministic sim would differ. Stands in
      for a sampled adversary / stochastic fallback.
  """

  def __init__(self, num_envs: int, state_fn, *, rollouts_per_env: int = 1,
               dt: float = 0.1, drift: torch.Tensor | None = None,
               device: str = "cpu"):
    self.n_live = int(num_envs)
    self.rollouts_per_env = int(rollouts_per_env)
    self.num_envs = self.n_live * self.rollouts_per_env
    self.device = device
    self.state_fn, self.dt = state_fn, float(dt)
    self.drift = None if drift is None else drift.to(device)
    self.x = torch.zeros(self.num_envs, device=device)
    self.seeds = 0                       # rollouts run (recertification count)
    self.steps = 0

  def seed(self) -> None:
    self.x = self.state_fn().to(self.device).repeat_interleave(
      self.rollouts_per_env).clone()
    self.seeds += 1

  def step(self, action: torch.Tensor) -> RolloutStep:
    push = action[:, 0]
    if self.drift is not None:
      push = push + self.drift.repeat(self.n_live)
    self.x = self.x + self.dt * push
    self.steps += 1
    g = self.x.clone()
    l = 1.0 - (self.x - 1.0).abs()
    obs = self.x.unsqueeze(-1).expand(self.num_envs, OBS_DIM).contiguous()
    return RolloutStep(ctx={"s_obs": obs}, g=g, l=l, done=g < -0.5)


def constant_fallback(value: float, act_dim: int = ACT_DIM):
  """A scripted pi^<: always command ``value`` on every action dimension."""
  def fn(s_obs, **_):
    return torch.full((s_obs.shape[0], act_dim), value, device=s_obs.device)
  return fn


def drive(step_fn, fixture: Fixture, steps: int = STEPS) -> dict:
  """Run ``step_fn(a_nom, speed=..., fresh=..., s_obs=...) -> (action, info)``.

  Returns stacked traces of everything a FilterInfo can carry.
  """
  acts, engaged, values, cautions, interventions = [], [], [], [], []
  for t in range(steps):
    action, info = step_fn(fixture.a_nom[t], speed=fixture.speed[t],
                           fresh=fixture.fresh[t], s_obs=fixture.obs[t])
    acts.append(action.detach().clone())
    engaged.append(info.engaged.detach().clone())
    values.append(info.value.detach().clone())
    cautions.append(None if info.caution is None
                    else info.caution.detach().clone())
    interventions.append(None if info.intervention is None
                         else info.intervention.detach().clone())
  out = {"action": torch.stack(acts),
         "engaged": torch.stack(engaged),
         "value": torch.stack(values)}
  if cautions[0] is not None:
    out["caution"] = torch.stack(cautions)
  if interventions[0] is not None:
    out["intervention"] = torch.stack(interventions)
  return out
