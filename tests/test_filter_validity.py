"""Def. 2 validity, as a property any composition can be run through.

Hsu, Hu & Fisac, "The Safety Filter: A Unified View" (arXiv:2309.05837),
Def. 2 — a filter phi is VALID w.r.t. its monitor Delta and fallback pi^< iff

    Delta(x, pi^<(x)) >= 0   =>   Delta(x, phi(x, u)) >= 0   for all u.

I.e. wherever the fallback is itself certified, no nominal action can talk the
filter into an uncertified output. ``assert_valid`` below takes any composition
and a context stream and checks exactly that, over adversarially varied nominal
actions.

One caveat worth stating loudly. The DEPLOYED LeastRestrictiveIntervention is
deliberately NOT Def-2 valid instant-by-instant: median smoothing, release
hysteresis and the rest-speed gate trade pointwise strictness for chatter
robustness (a single-step V dip at a contact event must not trigger a handover;
a release at speed must not hand back a state the nominal never trained on).
Those are field-validated heuristics, not bugs. The validity property is
therefore checked in the STRICT configuration (window=1, no hysteresis, no rest
gate), which is the un-smoothed switch the theory describes; the difference
between the two is precisely the safety margin the smoothing spends.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from filter_fixtures import (  # noqa: E402
  ACT_DIM, NUM_ENVS, STEPS, Fixture, ToyShadowSim)
from robot_safety_sandbox.filters import (  # noqa: E402
  gameplay_filter, qcbf_filter, rollout_filter, safety_critic_filter,
  safety_value_filter)

DEV = "cpu"
STRICT = dict(median_window=1, hysteresis=0.0, rest_speed=float("inf"),
              caution=0.0)


def assert_valid(filt, fx: Fixture, ctx_keys=("s_obs",), steps: int = 60,
                 tol: float = 0.0, on_step=None) -> int:
  """Check Def. 2 over ``steps`` of the fixture stream; return #states tested.

  For each step: where Delta(x, pi^<(x)) >= 0, require that the filtered action
  also satisfies Delta(x, phi(x, u)) >= -tol, for several nominal actions u
  (the recorded nominal, its negation, and the action-box corners).

  ``on_step(t)`` runs before each step's checks — for monitors that read a
  simulator rather than only the ctx (the rollout monitors), it is how the test
  advances the shadow sim's LIVE state in lockstep with the fixture stream.
  """
  checked = 0
  for t in range(steps):
    if on_step is not None:
      on_step(t)
    ctx = {k: getattr(fx, {"s_obs": "obs"}[k])[t] for k in ctx_keys}
    a_fb = filt.fallback(**ctx)
    delta_fb = filt.monitor(a_fb, fallback=filt.fallback, **ctx)
    premise = delta_fb >= 0
    if not bool(premise.any()):
      continue
    candidates = [fx.a_nom[t], -fx.a_nom[t],
                  torch.ones(NUM_ENVS, ACT_DIM), -torch.ones(NUM_ENVS, ACT_DIM)]
    for u in candidates:
      action, _ = filt(u, speed=fx.speed[t], fresh=fx.fresh[t], **ctx)
      delta = filt.monitor(action, fallback=filt.fallback, **ctx)
      bad = premise & (delta < -tol)
      assert not bool(bad.any()), (
        f"Def-2 violated at t={t} in {int(bad.sum())} env(s): "
        f"min Delta(x, phi) = {delta[bad].min().item():.4e} while "
        f"Delta(x, pi^<) >= 0")
      checked += int(premise.sum())
  return checked


def test_value_filter_is_valid():
  """Trivially valid: a state monitor cannot be moved by the action."""
  fx = Fixture()
  filt = safety_value_filter(NUM_ENVS, DEV, fx.value_fn, fx.fallback_fn,
                             **STRICT)
  assert assert_valid(filt, fx) > 0


def test_critic_filter_is_valid_in_strict_config():
  """The interesting case: Q moves with u, and the hard switch must catch it."""
  fx = Fixture()
  filt = safety_critic_filter(NUM_ENVS, DEV, fx.q_fn, fx.fallback_fn, **STRICT)
  assert assert_valid(filt, fx) > 0


def _rollout_stream(fx: Fixture):
  """A live-state holder driven by the fixture obs, plus the sim reading it.

  The rollout monitors read a simulator, not the ctx, so validity is only
  meaningful if the simulated state tracks the state the ctx describes. Here
  the toy sim's live x IS the fixture's first obs channel.
  """
  holder = {"x": fx.obs[0][:, 0] * 0.3}
  sim = ToyShadowSim(NUM_ENVS, lambda: holder["x"], dt=0.05)
  return holder, sim, (lambda t: holder.update(x=fx.obs[t][:, 0] * 0.3))


def test_rollout_filter_is_valid_in_strict_config():
  """The gameplay composition, run through the SAME Def-2 harness as the value
  and critic filters — the monitor is the only thing that changed."""
  fx = Fixture()
  holder, sim, on_step = _rollout_stream(fx)
  filt = rollout_filter(NUM_ENVS, DEV, fx.fallback_fn, sim, 6, **STRICT)
  assert assert_valid(filt, fx, steps=40, on_step=on_step) > 0


def test_adversarial_rollout_filter_is_valid_in_strict_config():
  fx = Fixture()
  holder, sim, on_step = _rollout_stream(fx)

  class _WithDstb(ToyShadowSim):
    """ctrl and dstb both push; the sim reads their sum on channel 0."""

    def step(self, action):
      a = action.clone()
      a[:, 0] = action[:, 0] + action[:, -1]
      return super().step(a)

  sim = _WithDstb(NUM_ENVS, lambda: holder["x"], dt=0.05)
  filt = gameplay_filter(
    NUM_ENVS, DEV, fx.fallback_fn, sim, 6,
    lambda s_obs, **_: -0.3 * torch.ones(s_obs.shape[0], 1), **STRICT)
  assert assert_valid(filt, fx, steps=40, on_step=on_step) > 0


@pytest.mark.parametrize("kappa", [0.0, 0.5, 0.8, 1.0])
def test_qcbf_filter_is_valid_for_kappa_in_unit_interval(kappa):
  """Q(x, u) >= kappa * V(x) with V >= 0 and kappa in [0, 1] implies Q >= 0."""
  fx = Fixture()
  filt = qcbf_filter(NUM_ENVS, DEV, fx.q_fn, fx.fallback_fn, kappa=kappa)
  # QCBFIntervention takes no speed/fresh; drive it through the same checker
  # by giving it a tolerant context.
  checked = 0
  for t in range(60):
    ctx = dict(s_obs=fx.obs[t])
    a_fb = filt.fallback(**ctx)
    premise = filt.monitor(a_fb, **ctx) >= 0
    if not bool(premise.any()):
      continue
    for u in (fx.a_nom[t], -fx.a_nom[t], torch.ones(NUM_ENVS, ACT_DIM)):
      action, _ = filt(u, **ctx)
      delta = filt.monitor(action, **ctx)
      bad = premise & (delta < 0)
      assert not bool(bad.any()), (
        f"Def-2 violated at t={t}, kappa={kappa}: "
        f"min Q = {delta[bad].min().item():.4e}")
    checked += int(premise.sum())
  assert checked > 0


def test_smoothed_config_is_documented_as_weaker():
  """The deployed (smoothed/hysteretic) switch may transiently violate Def. 2.

  Not a bug — the smoothing is what buys chatter robustness. This test pins the
  trade-off so it is a recorded property rather than a surprise: it asserts the
  STRICT config is valid on a stream where the DEPLOYED default is not.
  """
  fx = Fixture()
  strict = safety_critic_filter(NUM_ENVS, DEV, fx.q_fn, fx.fallback_fn,
                                **STRICT)
  assert assert_valid(strict, fx, steps=STEPS // 2) > 0

  loose = safety_critic_filter(NUM_ENVS, DEV, fx.q_fn, fx.fallback_fn)
  violations = 0
  for t in range(STEPS // 2):
    ctx = dict(s_obs=fx.obs[t])
    premise = loose.monitor(loose.fallback(**ctx), **ctx) >= 0
    action, _ = loose(fx.a_nom[t], speed=fx.speed[t], fresh=fx.fresh[t], **ctx)
    violations += int((premise & (loose.monitor(action, **ctx) < 0)).sum())
  assert violations > 0, ("expected the smoothed switch to be transiently "
                          "invalid on this stream; if this ever passes with 0, "
                          "the trade-off note above needs revisiting")
