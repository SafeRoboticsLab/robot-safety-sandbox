"""Unit tests for the three swappable filter modules and the composition."""

from __future__ import annotations

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from filter_fixtures import ACT_DIM, NUM_ENVS, Fixture  # noqa: E402
from robot_safety_sandbox.filters import (  # noqa: E402
  AdversarialRolloutMonitor, CriticMonitor, EngagementLog, FilterInfo,
  LeastRestrictiveIntervention, PolicyFallback, QCBFIntervention, RolloutMonitor,
  SafetyFilter, ValueMonitor, ZeroFallback, gameplay_filter, qcbf_filter,
  safety_critic_filter, safety_value_filter)

DEV = "cpu"


# --- fallback ----------------------------------------------------------------

def test_zero_fallback_shape_and_value():
  fb = ZeroFallback(NUM_ENVS, DEV, ACT_DIM)
  a = fb()
  assert a.shape == (NUM_ENVS, ACT_DIM)
  assert torch.all(a == 0)


def test_policy_fallback_forwards_ctx():
  fx = Fixture()
  fb = PolicyFallback(NUM_ENVS, DEV, fx.fallback_fn, ACT_DIM)
  assert torch.equal(fb(s_obs=fx.obs[0]), fx.fallback_fn(s_obs=fx.obs[0]))


# --- monitors ----------------------------------------------------------------

def test_value_monitor_ignores_action():
  fx = Fixture()
  m = ValueMonitor(NUM_ENVS, DEV, fx.value_fn, ACT_DIM)
  v0 = m(fx.a_nom[0], s_obs=fx.obs[0])
  v1 = m(-fx.a_nom[0], s_obs=fx.obs[0])
  assert v0.shape == (NUM_ENVS,)
  assert torch.equal(v0, v1)


def test_critic_monitor_uses_action_and_exposes_grad_fn():
  fx = Fixture()
  m = CriticMonitor(NUM_ENVS, DEV, fx.q_fn, ACT_DIM)
  q0 = m(fx.a_nom[0], s_obs=fx.obs[0])
  assert q0.shape == (NUM_ENVS,)
  assert not torch.equal(q0, m(-fx.a_nom[0], s_obs=fx.obs[0]))
  assert not q0.requires_grad          # __call__ is no-grad
  u = fx.a_nom[0].clone().requires_grad_(True)
  g = torch.autograd.grad(m.q_fn(action=u, s_obs=fx.obs[0]).sum(), u)[0]
  assert g.shape == u.shape and torch.isfinite(g).all()


@pytest.mark.parametrize("cls", [RolloutMonitor, AdversarialRolloutMonitor])
def test_rollout_monitors_are_gated_on_shadow_sim(cls):
  with pytest.raises(NotImplementedError, match="shadow-simulation"):
    cls(NUM_ENVS, DEV)


def test_gameplay_filter_builder_raises_on_the_monitor():
  fx = Fixture()
  with pytest.raises(NotImplementedError, match="shadow-simulation"):
    gameplay_filter(NUM_ENVS, DEV, fx.fallback_fn, action_dim=ACT_DIM)


# --- interventions -----------------------------------------------------------

def _ctx(fx, t=0):
  return dict(speed=fx.speed[t], fresh=fx.fresh[t], s_obs=fx.obs[t])


def test_least_restrictive_switches_hard():
  """Engaged envs get exactly pi^<; released envs get exactly the nominal."""
  fx = Fixture()
  filt = safety_value_filter(NUM_ENVS, DEV, fx.value_fn, fx.fallback_fn)
  a, info = filt(fx.a_nom[0], **_ctx(fx))
  a_safe = fx.fallback_fn(s_obs=fx.obs[0])
  eng = info.engaged
  assert eng.any() and (~eng).any(), "fixture must exercise both branches"
  assert torch.equal(a[eng], a_safe[eng])
  assert torch.equal(a[~eng], fx.a_nom[0][~eng])


def test_latch_and_history_clear_on_reset():
  """The 77%-livelock contract: reset(done) drops the latch."""
  fx = Fixture()
  filt = safety_value_filter(NUM_ENVS, DEV, fx.value_fn, fx.fallback_fn)
  for t in range(20):
    filt(fx.a_nom[t], **_ctx(fx, t))
  assert filt.intervention.engaged.any()
  filt.reset(torch.ones(NUM_ENVS, dtype=torch.bool))
  assert not filt.intervention.engaged.any()


def test_engagement_log_counts_and_resets():
  fx = Fixture()
  filt = safety_value_filter(NUM_ENVS, DEV, fx.value_fn, fx.fallback_fn)
  for t in range(10):
    filt(fx.a_nom[t], **_ctx(fx, t))
  log = filt.telemetry
  assert 0.0 < log.intervention_rate(10) <= 1.0
  assert log.engaged_steps.sum() > 0
  log.reset()
  assert log.engaged_steps.sum() == 0 and log.caution_steps.sum() == 0


def test_qcbf_passes_through_already_safe_actions():
  fx = Fixture()
  filt = qcbf_filter(NUM_ENVS, DEV, fx.q_fn, fx.fallback_fn)
  a, info = filt(fx.a_nom[0], s_obs=fx.obs[0])
  ok = ~info.engaged
  assert ok.any(), "fixture must exercise the pass-through branch"
  assert torch.equal(a[ok], fx.a_nom[0][ok])
  assert torch.allclose(info.intervention[ok], torch.zeros(int(ok.sum())))
  assert isinstance(filt.intervention, QCBFIntervention)


def test_qcbf_requires_a_critic_monitor():
  fx = Fixture()
  bad = SafetyFilter(PolicyFallback(NUM_ENVS, DEV, fx.fallback_fn, ACT_DIM),
                     ValueMonitor(NUM_ENVS, DEV, fx.value_fn, ACT_DIM),
                     QCBFIntervention(NUM_ENVS, DEV, action_dim=ACT_DIM))
  with pytest.raises(TypeError, match="q_fn"):
    bad(fx.a_nom[0], s_obs=fx.obs[0])


# --- composition -------------------------------------------------------------

def test_safety_critic_filter_composes():
  """The new-for-free composition: switching intervention on a Q monitor."""
  fx = Fixture()
  filt = safety_critic_filter(NUM_ENVS, DEV, fx.q_fn, fx.fallback_fn)
  assert isinstance(filt.monitor, CriticMonitor)
  assert isinstance(filt.intervention, LeastRestrictiveIntervention)
  a, info = filt(fx.a_nom[0], **_ctx(fx))
  assert a.shape == (NUM_ENVS, ACT_DIM)
  assert isinstance(info, FilterInfo) and info.value.shape == (NUM_ENVS,)
  # the Q monitor really scores the ACTION (a value monitor could not)
  a2, info2 = filt(-fx.a_nom[0], **_ctx(fx))
  assert not torch.equal(info.value, info2.value)


@pytest.mark.parametrize("attr,bad", [("num_envs", 3), ("action_dim", 99)])
def test_consistency_check_rejects_mismatched_modules(attr, bad):
  fx = Fixture()
  fb = PolicyFallback(NUM_ENVS, DEV, fx.fallback_fn, ACT_DIM)
  mon = ValueMonitor(NUM_ENVS, DEV, fx.value_fn, ACT_DIM)
  itv = LeastRestrictiveIntervention(NUM_ENVS, DEV, action_dim=ACT_DIM)
  setattr(mon, attr, bad)
  with pytest.raises(ValueError, match=attr):
    SafetyFilter(fb, mon, itv)


def test_consistency_check_rejects_wrong_module_type():
  fx = Fixture()
  fb = PolicyFallback(NUM_ENVS, DEV, fx.fallback_fn, ACT_DIM)
  itv = LeastRestrictiveIntervention(NUM_ENVS, DEV, action_dim=ACT_DIM)
  with pytest.raises(TypeError, match="monitor"):
    SafetyFilter(fb, fb, itv)


def test_custom_telemetry_object_is_used():
  fx = Fixture()
  log = EngagementLog(NUM_ENVS, DEV)
  filt = safety_value_filter(NUM_ENVS, DEV, fx.value_fn, fx.fallback_fn,
                             telemetry=log)
  filt(fx.a_nom[0], **_ctx(fx))
  assert filt.telemetry is log and log.engaged_steps.sum() > 0
