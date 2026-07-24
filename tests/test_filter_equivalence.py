"""The refactor's correctness anchor: composed filters == the old monoliths.

Replays the exact synthetic stream from ``characterize_filters.py`` through the
new ``SafetyFilter`` compositions and asserts the traces match the snapshot
recorded from the pre-refactor ``ValueShield`` / ``QCBFFilter`` (bitwise: both
are deterministic CPU float32 with an identical op order).
"""

from __future__ import annotations

import os
import sys
import warnings

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from filter_fixtures import Fixture, drive  # noqa: E402
from robot_safety_sandbox.filters import (  # noqa: E402
  QCBFFilter, ValueShield, qcbf_filter, safety_value_filter)

SNAPSHOT = os.path.join(_HERE, "fixtures", "filter_characterization.pt")


@pytest.fixture(scope="module")
def snapshot():
  if not os.path.exists(SNAPSHOT):
    pytest.skip(f"missing characterization snapshot: {SNAPSHOT} "
                "(regenerate with tests/characterize_filters.py)")
  return torch.load(SNAPSHOT, weights_only=True)


def _assert_same(new: dict, ref: dict, keys) -> None:
  for k in keys:
    assert k in new, f"trace lost key {k}"
    a, b = new[k], ref[k]
    assert a.shape == b.shape, f"{k}: shape {a.shape} != {b.shape}"
    if a.dtype == torch.bool:
      assert torch.equal(a, b), f"{k}: {(a != b).sum().item()} steps differ"
    else:
      assert torch.equal(a, b), (
        f"{k}: max |diff| = {(a - b).abs().max().item():.3e}")


def test_value_filter_matches_legacy_snapshot(snapshot):
  """SafetyFilter(PolicyFallback, ValueMonitor, LeastRestrictive) == ValueShield."""
  fx = Fixture()
  filt = safety_value_filter(fx.num_envs, fx.device, fx.value_fn,
                             fx.fallback_fn)
  trace = drive(filt, fx)
  _assert_same(trace, snapshot["value_filter"],
               ["action", "engaged", "value", "caution"])
  assert torch.equal(filt.telemetry.engaged_steps,
                     snapshot["value_filter"]["engaged_steps"])
  assert torch.equal(filt.telemetry.caution_steps,
                     snapshot["value_filter"]["caution_steps"])


def test_qcbf_filter_matches_legacy_snapshot(snapshot):
  """SafetyFilter(PolicyFallback, CriticMonitor, QCBFIntervention) == QCBFFilter."""
  fx = Fixture()
  filt = qcbf_filter(fx.num_envs, fx.device, fx.q_fn, fx.fallback_fn)
  trace = drive(filt, fx)
  _assert_same(trace, snapshot["qcbf"],
               ["action", "engaged", "value", "intervention"])
  assert torch.equal(filt.telemetry.engaged_steps,
                     snapshot["qcbf"]["engaged_steps"])


@pytest.mark.parametrize("alias,builder,fn", [
  (ValueShield, safety_value_filter, "value_fn"),
  (QCBFFilter, qcbf_filter, "q_fn"),
])
def test_deprecated_aliases_warn_and_compose(alias, builder, fn):
  fx = Fixture()
  with pytest.warns(DeprecationWarning):
    old = alias(fx.num_envs, fx.device, getattr(fx, fn), fx.fallback_fn)
  with warnings.catch_warnings():
    warnings.simplefilter("error")   # the builder itself must not warn
    new = builder(fx.num_envs, fx.device, getattr(fx, fn), fx.fallback_fn)
  assert type(old) is type(new)
  assert type(old.monitor) is type(new.monitor)
  assert type(old.intervention) is type(new.intervention)
