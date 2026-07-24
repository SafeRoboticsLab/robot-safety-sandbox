"""Record the reference (action, FilterInfo) traces of the monolithic filters.

Run this BEFORE the filter refactor, on the pre-refactor code:

    python tests/characterize_filters.py

It drives the legacy ``ValueShield`` and ``QCBFFilter`` over a fixed synthetic
episode stream (tests/filter_fixtures.py) and writes the traces to
``tests/data/filter_characterization.pt``. ``test_filter_equivalence.py`` then
replays the same stream through the composed ``SafetyFilter`` and asserts the
traces match, which is what makes the refactor behavior-preserving rather than
merely plausible.

The legacy classes are also reachable after the refactor through the deprecated
aliases, so re-running this script post-refactor regenerates an identical file.
"""

from __future__ import annotations

import os
import sys
import warnings

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from filter_fixtures import Fixture, drive  # noqa: E402

OUT = os.path.join(_HERE, "fixtures", "filter_characterization.pt")


def main() -> None:
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from robot_safety_sandbox.filters import QCBFFilter, ValueShield

  torch.manual_seed(0)
  fx = Fixture()

  vs = ValueShield(fx.num_envs, fx.device, fx.value_fn, fx.fallback_fn)
  value_trace = drive(vs.act, fx)
  value_trace["engaged_steps"] = vs.engaged_steps.clone()
  value_trace["caution_steps"] = vs.caution_steps.clone()

  fx2 = Fixture()
  qc = QCBFFilter(fx2.num_envs, fx2.device, fx2.q_fn, fx2.fallback_fn)
  qcbf_trace = drive(qc.act, fx2)
  qcbf_trace["engaged_steps"] = qc.engaged_steps.clone()

  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  torch.save({"value_shield": value_trace, "qcbf": qcbf_trace}, OUT)
  print(f"wrote {OUT}")
  print(f"  value: engaged {value_trace['engaged'].float().mean():.4f} "
        f"caution {value_trace['caution'].float().mean():.4f}")
  print(f"  qcbf:  engaged {qcbf_trace['engaged'].float().mean():.4f} "
        f"mean |du| {qcbf_trace['intervention'].mean():.4f}")


if __name__ == "__main__":
  main()
