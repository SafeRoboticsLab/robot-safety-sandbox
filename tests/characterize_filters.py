"""(Re)record the reference (action, FilterInfo) traces of the safety filters.

    python tests/characterize_filters.py

It drives the value filter and the Q-CBF filter over a fixed synthetic episode
stream (tests/filter_fixtures.py) and writes the traces to
``tests/fixtures/filter_characterization.pt``. ``test_filter_equivalence.py``
replays the same stream and asserts the traces match.

The COMMITTED snapshot was recorded on the pre-refactor monolithic filter
classes; that is what makes the decomposition behavior-preserving rather than
merely plausible. Re-running this script rewrites the anchor from the CURRENT
code — do it only when a behavior change is intended and reviewed.
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from filter_fixtures import Fixture, drive  # noqa: E402

OUT = os.path.join(_HERE, "fixtures", "filter_characterization.pt")


def main() -> None:
  from robot_safety_sandbox.filters import qcbf_filter, safety_value_filter

  torch.manual_seed(0)
  fx = Fixture()

  vs = safety_value_filter(fx.num_envs, fx.device, fx.value_fn, fx.fallback_fn)
  value_trace = drive(vs, fx)
  value_trace["engaged_steps"] = vs.telemetry.engaged_steps.clone()
  value_trace["caution_steps"] = vs.telemetry.caution_steps.clone()

  fx2 = Fixture()
  qc = qcbf_filter(fx2.num_envs, fx2.device, fx2.q_fn, fx2.fallback_fn)
  qcbf_trace = drive(qc, fx2)
  qcbf_trace["engaged_steps"] = qc.telemetry.engaged_steps.clone()

  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  torch.save({"value_filter": value_trace, "qcbf": qcbf_trace}, OUT)
  print(f"wrote {OUT}")
  print(f"  value: engaged {value_trace['engaged'].float().mean():.4f} "
        f"caution {value_trace['caution'].float().mean():.4f}")
  print(f"  qcbf:  engaged {qcbf_trace['engaged'].float().mean():.4f} "
        f"mean |du| {qcbf_trace['intervention'].mean():.4f}")


if __name__ == "__main__":
  main()
