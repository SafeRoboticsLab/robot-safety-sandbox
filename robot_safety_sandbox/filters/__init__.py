"""Runtime safety filters: three ways to deploy a trained safety policy/value.

All filters share one control flow (the safe_adaptation_dev convention:
propose -> monitor -> override), batched over parallel envs, torch end-to-end:

  ValueShield   (value-based shielding)  switch to the safety fallback policy
                when the learned V(s) crosses a threshold; latched with
                hysteresis + a caution band. The workhorse — what
                examples/eval_filter.py deploys.
  QCBFFilter    (R-CBF / Q-CBF projection)  minimally modify the nominal
                action so the learned Q(s, a) satisfies a discrete-time
                barrier condition; falls back to the safety policy when the
                projection cannot restore the constraint.
  RolloutShield (gameplay / rollout-based shielding)  certify the nominal by
                imagining the fallback's rollout from the successor state.
                Interface stub — implementation TBD.

Margin convention throughout the zoo: safe iff >= 0 (g, l, V, Q alike).
"""

from .base import FilterInfo, SafetyFilter
from .qcbf import QCBFFilter
from .rollout_shield import RolloutShield
from .value_shield import ValueShield

__all__ = ["SafetyFilter", "FilterInfo", "ValueShield", "QCBFFilter",
           "RolloutShield"]
