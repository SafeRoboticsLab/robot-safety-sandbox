"""Rollout-based (gameplay) shielding — interface stub, implementation TBD.

The third filter style: instead of trusting a learned scalar (V or Q), the
monitor *simulates* the fallback policy from the nominal's successor state
and certifies the nominal action only if that imagined rollout stays safe
(min g >= 0 along the horizon, optionally under an adversarial disturbance —
the "gameplay" variant, cf. Hsu et al.'s gameplay filter).

Design intent for the mjlab implementation (why this is a stub):
- Needs a *shadow simulation*: either a second batched mjlab env stepped with
  saved states (mjlab state save/restore cannot yet round-trip observation-
  history buffers — the same limitation that broke respawn-based commitment
  analysis; live-switch semantics avoided it, a rollout monitor cannot), or a
  learned dynamics model, or a paired env that accepts explicit state setting.
- Cost: one fallback rollout of horizon H per env per step (amortizable by
  only re-certifying every k steps and latching in between).

The interface is fixed now so filter-comparison code can be written against
all three styles; construction raises until the shadow-sim path exists.
"""

from __future__ import annotations

from .base import SafetyFilter


class RolloutShield(SafetyFilter):
  """Certify the nominal by imagined fallback rollouts (NOT YET IMPLEMENTED).

  Planned signature: RolloutShield(num_envs, device, shadow_env, fallback_fn,
  horizon, margin_fn, adversary_fn=None, recertify_every=1).
  """

  def __init__(self, *args, **kwargs):
    raise NotImplementedError(
      "RolloutShield needs a shadow-simulation path (state save/restore "
      "including observation-history buffers, or a learned model). The "
      "interface is reserved; see the module docstring for the design.")
