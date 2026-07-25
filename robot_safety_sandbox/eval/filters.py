"""One table from ``--filter <kind>`` to a composition, and nothing else.

The five recipes differ in exactly ONE module each (see
:mod:`robot_safety_sandbox.filters`), so this builder is a dispatch table over
what the twin supplies -- there is no per-filter branching anywhere else in the
evaluation stack.

  value     ValueMonitor       V(s)                     any PPO twin
  critic    CriticMonitor      Q(s, u_nom)              any SAC twin
  qcbf      CriticMonitor      Q(s, u), + QCBFIntervention (minimal modification)
  rollout   RolloutMonitor     H shadow-sim steps       any twin
  gameplay  AdversarialRolloutMonitor  the same, played against pi_dstb  2P twin

A shadow rollout must certify THE WORLD BEING FILTERED, so the shadow bridge is
built from the live env's own ``cfg_builder`` (which already carries any preset
surgery) rather than from the task's stock one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .. import filters as F
from ..base import MjlabTensorSafetyEnv
from ..registry import REACH_AVOID, spec

#: --filter -> one line on what it is and what the twin must supply.
FILTERS = {
  "value":    "V(s) from an on-policy twin, latched switch",
  "critic":   "Q(s, u_nom) from a SAC twin, same latched switch",
  "qcbf":     "Q(s, u) from a SAC twin, minimal modification (R-CBF)",
  "rollout":  "simulate pi^< for H steps in a shadow env, latched switch",
  "gameplay": "the same rollout, played against the twin's dstb actor",
}

#: which twin capability each filter needs, if any beyond the fallback.
FILTER_NEEDS = {"value": "value_fn", "critic": "q_fn", "qcbf": "q_fn",
                "rollout": None, "gameplay": "dstb_fn"}


@dataclass
class SwitchCfg:
  """LeastRestrictiveIntervention's knobs (the latched eps-switch)."""
  eps: float = 0.0
  caution: float = 0.45
  hysteresis: float = 0.15

  def kwargs(self) -> dict:
    return dict(eps=self.eps, caution=self.caution,
                hysteresis=self.hysteresis)


@dataclass
class RolloutCfg:
  """The shadow-rollout monitors' knobs (rollout / gameplay only).

  ``reach_avoid`` is NOT a free knob: which reduction certifies a rollout is a
  property of the TASK, exactly as the MAP law derives the learner from it (a
  ``mode="reach-avoid"`` task is valued by ``max_t min(l_t, min_{s<=t} g_s)``,
  a ``mode="safety"`` one by ``min_t g_t``). Leave it ``None`` -- the default,
  and the only setting an evaluation should use -- and :func:`build_filter`
  derives it from ``spec(env.task).mode``. An explicit bool may only CONFIRM
  what the mode derives (useful as an assertion); disagreeing with it is an
  error, because certifying a reach-avoid task with the avoid reduction proves
  "the fallback avoided failure for H steps" while never requiring the target
  to be reached -- a strictly weaker claim wearing the reach-avoid name.
  """
  horizon: int = 20
  rollouts: int = 1
  recertify_every: int = 1
  reach_avoid: Optional[bool] = None
  contact_history: str = "sync"


@dataclass
class FilterBundle:
  """The built filter plus whatever it owns and must later close."""
  filt: F.SafetyFilter
  shadow: Optional[F.MjlabShadowSim] = None
  shadow_bridge: Optional[MjlabTensorSafetyEnv] = None
  extras: dict = field(default_factory=dict)

  def close(self) -> None:
    if self.shadow_bridge is not None:
      self.shadow_bridge.close()


def _need(mods: dict, key: str, kind: str):
  if key not in mods:
    raise SystemExit(
      f"--filter {kind} needs '{key}', which the twin ({mods.get('twin')}) "
      "does not have. Q(s, a) comes from an off-policy twin "
      "({Safety,ReachAvoid}SAC{1P,2P}); a disturbance actor, from a 2P one. "
      f"This twin supplies: {sorted(k for k in mods if k.endswith('_fn'))}.")
  return mods[key]


def reach_avoid_reduction(task: str, override: Optional[bool] = None) -> bool:
  """Which rollout reduction certifies ``task`` -- derived from its mode.

  The same law that names the learner (``registry.algo_name``) fixes the
  reduction: the mode IS the outcome functional the task's margins define, and
  a shadow rollout is that functional's Monte-Carlo evaluation under
  (pi^<, pi_d). So there is nothing left to choose.

  :param override: normally ``None``. A bool is accepted only to ASSERT the
      derived value; disagreement raises, it never wins.
  """
  s = spec(task)
  derived = s.mode == REACH_AVOID
  if derived and not getattr(s.margin_fn, "has_target", True):
    raise SystemExit(
      f"task '{task}' declares mode={REACH_AVOID!r} but its margin_fn has no "
      "target channel (l is a zero placeholder), so the reach-avoid reduction "
      "max_t min(l, min_s<=t g) is degenerate. Fix the task's margins or "
      f"declare mode='safety'.")
  if override is not None and bool(override) != derived:
    want = "reach-avoid" if derived else "avoid"
    got = "reach-avoid" if override else "avoid"
    raise SystemExit(
      f"reach_avoid={override} contradicts task '{task}' (mode={s.mode!r}, "
      f"which requires the {want} reduction, not the {got} one). The rollout "
      "reduction is DERIVED from the task's mode, not chosen: certifying a "
      "reach-avoid task with min_t g would only prove the fallback avoided "
      "failure over the horizon, never that it reached the target. Drop the "
      "override (leave it None) to get the right one.")
  return derived


def build_shadow(env, mods: dict, rollout: RolloutCfg, *,
                 adversary: bool) -> tuple:
  """A shadow sim over THIS env's cfg (not the task's stock one).

  The evaluation env may be surgically modified by a preset (a pinned terrain
  parameter, a task-specific spawn), so the shadow is built from the SAME
  cfg_builder -- otherwise the rollout certifies a different world than the one
  being filtered.
  """
  s = spec(env.task)
  n_shadow = env.num_envs * rollout.rollouts
  bridge = MjlabTensorSafetyEnv(
    n_shadow, env.device, cfg_builder=env.cfg_builder, margin_fn=s.margin_fn,
    ctrl_dim=s.ctrl_dim, dstb_dim=s.dstb_dim, adversary=adversary,
    end_criterion=s.end_criterion, obs_key=env.safety_obs_key, **s.kwargs)
  shadow = F.MjlabShadowSim(
    env.mj, bridge, num_envs=env.num_envs,
    rollouts_per_env=rollout.rollouts,
    obs_adapter=lambda obs: {"s_obs": mods["norm"](obs.float())},
    contact_history=rollout.contact_history)
  return bridge, shadow


def build_filter(kind: str, mods: dict, env, *, switch: SwitchCfg | None = None,
                 rollout: RolloutCfg | None = None, kappa: float = 0.8,
                 shadow=None) -> FilterBundle:
  """Assemble the requested composition out of one twin's modules.

  :param kind: one of :data:`FILTERS`.
  :param mods: :func:`~robot_safety_sandbox.eval.policies.safety_modules`
      output, plus a ``"norm"`` entry (the twin's obs normalizer) when a
      shadow rollout is involved.
  :param env: the :class:`~robot_safety_sandbox.eval.envs.EvalEnv` being
      filtered -- supplies num_envs/device and, for the rollout monitors, the
      cfg the shadow must mirror.
  :param shadow: an already-built ShadowSim to use instead of instantiating one
      (rollout/gameplay only). The bundle then does not own it and will not
      close it.
  """
  if kind not in FILTERS:
    raise ValueError(f"unknown --filter {kind!r}; one of {sorted(FILTERS)}")
  switch = switch or SwitchCfg()
  rollout = rollout or RolloutCfg()
  n, device = env.num_envs, env.device

  if kind == "value":
    return FilterBundle(F.safety_value_filter(
      n, device, _need(mods, "value_fn", kind), mods["fallback_fn"],
      **switch.kwargs()))
  if kind == "critic":
    return FilterBundle(F.safety_critic_filter(
      n, device, _need(mods, "q_fn", kind), mods["fallback_fn"],
      **switch.kwargs()))
  if kind == "qcbf":
    return FilterBundle(F.qcbf_filter(
      n, device, _need(mods, "q_fn", kind), mods["fallback_fn"], kappa=kappa))

  adversarial = kind == "gameplay"
  dstb_fn = _need(mods, "dstb_fn", kind) if adversarial else None
  bridge = None
  if shadow is None:
    bridge, shadow = build_shadow(env, mods, rollout, adversary=adversarial)
  reach_avoid = reach_avoid_reduction(env.task, rollout.reach_avoid)
  print(f"[filter] {kind}: reduction = "
        + ("reach-avoid  max_t min(l_t, min_s<=t g_s)" if reach_avoid
           else "avoid  min_t g_t")
        + f"  (derived from mode={spec(env.task).mode!r})")
  common = dict(reach_avoid=reach_avoid,
                recertify_every=rollout.recertify_every, **switch.kwargs())
  if adversarial:
    filt = F.gameplay_filter(n, device, mods["fallback_fn"], shadow,
                             rollout.horizon, dstb_fn, **common)
  else:
    filt = F.rollout_filter(n, device, mods["fallback_fn"], shadow,
                            rollout.horizon, **common)
  return FilterBundle(filt, shadow=shadow, shadow_bridge=bridge)
