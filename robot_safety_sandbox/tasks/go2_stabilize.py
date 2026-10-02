"""Go2 flat-ground stabilization / locomotion vs an adversarial force.

The ORIGINAL task of this safety line and the simplest zoo entry: no curricula,
no staged pipeline, task-local margins. The reference starting point for porting
a task that needs no special machinery. Both tasks are mode="reach-avoid" and
support ``--adversary`` -> the two-player game (ReachAvoidPPO2P /
ReachAvoidSAC2P, the E042 config).
"""

from __future__ import annotations

import functools

from ..registry import REACH_AVOID, TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_stabilize.env_cfg import (
    go2_locomote_env_cfg,
    go2_stabilize_env_cfg,
    locomote_margins,
    stance_margins,
  )
  register(TaskSpec(
    task_id="go2_stabilize", cfg_builder=go2_stabilize_env_cfg,
    margin_fn=stance_margins, mode=REACH_AVOID,
    supports_adversary=True,
    description="Flat ground, zero command: return to a stable stand despite "
                "an adversarial base force (the original task of this line)."))
  register(TaskSpec(
    task_id="go2_locomote", cfg_builder=go2_locomote_env_cfg,
    margin_fn=locomote_margins, mode=REACH_AVOID,
    supports_adversary=True,
    description="Flat ground, constant forward command: keep tracking it "
                "despite an adversarial base force."))
  from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import (
    go2_weak_leg_env_cfg,
  )
  # Physics-limit ODD axis: FR-leg allowable torque as a fraction of nominal.
  # 1.0 = normal (go2_stabilize) ... 0.5 / 0.2 ... 0.0 = dead (infeasible).
  for _frac, _pct in ((0.5, "50"), (0.2, "20")):
    register(TaskSpec(
      task_id=f"go2_weak_leg_{_pct}",
      cfg_builder=functools.partial(go2_weak_leg_env_cfg, torque_frac=_frac),
      margin_fn=stance_margins, mode=REACH_AVOID,
      supports_adversary=True,
      description=f"[ODD: front-right leg WEAK — allowable torque scaled to {_pct}%; "
                  "else identical to go2_stabilize]"))
  # Back-compat alias: bare go2_weak_leg == 50%.
  register(TaskSpec(
    task_id="go2_weak_leg", cfg_builder=go2_weak_leg_env_cfg,
    margin_fn=stance_margins, mode=REACH_AVOID,
    supports_adversary=True,
    description="[ODD: front-right leg WEAK — allowable torque scaled to 50%; "
                "else identical to go2_stabilize]"))
  # FINE torque-fraction grid for the survival sweep / swap-matrix eval (E049):
  # go2_weak_leg_sweep_{pct}, pct in {100,70,50,30,20,10}. Build-time fixed θ per task
  # (the static-sweep analog of go2_payload_sweep_*), so each policy can be evaluated at
  # any FR-torque level, on- or off- its own training θ.
  for _pct in (100, 70, 50, 30, 20, 10):
    register(TaskSpec(
      task_id=f"go2_weak_leg_sweep_{_pct}",
      cfg_builder=functools.partial(go2_weak_leg_env_cfg, torque_frac=_pct / 100.0),
      margin_fn=stance_margins, mode=REACH_AVOID,
      supports_adversary=True,
      description=f"[sweep: FR-leg allowable torque {_pct}%] " + go2_weak_leg_env_cfg.__doc__.split(".")[0]))
  # ── LEG-DEGRADATION ODD arms: θ = FR allowable-torque fraction ∈ [0.2, 1.0], RANDOMIZED per-episode
  # (fresh θ each reset — a continuum, not fixed specialists). Three arms differ ONLY in what the policy
  # observes (mirrors the payload blind/conditioned/history phase).
  from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import (
    go2_weak_leg_randomized_env_cfg,
    go2_weak_leg_conditioned_env_cfg,
    go2_weak_leg_history_env_cfg,
  )
  register(TaskSpec(
    task_id="go2_weak_leg_blind", cfg_builder=go2_weak_leg_randomized_env_cfg,
    margin_fn=stance_margins, mode=REACH_AVOID, supports_adversary=True,
    description="[leg ODD: FR torque θ∈[0.2,1.0] randomized per-episode; θ HIDDEN — blind baseline]"))
  register(TaskSpec(
    task_id="go2_weak_leg_conditioned", cfg_builder=go2_weak_leg_conditioned_env_cfg,
    margin_fn=stance_margins, mode=REACH_AVOID, supports_adversary=True,
    description="[leg ODD: FR torque θ∈[0.2,1.0] randomized per-episode; θ EXPOSED to actor+critic]"))
  register(TaskSpec(
    task_id="go2_weak_leg_history", cfg_builder=go2_weak_leg_history_env_cfg,
    margin_fn=stance_margins, mode=REACH_AVOID, supports_adversary=True,
    description="[leg ODD: FR torque θ∈[0.2,1.0] randomized per-episode; θ HISTORY — K-frame proprio+action stack]"))
  # FIXED-θ EVAL tasks for the conditioned & history policies (survival sweep E054): θ pinned per grid
  # point so each arm gets its training obs surface (θ scalar / K-stack) at a single test torque.
  from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import (
    go2_weak_leg_conditioned_at_env_cfg,
    go2_weak_leg_history_at_env_cfg,
  )
  for _pct in (100, 70, 50, 30, 20, 10):
    register(TaskSpec(
      task_id=f"go2_weak_leg_conditioned_at_{_pct}",
      cfg_builder=functools.partial(go2_weak_leg_conditioned_at_env_cfg, theta=_pct / 100.0),
      margin_fn=stance_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[eval: conditioned policy @ fixed FR-torque {_pct}%]"))
    register(TaskSpec(
      task_id=f"go2_weak_leg_history_at_{_pct}",
      cfg_builder=functools.partial(go2_weak_leg_history_at_env_cfg, theta=_pct / 100.0),
      margin_fn=stance_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[eval: history policy @ fixed FR-torque {_pct}%]"))
  # ── SOFT-REST objective: safety = don't SLAM (<10N contact), target = stay LEVEL (small roll/pitch).
  # The ODD-AWARE margin for the degraded regime — a robot that can't stand may GRACEFULLY SIT instead of
  # fighting a pull it can't resist. Same envs/adversary as above; only margin_fn=soft_rest_margins.
  from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import (
    soft_rest_margins,
    go2_weak_leg_soft_env_cfg,
    go2_weak_leg_blind_soft_env_cfg,
    go2_weak_leg_conditioned_soft_env_cfg,
    go2_weak_leg_history_soft_env_cfg,
    go2_weak_leg_conditioned_soft_at_env_cfg,
    go2_weak_leg_history_soft_at_env_cfg,
  )
  register(TaskSpec(
    task_id="go2_weak_leg_20_soft",
    cfg_builder=functools.partial(go2_weak_leg_soft_env_cfg, torque_frac=0.2),
    margin_fn=soft_rest_margins, mode=REACH_AVOID, supports_adversary=True,
    description="[soft-rest: FR 20% torque specialist; safety=no-slam(<80N), target=level. Sit gracefully?]"))
  register(TaskSpec(
    task_id="go2_weak_leg_blind_soft", cfg_builder=go2_weak_leg_blind_soft_env_cfg,
    margin_fn=soft_rest_margins, mode=REACH_AVOID, supports_adversary=True,
    description="[soft-rest BLIND: θ∈[0.2,1.0] rand per-ep, θ hidden; safety=no-slam(<80N), target=level]"))
  register(TaskSpec(
    task_id="go2_weak_leg_conditioned_soft", cfg_builder=go2_weak_leg_conditioned_soft_env_cfg,
    margin_fn=soft_rest_margins, mode=REACH_AVOID, supports_adversary=True,
    description="[soft-rest CONDITIONED: θ exposed; safety=no-slam(<80N), target=level. Sit when leg dead?]"))
  register(TaskSpec(
    task_id="go2_weak_leg_history_soft", cfg_builder=go2_weak_leg_history_soft_env_cfg,
    margin_fn=soft_rest_margins, mode=REACH_AVOID, supports_adversary=True,
    description="[soft-rest HISTORY: K-frame stack; safety=no-slam(<80N), target=level]"))
  # fixed-θ soft-rest eval tasks (for the dynamic-ramp eval readout of conditioned/history)
  for _pct in (100, 50, 20):
    register(TaskSpec(
      task_id=f"go2_weak_leg_conditioned_soft_at_{_pct}",
      cfg_builder=functools.partial(go2_weak_leg_conditioned_soft_at_env_cfg, theta=_pct / 100.0),
      margin_fn=soft_rest_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[soft-rest eval: conditioned @ fixed FR-torque {_pct}%]"))
    register(TaskSpec(
      task_id=f"go2_weak_leg_history_soft_at_{_pct}",
      cfg_builder=functools.partial(go2_weak_leg_history_soft_at_env_cfg, theta=_pct / 100.0),
      margin_fn=soft_rest_margins, mode=REACH_AVOID, supports_adversary=True,
      description=f"[soft-rest eval: history @ fixed FR-torque {_pct}%]"))
  # ── DYNAMIC-θ soft-rest TRAINING arms (E064): θ changes MID-EPISODE during training (LegDeathCallback
  # drives θ per-env per-step; the reset randomization is dropped). Same soft-rest margin/adversary as the
  # static arms; the policies now PRACTICE the leg-death transition instead of only static per-episode θ.
  from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import (
    go2_weak_leg_blind_soft_dyn_env_cfg,
    go2_weak_leg_conditioned_soft_dyn_env_cfg,
    go2_weak_leg_history_soft_dyn_env_cfg,
  )
  register(TaskSpec(
    task_id="go2_weak_leg_blind_soft_dyn", cfg_builder=go2_weak_leg_blind_soft_dyn_env_cfg,
    margin_fn=soft_rest_margins, mode=REACH_AVOID, supports_adversary=True,
    description="[E064 dynamic-θ training: θ changes mid-episode] soft-rest BLIND: θ∈[0.2,1.0] driven "
                "per-step (step change at t~U[50,200]); θ hidden; safety=no-slam(<80N), target=level"))
  register(TaskSpec(
    task_id="go2_weak_leg_conditioned_soft_dyn", cfg_builder=go2_weak_leg_conditioned_soft_dyn_env_cfg,
    margin_fn=soft_rest_margins, mode=REACH_AVOID, supports_adversary=True,
    description="[E064 dynamic-θ training: θ changes mid-episode] soft-rest CONDITIONED: live θ exposed "
                "to actor+critic; safety=no-slam(<80N), target=level"))
  register(TaskSpec(
    task_id="go2_weak_leg_history_soft_dyn", cfg_builder=go2_weak_leg_history_soft_dyn_env_cfg,
    margin_fn=soft_rest_margins, mode=REACH_AVOID, supports_adversary=True,
    description="[E064 dynamic-θ training: θ changes mid-episode] soft-rest HISTORY: K-frame proprio+action "
                "stack (no θ); safety=no-slam(<80N), target=level"))
