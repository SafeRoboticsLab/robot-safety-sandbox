"""Go2-with-payload stabilization vs an adversarial force — the ODD-conditioned demonstration task.

Variant of go2_stabilize whose robot carries a sloshy/rigid payload (ODD = rigidity x total-mass).
Reuses go2_stabilize's stance_margins (payload-agnostic safety). Two-player (GameplaySAC via
--adversary) is the intended learner, matching the best go2_stabilize result.

Registers the ODD-randomized task + two fixed-ODD SPECIALISTS (light-rigid / heavy-sloshy) for the
bifurcation check (does the optimal strategy flip dodge<->brace across the ODD?).
"""

from __future__ import annotations

from ..registry import TaskSpec, register

_DESC = ("Go2 carrying a sloshy/rigid payload: return to a stable stand despite an adversarial base "
         "force. The payload changes the optimal strategy (dodge when light/rigid vs brace-in-place "
         "when heavy/sloshy).")


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_payload_stabilize.env_cfg import (
    go2_payload_blind_env_cfg,
    go2_payload_conditioned_env_cfg,
    go2_payload_conditioned_heavy_sloshy_env_cfg,
    go2_payload_conditioned_light_rigid_env_cfg,
    go2_payload_conditioned_ood_rigid_env_cfg,
    go2_payload_conditioned_ood_sloshy_env_cfg,
    go2_payload_descent_env_cfg,
    descent_margins,
    go2_payload_heavy_sloshy_env_cfg,
    go2_payload_light_rigid_env_cfg,
    go2_payload_ood_rigid_env_cfg,
    go2_payload_ood_sloshy_env_cfg,
    go2_payload_history_env_cfg,
    go2_payload_history_light_rigid_env_cfg,
    go2_payload_history_heavy_sloshy_env_cfg,
    go2_payload_history_ood_sloshy_env_cfg,
    go2_payload_history_ood_rigid_env_cfg,
    go2_payload_stabilize_env_cfg,
  )
  from robot_safety_sandbox.envs.go2_stabilize.env_cfg import stance_margins

  common = dict(margin_fn=stance_margins, mode="reach-avoid", supports_adversary=True)
  register(TaskSpec(task_id="go2_payload_stabilize", cfg_builder=go2_payload_stabilize_env_cfg,
                    description="[ODD: rigidity x total-mass RANDOMIZED per-env] " + _DESC, **common))
  # E008c-style A/B on Go2: one CONDITIONED policy (sees θ) vs one BLIND policy (does not), same
  # ODD-randomized env. Does the conditioned policy match BOTH specialists' strategies (dodge for
  # light-rigid θ, brace for heavy-sloshy θ) while the blind one is stuck on a single worst-case?
  register(TaskSpec(task_id="go2_payload_conditioned", cfg_builder=go2_payload_conditioned_env_cfg,
                    description="[ODD randomized + theta EXPOSED to actor+critic] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_blind", cfg_builder=go2_payload_blind_env_cfg,
                    description="[ODD randomized, theta HIDDEN — blind baseline] " + _DESC, **common))
  # SOFT-DESCENT / lie-down FALLBACK skill (its own reach-avoid margins, NOT stance_margins): from
  # standing, lower to a soft belly-down rest — reach low/level/slow WHILE keeping non-foot ground
  # contact gentle (contact-force safe set). Blind ODD-randomized env (robust to the hidden payload).
  # supports_adversary=True so it CAN take a pull later, but it is trained single-player first.
  register(TaskSpec(task_id="go2_payload_descent", cfg_builder=go2_payload_descent_env_cfg,
                    margin_fn=descent_margins, mode="reach-avoid", supports_adversary=True,
                    description="[soft-descent fallback: reach a soft belly-down rest, contact-force safe "
                                "set] Go2 (hidden randomized payload) lowers from standing to a low/level/"
                                "slow belly-down rest while keeping non-foot ground contact gentle."))
  register(TaskSpec(task_id="go2_payload_light_rigid", cfg_builder=go2_payload_light_rigid_env_cfg,
                    description="[specialist: light+rigid ~ normal Go2] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_heavy_sloshy", cfg_builder=go2_payload_heavy_sloshy_env_cfg,
                    description="[specialist: heavy+sloshy, brace regime] " + _DESC, **common))
  # EVAL-ONLY: the conditioned policy (57-dim obs) at each fixed-ODD extreme, for the read-out.
  register(TaskSpec(task_id="go2_payload_conditioned_light_rigid",
                    cfg_builder=go2_payload_conditioned_light_rigid_env_cfg,
                    description="[eval: conditioned policy @ fixed light-rigid theta] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_conditioned_heavy_sloshy",
                    cfg_builder=go2_payload_conditioned_heavy_sloshy_env_cfg,
                    description="[eval: conditioned policy @ fixed heavy-sloshy theta] " + _DESC, **common))
  # OOD-generalization eval tasks (mass 12kg > 7.5 max; rigid variant stiffness 400 > 300 max).
  register(TaskSpec(task_id="go2_payload_conditioned_ood_sloshy",
                    cfg_builder=go2_payload_conditioned_ood_sloshy_env_cfg,
                    description="[eval OOD: conditioned @ 12kg sloshy, extrapolated theta] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_ood_sloshy", cfg_builder=go2_payload_ood_sloshy_env_cfg,
                    description="[eval OOD: blind @ 12kg sloshy] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_conditioned_ood_rigid",
                    cfg_builder=go2_payload_conditioned_ood_rigid_env_cfg,
                    description="[eval OOD: conditioned @ 12kg rigid k=400, extrapolated theta] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_ood_rigid", cfg_builder=go2_payload_ood_rigid_env_cfg,
                    description="[eval OOD: blind @ 12kg rigid k=400] " + _DESC, **common))
  # E017 HISTORY arm: frame-stacked proprio+action history, no theta -> infer the payload from dynamics.
  register(TaskSpec(task_id="go2_payload_history", cfg_builder=go2_payload_history_env_cfg,
                    description="[history: ODD randomized, K-frame proprio+action history, no theta] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_history_light_rigid", cfg_builder=go2_payload_history_light_rigid_env_cfg,
                    description="[eval history @ light-rigid] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_history_heavy_sloshy", cfg_builder=go2_payload_history_heavy_sloshy_env_cfg,
                    description="[eval history @ heavy-sloshy] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_history_ood_sloshy", cfg_builder=go2_payload_history_ood_sloshy_env_cfg,
                    description="[eval history OOD @ 12kg sloshy] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_history_ood_rigid", cfg_builder=go2_payload_history_ood_rigid_env_cfg,
                    description="[eval history OOD @ 12kg rigid k=400] " + _DESC, **common))
  # E019 break-boundary sweep: find where the trained arms FALL. mass sweep x {blind,conditioned,history}.
  from functools import partial
  from robot_safety_sandbox.envs.go2_payload_stabilize.env_cfg import sweep_env_cfg
  for _m in (8, 12, 16, 20, 25):
    for _o in ("blind", "conditioned", "history"):
      register(TaskSpec(task_id=f"go2_payload_sweep_{_o}_{_m}",
                        cfg_builder=partial(sweep_env_cfg, mass=float(_m), stiffness=0.0, obs=_o),
                        description=f"[sweep: {_o} @ {_m}kg sloshy] " + _DESC, **common))
