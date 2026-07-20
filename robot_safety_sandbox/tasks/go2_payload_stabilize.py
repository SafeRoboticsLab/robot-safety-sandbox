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
    go2_payload_heavy_sloshy_env_cfg,
    go2_payload_light_rigid_env_cfg,
    go2_payload_stabilize_env_cfg,
  )
  from robot_safety_sandbox.envs.go2_stabilize.env_cfg import stance_margins

  common = dict(margin_fn=stance_margins, default_algo="ReachAvoidPPO", supports_adversary=True)
  register(TaskSpec(task_id="go2_payload_stabilize", cfg_builder=go2_payload_stabilize_env_cfg,
                    description="[ODD: rigidity x total-mass RANDOMIZED per-env] " + _DESC, **common))
  # E008c-style A/B on Go2: one CONDITIONED policy (sees θ) vs one BLIND policy (does not), same
  # ODD-randomized env. Does the conditioned policy match BOTH specialists' strategies (dodge for
  # light-rigid θ, brace for heavy-sloshy θ) while the blind one is stuck on a single worst-case?
  register(TaskSpec(task_id="go2_payload_conditioned", cfg_builder=go2_payload_conditioned_env_cfg,
                    description="[ODD randomized + theta EXPOSED to actor+critic] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_blind", cfg_builder=go2_payload_blind_env_cfg,
                    description="[ODD randomized, theta HIDDEN — blind baseline] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_light_rigid", cfg_builder=go2_payload_light_rigid_env_cfg,
                    description="[specialist: light+rigid ~ normal Go2] " + _DESC, **common))
  register(TaskSpec(task_id="go2_payload_heavy_sloshy", cfg_builder=go2_payload_heavy_sloshy_env_cfg,
                    description="[specialist: heavy+sloshy, brace regime] " + _DESC, **common))
