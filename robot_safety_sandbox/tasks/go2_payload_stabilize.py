"""Go2-with-payload stabilization vs an adversarial force — the ODD-conditioned demonstration task.

Variant of go2_stabilize whose robot carries a sloshy/rigid payload (ODD = rigidity x total-mass).
Reuses go2_stabilize's stance_margins (payload-agnostic safety). Two-player (GameplaySAC via
--adversary) is the intended learner, matching the best go2_stabilize result.
"""

from __future__ import annotations

from ..registry import TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_payload_stabilize.env_cfg import (
    go2_payload_stabilize_env_cfg,
  )
  from robot_safety_sandbox.envs.go2_stabilize.env_cfg import stance_margins

  register(TaskSpec(
    task_id="go2_payload_stabilize", cfg_builder=go2_payload_stabilize_env_cfg,
    margin_fn=stance_margins, default_algo="ReachAvoidPPO",
    supports_adversary=True,
    description="Go2 carrying a sloshy/rigid payload (ODD: rigidity x total-mass): return to a "
                "stable stand despite an adversarial base force. The payload changes the optimal "
                "strategy (dodge when light/rigid vs brace-in-place when heavy/sloshy)."))
