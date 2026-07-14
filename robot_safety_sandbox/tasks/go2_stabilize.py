"""Go2 flat-ground stabilization / locomotion vs an adversarial force.

The ORIGINAL task of this safety line (ISAACS Tier 2) and the simplest zoo
entry: no curricula, no staged pipeline, task-local margins. The reference
starting point for porting a task that needs no special machinery.
"""

from __future__ import annotations

from ..registry import TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_stabilize.env_cfg import (
    go2_locomote_env_cfg,
    go2_stabilize_env_cfg,
    locomote_margins,
    stance_margins,
  )
  register(TaskSpec(
    task_id="go2_stabilize", cfg_builder=go2_stabilize_env_cfg,
    margin_fn=stance_margins, default_algo="ReachAvoidPPO",
    supports_adversary=True,
    description="Flat ground, zero command: return to a stable stand despite "
                "an adversarial base force (the original ISAACS Tier-2 task)."))
  register(TaskSpec(
    task_id="go2_locomote", cfg_builder=go2_locomote_env_cfg,
    margin_fn=locomote_margins, default_algo="ReachAvoidPPO",
    supports_adversary=True,
    description="Flat ground, constant forward command: keep tracking it "
                "despite an adversarial base force."))
