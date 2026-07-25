"""TUNNEL crawl twins (SAC family): avoid vs reach-avoid on a shared uniform
randomized-pose/-velocity spawn under a VIRTUAL low bar -- the crawl campaign's
new off-distribution formulation.  Identical env / spawn distribution across the
twins; the ONLY difference is the reach term l (RA) vs none (avoid).

"SAC family" is a property of the RUN, not of these registrations: a task
declares only its ``mode`` (the MAP's **M**), and the trainer family supplies the
**A**. Trained through ``examples/train_off_policy.py`` these two resolve to
``ReachAvoidSAC1P`` / ``SafetySAC1P``; through the on-policy trainer, to
``ReachAvoidPPO1P`` / ``SafetyPPO1P``. Same tasks, same margins."""

from __future__ import annotations

from functools import partial

from ..margins import avoid_only
from ..registry import AVOID, REACH_AVOID, TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_crawl.tunnel import (
    tunnel_margins, unitree_go2_tunnel_env_cfg)

  cb = partial(unitree_go2_tunnel_env_cfg, bar_clearance=0.30, bar_depth=0.4)

  # Reach-avoid twin: RA target = completion just past the tunnel exit.
  register(TaskSpec(
    task_id="go2_tunnel_ra",
    cfg_builder=cb,
    margin_fn=tunnel_margins,
    mode=REACH_AVOID,                      # -> ReachAvoidSAC1P under train_off_policy.py
    end_criterion="reach-avoid",
    ctrl_dim=12,
    description="tunnel crawl reach-avoid @clearance 0.30, depth 0.4: uniform "
                "randomized pose+velocity spawn; RA target = completion just "
                "past the exit. Single-variable contrast vs _avoid."))

  # Avoid-only twin: same env, reach term stripped (avoid_only).
  register(TaskSpec(
    task_id="go2_tunnel_avoid",
    cfg_builder=cb,
    margin_fn=avoid_only(tunnel_margins),
    mode=AVOID,                            # -> SafetySAC1P under train_off_policy.py
    end_criterion="failure",
    ctrl_dim=12,
    description="tunnel crawl avoid-only @clearance 0.30, depth 0.4: uniform "
                "randomized pose+velocity spawn; no reach term (avoid_only)."))
