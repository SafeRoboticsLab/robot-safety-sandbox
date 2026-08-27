"""sRAS Phase I task. A single NEW task variant, distinct from the
`_ra*` / `_ras` / `_hando` arms (existing tasks untouched):

  go2_gap_brake_or_jump_sras

The hando recipe (brake_or_jump reverse curriculum @ gap 0.30, 50% harvested-
handover / 50% curriculum spawn mixture, mode REACH_AVOID -> ReachAvoidPPO1P, BARE)
EXCEPT the margin and one termination:

  - margin (g, l) = sras_margin_fn: reach target = Ω̂[stay] (the stay kernel,
    via vstay.stay_kernel_margin), failure = F ∪ (T\\Ω̂[stay]).
  - + a doomed-T failure termination (2-step debounce) when the robot lands in
    T\\Ω̂[stay].

Warm-start from the hando arm (same crossing skill / obs). See
robot_safety_sandbox/envs/go2_gap/brake_or_jump_sras.py for the construction.
"""
from __future__ import annotations

from functools import partial

from ..registry import REACH_AVOID, TaskSpec, register

SRAS_GAP_WIDTH = 0.30


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_gap.brake_or_jump_sras import (
    sras_margin_fn,
    unitree_go2_brake_or_jump_sras_env_cfg,
  )

  register(TaskSpec(
    task_id="go2_gap_brake_or_jump_sras",
    cfg_builder=partial(unitree_go2_brake_or_jump_sras_env_cfg,
                        gap_width=SRAS_GAP_WIDTH),
    margin_fn=sras_margin_fn,
    mode=REACH_AVOID,
    description=(
      "sRAS Phase I — brake_or_jump reverse curriculum @ gap 0.30 "
      "with INHERITED sets from the STAY game. reach target = Ω̂[stay] "
      "(vstay.stay_kernel_margin), failure = F ∪ (T\\Ω̂[stay]); + a doomed-T "
      "failure termination (2-step debounce) for landings outside the kernel. "
      "Same handover spawn mixture / curriculum, BARE ReachAvoidPPO1P, "
      "warm-start from the hando arm. Delta vs go2_gap_brake_or_jump_ra_hando = the "
      "(g, l) margin + the doomed-T termination.")))
