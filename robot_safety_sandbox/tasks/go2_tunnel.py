"""TUNNEL crawl twins (SAC family): avoid vs reach-avoid on a shared uniform
randomized-pose/-velocity spawn under a VIRTUAL low bar -- the crawl campaign's
new off-distribution formulation.  Identical env / spawn distribution across the
twins; the ONLY difference is the reach term l (RA) vs none (avoid).

NB on the SAC learner names (default_algo): the registry's ``algo_name`` resolver
(and its ``_ALGO_PROBLEM`` table) speak the PPO-family NAMES only -- they are the
canonical (problem x players) labels. The SAC trainer (examples/train_off_policy.py)
maps those PPO names to the SAC classes via ``PPO_TO_SAC``
(SafetyPPO->SafetySAC, ReachAvoidPPO->ReachAvoidSAC). So we register the PPO
names here (which algo_name accepts); running under train_off_policy.py yields
SafetySAC / ReachAvoidSAC. Registering "ReachAvoidSAC"/"SafetySAC" directly would
make algo_name raise (unknown learner)."""

from __future__ import annotations

from functools import partial

from ..margins import avoid_only
from ..registry import TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_crawl.tunnel import (
    tunnel_margins, unitree_go2_tunnel_env_cfg)

  cb = partial(unitree_go2_tunnel_env_cfg, bar_clearance=0.30, bar_depth=0.4)

  # Reach-avoid twin: RA target = completion just past the tunnel exit.
  register(TaskSpec(
    task_id="go2_tunnel_ra",
    cfg_builder=cb,
    margin_fn=tunnel_margins,
    default_algo="ReachAvoidPPO",          # -> ReachAvoidSAC via train_off_policy.py
    end_criterion="reach-avoid",
    ctrl_dim=12,
    kind="safety",
    description="tunnel crawl reach-avoid @clearance 0.30, depth 0.4: uniform "
                "randomized pose+velocity spawn; RA target = completion just "
                "past the exit. Single-variable contrast vs _avoid."))

  # Avoid-only twin: same env, reach term stripped (avoid_only).
  register(TaskSpec(
    task_id="go2_tunnel_avoid",
    cfg_builder=cb,
    margin_fn=avoid_only(tunnel_margins),
    default_algo="SafetyPPO",              # -> SafetySAC via train_off_policy.py
    end_criterion="failure",
    ctrl_dim=12,
    kind="safety",
    description="tunnel crawl avoid-only @clearance 0.30, depth 0.4: uniform "
                "randomized pose+velocity spawn; no reach term (avoid_only)."))
