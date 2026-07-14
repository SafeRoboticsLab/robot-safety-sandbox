"""Crawl R-CBF twins: avoid/RA pairs on the low-bar task, static + closing gate.

The low-bar half of the R-CBF claim (see tasks/go2_gap.py twins for the gap
half). Same one-line-backup contrast, two terrain conditions:

  static bar   : NO committed region -> theory predicts avoid == RA filter
                 (the negative control; a predicted null).
  closing gate : committed region (virtual descending ceiling after entry) ->
                 avoid-filter livelocks at the face, RA-filter drives through.

All four share the env family, spawns, budget; only backup/terrain flags vary.
Nominal for the filter eval: go2_walker_flat (the same blind walker as gap).
"""

from __future__ import annotations

from ..registry import TaskSpec, register


def register_all() -> None:
  from safe_mjlab_zoo.envs.go2_crawl.twins import (
    crawl_gate_twin_margins,
    crawl_twin_margins,
    unitree_go2_crawl_gate_twin_env_cfg,
    unitree_go2_crawl_twin_env_cfg,
  )

  def _l_zero_wrap(margin_fn):
    def fn(env):
      g, l = margin_fn(env)
      return g, l * 0.0   # avoid-only: same g, l identically 0
    return fn

  register(TaskSpec(
    task_id="go2_crawl_twin_avoid",
    cfg_builder=unitree_go2_crawl_twin_env_cfg,
    margin_fn=_l_zero_wrap(crawl_twin_margins), default_algo="SafetyPPO",
    description="STATIC-bar avoid twin (negative control). g = crawl "
                "integrity, l = 0."))
  register(TaskSpec(
    task_id="go2_crawl_twin_ra",
    cfg_builder=unitree_go2_crawl_twin_env_cfg,
    margin_fn=crawl_twin_margins, default_algo="ReachAvoidPPO",
    description="STATIC-bar RA twin (negative control). l = min(rest, "
                "past-bar) completion."))
  register(TaskSpec(
    task_id="go2_crawl_gate_avoid",
    cfg_builder=unitree_go2_crawl_gate_twin_env_cfg,
    margin_fn=_l_zero_wrap(crawl_gate_twin_margins), default_algo="SafetyPPO",
    description="CLOSING-GATE avoid twin: g += virtual descending ceiling "
                "after entry (+ crushed_by_gate termination). Prediction: "
                "certified set excludes the gate span -> filter livelocks."))
  register(TaskSpec(
    task_id="go2_crawl_gate_ra",
    cfg_builder=unitree_go2_crawl_gate_twin_env_cfg,
    margin_fn=crawl_gate_twin_margins, default_algo="ReachAvoidPPO",
    description="CLOSING-GATE RA twin: completion-l + gate g. Prediction: "
                "commits and crawls through while the window is open."))
