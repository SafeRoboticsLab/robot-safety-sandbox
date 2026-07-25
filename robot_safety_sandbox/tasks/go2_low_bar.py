"""LOW-BAR crawl split test: avoid vs reach-avoid on a shared reverse curriculum
under a VIRTUAL low bar -- the 2nd reach-avoid-liveness benchmark (structural
twin of ``go2_gap_brake_or_jump``).  Height chain at FIXED depth 0.4 (vary the
under-beam clearance only, for attribution): a bootstrap rung (0.39, near
standing) then h90 / h80 / h70.  Identical env / spawn distribution per rung;
the ONLY difference is the reach term l (RA) vs none (avoid)."""

from __future__ import annotations

from functools import partial

from ..margins import avoid_only
from ..registry import AVOID, CUMULATIVE, REACH_AVOID, TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_crawl.low_bar import (
    low_bar_margins, unitree_go2_low_bar_env_cfg)

  # Dense-reward crawl-forward BRIDGE (vanilla SB3 PPO, mode=CUMULATIVE -> the env
  # is auto-built in DENSE-reward mode; same proprioception obs as the twins).
  # reward=g (the safety framing) only rewards staying safe -> it LOITERS with no
  # forward drive (a crawl has no ballistic phase to carry it, unlike the gap
  # jump). The bridge supplies the missing forward drive via the env's dense
  # reward stack and learns to crawl THROUGH the bar; both twins then warm-start
  # from it and add only the reach-avoid decision. Analog of go2_gap_crossing.
  register(TaskSpec(
    task_id="go2_low_bar_bridge",
    cfg_builder=partial(unitree_go2_low_bar_env_cfg, bar_clearance=0.39,
                        bar_depth=0.4),
    mode=CUMULATIVE,
    description="Dense-reward crawl-forward bridge (vanilla PPO): learns to crawl "
                "through the bar; warm-starts both twins. Analog of go2_gap_crossing."))

  def _pair(clearance, suffix):
    cb = partial(unitree_go2_low_bar_env_cfg, bar_clearance=clearance,
                 bar_depth=0.4)
    register(TaskSpec(
      task_id=f"go2_low_bar_avoid{suffix}", cfg_builder=cb,
      margin_fn=avoid_only(low_bar_margins), mode=AVOID,
      description=f"low-bar crawl avoid-only @clearance {clearance}, depth 0.4: "
                  f"reverse-curriculum spawn; no reach term (avoid_only)."))
    register(TaskSpec(
      task_id=f"go2_low_bar_ra{suffix}", cfg_builder=cb,
      margin_fn=low_bar_margins, mode=REACH_AVOID,
      description=f"low-bar crawl reach-avoid @clearance {clearance}, depth 0.4: "
                  f"RA target = completion past the bar. Single-variable "
                  f"contrast vs _avoid."))

  # Bootstrap rung (0.39, near standing), then the height chain at fixed depth
  # 0.4. The intended warm-start lineage -- bridge -> the 0.39 twins, then each
  # rung from the previous rung's SAME twin -- is a property of a RUN (--load),
  # recorded in docs/log/experiments.md, not of the registration.
  _pair(0.39, "")
  _pair(0.342, "_h90")
  _pair(0.304, "_h80")
  _pair(0.266, "_h70")

  # CLOSING-GATE variant: the COMMITTED-region twin the static bar cannot
  # produce. Same env/spawn/curriculum/l as the static low-bar, but the virtual
  # bar becomes a descending ceiling + a crushed_by_gate termination. Fixed
  # clearance 0.342 (h90, a light duck) so the GATE -- not the duck difficulty
  # -- is the isolated variable. Both twins warm-start from the crawl bridge.
  from robot_safety_sandbox.envs.go2_crawl.low_bar_gate import (
    low_bar_gate_margins, unitree_go2_low_bar_gate_env_cfg)

  gate_cb = partial(unitree_go2_low_bar_gate_env_cfg, bar_clearance=0.342,
                    bar_depth=0.4, gate_close_rate=0.0015)
  register(TaskSpec(
    task_id="go2_low_bar_gate_avoid", cfg_builder=gate_cb,
    margin_fn=avoid_only(low_bar_gate_margins), mode=AVOID,
    description="low-bar CLOSING-GATE crawl avoid-only @clearance 0.342, depth "
                "0.4, close-rate 0.0015: descending ceiling once past the bar "
                "face; no reach term (avoid_only) -> livelocks (committed region)."))
  register(TaskSpec(
    task_id="go2_low_bar_gate_ra", cfg_builder=gate_cb,
    margin_fn=low_bar_gate_margins, mode=REACH_AVOID,
    description="low-bar CLOSING-GATE crawl reach-avoid @clearance 0.342, depth "
                "0.4, close-rate 0.0015: RA target = completion past the bar; "
                "drives through while the gate window is open (crosses the "
                "committed region the static bar cannot produce)."))
