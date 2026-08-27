"""RAS phase-2 (professor's B+) — Reach-Avoid-**Stay** at gap 0.30.

One task, one variable vs the proven naive-RA arm (`go2_gap_brake_or_jump_ra_w30`):
the {geometric handover gate + frozen lander + policy_mask} bundle. Everything
else — the brake_or_jump reverse-curriculum env at 0.30, `g = g_terrain_relative`,
`l = l_stable_far` — is the E040b naive-RA recipe verbatim.

The mechanism:
  - a reach-avoid PPO policy trains on the reverse curriculum;
  - when an env crosses the calibrated GEOMETRIC gate (`geom_landable_gate`:
    x_rel>=-0.06 AND vx>=1.23 — airborne, committed over the gap), base.py latches
    it to the FROZEN safelanding lander, which drives the env to episode end ("stay");
  - the lander-driven steps STAY in the rollout buffer for value/GAE propagation
    (so the reach-avoid value learns "reach a state the lander actually lands
    from"), but are EXCLUDED from the policy gradient via the per-step policy_mask.

** l is `l_stable_far` — the OUTCOME margin — NOT l_vhat. ** With the RA backup
min(g, max(l, gamma V')), an outcome-l is <0 at handover, so the lander's real
survival propagates back to the latch state; an l_vhat>=0 would FLOOR max(l,·)>=0
and a false-positive latch would stay attractive forever (Correction 1).

** MUST be trained with `ReachAvoidMaskedPPO1P` ** (the policy_mask learner), NOT
the MAP-default ReachAvoidPPO1P — pass `--algo ReachAvoidMaskedPPO1P` to
examples/train.py. With an all-ones mask (gate never fires) it degrades exactly
to naive-RA, the proven baseline. See docs/log/COMMANDS.md.

`hybrid_skill` / gate paths are portable via RAAS_LANDER_DIR (default = the local
the lander run), mirroring tasks/go2_gap_raas.py.
"""

from __future__ import annotations

import os
from functools import partial

from ..margins import compose, g_terrain_relative
from ..registry import REACH_AVOID, TaskSpec, register
from ..vhat import geom_landable_gate

# RAS phase-2 pins the same gap width as the naive-RA _w30 arm + the safelanding lander.
RAS_GAP_WIDTH = 0.30


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_gap.brake_or_jump import (
    l_stable_far, unitree_go2_brake_or_jump_env_cfg)

  # Frozen safelanding lander (the "stay" fallback). Portable path, same default as
  # tasks/go2_gap_raas.py so a checked-out repo resolves it without env vars.
  lander = os.environ.get("RAAS_LANDER_DIR",
                          "runs/raas_safelanding/go2_gap_raas_safelanding")

  register(TaskSpec(
    task_id="go2_gap_brake_or_jump_ras",
    cfg_builder=partial(unitree_go2_brake_or_jump_env_cfg, gap_width=RAS_GAP_WIDTH),
    margin_fn=compose(g_terrain_relative, l_stable_far),
    mode=REACH_AVOID,
    kwargs=dict(hybrid_skill=lander,
                latch_margin_fn=partial(geom_landable_gate, vx_min=2.55)),
    description=(
      "RAS phase-2 B+: brake_or_jump reverse curriculum @ gap 0.30, "
      "g=g_terrain_relative, l=l_stable_far (outcome margin). Frozen safelanding lander "
      "hands over on the geometric gate (x_rel>=-0.06 AND vx>=2.55); lander steps "
      "propagate to the value but are policy_masked. TRAIN WITH "
      "ReachAvoidMaskedPPO1P (--algo). One-variable vs go2_gap_brake_or_jump_ra_w30. "
      "vx_min RE-CALIBRATED 1.23->2.55: the 1.23 gate handed weak "
      "origination-launches to the lander (competent only for vx>=2.55 strong jumps) "
      "-> the weak-gate arm stalled at L8 (0.2% survival). 2.55 = lander local land-rate 0.915.")))
