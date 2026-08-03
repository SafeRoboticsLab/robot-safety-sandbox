"""Agility Digit v3 standing safety filter vs an adversarial torso force.

The humanoid analog of ``go2_stabilize`` and, like it, a "no-machinery" task: flat ground, stock spawns, zero command, no curriculum, no
staged pipeline — trainable from scratch in one run, robustified by toggling
``--adversary`` (the worst-case torso force replaces the random push the bridge
drops).

Two problems live here, and they take different backups (see margins.py):

  digit_stabilize                    mode=REACH-AVOID — a real, reachable
                                     target (``l_digit_stay``) ->
                                     ReachAvoidPPO1P, or ReachAvoidPPO2P
                                     with ``--adversary``.
  the other four (``*_avoid``,       mode=AVOID — no target set, no l ->
  ``*_stay``, box family)            SafetyPPO1P, or SafetyPPO2P with
                                     ``--adversary``.

The ``*_stay`` tasks are avoid despite wanting a stance: "remain in the stance
set forever" is VIABILITY, folded into g as ``min(g_fall, l_stance)`` (see
``g_digit_stabilize``), not a reach. They previously declared ``l = -CLAMP``
(``l_neg``) to make a reach-avoid learner emulate avoid; that trick is invalid
(it yields an EMPTY safe set under the corrected anchor) and was removed
2026-07-17 — avoid problems now use the avoid learners.

  g = don't fall (torso height, uprightness <80 deg, no non-foot ground contact);
      FALL-ONLY — no planted-stance term, so recovery STEPS are safe
  l = settled in place (roughly upright <20 deg, low planar speed, near spawn) ->
      V > 0 == "can settle back in place despite the force, stepping if needed"

  DYNAMIC-RECOVERY design (2026-07-08): sim2sim showed a static stance depends
  on ankle-roll authority mjlab lacks (a PD statue falls ~3.5 s) while dynamic
  stepping balance transfers (a walk policy survives 20 s in both sims). So the
  filter no longer forces a planted stance; it lets the robot step to catch
  itself — the transferable strategy. See margins.py module docstring.

Digit specifics vs the go2 default: ``ctrl_dim=20`` (actuators); ``ctrl_gain=12``
(the per-joint mjlab action scale is ~0.062 rad/unit, so 12 maps SB3's natural
[-1, 1] action to ~+-0.75 rad of joint authority — go2's effective range); the
adversary force is applied to the ``torso`` body.

The Digit asset and env builders are vendored into the zoo
(``envs/assets/digit`` + ``envs/digit_safety/builders.py``); no mjlab-fork
dependency remains.
"""

from __future__ import annotations

from ..margins import compose
from ..registry import AVOID, CUMULATIVE, REACH_AVOID, TaskSpec, register


def register_all() -> None:
  from robot_safety_sandbox.envs.digit_safety.env_cfg import (
    digit_avoid_from_walk_env_cfg,
    digit_box_stabilize_env_cfg,
    digit_stabilize_env_cfg,
    digit_walk_rigidtoe_env_cfg,
  )
  from robot_safety_sandbox.envs.digit_safety.margins import (
    g_digit_box_stabilize,
    g_digit_box_stand,
    g_digit_stabilize,
    g_digit_stabilize_dense,
    g_digit_stand,
    g_digit_stand_dense,
    l_digit_stay,
  )

  # BOX family (the ORIGINAL project task, on the rigidtoe plant): balance a
  # free box on the forearms while standing. Box drop/spill is terminal ->
  # box margins live in g at every stage. Same staged pipeline as the no-box
  # family: avoid (from scratch) -> stay+planted anneal (warm-start) ->
  # the two-player game via --adversary. Actor obs includes box_pose/box_vel (no-box
  # checkpoints are not warm-start compatible).
  register(TaskSpec(
    task_id="digit_box_stabilize_avoid",
    cfg_builder=digit_box_stabilize_env_cfg,
    margin_fn=compose(g_digit_box_stand),  # avoid-only: no target set
    ctrl_dim=20,
    mode=AVOID,  # +--adversary -> SafetyPPO2P (the two-player avoid game)
    supports_adversary=True,
    kwargs={"ctrl_gain": 12.0, "adversary_body": "torso"},
    description="Box stage 1: don't fall AND don't drop/spill the box "
                "(fall-only g + box terms; no stance constraints).",
  ))

  register(TaskSpec(
    task_id="digit_box_stabilize_stay",
    cfg_builder=digit_box_stabilize_env_cfg,
    margin_fn=compose(g_digit_box_stabilize),  # avoid-only: no target set
    ctrl_dim=20,
    mode=AVOID,  # +--adversary -> SafetyPPO2P (the two-player avoid game)
    supports_adversary=True,
    kwargs={"ctrl_gain": 12.0, "adversary_body": "torso"},
    description="Box STAY: remain upright/settled/planted (annealed via "
                "_l_alpha) AND keep the box balanced forever; two-player avoid "
                "game via --adversary.",
  ))

  # STAY formulation (the fix for the reach-avoid structural degeneracy — see
  # g_digit_stabilize): avoid-only (SafetyPPO1P) on min(g_fall, l_stance), fall-only
  # termination, stance annealed via _l_alpha. "Remain in the stance set
  # forever" instead of "touch it once" (trivially satisfied at spawn).
  register(TaskSpec(
    task_id="digit_stabilize_stay",
    cfg_builder=digit_stabilize_env_cfg,
    # No l: the STAY requirement is already folded into g as
    # g' = min(g_fall, l_stance), and "remain in g' forever" is the AVOID
    # backup. There is no target set, so there is no l to declare — the old
    # l_neg trick (make a reach-avoid learner emulate avoid) is invalid under
    # the corrected anchor and is gone; see margins.py.
    margin_fn=compose(g_digit_stabilize),
    ctrl_dim=20,
    mode=AVOID,  # +--adversary -> SafetyPPO2P (the two-player avoid game)
    supports_adversary=True,
    kwargs={"ctrl_gain": 12.0, "adversary_body": "torso"},
    description="Flat ground: STAY upright + settled forever (viability of the "
                "stance set, min(g_fall, l_stance) in the avoid backup; "
                "fall-only termination; stance annealed by _l_alpha).",
  ))

  register(TaskSpec(
    task_id="digit_stabilize",
    cfg_builder=digit_stabilize_env_cfg,
    # The one GENUINE reach-avoid task of the family: l_digit_stay is a real,
    # reachable target set, so the reach-avoid backup applies as written.
    margin_fn=compose(g_digit_stand, l_digit_stay),
    ctrl_dim=20,
    mode=REACH_AVOID,  # +--adversary -> ReachAvoidPPO2P (two-player RA game)
    supports_adversary=True,
    kwargs={"ctrl_gain": 12.0, "adversary_body": "torso"},
    description="Flat ground: stand in place (upright, at rest, near spawn) — a "
                "REACHABLE reach-avoid target — despite an adversarial torso "
                "force (the humanoid two-player reach-avoid game).",
  ))

  # Avoid-only twin: identical env/g, NO reach target (the backup differs, not
  # just the margin). Isolation test — does the safety value (g) converge the
  # same with vs without the reach term? If this and digit_stabilize plateau
  # alike, l is irrelevant to g-convergence (as the reach-avoid formulation
  # predicts); if this converges much better, l is corrupting the g backup.
  register(TaskSpec(
    task_id="digit_stabilize_avoid",
    cfg_builder=digit_stabilize_env_cfg,
    margin_fn=compose(g_digit_stand),  # avoid-only: no target set
    ctrl_dim=20,
    mode=AVOID,  # +--adversary -> SafetyPPO2P (the two-player avoid game)
    supports_adversary=True,
    kwargs={"ctrl_gain": 12.0, "adversary_body": "torso"},
    description="Avoid-only twin of digit_stabilize: don't fall, no reach "
                "target. Isolation test for whether l affects g-convergence.",
  ))

  # --- Filter family (E034/E035): the nominal walker + its safety twin --------
  # These two compose into the deployable FILTER: a CUMULATIVE walker (pi_task)
  # shielded by a SafetySAC1P avoid twin whose Q(s,a) is valid on the walker's
  # MOVING states (reset_from_walk_states), for the least-restrictive/QCBF filter.

  # NOMINAL rigidtoe walker (mode=CUMULATIVE): stock SB3 PPO on dense reward.
  # supports_adversary so the filter eval can drive a torso force on the walking
  # nominal (the dstb channel); the walker itself trains adversary-off (there is
  # no two-player cumulative learner — algo_name refuses adversary+cumulative).
  register(TaskSpec(
    task_id="digit_walk_rigidtoe",
    cfg_builder=digit_walk_rigidtoe_env_cfg,
    mode=CUMULATIVE,
    ctrl_dim=20,
    supports_adversary=True,
    kwargs={"ctrl_gain": 12.0, "adversary_body": "torso"},
    description="Digit v3 rigidtoe velocity walker (dense reward, stock SB3 PPO) "
                "— same plant + obs-92 as the E012 safety twin, so it composes "
                "with the filter. Train: train.py --family on_policy.",
  ))

  # SafetySAC1P avoid twin trained on the WALKER's state distribution (E035):
  # Q_safe(s,a) valid on moving states, for the QCBF/critic filter.
  register(TaskSpec(
    task_id="digit_avoid_from_walk",
    cfg_builder=digit_avoid_from_walk_env_cfg,
    margin_fn=compose(g_digit_stand),  # avoid-only: fall-only g, no target
    ctrl_dim=20,
    mode=AVOID,
    supports_adversary=False,
    kwargs={"ctrl_gain": 12.0, "adversary_body": "torso"},
    description="SafetySAC1P avoid whose reset distribution is the E034 walker's "
                "states (incl. shoved) — Q_safe(s,a) valid on MOVING states for "
                "the QCBF filter. Fall-only g. Train: train.py --family off_policy.",
  ))

  # --- Dense certificate-preserving avoid ladder (T003, E039+) ------------------
  # AVOID on the DENSE fall margin g̃_fall (= g_digit_stand densified by a 6-channel
  # interior-quality field: upright/stillness/capture-point/ang-mom/pose/planted).
  # sign(g̃)==sign(g) pointwise, so the {V≥0} certificate is unchanged; the density
  # gives the min-backup the per-step improvement gradient it otherwise starves for
  # (the E036/E037 failure). Same env as digit_stabilize (zero command, stock resets,
  # fall-only termination). Rung 1 = SafetySAC1P bootstrap; later rungs add kick/
  # walker-state resets + a push curriculum (Digit-local cfg/reset changes).
  register(TaskSpec(
    task_id="digit_avoid_dense",
    cfg_builder=digit_stabilize_env_cfg,
    margin_fn=compose(g_digit_stand_dense),  # dense g̃_fall (V_fall engagement margin)
    ctrl_dim=20,
    mode=AVOID,
    supports_adversary=True,
    kwargs={"ctrl_gain": 12.0, "adversary_body": "torso"},
    description="Dense-avoid bootstrap (T003/E039): SafetySAC on g̃_fall — "
                "g_digit_stand + certificate-preserving 6-channel shaping. Zero "
                "command, stock resets, fall-only. Train: train.py --family off_policy.",
  ))

  # Dense STAY margin g̃_stay (release/handoff certificate; α-annealed via _l_alpha).
  # Used at the stay-tightening rung (E041), not for engagement.
  register(TaskSpec(
    task_id="digit_stay_dense",
    cfg_builder=digit_stabilize_env_cfg,
    margin_fn=compose(g_digit_stabilize_dense),  # dense g̃_stay (V_stay release margin)
    ctrl_dim=20,
    mode=AVOID,
    supports_adversary=True,
    kwargs={"ctrl_gain": 12.0, "adversary_body": "torso"},
    description="Dense-stay (T003/E041): SafetySAC on g̃_stay = min(g_fall,l_stance) "
                "densified; the settled-set RELEASE certificate. Train: --family off_policy.",
  ))
