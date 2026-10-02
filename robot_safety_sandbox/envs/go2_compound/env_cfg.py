"""Go2 losing its FR leg WHILE carrying a high-CoM load — the COMPOUND two-mode ODD (T004 / E078, DEMO 2).

The COMPOSITION of the two ODD mechanisms already verified separately, both driven together every episode:
  * a CONSTANT carried load W = COMPOUND_W (N) at height h = COMPOUND_H (the inverted-pendulum lever from
    go2_weight_ladder — once tilted it torques the body further over); AND
  * the FR-leg allowable-torque fraction θ (the per-episode ``randomize_fr_torque`` axis from go2_broken_leg),
    the DYNAMIC ODD parameter the modes are conditioned on.

WHY the compound axis (the E078 gate, PASS): the UNLOADED leg-at-stance certificate was FLAT — a reactive
policy stands down to θ≈0.1 (E076 PART B, the paper's negative control). But WITH the high-CoM load, standing
on 3.5 legs under the pendulum genuinely FAILS below θ_c≈0.3 (E078 gate: E075-recal stand tips 0.03→0.14→0.25→
0.48 as θ falls 0.4→0.3→0.2→0.1; slam→0.50 at θ=0.1), while a scripted descent stays gentle everywhere
(v_touch≈0.05-0.14 m/s, tilt≤0.3, settled contact 38-51 N ≪ SLAM_CAP(80)=184 N). So V_compound_stand(x, θ)
CONTRACTS as θ falls → a REAL certified leg handoff, the "cargo robot loses a motor" story.

Two conditioned reach-avoid modes, each carrying the CONSTANT load W (so the ONLY randomized ODD is θ, exposed
to actor+critic — the W obs is NOT added, W is fixed):
  * COMPOUND STAND — ``stance_margins``: hold a stable stand despite the weak leg + the carried load +
    adversary. θ ∈ [STAND_THETA_LO, 1.0] = the comfortably-feasible band ABOVE the gate's θ_c (E075 lesson:
    train on the feasible band for a calibrated certificate).
  * COMPOUND REST  — ``weight_rest_margins``: come down softly to a LOW LEVEL belly rest WITHOUT slamming,
    with the LOAD-CONDITIONED slam cap SLAM_CAP(W)=80+1.3W (=184 N at W=80; the measured settled rest force
    ~38-51 N sits comfortably under it). θ ∈ [0.0, 1.0] INCLUDING a fully-dead leg. ``illegal_contact`` raised
    to REST_CONTACT_N so a belly rest under load does not terminate.

ADDITIVE — built on ``go2_stabilize`` + the ``randomize_base_load`` load channel (go2_weight_ladder) + the FR-
torque θ machinery (go2_broken_leg); no existing task or margin is modified.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg

from robot_safety_sandbox.envs.go2_stabilize.env_cfg import (  # reused verbatim
  go2_stabilize_env_cfg,
  stance_margins,
)
from robot_safety_sandbox.envs.go2_weight_ladder.env_cfg import (  # the carried-load channel + rest margins
  randomize_base_load,
  weight_rest_margins,
  REST_CONTACT_N,
)
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import (  # the FR-torque θ machinery
  cache_fr_nominal_forcerange,
  randomize_fr_torque,
  _add_fr_conditioning_obs,
  _raise_contact_termination,
)

# ── Compound-ODD constants ──────────────────────────────────────────────────────────────────────
COMPOUND_W = 80.0        # constant carried load (N) — the E078 gate operating point
COMPOUND_H = 0.25        # high-CoM lever (m) — inverted-pendulum torque once tilted (validated E073)
# STAND θ band: just ABOVE the E078 gate's θ_c≈0.3 (stand comfortably feasible for θ≥0.4: tip≤0.06). The
# E075 lesson — train on the comfortably-feasible band so V_stand is a calibrated certificate near its edge.
STAND_THETA_LO, STAND_THETA_HI = 0.4, 1.0
REST_THETA_LO, REST_THETA_HI = 0.0, 1.0   # REST covers the fully-dead FR leg (θ=0 — lying down needs no leg)


def _add_constant_load(cfg: ManagerBasedRlEnvCfg, W: float, load_height: float) -> None:
  """Attach a CONSTANT high-CoM carried load: ``randomize_base_load`` with lo=hi=W (so every episode gets the
  same W) at ``load_height``. Stores ``env._weight_W``=W and ``env._weight_h``=h each reset; ``base.py``'s
  ``_apply_dstb`` adds the downward wrench [0,0,-W] + the inverted-pendulum torque τ=R·[0,0,h]×[0,0,-W]."""
  cfg.events["randomize_base_load"] = EventTermCfg(
    func=randomize_base_load, mode="reset", params={"lo": W, "hi": W, "load_height": load_height})


def _add_fr_theta_events(cfg: ManagerBasedRlEnvCfg, lo: float, hi: float) -> None:
  """Attach the FR-leg torque θ machinery: the startup cache (snapshots nominal forcerange + allocates the θ
  store, ``@requires_model_fields`` expands per-world forcerange) + the per-episode reset randomization θ~U[lo,
  hi] (absolute write nominal*θ, no compounding). Mirrors ``go2_weak_leg_randomized_env_cfg``'s two events."""
  cfg.events["cache_fr_nominal"] = EventTermCfg(func=cache_fr_nominal_forcerange, mode="startup", params={})
  cfg.events["randomize_fr_torque"] = EventTermCfg(
    func=randomize_fr_torque, mode="reset", params={"lo": lo, "hi": hi})


def go2_compound_stand_env_cfg(
  play: bool = False, lo: float = STAND_THETA_LO, hi: float = STAND_THETA_HI,
  W: float = COMPOUND_W, load_height: float = COMPOUND_H,
) -> ManagerBasedRlEnvCfg:
  """COMPOUND STAND mode: ``go2_stabilize`` + CONSTANT high-CoM load W@h + FR torque θ~U[lo,hi] randomized
  per-episode + θ exposed to actor+critic (obs 48 = 47 + θ; the W obs is NOT added — W is constant). Margins =
  ``stance_margins``: hold a stand despite the weak leg + the load + the adversary. θ band defaults to the
  comfortably-feasible [0.4, 1.0] (above the E078 gate's θ_c). Everything else inherited from go2_stabilize."""
  cfg = go2_stabilize_env_cfg(play=play)
  _add_constant_load(cfg, W, load_height)
  _add_fr_theta_events(cfg, lo, hi)
  _add_fr_conditioning_obs(cfg)
  return cfg


def go2_compound_rest_env_cfg(
  play: bool = False, lo: float = REST_THETA_LO, hi: float = REST_THETA_HI,
  W: float = COMPOUND_W, load_height: float = COMPOUND_H,
) -> ManagerBasedRlEnvCfg:
  """COMPOUND REST mode: ``go2_stabilize`` + CONSTANT high-CoM load W@h + FR torque θ~U[lo,hi] randomized
  per-episode (span [0.0, 1.0] INCLUDING a fully-dead FR leg) + θ exposed to actor+critic + ``illegal_contact``
  raised to REST_CONTACT_N so a belly rest under load does not terminate. Margins = ``weight_rest_margins``
  (LOAD-CONDITIONED no-slam cap SLAM_CAP(W)=80+1.3W = 184 N at W=80 + LOW/level/settled): come down softly to a
  low level rest when standing is infeasible."""
  cfg = go2_stabilize_env_cfg(play=play)
  _add_constant_load(cfg, W, load_height)
  _add_fr_theta_events(cfg, lo, hi)
  _add_fr_conditioning_obs(cfg)
  _raise_contact_termination(cfg, REST_CONTACT_N)
  return cfg


__all__ = [
  "go2_compound_stand_env_cfg", "go2_compound_rest_env_cfg",
  "stance_margins", "weight_rest_margins",
  "COMPOUND_W", "COMPOUND_H", "STAND_THETA_LO", "STAND_THETA_HI", "REST_THETA_LO", "REST_THETA_HI",
  "REST_CONTACT_N",
]
