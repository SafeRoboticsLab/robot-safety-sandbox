"""Go2-with-payload stabilization vs an adversarial force — the ODD-conditioned demonstration task.

A drop-in variant of ``go2_stabilize``: same flat terrain, zero-command stance target, and
``stance_margins`` (safety = trunk-corner height + stance band — payload-agnostic, so it is reused
verbatim). The ONLY change is the robot: the Go2 carries a sloshy/rigid payload (``assets_go2_payload``).

The payload is the ODD. Its config (rigidity via hinge stiffness × total mass — the two primary axes,
plus n_layers / mass distribution) changes the OPTIMAL stabilization STRATEGY under the pull adversary:
light/rigid → dodge toward the pull (dynamic); heavy/tall/sloshy → brace in place (moving would excite
the slosh and topple). A single blind policy must worst-case; an ODD-conditioned one adapts.

NOTE: this builder pins ONE payload config (DEFAULT_PAYLOAD). Randomizing rigidity × total-mass PER-ENV
(the ODD distribution) + exposing the estimate to the critic is the next layer (mjlab reset events on
the payload joint stiffness / block masses).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from robot_safety_sandbox.envs.assets_go2_payload import get_go2_payload_robot_cfg
from robot_safety_sandbox.envs.go2_stabilize.env_cfg import _pin_twist, stance_margins  # reused
from robot_safety_sandbox.envs.velocity.go2 import unitree_go2_flat_env_cfg

# The 12 Go2 leg joints (payload hinges are named ``payload_j*`` ⇒ excluded by ``_joint`` suffix).
_LEG_JOINTS = "^(FL|FR|RL|RR)_.*_joint$"

# ── THE ODD distribution (per-env, sampled at startup) — rigidity × total-mass ──────────────────
RIGIDITY_RANGE = (0.0, 300.0)     # payload hinge stiffness: 0 = water-like slosh … 300 ≈ rigid box
MASS_SCALE_RANGE = (0.4, 2.5)     # × DEFAULT_PAYLOAD total_mass (3 kg) ⇒ ~[1.2, 7.5] kg


def _add_odd_events(cfg: ManagerBasedRlEnvCfg) -> None:
  """Randomize the payload ODD (rigidity × total-mass) PER-ENV at startup — one θ per env, so the
  parallel envs sample the ODD distribution. Read back live from model.jnt_stiffness / body_mass."""
  cfg.events["payload_rigidity"] = EventTermCfg(   # RIGIDITY axis (hinge stiffness, shared per env)
    func=dr.joint_stiffness, mode="startup",
    params={"asset_cfg": SceneEntityCfg("robot", joint_names="payload_j.*"),
            "ranges": RIGIDITY_RANGE, "operation": "abs", "shared_random": True})
  cfg.events["payload_mass"] = EventTermCfg(        # TOTAL-MASS axis (one scale per env, all blocks)
    func=dr.body_mass, mode="startup",
    params={"asset_cfg": SceneEntityCfg("robot", body_names="payload_.*"),
            "ranges": MASS_SCALE_RANGE, "operation": "scale", "shared_random": True})


def _scope_joint_rewards_to_legs(cfg: ManagerBasedRlEnvCfg) -> None:
  """The inherited velocity dense-reward terms (``pose``, ``stand_still``) default to ``joint_names='.*'``
  = ALL robot joints; the payload's 4 passive hinges then break the per-joint std shapes. Re-scope any
  such all-joints robot term to the 12 leg joints. (Payload joint state can be surfaced separately as
  an obs term for conditioning — it is not a control joint.)"""
  for term in cfg.rewards.values():
    ac = (getattr(term, "params", None) or {}).get("asset_cfg")
    if isinstance(ac, SceneEntityCfg) and ac.name == "robot" and ac.joint_names == ".*":
      ac.joint_names = _LEG_JOINTS

# Default ODD operating point (rigidity × total-mass). A mid rigidity + moderate mass so the base task
# loads and stands; the ODD sweep varies stiffness ∈ [0, ~300] and total_mass ∈ [~0.5, ~6].
DEFAULT_PAYLOAD = dict(n_layers=4, total_mass=3.0, stiffness=20.0, damping=0.05, profile="uniform")

__all__ = ["go2_payload_stabilize_env_cfg", "stance_margins", "DEFAULT_PAYLOAD"]


def go2_payload_stabilize_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_flat_env_cfg(play=play)
  # swap the base Go2 for the Go2+payload (same base_link / feet / leg joints ⇒ sensors & margins hold)
  cfg.scene.entities["robot"] = get_go2_payload_robot_cfg(**DEFAULT_PAYLOAD)
  _scope_joint_rewards_to_legs(cfg)   # keep dense joint rewards on the 12 legs, not the payload hinges
  _add_odd_events(cfg)                 # per-env ODD: rigidity × total-mass
  _pin_twist(cfg, 0.0)   # zero command: the target is a stable stand
  return cfg
