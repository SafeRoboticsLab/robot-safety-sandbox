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

# Fixed-ODD SPECIALIST extremes for the bifurcation check (professor's cheap gate): does the optimal
# strategy actually flip dodge↔brace across the ODD? Light+rigid ≈ a normal Go2 (dodge toward the pull);
# heavy+sloshy should force a brace-in-place (moving would excite the slosh and topple).
LIGHT_RIGID = dict(n_layers=4, total_mass=1.2, stiffness=300.0, damping=0.05, profile="uniform")
HEAVY_SLOSHY = dict(n_layers=4, total_mass=7.0, stiffness=0.0, damping=0.05, profile="top_heavy")

__all__ = ["go2_payload_stabilize_env_cfg", "go2_payload_light_rigid_env_cfg",
           "go2_payload_heavy_sloshy_env_cfg", "stance_margins", "DEFAULT_PAYLOAD"]


def _go2_payload_env_cfg(play: bool, payload: dict, randomize_odd: bool) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_flat_env_cfg(play=play)
  # swap the base Go2 for the Go2+payload (same base_link / feet / leg joints ⇒ sensors & margins hold)
  cfg.scene.entities["robot"] = get_go2_payload_robot_cfg(**payload)
  _scope_joint_rewards_to_legs(cfg)   # keep dense joint rewards on the 12 legs, not the payload hinges
  if randomize_odd:
    _add_odd_events(cfg)              # per-env ODD: rigidity × total-mass
  # The payload's 4 limited hinges add limit-constraints; base go2 njmax=300 overflows (~400 peak) and
  # mujoco-warp silently DROPS the excess. Raise the per-world constraint budget with margin.
  cfg.sim.njmax = 600
  # STIFF-SPRING STABILITY: implicitfast integrates hinge STIFFNESS explicitly, so at the base go2
  # dt=0.005 any payload stiffness ≳100 blows up (ω>2/dt), kicking the robot into instant termination
  # (measured: k=300 -> |qvel|~2400, diverges). The `implicit` integrator does NOT help (springs are
  # position-, not velocity-, dependent). Halve the physics dt: k=300 is stable at dt=0.002 with ~2.3x
  # margin across the whole ODD, and — unlike armature — it preserves the slosh dynamics exactly.
  cfg.sim.mujoco.timestep = 0.002
  _pin_twist(cfg, 0.0)   # zero command: the target is a stable stand
  return cfg


def go2_payload_stabilize_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The ODD-conditioned task: payload rigidity × total-mass randomized per-env."""
  return _go2_payload_env_cfg(play, DEFAULT_PAYLOAD, randomize_odd=True)


def go2_payload_light_rigid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Specialist: light + rigid payload (≈ normal Go2). Bifurcation-check extreme."""
  return _go2_payload_env_cfg(play, LIGHT_RIGID, randomize_odd=False)


def go2_payload_heavy_sloshy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Specialist: heavy + sloshy top-heavy payload. Bifurcation-check extreme."""
  return _go2_payload_env_cfg(play, HEAVY_SLOSHY, randomize_odd=False)
