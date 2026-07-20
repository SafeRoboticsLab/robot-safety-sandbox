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

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from robot_safety_sandbox.envs.assets_go2_payload import get_go2_payload_robot_cfg
from robot_safety_sandbox.envs.go2_stabilize.env_cfg import _pin_twist, stance_margins  # reused
from robot_safety_sandbox.envs.velocity.go2 import unitree_go2_flat_env_cfg

# The 12 Go2 leg joints (payload hinges are named ``payload_j*`` ⇒ excluded by ``_joint`` suffix).
_LEG_JOINTS = "^(FL|FR|RL|RR)_.*_joint$"

# ── THE ODD distribution (per-env, sampled at startup) — rigidity × total-mass ──────────────────
RIGIDITY_RANGE = (0.0, 300.0)     # payload hinge stiffness: 0 = water-like slosh … 300 ≈ rigid box
MASS_SCALE_RANGE = (0.4, 2.5)     # × DEFAULT_PAYLOAD total_mass (3 kg) ⇒ ~[1.2, 7.5] kg

# ── ODD-CONDITIONING observation (the oracle θ signal) ──────────────────────────────────────────
# θ = (rigidity, total-mass), the two randomized ODD axes, normalized to [-1, 1] per-env. Read LIVE
# from the (per-env) model that the startup events wrote: payload-hinge ``jnt_stiffness`` and summed
# payload ``body_mass``. The CONDITIONED policy sees θ; the BLIND policy does not (identical env
# otherwise) — the E008c-style A/B on Go2. Normalization spans match the randomization ranges so a
# fully-slosh light payload → ≈-1 and a rigid heavy one → ≈+1 on each axis.
_STIFF_LO, _STIFF_HI = RIGIDITY_RANGE                          # rigidity axis span
_MASS_LO, _MASS_HI = 3.0 * MASS_SCALE_RANGE[0], 3.0 * MASS_SCALE_RANGE[1]  # total-mass span ~[1.2,7.5]


def _resolve_payload_odd_idx(env) -> tuple[torch.Tensor, torch.Tensor]:
  """Global model column indices of the payload hinge joints and payload bodies, by NAME (robust to
  model layout). Payload joints ``robot/payload_j{1..4}``; bodies ``robot/payload_mount`` + ``_b{1..4}``
  (block_0's mass rides on the mount body)."""
  m = env.sim.mj_model
  jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
  bnames = ["robot/payload_mount"] + [f"robot/payload_b{i}" for i in range(1, 5)]
  bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in bnames]
  dev = env.device
  return (torch.tensor([j for j in jids if j >= 0], device=dev, dtype=torch.long),
          torch.tensor([b for b in bids if b >= 0], device=dev, dtype=torch.long))


def payload_odd(env) -> torch.Tensor:
  """Per-env ODD θ=(rigidity, total-mass) normalized to [-1,1]. Shape (num_envs, 2)."""
  if getattr(env, "_payload_odd_jidx", None) is None:
    env._payload_odd_jidx, env._payload_odd_bidx = _resolve_payload_odd_idx(env)
  stiff = env.sim.model.jnt_stiffness[:, env._payload_odd_jidx].mean(dim=1)   # hinge spring = rigidity
  mass = env.sim.model.body_mass[:, env._payload_odd_bidx].sum(dim=1)          # total payload mass
  s = 2.0 * (stiff - _STIFF_LO) / (_STIFF_HI - _STIFF_LO) - 1.0
  m = 2.0 * (mass - _MASS_LO) / (_MASS_HI - _MASS_LO) - 1.0
  return torch.stack([s, m], dim=1)


def _add_odd_conditioning_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Expose θ to BOTH the actor (so the policy adapts strategy per-ODD) and the critic (so the value
  is θ-correct). Separate cfg instances per group (the manager owns per-term state)."""
  for group in ("actor", "critic"):
    cfg.observations[group].terms["payload_odd"] = ObservationTermCfg(func=payload_odd)


def _add_odd_events(cfg: ManagerBasedRlEnvCfg) -> None:
  """Randomize the payload ODD (rigidity × total-mass) PER-EPISODE at reset — a FRESH θ each episode,
  so training sees a continuum of the ODD distribution (not just num_envs fixed points), and the
  conditioned policy is pressured to interpolate across θ. (Professor rec: θ committed per episode.)
  Both ops read from the DEFAULT model field each reset (abs sets, scale multiplies the default — no
  compounding drift; verified in dr/_core.py). Read back live from model.jnt_stiffness / body_mass."""
  cfg.events["payload_rigidity"] = EventTermCfg(   # RIGIDITY axis (hinge stiffness, shared per env)
    func=dr.joint_stiffness, mode="reset",
    params={"asset_cfg": SceneEntityCfg("robot", joint_names="payload_j.*"),
            "ranges": RIGIDITY_RANGE, "operation": "abs", "shared_random": True})
  cfg.events["payload_mass"] = EventTermCfg(        # TOTAL-MASS axis (one scale of DEFAULT per episode)
    func=dr.body_mass, mode="reset",
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


def _scope_joint_obs_to_legs(cfg: ManagerBasedRlEnvCfg) -> None:
  """CRITICAL for the hidden-ODD premise: the inherited ``joint_pos``/``joint_vel`` obs terms default to
  ALL robot joints, which would expose the 4 payload HINGE angles+velocities (8 dims) — i.e. the policy
  would DIRECTLY SENSE the slosh state it is supposed to be blind to (no real sloshy payload has hinge
  encoders). Re-scope the joint obs (actor AND critic) to the 12 legs, so the payload is observable ONLY
  through its EFFECT on the base (IMU / projected-gravity) + leg loading — never measured directly. This
  is what makes the ODD a genuinely hidden parameter (and theta genuinely informative, not redundant)."""
  for group in ("actor", "critic"):
    grp = cfg.observations.get(group)
    if grp is None:
      continue
    for name in ("joint_pos", "joint_vel"):
      term = grp.terms.get(name)
      if term is None:
        continue
      p = dict(getattr(term, "params", None) or {})
      p["asset_cfg"] = SceneEntityCfg("robot", joint_names=_LEG_JOINTS)
      term.params = p

# Default ODD operating point (rigidity × total-mass). A mid rigidity + moderate mass so the base task
# loads and stands; the ODD sweep varies stiffness ∈ [0, ~300] and total_mass ∈ [~0.5, ~6].
DEFAULT_PAYLOAD = dict(n_layers=4, total_mass=3.0, stiffness=20.0, damping=0.05, profile="uniform")

# Fixed-ODD SPECIALIST extremes for the bifurcation check (professor's cheap gate): does the optimal
# strategy actually flip dodge↔brace across the ODD? Light+rigid ≈ a normal Go2 (dodge toward the pull);
# heavy+sloshy should force a brace-in-place (moving would excite the slosh and topple).
LIGHT_RIGID = dict(n_layers=4, total_mass=1.2, stiffness=300.0, damping=0.05, profile="uniform")
HEAVY_SLOSHY = dict(n_layers=4, total_mass=7.0, stiffness=0.0, damping=0.05, profile="top_heavy")

__all__ = ["go2_payload_stabilize_env_cfg", "go2_payload_light_rigid_env_cfg",
           "go2_payload_heavy_sloshy_env_cfg", "go2_payload_conditioned_env_cfg",
           "go2_payload_blind_env_cfg", "go2_payload_conditioned_light_rigid_env_cfg",
           "go2_payload_conditioned_heavy_sloshy_env_cfg", "stance_margins", "DEFAULT_PAYLOAD"]


def _go2_payload_env_cfg(play: bool, payload: dict, randomize_odd: bool) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_flat_env_cfg(play=play)
  # swap the base Go2 for the Go2+payload (same base_link / feet / leg joints ⇒ sensors & margins hold)
  cfg.scene.entities["robot"] = get_go2_payload_robot_cfg(**payload)
  _scope_joint_rewards_to_legs(cfg)   # keep dense joint rewards on the 12 legs, not the payload hinges
  _scope_joint_obs_to_legs(cfg)       # payload is HIDDEN: no direct hinge-state obs, only its base effect
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
  # CRUCIAL: raise decimation in lockstep so the ENV step stays 0.02 s (50 Hz control, as base go2).
  # Otherwise env_dt = 0.002×4 = 0.008 (125 Hz) — mistunes the velocity task (gait/action-rate/horizon)
  # and plays eval videos ~2.5× slow. 0.002 × 10 = 0.02 keeps the control rate identical to base go2.
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10
  _pin_twist(cfg, 0.0)   # zero command: the target is a stable stand
  return cfg


def go2_payload_stabilize_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The ODD-conditioned task: payload rigidity × total-mass randomized per-env."""
  return _go2_payload_env_cfg(play, DEFAULT_PAYLOAD, randomize_odd=True)


def go2_payload_conditioned_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """CONDITIONED arm: ODD randomized per-env AND θ=(rigidity, total-mass) exposed to actor+critic —
  one policy that can adapt strategy per-ODD (the oracle upper bound of the E008c A/B)."""
  cfg = _go2_payload_env_cfg(play, DEFAULT_PAYLOAD, randomize_odd=True)
  _add_odd_conditioning_obs(cfg)
  return cfg


def go2_payload_blind_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """BLIND arm: ODD randomized per-env, θ NOT exposed (identical to go2_payload_stabilize) — one
  policy that must worst-case across the ODD. The control baseline for the conditioned arm."""
  return _go2_payload_env_cfg(play, DEFAULT_PAYLOAD, randomize_odd=True)


def go2_payload_light_rigid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Specialist: light + rigid payload (≈ normal Go2). Bifurcation-check extreme."""
  return _go2_payload_env_cfg(play, LIGHT_RIGID, randomize_odd=False)


def go2_payload_heavy_sloshy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Specialist: heavy + sloshy top-heavy payload. Bifurcation-check extreme."""
  return _go2_payload_env_cfg(play, HEAVY_SLOSHY, randomize_odd=False)


# EVAL-ONLY fixed-θ envs for the CONDITIONED policy (its obs is 57-dim): specialist physics + the θ
# obs term, so θ is read live from the fixed payload and matches what the policy saw for that ODD.
# The conditioned-vs-blind read-out evals: conditioned on these two, blind on the 55-dim specialists.
def go2_payload_conditioned_light_rigid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Conditioned policy at FIXED light-rigid θ (specialist physics + θ obs)."""
  cfg = _go2_payload_env_cfg(play, LIGHT_RIGID, randomize_odd=False)
  _add_odd_conditioning_obs(cfg)
  return cfg


def go2_payload_conditioned_heavy_sloshy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Conditioned policy at FIXED heavy-sloshy θ (specialist physics + θ obs)."""
  cfg = _go2_payload_env_cfg(play, HEAVY_SLOSHY, randomize_odd=False)
  _add_odd_conditioning_obs(cfg)
  return cfg
