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

# OUT-OF-DISTRIBUTION stress points (OUTSIDE training: mass > 7.5 kg, stiffness > 300) for the OOD-
# generalization test: does the conditioned policy, given the EXTRAPOLATED theta (payload_odd reads the
# OOD model -> theta > 1), survive OOD payloads better than the blind policy (which gets no theta)?
OOD_HEAVY_SLOSHY = dict(n_layers=4, total_mass=12.0, stiffness=0.0, damping=0.05, profile="top_heavy")
OOD_HEAVY_RIGID = dict(n_layers=4, total_mass=12.0, stiffness=400.0, damping=0.05, profile="top_heavy")


# ── SOFT-DESCENT / lie-down REACH-AVOID margins ───────────────────────────────────────────────────
# A FALLBACK skill (not the stabilize adversary game): from standing, the Go2 lowers itself to a soft
# belly-down rest. TARGET l = a low, level, slow pose; SAFE-SET g = keep the non-foot ground contact
# GENTLE (a settle registers ~60-70 N, a slam ~4700 N — so F_SAFE=200 N cleanly separates soft from
# hard) AND stay off the robot's side/back (up > -0.3). Payload-agnostic; reads the base state
# (like stance_margins) plus the inherited ``nonfoot_ground_touch`` contact sensor.
F_SAFE, F_SCALE = 200.0, 200.0     # settle ~60-70N safe, slam ~4700N unsafe -> 200N separates
H_LIE, HS       = 0.14, 0.06       # target base height for the lie-down
V_TOL, VS       = 0.30, 0.30       # linear speed tolerance at rest
W_TOL, WS       = 0.60, 0.60       # angular speed tolerance at rest
UP_MIN, US      = 0.50, 0.50       # must stay level-ish (belly-down, not on side/inverted)


def descent_margins(env):
  """(g, l): soft-contact + level safety, low/slow/level lie-down target. Mirrors ``stance_margins``'s
  base-state API and reads the ``nonfoot_ground_touch`` contact sensor for the contact-force safe set."""
  d = env.scene["robot"].data
  s = env.scene["nonfoot_ground_touch"]
  fh = s.data.force_history if s.data.force_history is not None else s.data.force
  force = torch.norm(fh, dim=-1).flatten(1).amax(1)          # per-env max non-foot ground force
  up = -d.projected_gravity_b[:, 2]                          # 1 = upright/level, 0 = on side, -1 = inverted
  base_z = d.root_link_pos_w[:, 2]
  v_b = torch.linalg.norm(d.root_link_lin_vel_b, dim=1)
  w_b = torch.linalg.norm(d.root_link_ang_vel_b, dim=1)
  # SAFE: contact force <= F_SAFE AND not tipped past its side (up > -0.3).
  g = torch.minimum((F_SAFE - force) / F_SCALE, (up + 0.3) / 1.0)
  # TARGET: base low, linear + angular speed small, still belly-down level.
  l = torch.minimum(
    torch.minimum((H_LIE - base_z) / HS, (V_TOL - v_b) / VS),
    torch.minimum((W_TOL - w_b) / WS, (up - UP_MIN) / US))
  return g, l


__all__ = ["go2_payload_stabilize_env_cfg", "go2_payload_light_rigid_env_cfg",
           "go2_payload_heavy_sloshy_env_cfg", "go2_payload_conditioned_env_cfg",
           "go2_payload_blind_env_cfg", "go2_payload_conditioned_light_rigid_env_cfg",
           "go2_payload_conditioned_heavy_sloshy_env_cfg", "go2_payload_descent_env_cfg",
           "stance_margins", "descent_margins", "DEFAULT_PAYLOAD"]


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


def go2_payload_descent_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """SOFT-DESCENT / lie-down FALLBACK: from standing, lower to a soft belly-down rest. Reuses the BLIND
  payload cfg (ODD rigidity × total-mass randomized PER-EPISODE and HIDDEN — blind 47-dim obs) so the
  descent is robust to the payload WITHOUT seeing it. Identical env to go2_payload_blind (zero-command
  stance base + hidden ODD); only the reach-avoid MARGINS differ (``descent_margins`` — soft-contact
  safe set + low/slow/level target — vs ``stance_margins``). The ``nonfoot_ground_touch`` contact
  sensor that ``descent_margins`` reads is inherited from the go2 velocity base and preserved here."""
  return go2_payload_blind_env_cfg(play=play)


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


# OOD-generalization eval envs: fixed OOD physics; conditioned variants add the θ obs (which yields the
# extrapolated θ>1 read from the OOD model). Conditioned evals on the *_conditioned_* envs, blind on the
# plain OOD envs.
def go2_payload_conditioned_ood_sloshy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _go2_payload_env_cfg(play, OOD_HEAVY_SLOSHY, randomize_odd=False)
  _add_odd_conditioning_obs(cfg)
  return cfg


def go2_payload_ood_sloshy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_payload_env_cfg(play, OOD_HEAVY_SLOSHY, randomize_odd=False)


def go2_payload_conditioned_ood_rigid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _go2_payload_env_cfg(play, OOD_HEAVY_RIGID, randomize_odd=False)
  _add_odd_conditioning_obs(cfg)
  return cfg


def go2_payload_ood_rigid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _go2_payload_env_cfg(play, OOD_HEAVY_RIGID, randomize_odd=False)


# ── E017 HISTORY / introspective arm ────────────────────────────────────────────────────────────
# The policy sees a FRAME-STACKED history of proprio+action (no theta) and must INFER the payload from
# the dynamics it has felt — the deployable "introspective" alternative to a raw theta-input (which was
# OOD-fragile: an extrapolated theta induced the wrong strategy). The stacked history is an IMPLICIT,
# bounded, strategy-relevant embedding (RMA-style): the MLP encodes K past frames into its hidden state.
HISTORY_K = 16   # frames of history (0.32 s @ 50 Hz); captures a chunk of the payload's slosh response


def _set_obs_history(cfg: ManagerBasedRlEnvCfg, k: int) -> None:
  for group in ("actor", "critic"):
    g = cfg.observations.get(group)
    if g is not None:
      g.history_length = k


def go2_payload_history_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """HISTORY arm (train): ODD randomized per-episode, NO theta, but a K-frame proprio+action history."""
  cfg = _go2_payload_env_cfg(play, DEFAULT_PAYLOAD, randomize_odd=True)
  _set_obs_history(cfg, HISTORY_K)
  return cfg


def go2_payload_history_light_rigid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _go2_payload_env_cfg(play, LIGHT_RIGID, randomize_odd=False); _set_obs_history(cfg, HISTORY_K); return cfg


def go2_payload_history_heavy_sloshy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _go2_payload_env_cfg(play, HEAVY_SLOSHY, randomize_odd=False); _set_obs_history(cfg, HISTORY_K); return cfg


def go2_payload_history_ood_sloshy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _go2_payload_env_cfg(play, OOD_HEAVY_SLOSHY, randomize_odd=False); _set_obs_history(cfg, HISTORY_K); return cfg


def go2_payload_history_ood_rigid_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _go2_payload_env_cfg(play, OOD_HEAVY_RIGID, randomize_odd=False); _set_obs_history(cfg, HISTORY_K); return cfg


# ── E019 BREAK-BOUNDARY sweep ────────────────────────────────────────────────────────────────────
# Parameterized fixed-ODD eval env for the mass/rigidity sweep: find where the trained arms FALL. Built
# at the target mass (model + inertia consistent — no runtime-override blowup). obs selects the arm's
# input surface (blind 47 / conditioned +theta 49 / history stacked 752).
def sweep_env_cfg(play: bool = False, mass: float = 12.0, stiffness: float = 0.0,
                  profile: str = "top_heavy", obs: str = "blind") -> ManagerBasedRlEnvCfg:
  payload = dict(n_layers=4, total_mass=mass, stiffness=stiffness, damping=0.05, profile=profile)
  cfg = _go2_payload_env_cfg(play, payload, randomize_odd=False)
  if obs == "conditioned":
    _add_odd_conditioning_obs(cfg)
  elif obs == "history":
    _set_obs_history(cfg, HISTORY_K)
  return cfg
