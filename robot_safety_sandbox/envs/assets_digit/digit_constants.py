"""Agility Robotics Digit v3 constants.

Vendored from the mjlab fork (MjlabSafety_Digit
``src/mjlab/asset_zoo/robots/digit_v3/digit_constants.py``); the MJCF + meshes
live in the zoo under ``envs/assets_digit/xmls/`` (same pattern as
``envs/assets_go2``). Only stock mjlab APIs are used.
"""

import dataclasses
import re
from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

ZOO_ENVS_PATH = Path(__file__).resolve().parents[1]

DIGIT_XML: Path = ZOO_ENVS_PATH / "assets_digit" / "xmls" / "digit.xml"
assert DIGIT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, DIGIT_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(DIGIT_XML))
  spec.assets = get_assets(spec.meshdir)
  # Make the XML floor plane invisible — the terrain system provides its
  # own ground.  Having both visible causes z-fighting (flickering).
  # The geom is kept (not deleted) so it can be re-activated if needed.
  for g in spec.worldbody.geoms:
    if g.name == "floor":
      g.rgba = (0, 0, 0, 0)
      break
  return spec


# Per-joint overrides applied on top of the all-zero upright pose.
# Both left and right hip_roll share the same world-frame axis, so the same
# sign moves both legs in the same direction.  Tune these values with:
#   uv run python src/mjlab/asset_zoo/robots/digit_v3/digit_constants.py
_UPRIGHT_JOINT_OVERRIDES: dict[str, float] = {
  # Leg joints — match hardware initial pose.
  "left_hip_roll_joint": 0.375,
  "right_hip_roll_joint": -0.375,
  "left_hip_yaw_joint": 0.000,
  "right_hip_yaw_joint": 0.000,
  "left_hip_pitch_joint": 0.311,
  "right_hip_pitch_joint": -0.311,
  "left_knee_joint": 0.344,
  "right_knee_joint": -0.344,
  "left_toe_A_joint": -0.123,
  "right_toe_A_joint": 0.123,
  "left_toe_B_joint": 0.123,
  "right_toe_B_joint": -0.123,
  # Arm joints — match hardware initial pose.
  "left_shoulder_roll_joint": -0.0773,
  "right_shoulder_roll_joint": 0.0773,
  "left_shoulder_pitch_joint": 1.145,
  "right_shoulder_pitch_joint": -1.145,
  "shoulder_yaw_joint_left": 0.0013,
  "shoulder_yaw_joint_right": -0.0013,
  "left_elbow_joint": -0.043,
  "right_elbow_joint": 0.043,
}


def get_spec_upright() -> mujoco.MjSpec:
  """Get Digit spec with all hinge joints zeroed (upright standing pose).

  The Digit V3 MJCF is designed so that zero hinge-joint positions produce a
  symmetric, upright stance.  The Agility 'standing' keyframe uses non-zero
  hip_pitch/knee/tarsus values whose sign conventions differ between left and
  right sides (mirrored joint axes), which causes an asymmetric initial pose.
  Zeroing all hinges avoids that problem entirely.  _UPRIGHT_JOINT_OVERRIDES
  can fine-tune specific joints after zeroing.

  After setting hinge positions, a short physics settle is run (with gravity
  disabled) so the ball joints on the closed-loop linkages (Achilles rods,
  toe rods) find orientations that satisfy the connect constraints.  The
  settled ball-joint quaternions are written back into the keyframe.
  """
  spec = get_spec()
  model = spec.compile()
  key_qpos = list(model.key("standing").qpos)
  for i in range(model.njnt):
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
      key_qpos[model.jnt_qposadr[i]] = 0.0
  for joint_name, angle in _UPRIGHT_JOINT_OVERRIDES.items():
    key_qpos[model.jnt_qposadr[model.joint(joint_name).id]] = angle
  # Initial root height — refined automatically after the settle below so the
  # lowest foot collision geom rests on z=0.
  key_qpos[2] = 1.0

  # Settle closed-loop linkages (Achilles rods, toe rods) with gravity
  # disabled.  The root pose and user-overridden hinges are pinned; linkage
  # joints (ball joints + non-overridden hinges like tarsus, toe, hip_pitch)
  # are free to adjust so the connect constraints are satisfied.
  data = mujoco.MjData(model)
  data.qpos[:] = key_qpos
  data.qvel[:] = 0
  saved_gravity = model.opt.gravity.copy()
  model.opt.gravity[:] = 0
  pinned_addrs: dict[int, float] = {}
  for jn in _UPRIGHT_JOINT_OVERRIDES:
    jid = model.joint(jn).id
    addr = model.jnt_qposadr[jid]
    pinned_addrs[addr] = key_qpos[addr]
  for _ in range(2000):
    mujoco.mj_step(model, data)
    data.qpos[:7] = key_qpos[:7]
    for addr, val in pinned_addrs.items():
      data.qpos[addr] = val
    data.qvel[:] = 0
  model.opt.gravity[:] = saved_gravity

  # Auto-adjust root z so the lowest corner of either foot box sits on z=0.
  mujoco.mj_forward(model, data)
  corners = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=float,
  )
  lowest_z = float("inf")
  for name in ("left_toe_roll_collision", "right_toe_roll_collision"):
    xpos = data.geom(name).xpos
    xmat = data.geom(name).xmat.reshape(3, 3)
    size = model.geom(name).size
    world = xpos + (corners * size) @ xmat.T
    lowest_z = min(lowest_z, float(world[:, 2].min()))
  data.qpos[2] -= lowest_z

  # Write the fully settled qpos (ball joints + linkage hinges) back.
  spec.keys[0].qpos = data.qpos.tolist()
  return spec


##
# Actuator config.
##

# Natural frequency and damping ratio for PD control tuning.
_NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
_DAMPING_RATIO = 2.0

# Armature values from digit.xml joint definitions.
_ARMATURE_A = 0.1728  # hip_roll, shoulder_roll, shoulder_pitch, elbow
_ARMATURE_B = 0.0675  # hip_yaw, shoulder_yaw
_ARMATURE_C = 0.120576  # hip_pitch, knee
_ARMATURE_D = 0.035  # toe_A, toe_B

_STIFFNESS_A = _ARMATURE_A * _NATURAL_FREQ**2
_STIFFNESS_B = _ARMATURE_B * _NATURAL_FREQ**2
_STIFFNESS_C = _ARMATURE_C * _NATURAL_FREQ**2
_STIFFNESS_D = _ARMATURE_D * _NATURAL_FREQ**2

_DAMPING_A = 2.0 * _DAMPING_RATIO * _ARMATURE_A * _NATURAL_FREQ
_DAMPING_B = 2.0 * _DAMPING_RATIO * _ARMATURE_B * _NATURAL_FREQ
_DAMPING_C = 2.0 * _DAMPING_RATIO * _ARMATURE_C * _NATURAL_FREQ
_DAMPING_D = 2.0 * _DAMPING_RATIO * _ARMATURE_D * _NATURAL_FREQ

# Effort limits (Nm) — peak joint torques from the manufacturer MJCF
# (digit-v3-mjcf/digit-v3.xml), computed as gear × max(|ctrlrange|).
_EFFORT_A = (
  113.0  # hip_roll, shoulder_roll, shoulder_pitch, elbow  (gear=80, ctrl=±1.4125)
)
_EFFORT_B = (
  79.2  # hip_yaw, shoulder_yaw                           (gear=50, ctrl=±1.5835)
)
_EFFORT_C_HIP_PITCH = (
  150.0  # hip_pitch (reduced from spec 216.9 Nm to tame motor torque)
)
_EFFORT_C_KNEE = 150.0  # knee      (reduced from spec 231.3 Nm to tame motor torque)
_EFFORT_D = (
  42.0  # toe_A, toe_B                                    (gear=50, ctrl=±0.8395)
)

DIGIT_ACTUATOR_A = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_pitch_joint",
    ".*_elbow_joint",
  ),
  stiffness=_STIFFNESS_A,
  damping=_DAMPING_A,
  effort_limit=_EFFORT_A,
  armature=_ARMATURE_A,
)

DIGIT_ACTUATOR_B = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
    "shoulder_yaw_joint_.*",
  ),
  stiffness=_STIFFNESS_B,
  damping=_DAMPING_B,
  effort_limit=_EFFORT_B,
  armature=_ARMATURE_B,
)

DIGIT_ACTUATOR_C_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=_STIFFNESS_C,
  damping=_DAMPING_C,
  effort_limit=_EFFORT_C_HIP_PITCH,
  armature=_ARMATURE_C,
)

DIGIT_ACTUATOR_C_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=_STIFFNESS_C,
  damping=_DAMPING_C,
  effort_limit=_EFFORT_C_KNEE,
  armature=_ARMATURE_C,
)

DIGIT_ACTUATOR_D = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_toe_A_joint",
    ".*_toe_B_joint",
  ),
  stiffness=_STIFFNESS_D,
  damping=_DAMPING_D,
  effort_limit=_EFFORT_D,
  armature=_ARMATURE_D,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.0366),  # Matches the settled stance height from get_spec_upright().
  joint_pos=None,  # Use 'standing' keyframe from digit.xml (ball joints need qpos format).
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Foot geoms get condim=3 with friction; body geoms use XML defaults (condim=3).
FEET_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_toe_roll_collision$",),
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

DIGIT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    DIGIT_ACTUATOR_A,
    DIGIT_ACTUATOR_B,
    DIGIT_ACTUATOR_C_HIP_PITCH,
    DIGIT_ACTUATOR_C_KNEE,
    DIGIT_ACTUATOR_D,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_digit_robot_cfg() -> EntityCfg:
  """Get a fresh Digit v3 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_COLLISION,),
    spec_fn=get_spec_upright,
    articulation=DIGIT_ARTICULATION,
  )


##
# Sim2sim-calibrated model (see sim2sim/ and report.md).
#
# Parameters identified against the Agility ar-control simulator
# (2023.01.13b) via matched excitation experiments (per-joint chirps,
# steps, multisine; hanging teststand config).  Halves the trajectory
# mismatch on held-out excitations vs the uncalibrated model.
##

# Fitted per joint type: (armature, frictionloss, joint_damping).
# joint_damping == None means keep the XML value.
_CALIBRATED_JOINT_PARAMS: dict[str, tuple[float, float, float | None]] = {
  ".*_hip_roll_joint": (0.5203, 0.3871, None),
  ".*_hip_yaw_joint": (0.1960, 0.4683, None),
  ".*_hip_pitch_joint": (0.4533, 0.3882, None),
  ".*_knee_joint": (0.4113, 0.4583, None),
  ".*_toe_A_joint": (0.0677, 1.6439, 0.22),
  ".*_toe_B_joint": (0.0635, 1.7514, 0.0),
  ".*_shoulder_roll_joint": (0.5781, 0.4422, 2.757),
  ".*_shoulder_pitch_joint": (0.8478, 0.5910, 13.364),
  "shoulder_yaw_joint_.*": (0.2116, 3.6618, None),
  ".*_elbow_joint": (0.5682, 0.4668, None),
}

# Leaf springs from the manufacturer MJCF, absent in digit.xml where the
# shin and heel-spring bodies are welded rigid.  ar-control simulates
# them; without them the foot sole sits ~2.9 deg off the ground at the
# default pose (edge contact).  The added armature keeps the stiff
# explicit spring stable at the 5 ms training timestep by lowering its
# resonance below the integrator limit; static stiffness is unchanged.
_SPRING_JOINTS: dict[str, tuple[str, float, tuple[float, float] | None]] = {
  "left_shin_joint": ("left_shin", 6000.0, None),
  "right_shin_joint": ("right_shin", 6000.0, None),
  "left_heel_spring_joint": ("left_heel_spring", 4375.0, (-0.105, 0.105)),
  "right_heel_spring_joint": ("right_heel_spring", 4375.0, (-0.105, 0.105)),
}
_SPRING_DAMPING = 5.0
_SPRING_ARMATURE = 0.05


# Rigid toe transmission (the `_rigidtoe` model variant, 2026-07-10).
#
# The MJCF's toe pushrods (rod bodies with free BALL joints + connect
# equalities) leak a passive DOF under load: with all 12 leg motors welded, a
# standing robot still rolls its feet 0.44 rad and falls in ~2 s — the rod
# geometry reconfigures under body weight (loaded transmission ratio ~3.4 vs
# free-swing 1.586), which is why a PD statue stands 28 s in ar-control but
# ~2 s in mjlab, and why standing policies trained here fail to transfer while
# walking policies (free-swing regime) transfer cleanly. ar-control treats the
# linkage as a rigid kinematic map. This variant does the same: the rod BALL
# joints are deleted (rods weld to the toe arms; mass/visuals kept) and the
# motor->toe map is enforced directly as fixed-tendon equalities using the
# transmission slopes measured on the ar-control hanging-step data
# (side-symmetric, linear to NRMSE 0.01 over the +-0.15 rad working range):
#
#     toe_pitch = -0.513*toe_A + 0.507*toe_B
#     toe_roll  = +1.594*toe_A + 1.593*toe_B
#
# See sim2sim/README.md findings 5-7 and report.md (2026-07-10).
_RIGID_TOE_RODS = ("toe_A_rod", "toe_B_rod")
# tendon terms: (target joint, [(joint suffix, coef), ...]) with the constraint
# T = target - sum(coef * joint) held at its standing-keyframe value.
_RIGID_TOE_MAP: dict[str, tuple[tuple[str, float], ...]] = {
  "toe_pitch_joint": (("toe_A_joint", -0.513), ("toe_B_joint", 0.507)),
  "toe_roll_joint": (("toe_A_joint", 1.594), ("toe_B_joint", 1.593)),
}
_RIGID_TOE_SOLREF = (0.001, 1.0)
_RIGID_TOE_SOLIMP = (0.99, 0.999, 0.0005)


def apply_rigid_toe_transmission(
  spec: mujoco.MjSpec, key_by_joint: dict[str, float]
) -> None:
  """Replace the toe pushrod mechanism with rigid joint-space couplings.

  Args:
    spec: Digit spec (keyframes must already be dropped or rebuilt after).
    key_by_joint: standing-pose joint values BY NAME (used to anchor the
      linear map's constant so the settled pose is constraint-consistent).
  """
  # 1. Drop the rod ball joints (rods weld to the toe arms) + connect eqs.
  rod_names = {f"{s}_{r}" for s in ("left", "right") for r in _RIGID_TOE_RODS}
  for j in list(spec.joints):
    if j.name and j.name.removesuffix("_joint") in rod_names:
      spec.delete(j)
  for e in list(spec.equalities):
    if e.name in rod_names:
      spec.delete(e)
  # Hide the rod meshes: with the joints + connects gone the rods are welded
  # to the drive arms and their foot-side ends attach to NOTHING — leaving
  # them visible shows disconnected sticks floating beside the feet in eval
  # videos. Mass/inertia are kept (rigidly carried by the arm); only the
  # visuals go. The actual transmission is the fixed-tendon coupling below.
  for bname in rod_names:
    for g in spec.body(bname).geoms:
      g.rgba = (0.0, 0.0, 0.0, 0.0)

  # 2. Enforce the measured motor->toe map via fixed-tendon equalities.
  for side in ("left", "right"):
    for target, terms in _RIGID_TOE_MAP.items():
      tname = f"{side}_{target.removesuffix('_joint')}_coupling"
      tendon = spec.add_tendon()
      tendon.name = tname
      tendon.wrap_joint(f"{side}_{target}", 1.0)
      offset = key_by_joint[f"{side}_{target}"]
      for jsuffix, coef in terms:
        tendon.wrap_joint(f"{side}_{jsuffix}", -coef)
        offset -= coef * key_by_joint[f"{side}_{jsuffix}"]
      eq = spec.add_equality()
      eq.name = tname
      eq.type = mujoco.mjtEq.mjEQ_TENDON
      eq.name1 = tname
      eq.data[0] = offset  # T(q) = T(standing key); tendon length at qpos0 = 0
      eq.solref[:2] = _RIGID_TOE_SOLREF
      eq.solimp[:3] = _RIGID_TOE_SOLIMP


def get_spec_upright_calibrated(
  toe_roll_stiffness: float = 0.0,
  rigid_toe: bool = False,
) -> mujoco.MjSpec:
  """Digit spec with sim2sim-calibrated leaf springs and joint damping.

  Args:
    toe_roll_stiffness: optional effective ankle-roll stiffness (Nm/rad)
      on the passive toe_roll joints.  0 keeps the physically-faithful
      free ankle (matches ar-control hanging response); ~300 makes
      quiet PD standing statically stable (MuJoCo's soft-constraint
      pushrods give only ~24 Nm/rad of ankle-roll authority under load
      vs ar-control's rigid-linkage transmission).
  """
  spec = get_spec_upright()

  # Capture the settled keyframe by joint name before surgery changes nq.
  model_old = spec.compile()
  old_qpos = np.array(model_old.key("standing").qpos)
  saved: dict[str, np.ndarray] = {}
  for jid in range(model_old.njnt):
    name = mujoco.mj_id2name(model_old, mujoco.mjtObj.mjOBJ_JOINT, jid)
    adr = model_old.jnt_qposadr[jid]
    dim = {
      mujoco.mjtJoint.mjJNT_FREE: 7,
      mujoco.mjtJoint.mjJNT_BALL: 4,
    }.get(model_old.jnt_type[jid], 1)
    saved[name or "__free__"] = old_qpos[adr : adr + dim]

  # Drop the keyframe up front: it no longer matches nq after surgery,
  # and spec.delete() reverts uncommitted attribute mutations on parsed
  # elements, so it must happen before any joint edits below.
  for k in list(spec.keys):
    spec.delete(k)

  if rigid_toe:
    key_scalars = {k: float(v[0]) for k, v in saved.items() if len(v) == 1}
    apply_rigid_toe_transmission(spec, key_scalars)

  for jname, (bname, stiffness, jrange) in _SPRING_JOINTS.items():
    body = spec.body(bname)
    j = body.add_joint(name=jname)
    j.type = mujoco.mjtJoint.mjJNT_HINGE
    j.axis[:] = np.array([0.0, 0.0, 1.0])
    j.pos[:] = np.array([0.0, 0.0, 0.0])
    j.stiffness = stiffness
    j.damping = _SPRING_DAMPING
    j.armature = _SPRING_ARMATURE
    if jrange is not None:
      j.limited = True
      j.range[:] = np.array(jrange)

  # Fitted joint damping where it differs from the XML.
  for pattern, (_, _, damping) in _CALIBRATED_JOINT_PARAMS.items():
    if damping is None:
      continue
    for j in spec.joints:
      if j.name and re.fullmatch(pattern, j.name):
        j.damping = damping

  if toe_roll_stiffness > 0.0:
    # NOTE: mutate via spec.joints iteration handles — in mujoco 3.6,
    # edits through spec.joint(name) lookups can fail to persist after
    # other spec surgery in the same pass.
    for j in spec.joints:
      if j.name in ("left_toe_roll_joint", "right_toe_roll_joint"):
        j.stiffness = toe_roll_stiffness
        j.damping = 3.0

  # Rebuild the 'standing' keyframe for the new nq: old joints keep their
  # settled values, the spring joints start at 0 (their unloaded state).
  model_new = spec.compile()
  new_qpos = np.array(model_new.qpos0)
  for jid in range(model_new.njnt):
    name = mujoco.mj_id2name(model_new, mujoco.mjtObj.mjOBJ_JOINT, jid)
    vals = saved.get(name or "__free__")
    if vals is not None:
      adr = model_new.jnt_qposadr[jid]
      new_qpos[adr : adr + len(vals)] = vals
  key = spec.add_key()
  key.name = "standing"
  key.qpos = new_qpos.tolist()
  return spec


def _calibrated_actuator(
  base: BuiltinPositionActuatorCfg, pattern: str
) -> BuiltinPositionActuatorCfg:
  armature, frictionloss, _ = _CALIBRATED_JOINT_PARAMS[pattern]
  return dataclasses.replace(
    base,
    target_names_expr=(pattern,),
    armature=armature,
    frictionloss=frictionloss,
  )


DIGIT_CALIBRATED_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _calibrated_actuator(DIGIT_ACTUATOR_A, ".*_hip_roll_joint"),
    _calibrated_actuator(DIGIT_ACTUATOR_A, ".*_shoulder_roll_joint"),
    _calibrated_actuator(DIGIT_ACTUATOR_A, ".*_shoulder_pitch_joint"),
    _calibrated_actuator(DIGIT_ACTUATOR_A, ".*_elbow_joint"),
    _calibrated_actuator(DIGIT_ACTUATOR_B, ".*_hip_yaw_joint"),
    _calibrated_actuator(DIGIT_ACTUATOR_B, "shoulder_yaw_joint_.*"),
    _calibrated_actuator(DIGIT_ACTUATOR_C_HIP_PITCH, ".*_hip_pitch_joint"),
    _calibrated_actuator(DIGIT_ACTUATOR_C_KNEE, ".*_knee_joint"),
    _calibrated_actuator(DIGIT_ACTUATOR_D, ".*_toe_A_joint"),
    _calibrated_actuator(DIGIT_ACTUATOR_D, ".*_toe_B_joint"),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_digit_robot_cfg_calibrated() -> EntityCfg:
  """Digit config with the sim2sim-calibrated model.

  NOTE: the spring joints add 4 passive DOFs; observation terms that
  enumerate all joints change dimension, so policies trained on the
  uncalibrated model are incompatible (retraining required) and the
  deployment script must include shin/heel-spring positions from the
  LLAPI unactuated-joint array in the observation vector.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_COLLISION,),
    spec_fn=get_spec_upright_calibrated,
    articulation=DIGIT_CALIBRATED_ARTICULATION,
  )


# Effective holding-gain scales for the rigidtoe TRAINING model, fitted on the
# ar-control loaded suite (sim2sim/fit_gain_scales.py, 2026-07-10): kp&kd
# co-scaled per leg joint type until loaded tracking errors match ar-control
# (mjlab's pure-PD joints otherwise CREEP under sustained load — ar's servo
# cascade holds creep-free; see sim2sim/README finding 6). With these scales +
# the rigid toe transmission the mirror survives the FULL 28 s hard sway that
# ar-control survives (vs 1.5 s nominal). Free-swing NRMSE cost: 0.06-0.15 vs
# 0.01-0.025 baseline. Arms stay nominal.
# DEPLOYMENT PARITY: these are TRAINING-ONLY effective gains — exported policy
# metadata must carry the NOMINAL kp/kd (the real robot's servo provides the
# extra holding itself). Export via the *Calibrated* task keeps this automatic.
DIGIT_RIGIDTOE_GAIN_SCALES: dict[str, float] = {
  ".*_hip_roll_joint": 4.10,
  ".*_hip_yaw_joint": 3.75,
  ".*_hip_pitch_joint": 3.88,
  ".*_knee_joint": 4.17,
  ".*_toe_A_joint": 4.19,
  ".*_toe_B_joint": 4.09,
}


def _rigidtoe_actuator(
  base: BuiltinPositionActuatorCfg, pattern: str
) -> BuiltinPositionActuatorCfg:
  act = _calibrated_actuator(base, pattern)
  scale = DIGIT_RIGIDTOE_GAIN_SCALES.get(pattern, 1.0)
  assert act.stiffness is not None and act.damping is not None
  return dataclasses.replace(
    act,
    stiffness=act.stiffness * scale,
    damping=act.damping * scale,
  )


DIGIT_RIGIDTOE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _rigidtoe_actuator(DIGIT_ACTUATOR_A, ".*_hip_roll_joint"),
    _rigidtoe_actuator(DIGIT_ACTUATOR_A, ".*_shoulder_roll_joint"),
    _rigidtoe_actuator(DIGIT_ACTUATOR_A, ".*_shoulder_pitch_joint"),
    _rigidtoe_actuator(DIGIT_ACTUATOR_A, ".*_elbow_joint"),
    _rigidtoe_actuator(DIGIT_ACTUATOR_B, ".*_hip_yaw_joint"),
    _rigidtoe_actuator(DIGIT_ACTUATOR_B, "shoulder_yaw_joint_.*"),
    _rigidtoe_actuator(DIGIT_ACTUATOR_C_HIP_PITCH, ".*_hip_pitch_joint"),
    _rigidtoe_actuator(DIGIT_ACTUATOR_C_KNEE, ".*_knee_joint"),
    _rigidtoe_actuator(DIGIT_ACTUATOR_D, ".*_toe_A_joint"),
    _rigidtoe_actuator(DIGIT_ACTUATOR_D, ".*_toe_B_joint"),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_spec_upright_calibrated_rigidtoe() -> mujoco.MjSpec:
  """Calibrated spec + RIGID toe transmission (`_rigidtoe` variant).

  See ``apply_rigid_toe_transmission``: deletes the pushrod ball joints
  (which leak a passive DOF under load — the root cause of the standing
  sim2sim gap) and enforces the measured motor->toe kinematic map as
  fixed-tendon equalities. The base calibrated model is unchanged; this is
  the opt-in variant for standing/safety training.
  """
  return get_spec_upright_calibrated(rigid_toe=True)


def get_digit_robot_cfg_calibrated_rigidtoe() -> EntityCfg:
  """Digit config: calibrated model + rigid toe transmission + fitted
  effective holding gains (``DIGIT_RIGIDTOE_GAIN_SCALES``).

  NOTE: removing the 4 rod BALL joints changes nq vs the calibrated model
  (rods weld to the toe arms), but the actuated joints, the 10 LLAPI
  unactuated joints, and the leaf springs are unchanged — so
  joint-enumerating OBSERVATIONS keep the calibrated model's dimensions
  and checkpoints remain obs-compatible across the two variants.
  ACTION-SCALE / METADATA PARITY: ``DIGIT_ACTION_SCALE`` is computed from
  the NOMINAL articulation and deployment metadata must keep nominal
  kp/kd — export via the *Calibrated* task, not this one.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_COLLISION,),
    spec_fn=get_spec_upright_calibrated_rigidtoe,
    articulation=DIGIT_RIGIDTOE_ARTICULATION,
  )


DIGIT_ACTION_SCALE: dict[str, float] = {}
for _a in DIGIT_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    DIGIT_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  spec = get_spec_upright()
  # Re-enable the floor for standalone viewing (no terrain system here).
  for g in spec.worldbody.geoms:
    if g.name == "floor":
      g.rgba = (0.5, 0.5, 0.5, 1.0)
      break
  model = spec.compile()
  data = mujoco.MjData(model)
  # Apply the standing keyframe so the viewer shows the tuned initial pose.
  mujoco.mj_resetDataKeyframe(model, data, model.key("standing").id)
  mujoco.mj_forward(model, data)
  print(f"root z (qpos[2])      = {data.qpos[2]:.4f}")
  print(f"torso body z          = {data.body('torso').xpos[2]:.4f}")
  print(f"left  toe_roll geom z = {data.geom('left_toe_roll_collision').xpos[2]:.4f}")
  print(f"right toe_roll geom z = {data.geom('right_toe_roll_collision').xpos[2]:.4f}")
  # launch_passive keeps the pose frozen (physics not stepped) until closed.
  with viewer.launch_passive(model, data) as v:
    v.cam.lookat[:] = [0, 0, 0.8]
    v.cam.distance = 3.0
    print("Showing static standing pose — close window to exit.")
    while v.is_running():
      v.sync()
