"""Digit v3 specs with an open-top box.

Two variants are provided:

1. ``get_spec_upright_with_box`` — box rigidly welded to both arms
   (original, used for locomotion tasks).

2. ``get_spec_upright_with_box_on_arms`` — box as a free body resting
   on wide forearm platforms.  The box has 6-DOF joints (3 slides +
   3 hinges) relative to the torso, so it can translate and rotate
   freely.  Contact between the box and two flat platform geoms on
   the forearms is the only thing holding it up — both arms must
   stay extended or the box falls.
"""

from typing import Callable

import mujoco
import numpy as np

from safe_mjlab_zoo.envs.assets_digit.digit_constants import (
  _UPRIGHT_JOINT_OVERRIDES,
  get_spec_upright,
  get_spec_upright_calibrated,
  get_spec_upright_calibrated_rigidtoe,
)

# Inner cavity half-dimensions (metres).
_INNER_HX = 0.20
_INNER_HY = 0.15
_WALL_HEIGHT = 0.3

_WALL_HALF_THICKNESS = (
  0.03  # 6 cm total — must exceed elbow radius + margin to prevent penetration
)
_BOX_MASS = 5.0  # kg — total mass spread across panels
_BOX_RGBA = np.array([0.6, 0.4, 0.2, 1.0], dtype=np.float32)

# Hand endpoint in each elbow body frame (same geometry left and right).
_HAND_LOCAL = np.array([0.37937, 0.0, -0.061912])

# Each panel: (name, half-sizes, local offset from box body origin).
_PANELS: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = [
  # Bottom
  (
    "bottom",
    (_INNER_HX, _INNER_HY, _WALL_HALF_THICKNESS),
    (0.0, 0.0, 0.0),
  ),
  # +X wall (front)
  (
    "wall_px",
    (_WALL_HALF_THICKNESS, _INNER_HY, _WALL_HEIGHT / 2),
    (_INNER_HX, 0.0, _WALL_HEIGHT / 2),
  ),
  # -X wall (back)
  (
    "wall_nx",
    (_WALL_HALF_THICKNESS, _INNER_HY, _WALL_HEIGHT / 2),
    (-_INNER_HX, 0.0, _WALL_HEIGHT / 2),
  ),
  # +Y wall (left arm side)
  (
    "wall_py",
    (_INNER_HX, _WALL_HALF_THICKNESS, _WALL_HEIGHT / 2),
    (0.0, _INNER_HY, _WALL_HEIGHT / 2),
  ),
  # -Y wall (right arm side)
  (
    "wall_ny",
    (_INNER_HX, _WALL_HALF_THICKNESS, _WALL_HEIGHT / 2),
    (0.0, -_INNER_HY, _WALL_HEIGHT / 2),
  ),
]

_PANEL_MASS = _BOX_MASS / len(_PANELS)

_SQRT2_2 = np.sqrt(2.0) / 2.0
# 90° rotation around local X: open top (+Z local) faces upward in world.
_BOX_QUAT = np.array([_SQRT2_2, _SQRT2_2, 0.0, 0.0])

# ── Arm joint overrides for tray pose ───────────────────────────────
# Both forearms extended forward and roughly level, creating a "tray"
# surface for the box to rest on.
_TRAY_ARM_OVERRIDES: dict[str, float] = {
  "left_shoulder_roll_joint": 0.0,
  "right_shoulder_roll_joint": 0.0,
  "left_shoulder_pitch_joint": 0.0,
  "right_shoulder_pitch_joint": 0.0,
  "shoulder_yaw_joint_left": 0.0,
  "shoulder_yaw_joint_right": 0.0,
  "left_elbow_joint": 0.0,
  "right_elbow_joint": 0.0,
}

# ── Platform geom parameters ───────────────────────────────────────
# Flat, invisible platforms attached to each forearm body.  These are
# much wider than the thin forearm cylinders (∅ 2.6 cm) so the box
# has a real surface to rest on.
_PLATFORM_HALF_LENGTH = (
  0.14  # along the forearm axis (must fit inside ±0.17 clear space)
)
_PLATFORM_HALF_WIDTH = 0.04  # lateral extent (must fit inside ±0.12 clear space)
_PLATFORM_HALF_THICKNESS = 0.01  # 2 cm thick slab


def _find_body(root: mujoco.MjsBody, name: str) -> mujoco.MjsBody | None:
  if root.name == name:
    return root
  child = root.first_body()
  while child:
    result = _find_body(child, name)
    if result is not None:
      return result
    child = root.next_body(child)
  return None


def _name_all_geoms(spec: mujoco.MjSpec) -> None:
  """Give unique names to every unnamed geom in the spec.

  The entity system's ``CollisionCfg`` looks up geoms by name, so
  unnamed geoms (``g.name == ""``) are invisible to
  ``disable_other_geoms``.  After this call, every geom has a name
  of the form ``{body_name}_geom_{index}`` and the entity system can
  properly enable/disable them all.
  """
  # Build a map of body-name → count for generating unique names.
  counts: dict[str, int] = {}
  for g in spec.geoms:
    if g.name:
      continue
    # Walk up to find the parent body.
    parent = g.parent
    body_name = parent.name if parent and parent.name else "world"
    idx = counts.get(body_name, 0)
    g.name = f"{body_name}_geom_{idx}"
    counts[body_name] = idx + 1


def _add_box_geoms(
  box_body: mujoco.MjsBody,
  *,
  enable_collision: bool,
) -> None:
  """Add panel geoms to the box body.

  Args:
    box_body: The MjsBody to add geoms to.
    enable_collision: If True, set contype/conaffinity=1 so the box
      collides with platforms and ground.  If False, 0/0.
  """
  ct = 1 if enable_collision else 0
  ca = 1 if enable_collision else 0
  for name, size, pos in _PANELS:
    geom = box_body.add_geom()
    geom.name = f"box_{name}"
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = np.array(size)
    geom.pos = np.array(pos)
    geom.rgba = _BOX_RGBA
    geom.mass = _PANEL_MASS
    geom.condim = 4
    geom.friction = (1.0, 0.005, 0.0001)
    geom.contype = ct
    geom.conaffinity = ca
    # Arm-through-wall penetration is prevented by ±90° shoulder
    # pitch limits, so default contact settings are sufficient.
    # Non-zero margins or stiff solref launch the box off the
    # platforms at initialisation.


# ────────────────────────────────────────────────────────────────────
# Variant 1: rigid weld (original)
# ────────────────────────────────────────────────────────────────────


def get_spec_upright_with_box() -> mujoco.MjSpec:
  """Digit v3 upright spec with an open-top box held by both arms."""
  spec = get_spec_upright()

  # Compile once to locate both hand endpoints.
  tmp_model = spec.compile()
  tmp_data = mujoco.MjData(tmp_model)
  mujoco.mj_resetDataKeyframe(tmp_model, tmp_data, tmp_model.key("standing").id)
  mujoco.mj_forward(tmp_model, tmp_data)

  lid = tmp_model.body("left_elbow").id
  rid = tmp_model.body("right_elbow").id
  l_rot = tmp_data.xmat[lid].reshape(3, 3)
  l_pos = tmp_data.xpos[lid]
  r_rot = tmp_data.xmat[rid].reshape(3, 3)
  r_pos = tmp_data.xpos[rid]

  lhand_world = l_pos + l_rot @ _HAND_LOCAL
  rhand_world = r_pos + r_rot @ _HAND_LOCAL
  midpoint_world = (lhand_world + rhand_world) / 2

  # Box body origin at the midpoint, in left_elbow frame.
  box_pos_in_elbow = l_rot.T @ (midpoint_world - l_pos)

  elbow = _find_body(spec.worldbody, "left_elbow")
  assert elbow is not None, "left_elbow body not found in spec"

  box_body = elbow.add_body()
  box_body.name = "box_load"
  box_body.pos = box_pos_in_elbow
  box_body.quat = _BOX_QUAT

  _add_box_geoms(box_body, enable_collision=False)

  # Weld the right hand to the right side wall of the box.
  # Recompile with the box to get exact world-frame transforms.
  tmp2 = spec.compile()
  d2 = mujoco.MjData(tmp2)
  mujoco.mj_resetDataKeyframe(tmp2, d2, tmp2.key("standing").id)
  mujoco.mj_forward(tmp2, d2)

  box_id = tmp2.body("box_load").id
  relbow_id = tmp2.body("right_elbow").id
  box_rot = d2.xmat[box_id].reshape(3, 3)
  box_wpos = d2.xpos[box_id]
  relbow_rot = d2.xmat[relbow_id].reshape(3, 3)
  relbow_pos = d2.xpos[relbow_id]

  anchor = box_rot.T @ (relbow_pos - box_wpos)

  rel_rot = box_rot.T @ relbow_rot
  rel_quat = np.zeros(4)
  mujoco.mju_mat2Quat(rel_quat, rel_rot.flatten())

  b2_y_in_b1 = box_rot.T @ (relbow_rot @ np.array([0.0, 1.0, 0.0]))

  weld = spec.add_equality()
  weld.type = mujoco.mjtEq.mjEQ_WELD
  weld.name = "box_right_hand_weld"
  weld.name1 = "box_load"
  weld.name2 = "right_elbow"
  weld.objtype = mujoco.mjtObj.mjOBJ_BODY
  weld.data[0:3] = np.array([0.0, 1.0, 0.0])
  weld.data[3:6] = anchor + b2_y_in_b1
  weld.data[6:10] = rel_quat
  weld.data[10] = 1.0

  return spec


# ────────────────────────────────────────────────────────────────────
# Variant 2: free box on forearm platforms
# ────────────────────────────────────────────────────────────────────


def get_spec_upright_with_box_on_arms(
  base_spec_fn: Callable[[], mujoco.MjSpec] = get_spec_upright,
) -> mujoco.MjSpec:
  """Digit v3 spec with a free box resting on forearm platforms.

  The robot's arms are set to a "tray" pose (forearms extended forward
  and level).  The box is parented to ``torso`` with 6-DOF joints
  (3 slides + 3 hinges) so it can translate and rotate freely.

  Two flat, invisible platform geoms (``left_forearm_platform`` and
  ``right_forearm_platform``) are attached to each elbow body.  These
  are much wider than the thin forearm cylinders and provide the
  physical surface the box rests on.  The box, platforms, and feet
  all use ``contype=1, conaffinity=1`` — collision is determined
  entirely by the entity's ``CollisionCfg`` which enables only the
  geoms that should participate.

  All geoms are named (via ``_name_all_geoms``) so the entity
  system's ``CollisionCfg(disable_other_geoms=True)`` can properly
  disable every geom that shouldn't collide.
  """
  spec = base_spec_fn()

  # ── Step 1: Name all geoms ────────────────────────────────────────
  # Must happen BEFORE we compile, so that all geom names are stable.
  _name_all_geoms(spec)

  # ── Step 1b: Restrict shoulder pitch range for box task ──────────
  # The XML has ±145° which allows the arm to fold back into the
  # torso.  For box carrying the arms stay in front, so ±90° is
  # plenty and physically prevents torso penetration.
  _SHOULDER_PITCH_LIMIT = 1.5708  # 90 degrees
  _restrict = {"left_shoulder_pitch_joint", "right_shoulder_pitch_joint"}
  for j in spec.joints:
    if j.name in _restrict:
      j.range = (-_SHOULDER_PITCH_LIMIT, _SHOULDER_PITCH_LIMIT)

  # ── Step 2: Override arm angles for tray pose ─────────────────────
  robot_model = spec.compile()
  key_qpos = np.array(spec.keys[0].qpos)

  for jname, angle in _TRAY_ARM_OVERRIDES.items():
    jid = robot_model.joint(jname).id
    key_qpos[robot_model.jnt_qposadr[jid]] = angle

  spec.keys[0].qpos = key_qpos.tolist()

  # ── Step 3: Compute forearm geometry in tray pose ─────────────────
  robot_data = mujoco.MjData(robot_model)
  robot_data.qpos[:] = key_qpos
  mujoco.mj_forward(robot_model, robot_data)

  lid = robot_model.body("left_elbow").id
  rid = robot_model.body("right_elbow").id
  l_rot = robot_data.xmat[lid].reshape(3, 3)
  l_pos = robot_data.xpos[lid]
  r_rot = robot_data.xmat[rid].reshape(3, 3)
  r_pos = robot_data.xpos[rid]

  # Midpoint of both forearms (halfway along each forearm).
  l_mid = l_pos + l_rot @ (_HAND_LOCAL * 0.5)
  r_mid = r_pos + r_rot @ (_HAND_LOCAL * 0.5)
  midpoint = (l_mid + r_mid) / 2

  # Find the highest forearm cylinder surface at the midpoint.
  forearm_top_z = -np.inf
  for body_id in (lid, rid):
    for gid in range(robot_model.ngeom):
      if (
        robot_model.geom_bodyid[gid] != body_id
        or robot_model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CYLINDER
      ):
        continue
      gpos = robot_data.geom_xpos[gid]
      gaxis = robot_data.geom_xmat[gid].reshape(3, 3)[:, 2]
      radius = robot_model.geom_size[gid, 0]
      half_len = robot_model.geom_size[gid, 1]
      if abs(gaxis[0]) > 0.001:
        t = np.clip((midpoint[0] - gpos[0]) / gaxis[0], -half_len, half_len)
        z_at_mid = gpos[2] + t * gaxis[2] + radius
        forearm_top_z = max(forearm_top_z, z_at_mid)

  # ── Step 4: Add platform geoms to each forearm ────────────────────
  # Flat slabs whose top surface sits at forearm_top_z.  They move
  # with the elbow body so the box stays supported as long as the
  # arms are extended.  Because they're children of the elbow bodies
  # (not torso), MuJoCo does NOT filter contacts with the box (which
  # is a child of torso — different kinematic branch).
  for elbow_name, body_id in [("left_elbow", lid), ("right_elbow", rid)]:
    elbow_body = _find_body(spec.worldbody, elbow_name)
    assert elbow_body is not None
    e_rot = robot_data.xmat[body_id].reshape(3, 3)
    e_pos = robot_data.xpos[body_id]

    # Platform centre: forearm midpoint, shifted slightly toward
    # the body centreline (Y=0).
    plat_world = e_pos + e_rot @ (_HAND_LOCAL * 0.5)
    plat_world[1] *= 0.2  # close to midline so platforms fit inside box cavity
    # Top surface flush with the forearm cylinder top.
    plat_world[2] = forearm_top_z - _PLATFORM_HALF_THICKNESS

    # Convert world position to elbow-local frame.
    plat_local = e_rot.T @ (plat_world - e_pos)

    # World-aligned orientation in elbow frame (cancel elbow rotation).
    plat_quat = np.zeros(4)
    mujoco.mju_negQuat(plat_quat, robot_data.xquat[body_id])

    geom = elbow_body.add_geom()
    geom.name = f"{elbow_name}_platform"
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = np.array(
      [
        _PLATFORM_HALF_LENGTH,
        _PLATFORM_HALF_WIDTH,
        _PLATFORM_HALF_THICKNESS,
      ]
    )
    geom.pos = plat_local
    geom.quat = plat_quat
    geom.rgba = np.array([0.0, 0.0, 0.0, 0.0])  # invisible
    geom.mass = 0.01
    geom.contype = 1
    geom.conaffinity = 1
    geom.condim = 4
    geom.friction = (1.0, 0.005, 0.0001)

  # ── Step 5: Add box body to worldbody with freejoint ──────────
  torso_id = robot_model.body("torso").id
  t_rot = robot_data.xmat[torso_id].reshape(3, 3)
  t_pos = robot_data.xpos[torso_id]

  box_world_pos = midpoint.copy()
  # Place box bottom surface just above the platform top, outside the
  # contact margin zone so there's no initial repulsive force.  The
  # box will settle onto the platforms under gravity.
  box_world_pos[2] = forearm_top_z + _WALL_HALF_THICKNESS + 0.01

  box_body = spec.worldbody.add_body()
  box_body.name = "box_load"
  box_body.pos = box_world_pos
  box_body.quat = np.array([1.0, 0.0, 0.0, 0.0])  # Native upright orientation

  # Add a single free joint for realistic 6-DOF physics without torso parenting constraints
  fj = box_body.add_freejoint()
  fj.name = "box_freejoint"

  _add_box_geoms(box_body, enable_collision=True)

  # ── Step 6: Build keyframe ────────────────────────────────────────
  # Rebuild the keyframe by mapping each joint's value from the old model
  old_key_qpos = np.array(spec.keys[0].qpos)
  spec.keys[0].qpos = [0.0] * (len(old_key_qpos) + 7)  # +7 for freejoint
  new_model = spec.compile()

  new_key_qpos = np.copy(new_model.qpos0)
  for jid_new in range(new_model.njnt):
    jname = mujoco.mj_id2name(new_model, mujoco.mjtObj.mjOBJ_JOINT, jid_new)
    if jname == "box_freejoint":
      continue  # box freejoint position is set by worldbody placement

    new_qadr = new_model.jnt_qposadr[jid_new]
    jtype = new_model.jnt_type[jid_new]
    nq = {0: 7, 1: 4, 2: 1, 3: 1}[int(jtype)]

    if jname is None:
      # Unnamed joint (robot root freejoint): copy by qpos address
      # from the old keyframe directly to preserve standing height.
      if new_qadr < len(old_key_qpos):
        new_key_qpos[new_qadr : new_qadr + nq] = old_key_qpos[new_qadr : new_qadr + nq]
      continue

    try:
      jid_old = robot_model.joint(jname).id
      old_qadr = robot_model.jnt_qposadr[jid_old]
      new_key_qpos[new_qadr : new_qadr + nq] = old_key_qpos[old_qadr : old_qadr + nq]
    except KeyError:
      pass  # other joints stay at qpos0

  spec.keys[0].qpos = new_key_qpos.tolist()

  return spec


def get_spec_upright_calibrated_rigidtoe_with_box_on_arms() -> mujoco.MjSpec:
  """Box-on-arms spec on the `_rigidtoe + fitted gains` Digit model.

  Uses the creep-corrected plant (rigid toe transmission via fixed-tendon
  couplings + effective holding gains fitted on the ar-control loaded suite)
  instead of the deprecated toe_roll_stiffness=300 workaround. This is the
  plant on which the no-box safety policy first survived the full 20 s
  ar-control benchmark (2026-07-10).
  """
  return get_spec_upright_with_box_on_arms(
    base_spec_fn=get_spec_upright_calibrated_rigidtoe
  )


def get_spec_upright_calibrated_with_box_on_arms() -> mujoco.MjSpec:
  """Box-on-arms spec built on the sim2sim-calibrated Digit model.

  Uses the calibrated base (shin/heel leaf springs, fitted joint
  damping) with toe_roll stiffness 300 Nm/rad — the standing-stable
  ankle calibration, appropriate for this standing-dominant task
  (see sim2sim/README.md finding 4).
  """
  return get_spec_upright_with_box_on_arms(
    base_spec_fn=lambda: get_spec_upright_calibrated(toe_roll_stiffness=300.0)
  )


if __name__ == "__main__":
  import argparse

  import mujoco.viewer as viewer

  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--variant",
    choices=["welded", "free"],
    default="free",
    help="Which box variant to view.",
  )
  args = parser.parse_args()

  if args.variant == "welded":
    model = get_spec_upright_with_box().compile()
  else:
    model = get_spec_upright_with_box_on_arms().compile()

  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("standing").id)
  mujoco.mj_forward(model, data)
  label = "welded" if args.variant == "welded" else "free (platforms)"
  print(f"Viewing {label} box variant — close window to exit.")
  with viewer.launch(model, data) as v:
    pass
