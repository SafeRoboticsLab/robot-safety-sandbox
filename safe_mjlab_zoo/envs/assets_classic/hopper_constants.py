"""Hopper entity (gym Hopper-v4 lineage) for mjlab.

Planar 3-link hopper: unactuated root joints (rootx slide, rootz slide with
ref=1.25, rooty hinge) + 3 actuated hinges. Motors replicate gym exactly:
ctrl range [-1, 1], gear 200 (joint torque = 200 * ctrl).

qpos layout (XML order): [rootx, rootz, rooty, thigh, leg, foot]
gym obs = [qpos[1:], clip(qvel, +-10)] -> 11 dims.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

HOPPER_XML = Path(__file__).parent / "xmls" / "hopper.xml"

# gym motor: ctrlrange +-1.0, gear 200 -> forcerange +-1 pre-gear (equivalent).
# armature=1.0 replicates the XML's joint default (mjlab's actuator edit_spec
# OVERWRITES joint armature with the cfg value — 0 would change the dynamics).
HOPPER_ACTUATOR = BuiltinMotorActuatorCfg(
  target_names_expr=("thigh_joint", "leg_joint", "foot_joint"),
  effort_limit=1.0,
  gear=200.0,
  armature=1.0,
)

HOPPER_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(HOPPER_ACTUATOR,),
)


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(HOPPER_XML))


def get_hopper_robot_cfg() -> EntityCfg:
  """Fresh EntityCfg per call (avoid shared-mutation issues)."""
  return EntityCfg(
    # mjlab resets joints to default_joint_pos (NOT the XML qpos0), so the
    # rootz slide's ref=1.25 must be restated here — qpos[rootz]=1.25 is zero
    # displacement (torso at 1.25 m); 0 would put the torso INSIDE the floor.
    init_state=EntityCfg.InitialStateCfg(
      joint_pos={"rootz": 1.25},
      joint_vel={".*": 0.0},
    ),
    spec_fn=get_spec,
    articulation=HOPPER_ARTICULATION,
  )
