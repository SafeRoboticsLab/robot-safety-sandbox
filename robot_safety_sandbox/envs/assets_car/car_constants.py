"""Differential-drive car constants (robot-from-MJCF + wheel velocity servos).

Mirrors the go2/digit ``*_constants.py`` pattern: a ``get_spec`` that loads the
MJCF, and an ``EntityCfg`` with an ``EntityArticulationInfoCfg`` whose actuators
mjlab injects into the spec. The two hinge wheels ("left"/"right") are driven by
built-in <velocity> servos (velocity targets, matching the diff-drive control of
the safety-gymnasium car); the passive ball caster ("rear") and the free-joint
chassis carry no actuator.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ZOO_ENVS_PATH = Path(__file__).resolve().parents[1]

CAR_XML: Path = ZOO_ENVS_PATH / "assets_car" / "xmls" / "car.xml"
assert CAR_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(CAR_XML))


##
# Actuator config: one velocity servo per wheel joint.
##

WHEEL_ACTUATOR = BuiltinVelocityActuatorCfg(
  target_names_expr=("left", "right"),
  damping=5.0,          # velocity-servo gain (kv)
  effort_limit=2.0,     # plenty for the ~20 g car; wheels saturate to target fast
  armature=0.001,
)

##
# Initial state: chassis just above the ground, wheels at rest.
#
# The car's mechanical FORWARD is its body -y axis (the front bumper sits at -y,
# and the wheels roll about the body x-axis). We yaw the default pose +90 deg
# about z so that "forward" points along world +x -- the direction the goal /
# obstacle corridor is laid out (see car_arena.py). rot is (w, x, y, z).
##

_SQRT_HALF = 0.7071067811865476  # cos(45 deg) = sin(45 deg): +90 deg about z

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.1),
  rot=(_SQRT_HALF, 0.0, 0.0, _SQRT_HALF),
  joint_pos={"left": 0.0, "right": 0.0},
  joint_vel={".*": 0.0},
)

CAR_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(WHEEL_ACTUATOR,),
  soft_joint_pos_limit_factor=1.0,
)


def get_car_robot_cfg() -> EntityCfg:
  """A fresh diff-drive car EntityCfg (new instance each call)."""
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=CAR_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_car_robot_cfg())
  viewer.launch(robot.spec.compile())
