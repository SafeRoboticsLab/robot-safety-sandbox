"""Unitree Go2 with a sloshy / rigid payload on the trunk — the ODD-conditioned demonstration robot.

NON-INVASIVE: loads the stock Go2 spec (``assets.go2.go2_constants.get_spec``, unmodified) and adds a stacked,
hinged payload to ``base_link`` via the MjSpec API (the pattern from ``assets_digit/digit_with_box``).
Ported from LINC-POC (``linc/envs/pybullet/env_hexapod.py``, envtype='spring'): a vertical stack of
boxes — block_0 rigid on the trunk, then N boxes each on a revolute hinge with ALTERNATING roll/pitch
axes (so it can arch any direction), range ±0.5 rad, viscous damping.

ODD AXES (all constructor args):
  * stiffness   — hinge spring = RIGIDITY (0 = water-like slosh … large = rigid box). PRIMARY axis.
  * total_mass  — payload mass, distributed over blocks (see mass_profile). PRIMARY axis.
  * n_layers    — number of hinged blocks (height / CoM). structural.
  * mass_profile— 'uniform' | 'top_heavy' | 'bottom_heavy' (CoM shift). structural.
  * damping     — hinge viscous damping.

Payload joints are named ``payload_j*`` (unmatched by Go2's leg-actuator regexes ⇒ PASSIVE) and
payload geoms are ``payload_b*`` with contype/conaffinity=0 (⇒ non-colliding, unmatched by the
``.*_collision`` collision cfg). So the base Go2 articulation / collision / init are reused verbatim.
"""

from functools import partial
from typing import List

import mujoco
import numpy as np

from mjlab.entity import EntityCfg

from robot_safety_sandbox.envs.assets.go2 import go2_constants as _g2

# payload block geometry — LINC 0.15×0.15×0.05 (full) ⇒ MuJoCo half-extents:
_HW, _HH = 0.075, 0.025
_LAYER_DZ = 2 * _HH                       # vertical spacing between block frames (block height)
_MOUNT_Z = 0.06                           # sit the stack just above the trunk top (base geom ~0.057)
_RANGE = 0.5                              # hinge limit (rad), LINC value
_COLORS = (np.array([0.85, 0.75, 0.1, 1.0]), np.array([0.2, 0.7, 0.35, 1.0]))   # Yellow / Green


def mass_profile(n_blocks: int, total_mass: float, profile: str = "uniform") -> List[float]:
  """Per-block masses (n_blocks = 1 fixed base + hinged layers) summing to total_mass. The
  mass-DISTRIBUTION ODD axis (shifts CoM): 'uniform' | 'top_heavy' (↑) | 'bottom_heavy' (↓)."""
  if profile == "uniform":
    w = [1.0] * n_blocks
  elif profile == "top_heavy":
    w = [float(i + 1) for i in range(n_blocks)]
  elif profile == "bottom_heavy":
    w = [float(n_blocks - i) for i in range(n_blocks)]
  else:
    raise ValueError(profile)
  s = sum(w)
  return [total_mass * wi / s for wi in w]


def _add_block(body: "mujoco.MjsBody", name: str, mass: float, color: np.ndarray) -> None:
  g = body.add_geom()
  g.name = name
  g.type = mujoco.mjtGeom.mjGEOM_BOX
  g.size = np.array([_HW, _HW, _HH])
  g.pos = np.array([0.0, 0.0, _HH])       # block sits just above the body frame (hinge at its base)
  g.rgba = color
  g.mass = float(mass)
  g.contype = 0                           # payload does not collide (self or robot/world)
  g.conaffinity = 0


def add_payload(spec: mujoco.MjSpec, parent: str = "base_link", n_layers: int = 4,
                total_mass: float = 3.0, stiffness: float = 0.0, damping: float = 0.05,
                profile: str = "uniform", rng: float = _RANGE) -> mujoco.MjSpec:
  """Add the stacked hinged payload under `parent`. Mutates and returns `spec`."""
  masses = mass_profile(n_layers + 1, total_mass, profile)
  base = spec.body(parent)
  mount = base.add_body()
  mount.name = "payload_mount"
  mount.pos = np.array([0.0, 0.0, _MOUNT_Z])
  _add_block(mount, "payload_b0", masses[0], _COLORS[0])     # block_0 rigid on the mount
  parent_body = mount
  for i in range(1, n_layers + 1):
    b = parent_body.add_body()
    b.name = f"payload_b{i}"
    b.pos = np.array([0.0, 0.0, _LAYER_DZ])
    j = b.add_joint(name=f"payload_j{i}")
    j.type = mujoco.mjtJoint.mjJNT_HINGE
    j.axis[:] = np.array([1.0, 0.0, 0.0]) if i % 2 else np.array([0.0, 1.0, 0.0])  # alternate roll/pitch
    j.pos[:] = np.zeros(3)
    j.stiffness = float(stiffness)        # RIGIDITY ODD
    j.damping = float(damping)
    j.limited = True
    j.range[:] = np.array([-rng, rng])
    _add_block(b, f"payload_b{i}", masses[i], _COLORS[i % 2])
    parent_body = b
  return spec


def get_spec(n_layers: int = 4, total_mass: float = 3.0, stiffness: float = 0.0,
             damping: float = 0.05, profile: str = "uniform") -> mujoco.MjSpec:
  """Stock Go2 spec + payload (base Go2 XML untouched)."""
  spec = _g2.get_spec()
  add_payload(spec, n_layers=n_layers, total_mass=total_mass, stiffness=stiffness,
              damping=damping, profile=profile)
  return spec


def get_go2_payload_robot_cfg(n_layers: int = 4, total_mass: float = 3.0, stiffness: float = 0.0,
                              damping: float = 0.05, profile: str = "uniform") -> EntityCfg:
  """Fresh EntityCfg for the Go2+payload robot (reuses Go2's init / articulation / collision)."""
  return EntityCfg(
    init_state=_g2.INIT_STATE,
    collisions=(_g2.FULL_COLLISION,),
    spec_fn=partial(get_spec, n_layers=n_layers, total_mass=total_mass,
                    stiffness=stiffness, damping=damping, profile=profile),
    articulation=_g2.GO2_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity
  robot = Entity(get_go2_payload_robot_cfg())
  viewer.launch(robot.spec.compile())
