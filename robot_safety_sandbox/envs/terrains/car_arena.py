"""Car-goal arena terrain: a flat plate, a corridor of obstacle cylinders, and a
visual goal disk -- the mjlab analog of ``bicycle5d.py``'s planar scene.

Layout along +x (the car spawns at the patch ORIGIN and drives toward the goal)::

    car spawn (origin) ....  o   o   o  .... [ GOAL disk ]
                            obstacles          x = START_TO_GOAL

The patch ORIGIN is returned at the spawn point, so every world position is read
in margins as ``pos - env_origins`` (see ``car_margins`` in car_goal/env_cfg.py):
an obstacle placed at local ``origin + (dx, dy)`` sits at ``(dx, dy)`` relative to
the spawn, so the module-level ``OBSTACLES`` / ``START_TO_GOAL`` offsets ARE the
coordinates the margin math uses -- the geometry and the margins can never drift
apart. The layout is fixed (identical across patches); per-episode variety comes
from the randomized car spawn (position + heading), which keeps the visual
obstacle geoms and the analytic ``g`` margin exactly consistent.

  * obstacle cylinders  -- VISUAL ONLY (contype = conaffinity = 0). They mark the
                           KEEP-OUT region {g<0}; the failure set is DEFINED by g,
                           not by physics. Collidable cylinders would block the car
                           at their surface so g never reaches <0 and the collision
                           termination could never fire (matches bicycle5d, where
                           obstacles are circles in the margin, not walls).
  * goal disk           -- VISUAL ONLY (contype = conaffinity = 0): reaching it is
                           an analytic ``l`` event, it must not push the car.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

# --- scene geometry (all offsets are RELATIVE TO THE CAR SPAWN, +x forward) ---
START_TO_GOAL = 2.0        # goal disk center, meters ahead of the spawn
GOAL_RADIUS = 0.40         # goal disk radius (l >= 0 inside)
#: (dx, dy, radius) -- TWO obstacle cylinders forming a light slalom between the
#: spawn and the goal: a straight run clips them, so the car must weave (swerve
#: away from the +y one, then the -y one) and then enter the goal.
OBSTACLES: tuple[tuple[float, float, float], ...] = (
  (0.75, 0.32, 0.25),
  (1.40, -0.32, 0.25),
)

_OBSTACLE_HEIGHT = 0.30    # cylinder half-height (visual only in z)
_SPAWN_X = 1.0            # spawn this far from the patch's -x edge
_ARENA = (5.0, 3.0)       # patch (length x, width y): fits spawn + obstacles + goal

# A clean, high-contrast demo palette (this scene is the showcase GIF):
# light neutral floor, vivid PURPLE obstacles (hazard), vivid GREEN goal (target).
_FLOOR_RGBA = (0.87, 0.88, 0.91, 1.0)      # clean light-grey backdrop
_OBSTACLE_RGBA = (0.55, 0.30, 0.80, 1.0)   # purple — the hazard
_GOAL_RGBA = (0.16, 0.73, 0.40, 0.55)      # green, semi-transparent — the target


@dataclass(kw_only=True)
class CarArenaTerrainCfg(SubTerrainCfg):
  def function(self, difficulty, spec, rng) -> TerrainOutput:
    body = spec.body("terrain")
    geoms: list[TerrainGeometry] = []
    L, W = self.size
    ox, oy = _SPAWN_X, W / 2   # spawn origin in the patch-local frame

    # --- floor plate (collidable driving surface) ---
    floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX, pos=(L / 2, W / 2, -0.01),
      size=(L / 2, W / 2, 0.01))
    floor.rgba = _FLOOR_RGBA
    geoms.append(TerrainGeometry(geom=floor, color=_FLOOR_RGBA))

    # --- obstacle cylinders (VISUAL-ONLY analytic keep-out) ---
    # contype=conaffinity=0: NO physics collision. The obstacle is a KEEP-OUT
    # REGION {g<0}, not a wall -- the failure set is DEFINED by the g margin (its
    # zero-level set is the boundary). If the cylinders were collidable, physics
    # would stop the car at the surface so g never reaches <0 and the collision
    # termination could never fire. Visual-only lets the car ENTER the region,
    # g<0 triggers car_collision -> the episode ends and base.py anchors g to
    # failure there. Same convention as the goal disk and the zoo "island"/
    # virtual-bar hazards; matches bicycle5d (obstacles are circles in the margin).
    for dx, dy, r in OBSTACLES:
      cyl = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(ox + dx, oy + dy, _OBSTACLE_HEIGHT),
        size=(r, _OBSTACLE_HEIGHT, r))
      cyl.rgba = _OBSTACLE_RGBA
      cyl.contype = 0
      cyl.conaffinity = 0
      geoms.append(TerrainGeometry(geom=cyl, color=_OBSTACLE_RGBA))

    # --- goal disk (VISUAL ONLY: no collision) ---
    goal = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      pos=(ox + START_TO_GOAL, oy, 0.005), size=(GOAL_RADIUS, 0.005, GOAL_RADIUS))
    goal.rgba = _GOAL_RGBA
    goal.contype = 0
    goal.conaffinity = 0
    geoms.append(TerrainGeometry(geom=goal, color=_GOAL_RGBA))

    origin = np.array([ox, oy, 0.0])
    return TerrainOutput(origin=origin, geometries=geoms)


CAR_ARENA_CFG = TerrainGeneratorCfg(
  curriculum=False,
  size=_ARENA,
  border_width=0.0,     # no perimeter wall (the arena plate is the whole world)
  num_rows=1,
  num_cols=10,
  # "height" honors each geom's per-geometry `color` (set in function() below);
  # "none" would force every terrain geom to uniform grey. So we get the purple
  # obstacles / green goal / light floor palette instead of grey.
  color_scheme="height",
  sub_terrains={"car_arena": CarArenaTerrainCfg(proportion=1.0, size=_ARENA)},
)
