"""Low-bar crawl terrain (parameterized): flat approach -> a VIRTUAL low bar -> rest.

Layout along +x (patch origin at the BAR FACE, so x_rel = 0 is the bar's near
edge)::

    [ approach ] | [ under-bar: flat floor + VIRTUAL beam @clearance ] | [ rest ]
       spawn here  origin   crawl under (depth = bar_depth)              past-bar

The floor is a single continuous PHYSICAL plane at z = 0 spanning approach +
under-bar + rest.  The bar is **VIRTUAL**: a visual-only beam geom rendered at
``bar_clearance`` with ``contype = conaffinity = 0`` so it has NO collision.

Why virtual (E032 lesson): a *physical* bar on the ``terrain`` body let a
leg-vs-bar strike read as a legal ground contact (contact aliasing) -- the
margin code could not tell "clipped the beam" from "foot on the floor".  Here
the bar never touches the physics: the g margin rules a strike analytically
(``bar_clearance - trunk_top`` while under the span), and it also rules out
jumping-over analytically (same term goes < 0 when the trunk is too high), so no
physical anti-jump wall / pillars are needed.

Two knobs the env cfg overrides per difficulty:
  * ``bar_clearance`` -- under-beam clearance in m (lower = must duck more).
  * ``bar_depth``     -- bar thickness along the track = the crawl width the
                         robot must stay ducked to cover.
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

BEAM_THICKNESS = 0.04


def _add_box(body, geoms, pos, size, rgba=(0.5, 0.5, 0.5, 1.0), collide=True):
  geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)
  geom.rgba = rgba
  if not collide:
    # Visual-only geom: no collision (contype/conaffinity 0). The virtual bar
    # never enters the physics -> no contact aliasing with the ground (E032).
    geom.contype = 0
    geom.conaffinity = 0
  geoms.append(TerrainGeometry(geom=geom, color=rgba))


@dataclass(kw_only=True)
class LowBarTerrainCfg(SubTerrainCfg):
  bar_clearance: float = 0.39    # under-beam clearance (m); trunk-top must fit
  bar_depth: float = 0.4         # bar thickness along track = required crawl width
  approach_length: float = 2.2   # runway before the bar (spawn lives here; long
                                  # enough that the far standstill spawns ~-2.0 m
                                  # sit on the coloured approach, not the border)

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    body = spec.body("terrain")
    geoms: list[TerrainGeometry] = []
    tl, tw = self.size[0], self.size[1]
    al, W, H = self.approach_length, self.bar_depth, self.bar_clearance

    # --- PHYSICAL floor (collidable), z = 0, coloured by section --------------
    # Approach (blue-gray), 0 .. al.
    _add_box(body, geoms, pos=(al / 2, tw / 2, 0.0),
             size=(al / 2, tw / 2, 0.01), rgba=(0.45, 0.52, 0.65, 1.0))
    # Under-bar crawl surface (tan), al .. al + W.
    _add_box(body, geoms, pos=(al + W / 2, tw / 2, 0.0),
             size=(W / 2, tw / 2, 0.01), rgba=(0.72, 0.62, 0.45, 1.0))
    # Rest zone (green), al + W .. tl.
    x1 = al + W
    rest = max(1.0, tl - x1)
    _add_box(body, geoms, pos=(x1 + rest / 2, tw / 2, 0.0),
             size=(rest / 2, tw / 2, 0.01), rgba=(0.45, 0.62, 0.48, 1.0))

    # --- VIRTUAL beam (visual only, NO collision) -----------------------------
    # Bottom of the beam sits at the under-clearance H; semi-transparent red so
    # the crawling robot is visible through it in eval videos.
    _add_box(body, geoms, pos=(al + W / 2, tw / 2, H + BEAM_THICKNESS / 2),
             size=(W / 2, tw / 2, BEAM_THICKNESS / 2),
             rgba=(0.85, 0.15, 0.15, 0.4), collide=False)

    origin = np.array([al, tw / 2, 0.0])   # AT the bar face -> x_rel measured here
    return TerrainOutput(origin=origin, geometries=geoms)


LOW_BAR_TERRAINS_CFG = TerrainGeneratorCfg(
  curriculum=False,
  size=(6.0, 3.0),
  border_width=5.0,
  num_rows=1,
  num_cols=10,
  color_scheme="none",
  sub_terrains={"low_bar": LowBarTerrainCfg(proportion=1.0)},
)
