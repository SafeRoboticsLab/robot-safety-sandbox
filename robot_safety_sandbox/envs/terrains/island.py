"""Trapped-island forced-crossing terrain for the ISAACS gap-safety scenario.

Layout along +x (patch origin at the island's FRONT edge = near edge of the gap)::

    [ back pit ] [ short island ] | [ front gap ] | [ long far platform ]
       fall         spawn here    origin  fall          robustly safe

The robot spawns on the short island. A back pit means a backward shove falls;
the front gap must be crossed to reach the long (robustly-safe) far platform.
Under an adversary strong enough that no stationary stance survives on the small
island, the ONLY V>0 option is to cross forward — the "fail unless you cross"
scenario. Gap width scales with difficulty (0.05 -> 0.6 m).
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


def _add_box(body, geoms, pos, size, rgba=(0.5, 0.5, 0.5, 1.0)):
  geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)
  geom.rgba = rgba
  geoms.append(TerrainGeometry(geom=geom, color=rgba))


@dataclass(kw_only=True)
class IslandCrossingTerrainCfg(SubTerrainCfg):
  gap_width_range: tuple[float, float] = (0.05, 0.6)
  gap_depth: float = 1.0
  island_length: float = 1.0
  back_pit_length: float = 0.8

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    body = spec.body("terrain")
    geoms: list[TerrainGeometry] = []

    gap_width = self.gap_width_range[0] + difficulty * (
      self.gap_width_range[1] - self.gap_width_range[0]
    )
    tl, tw = self.size[0], self.size[1]
    wh = self.gap_depth

    island_back = self.back_pit_length
    island_front = island_back + self.island_length  # == patch origin x

    # Island (spawn) platform.
    _add_box(body, geoms, pos=(island_back + self.island_length / 2, tw / 2, 0.0),
             size=(self.island_length / 2, tw / 2, 0.01))
    # Back pit floor (visual; deep -> falling in terminates).
    _add_box(body, geoms, pos=(island_back / 2, tw / 2, -wh),
             size=(island_back / 2, tw / 2, 0.02), rgba=(0.15, 0.05, 0.05, 1.0))
    # Front gap floor (visual).
    _add_box(body, geoms, pos=(island_front + gap_width / 2, tw / 2, -wh),
             size=(gap_width / 2, tw / 2, 0.02), rgba=(0.15, 0.05, 0.05, 1.0))
    # Long far (robustly-safe) platform.
    far_start = island_front + gap_width
    far_len = tl - far_start
    if far_len > 0.01:
      _add_box(body, geoms, pos=(far_start + far_len / 2, tw / 2, 0.0),
               size=(far_len / 2, tw / 2, 0.01))

    origin = np.array([island_front, tw / 2, 0.0])  # AT the island front edge
    return TerrainOutput(origin=origin, geometries=geoms)


ISLAND_CROSSING_TERRAINS_CFG = TerrainGeneratorCfg(
  curriculum=False,
  size=(6.0, 2.0),
  border_width=5.0,
  num_rows=10,
  num_cols=10,
  color_scheme="none",
  sub_terrains={
    "island": IslandCrossingTerrainCfg(
      proportion=1.0,
      gap_width_range=(0.05, 0.6),
      gap_depth=1.0,
      island_length=0.7,  # smaller island -> harder to just balance in place
      back_pit_length=0.8,
    ),
  },
)
