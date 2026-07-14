"""Safety-FILTER terrain: arrive with momentum, reach a safe stop.

Each patch is a decision scenario for the safety-filter objective (reach safe
rest given the arrival momentum):

    [ approach ] [ gap ]( sep )[ gap ] ... [ gap ] [ ----- rest zone ----- ]
      spawn+run     cluster of 1..n closely-spaced gaps        settle here

The robot spawns on the ``approach`` with random forward momentum (set by the
env's reset).  If it is slow enough it can brake to rest on the approach BEFORE
the gaps (no jump needed); if it is too fast to brake, it must jump — and if the
gaps come as a cluster (the ``sep`` platforms are too short to stop on), it must
CHAIN through until it reaches the long ``rest zone``.  Which behavior is safe is
dictated by momentum + geometry, exactly the reach-avoid value decides.

Difficulty raises the gap width and the cluster size (1 -> n_max).  Gaps are
SHALLOW (``gap_depth`` ~0.4 m) so a missed jump ends in a clean early
termination instead of a deep free-fall that blows up the contact solver.
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
class SafetyFilterTerrainCfg(SubTerrainCfg):
  approach_length: float = 2.5      # room to brake before the gaps
  approach_jitter: float = 0.0      # keep 0: resets compute exact distance-to-gap
  gap_width_range: tuple[float, float] = (0.15, 0.50)  # by difficulty
  gap_jitter: float = 0.03
  n_gaps_max: int = 3               # cluster size grows with difficulty
  separator: float = 0.35          # inter-gap platform: too short to stop on
  gap_depth: float = 0.4

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    body = spec.body("terrain")
    geoms: list[TerrainGeometry] = []
    tw = self.size[1]
    d = float(difficulty)

    approach = self.approach_length + float(rng.uniform(-self.approach_jitter,
                                                        self.approach_jitter))
    gap_w = self.gap_width_range[0] + d * (
      self.gap_width_range[1] - self.gap_width_range[0]
    )
    # cluster size: 1 at difficulty 0, up to n_gaps_max at difficulty 1.
    n = 1 + int(rng.integers(0, round(d * (self.n_gaps_max - 1)) + 1))

    # Distinct colors so videos are legible: blue-gray approach, dark-red pits,
    # tan separators, green-tinted rest zone.
    _add_box(body, geoms, pos=(approach / 2, tw / 2, 0.0),
             size=(approach / 2, tw / 2, 0.01), rgba=(0.45, 0.52, 0.65, 1.0))
    x = approach
    for i in range(n):
      w = max(0.08, gap_w + float(rng.uniform(-self.gap_jitter, self.gap_jitter)))
      _add_box(body, geoms, pos=(x + w / 2, tw / 2, -self.gap_depth),
               size=(w / 2, tw / 2, 0.02), rgba=(0.25, 0.06, 0.06, 1.0))
      x += w
      if i < n - 1:  # short separator (unstoppable) inside the cluster
        _add_box(body, geoms, pos=(x + self.separator / 2, tw / 2, 0.0),
                 size=(self.separator / 2, tw / 2, 0.01), rgba=(0.72, 0.62, 0.45, 1.0))
        x += self.separator

    # Long rest zone (robustly safe: settle to a stop here after the cluster).
    rest = max(1.0, self.size[0] - x)
    _add_box(body, geoms, pos=(x + rest / 2, tw / 2, 0.0),
             size=(rest / 2, tw / 2, 0.01), rgba=(0.45, 0.62, 0.48, 1.0))

    origin = np.array([0.0, tw / 2, 0.0])  # spawn at approach start
    return TerrainOutput(origin=origin, geometries=geoms)


SAFETY_FILTER_TERRAINS_CFG = TerrainGeneratorCfg(
  curriculum=True,
  size=(12.0, 3.0),
  border_width=5.0,
  num_rows=10,     # difficulty levels
  num_cols=10,     # variations per level
  difficulty_range=(0.0, 1.0),
  color_scheme="none",
  sub_terrains={
    "sf": SafetyFilterTerrainCfg(proportion=1.0),
  },
)
