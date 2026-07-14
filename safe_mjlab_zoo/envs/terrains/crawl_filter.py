"""Crawl-filter terrain: stop-or-crawl decision scenarios under a low bar.

Each patch is one decision scenario (mirrors ``safety_filter_terrain.py``):

    [ approach (2.5 m exact) ] [ BAR: beam + anti-jump wall + pillars ] [ rest zone ]

The bar's under-beam CLEARANCE comes from an explicit per-row table spanning
trivially-passable (0.35 m) down to the Go2's crouch feasibility floor
(~0.22 m) and TWO IMPOSSIBLE rows (0.18 / 0.15 m — below any crouch): the
value function needs data below the floor to learn the STOP decision, and the
benchmark's impossible columns must be in-distribution.

Bar geometry mirrors ``parkour/terrains.py::CrawlTerrainCfg``: a thin beam at
the clearance height, a solid wall stacked above it (so jumping over is never
an option), and side pillars. All geoms live on the ``terrain`` body, so bar
strikes register on the existing ``nonfoot_ground_touch`` contact sensor and
drive g < 0 through the reach-avoid wrapper's contact margin unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

# Bar face x (relative to patch origin) == approach length; keep exact (no
# jitter) so resets/evals compute distance-to-bar analytically.
_BAR_X = 2.5
BAR_DEPTH = 0.8       # bar section length along the track
BEAM_THICKNESS = 0.04
WALL_HEIGHT = 0.6     # anti-jump wall stacked above the beam
PILLAR_WIDTH = 0.05

# Explicit clearance (bottom of beam) per difficulty row, HIGH -> LOW so the
# forward curriculum starts trivial: row 0 (0.50 m) clears the standing Go2
# (trunk top ~0.38 m walking) untouched -> the walking behavior passes with no
# new skill.  Rows descend one small notch at a time through the crouch
# feasibility floor (~0.22 m) so ducking is acquired continuously, ending in
# two IMPOSSIBLE rows below the floor (correct behavior there: STOP).
ROW_CLEARANCES = (
  0.50, 0.46, 0.42, 0.39, 0.36, 0.33, 0.30, 0.27, 0.25, 0.235, 0.22, 0.18, 0.15
)
N_ROWS = len(ROW_CLEARANCES)


def bar_clearance_for_level(levels: torch.Tensor) -> torch.Tensor:
  """Per-env under-beam clearance (m) from the terrain difficulty row."""
  table = torch.tensor(ROW_CLEARANCES, device=levels.device, dtype=torch.float32)
  return table[levels.clamp(0, N_ROWS - 1).long()]


def is_impossible_level(levels: torch.Tensor) -> torch.Tensor:
  return bar_clearance_for_level(levels) <= 0.20


def _add_box(body, geoms, pos, size, rgba=(0.5, 0.5, 0.5, 1.0)):
  geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)
  geom.rgba = rgba
  geoms.append(TerrainGeometry(geom=geom, color=rgba))


@dataclass(kw_only=True)
class CrawlFilterTerrainCfg(SubTerrainCfg):
  approach_length: float = _BAR_X

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    body = spec.body("terrain")
    geoms: list[TerrainGeometry] = []
    tw = self.size[1]

    # Difficulty (0..1 within-row jitter included) -> row index -> clearance.
    row = min(int(difficulty * N_ROWS), N_ROWS - 1)
    clearance = ROW_CLEARANCES[row]
    impossible = clearance <= 0.20

    # Approach (blue-gray).
    _add_box(body, geoms, pos=(self.approach_length / 2, tw / 2, 0.0),
             size=(self.approach_length / 2, tw / 2, 0.01),
             rgba=(0.45, 0.52, 0.65, 1.0))

    x0 = self.approach_length
    # Ground continues under the bar (crawling surface, tan).
    _add_box(body, geoms, pos=(x0 + BAR_DEPTH / 2, tw / 2, 0.0),
             size=(BAR_DEPTH / 2, tw / 2, 0.01), rgba=(0.72, 0.62, 0.45, 1.0))
    # Beam at the clearance height (red; darker when impossible). Semi-
    # transparent so the robot is visible THROUGH the bar structure in videos
    # (a solid wall occludes it from every non-opening angle).
    beam_rgba = (0.55, 0.05, 0.05, 0.65) if impossible else (0.85, 0.15, 0.15, 0.6)
    _add_box(body, geoms,
             pos=(x0 + BAR_DEPTH / 2, tw / 2, clearance + BEAM_THICKNESS / 2),
             size=(BAR_DEPTH / 2, tw / 2, BEAM_THICKNESS / 2), rgba=beam_rgba)
    # Anti-jump wall above the beam (mostly transparent — its only job is to
    # rule out jumping over, it should not hide the robot).
    wall_z = clearance + BEAM_THICKNESS + WALL_HEIGHT / 2
    _add_box(body, geoms, pos=(x0 + BAR_DEPTH / 2, tw / 2, wall_z),
             size=(BAR_DEPTH / 2, tw / 2, WALL_HEIGHT / 2),
             rgba=(0.35, 0.35, 0.4, 0.25))
    # Side pillars (full height of beam+wall).
    pillar_h = clearance + BEAM_THICKNESS + WALL_HEIGHT
    for y in (PILLAR_WIDTH, tw - PILLAR_WIDTH):
      _add_box(body, geoms, pos=(x0 + BAR_DEPTH / 2, y, pillar_h / 2),
               size=(BAR_DEPTH / 2, PILLAR_WIDTH, pillar_h / 2),
               rgba=(0.3, 0.3, 0.3, 0.25))  # semi-transparent so the side
      # follow-camera (azimuth 90) sees the robot through the near pillar

    # Long rest zone (green-tinted).
    x1 = x0 + BAR_DEPTH
    rest = max(1.0, self.size[0] - x1)
    _add_box(body, geoms, pos=(x1 + rest / 2, tw / 2, 0.0),
             size=(rest / 2, tw / 2, 0.01), rgba=(0.45, 0.62, 0.48, 1.0))

    origin = np.array([0.0, tw / 2, 0.0])  # spawn at approach start
    return TerrainOutput(origin=origin, geometries=geoms)


CRAWL_FILTER_TERRAINS_CFG = TerrainGeneratorCfg(
  curriculum=True,
  size=(12.0, 3.0),
  border_width=5.0,
  num_rows=N_ROWS,   # one difficulty row per clearance table entry
  num_cols=10,
  difficulty_range=(0.0, 1.0),
  color_scheme="none",
  sub_terrains={"crawl": CrawlFilterTerrainCfg(proportion=1.0)},
)
