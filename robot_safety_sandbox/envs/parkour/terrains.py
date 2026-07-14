"""Parkour terrain generators for gap jumps, crawl barriers, and rugged ground.

Each terrain type is a SubTerrainCfg subclass that generates MuJoCo geometry
via the mjlab terrain generator system. Difficulty (0-1) controls obstacle
parameters (gap width, barrier height, roughness amplitude).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)



# ---------------------------------------------------------------------------
# Gap Jump Terrain
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class GapJumpTerrainCfg(SubTerrainCfg):
  """Terrain with gaps the robot must jump across.

  The track is a flat runway with one or more gaps cut into it.
  Gap width scales with difficulty.
  """

  gap_width_range: tuple[float, float] = (0.01, 0.6)
  """Min/max gap width in meters, interpolated by difficulty."""
  gap_depth: float = 1.5
  """Depth of the gap (how far down it goes)."""
  num_gaps: int = 3
  """Number of gaps along the track."""
  platform_height: float = 0.0
  """Height of the running surface."""
  border_width: float = 0.25
  """Flat border at the start/end of the track."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []

    gap_width = self.gap_width_range[0] + difficulty * (
      self.gap_width_range[1] - self.gap_width_range[0]
    )

    track_length = self.size[0]
    track_width = self.size[1]

    # Divide the track into segments separated by gaps.
    total_gap = gap_width * self.num_gaps
    total_platform = track_length - total_gap - 2 * self.border_width
    num_segments = self.num_gaps + 1
    segment_length = max(total_platform / num_segments, 0.3)

    cursor_x = 0.0

    # Start border.
    _add_box(
      body, geometries,
      pos=(cursor_x + self.border_width / 2, track_width / 2, self.platform_height / 2),
      size=(self.border_width / 2, track_width / 2, max(self.platform_height / 2, 0.01)),
    )
    cursor_x += self.border_width

    for i in range(num_segments):
      # Platform segment.
      _add_box(
        body, geometries,
        pos=(cursor_x + segment_length / 2, track_width / 2, self.platform_height / 2),
        size=(segment_length / 2, track_width / 2, max(self.platform_height / 2, 0.01)),
      )
      cursor_x += segment_length

      # Gap (no geometry — just empty space). Add walls on the sides of the gap.
      if i < self.num_gaps:
        # Add thin walls at gap edges to make the gap visible.
        wall_thickness = 0.02
        wall_height = self.gap_depth
        _add_box(
          body, geometries,
          pos=(cursor_x + wall_thickness / 2, track_width / 2, -wall_height / 2),
          size=(wall_thickness / 2, track_width / 2, wall_height / 2),
          rgba=(0.3, 0.1, 0.1, 1.0),
        )
        _add_box(
          body, geometries,
          pos=(cursor_x + gap_width - wall_thickness / 2, track_width / 2, -wall_height / 2),
          size=(wall_thickness / 2, track_width / 2, wall_height / 2),
          rgba=(0.3, 0.1, 0.1, 1.0),
        )
        # Bottom of the gap.
        _add_box(
          body, geometries,
          pos=(cursor_x + gap_width / 2, track_width / 2, -(wall_height)),
          size=(gap_width / 2, track_width / 2, 0.02),
          rgba=(0.15, 0.05, 0.05, 1.0),
        )
        cursor_x += gap_width

    # End border.
    remaining = track_length - cursor_x
    if remaining > 0.01:
      _add_box(
        body, geometries,
        pos=(cursor_x + remaining / 2, track_width / 2, self.platform_height / 2),
        size=(remaining / 2, track_width / 2, max(self.platform_height / 2, 0.01)),
      )

    # Side walls to keep robot on track.
    wall_height = 0.5
    wall_thickness = 0.05
    for y_pos in [wall_thickness / 2, track_width - wall_thickness / 2]:
      _add_box(
        body, geometries,
        pos=(track_length / 2, y_pos, wall_height / 2),
        size=(track_length / 2, wall_thickness / 2, wall_height / 2),
        rgba=(0.4, 0.4, 0.4, 1.0),
      )

    origin = np.array([self.border_width + 0.3, track_width / 2, self.platform_height])
    return TerrainOutput(origin=origin, geometries=geometries)


# ---------------------------------------------------------------------------
# Crawl Terrain
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class CrawlTerrainCfg(SubTerrainCfg):
  """Terrain with low barriers the robot must crawl under.

  Barriers are horizontal beams at a height that decreases with difficulty,
  forcing the robot to lower its body.
  """

  barrier_height_range: tuple[float, float] = (0.35, 0.22)
  """Barrier clearance height range (easy, hard). Note: easy > hard."""
  barrier_depth: float = 0.8
  """Length of the barrier along the track direction."""
  barrier_wall_height: float = 0.6
  """Height of the wall above the barrier beam."""
  num_barriers: int = 3
  """Number of crawl barriers."""
  border_width: float = 0.25
  """Flat border at the start/end."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []

    barrier_height = self.barrier_height_range[0] + difficulty * (
      self.barrier_height_range[1] - self.barrier_height_range[0]
    )

    track_length = self.size[0]
    track_width = self.size[1]

    # Ground plane.
    _add_box(
      body, geometries,
      pos=(track_length / 2, track_width / 2, -0.01),
      size=(track_length / 2, track_width / 2, 0.01),
    )

    # Place barriers evenly along the track.
    usable_length = track_length - 2 * self.border_width
    spacing = usable_length / (self.num_barriers + 1)

    for i in range(self.num_barriers):
      bx = self.border_width + spacing * (i + 1)

      # Barrier beam (the thing robot crawls under).
      beam_thickness = 0.04
      _add_box(
        body, geometries,
        pos=(bx, track_width / 2, barrier_height + beam_thickness / 2),
        size=(self.barrier_depth / 2, track_width / 2, beam_thickness / 2),
        rgba=(0.7, 0.2, 0.2, 1.0),
      )

      # Wall above barrier to prevent jumping over.
      wall_h = self.barrier_wall_height
      _add_box(
        body, geometries,
        pos=(bx, track_width / 2, barrier_height + beam_thickness + wall_h / 2),
        size=(self.barrier_depth / 2, track_width / 2, wall_h / 2),
        rgba=(0.5, 0.15, 0.15, 1.0),
      )

      # Support pillars on each side.
      pillar_width = 0.05
      for y_offset in [pillar_width / 2, track_width - pillar_width / 2]:
        _add_box(
          body, geometries,
          pos=(bx, y_offset, (barrier_height + beam_thickness + wall_h) / 2),
          size=(self.barrier_depth / 2, pillar_width / 2, (barrier_height + beam_thickness + wall_h) / 2),
          rgba=(0.4, 0.1, 0.1, 1.0),
        )

    # Side walls.
    wall_height = 0.6
    wall_thickness = 0.05
    for y_pos in [wall_thickness / 2, track_width - wall_thickness / 2]:
      _add_box(
        body, geometries,
        pos=(track_length / 2, y_pos, wall_height / 2),
        size=(track_length / 2, wall_thickness / 2, wall_height / 2),
        rgba=(0.4, 0.4, 0.4, 1.0),
      )

    origin = np.array([self.border_width + 0.3, track_width / 2, 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)


# ---------------------------------------------------------------------------
# Rugged Terrain (heightfield-based rough ground with random obstacles)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class RuggedTerrainCfg(SubTerrainCfg):
  """Rough terrain with random bumps and height steps using box primitives.

  Amplitude scales with difficulty for progressively harder footing.
  Uses boxes instead of heightfields to avoid MuJoCo hfield collision overflow.
  """

  noise_amplitude_range: tuple[float, float] = (0.02, 0.12)
  """Min/max noise amplitude (meters), interpolated by difficulty."""
  step_height_range: tuple[float, float] = (0.02, 0.08)
  """Random step heights added on top of noise."""
  num_steps: int = 8
  """Number of random height steps along the track."""
  num_blocks_x: int = 16
  """Number of terrain blocks along the track (X)."""
  num_blocks_y: int = 3
  """Number of terrain blocks across the track (Y)."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []

    amplitude = self.noise_amplitude_range[0] + difficulty * (
      self.noise_amplitude_range[1] - self.noise_amplitude_range[0]
    )
    step_height = self.step_height_range[0] + difficulty * (
      self.step_height_range[1] - self.step_height_range[0]
    )

    track_length = self.size[0]
    track_width = self.size[1]
    block_len = track_length / self.num_blocks_x
    block_wid = track_width / self.num_blocks_y

    # Build a height profile with cumulative random steps.
    step_positions = np.sort(rng.integers(0, self.num_blocks_x, self.num_steps))
    base_heights = np.zeros(self.num_blocks_x)
    current_height = 0.0
    for pos in step_positions:
      current_height += rng.uniform(-step_height, step_height)
      base_heights[pos:] += current_height

    # Place boxes with per-cell random noise on top of the step profile.
    for ix in range(self.num_blocks_x):
      for iy in range(self.num_blocks_y):
        h = base_heights[ix] + rng.uniform(-amplitude, amplitude)
        cx = ix * block_len + block_len / 2
        cy = iy * block_wid + block_wid / 2
        half_h = max(abs(h) / 2, 0.005)
        _add_box(
          body, geometries,
          pos=(cx, cy, h / 2),
          size=(block_len / 2, block_wid / 2, half_h),
          rgba=(0.45, 0.45, 0.50, 1.0),
        )

    origin = np.array([track_length * 0.1, track_width / 2, base_heights[1]])
    return TerrainOutput(origin=origin, geometries=geometries)


# ---------------------------------------------------------------------------
# Mixed Parkour Terrain (combines all obstacle types in a track)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class MixedParkourTerrainCfg(SubTerrainCfg):
  """A single track that mixes gap jumps, crawl barriers, and rough sections.

  Each segment type is placed sequentially with flat transitions between them.
  Difficulty controls all obstacle parameters simultaneously.
  """

  gap_width_range: tuple[float, float] = (0.1, 0.5)
  """Gap width range for jump sections."""
  barrier_height_range: tuple[float, float] = (0.35, 0.22)
  """Barrier clearance for crawl sections (easy→hard = high→low)."""
  barrier_depth: float = 0.6
  """Crawl barrier length along track."""
  rough_amplitude_range: tuple[float, float] = (0.02, 0.08)
  """Roughness amplitude range."""
  border_width: float = 0.3
  """Flat run-in at start."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []

    track_length = self.size[0]
    track_width = self.size[1]

    gap_width = self.gap_width_range[0] + difficulty * (
      self.gap_width_range[1] - self.gap_width_range[0]
    )
    barrier_height = self.barrier_height_range[0] + difficulty * (
      self.barrier_height_range[1] - self.barrier_height_range[0]
    )
    rough_amp = self.rough_amplitude_range[0] + difficulty * (
      self.rough_amplitude_range[1] - self.rough_amplitude_range[0]
    )

    cursor_x = 0.0

    # --- Section 1: Flat start ---
    flat_len = self.border_width
    _add_box(body, geometries,
      pos=(cursor_x + flat_len / 2, track_width / 2, -0.01),
      size=(flat_len / 2, track_width / 2, 0.01),
    )
    cursor_x += flat_len

    # --- Section 2: Rough terrain (using boxes to approximate) ---
    rough_len = (track_length - 2 * self.border_width) / 3
    num_rough_blocks = 12
    block_len = rough_len / num_rough_blocks
    for i in range(num_rough_blocks):
      h = rng.uniform(-rough_amp, rough_amp)
      _add_box(body, geometries,
        pos=(cursor_x + block_len / 2, track_width / 2, h / 2),
        size=(block_len / 2, track_width / 2, max(abs(h) / 2, 0.005)),
        rgba=(0.45, 0.45, 0.50, 1.0),
      )
      cursor_x += block_len

    # --- Section 3: Gap jump ---
    # Pre-gap platform.
    pre_gap = 0.5
    _add_box(body, geometries,
      pos=(cursor_x + pre_gap / 2, track_width / 2, -0.01),
      size=(pre_gap / 2, track_width / 2, 0.01),
    )
    cursor_x += pre_gap

    # Gap (empty space with bottom).
    _add_box(body, geometries,
      pos=(cursor_x + gap_width / 2, track_width / 2, -1.5),
      size=(gap_width / 2, track_width / 2, 0.02),
      rgba=(0.15, 0.05, 0.05, 1.0),
    )
    cursor_x += gap_width

    # Post-gap platform.
    post_gap = 0.5
    _add_box(body, geometries,
      pos=(cursor_x + post_gap / 2, track_width / 2, -0.01),
      size=(post_gap / 2, track_width / 2, 0.01),
    )
    cursor_x += post_gap

    # --- Section 4: Crawl barrier ---
    # Approach.
    approach = 0.3
    _add_box(body, geometries,
      pos=(cursor_x + approach / 2, track_width / 2, -0.01),
      size=(approach / 2, track_width / 2, 0.01),
    )
    cursor_x += approach

    # Barrier beam.
    beam_thickness = 0.04
    _add_box(body, geometries,
      pos=(cursor_x + self.barrier_depth / 2, track_width / 2, barrier_height + beam_thickness / 2),
      size=(self.barrier_depth / 2, track_width / 2, beam_thickness / 2),
      rgba=(0.7, 0.2, 0.2, 1.0),
    )
    # Wall above barrier.
    wall_h = 0.5
    _add_box(body, geometries,
      pos=(cursor_x + self.barrier_depth / 2, track_width / 2, barrier_height + beam_thickness + wall_h / 2),
      size=(self.barrier_depth / 2, track_width / 2, wall_h / 2),
      rgba=(0.5, 0.15, 0.15, 1.0),
    )
    # Ground under barrier.
    _add_box(body, geometries,
      pos=(cursor_x + self.barrier_depth / 2, track_width / 2, -0.01),
      size=(self.barrier_depth / 2, track_width / 2, 0.01),
    )
    cursor_x += self.barrier_depth

    # --- Section 5: Flat finish ---
    remaining = track_length - cursor_x
    if remaining > 0.01:
      _add_box(body, geometries,
        pos=(cursor_x + remaining / 2, track_width / 2, -0.01),
        size=(remaining / 2, track_width / 2, 0.01),
      )

    # Side walls.
    wall_height = 0.6
    wall_thickness = 0.05
    for y_pos in [wall_thickness / 2, track_width - wall_thickness / 2]:
      _add_box(body, geometries,
        pos=(track_length / 2, y_pos, wall_height / 2),
        size=(track_length / 2, wall_thickness / 2, wall_height / 2),
        rgba=(0.4, 0.4, 0.4, 1.0),
      )

    origin = np.array([self.border_width + 0.2, track_width / 2, 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)


# ---------------------------------------------------------------------------
# Terrain config presets
# ---------------------------------------------------------------------------

from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg


PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
  curriculum=True,
  size=(8.0, 3.0),
  border_width=5.0,
  num_rows=5,
  num_cols=10,
  color_scheme="none",
  sub_terrains={
    "flat": BoxFlatTerrainCfg(proportion=0.1),
    "rugged": RuggedTerrainCfg(proportion=0.2),
    "gap_jump": GapJumpTerrainCfg(proportion=0.25),
    "crawl": CrawlTerrainCfg(proportion=0.25),
    "mixed": MixedParkourTerrainCfg(proportion=0.2),
  },
)


# Gaps-only terrain for the minimal reach-avoid gap experiment.  A single gap
# per patch (flat approach -> gap -> flat landing), with gap width spanning
# clearly-jumpable to clearly-uncrossable across the difficulty rows.
# ``curriculum=False`` so every width is present from the start (paired with a
# ``randomize_terrain`` reset event the policy faces a random width each
# episode), sidestepping the distance-based-curriculum stall.
GAP_ONLY_TERRAINS_CFG = TerrainGeneratorCfg(
  curriculum=False,
  size=(8.0, 3.0),
  border_width=5.0,
  num_rows=10,
  num_cols=10,
  color_scheme="none",
  sub_terrains={
    "gap_jump": GapJumpTerrainCfg(
      proportion=1.0,
      num_gaps=1,
      gap_width_range=(0.1, 1.0),
      gap_depth=1.0,
    ),
  },
)


@dataclass(kw_only=True)
class GapEdgeTerrainCfg(SubTerrainCfg):
  """Single gap with the near edge at a FIXED location (the patch origin).

  Layout along +x: short approach platform -> gap -> landing platform.  The
  patch ORIGIN is placed at the near edge, so the reset event can spawn the
  robot relative to the edge (just before it for the commit-or-refuse decision,
  or mid-air just past it for finish-the-jump) using ``env_origins`` alone,
  without needing the per-patch gap width.  This is the **pure-jump** terrain:
  Policy 2 only ever acts near/over the gap, never walking the approach.
  """

  gap_width_range: tuple[float, float] = (0.1, 1.0)
  gap_depth: float = 1.0
  approach_length: float = 2.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries: list[TerrainGeometry] = []

    gap_width = self.gap_width_range[0] + difficulty * (
      self.gap_width_range[1] - self.gap_width_range[0]
    )
    track_length = self.size[0]
    track_width = self.size[1]
    edge_x = self.approach_length  # near edge fixed here (== patch origin x)

    # Approach platform [0, edge_x].
    _add_box(
      body, geometries,
      pos=(edge_x / 2, track_width / 2, 0.0),
      size=(edge_x / 2, track_width / 2, 0.01),
    )

    # Gap edge walls + floor [edge_x, edge_x + gap_width].
    wt = 0.02
    wh = self.gap_depth
    _add_box(
      body, geometries,
      pos=(edge_x + wt / 2, track_width / 2, -wh / 2),
      size=(wt / 2, track_width / 2, wh / 2), rgba=(0.3, 0.1, 0.1, 1.0),
    )
    _add_box(
      body, geometries,
      pos=(edge_x + gap_width - wt / 2, track_width / 2, -wh / 2),
      size=(wt / 2, track_width / 2, wh / 2), rgba=(0.3, 0.1, 0.1, 1.0),
    )
    _add_box(
      body, geometries,
      pos=(edge_x + gap_width / 2, track_width / 2, -wh),
      size=(gap_width / 2, track_width / 2, 0.02), rgba=(0.15, 0.05, 0.05, 1.0),
    )

    # Landing platform [edge_x + gap_width, track_length].
    landing_start = edge_x + gap_width
    landing_len = track_length - landing_start
    if landing_len > 0.01:
      _add_box(
        body, geometries,
        pos=(landing_start + landing_len / 2, track_width / 2, 0.0),
        size=(landing_len / 2, track_width / 2, 0.01),
      )

    # Low side curbs (just lane markers; kept short so the follow camera can
    # see the robot and gap over them).
    swh, swt = 0.08, 0.05
    for y_pos in [swt / 2, track_width - swt / 2]:
      _add_box(
        body, geometries,
        pos=(track_length / 2, y_pos, swh / 2),
        size=(track_length / 2, swt / 2, swh / 2), rgba=(0.4, 0.4, 0.4, 1.0),
      )

    origin = np.array([edge_x, track_width / 2, 0.0])  # AT the near edge
    return TerrainOutput(origin=origin, geometries=geometries)


GAP_EDGE_TERRAINS_CFG = TerrainGeneratorCfg(
  curriculum=False,
  size=(6.0, 2.0),
  border_width=5.0,
  num_rows=10,
  num_cols=10,
  color_scheme="none",
  sub_terrains={
    "gap_edge": GapEdgeTerrainCfg(
      proportion=1.0,
      gap_width_range=(0.1, 1.0),
      gap_depth=1.0,
      approach_length=2.0,
    ),
  },
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_box(
  body: mujoco.MjsBody,
  geometries: list[TerrainGeometry],
  pos: tuple[float, float, float],
  size: tuple[float, float, float],
  rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
) -> None:
  """Add a box geom to the terrain body and geometry list."""
  geom = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=pos,
    size=size,
  )
  geom.rgba = rgba
  geometries.append(TerrainGeometry(geom=geom, color=rgba))
