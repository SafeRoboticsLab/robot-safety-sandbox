"""Parkour curriculum learning strategies.

Progress terrain difficulty based on how far the robot gets along the track.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terrain_levels_parkour(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
) -> torch.Tensor | None:
  """Progress terrain difficulty based on forward distance traveled.

  Move up if robot walked >50% of the terrain patch length.
  Move down if robot walked <20%.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_generator is None:
    return None

  patch_length = terrain.cfg.terrain_generator.size[0]

  distance = torch.norm(
    env.scene["robot"].data.root_link_pos_w[env_ids, :2]
    - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  move_up = distance > 0.5 * patch_length
  move_down = distance < 0.2 * patch_length
  move_down = move_down & ~move_up

  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())
