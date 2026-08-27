"""RAAS v2 lander RETRAIN env: the safelanding env at gap 0.30, but
the reset spawns from the HARVESTED weak-launch / in-gap-descent manifold that the
frozen RAS reach policy actually produces from rest
(``scratchpad/ras_harvest_weaklaunch.py``) — NOT the ballistic S_airborne launch box
the safelanding lander (``landing.py::reset_midair_land``) trained on.

Everything else is the safelanding env verbatim: same island terrain @0.30, same
1175-d obs, same avoid-only ``g_terrain_relative`` margin. Only the spawn distribution
changes, so the retrained lander is competent on the WEAK launches (vx~0.5-2.0)
origination-from-rest produces, where the base lander (trained on vx 2.5-3.5) is not.

Rows are the 37-col jump_bank format (x_rel, z, quat, linv, angv, jpos, jvel, w);
restored via ``brake_or_jump._restore`` at their own x_rel with NO momentum scaling.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from robot_safety_sandbox.envs.go2_gap.brake_or_jump import _restore
from robot_safety_sandbox.envs.go2_gap.landing import unitree_go2_landing_env_cfg

_MANIFOLD_PATH = os.environ.get(
  "RAAS_MANIFOLD_PATH",
  os.path.expanduser(
    "~/artifacts/robot-safety-sandbox/lander-v2/weaklaunch_manifold.npz"))
_BANK: dict = {}                # cache per (path, split)


def _load_manifold(env, split: str = "train") -> torch.Tensor:
  key = (_MANIFOLD_PATH, split)
  if key not in _BANK:
    d = np.load(_MANIFOLD_PATH)
    rows = d["rows"]
    if split == "train":
      rows = rows[d["train_idx"]]
    elif split == "val":
      rows = rows[d["val_idx"]]
    elif split != "all":
      raise ValueError(f"split must be train|val|all, got {split!r}")
    _BANK[key] = torch.tensor(np.ascontiguousarray(rows), dtype=torch.float32,
                              device=env.device)
    print(f"[safelanding_v2] {_BANK[key].shape[0]} manifold rows ({split}) "
          f"<- {_MANIFOLD_PATH}")
  return _BANK[key]


def reset_from_manifold(env, env_ids, asset_cfg=SceneEntityCfg("robot"),
                        split: str = "train"):
  """Reset event: place each resetting env at a random harvested manifold row
  (restored at its own x_rel, no momentum scaling)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  bank = _load_manifold(env, split)
  n = int(len(env_ids))
  pick = torch.randint(0, bank.shape[0], (n,), device=bank.device)
  rows = bank[pick].to(env.device)
  xt = rows[:, 0]                                   # spawn at the harvested x_rel
  ms = torch.ones(n, device=env.device)             # NO momentum scaling
  _restore(env, env_ids, rows, xt, ms)


def unitree_go2_safelanding_v2_env_cfg(play: bool = False, gap_width: float = 0.30,
                                       split: str = "train"
                                       ) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_landing_env_cfg(play=play, gap_width=gap_width)
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_from_manifold, mode="reset", params={"split": split})
  # _restore writes the FULL joint state -> drop the default-pose joint reset
  # (mirrors brake_or_jump; otherwise reset_robot_joints would overwrite it).
  cfg.events.pop("reset_robot_joints", None)
  if "push_robot" in cfg.events:
    cfg.events.pop("push_robot", None)
  return cfg
