"""sRAS Phase II — the STAY (viability) game env.

Spawns the robot at HARVESTED LANDED states (the far-side T-region states the
deployed handover-composed system actually lands in, harvested offline)
and asks it to PERSIST without failing: avoid-only, mode SAFETY -> SafetyPPO1P,
so the learned value is V_stay = the viability value of the landed state. Its
non-negative super-level set (after conservative calibration) is Omega[stay] =
viab(T\\F), the sRAS Phase-II object.

Built on the brake_or_jump gap world (same terrain / obs / g as the crossing arm,
so V_stay transfers straight into the Phase-I margin), with three changes:
  - reset = restore a random harvested landed row (37-col bank format, via the
    shared `brake_or_jump._restore`, at its own x_rel with NO momentum scaling)
    -- the safelanding_v2 reset-from-bank pattern.
  - NO curriculum (persisting on flat far ground is not a curriculum problem).
  - episode ~5 s (persistence horizon).
FULL margin: `illegal_contact` stays ACTIVE (the default brake_or_jump env keeps
it -- only the belly-allowed EVAL harness pops it), because the stay claim is
strict: a belly/limb slam on the far side is a stay FAILURE.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from robot_safety_sandbox.envs.go2_gap.brake_or_jump import (
  _restore,
  unitree_go2_brake_or_jump_env_cfg,
)

_LANDED_PATH = os.environ.get(
  "SRAS_LANDED_STATES",
  os.path.expanduser(
    "~/artifacts/robot-safety-sandbox/stay-game/landed_states.npz"))
_BANK: dict = {}                # cache per (path, split) -> (rows tensor, global idx)


def _load_landed(env, split: str = "train"):
  """Return (rows[N,37] tensor, global_idx[N] tensor into the npz `rows`)."""
  key = (_LANDED_PATH, split)
  if key not in _BANK:
    d = np.load(_LANDED_PATH)
    n_all = len(d["rows"])
    if split == "train":
      gidx = d["train_idx"]
    elif split == "val":
      gidx = d["val_idx"]
    elif split == "all":
      gidx = np.arange(n_all)
    else:
      raise ValueError(f"split must be train|val|all, got {split!r}")
    rows = d["rows"][gidx]
    _BANK[key] = (
      torch.tensor(np.ascontiguousarray(rows), dtype=torch.float32,
                   device=env.device),
      torch.tensor(np.ascontiguousarray(gidx), dtype=torch.long,
                   device=env.device))
    print(f"[stay] {rows.shape[0]} landed rows ({split}) <- {_LANDED_PATH}",
          flush=True)
  return _BANK[key]


def reset_from_landed(env, env_ids, asset_cfg=SceneEntityCfg("robot"),
                      split: str = "train"):
  """Reset event: place each resetting env at a random harvested landed row
  (restored at its own x_rel, no momentum scaling)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  bank, gidx = _load_landed(env, split)
  n = int(len(env_ids))
  pick = torch.randint(0, bank.shape[0], (n,), device=bank.device)
  rows = bank[pick].to(env.device)
  xt = rows[:, 0]                                    # spawn at the harvested x_rel
  ms = torch.ones(n, device=env.device)              # NO momentum scaling
  _restore(env, env_ids, rows, xt, ms)
  # record which npz rows each env was spawned from (for offline V_stay mapping).
  if not hasattr(env, "_stay_pick_global"):
    env._stay_pick_global = torch.full((env.num_envs,), -1, dtype=torch.long,
                                       device=env.device)
  env._stay_pick_global[env_ids.long()] = gidx[pick]
  # brake_or_jump keeps per-env airborne-clean latches; reset them if present so
  # a stale latch from a prior episode never leaks (harmless for avoid-only, but
  # tidy). Guarded: the flags only exist once _ensure has run.
  if hasattr(env, "_td_og"):
    env._td_og[env_ids.long()] = False
  if hasattr(env, "_bodyplant"):
    env._bodyplant[env_ids.long()] = False


def unitree_go2_stay_env_cfg(play: bool = False, gap_width: float = 0.30,
                             split: str = "train") -> ManagerBasedRlEnvCfg:
  """The brake_or_jump gap world @ gap 0.30 (full margin, illegal_contact ACTIVE),
  reset from harvested landed rows, NO curriculum, 5 s episodes."""
  cfg = unitree_go2_brake_or_jump_env_cfg(play=play, gap_width=gap_width,
                                          clean=False)
  cfg.episode_length_s = 5.0
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_from_landed, mode="reset", params={"split": split})
  # _restore writes the full joint state -> the default-pose joint reset is
  # already popped by the base env; leave it.
  cfg.curriculum = {}                                # persisting is not a curriculum
  return cfg
