"""brake_or_jump reverse curriculum @ gap 0.30 with a HANDOVER-STATE
spawn mixture that EXTENDS the arm's origination range from D<=0.35 to D>=0.45.

The origination wall: the JUMPABLE decision window (arm originates a crossing) is D<=0.35
and the full-margin BRAKEABLE window is D>=0.40 — DISJOINT. The named lever is a
stronger jumper that originates from >=0.45 m out. This is NOT a new skill (Stage-2's
approach-from-rest failed): the WALKER still does the approach; the arm only needs
to originate from 0.40-0.55 m out at the walker's real ~0.8 m/s handover momentum —
states 0.1-0.2 m upstream of ones the RAS arm already handles at 0.77 success. A pure
spawn-DISTRIBUTION extension.

Reset mixture (global, per reset):
  50% HANDOVER  — a harvested real walking-handover state (scratchpad/e098_harvest.py),
                  uniform over D in {0.30..0.55}, restored at its own x_rel with its
                  own ~0.8 m/s momentum (mom_scale=1). These are the origination
                  targets, L-independent.
  50% CURRICULUM — the existing reverse-curriculum bank behaviour (`_spawn_level`
                  at the current level, incl. committed bands at low L), which keeps
                  committed-band completion competence.

Margin / l / g / curriculum gate are UNCHANGED (`brake_or_jump_levels`). Only the
reset event differs from the naive-RA `_w30` env. Trained BARE (ReachAvoidPPO1P) —
no lander / gate / policy_mask — since the arm deploys bare inside the D-gate.
"""
from __future__ import annotations

import os

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from robot_safety_sandbox.envs.go2_gap.brake_or_jump import (
  D,
  _ensure,
  _restore,
  _spawn_level,
  unitree_go2_brake_or_jump_env_cfg,
)

_HANDO_PATH = os.environ.get(
  "RAAS_HANDOVER_BANK",
  os.path.expanduser(
    "~/artifacts/robot-safety-sandbox/handover-range/handover_bank.pt"))
_F_HANDO = 0.50
_HANDO_BANK: dict = {}


def _hando_bank(env) -> torch.Tensor:
  """The harvested handover states as a [N, 37] bank-format tensor (cached).
  Balanced across D so a uniform row draw is uniform over D in {0.30..0.55}."""
  if _HANDO_PATH not in _HANDO_BANK:
    d = torch.load(_HANDO_PATH, map_location=env.device, weights_only=True)
    _HANDO_BANK[_HANDO_PATH] = d["bank"].float().to(env.device)
    print(f"[brake_or_jump_hando] {d['bank'].shape[0]} handover states "
          f"(uniform over D={d.get('ds')}, {d.get('per_D')}/D) <- {_HANDO_PATH}",
          flush=True)
  return _HANDO_BANK[_HANDO_PATH]


def _spawn_handover(env, ids):
  """Restore a random harvested handover state at its own harvested x_rel (~ -D)
  with its own momentum (no scaling), via the shared bank `_restore`."""
  bank = _hando_bank(env)
  dev = env.device
  n = int(len(ids))
  pick = torch.randint(0, bank.shape[0], (n,), device=bank.device)
  rows = bank[pick].to(dev)
  _restore(env, ids, rows, rows[:, 0], torch.ones(n, device=dev))


def reset_brake_or_jump_hando(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """Mixture reset: 50% harvested handover states (uniform over D), 50% the
  existing curriculum spawn. peek/level logic mirrors reset_brake_or_jump; the
  curriculum gate is unchanged."""
  _ensure(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  dev = env.device
  # peek/level logic mirrors reset_brake_or_jump exactly (gate unchanged).
  peek = torch.rand(len(env_ids), device=dev) < 0.20
  env._peek[env_ids] = peek
  lvl = torch.where(peek, min(env._L + 1, D), env._L).long()

  hando = torch.rand(len(env_ids), device=dev) < _F_HANDO
  curric = ~hando

  # per-env spawn-kind record (0 curriculum, 1 handover) for smoke/analysis.
  if not hasattr(env, "_hando_kind"):
    env._hando_kind = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
  env._hando_kind[env_ids.long()] = hando.long()

  if bool(curric.any()):
    _spawn_level(env, env_ids[curric], lvl[curric])
  if bool(hando.any()):
    _spawn_handover(env, env_ids[hando])

  # airborne-clean latches: fresh per episode (mirrors reset_brake_or_jump).
  env._td_og[env_ids.long()] = False
  env._bodyplant[env_ids.long()] = False


def unitree_go2_brake_or_jump_hando_env_cfg(play: bool = False,
                                            gap_width: float = 0.30
                                            ) -> ManagerBasedRlEnvCfg:
  """The naive-RA `_w30` env verbatim (curriculum `brake_or_jump_levels`, margins
  set at the task), with the reset event swapped for the handover-state mixture."""
  cfg = unitree_go2_brake_or_jump_env_cfg(play=play, gap_width=gap_width,
                                          clean=False)
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_brake_or_jump_hando, mode="reset", params={})
  return cfg
