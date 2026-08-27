"""sRAS Phase I: brake_or_jump reverse curriculum @ gap 0.30 with
INHERITED sets from the STAY game.

Phase II delivered Ω̂[stay] = viab(T\\F) as ``vstay.stay_kernel_margin(env)``
(k(s) >= 0 <=> the landed far-side state is persistable). Phase I re-solves the
reach-avoid crossing game with those sets INHERITED:

  T             = grounded past far_x(env) (= gap_width + 0.15 = 0.45)  (target region)
  reach target  = Ω̂[stay]                     (NOT the whole T region)
  failure set   = F ∪ (T \\ Ω̂[stay])          (landing outside the kernel is a FAILURE)

so that entering the target region OUTSIDE the kernel is a failure AT THE VALUE
LEVEL (kills the cross-then-fall tail) and the handover surface becomes a
DERIVED object (= Ω̂[stay]) instead of a hand-calibrated gate.

Margin construction (see the module functions below):

  k(s)   = stay_kernel_margin(env)                       # clamped [-3, 3]
  l'     = min( (x - far_x)/pos_norm, grounded_term, k ) # reach = Ω̂[stay]
  g'     = min( g_terrain_relative, doomed_T_term )      # failure = F ∪ (T\\Ω̂[stay])
           doomed_T_term = k       when (grounded AND past far_x)
                         = +CLAMP  otherwise (neutral)

  grounded_term = feet_in_contact - 0.5  (>=0 <=> a foot is loaded) — GUARDS the
  reach against firing mid-flight over T, where the distilled k (trained on LANDED
  states) is OOD. Before T the position term keeps l' < 0, so k there is harmless.

Plus a DOOMED-T failure TERMINATION (2-step debounce): the episode ends as a
failure when (grounded in T) AND (k < 0) persists for 2 consecutive control steps.
The conservative τ (stay game) can make a 1-step touchdown-transient misfire possible,
so the debounce tolerates it; the firing rate is reported at smoke time.

Everything else — env, gap 0.30, the handover spawn mixture (50% harvested
handover states D∈[0.30,0.55] + 50% curriculum bank), curriculum machinery — is
IDENTICAL to the hando task (``brake_or_jump_hando``). Only the (g, l) margin
and the added doomed-T termination differ. Existing tasks are untouched.
"""
from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from robot_safety_sandbox.margins import CLAMP, g_terrain_relative
from robot_safety_sandbox.envs.go2_gap.brake_or_jump import (
  _TOUCHDOWN_FORCE,
  far_x,
)
from robot_safety_sandbox.envs.go2_gap.brake_or_jump_hando import (
  reset_brake_or_jump_hando,
  unitree_go2_brake_or_jump_hando_env_cfg,
)
from robot_safety_sandbox.vstay import stay_kernel_margin

_POS_NORM = 0.20          # matches l_stable_far's position normalization
_DEBOUNCE = 2             # doomed-T termination fires after this many consecutive steps


def _feet_down_count(env) -> torch.Tensor:
  """Number of feet loaded above _TOUCHDOWN_FORCE, per env [N] (int). Same
  contact read as brake_or_jump._clean_flag (feet_ground_contact netforce)."""
  feet = env.scene["feet_ground_contact"]
  fmag = torch.norm(feet.data.force, dim=-1)        # [N, F] per-foot netforce mag
  foot_down = fmag > _TOUCHDOWN_FORCE
  while foot_down.dim() > 2:
    foot_down = foot_down.any(dim=-1)               # collapse any slot dim
  return foot_down.sum(dim=1)                        # [N] number of feet loaded


def _sras_signals(env) -> dict:
  """(x_rel, past_far, feet_down, grounded, in_T, k) for THIS step, memoized per
  ``common_step_counter`` so the termination (mjlab line 148) and the margin
  (line 152) read identical values — both run after the same physics step, before
  mjlab's in-step auto-reset. k = stay_kernel_margin (already clamped [-3, 3])."""
  tok = env.common_step_counter
  cache = getattr(env, "_sras_cache", None)
  if cache is not None and cache[0] == tok:
    return cache[1]
  robot = env.scene["robot"]
  x_rel = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  past_far = x_rel > far_x(env)
  feet_down = _feet_down_count(env)
  grounded = feet_down >= 1
  in_T = grounded & past_far
  k = stay_kernel_margin(env)                        # clamp=3.0 by default
  sig = {"x_rel": x_rel, "past_far": past_far, "feet_down": feet_down,
         "grounded": grounded, "in_T": in_T, "k": k}
  env._sras_cache = (tok, sig)
  return sig


def sras_margin_fn(env):
  """(g', l') for the inherited-set Phase-I game (see module docstring).

  l' = min( (x-far_x)/pos_norm, feet_down-0.5, k )         (reach = Ω̂[stay])
  g' = min( g_terrain_relative, k|_{in T} else +CLAMP )    (failure = F ∪ (T\\Ω̂))
  """
  s = _sras_signals(env)
  pos = (s["x_rel"] - far_x(env)) / _POS_NORM
  grounded_term = s["feet_down"].float() - 0.5
  l = torch.minimum(torch.minimum(pos, grounded_term), s["k"])

  g_terr = g_terrain_relative(env)
  neutral = torch.full_like(s["k"], CLAMP)
  doomed_T = torch.where(s["in_T"], s["k"], neutral)
  g = torch.minimum(g_terr, doomed_T)

  return g.clamp(-CLAMP, CLAMP), l.clamp(-CLAMP, CLAMP)


sras_margin_fn.has_target = True   # a real reach target (Ω̂[stay]); ReachAvoid learner


def doomed_T_termination(env, env_ids=None) -> torch.Tensor:
  """Failure termination: fire when (grounded in T) AND k < 0 persists for
  ``_DEBOUNCE`` consecutive control steps (debounced against touchdown-transient
  misfires). Non-time_out DoneTerm -> contributes to ``terminated`` (the reward
  hook anchors g to the failure margin). ``_doomed_count`` is zeroed per-reset in
  reset_sras. The per-step fire mask is stashed on env for smoke instrumentation."""
  s = _sras_signals(env)
  doomed_now = s["in_T"] & (s["k"] < 0.0)
  if not hasattr(env, "_doomed_count"):
    env._doomed_count = torch.zeros(env.num_envs, dtype=torch.long,
                                    device=env.device)
  env._doomed_count = torch.where(doomed_now, env._doomed_count + 1,
                                  torch.zeros_like(env._doomed_count))
  fire = env._doomed_count >= _DEBOUNCE
  env._sras_last_fire = fire
  return fire


def reset_sras(env, env_ids, asset_cfg=SceneEntityCfg("robot")):
  """The hando mixture reset, plus zeroing the doomed-T debounce counter for
  the resetting envs (fresh per episode)."""
  reset_brake_or_jump_hando(env, env_ids, asset_cfg=asset_cfg)
  if env_ids is None:
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    ids = env_ids.long()
  if not hasattr(env, "_doomed_count"):
    env._doomed_count = torch.zeros(env.num_envs, dtype=torch.long,
                                    device=env.device)
  env._doomed_count[ids] = 0


def unitree_go2_brake_or_jump_sras_env_cfg(play: bool = False,
                                           gap_width: float = 0.30
                                           ) -> ManagerBasedRlEnvCfg:
  """The hando env verbatim (same spawn mixture / curriculum / gap 0.30),
  plus the doomed-T failure termination. The margin (g', l') is set at the task
  (sras_margin_fn); the env only wires the reset-counter reset + the termination."""
  cfg = unitree_go2_brake_or_jump_hando_env_cfg(play=play, gap_width=gap_width)
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_sras, mode="reset", params={})
  cfg.terminations["sras_doomed_T"] = TerminationTermCfg(
    func=doomed_T_termination, params={})
  return cfg
