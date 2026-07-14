"""Base classes: run any mjlab task with safety_sb3 (tensor or numpy path).

The zoo's env contract (matches what every safety_sb3 learner consumes):

    reward   = g(s)   the physical safety margin  (NEVER normalize/reshape it)
    l_x      = l(s)   the target margin           (zeros for avoid-only tasks)
    dones    = terminated | truncated  (mjlab auto-resets internally)
    timeouts = truncated & ~terminated (no value bootstrap: g is absolute)

A task is fully specified by two callables (see :mod:`registry`):

    cfg_builder(play: bool) -> ManagerBasedRlEnvCfg   # the mjlab env
    margin_fn(env)          -> (g, l) batched tensors # the reach-avoid margins

Everything else here is plumbing: :class:`MjlabTensorSafetyEnv` exposes the
task as a safety_sb3 ``TensorVecEnv`` (GPU-resident, ~50k steps/s on a 12GB
card at 2048 envs); :class:`MjlabNumpySafetyEnv` as a classic SB3 ``VecEnv``
(for the SAC family / plain SB3 tooling). Curriculum levels and task metrics
(mjlab ``extras['log']``) are forwarded via ``metrics()`` so training logs
always show curriculum progression — silent curriculum stalls were the single
biggest source of lost weeks in the reference tasks.

Porting a new task = write a cfg_builder (spawn events + curricula, plain
mjlab) + a margin_fn (compose from :mod:`margins`), then ``register()`` it.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from gymnasium import spaces

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg

from safety_sb3.tensor_env import TensorVecEnv

TERMINAL_MARGIN = -0.1  # g anchor on failure terminations (pre-reset hook)


def safety_margin_hook(env, margin_fn=None) -> torch.Tensor:
  """Reward-term hook: computes (g, l) BEFORE mjlab's auto-reset (terminal-
  correct) and stashes them in ``env.extras``. g is anchored to the terminal
  failure value on real terminations."""
  g, l = margin_fn(env)
  failed = env.termination_manager.terminated
  g = torch.where(failed, torch.minimum(g, torch.full_like(g, TERMINAL_MARGIN)), g)
  env.extras["zoo_g"] = g
  env.extras["zoo_l"] = l
  return g


def build_task_cfg(cfg_builder: Callable, margin_fn: Callable, num_envs: int,
                   drop_events: tuple[str, ...] = ("push_robot",)):
  """Assemble an mjlab cfg for the zoo: task cfg + the margin hook.

  ``drop_events`` removes events a learned adversary replaces (default: the
  random push)."""
  cfg = cfg_builder(play=False)
  cfg.scene.num_envs = int(num_envs)
  if cfg.events is not None:
    for e in drop_events:
      cfg.events.pop(e, None)
  cfg.rewards["zoo_safety_hook"] = RewardTermCfg(
    func=safety_margin_hook, weight=1.0, params={"margin_fn": margin_fn})
  return cfg


class _MjlabCore:
  """Shared mjlab plumbing for both bridges."""

  def _init_core(self, num_envs, device, cfg_builder, margin_fn, *,
                 ctrl_dim, dstb_dim, ctrl_gain, force_max, adversary,
                 adversary_body, render_mode, obs_key=None):
    self.obs_key = obs_key  # resolved after first reset (auto-detect)
    self.ctrl_dim = int(ctrl_dim)
    self.dstb_dim = int(dstb_dim)
    self.ctrl_gain = float(ctrl_gain)
    self.force_max = float(force_max)
    self.adversary = bool(adversary)
    self.render_mode = render_mode
    self.mj = ManagerBasedRlEnv(
      cfg=build_task_cfg(cfg_builder, margin_fn, num_envs),
      device=device, render_mode="rgb_array" if render_mode else None)
    self._robot = self.mj.scene["robot"]
    body_ids, _ = self._robot.find_bodies(adversary_body)
    self._adv_body_ids = list(body_ids)
    self._all_ids = torch.arange(int(num_envs), device=device)
    self._zero_wrench = torch.zeros((int(num_envs), 1, 3), device=device)
    self._log: dict = {}
    obs_dict, _ = self.mj.reset()
    if self.obs_key is None:
      # Obs group naming is per-task-family ("proprioception" in the parkour
      # cfgs, "actor"/"policy" in velocity-style cfgs). Auto-detect the actor
      # group; pass obs_key explicitly to override.
      for cand in ("proprioception", "actor", "policy"):
        if cand in obs_dict:
          self.obs_key = cand
          break
      else:
        self.obs_key = next(iter(obs_dict.keys()))
    obs0 = obs_dict[self.obs_key]
    obs_space = spaces.Box(-np.inf, np.inf, shape=(int(obs0.shape[1]),),
                           dtype=np.float32)
    act_dim = self.ctrl_dim + (self.dstb_dim if self.adversary else 0)
    act_space = spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
    return obs_space, act_space

  def _apply_dstb(self, a_dstb: torch.Tensor) -> None:
    unit = a_dstb / a_dstb.norm(dim=1, keepdim=True).clamp_min(1e-6)
    mag = self.force_max
    scale = getattr(self, "force_scale", None)  # per-env (survival curriculum)
    if scale is not None:
      mag = mag * scale.reshape(-1, 1)
    forces = (unit * mag).reshape(-1, 1, 3)
    self._robot.write_external_wrench_to_sim(
      forces, self._zero_wrench, body_ids=self._adv_body_ids,
      env_ids=self._all_ids)

  def _mj_step(self, actions: torch.Tensor):
    if self.adversary:
      self._apply_dstb(actions[:, self.ctrl_dim:])
    ctrl = actions[:, :self.ctrl_dim] * self.ctrl_gain
    obs_dict, _r, terminated, truncated, extras = self.mj.step(ctrl)
    obs = obs_dict[self.obs_key].float()
    g = extras["zoo_g"].float()
    l = extras["zoo_l"].float()
    log = extras.get("log", {})
    for k, v in log.items():
      try:
        self._log[k] = float(v)
      except (TypeError, ValueError):
        pass
    return obs, g, terminated, truncated, l

  def metrics(self) -> dict[str, float]:
    out, self._log = self._log, {}
    return out

  def render(self):
    return self.mj.render()

  def close(self):
    try:
      self.mj.close()
    except Exception:
      pass


class MjlabTensorSafetyEnv(_MjlabCore, TensorVecEnv):
  """GPU-resident bridge (primary): torch end-to-end, no numpy bounce.
  Pair with safety_sb3 SafetyPPO / ReachAvoidPPO (auto-detected)."""

  def __init__(self, num_envs=2048, device="cuda:0", *, cfg_builder,
               margin_fn, ctrl_dim=12, dstb_dim=3, ctrl_gain=3.0,
               force_max=50.0, adversary=False, adversary_body="base_link",
               render_mode=None, obs_key=None):
    obs_space, act_space = self._init_core(
      num_envs, device, cfg_builder, margin_fn, ctrl_dim=ctrl_dim,
      dstb_dim=dstb_dim, ctrl_gain=ctrl_gain, force_max=force_max,
      adversary=adversary, adversary_body=adversary_body,
      render_mode=render_mode, obs_key=obs_key)
    TensorVecEnv.__init__(self, int(num_envs), obs_space, act_space, device)

  def reset(self) -> torch.Tensor:
    obs_dict, _ = self.mj.reset()
    return obs_dict[self.obs_key].float()

  def step_tensor(self, actions: torch.Tensor):
    obs, g, terminated, truncated, l = self._mj_step(actions)
    dones = terminated | truncated
    timeouts = truncated & ~terminated
    return obs, g, dones, timeouts, l


from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class MjlabNumpySafetyEnv(_MjlabCore, VecEnv):
  """Classic numpy SB3 ``VecEnv`` bridge (for the SAC family / stock SB3
  tooling). Slower (device<->host each step); prefer the tensor bridge for
  on-policy training."""

  def __init__(self, num_envs=64, device="cuda:0", *, cfg_builder, margin_fn,
               ctrl_dim=12, dstb_dim=3, ctrl_gain=3.0, force_max=50.0,
               adversary=False, adversary_body="base_link", render_mode=None,
               obs_key=None):
    obs_space, act_space = self._init_core(
      num_envs, device, cfg_builder, margin_fn, ctrl_dim=ctrl_dim,
      dstb_dim=dstb_dim, ctrl_gain=ctrl_gain, force_max=force_max,
      adversary=adversary, adversary_body=adversary_body,
      render_mode=render_mode, obs_key=obs_key)
    self._device = device
    VecEnv.__init__(self, int(num_envs), obs_space, act_space)
    self._actions = None

  def reset(self):
    obs_dict, _ = self.mj.reset()
    return obs_dict[self.obs_key].float().cpu().numpy()

  def step_async(self, actions):
    self._actions = np.asarray(actions, dtype=np.float32)

  def step_wait(self):
    a = torch.as_tensor(self._actions, device=self._device)
    obs, g, terminated, truncated, l = self._mj_step(a)
    term = terminated.cpu().numpy()
    trunc = truncated.cpu().numpy()
    dones = np.logical_or(term, trunc)
    obs_np = obs.cpu().numpy()
    l_np = l.cpu().numpy()
    infos = []
    for i in range(self.num_envs):
      info = {"l_x": float(l_np[i])}
      if dones[i]:
        info["terminal_observation"] = obs_np[i]
        info["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
      infos.append(info)
    return obs_np, g.cpu().numpy(), dones, infos

  # VecEnv boilerplate
  def _indices(self, indices):
    if indices is None:
      return range(self.num_envs)
    return [indices] if isinstance(indices, int) else indices

  def get_attr(self, attr_name, indices=None):
    return [getattr(self, attr_name, None) for _ in self._indices(indices)]

  def set_attr(self, attr_name, value, indices=None):
    setattr(self, attr_name, value)

  def env_method(self, method_name, *args, indices=None, **kwargs):
    return [None for _ in self._indices(indices)]

  def env_is_wrapped(self, wrapper_class, indices=None):
    return [False for _ in self._indices(indices)]
