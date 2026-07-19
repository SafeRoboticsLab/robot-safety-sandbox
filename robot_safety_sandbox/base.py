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
from mjlab.managers.termination_manager import TerminationTermCfg

# The tensor bridge pairs with safety_sb3 learners; the NUMPY bridge (nominal
# task policies, vanilla SB3) must work without safety_sb3 installed.
try:
  from safety_sb3.tensor_env import TensorVecEnv
  _HAS_SAFETY_SB3 = True
except ImportError:  # nominal-only install: tensor path disabled
  TensorVecEnv = object
  _HAS_SAFETY_SB3 = False

TERMINAL_MARGIN = -0.1  # g anchor on failure terminations (pre-reset hook)

#: name of the uniform success DoneTerm added by end_criterion="reach-avoid"
SUCCESS_TERM = "zoo_reach_success"


def _cached_margins(env, margin_fn):
  """(g, l) for THIS step. mjlab computes terminations (line 148 of
  ManagerBasedRlEnv.step) BEFORE rewards (line 152), both after the single
  ``common_step_counter`` bump — so the reach-avoid success DoneTerm runs first
  and stashes the margins it computed under a step token; the reward hook then
  reuses them instead of calling the (geometry) margin_fn a second time. The
  token guards against reading a stale cache when no DoneTerm ran this step
  (failure/timeout mode), in which case we compute fresh — identical result,
  since margin_fn is a pure function of the (unchanged between 148 and 152)
  scene state."""
  cache = getattr(env, "_zoo_gl_cache", None)
  if cache is not None and cache[2] == env.common_step_counter:
    return cache[0], cache[1]
  g, l = margin_fn(env)
  env._zoo_gl_cache = (g, l, env.common_step_counter)
  return g, l


def zoo_reach_success(env, margin_fn=None) -> torch.Tensor:
  """Uniform reach-avoid success termination (added ONLY when the env's
  end_criterion is "reach-avoid"): fire when the target is reached while safe,
  i.e. g >= 0 AND l >= 0. Registered as a mjlab DoneTerm (non-time_out) so
  mjlab's internal auto-reset fires on the SAME step the target is reached —
  a bridge-side augmentation of ``terminated`` would reset one step late
  (mjlab already auto-reset during self.mj.step) and desync the rollout buffer.
  Computes (and caches) the margins here so the later reward hook need not
  recompute them (see :func:`_cached_margins`)."""
  g, l = _cached_margins(env, margin_fn)
  return (g >= 0.0) & (l >= 0.0)


def safety_margin_hook(env, margin_fn=None) -> torch.Tensor:
  """Reward-term hook: computes (g, l) BEFORE mjlab's auto-reset (terminal-
  correct) and stashes them in ``env.extras``. g is anchored to the terminal
  failure value on real terminations."""
  g, l = _cached_margins(env, margin_fn)
  failed = env.termination_manager.terminated
  # A reach-avoid SUCCESS is a terminated step too (its DoneTerm sets terminated),
  # but it must NOT be anchored to the failure value: its terminal reach-avoid
  # target min(l, g) is >= 0 (a win), and clamping g to -0.1 would flip it to a
  # loss. Exclude it. In failure/timeout mode the term is absent, so this is a
  # no-op and the anchor is bit-identical to before.
  if SUCCESS_TERM in env.termination_manager.active_terms:
    failed = failed & ~env.termination_manager.get_term(SUCCESS_TERM)
  g = torch.where(failed, torch.minimum(g, torch.full_like(g, TERMINAL_MARGIN)), g)
  # NaN SANITATION: a physics NaN makes margins NaN in the SAME step the
  # nan_detection termination fires; the reset happens after the reward is
  # recorded, so one NaN margin can poison the rollout buffer and NaN the
  # policy weights (box stage-2 run 10d774v3 died this way at ~75M despite
  # nan_term). Grade corrupted states as failures, never as poison.
  g = torch.nan_to_num(g, nan=TERMINAL_MARGIN, posinf=3.0, neginf=-3.0)
  l = torch.nan_to_num(l, nan=-3.0, posinf=3.0, neginf=-3.0)
  env.extras["zoo_g"] = g
  env.extras["zoo_l"] = l
  return g


def build_task_cfg(cfg_builder: Callable, margin_fn: Callable, num_envs: int,
                   drop_events: tuple[str, ...] = ("push_robot",),
                   dense: bool = False, end_criterion: str = "failure",
                   cfg_overrides: dict | None = None):
  """Assemble an mjlab cfg for the zoo: task cfg + the margin hook.

  ``drop_events`` removes events a learned adversary replaces (default: the
  random push). ``dense=True`` is the STAGE-1 locomotion mode: the reward stays
  the env's own dense reward stack (`_r`) and the safety hook is NOT added, so a
  stock PPO shapes a proper gait (reach-avoid g/l are not used here).

  ``end_criterion`` (see :data:`registry.END_CRITERIA`) sets WHEN episodes end,
  UNIFORMLY across tasks. The task cfg's own terminations already encode the
  FAILURE set (fell_over / illegal_contact / ... -> the reward hook anchors g<0
  there) plus the env timeout, so:
    * "failure"     : add nothing. The task's existing terminations stand as-is
                      -> bit-identical to pre-end_criterion behavior. Reaching
                      the target does not end the episode; the agent reaches
                      deeper (l keeps climbing to the g ceiling).
    * "reach-avoid" : ALSO add the uniform success DoneTerm (g>=0 AND l>=0) on
                      top of the failure terminations -> reach-and-stop.
    * "timeout"     : strip every non-time_out (failure) termination so ONLY the
                      env timeout ends episodes, and add no success term (pure
                      value-learning / diagnostic). g is still anchored nowhere,
                      so failures live on in the margin, they just don't reset.
  """
  # cfg_overrides: experiment-level env/task params forwarded to the cfg_builder
  # (partial call-kwargs override the task registration's baked values, e.g.
  # {"gate_close_rate": 0.003}). Fail loud on a param the cfg_builder rejects.
  try:
    cfg = cfg_builder(play=False, **(cfg_overrides or {}))
  except TypeError as e:
    if cfg_overrides:
      raise SystemExit(
        f"[cfg_overrides] {list(cfg_overrides)} not all accepted by the task's "
        f"cfg_builder: {e}") from e
    raise
  cfg.scene.num_envs = int(num_envs)
  if cfg.events is not None:
    for e in drop_events:
      cfg.events.pop(e, None)
  if not dense:
    if margin_fn is None:
      raise ValueError(
        "margin_fn=None with dense=False: nominal tasks (kind='nominal') "
        "train on the dense env reward — use train_nominal.py / dense mode.")
    cfg.rewards["zoo_safety_hook"] = RewardTermCfg(
      func=safety_margin_hook, weight=1.0, params={"margin_fn": margin_fn})
    if end_criterion == "reach-avoid":
      cfg.terminations[SUCCESS_TERM] = TerminationTermCfg(
        func=zoo_reach_success, params={"margin_fn": margin_fn})
    elif end_criterion == "timeout":
      # Keep only the env timeout (time_out=True); drop the failure set so the
      # episode always runs to the horizon (diagnostic / pure value-learning).
      for name in [n for n, t in cfg.terminations.items()
                   if not getattr(t, "time_out", False)]:
        cfg.terminations.pop(name)
  return cfg


class _MjlabCore:
  """Shared mjlab plumbing for both bridges."""

  def _init_core(self, num_envs, device, cfg_builder, margin_fn, *,
                 ctrl_dim, dstb_dim, ctrl_gain, force_max, adversary,
                 adversary_body, render_mode, obs_key=None, dense_reward=False,
                 dstb_mode="wrench", dstb_gain=0.25, hybrid_skill=None,
                 latch_margin_fn=None, end_criterion="failure",
                 cfg_overrides=None):
    self.obs_key = obs_key  # resolved after first reset (auto-detect)
    self.end_criterion = str(end_criterion)
    self.ctrl_dim = int(ctrl_dim)
    self.dstb_dim = int(dstb_dim)
    self.ctrl_gain = float(ctrl_gain)
    self.force_max = float(force_max)
    # dstb channel: "wrench" = external force on adversary_body (legged tasks);
    # "action" = ACTION-ADDITIVE disturbance, ctrl += dstb_gain * a_dstb — the
    # Robust-Gymnasium / classic-control ISAACS convention (e.g. hopper +-25%
    # of the ctrl bound). MuJoCo clamps ctrl to ctrlrange afterward.
    self.dstb_mode = str(dstb_mode)
    self.dstb_gain = float(dstb_gain)
    self.adversary = bool(adversary)
    self.dense_reward = bool(dense_reward)  # Stage-1: train on the env reward
    self.render_mode = render_mode
    self.mj = ManagerBasedRlEnv(
      cfg=build_task_cfg(cfg_builder, margin_fn, num_envs, dense=dense_reward,
                         end_criterion=self.end_criterion,
                         cfg_overrides=cfg_overrides),
      device=device, render_mode="rgb_array" if render_mode else None)
    self._robot = self.mj.scene["robot"]
    if self.dstb_mode == "wrench":
      body_ids, _ = self._robot.find_bodies(adversary_body)
      self._adv_body_ids = list(body_ids)
    else:
      self._adv_body_ids = []  # action-mode dstb needs no body
    self._all_ids = torch.arange(int(num_envs), device=device)
    self._zero_wrench = torch.zeros((int(num_envs), 1, 3), device=device)
    self._log: dict = {}
    # HYBRID ROLLOUTS (funnel composition): when hybrid_skill is set, any env
    # whose LAST-step reach margin l went positive is latched to the FROZEN
    # skill's deterministic actions until its episode ends. The RA learner
    # then optimizes the value of the COMPOSED system (approach policy ->
    # frozen skill), which is exactly what the l = V_hat certificate certifies.
    self.hybrid_skill = hybrid_skill
    # dense-mode latch source (v10.3): geometric proposal margin computed for
    # the hybrid latch ONLY — never a reward term (anti-Goodhart split).
    self.latch_margin_fn = latch_margin_fn
    self._hyb = None       # lazy (policy, obs_mean, obs_var)
    self._hyb_latch = torch.zeros(int(num_envs), dtype=torch.bool, device=device)
    self._last_l = None
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
    # Cache the CURRENT raw actor obs on the mjlab env so margin fns that need
    # the observation (e.g. l_vhat) can read it INSIDE the reward hook, where
    # this step's obs are not computed yet (1-step lag, see l_vhat).
    self.mj._zoo_last_obs = obs0.float()
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

  def _hyb_load(self):
    """Lazy-load the frozen hybrid skill (policy + its frozen obs normalizer).
    Relative paths resolve against the zoo repo root."""
    if self._hyb is None:
      import os
      from safety_sb3 import SafetyPPO
      root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      d = os.path.expanduser(self.hybrid_skill)
      if not os.path.isabs(d):
        d = os.path.join(root, d)
      device = self.mj.device
      pol = SafetyPPO.load(os.path.join(d, "final_model.zip"), device=device,
                           custom_objects={"tensorboard_log": None}).policy
      pol.set_training_mode(False)
      st = torch.load(os.path.join(d, "tensornormalize.pt"),
                      map_location=device, weights_only=True)
      self._hyb = (pol, st["obs_mean"].to(device), st["obs_var"].to(device))
      print(f"[hybrid] frozen skill loaded from {d}")
    return self._hyb

  def _mj_step(self, actions: torch.Tensor):
    ctrl = actions[:, :self.ctrl_dim] * self.ctrl_gain
    if self.adversary:
      if self.dstb_mode == "action":
        ctrl = ctrl + self.dstb_gain * actions[:, self.ctrl_dim:]
      else:
        self._apply_dstb(actions[:, self.ctrl_dim:])
    if self.hybrid_skill is not None and self._last_l is not None:
      # latch on LAST step's l (computed inside the previous env.step); once
      # latched, the frozen skill controls the env until its episode ends.
      self._hyb_latch |= self._last_l > 0.0
      if bool(self._hyb_latch.any()):
        pol, h_mean, h_var = self._hyb_load()
        with torch.no_grad():
          nobs = torch.clamp(
            (self.mj._zoo_last_obs - h_mean) / torch.sqrt(h_var + 1e-8),
            -10.0, 10.0)
          a_f = torch.clamp(pol._predict(nobs, deterministic=True), -1.0, 1.0)
        ctrl = torch.where(self._hyb_latch.unsqueeze(-1),
                           a_f * self.ctrl_gain, ctrl)
    obs_dict, _r, terminated, truncated, extras = self.mj.step(ctrl)
    obs = obs_dict[self.obs_key].float()
    self.mj._zoo_last_obs = obs
    if self.dense_reward:
      # Stage-1 locomotion: the reward IS the env's dense reward stack (_r); no
      # safety g/l (a stock PPO with standard GAE shapes the gait).
      g = _r.float()
      if self.latch_margin_fn is not None:
        l = self.latch_margin_fn(self.mj).float()
      else:
        l = torch.zeros_like(g)
    else:
      g = extras["zoo_g"].float()
      l = extras["zoo_l"].float()
    if self.hybrid_skill is not None:
      done = terminated | truncated
      self._hyb_latch &= ~done
      self._last_l = l
      self._log["hybrid/latched_frac"] = float(self._hyb_latch.float().mean())
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
               render_mode=None, obs_key=None, dense_reward=False,
               dstb_mode="wrench", dstb_gain=0.25, hybrid_skill=None,
                 latch_margin_fn=None, end_criterion="failure",
               cfg_overrides=None):
    if not _HAS_SAFETY_SB3:
      raise ImportError(
        "safety_sb3 is required for the tensor bridge (pip install it or put "
        "the safety-stable-baselines repo on sys.path); the numpy bridge "
        "(make_numpy) works with vanilla stable_baselines3 only.")
    obs_space, act_space = self._init_core(
      num_envs, device, cfg_builder, margin_fn, ctrl_dim=ctrl_dim,
      dstb_dim=dstb_dim, ctrl_gain=ctrl_gain, force_max=force_max,
      adversary=adversary, adversary_body=adversary_body,
      render_mode=render_mode, obs_key=obs_key, dense_reward=dense_reward,
      dstb_mode=dstb_mode, dstb_gain=dstb_gain, hybrid_skill=hybrid_skill,
      latch_margin_fn=latch_margin_fn, end_criterion=end_criterion,
      cfg_overrides=cfg_overrides)
    TensorVecEnv.__init__(self, int(num_envs), obs_space, act_space, device)

  def reset(self) -> torch.Tensor:
    obs_dict, _ = self.mj.reset()
    obs = obs_dict[self.obs_key].float()
    self.mj._zoo_last_obs = obs
    self._hyb_latch.zero_()
    self._last_l = None
    return obs

  def step_tensor(self, actions: torch.Tensor):
    obs, g, terminated, truncated, l = self._mj_step(actions)
    # NaN sanitation (see safety_margin_hook): a NaN obs entering the rollout
    # buffer NaNs the policy update even though the env resets via nan_term.
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
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
               obs_key=None, dense_reward=False, dstb_mode="wrench",
               dstb_gain=0.25, hybrid_skill=None, latch_margin_fn=None,
               end_criterion="failure", cfg_overrides=None):
    obs_space, act_space = self._init_core(
      num_envs, device, cfg_builder, margin_fn, ctrl_dim=ctrl_dim,
      dstb_dim=dstb_dim, ctrl_gain=ctrl_gain, force_max=force_max,
      adversary=adversary, adversary_body=adversary_body,
      render_mode=render_mode, obs_key=obs_key, dense_reward=dense_reward,
      dstb_mode=dstb_mode, dstb_gain=dstb_gain, hybrid_skill=hybrid_skill,
      latch_margin_fn=latch_margin_fn, end_criterion=end_criterion,
      cfg_overrides=cfg_overrides)
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
