"""Training callbacks for zoo runs: wandb videos, normalizer persistence,
adversary force ramp, warm-start normalizer freeze.

All battle-tested on the Go2 gap pipeline; see PORTING.md for when to use
each. Everything logs through SB3's logger (wandb via sync_tensorboard).
"""

from __future__ import annotations

import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class VideoWandbCallback(BaseCallback):
  """Upload eval/video clips of the current deterministic policy.

  Uses a SEPARATE small render env (build with adversary off — show the
  deployable ctrl policy). First clip fires immediately (warm-start
  behavior), then every ``interval`` env-steps. Requires wandb.init'd."""

  def __init__(self, eval_env_fn, interval=25_000_000, video_len=300):
    super().__init__()
    self.eval_env_fn, self.interval, self.video_len = eval_env_fn, interval, video_len
    self._env = None
    self._last = -10**15

  def _on_training_start(self):
    self._env = self.eval_env_fn()

  def _on_step(self):
    if self.num_timesteps - self._last >= self.interval:
      self._last = self.num_timesteps
      self._log_video()
    return True

  def _log_video(self):
    import torch as th
    import wandb
    tvn = self.model.env if hasattr(self.model.env, "normalize_obs") else None
    obs = self._env.reset()
    frames = []
    for _ in range(self.video_len):
      o = obs if tvn is None else tvn.normalize_obs(obs)
      with th.no_grad():
        act = self.model.policy._predict(o, deterministic=True)
      obs, _g, _d, _t, _l = self._env.step_tensor(
        th.clamp(act, -1.0, 1.0))
      frames.append(np.asarray(self._env.render()))
    vid = np.stack(frames).transpose(0, 3, 1, 2)
    wandb.log({"eval/video": wandb.Video(vid, fps=30, format="mp4")},
              step=self.num_timesteps)


class TensorNormSaveCallback(BaseCallback):
  """Save TensorVecNormalize stats alongside periodic checkpoints (needed for
  faithful mid-run eval / warm-starting)."""

  def __init__(self, outdir, save_freq_steps=25_000_000):
    super().__init__()
    self.outdir, self.freq = outdir, save_freq_steps
    self._last = 0

  def _on_step(self):
    if self.num_timesteps - self._last >= self.freq:
      self._last = self.num_timesteps
      if hasattr(self.model.env, "save"):
        os.makedirs(self.outdir, exist_ok=True)
        self.model.env.save(os.path.join(
          self.outdir, f"tensornorm_{self.num_timesteps}.pt"))
    return True


class NormFreezeCallback(BaseCallback):
  """Freeze obs-normalizer updates for the first ``freeze_steps`` of a
  WARM-STARTED run: the inherited policy expects the loaded stats; letting
  them drift while the policy adapts to a new task distribution destabilizes
  the transfer."""

  def __init__(self, freeze_steps=5_000_000):
    super().__init__()
    self.freeze_steps = freeze_steps
    self._released = False

  def _on_training_start(self):
    if hasattr(self.model.env, "training"):
      self.model.env.training = False

  def _on_step(self):
    if not self._released and self.num_timesteps >= self.freeze_steps:
      self._released = True
      if hasattr(self.model.env, "training"):
        self.model.env.training = True
    return True


class PerEnvForceScaleCallback(BaseCallback):
  """rsl_rl-inc1b anti-collapse lever for adversarial (ISAACS) training: scale
  the adversary's force PER ENV by a survival curriculum — envs that fail get
  a weaker adversary (+recover), envs that survive get a stronger one. This is
  what stabilized the reference two-player game after pinned curricula and the
  global force ramp were not enough."""

  def __init__(self, step=0.05, lo=0.3, hi=1.0, init=0.5):
    super().__init__()
    self.step, self.lo, self.hi, self.init = step, lo, hi, init
    self._bridge = None
    self._scale = None

  def _on_training_start(self):
    import torch as th
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._bridge = e
    self._scale = th.full((e.num_envs,), self.init, device=e.device)
    e.force_scale = self._scale

  def _on_step(self):
    import torch as th
    dones = self.locals.get("dones")
    timeouts = self.locals.get("timeouts")
    if dones is None or not th.is_tensor(dones) or not bool(dones.any()):
      return True
    d = dones.bool()
    survived = d & timeouts.bool()
    failed = d & ~timeouts.bool()
    self._scale += self.step * survived.float() - self.step * failed.float()
    self._scale.clamp_(self.lo, self.hi)
    self.logger.record("isaacs/force_scale_mean", float(self._scale.mean()))
    return True


class ForceRampCallback(BaseCallback):
  """Adversarial (ISAACS) runs: ramp the adversary force from ``force_start``
  to ``force_max`` over ``ramp_steps`` so ctrl adapts to a strengthening
  attacker instead of collapsing under the full bound from step one."""

  def __init__(self, force_max=50.0, ramp_steps=1_000_000_000, force_start=8.0):
    super().__init__()
    self.fmax, self.ramp, self.fstart = force_max, ramp_steps, force_start
    self._bridge = None

  def _on_training_start(self):
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._bridge = e

  def _on_step(self):
    frac = min(1.0, self.num_timesteps / max(1, self.ramp))
    self._bridge.force_max = self.fstart + (self.fmax - self.fstart) * frac
    return True
