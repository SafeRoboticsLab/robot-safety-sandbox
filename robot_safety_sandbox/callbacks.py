"""Training callbacks for zoo runs: wandb videos, normalizer persistence,
adversary force ramp, warm-start normalizer freeze.

All battle-tested on the Go2 gap pipeline; see PORTING.md for when to use
each. Everything logs through SB3's logger (wandb via sync_tensorboard).
"""

from __future__ import annotations

import math
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class StdFloorCallback(BaseCallback):
  """Bound the policy action log_std into [min_std, max_std] at each rollout
  start (right after the prior update). The floor prevents the premature std
  COLLAPSE that freezes a policy in a sub-optimal gait; the ceiling prevents the
  entropy-bonus log-std RUNAWAY that inflates std into aggressive thrashing
  (robots fall over -> ep_len crashes). Together = sustained BOUNDED exploration."""

  def __init__(self, min_std=0.2, max_std=None):
    super().__init__()
    self.min_log = math.log(min_std)
    self.max_log = math.log(max_std) if max_std is not None else None

  def _on_rollout_start(self):
    import torch as th
    ls = getattr(self.model.policy, "log_std", None)
    if ls is not None:
      with th.no_grad():
        ls.clamp_(min=self.min_log, max=self.max_log)

  def _on_step(self):
    return True


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
    # mjlab renders the tracked env + its nearest neighbors in ONE scene (a herd,
    # like the unitree-rl-mjlab videos). Draw all the eval envs together.
    try:
      self._env.mj.cfg.viewer.max_extra_envs = max(1, int(self._env.num_envs) - 1)
    except Exception:
      pass

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
      frames.append(np.asarray(self._env.render()))  # native herd render
    vid = np.stack(frames).transpose(0, 3, 1, 2)
    wandb.log({"eval/video": wandb.Video(vid, fps=30, format="mp4")},
              step=self.num_timesteps)


class DenseMetricsCallback(BaseCallback):
  """Forward the numpy bridge's ``env.metrics()`` (mjlab per-term reward /
  termination / curriculum logs) to the SB3 logger -> wandb, once per rollout.

  Stock SB3 PPO knows nothing about the bridge's ``metrics()`` (the tensor-path
  SafetyPPO drains it itself), so without this the per-term reward breakdown is
  collected and silently dropped. Reachable through the VecMonitor/VecNormalize
  wrappers via VecEnvWrapper attribute forwarding."""

  def _on_rollout_end(self):
    fn = getattr(self.model.env, "metrics", None)
    if not callable(fn):
      return
    for k, v in (fn() or {}).items():
      self.logger.record(f"env/{k}", float(v))

  def _on_step(self):
    return True


class DenseVideoWandbCallback(BaseCallback):
  """eval/video for the STAGE-1 dense walker (stock SB3 PPO on the NUMPY bridge).

  Renders on a SEPARATE tensor env built in DENSE mode (no force / no safety
  hook), so the clip shows the UNAIDED gait. Bounces the render env's tensor obs
  -> numpy, normalizes with the training VecNormalize, predicts with the stock
  policy, steps the tensor env.

  The trigger fires on ROLLOUT boundaries (once per policy update), not per env
  step: within a rollout the policy is frozen, so a clip can only change after an
  update -> one-per-rollout is the finest MEANINGFUL cadence. At N envs a rollout
  advances num_timesteps by N*n_steps, so any ``interval`` below that resolves to
  "every rollout"."""

  def __init__(self, eval_env_fn, interval=10_000_000, video_len=300):
    super().__init__()
    self.eval_env_fn, self.interval, self.video_len = eval_env_fn, interval, video_len
    self._env = None
    self._last = -10**15

  def _on_training_start(self):
    self._env = self.eval_env_fn()
    try:
      self._env.mj.cfg.viewer.max_extra_envs = max(1, int(self._env.num_envs) - 1)
    except Exception:
      pass

  def _on_step(self):
    return True

  def _on_rollout_end(self):
    if self.num_timesteps - self._last >= self.interval:
      self._last = self.num_timesteps
      try:
        self._log_video()
      except Exception as e:  # a bad clip must never kill a long train
        print(f"[dense-video] skipped: {e}")

  def _log_video(self):
    import torch as th
    import wandb
    vn = self.model.env if hasattr(self.model.env, "normalize_obs") else None
    obs = self._env.reset()
    dev = obs.device
    frames = []
    for _ in range(self.video_len):
      o = obs.detach().cpu().numpy()
      if vn is not None:
        o = vn.normalize_obs(o)
      act, _ = self.model.predict(o, deterministic=True)  # stock PPO, numpy
      obs, _g, _d, _t, _l = self._env.step_tensor(
        th.as_tensor(np.clip(act, -1.0, 1.0), dtype=th.float32, device=dev))
      frames.append(np.asarray(self._env.render()))  # native herd render
    vid = np.stack(frames).transpose(0, 3, 1, 2)
    wandb.log({"eval/video": wandb.Video(vid, fps=30, format="mp4")},
              step=self.num_timesteps)


class VecNormSaveCallback(BaseCallback):
  """Save the numpy VecNormalize obs stats alongside periodic checkpoints.

  The stock CheckpointCallback saves only the policy; without the matching
  obs-normalizer a mid-run checkpoint can't be evaluated (or handed to Stage 2)
  faithfully. Writes ``vecnormalize_<step>.pkl`` next to the model checkpoints."""

  def __init__(self, outdir, save_freq_steps=10_000_000):
    super().__init__()
    self.outdir, self.freq = outdir, save_freq_steps
    self._last = 0

  def _on_step(self):
    if self.num_timesteps - self._last >= self.freq:
      self._last = self.num_timesteps
      if hasattr(self.model.env, "save"):
        os.makedirs(self.outdir, exist_ok=True)
        self.model.env.save(os.path.join(
          self.outdir, f"vecnormalize_{self.num_timesteps}.pkl"))
    return True


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


class FwdForceAnnealCallback(BaseCallback):
  """Anneal the crawl constant forward-current force (bootstraps forward
  locomotion) from ``start`` N -> 0 over the run: hold ``start`` for ``hold_steps``
  (let the gait form), then linearly decay to 0 over ``anneal_steps``, then 0.
  Sets env._fwd_force_scale, read by crawl_duck_margins. Final policy crawls
  unaided (the 0-force ablation showed it already partly can)."""

  def __init__(self, start=15.0, hold_steps=40_000_000, anneal_steps=160_000_000):
    super().__init__()
    self.start, self.hold, self.anneal = start, hold_steps, anneal_steps
    self._mj = None

  def _on_training_start(self):
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._mj = getattr(e, "mj", None)

  def _on_step(self):
    t = self.num_timesteps
    if t < self.hold:
      val = self.start
    elif t < self.hold + self.anneal:
      val = self.start * (1.0 - (t - self.hold) / self.anneal)
    else:
      val = 0.0
    if self._mj is not None:
      self._mj._fwd_force_scale = val
    self.logger.record("crawl/fwd_force", float(val))
    return True


class GaitThreshRampCallback(BaseCallback):
  """Trot-threshold curriculum for the crawl gait-l. Keeps env._gait_thresh just
  above the achieved trot-score (EMA) so the gait sub-margin stays a mild-negative,
  reachable gradient (a fixed high threshold is masked by gamma*V' -> no learning).
  Monotonic-up; creeps toward trot_ema + gap, capped at `cap`."""

  def __init__(self, start=0.56, cap=0.88, gap=0.05, ema=0.01, step=0.002):
    super().__init__()
    self.start, self.cap, self.gap, self.ema_a, self.step = start, cap, gap, ema, step
    self._mj = None
    self._ema = start

  def _on_training_start(self):
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._mj = getattr(e, "mj", None)
    if self._mj is not None:
      self._mj._gait_thresh = self.start

  def _on_step(self):
    if self._mj is None:
      return True
    tr = getattr(self._mj, "_gait_trot_last", None)
    if tr is not None:
      self._ema += self.ema_a * (float(tr) - self._ema)
      cur = getattr(self._mj, "_gait_thresh", self.start)
      target = min(self.cap, self._ema + self.gap)
      if target > cur:
        self._mj._gait_thresh = min(self.cap, cur + self.step)
    self.logger.record("crawl/gait_thresh",
                       float(getattr(self._mj, "_gait_thresh", self.start)))
    self.logger.record("crawl/trot_ema", float(self._ema))
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


class LAnnealCallback(BaseCallback):
  """Reach-set curriculum for warm-started reach-avoid: ramp ``env._l_alpha``
  from 0 -> 1 so the reach target ``l`` contracts from LOOSE (alpha=0, ~= the
  non-fallen set the avoid base already satisfies -> reach-avoid ~= avoid-only,
  no OOD jump) to STRICT (alpha=1, the true stay-in-place set). Holds alpha=0
  for ``hold_steps`` first (let the re-inflated policy re-settle onto the warm
  base), then linearly to 1 over ``anneal_steps``, then strict. Sets the attr
  the margin reads via ``getattr(env, '_l_alpha', 1.0)``. Same env-reach pattern
  as FwdForceAnnealCallback (through the bridge's ``.mj``)."""

  def __init__(self, anneal_steps=120_000_000, hold_steps=20_000_000):
    super().__init__()
    self.anneal = max(1, anneal_steps)
    self.hold = hold_steps
    self._mj = None

  def _on_training_start(self):
    e = self.model.env
    while hasattr(e, "venv"):
      e = e.venv
    self._mj = getattr(e, "mj", None)
    if self._mj is not None:
      self._mj._l_alpha = 0.0

  def _on_step(self):
    t = self.num_timesteps
    if t < self.hold:
      a = 0.0
    else:
      a = min(1.0, (t - self.hold) / self.anneal)
    if self._mj is not None:
      self._mj._l_alpha = a
    self.logger.record("curriculum/l_alpha", float(a))
    return True
