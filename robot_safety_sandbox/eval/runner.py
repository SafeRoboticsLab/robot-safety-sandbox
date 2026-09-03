"""The single evaluation rollout loop.

Every evaluation in this repo is the same loop -- ask the nominal, filter it,
maybe deliver a disturbance, step, measure -- and the four axes it composes
(environment / nominal / filter / metrics) are independent objects it does not
introspect. There is one branch in here that matters: whether an adversary is
configured, which decides whether the env is stepped with ``a_ctrl`` or with
``cat([a_ctrl, a_dstb])``.

The filter's ``reset(done)`` contract is honored every step (an uncleared latch
once left a fallback driving fresh episodes for a 77% livelock), and the
nominal's own episode-freshness flag is the PREVIOUS step's done mask, which is
what the least-restrictive intervention reseeds its margin history from.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch

from .metrics import Metric, StepRecord


class NominalPolicy:
  """A stock-SB3 nominal policy driven off the env's own obs group.

  Owns the numpy bounce and the VecNormalize statistics so the loop does not:
  the nominal is the one component that lives in SB3's numpy world while
  everything else is torch on the device.
  """

  def __init__(self, model, vecnormalize=None, device: str = "cuda:0",
               bound: float = 1.0):
    self.model, self.vn, self.device = model, vecnormalize, device
    # The action clamp MUST match what the policy was TRAINED under (the bridge's action_bound). A
    # policy trained unclamped (wide bound) but re-clipped to +-1 here would be a DIFFERENT policy at
    # eval than in training, silently corrupting every speed number (T006). Pass the spec's
    # interface.action.clip so the two read the same source; default 1.0 is the historical bound.
    self.bound = float(bound)

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    o = obs.detach().cpu().numpy()
    if self.vn is not None:
      o = self.vn.normalize_obs(o)
    a, _ = self.model.predict(o, deterministic=True)
    return torch.as_tensor(np.clip(a, -self.bound, self.bound), dtype=torch.float32,
                           device=self.device)


class ZeroNominal:
  """The do-nothing nominal (a filter-only / fallback-only control run)."""

  def __init__(self, num_envs: int, ctrl_dim: int, device: str):
    self.a = torch.zeros(num_envs, ctrl_dim, device=device)

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    return self.a


class TwinNominal:
  """The twin's OWN control policy, driven as the nominal.

  What a disturbance-effect probe evaluates: the deployable policy the two-
  player game produced, run against its own adversary at a swept strength. It
  reads the SAFETY obs group through the twin's normalizer, not the nominal
  group, because it is the twin.
  """

  #: read by run_eval: this nominal is fed the SAFETY group, not the nominal
  #: one (they are different groups whenever an env emits both).
  uses_safety_obs = True

  def __init__(self, fallback_fn, norm):
    self.fallback_fn, self.norm = fallback_fn, norm

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    return self.fallback_fn(s_obs=self.norm(obs.float()))


class VideoRecorder:
  """Frames from the live env, tinted by how much of the herd the filter drives.

  Border colour goes from green (every robot on the nominal) to red (every
  robot on the fallback), which is what makes a filter handover legible in a
  batched rollout. Requires the env to have been built with
  ``render_mode="rgb_array"``.
  """

  def __init__(self, path: str, fps: int = 30, border: int = 14):
    self.path, self.fps, self.border = path, fps, border
    self.frames: list = []

  def __call__(self, env, engaged: torch.Tensor) -> None:
    import numpy as np_
    frac = float(engaged.float().mean())
    f = np_.asarray(env.bridge.render()).copy()
    b = self.border
    c = np_.array([int(255 * frac), int(200 * (1 - frac)), 30], dtype=np_.uint8)
    f[:b, :] = c
    f[-b:, :] = c
    f[:, :b] = c
    f[:, -b:] = c
    self.frames.append(f)

  def save(self) -> None:
    if not self.frames:
      return
    import imageio
    imageio.mimwrite(self.path, self.frames, fps=self.fps, macro_block_size=1)
    print(f"[video] {len(self.frames)} frames -> {self.path}")


# --- disturbance sources -----------------------------------------------------

def dstb_none(num_envs: int, dstb_dim: int, device: str):
  z = torch.zeros(num_envs, dstb_dim, device=device)
  return lambda s_obs: z


def dstb_random(num_envs: int, dstb_dim: int, device: str):
  def fn(s_obs):
    return torch.rand(num_envs, dstb_dim, device=device) * 2.0 - 1.0
  return fn


def dstb_policy(dstb_fn):
  """The trained min-player: the worst-case attack the twin learned."""
  return lambda s_obs: dstb_fn(s_obs=s_obs)


DSTB_SOURCES = ("none", "random", "policy")


def make_dstb(kind: str, env, mods: Optional[dict] = None):
  """Resolve ``--dstb`` to a callable ``(s_obs) -> (N, dstb_dim)``."""
  if kind not in DSTB_SOURCES:
    raise ValueError(f"unknown --dstb {kind!r}; one of {DSTB_SOURCES}")
  if kind == "none":
    return dstb_none(env.num_envs, env.dstb_dim, env.device)
  if kind == "random":
    return dstb_random(env.num_envs, env.dstb_dim, env.device)
  if not mods or "dstb_fn" not in mods:
    raise SystemExit(
      "--dstb policy needs a two-player twin to attack with: the twin at "
      f"--twin ({(mods or {}).get('twin')}) has no disturbance actor. Point "
      "--dstb-twin at a *2P checkpoint, or use --dstb random.")
  return dstb_policy(mods["dstb_fn"])


# --- the loop ----------------------------------------------------------------

def run_eval(env, nominal, filt, metrics: Metric, *, steps: int,
             norm: Callable, dstb_fn: Optional[Callable] = None,
             dstb_scale: float = 1.0, no_filter: bool = False,
             command_surgery: Optional[Callable] = None,
             video: Optional[Callable] = None,
             seed: Optional[int] = None,
             progress_every: int = 0) -> dict:
  """Roll ``steps`` control steps and return the metric set's report.

  :param env: an :class:`~robot_safety_sandbox.eval.envs.EvalEnv`.
  :param nominal: ``(obs) -> (N, ctrl)`` in [-1, 1] (see :class:`NominalPolicy`).
  :param filt: a :class:`~robot_safety_sandbox.filters.SafetyFilter`.
  :param metrics: a :class:`~robot_safety_sandbox.eval.metrics.Metric`.
  :param norm: the twin's obs normalizer -- applied to the SAFETY obs group
      before anything the twin evaluates sees it.
  :param dstb_fn: ``(s_obs) -> (N, dstb_dim)``, or None for no attack. Requires
      an env built with ``adversary=True``.
  :param dstb_scale: continuous attack strength (see ``EvalEnv.set_dstb_scale``).
  :param no_filter: run the nominal unfiltered -- the CONTROL arm. The filter is
      still evaluated (so its monitor is still measured) but never applied, and
      its telemetry is zeroed so engagement reads 0.
  :param command_surgery: ``(env, engaged, caution) -> None``, applied after the
      filter decides and before the env steps (see ``TwistCommandSurgery``).
  :param video: ``(env, engaged) -> None`` called once per step after the step
      (see :class:`VideoRecorder`).
  """
  if dstb_fn is not None and not env.adversary:
    raise ValueError(
      "a disturbance source was given but the env has no adversary channel; "
      "build it with adversary=True (the task must declare supports_adversary)")
  env.set_dstb_scale(dstb_scale)
  env.reset(seed=seed)
  n, device = env.num_envs, env.device
  prev_done = torch.ones(n, dtype=torch.bool, device=device)  # t=0 is fresh
  # a nominal that IS the twin reads the twin's own group (see TwinNominal)
  nominal_obs = (env.safety_obs if getattr(nominal, "uses_safety_obs", False)
                 else env.nominal_obs)

  for t in range(steps):
    t0 = time.perf_counter()
    a_nom = nominal(nominal_obs())
    s_obs = norm(env.safety_obs())
    speed = torch.norm(env.robot.data.root_link_lin_vel_w[:, :2], dim=1)
    action, info = filt(a_nom, speed=speed, fresh=prev_done, s_obs=s_obs)
    engaged = info.engaged
    # QCBFIntervention modifies rather than switches, so it reports no caution
    # band; a caution-driven command surgery then simply never fires.
    caution = (info.caution if info.caution is not None
               else torch.zeros_like(engaged))
    if no_filter:
      if hasattr(filt.intervention, "engaged"):
        filt.intervention.engaged.zero_()
      filt.telemetry.reset()
      action = a_nom
      engaged = torch.zeros_like(engaged)
      caution = torch.zeros_like(caution)
    a_dstb = dstb_fn(s_obs) if dstb_fn is not None else None
    if command_surgery is not None:
      command_surgery(env, engaged, caution)
    t1 = time.perf_counter()
    out = env.step(action, a_dstb)
    t2 = time.perf_counter()

    metrics.update(StepRecord(
      t=t, env=env, a_nom=a_nom, a_filt=action, a_dstb=a_dstb,
      info=info, out=out, filter_s=t1 - t0, env_s=t2 - t1))

    if video is not None:
      video(env, engaged)
    done = out.done
    prev_done = done.clone()
    if bool(done.any()):
      filt.reset(done)
    if progress_every and (t + 1) % progress_every == 0:
      print(f"  [{t + 1}/{steps}] engaged={float(engaged.float().mean()):.2f} "
            f"g_min={float(out.g.min()):+.3f}", flush=True)

  return metrics.result()
