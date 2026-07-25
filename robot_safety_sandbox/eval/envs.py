"""The evaluation environment: one registry-driven builder, flat or not.

An evaluation env is a :class:`~robot_safety_sandbox.base.MjlabTensorSafetyEnv`
and never a raw ``ManagerBasedRlEnv``, because the bridge is what supplies the
three things every evaluation needs and a raw env does not:

  * the task's ``(g, l)`` margins under the SAME contract training used
    (terminal anchor, NaN sanitation) -- so "safe" and "reached" mean in eval
    exactly what they meant in training;
  * the ADVERSARY channel: ``step_tensor(cat([a_ctrl, a_dstb]))`` actually
    delivers a disturbance, where a raw env has nowhere to put one. The old
    gap gauntlet built a raw env, so its disturbance actor could only ever
    PREDICT an attack inside a shadow rollout, never deliver one;
  * obs-group auto-detection, so no call site hardcodes "proprioception".

Nothing here knows about any particular task's geometry. Task-shaped cfg
surgery (a pinned terrain parameter, a task-specific spawn) arrives as an
:class:`~robot_safety_sandbox.eval.presets.EvalPreset`, whose transform is
applied to the cfg the task's own builder returns; ``env_overrides`` forwards
experiment-level params to that builder, the same dict ``make_tensor`` takes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from ..base import MjlabTensorSafetyEnv
from ..registry import CUMULATIVE, spec

#: Obs-group name preference. The SAFETY twin's group is called
#: "proprioception" in the parkour-derived cfgs and "actor" in velocity-style
#: ones; the NOMINAL policy's is "actor" wherever both exist side by side (the
#: gap gauntlet grafts the blind walker's group in under that name). Detection
#: order therefore differs by role, and both are overridable.
SAFETY_OBS_ORDER = ("proprioception", "actor", "policy")
NOMINAL_OBS_ORDER = ("actor", "policy", "proprioception")


def detect_obs_key(obs_dict: dict, order: tuple[str, ...]) -> str:
  for cand in order:
    if cand in obs_dict:
      return cand
  return next(iter(obs_dict.keys()))


@dataclass
class StepOut:
  """One environment transition, as the runner and the metrics see it."""
  g: torch.Tensor            # (N,) safety margin, safe iff >= 0
  l: torch.Tensor            # (N,) target margin, in-target iff >= 0
  terminated: torch.Tensor   # (N,) bool
  truncated: torch.Tensor    # (N,) bool

  @property
  def done(self) -> torch.Tensor:
    return self.terminated | self.truncated


class EvalEnv:
  """A task instantiated for evaluation: margins, both obs groups, adversary.

  Thin on purpose -- it owns the bridge and the adversary bookkeeping and
  forwards everything else. ``self.mj`` is the raw mjlab env for the things
  only it has (the scene, the command manager); ``self.cfg_builder`` is the
  EXACT builder this env was made from, so a shadow sim certifies the same
  world (the gauntlet's cfg surgery must not silently differ between them).
  """

  def __init__(self, bridge: MjlabTensorSafetyEnv, *, task: str,
               cfg_builder: Callable, nominal_obs_key: Optional[str] = None):
    self.bridge = bridge
    self.task = task
    self.cfg_builder = cfg_builder
    self.device = str(bridge.mj.device)
    self.num_envs = int(bridge.num_envs)
    self.adversary = bool(bridge.adversary)
    self.ctrl_dim, self.dstb_dim = bridge.ctrl_dim, bridge.dstb_dim
    self.safety_obs_key = bridge.obs_key
    self._nominal_obs_key = nominal_obs_key
    self._dstb_scale = 1.0

  # --- handles ---------------------------------------------------------------

  @property
  def mj(self):
    return self.bridge.mj

  @property
  def robot(self):
    return self.bridge.mj.scene["robot"]

  @property
  def obs(self) -> dict:
    return self.bridge.obs_groups()

  @property
  def nominal_obs_key(self) -> str:
    if self._nominal_obs_key is None:
      self._nominal_obs_key = detect_obs_key(self.obs, NOMINAL_OBS_ORDER)
    return self._nominal_obs_key

  def nominal_obs(self) -> torch.Tensor:
    return self.obs[self.nominal_obs_key].float()

  def safety_obs(self) -> torch.Tensor:
    return self.obs[self.safety_obs_key].float()

  # --- adversary strength ----------------------------------------------------

  def set_dstb_scale(self, scale: float) -> None:
    """Attack STRENGTH as a continuous multiplier of the task's nominal one.

    The two disturbance channels take it in different places, because they
    parameterize strength differently (see ``base._MjlabCore._apply_dstb``):

      wrench  the action only picks a DIRECTION -- it is unit-normalized before
              being scaled by ``force_max`` -- so scaling the action is a no-op
              and the knob is the per-env ``force_scale`` the bridge already
              reads for the survival curriculum. Delivered force = force_max *
              scale, in the attack direction the adversary chose.
      action  the disturbance is added to the control (ctrl += dstb_gain *
              a_dstb), so the knob is the action magnitude itself.

    scale=1.0 reproduces the task's own training-time attack; 0.0 disables it
    while keeping the two-player action space intact.
    """
    self._dstb_scale = float(scale)
    if self.bridge.dstb_mode == "wrench":
      self.bridge.force_scale = torch.full((self.num_envs,), float(scale),
                                           device=self.device)

  def scale_dstb_action(self, a_dstb: torch.Tensor) -> torch.Tensor:
    """Apply the strength knob on the channel where it belongs (see above)."""
    if self.bridge.dstb_mode == "wrench":
      return a_dstb
    return a_dstb * self._dstb_scale

  # --- stepping --------------------------------------------------------------

  def reset(self, seed: Optional[int] = None) -> None:
    if seed is not None:
      self.mj.seed(seed)
    self.bridge.reset()

  def step(self, a_ctrl: torch.Tensor,
           a_dstb: Optional[torch.Tensor] = None) -> StepOut:
    """One control step. Delivers ``a_dstb`` iff the env has an adversary.

    ``a_ctrl`` is in the policy's own [-1, 1] units; the bridge applies the
    task's ``ctrl_gain``, exactly as in training.
    """
    if self.adversary:
      if a_dstb is None:
        a_dstb = torch.zeros(self.num_envs, self.dstb_dim, device=self.device)
      action = torch.cat([a_ctrl, self.scale_dstb_action(a_dstb)], dim=-1)
    else:
      if a_dstb is not None:
        raise ValueError(
          f"a disturbance was supplied but the env for '{self.task}' was built "
          "without an adversary; build it with adversary=True (the task must "
          "declare supports_adversary)")
      action = a_ctrl
    _obs, g, _dones, _timeouts, l = self.bridge.step_tensor(action)
    # terminated/truncated come from the STEP's return, not from the manager:
    # mjlab auto-resets inside step and clears the manager's buffers.
    return StepOut(g=g, l=l,
                   terminated=self.bridge._last_terminated,
                   truncated=self.bridge._last_truncated)

  def close(self) -> None:
    self.bridge.close()


def build_eval_env(task: str, num_envs: int, device: str = "cuda:0", *,
                   adversary: bool = False,
                   env_overrides: Optional[dict] = None,
                   cfg_transform: Optional[Callable] = None,
                   end_criterion: Optional[str] = None,
                   safety_obs_key: Optional[str] = None,
                   nominal_obs_key: Optional[str] = None,
                   play: bool = True,
                   **bridge_kwargs) -> EvalEnv:
  """Build the evaluation env for ``task``.

  :param adversary: give the env the concatenated ``[ctrl, dstb]`` action space
      so a disturbance can actually be DELIVERED (refused unless the task
      declares ``supports_adversary``).
  :param env_overrides: params forwarded to the task's own cfg_builder (the
      same dict ``make_tensor(cfg_overrides=...)`` takes, e.g.
      ``{"cmd_vx": 0.6}``).
  :param cfg_transform: ``cfg -> cfg`` applied AFTER the task builds it -- the
      hook an :class:`EvalPreset` uses for task-shaped surgery. Nothing in this
      module supplies one.
  :param play: build the cfg in play mode (evaluation default: no training-only
      randomization). Consecutive evaluation runs are still not BITWISE
      reproducible, but observation noise is not why: the go2 cfgs set
      ``enable_corruption=False`` on the policy group in play mode (measured: a
      shadow seeded from a live eval env reads max|diff| 0.0 on both obs
      groups). What remains is mujoco_warp's own reduction nondeterminism,
      ~2.5e-6 in qpos after one step.
  """
  s = spec(task)
  if adversary and not s.supports_adversary:
    raise ValueError(
      f"task '{task}' does not define an adversary (supports_adversary=False), "
      "so there is no disturbance channel to attack through. Pick the task's "
      "two-player variant, or drop the adversary.")
  if s.mode == CUMULATIVE:
    raise ValueError(
      f"task '{task}' is mode='cumulative': its 'g' is the env's shaped reward, "
      "not a safety margin, so nothing in an evaluation could be judged safe. "
      "Evaluate on the SAFETY task whose margins define the failure set and "
      "load the cumulative policy as the NOMINAL.")

  overrides = dict(env_overrides or {})
  _play = bool(play)

  def cfg_builder(play=None, **kw):
    # The evaluation fixes `play`, not build_task_cfg (which always asks for
    # play=False, the TRAINING setting) — so the incoming value is dropped on
    # purpose. Caller kwargs win over the baked-in overrides, which lets a
    # shadow sim be built from this very closure.
    del play
    cfg = s.cfg_builder(play=_play, **{**overrides, **kw})
    return cfg_transform(cfg) if cfg_transform is not None else cfg

  kw = dict(s.kwargs)
  kw.update(bridge_kwargs)
  if safety_obs_key is not None:
    kw["obs_key"] = safety_obs_key
  ec = end_criterion if end_criterion is not None else s.end_criterion
  bridge = MjlabTensorSafetyEnv(
    num_envs, device, cfg_builder=cfg_builder, margin_fn=s.margin_fn,
    ctrl_dim=s.ctrl_dim, dstb_dim=s.dstb_dim, adversary=adversary,
    end_criterion=ec, **kw)
  return EvalEnv(bridge, task=task, cfg_builder=cfg_builder,
                 nominal_obs_key=nominal_obs_key)


# --- the nominal's velocity command ------------------------------------------

@dataclass
class TwistCommandSurgery:
  """Drive the nominal's velocity command from the filter's own verdict.

  Part of the deployed least-restrictive protocol, not a metric: while the
  filter is in its CAUTION band the nominal's forward command is zeroed so it
  decelerates itself, and while the fallback is ENGAGED the command is held at
  the value the twin was TRAINED under (feeding a twin cmd=0 when it learned
  under a constant forward command is out of distribution and turns a traverse
  fallback into a braker -- measured, 2026-07-11).

  Generic: it applies to any task whose command manager has ``term``. Tasks
  without it get ``available=False`` and the runner skips it.
  """
  cmd_vx: float = 1.0
  engaged_cmd_vx: float = 1.0
  term: str = "twist"
  index: int = 0
  available: bool = field(init=False, default=False)

  def bind(self, env: EvalEnv) -> "TwistCommandSurgery":
    self.available = self.term in getattr(
      env.mj.command_manager, "active_terms", ())
    return self

  def __call__(self, env: EvalEnv, engaged: torch.Tensor,
               caution: torch.Tensor) -> None:
    if not self.available:
      return
    cmd = env.mj.command_manager.get_command(self.term)
    col = cmd[:, self.index]
    cmd[:, self.index] = torch.where(
      engaged, torch.full_like(col, self.engaged_cmd_vx),
      torch.where(caution, torch.zeros_like(col),
                  torch.full_like(col, self.cmd_vx)))
