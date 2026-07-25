"""Shadow simulation: the imagined future a rollout monitor certifies against.

A rollout (Gameplay) monitor does not trust a learned scalar; it SIMULATES.
That needs a second, resettable copy of the world -- a *shadow sim* -- which per
control step is re-seeded from the live state, driven for H steps, and read for
its margins. This module supplies the abstraction (:class:`ShadowSim`) and the
mjlab implementation (:class:`MjlabShadowSim`); the monitors themselves live in
``monitor.py`` and never touch mjlab.

Why an ABC rather than "just use the env": the monitor's semantics (min over the
horizon, min over parallel rollouts, recertification latching) are pure
bookkeeping and are unit-testable against a toy sim in milliseconds. Only
:class:`MjlabShadowSim` needs a GPU.

State seeding -- the part an earlier docstring wrongly called impossible
------------------------------------------------------------------------
mjlab's batched writers (``write_root_link_pose_to_sim`` and friends) are an
in-use pattern for exactly this (see
``envs/go2_gap/brake_or_jump.py::_restore``). What they do NOT cover is
everything *around* the rigid-body state: contact-force history, air-time
tracking, observation history, the action manager's last/prev action, command
buffers. Those are all plain torch tensors hanging off the managers -- nested up
to four containers deep (manager -> {group: {term: CircularBuffer}} -> tensor)
but nothing more exotic than that -- so :func:`sync_env_state` copies them too,
row by row. **The claim that observation-history buffers cannot be
round-tripped is retracted**: they can, and they are, here.

Three details that cost real accuracy, all learned the hard way:

* Copy ``sim.data.qpos``/``qvel`` WHOLESALE, not ``entity.data.joint_pos``.
  The latter covers only NAMED joints -- car_goal's caster ball joint is not
  one -- and omitting it desynced the shadow by 1e-4 m within one step and
  0.15 in g over thirty.
* ``qacc_warmstart`` is state. Without it the solver starts from a different
  guess and the trajectories part company immediately.
* These fields are mjlab ``TorchArray`` proxies, NOT ``torch.Tensor``
  (``torch.is_tensor`` is False on them). Write with ``dst[:] = src``, which
  routes through warp's own CUDA stream; ``.copy_()`` delegates to the wrapped
  tensor on the default stream.

Measured fidelity (RTX 4070). On go2_stabilize -- the env with a 4-substep
contact-force history AND seven observation-history buffers, i.e. the one this
was said to be impossible for -- a seeded shadow matches the live env's
``qpos``/``qvel`` to 1.2e-7, every observation-history buffer bitwise, and the
task's ``g`` margin EXACTLY (dg = 0). On car_goal the shadow reproduces the live
trajectory to 2.4e-7 m on the first step and then diverges only at the system's
own Lyapunov rate (~1e-3 m by step 10 under random actions).

Two things do NOT come across, both by nature rather than by omission:

* observation NOISE. go2_stabilize adds +-1.5 to ``joint_vel``, so the shadow's
  observations differ from the live ones by up to a full noise width. That is a
  fresh draw, not a state gap -- the MARGIN, a function of state, is identical.
* bitwise step determinism. Two shadows seeded identically from the same live
  env and given the same action land 4.5e-4 apart in ``g`` after one step; that
  is mujoco_warp's own reduction nondeterminism and bounds how sharp any
  rollout verdict can be.

The contact-history wrinkle
---------------------------
``ContactSensorCfg.history_length=4`` (envs/velocity/go2.py) feeds
``margins.g_terrain_relative``, which PREFERS ``force_history`` over the
instantaneous ``force``. A shadow env seeded without that buffer would evaluate
its margin off a half-filled window. Three ways out, chosen by
``contact_history=``:

  "sync"           copy the live buffer over (default). Exact -- and the reason
                   this is not, in fact, a limitation.
  "instantaneous"  zero it and set ``env._zoo_instantaneous_contact``, which
                   makes ``margins.g_terrain_relative`` read
                   ``sensor.data.force`` instead. The mitigation for a shadow
                   whose sensor set does not line up with the live one.
  "ignore"         leave it. Sound only when ``decimation >= history_length``
                   (one control step then fully refills the window, and the
                   monitor never reads a margin before its first step) -- which
                   is ASSERTED, not assumed.

There is deliberately no fourth option that lets a half-filled history through
silently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import torch


class RolloutStep(NamedTuple):
  """One step of an imagined rollout, batched over the shadow sim's envs.

  ``ctx`` is the per-step keyword context the fallback / adversary policies are
  called with (the same ``**ctx`` convention the live filter uses, e.g.
  ``{"s_obs": normalized_obs}``), so a policy cannot tell whether it is driving
  the real env or an imagined one.
  """

  ctx: dict
  g: torch.Tensor      # (n,) safety margin, safe iff >= 0
  l: torch.Tensor      # (n,) target margin, in-target iff >= 0
  done: torch.Tensor   # (n,) bool, the episode ended on this step


class ShadowSim(ABC):
  """A second simulation the monitor plays imagined futures in.

  Contract:

  * ``num_envs`` is ``N * rollouts_per_env`` -- shadow env ``j`` mirrors live
    env ``j // rollouts_per_env``. The monitor relies on that block layout to
    reshape to ``(N, R)`` and take the min over rollouts.
  * :meth:`seed` copies the CURRENT live state into every shadow env. It must be
    IDEMPOTENT: seeding twice from an unchanged live env and replaying the same
    actions must give the same margins. The Def-2 validity property evaluates
    the monitor several times at one state and would otherwise be meaningless.
  * :meth:`step` advances every shadow env by one control step.
  """

  num_envs: int
  device: str
  rollouts_per_env: int = 1

  @abstractmethod
  def seed(self) -> None:
    """Re-initialize every shadow env from the live env's current state."""

  @abstractmethod
  def step(self, action: torch.Tensor) -> RolloutStep:
    """Advance the shadow sim one control step with ``action`` (n, A)."""


# --- generic per-env state copying -------------------------------------------
# Everything below duck-types on tensor SHAPES rather than on mjlab classes, so
# a manager or sensor that grows a new per-env buffer is round-tripped without a
# code change here. The rule: a tensor whose LEADING dim is the live env count
# is a per-env buffer and is reindexed by the env map; a tensor whose SECOND dim
# is (mjlab's time-first CircularBuffer storage) is reindexed on dim 1.


def _reindex(src: torch.Tensor, dst: torch.Tensor | None, m: torch.Tensor,
             n_live: int, n_shadow: int) -> torch.Tensor | None:
  """Fill ``dst`` from ``src`` along whichever axis is the env axis.

  Returns the destination tensor (allocated if ``dst`` was None, e.g. a
  CircularBuffer that has not been appended to yet), or None if ``src`` is not
  a per-env buffer / the shapes are incompatible.
  """
  if src.shape[0] == n_live and (dst is None or dst.shape[0] == n_shadow):
    if dst is None:
      dst = src.new_zeros((n_shadow, *src.shape[1:]))
    if dst.shape[1:] == src.shape[1:]:
      dst.copy_(src[m])
      return dst
    return None
  if (src.dim() > 1 and src.shape[1] == n_live
      and (dst is None or (dst.dim() > 1 and dst.shape[1] == n_shadow))):
    if dst is None:
      dst = src.new_zeros((src.shape[0], n_shadow, *src.shape[2:]))
    if dst.shape[0] == src.shape[0] and dst.shape[2:] == src.shape[2:]:
      dst.copy_(src[:, m])
      return dst
    return None
  return None


#: attributes that LOOK like per-env buffers but are identity, not state: their
#: correct contents depend on the DESTINATION's env count, so copying them from
#: a differently sized live env corrupts them (this bites only when R > 1).
_NOT_STATE = ("_all_indices", "_ALL_INDICES", "_all_ids", "_env_ids",
              "_all_env_ids", "_max_len_tensor")

#: attributes that are BACK-REFERENCES to the world, not state. A manager term
#: holds the env and the asset it acts on; following those would walk the whole
#: object graph (and re-enter the scene and the simulation) instead of the
#: handful of buffers we are after.
_NOT_TRAVERSED = ("_env", "env", "_asset", "asset", "scene", "sim", "_sim",
                  "cfg", "_cfg", "_data", "_slots")


def _copy_state(src: Any, dst: Any, m: torch.Tensor, n_live: int,
                n_shadow: int, max_depth: int = 0, _depth: int = 0) -> None:
  """Copy every per-env tensor of ``src`` into ``dst``, reindexed by ``m``.

  ``max_depth`` bounds recursion into nested state containers -- the
  observation manager holds a dict of groups holding a dict of terms holding a
  CircularBuffer holding the tensor, which is exactly the "observation history"
  a rollout was once thought unable to round-trip. Use ``max_depth=0`` on
  objects that also reference the whole world (the env) so the walk stops at
  tensors and never wanders into the scene or the simulation.
  """
  if src is dst or src is None or dst is None or not hasattr(src, "__dict__"):
    return
  _copy_items(vars(src).items(),
              lambda k: getattr(dst, k, None),
              lambda k, v: setattr(dst, k, v),
              m, n_live, n_shadow, max_depth, _depth)


def _copy_items(items, get, put, m, n_live, n_shadow, max_depth, _depth) -> None:
  """The shared body of the attribute walk and the dict walk."""
  for name, val in list(items):
    if name in _NOT_STATE:
      continue
    ref = get(name)
    if torch.is_tensor(val):
      out = _reindex(val, ref if torch.is_tensor(ref) else None, m, n_live,
                     n_shadow)
      if out is not None and out is not ref:
        put(name, out)
    elif name == "_pointer" and isinstance(val, int):
      put(name, val)                   # CircularBuffer's global write cursor
    elif _depth >= max_depth or ref is None or name in _NOT_TRAVERSED:
      continue
    elif isinstance(val, dict) and isinstance(ref, dict):
      _copy_items(val.items(), ref.get, ref.__setitem__, m, n_live, n_shadow,
                  max_depth, _depth + 1)
    elif type(val) is type(ref) and hasattr(val, "__dict__"):
      _copy_state(val, ref, m, n_live, n_shadow, max_depth, _depth + 1)


#: raw mjwarp Data fields that ARE the physics state. Everything else there is
#: derived and gets recomputed by the forward() at the end of seeding.
_SIM_FIELDS = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "time",
               "xfrc_applied", "mocap_pos", "mocap_quat")

#: managers whose own buffers and per-term buffers are per-env state
_MANAGERS = ("action_manager", "observation_manager", "command_manager",
             "event_manager", "termination_manager", "curriculum_manager")


def sync_env_state(live, shadow, env_map: torch.Tensor, *,
                   contact_history: str = "sync") -> None:
  """Make ``shadow`` (an mjlab ``ManagerBasedRlEnv``) a copy of ``live``.

  ``env_map`` is (n_shadow,) long: shadow env j takes live env ``env_map[j]``.
  Positions are re-based onto the shadow's own tile origins, so the two envs
  need not have the same env count -- but they DO need the same terrain per
  tile, which is the caller's responsibility (see :class:`MjlabShadowSim`).
  """
  n_live, n_shadow, m = live.num_envs, shadow.num_envs, env_map

  # 1. raw physics state, wholesale (see the module docstring on why not
  #    entity.data.joint_pos). These are mjlab TorchArray proxies, not
  #    torch.Tensor: they are NOT instances of Tensor (torch.is_tensor is False)
  #    and must be written with `dst[:] = ...`, which routes the write through
  #    warp's own CUDA stream. `.copy_()` would delegate to the wrapped tensor
  #    on the default stream instead.
  for f in _SIM_FIELDS:
    a, b = getattr(live.sim.data, f, None), getattr(shadow.sim.data, f, None)
    if a is None or b is None or not hasattr(a, "shape") or not a.numel():
      continue
    if a.shape[0] == n_live and b.shape[0] == n_shadow and a.shape[1:] == b.shape[1:]:
      b[:] = a[m]

  # 2. re-base each entity's root onto the shadow's tile origin (step 1 carried
  #    the LIVE world position across in qpos), and push the named joint state
  #    through the entity API so its own cache agrees with the raw copy (a
  #    disagreement would be written back out by write_data_to_sim below).
  d_origin = shadow.scene.env_origins - live.scene.env_origins[m]
  same_tiles = bool((d_origin == 0).all())
  for name, ent in shadow.scene.entities.items():
    src = live.scene.entities.get(name)
    if src is None:
      continue
    if src.data.joint_pos is not None and src.data.joint_pos.shape[1]:
      ent.write_joint_state_to_sim(src.data.joint_pos[m], src.data.joint_vel[m])
    # a fixed-base entity (the terrain is one) has no root to re-base; its world
    # pose is baked into the model, per tile, and needs no correction.
    if getattr(ent, "is_fixed_base", True):
      continue
    pos = src.data.root_link_pos_w[m]
    if not same_tiles:
      pos = pos + d_origin
    ent.write_root_link_pose_to_sim(
      torch.cat([pos, src.data.root_link_quat_w[m]], dim=-1))
    ent.write_root_link_velocity_to_sim(
      torch.cat([src.data.root_link_lin_vel_w[m],
                 src.data.root_link_ang_vel_w[m]], dim=-1))

  # 3. env + manager bookkeeping. The env pass is tensor-only (max_depth=0), so
  #    it picks up episode_length_buf and any task-local per-env buffer (e.g.
  #    brake_or_jump's `_peek`) without walking into scene/sim. Manager TERMS go
  #    one level deeper: that is where the observation history CircularBuffers,
  #    the delay buffers and the command buffers live.
  _copy_state(live, shadow, m, n_live, n_shadow, max_depth=0)
  shadow.common_step_counter = live.common_step_counter
  for mgr in _MANAGERS:
    a, b = getattr(live, mgr, None), getattr(shadow, mgr, None)
    if a is None or b is None:
      continue
    # depth 4 reaches the deepest per-env buffer mjlab nests: manager ->
    # {group: {term: CircularBuffer}} -> tensor (the observation history), and
    # manager -> {term: DelayBuffer} -> CircularBuffer -> tensor.
    _copy_state(a, b, m, n_live, n_shadow, max_depth=4)

  # 4. sensors: contact-force history + air time (see the module docstring).
  _sync_sensors(live, shadow, m, n_live, n_shadow, contact_history)

  shadow.scene.write_data_to_sim()
  shadow.sim.forward()


def _sync_sensors(live, shadow, m, n_live: int, n_shadow: int,
                  policy: str) -> None:
  if policy not in ("sync", "instantaneous", "ignore"):
    raise ValueError(f"contact_history={policy!r}; one of "
                     "('sync', 'instantaneous', 'ignore')")
  missing = []
  for name, sensor in shadow.scene.sensors.items():
    hist = getattr(sensor, "_history_state", None)
    if hist is None and getattr(sensor, "_air_time_state", None) is None:
      continue                                   # nothing stateful to carry
    src = live.scene.sensors.get(name)
    if policy == "sync":
      if src is None:
        missing.append(name)
      else:
        _copy_state(src, sensor, m, n_live, n_shadow, max_depth=1)
    elif policy == "instantaneous" and hist is not None:
      for buf in hist.values():
        buf.zero_()
  if missing:
    raise ValueError(
      f"contact_history='sync' but the live env has no sensor(s) {missing} to "
      "copy history from, so the rollout would evaluate its margin off a "
      "half-filled window. Build the shadow from the SAME task, or pass "
      "contact_history='instantaneous' (margins then read sensor.data.force) "
      "or 'ignore' (sound only when decimation >= history_length).")
  if policy == "instantaneous":
    # honored by margins.g_terrain_relative and go2_crawl.tunnel.g_crawl
    shadow._zoo_instantaneous_contact = True
  elif policy == "ignore":
    dec = int(shadow.cfg.decimation)
    for name, sensor in shadow.scene.sensors.items():
      h = int(getattr(getattr(sensor, "cfg", None), "history_length", 0) or 0)
      if h > dec:
        raise ValueError(
          f"contact_history='ignore' but sensor {name!r} keeps {h} substeps of "
          f"history while the env decimates by {dec}: one rollout step would "
          "NOT refill the window, so the first margins would be stale. Use "
          "'sync' or 'instantaneous'.")


def _mj(env):
  """The raw mjlab env behind a zoo bridge (or the env itself)."""
  return getattr(env, "mj", env)


class MjlabShadowSim(ShadowSim):
  """An mjlab env stepped as the shadow of another mjlab env.

  :param live_env: the env being filtered -- a ``ManagerBasedRlEnv`` or any of
      the zoo bridges (anything exposing ``.mj``).
  :param shadow_env: a zoo tensor bridge (``MjlabTensorSafetyEnv``) built from
      the SAME task with ``N * rollouts_per_env`` envs. A bridge rather than a
      raw mjlab env because its ``step_tensor`` already returns the task's
      (g, l) with the terminal failure anchor applied -- a rollout must not
      reimplement the margin contract it is certifying against.
  :param num_envs: the LIVE env count N (defaults to the live env's own).
  :param rollouts_per_env: R. ``shadow_env`` must have exactly ``N * R`` envs;
      shadow ``j`` mirrors live ``j // R``.
  :param obs_adapter: callable(raw_obs (n, O)) -> ctx dict for the fallback and
      adversary. Defaults to ``{"s_obs": obs}``; pass the twin's normalizer
      here, exactly as the live call site does.
  :param contact_history: see the module docstring.

  **Terrain caveat.** Shadow tile j must carry the same terrain as live tile
  ``env_map[j]``. With ``R == 1``, the same task and the same env count, mjlab
  lays out an identical grid and this holds by construction (verified: the env
  origins come out bitwise equal, which the seeding also uses as a fast path).
  With ``R > 1``, or a terrain curriculum that assigns different sub-terrains
  per tile, it does NOT hold in general, so such a task must either use R=1 or a
  generator with a single repeated sub-terrain. Checkable only in part; stated
  here because the rest is the caller's contract.
  """

  def __init__(self, live_env, shadow_env, *, num_envs: int | None = None,
               rollouts_per_env: int = 1, obs_adapter=None,
               contact_history: str = "sync"):
    if not hasattr(shadow_env, "step_tensor"):
      raise TypeError(
        "shadow_env must be a zoo tensor bridge (MjlabTensorSafetyEnv) so the "
        "rollout reads the task's (g, l) through the same contract training "
        f"does; got {type(shadow_env).__name__}.")
    if getattr(shadow_env, "dense_reward", False):
      raise ValueError(
        "the shadow bridge is in DENSE-REWARD mode (a mode='cumulative' task), "
        "so its 'g' is the env's shaped reward, not a safety margin — there is "
        "nothing for a rollout to certify. Build the shadow from the SAFETY "
        "task whose margins define the failure set.")
    self.live_env, self.shadow_env = live_env, shadow_env
    self._live, self._shadow = _mj(live_env), _mj(shadow_env)
    self.num_envs = int(self._shadow.num_envs)
    self.device = str(self._shadow.device)
    self.n_live = int(num_envs if num_envs is not None else self._live.num_envs)
    self.rollouts_per_env = int(rollouts_per_env)
    if self.num_envs != self.n_live * self.rollouts_per_env:
      raise ValueError(
        f"shadow env has {self.num_envs} envs; expected num_envs * "
        f"rollouts_per_env = {self.n_live} * {self.rollouts_per_env}")
    self.env_map = torch.arange(
      self.n_live, device=self.device).repeat_interleave(self.rollouts_per_env)
    self.obs_adapter = obs_adapter or (lambda obs: {"s_obs": obs})
    self.contact_history = contact_history
    self.seeds = 0                     # rollouts run; telemetry + tests

  def seed(self) -> None:
    sync_env_state(self._live, self._shadow, self.env_map,
                   contact_history=self.contact_history)
    self.seeds += 1

  def step(self, action: torch.Tensor) -> RolloutStep:
    obs, g, dones, _timeouts, l = self.shadow_env.step_tensor(action)
    return RolloutStep(ctx=self.obs_adapter(obs), g=g, l=l, done=dones)

  @classmethod
  def from_task(cls, task_id: str, live_env, num_envs: int, *,
                rollouts_per_env: int = 1, device: str = "cuda:0",
                adversary: bool = False, obs_adapter=None,
                contact_history: str = "sync",
                **make_kwargs) -> "MjlabShadowSim":
    """Build the shadow bridge for ``task_id`` and wrap it.

    ``adversary=True`` gives the bridge the concatenated ``[ctrl, dstb]`` action
    space that an :class:`~.monitor.AdversarialRolloutMonitor` drives.
    """
    from ..registry import make_tensor
    shadow = make_tensor(task_id, num_envs=num_envs * rollouts_per_env,
                         device=device, adversary=adversary, **make_kwargs)
    return cls(live_env, shadow, num_envs=num_envs,
               rollouts_per_env=rollouts_per_env, obs_adapter=obs_adapter,
               contact_history=contact_history)
