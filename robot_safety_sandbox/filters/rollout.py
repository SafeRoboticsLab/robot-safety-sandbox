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

Identity seeding -- copying the robot, not just its state
---------------------------------------------------------
State is only half of "be the live env". ``mode="startup"`` events give every
env its OWN ROBOT once at construction -- on the velocity/parkour/digit stacks:
``geom_friction`` (foot friction, 0.3-1.2), ``body_ipos`` (base COM, +-2.5/3 cm)
and ``entity.data.encoder_bias`` (+-0.015 rad). Those live in ``sim.model`` and
``entity.data``, which the buffer walk deliberately cannot reach, and a shadow
built from the same cfg REDRAWS them. Unsynced, the rollout then certifies a
robot from the same distribution rather than THE robot being filtered, which
voids the H = T construction guarantee however exact the state copy is.
:func:`sync_dr_identity` copies the REALIZED per-env values (live j -> every one
of live j's R shadow lanes, identically -- R independent draws would be a
robust certificate over the DR distribution, a different guarantee), and
:func:`assert_dr_identity` re-checks them after every seed, because an identity
gap corrupts verdicts invisibly. Two notes for anyone extending this:

* the field list is DERIVED, from ``EventManager.domain_randomization_fields``
  (what ``@requires_model_fields`` had ``sim.expand_model_fields()`` allocate
  per-world memory for). It includes the constants recomputed from a DR'd field
  -- ``body_subtreemass``, ``dof_invweight0``, ``body_invweight0``, the tendon
  pair -- so copying them all is exact and no ``recompute_constants`` is needed.
  A startup event whose function declares no ``model_fields`` and is not in
  :data:`_DR_ENTITY_FIELDS` RAISES rather than being skipped.
* a model field that was never expanded per-world is a BROADCAST view (nworld
  rows sharing one row of memory; ``TorchArray`` does the ``expand()``), and
  writing to it raises "more than one element of the written-to tensor refers
  to a single memory location". That is not an obstacle to route around: it
  means the field holds no per-env value at all, so there is nothing to carry
  and the two envs agree by construction. Only an ASYMMETRY -- live expanded,
  shadow not -- is an error, and it is raised as one.

Measured fidelity (RTX 4070, go2_locomote, 64 envs, identical action sequences).
After a seed, all six DR fields match EXACTLY (0.0) and the first step lands at
1.2e-7 ``qpos`` / 2.8e-6 ``qvel`` with ``dg`` EXACTLY 0. WITHOUT the identity
copy the same seed leaves ``geom_friction`` 7.7e-1, ``body_ipos`` 5.9e-2 and
``encoder_bias`` 2.9e-2 apart, and one step costs 9.0e-3 qpos / 5.8e-1 qvel and
1.6e-3 in ``g``. On go2_stabilize the seeded shadow also reproduces every
observation-history buffer bitwise and the ``g`` margin exactly; on car_goal it
tracks the live trajectory to 2.4e-7 m on the first step, then diverges at the
system's own Lyapunov rate (~1e-3 m by step 10 under random actions).

The remaining floor is mujoco_warp's reduction nondeterminism, and it is small:
two shadows seeded identically from ONE live env and given the same action land
2.5e-6 (qpos) / 4.3e-5 (qvel) / 2.1e-7 (g) apart after a step -- i.e. FURTHER
apart than the seeded shadow is from the live env itself. (The earlier "4.5e-4
in g from warp nondeterminism" figure is withdrawn: it compared two envs with
independent DR draws, so it measured this bug, not the solver.)

Observation noise is likewise not a caveat on the evaluation path: the zoo's
eval envs are built ``play=True``, which sets ``enable_corruption=False`` on the
``actor`` group (the nominal AND safety group here), and a seeded shadow's
``actor``/``critic`` observations come out at max|diff| 0.0 against the live
env's -- exactly, encoder bias included. A shadow built for TRAINING-mode cfgs
(``play=False``) does draw its own noise; the MARGIN, a function of state, is
unaffected either way.

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

# --- per-env IDENTITY: the domain-randomization draw --------------------------
# Everything above copies the robot's STATE. A ``mode="startup"`` event gives
# each env its own ROBOT -- foot friction, base COM offset, encoder bias -- drawn
# once at construction and living in `sim.model` / `entity.data`, neither of
# which the buffer walk above can reach (`scene` and `sim` are _NOT_TRAVERSED,
# and rightly so). A shadow built from the same cfg redraws its own, so without
# the copy below the rollout certifies a DIFFERENT ROBOT from the same
# distribution -- which voids the H = T guarantee no matter how good the state
# copy is. The values are copied REALIZED and per-env (live j -> every lane of
# shadow j); nothing is ever re-sampled here.

#: DR whose draw does NOT land in ``sim.model`` (so ``model_fields`` cannot find
#: it), by event-function name -> the ``entity.data`` buffer it writes.
_DR_ENTITY_FIELDS = {"encoder_bias": "encoder_bias"}


def _native(x):
  """The plain ``torch.Tensor`` behind an mjlab ``TorchArray`` proxy."""
  return getattr(x, "_tensor", x)


def dr_identity_fields(live) -> tuple[tuple[str, ...], tuple[str, ...]]:
  """The (model, entity-data) buffers that hold ``live``'s per-env DR draw.

  Derived from the env's own event manager rather than hard-coded, so a task
  that randomizes a new field is covered without a change here:
  ``EventManager.domain_randomization_fields`` is exactly the set
  ``@requires_model_fields`` asked ``sim.expand_model_fields()`` to give real
  per-world memory to (the DR'd fields AND the constants recomputed from them,
  which is why seeding needs no ``recompute_constants``).

  A DR function that declares no ``model_fields`` writes its draw somewhere this
  cannot see; those are enumerated in :data:`_DR_ENTITY_FIELDS`, and an unknown
  one RAISES rather than being silently skipped.
  """
  em = getattr(live, "event_manager", None)
  model_fields = tuple(getattr(em, "domain_randomization_fields", ()) or ())
  entity_fields: list[str] = []
  events = getattr(getattr(live, "cfg", None), "events", None) or {}
  for name, term in events.items():
    fn = getattr(term, "func", None)
    fname = getattr(fn, "__name__", type(fn).__name__)
    if getattr(fn, "model_fields", None):
      continue                                 # lands in sim.model; covered
    if fname in _DR_ENTITY_FIELDS:
      entity_fields.append(_DR_ENTITY_FIELDS[fname])
    elif getattr(fn, "_zoo_shadow_identity_safe", False):
      continue                                 # declared per-env-identity-free
    elif getattr(term, "mode", None) == "startup":
      raise ValueError(
        f"startup event {name!r} (func {fname!r}) declares no model_fields, so "
        "a shadow sim cannot tell whether it gave each env a different ROBOT. "
        "If it is domain randomization, add its buffer to "
        "filters.rollout._DR_ENTITY_FIELDS so seeding copies the realized "
        "draw; if it randomizes nothing per-env, mark it explicitly with "
        f"`{fname}._zoo_shadow_identity_safe = True`. Refusing to certify a "
        "rollout whose robot may not be the live one.")
  return model_fields, tuple(dict.fromkeys(entity_fields))


def _dr_pairs(live, shadow, m: torch.Tensor, n_live: int, n_shadow: int):
  """Yield ``(name, live_buffer, shadow_buffer)`` for every DR field to carry."""
  model_fields, entity_fields = dr_identity_fields(live)
  for f in model_fields:
    a = getattr(live.sim.model, f, None)
    b = getattr(shadow.sim.model, f, None)
    if a is None or b is None:
      continue
    ta, tb = _native(a), _native(b)
    if not ta.numel() or ta.shape[0] != n_live or tb.shape[0] != n_shadow:
      continue
    if ta.shape[1:] != tb.shape[1:]:
      continue
    if tb.stride(0) == 0:
      # A model field that was never expanded per-world is a BROADCAST view --
      # `nworld` rows sharing one row of memory (TorchArray does the expand()).
      # Writing to it raises "more than one element ... refers to a single
      # memory location", and no per-env value can be stored in it anyway.
      if ta.stride(0) == 0:
        continue        # neither side is per-world: identical by construction
      raise RuntimeError(
        f"live env randomizes model field {f!r} per env but the shadow's copy "
        "is a broadcast view (never expanded per-world), so the realized draw "
        "cannot be stored in it. Build the shadow from the SAME cfg (its DR "
        "events are what call sim.expand_model_fields).")
    yield f, a, b
  for name, ent in shadow.scene.entities.items():
    src = live.scene.entities.get(name)
    if src is None:
      continue
    for attr in entity_fields:
      a = getattr(src.data, attr, None)
      b = getattr(ent.data, attr, None)
      if a is None or b is None or not torch.is_tensor(a) or not a.numel():
        continue
      if (a.shape[0] == n_live and b.shape[0] == n_shadow
          and a.shape[1:] == b.shape[1:]):
        yield f"{name}.{attr}", a, b


def sync_dr_identity(live, shadow, m: torch.Tensor, n_live: int,
                     n_shadow: int) -> tuple[str, ...]:
  """Copy live env ``m[j]``'s REALIZED DR values into shadow env ``j``.

  Deterministic and re-indexed by the same ``env_map`` as the rest of the state
  copy, so with ``rollouts_per_env = R > 1`` all R lanes of live env j get live
  j's values IDENTICALLY. Deliberately not R independent draws: that would be a
  robust certificate over the DR distribution, a different guarantee, and it
  must never happen by accident.
  """
  copied = []
  for name, a, b in _dr_pairs(live, shadow, m, n_live, n_shadow):
    b[:] = _native(a)[m]                  # `[:] =`: warp's stream, not copy_()
    copied.append(name)
  return tuple(copied)


def dr_identity_diff(live, shadow, m: torch.Tensor, n_live: int,
                     n_shadow: int) -> dict[str, float]:
  """``{field: max|live[m] - shadow|}`` over the DR fields (0.0 when matched)."""
  return {name: float((_native(b) - _native(a)[m]).abs().max())
          for name, a, b in _dr_pairs(live, shadow, m, n_live, n_shadow)}


def assert_dr_identity(live, shadow, m: torch.Tensor, n_live: int,
                       n_shadow: int) -> None:
  """Fail loudly if the shadow is not the same ROBOT as the live env.

  Cheap (a handful of tiny reductions) and run on every seed on purpose: a
  silently unsynced identity is invisible in the verdicts -- it reads as a noisy
  monitor -- and cost this project a whole invalidated experiment once.
  """
  bad = {k: v for k, v in dr_identity_diff(live, shadow, m, n_live,
                                           n_shadow).items() if v != 0.0}
  if bad:
    raise RuntimeError(
      "shadow sim is not the same robot as the live env after seeding: "
      + ", ".join(f"{k} max|diff|={v:.3e}" for k, v in sorted(bad.items()))
      + ". The rollout would certify a system drawn from the same domain-"
        "randomization distribution rather than THE system being filtered.")

#: managers whose own buffers and per-term buffers are per-env state
_MANAGERS = ("action_manager", "observation_manager", "command_manager",
             "event_manager", "termination_manager", "curriculum_manager")


def sync_env_state(live, shadow, env_map: torch.Tensor, *,
                   contact_history: str = "sync",
                   verify_identity: bool = True) -> None:
  """Make ``shadow`` (an mjlab ``ManagerBasedRlEnv``) a copy of ``live``.

  ``env_map`` is (n_shadow,) long: shadow env j takes live env ``env_map[j]``.
  Positions are re-based onto the shadow's own tile origins, so the two envs
  need not have the same env count -- but they DO need the same terrain per
  tile, which is the caller's responsibility (see :class:`MjlabShadowSim`).

  Copies both halves of "be the live env": its STATE (physics, manager and
  sensor buffers) and its IDENTITY (the per-env domain-randomization draw, see
  :func:`sync_dr_identity`). ``verify_identity`` re-reads the DR fields
  afterwards and raises if any differ -- on by default, because an identity gap
  is invisible in the verdicts it corrupts.
  """
  n_live, n_shadow, m = live.num_envs, shadow.num_envs, env_map

  # 0. per-env IDENTITY (the startup-DR draw). Before the state copy, so the
  #    forward() at the end runs on the live env's own robot.
  sync_dr_identity(live, shadow, m, n_live, n_shadow)

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

  if verify_identity:
    assert_dr_identity(live, shadow, m, n_live, n_shadow)


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
