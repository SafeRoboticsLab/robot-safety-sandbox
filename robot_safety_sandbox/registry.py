"""Task registry: one place future work looks to run or add benchmark tasks.

    from robot_safety_sandbox import make_tensor, list_tasks
    env = make_tensor("go2_gap_chain", num_envs=2048)   # -> TensorVecEnv
    model = ReachAvoidPPO("MlpPolicy", env, normalize_obs=True, ...)

A :class:`TaskSpec` pins everything a benchmark run needs: the mjlab cfg
builder (spawn events + curricula), the reach-avoid margins, action dims, the
recommended learner, and the warm-start lineage (curriculum pipelines like
landing -> crossing -> chain are first-class here — they are how the hard
skills were actually learned).

Every task declares ONE axis, its ``mode`` — the safety_sb3 BACKUP it is trained
under (see :data:`MODES`). It replaces the old ``kind="safety"|"nominal"`` split,
which was redundant once ``backups.CUMULATIVE`` made plain reward-maximizing RL
a mode of the same framework:

  mode="safety"       AVOID:       V = min(g, gamma V')          margins, no l
  mode="reach-avoid"  REACH_AVOID: V = min(g, max(l, gamma V'))  margins with l
  mode="cumulative"   CUMULATIVE:  V = r + gamma (1-d) V'        dense env reward,
                      i.e. the plain-RL TASK policy a filter wraps (margin_fn=None;
                      envs are auto-built in dense mode, trained with STOCK SB3).

A full filter experiment needs both layers: a cumulative task policy (pi_task) and
a safety task supplying the certificate V(s) + fallback.

The two SAFETY modes have FOUR learners, one per (mode, players) cell — the mode
is a property of the TASK's margins, the player count a property of the RUN
(``--adversary``):

                       avoid (no l)      reach-avoid (real l)
        single-player  SafetyPPO         ReachAvoidPPO
        two-player     IsaacsPPO         GameplayPPO

``mode`` picks the COLUMN; :func:`algo_name` picks the row from the run's
``adversary`` flag and returns the cell. ``default_algo`` is DERIVED from the
mode (explicitly overridable, and a legacy override still fixes the mode).
NB ``IsaacsPPO``/``IsaacsSAC`` CHANGED MEANING in safety_sb3 v0.2.0: they are now
the two-player AVOID game (ISAACS eq. 7, no target set); the two-player
reach-avoid learner they used to be is now ``GameplayPPO``/``GameplaySAC``.
"""

from __future__ import annotations

import warnings
from dataclasses import InitVar, dataclass, field
from typing import Callable, Optional

_REGISTRY: dict[str, "TaskSpec"] = {}

#: The task MODE = which safety_sb3 backup values a rollout of it. These values
#: are duplicated as LITERALS from ``safety_sb3.backups`` ON PURPOSE: a
#: CUMULATIVE-ONLY install (dense reward + vanilla SB3, no safety_sb3 — see
#: base.py) must still import this registry, so this module never imports
#: safety_sb3. ``tests/test_registry_mode.py`` asserts the two stay in sync
#: whenever safety_sb3 IS importable.
AVOID = "safety"               # backups.AVOID
REACH_AVOID = "reach-avoid"    # backups.REACH_AVOID
CUMULATIVE = "cumulative"      # backups.CUMULATIVE (plain RL: reward + gamma V')
MODES = (AVOID, REACH_AVOID, CUMULATIVE)
#: the modes that need margins (g, l) and a safety_sb3 learner
SAFETY_MODES = (AVOID, REACH_AVOID)

#: mode -> the single-player learner it defaults to (what ``default_algo``
#: resolves to when a registration doesn't name one).
_MODE_ALGO = {AVOID: "SafetyPPO", REACH_AVOID: "ReachAvoidPPO",
              CUMULATIVE: "PPO"}
#: ...and the inverse: naming a learner fixes the mode (how a legacy
#: ``default_algo=`` registration keeps working without a ``mode=``).
_ALGO_MODE = {
  "SafetyPPO": AVOID,           "IsaacsPPO": AVOID,          # ISAACS eq. 7
  "ReachAvoidPPO": REACH_AVOID, "GameplayPPO": REACH_AVOID,  # Gameplay eq. 6a
  "PPO": CUMULATIVE,                                         # stock SB3
}
#: (mode, n_players) -> learner
_LEARNER = {
  (AVOID, 1): "SafetyPPO",           (AVOID, 2): "IsaacsPPO",
  (REACH_AVOID, 1): "ReachAvoidPPO", (REACH_AVOID, 2): "GameplayPPO",
  (CUMULATIVE, 1): "PPO",            # no two-player cumulative game
}
#: retired ``kind`` axis -> mode (deprecated compat path)
_KIND_MODE = {"safety": AVOID, "nominal": CUMULATIVE}

#: WHEN an episode ends, in terms of the task's (g, l) margins (g>=0 safe,
#: l>=0 in-target). The ENV-side companion to the learner's ``terminal_type``
#: (safety_sb3, how a terminal step is VALUED); the two are orthogonal — every
#: pairing is valid. See :func:`base.build_task_cfg` for the mechanism.
#:   "failure"     terminate on the env's failure set (g<0) + its timeout;
#:                 NEVER on reach -> the agent keeps going after reaching and
#:                 learns to reach DEEPER (default; reproduces today's behavior,
#:                 where every task already terminates on physical failure only).
#:   "reach-avoid" also terminate on success (g>=0 AND l>=0) -> reach-and-stop.
#:   "timeout"     neither failure nor success ends it, only the env timeout
#:                 (diagnostic / pure value-learning).
END_CRITERIA = ("failure", "reach-avoid", "timeout")


@dataclass
class TaskSpec:
  task_id: str
  cfg_builder: Callable          # (play: bool) -> ManagerBasedRlEnvCfg
  margin_fn: Optional[Callable] = None  # (env) -> (g, l); None for cumulative
  description: str = ""
  ctrl_dim: int = 12
  dstb_dim: int = 3              # adversary force dims (ISAACS)
  # Which BACKUP values this task: one of MODES. Fixes the learner COLUMN
  # (algo_name() picks the row from the run's adversary flag), whether the env
  # is built in dense-reward mode, and whether margins are required. Defaults
  # to AVOID unless a legacy default_algo= implies another mode.
  mode: Optional[str] = None
  # DERIVED from mode (_MODE_ALGO); set it only to pin a specific learner name
  # — it must agree with the mode, and naming one is enough to fix the mode.
  default_algo: Optional[str] = None
  warmstart_from: Optional[str] = None  # previous pipeline stage task_id
  supports_adversary: bool = False
  # WHEN the episode ends from (g, l); one of END_CRITERIA. "failure" (default)
  # reproduces today's behavior for EVERY registered task — an audit (2026-07-17)
  # found none currently terminate on success. A run may override it via the
  # trainer's --end-criterion flag; the two knobs (this + terminal_type) are
  # orthogonal. Set "reach-avoid" on a task only if it should end on reach.
  end_criterion: str = "failure"
  kwargs: dict = field(default_factory=dict)  # extra bridge kwargs
  #: DEPRECATED: the retired "safety" | "nominal" axis, mapped onto ``mode``.
  #: InitVar -> constructor-only, never an attribute of the spec.
  kind: InitVar[Optional[str]] = None

  def __post_init__(self, kind):
    if kind is not None:
      if kind not in _KIND_MODE:
        raise ValueError(f"task '{self.task_id}': unknown kind={kind!r}; "
                         f"the axis is now mode= (one of {MODES})")
      warnings.warn(
        f"TaskSpec(kind={kind!r}) is deprecated: the kind axis was replaced by "
        f"mode={_KIND_MODE[kind]!r} (one of {MODES}).",
        DeprecationWarning, stacklevel=3)
      if self.mode is not None and self.mode != _KIND_MODE[kind]:
        raise ValueError(
          f"task '{self.task_id}' sets both mode={self.mode!r} and the "
          f"deprecated kind={kind!r} (-> {_KIND_MODE[kind]!r}); drop the kind.")
      self.mode = _KIND_MODE[kind]
    if self.default_algo is not None and self.default_algo not in _ALGO_MODE:
      raise ValueError(
        f"task '{self.task_id}' declares default_algo={self.default_algo!r}, "
        f"which is not a known learner; known: {sorted(_ALGO_MODE)}")
    if self.mode is None:   # a legacy default_algo= still fixes the mode
      self.mode = (AVOID if self.default_algo is None
                   else _ALGO_MODE[self.default_algo])
    if self.mode not in MODES:
      raise ValueError(f"task '{self.task_id}' has mode={self.mode!r}; "
                       f"must be one of {MODES}")
    if self.default_algo is None:
      self.default_algo = _MODE_ALGO[self.mode]
    elif _ALGO_MODE[self.default_algo] != self.mode:
      raise ValueError(
        f"task '{self.task_id}' declares mode={self.mode!r} but "
        f"default_algo={self.default_algo!r}, which solves "
        f"{_ALGO_MODE[self.default_algo]!r}")
    if self.mode != CUMULATIVE and self.margin_fn is None:
      raise ValueError(
        f"task '{self.task_id}' is mode={self.mode!r} and needs a margin_fn "
        f"(only mode={CUMULATIVE!r} trains without margins, on dense reward)")
    if self.end_criterion not in END_CRITERIA:
      raise ValueError(
        f"task '{self.task_id}' has end_criterion={self.end_criterion!r}; "
        f"must be one of {END_CRITERIA}")


def register(spec: TaskSpec) -> None:
  if spec.task_id in _REGISTRY:
    raise ValueError(f"task '{spec.task_id}' already registered")
  _REGISTRY[spec.task_id] = spec


def list_tasks(mode: Optional[str] = None, kind: Optional[str] = None
               ) -> list[str]:
  """Registered task ids, optionally filtered to one :data:`MODES` entry.

  ``kind=`` (and the retired values "safety"/"nominal" passed positionally) is
  a deprecated alias for ``mode=``."""
  if kind is not None:
    warnings.warn("list_tasks(kind=...) is deprecated; use mode= (one of "
                  f"{MODES}).", DeprecationWarning, stacklevel=2)
    if mode is not None and mode != _KIND_MODE.get(kind, kind):
      raise ValueError(f"list_tasks: mode={mode!r} and kind={kind!r} disagree")
    mode = _KIND_MODE.get(kind, kind)
  if mode == "nominal":   # retired kind value passed positionally
    warnings.warn("list_tasks('nominal') is deprecated; the mode is "
                  f"{CUMULATIVE!r}.", DeprecationWarning, stacklevel=2)
    mode = CUMULATIVE
  if mode is not None and mode not in MODES:
    raise ValueError(f"list_tasks: unknown mode={mode!r}; one of {MODES}")
  return sorted(t for t, s in _REGISTRY.items()
                if mode is None or s.mode == mode)


def spec(task_id: str) -> TaskSpec:
  if task_id not in _REGISTRY:
    raise KeyError(
      f"unknown task '{task_id}'. Registered: {list_tasks()}. "
      "(Some tasks require their source repo on sys.path during the "
      "phase-1 compat period — see tasks/*.py and MIGRATION.md.)")
  return _REGISTRY[task_id]


def algo_name(task_id: str, adversary: bool = False) -> str:
  """The learner CLASS NAME for running ``task_id`` (names only — this module
  never imports safety_sb3, so the registry stays importable without it).

  The task's ``mode`` fixes the BACKUP (avoid vs reach-avoid); ``adversary``
  fixes the PLAYER COUNT. Resolving both together is what keeps the 2x2 honest —
  the old code hardcoded IsaacsPPO for every adversarial run, which since
  safety_sb3 v0.2.0 (where that name means the AVOID game) would silently turn
  every reach-avoid task into an avoid game.

  mode=CUMULATIVE returns ``"PPO"``: the plain-RL task policy trains with STOCK
  stable_baselines3 (identical to SafetyPPO's cumulative mode, but it keeps the
  checkpoint a vanilla SB3 zip). There is no two-player cumulative game.

  Also refuses the one pairing that is silently wrong: a reach-avoid learner on
  an avoid-only task (no target set). Pinning l to a constant does NOT make the
  reach-avoid backup compute the avoid value — a negative l empties the safe
  set, a non-negative one strips the lookahead — so it has no valid formulation
  and must not be reachable by accident. See margins.py.
  """
  s = spec(task_id)
  if adversary and not s.supports_adversary:
    raise ValueError(f"task '{task_id}' does not define an adversary")
  if adversary and s.mode == CUMULATIVE:
    raise ValueError(f"task '{task_id}' is mode={CUMULATIVE!r}: there is no "
                     f"two-player cumulative learner")
  # margin_fns built by margins.compose/avoid_only carry has_target; anything
  # else (task-local margin builders) is assumed to declare a real l.
  if s.mode == REACH_AVOID and not getattr(s.margin_fn, "has_target", True):
    raise ValueError(
      f"task '{task_id}' is AVOID-ONLY (its margin_fn declares no target set) "
      f"but declares mode={REACH_AVOID!r}. Avoid is not a reach-avoid instance "
      f"for ANY constant l — declare mode={AVOID!r} (--adversary then gives the "
      f"two-player IsaacsPPO), or give the task a real reach margin l. See "
      f"margins.py / safety_sb3 RELEASE_NOTES v0.2.0.")
  return _LEARNER[(s.mode, 2 if adversary else 1)]


def make_tensor(task_id: str, num_envs: int = 2048, device: str = "cuda:0",
                adversary: bool = False, end_criterion: Optional[str] = None,
                cfg_overrides: Optional[dict] = None, **kw):
  """GPU-resident env (primary path; pair with safety_sb3 PPO learners).

  ``end_criterion`` (None -> the task's TaskSpec value; else an explicit
  override, one of :data:`END_CRITERIA`) sets WHEN the episode ends from (g, l).
  ``cfg_overrides`` (dict) forwards experiment-level env/task params to the
  task's cfg_builder, overriding the values baked into its registration (e.g.
  ``{"gate_close_rate": 0.003}``) -- lets a config recipe tune the env without
  editing the task or the trainer flags.
  """
  from .base import MjlabTensorSafetyEnv
  s = spec(task_id)
  if adversary and not s.supports_adversary:
    raise ValueError(f"task '{task_id}' does not define an adversary")
  kw.setdefault("dense_reward", s.mode == CUMULATIVE)  # cumulative => dense
  ec = end_criterion if end_criterion is not None else s.end_criterion
  return MjlabTensorSafetyEnv(
    num_envs, device, cfg_builder=s.cfg_builder, margin_fn=s.margin_fn,
    ctrl_dim=s.ctrl_dim, dstb_dim=s.dstb_dim, adversary=adversary,
    end_criterion=ec, cfg_overrides=cfg_overrides, **{**s.kwargs, **kw})


def make_numpy(task_id: str, num_envs: int = 64, device: str = "cuda:0",
               adversary: bool = False, end_criterion: Optional[str] = None,
               cfg_overrides: Optional[dict] = None, **kw):
  """Classic SB3 VecEnv (for the SAC family / stock SB3 tooling).

  ``end_criterion`` (None -> the task's TaskSpec value; else an override) sets
  WHEN the episode ends from (g, l). ``cfg_overrides`` (dict) forwards env/task
  params to the task's cfg_builder. See :func:`make_tensor`.
  """
  from .base import MjlabNumpySafetyEnv
  s = spec(task_id)
  if adversary and not s.supports_adversary:
    raise ValueError(f"task '{task_id}' does not define an adversary")
  kw.setdefault("dense_reward", s.mode == CUMULATIVE)  # cumulative => dense
  ec = end_criterion if end_criterion is not None else s.end_criterion
  return MjlabNumpySafetyEnv(
    num_envs, device, cfg_builder=s.cfg_builder, margin_fn=s.margin_fn,
    ctrl_dim=s.ctrl_dim, dstb_dim=s.dstb_dim, adversary=adversary,
    end_criterion=ec, cfg_overrides=cfg_overrides, **{**s.kwargs, **kw})
