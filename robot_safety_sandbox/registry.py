"""Task registry: one place future work looks to run or add benchmark tasks.

    from robot_safety_sandbox import make_tensor, list_tasks
    env = make_tensor("go2_gap_chain", num_envs=2048)   # -> TensorVecEnv
    model = ReachAvoidPPO("MlpPolicy", env, normalize_obs=True, ...)

A :class:`TaskSpec` pins everything a benchmark run needs: the mjlab cfg
builder (spawn events + curricula), the reach-avoid margins, action dims, the
recommended learner, and the warm-start lineage (curriculum pipelines like
landing -> crossing -> chain are first-class here — they are how the hard
skills were actually learned).

Two task KINDS live side by side (a full filter experiment needs both):
  kind="safety"   margins (g, l) + a safety learner -> the certificate V(s) +
                  fallback policy. FOUR learners, one per (problem, players)
                  cell — the problem is a property of the TASK's margins, the
                  player count is a property of the RUN (``--adversary``):

                                   avoid (no l)      reach-avoid (real l)
                    single-player  SafetyPPO         ReachAvoidPPO
                    two-player     IsaacsPPO         GameplayPPO

                  ``default_algo`` picks the COLUMN (its problem is the task's
                  problem); :func:`algo_name` picks the row from the run's
                  ``adversary`` flag and returns the cell.
                  NB ``IsaacsPPO``/``IsaacsSAC`` CHANGED MEANING in safety_sb3
                  v0.2.0: they are now the two-player AVOID game (ISAACS eq. 7,
                  no target set); the two-player reach-avoid learner they used
                  to be is now ``GameplayPPO``/``GameplaySAC``.
  kind="nominal"  the TASK policy a filter wraps: dense env reward + VANILLA
                  SB3 (margin_fn=None; envs are auto-built in dense mode).
                  Registered under nominal/, trained with train_nominal.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

_REGISTRY: dict[str, "TaskSpec"] = {}

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
  margin_fn: Optional[Callable] = None  # (env) -> (g, l); None for nominal
  description: str = ""
  ctrl_dim: int = 12
  dstb_dim: int = 3              # adversary force dims (ISAACS)
  # Declares the task's PROBLEM via a learner name; algo_name() swaps in the
  # two-player learner of the same problem when a run passes adversary=True.
  # SafetyPPO | ReachAvoidPPO | IsaacsPPO | GameplayPPO | PPO (nominal).
  default_algo: str = "SafetyPPO"
  warmstart_from: Optional[str] = None  # previous pipeline stage task_id
  supports_adversary: bool = False
  kind: str = "safety"           # "safety" (margins) | "nominal" (dense task)
  # WHEN the episode ends from (g, l); one of END_CRITERIA. "failure" (default)
  # reproduces today's behavior for EVERY registered task — an audit (2026-07-17)
  # found none currently terminate on success. A run may override it via the
  # trainer's --end-criterion flag; the two knobs (this + terminal_type) are
  # orthogonal. Set "reach-avoid" on a task only if it should end on reach.
  end_criterion: str = "failure"
  kwargs: dict = field(default_factory=dict)  # extra bridge kwargs

  def __post_init__(self):
    if self.kind == "safety" and self.margin_fn is None:
      raise ValueError(f"safety task '{self.task_id}' needs a margin_fn")
    if self.end_criterion not in END_CRITERIA:
      raise ValueError(
        f"task '{self.task_id}' has end_criterion={self.end_criterion!r}; "
        f"must be one of {END_CRITERIA}")


def register(spec: TaskSpec) -> None:
  if spec.task_id in _REGISTRY:
    raise ValueError(f"task '{spec.task_id}' already registered")
  _REGISTRY[spec.task_id] = spec


def list_tasks(kind: Optional[str] = None) -> list[str]:
  return sorted(t for t, s in _REGISTRY.items()
                if kind is None or s.kind == kind)


def spec(task_id: str) -> TaskSpec:
  if task_id not in _REGISTRY:
    raise KeyError(
      f"unknown task '{task_id}'. Registered: {list_tasks()}. "
      "(Some tasks require their source repo on sys.path during the "
      "phase-1 compat period — see tasks/*.py and MIGRATION.md.)")
  return _REGISTRY[task_id]


AVOID = "avoid"
REACH_AVOID = "reach-avoid"

#: which PROBLEM (backup) each learner solves — ``default_algo`` names one of
#: these, and that is what fixes the task's problem; the player count comes from
#: the run. Mirrors safety_sb3.backups: avoid anchors on g, reach-avoid on
#: min(l, g).
_ALGO_PROBLEM = {
  "SafetyPPO": AVOID,           "IsaacsPPO": AVOID,          # ISAACS eq. 7
  "ReachAvoidPPO": REACH_AVOID, "GameplayPPO": REACH_AVOID,  # Gameplay eq. 6a
}
#: (problem, n_players) -> learner
_LEARNER = {
  (AVOID, 1): "SafetyPPO",       (AVOID, 2): "IsaacsPPO",
  (REACH_AVOID, 1): "ReachAvoidPPO", (REACH_AVOID, 2): "GameplayPPO",
}


def algo_name(task_id: str, adversary: bool = False) -> str:
  """The learner CLASS NAME for running ``task_id`` (names only — this module
  never imports safety_sb3, so the registry stays importable without it).

  The task's margins fix the PROBLEM (avoid vs reach-avoid); ``adversary`` fixes
  the PLAYER COUNT. Resolving both together is what keeps the 2x2 honest — the
  old code hardcoded IsaacsPPO for every adversarial run, which since safety_sb3
  v0.2.0 (where that name means the AVOID game) would silently turn every
  reach-avoid task into an avoid game.

  Also refuses the one pairing that is silently wrong: a reach-avoid learner on
  an avoid-only task (no target set). Pinning l to a constant does NOT make the
  reach-avoid backup compute the avoid value — a negative l empties the safe
  set, a non-negative one strips the lookahead — so it has no valid formulation
  and must not be reachable by accident. See margins.py.
  """
  s = spec(task_id)
  if s.default_algo not in _ALGO_PROBLEM:
    raise ValueError(
      f"task '{task_id}' declares default_algo='{s.default_algo}', which is "
      f"not a safety learner; known: {sorted(_ALGO_PROBLEM)}")
  if adversary and not s.supports_adversary:
    raise ValueError(f"task '{task_id}' does not define an adversary")
  problem = _ALGO_PROBLEM[s.default_algo]
  # margin_fns built by margins.compose/avoid_only carry has_target; anything
  # else (task-local margin builders) is assumed to declare a real l.
  if problem == REACH_AVOID and not getattr(s.margin_fn, "has_target", True):
    raise ValueError(
      f"task '{task_id}' is AVOID-ONLY (its margin_fn declares no target set) "
      f"but declares the reach-avoid learner '{s.default_algo}'. Avoid is not "
      f"a reach-avoid instance for ANY constant l — declare SafetyPPO (avoid; "
      f"--adversary then gives the two-player IsaacsPPO), or give the task a "
      f"real reach margin l. See margins.py / safety_sb3 RELEASE_NOTES v0.2.0.")
  return _LEARNER[(problem, 2 if adversary else 1)]


def make_tensor(task_id: str, num_envs: int = 2048, device: str = "cuda:0",
                adversary: bool = False, end_criterion: Optional[str] = None,
                **kw):
  """GPU-resident env (primary path; pair with safety_sb3 PPO learners).

  ``end_criterion`` (None -> the task's TaskSpec value; else an explicit
  override, one of :data:`END_CRITERIA`) sets WHEN the episode ends from (g, l).
  """
  from .base import MjlabTensorSafetyEnv
  s = spec(task_id)
  if adversary and not s.supports_adversary:
    raise ValueError(f"task '{task_id}' does not define an adversary")
  kw.setdefault("dense_reward", s.kind == "nominal")  # nominal => dense
  ec = end_criterion if end_criterion is not None else s.end_criterion
  return MjlabTensorSafetyEnv(
    num_envs, device, cfg_builder=s.cfg_builder, margin_fn=s.margin_fn,
    ctrl_dim=s.ctrl_dim, dstb_dim=s.dstb_dim, adversary=adversary,
    end_criterion=ec, **{**s.kwargs, **kw})


def make_numpy(task_id: str, num_envs: int = 64, device: str = "cuda:0",
               adversary: bool = False, end_criterion: Optional[str] = None,
               **kw):
  """Classic SB3 VecEnv (for the SAC family / stock SB3 tooling).

  ``end_criterion`` (None -> the task's TaskSpec value; else an override) sets
  WHEN the episode ends from (g, l). See :func:`make_tensor`.
  """
  from .base import MjlabNumpySafetyEnv
  s = spec(task_id)
  if adversary and not s.supports_adversary:
    raise ValueError(f"task '{task_id}' does not define an adversary")
  kw.setdefault("dense_reward", s.kind == "nominal")  # nominal => dense
  ec = end_criterion if end_criterion is not None else s.end_criterion
  return MjlabNumpySafetyEnv(
    num_envs, device, cfg_builder=s.cfg_builder, margin_fn=s.margin_fn,
    ctrl_dim=s.ctrl_dim, dstb_dim=s.dstb_dim, adversary=adversary,
    end_criterion=ec, **{**s.kwargs, **kw})
