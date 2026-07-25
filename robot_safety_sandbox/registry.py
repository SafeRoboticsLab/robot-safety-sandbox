"""Task registry: one place future work looks to run or add benchmark tasks.

    from robot_safety_sandbox import make_tensor, list_tasks
    env = make_tensor("go2_gap_chain", num_envs=2048)   # -> TensorVecEnv
    model = ReachAvoidPPO1P("MlpPolicy", env, normalize_obs=True, ...)

A :class:`TaskSpec` pins everything a benchmark run needs: the mjlab cfg
builder (spawn events + curricula), the reach-avoid margins, the action dims,
and the task's ``mode``. Curriculum LINEAGE (landing -> crossing -> chain) is
deliberately NOT a field here: it lives in ``docs/log/experiments.md``, which is
where it is actually maintained.

Here's a MAP to navigate the codebase — **Mode. Algorithm. Players.**

    M = Mode       Safety | ReachAvoid | Cumulative    the Bellman operator
    A = Algorithm  PPO | SAC | A2C | DQN               the RL update rule
    P = Players    1P | 2P                             single-player | zero-sum

A learner's NAME is those three letters concatenated in that order —
``SafetyPPO1P``, ``ReachAvoidSAC2P`` — and :func:`algo_name` is that
concatenation and nothing else. No lookup table, no per-task override:

    M comes from the TASK   its ``mode`` (below), a property of its margins
    A comes from the RUN    the trainer family (on_policy -> PPO, off_policy -> SAC)
    P comes from the RUN    the ``--adversary`` flag

Every task declares exactly one axis of its own, its ``mode`` — the safety_sb3
BACKUP it is trained under (see :data:`MODES`):

  mode="safety"       AVOID:       V = min(g, gamma V')          margins, no l
  mode="reach-avoid"  REACH_AVOID: V = min(g, max(l, gamma V'))  margins with l
  mode="cumulative"   CUMULATIVE:  V = r + gamma (1-d) V'        dense env reward,
                      i.e. the plain-RL TASK policy a filter wraps (margin_fn=None;
                      envs are auto-built in dense mode, trained with STOCK SB3).

A full filter experiment needs both layers: a cumulative task policy (pi_task) and
a safety task supplying the certificate V(s) + fallback.

The two SAFETY modes give four learners per algorithm, one per (M, P) cell — the
mode is a property of the TASK's margins, the player count a property of the RUN:

                       avoid (no l)      reach-avoid (real l)
        single-player  SafetyPPO1P       ReachAvoidPPO1P
        two-player     SafetyPPO2P       ReachAvoidPPO2P

``mode`` picks the COLUMN, ``--adversary`` the row. CUMULATIVE has no P axis at
all: it is stock ``stable_baselines3`` (``"PPO"`` / ``"SAC"``, no suffix) and
there is no two-player cumulative game.

NB the 2P cell is a DIFFERENT ALGORITHM in the two families, not the same game
with a different optimizer: ``*SAC2P`` is the minimax game on one shared
joint-action critic ``Q(s, [a_ctrl, a_dstb])``, while ``*PPO2P`` is an
alternating best-response approximation with two independent ``V(s)`` nets, two
rollout buffers and a phase machine. See docs/API.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

#: MAP letter **M** — mode -> the learner-name prefix that Bellman operator earns.
#: CUMULATIVE earns none: it is stock stable_baselines3, not a safety_sb3 learner.
_PREFIX = {AVOID: "Safety", REACH_AVOID: "ReachAvoid", CUMULATIVE: ""}
#: MAP letter **A** — trainer family -> the algorithm it runs. The A slot also
#: has A2C / DQN in principle; neither is wired to a trainer in this zoo yet.
_ALG = {"on_policy": "PPO", "off_policy": "SAC"}
#: the trainer families (``examples/train.py --family``)
FAMILIES = tuple(_ALG)

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
  dstb_dim: int = 3              # adversary force dims (two-player games)
  # REQUIRED. Which BACKUP values this task: one of MODES. This is the task's
  # whole say in the MAP — it supplies the **M**, and nothing else here names a
  # learner. It also fixes whether the env is built in dense-reward mode and
  # whether margins are required.
  mode: Optional[str] = None
  supports_adversary: bool = False
  # WHEN the episode ends from (g, l); one of END_CRITERIA. "failure" (default)
  # reproduces today's behavior for EVERY registered task — an audit (2026-07-17)
  # found none currently terminate on success. A run may override it via the
  # trainer's --end-criterion flag; the two knobs (this + terminal_type) are
  # orthogonal. Set "reach-avoid" on a task only if it should end on reach.
  end_criterion: str = "failure"
  kwargs: dict = field(default_factory=dict)  # extra bridge kwargs

  def __post_init__(self):
    if self.mode is None:
      raise ValueError(
        f"task '{self.task_id}' declares no mode=; every task must name the "
        f"backup it trains under, one of {MODES}. The learner is DERIVED from "
        f"it (see algo_name), so there is nothing else to declare.")
    if self.mode not in MODES:
      raise ValueError(f"task '{self.task_id}' has mode={self.mode!r}; "
                       f"must be one of {MODES}")
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


def list_tasks(mode: Optional[str] = None) -> list[str]:
  """Registered task ids, optionally filtered to one :data:`MODES` entry."""
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


def algo_name(task_id: str, adversary: bool = False,
              family: str = "on_policy") -> str:
  """The learner CLASS NAME for running ``task_id`` — the MAP, spelled out.

  It is a FORMULA, not a lookup: **M**ode + **A**lgorithm + **P**layers,
  concatenated. Nothing in the registry can override it; a task supplies only
  its ``mode``.

      algo_name("go2_stabilize")                            -> ReachAvoidPPO1P
      algo_name("go2_stabilize", adversary=True)            -> ReachAvoidPPO2P
      algo_name("go2_stabilize", family="off_policy")       -> ReachAvoidSAC1P
      algo_name("digit_stabilize_avoid", adversary=True)    -> SafetyPPO2P

  Names only — this module never imports safety_sb3, so a cumulative-only
  install (dense reward + vanilla SB3, see base.py) still imports the registry.

  mode=CUMULATIVE returns the bare algorithm (``"PPO"`` / ``"SAC"``): the
  plain-RL task policy trains with STOCK stable_baselines3, which keeps the
  checkpoint a vanilla SB3 zip. It has no P axis — there is no two-player
  cumulative game.

  Two pairings are refused rather than resolved:

  * a two-player CUMULATIVE run — no such game exists;
  * a reach-avoid learner on an avoid-only task (no target set). Pinning l to a
    constant does NOT make the reach-avoid backup compute the avoid value — a
    negative l empties the safe set, a non-negative one strips the lookahead —
    so it has no valid formulation and must not be reachable by accident. See
    margins.py.
  """
  if family not in _ALG:
    raise ValueError(f"unknown family={family!r}; one of {FAMILIES}")
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
      f"two-player Safety{_ALG[family]}2P), or give the task a real reach "
      f"margin l. See margins.py.")
  if s.mode == CUMULATIVE:
    return _ALG[family]                      # stock stable_baselines3, 1 player
  return f"{_PREFIX[s.mode]}{_ALG[family]}{'2P' if adversary else '1P'}"


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
