"""Task registry: one place future work looks to run or add benchmark tasks.

    from safe_mjlab_zoo import make_tensor, list_tasks
    env = make_tensor("go2_gap_chain", num_envs=2048)   # -> TensorVecEnv
    model = ReachAvoidPPO("MlpPolicy", env, normalize_obs=True, ...)

A :class:`TaskSpec` pins everything a benchmark run needs: the mjlab cfg
builder (spawn events + curricula), the reach-avoid margins, action dims, the
recommended learner, and the warm-start lineage (curriculum pipelines like
landing -> crossing -> chain are first-class here — they are how the hard
skills were actually learned).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

_REGISTRY: dict[str, "TaskSpec"] = {}


@dataclass
class TaskSpec:
  task_id: str
  cfg_builder: Callable          # (play: bool) -> ManagerBasedRlEnvCfg
  margin_fn: Callable            # (env) -> (g, l)
  description: str = ""
  ctrl_dim: int = 12
  dstb_dim: int = 3              # adversary force dims (ISAACS)
  default_algo: str = "SafetyPPO"   # SafetyPPO | ReachAvoidPPO | IsaacsPPO
  warmstart_from: Optional[str] = None  # previous pipeline stage task_id
  supports_adversary: bool = False
  kwargs: dict = field(default_factory=dict)  # extra bridge kwargs


def register(spec: TaskSpec) -> None:
  if spec.task_id in _REGISTRY:
    raise ValueError(f"task '{spec.task_id}' already registered")
  _REGISTRY[spec.task_id] = spec


def list_tasks() -> list[str]:
  return sorted(_REGISTRY)


def spec(task_id: str) -> TaskSpec:
  if task_id not in _REGISTRY:
    raise KeyError(
      f"unknown task '{task_id}'. Registered: {list_tasks()}. "
      "(Some tasks require their source repo on sys.path during the "
      "phase-1 compat period — see tasks/*.py and MIGRATION.md.)")
  return _REGISTRY[task_id]


def make_tensor(task_id: str, num_envs: int = 2048, device: str = "cuda:0",
                adversary: bool = False, **kw):
  """GPU-resident env (primary path; pair with safety_sb3 PPO learners)."""
  from .base import MjlabTensorSafetyEnv
  s = spec(task_id)
  if adversary and not s.supports_adversary:
    raise ValueError(f"task '{task_id}' does not define an adversary")
  return MjlabTensorSafetyEnv(
    num_envs, device, cfg_builder=s.cfg_builder, margin_fn=s.margin_fn,
    ctrl_dim=s.ctrl_dim, dstb_dim=s.dstb_dim, adversary=adversary,
    **{**s.kwargs, **kw})


def make_numpy(task_id: str, num_envs: int = 64, device: str = "cuda:0",
               adversary: bool = False, **kw):
  """Classic SB3 VecEnv (for the SAC family / stock SB3 tooling)."""
  from .base import MjlabNumpySafetyEnv
  s = spec(task_id)
  if adversary and not s.supports_adversary:
    raise ValueError(f"task '{task_id}' does not define an adversary")
  return MjlabNumpySafetyEnv(
    num_envs, device, cfg_builder=s.cfg_builder, margin_fn=s.margin_fn,
    ctrl_dim=s.ctrl_dim, dstb_dim=s.dstb_dim, adversary=adversary,
    **{**s.kwargs, **kw})
