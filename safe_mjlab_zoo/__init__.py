"""safe_mjlab_zoo: mjlab safety-benchmark environments for safety_sb3.

    from safe_mjlab_zoo import make_tensor, list_tasks
    env = make_tensor("go2_gap_chain", num_envs=2048)

Tasks register lazily on import; phase-1 compat tasks additionally need their
source repo on sys.path (see tasks/*.py + MIGRATION.md).
"""

from .base import MjlabNumpySafetyEnv, MjlabTensorSafetyEnv, build_task_cfg
from .registry import TaskSpec, list_tasks, make_numpy, make_tensor, register, spec

from .tasks import digit_safety as _digit_safety
from .tasks import go2_crawl as _go2_crawl
from .tasks import go2_gap as _go2_gap
from .tasks import go2_stabilize as _go2_stabilize

_go2_gap.register_all()
_go2_crawl.register_all()
_go2_stabilize.register_all()
_digit_safety.register_all()

__all__ = [
  "MjlabTensorSafetyEnv", "MjlabNumpySafetyEnv", "build_task_cfg",
  "TaskSpec", "register", "spec", "list_tasks", "make_tensor", "make_numpy",
]
