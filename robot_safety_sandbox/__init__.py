"""robot_safety_sandbox: mjlab safety-benchmark environments for safety_sb3.

    from robot_safety_sandbox import make_tensor, list_tasks
    env = make_tensor("go2_gap_chain", num_envs=2048)

Tasks register lazily on import; phase-1 compat tasks additionally need their
source repo on sys.path (see tasks/*.py + MIGRATION.md).
"""

from .base import MjlabNumpySafetyEnv, MjlabTensorSafetyEnv, build_task_cfg
from .registry import (
  TaskSpec, algo_name, list_tasks, make_numpy, make_tensor, register, spec)

from .tasks import digit_safety as _digit_safety
from .tasks import go2_crawl as _go2_crawl
from .tasks import go2_gap as _go2_gap
from .tasks import go2_stabilize as _go2_stabilize
from .tasks import go2_crawl_twins as _go2_crawl_twins
from .tasks import go2_gap_brake_or_jump as _go2_gap_brake_or_jump
from .nominal import go2_crawl_walker as _go2_crawl_walker
from .nominal import go2_walker as _go2_walker

# safety tasks (margins + safety_sb3 learners)
_go2_gap.register_all()
_go2_crawl.register_all()
_go2_stabilize.register_all()
_digit_safety.register_all()
_go2_crawl_twins.register_all()
_go2_gap_brake_or_jump.register_all()  # split test: harvested-state RA vs avoid twins
# nominal task policies (dense reward + vanilla SB3) — what filters wrap
_go2_walker.register_all()
_go2_crawl_walker.register_all()

__all__ = [
  "MjlabTensorSafetyEnv", "MjlabNumpySafetyEnv", "build_task_cfg",
  "TaskSpec", "register", "spec", "list_tasks", "make_tensor", "make_numpy",
  "algo_name",
]
