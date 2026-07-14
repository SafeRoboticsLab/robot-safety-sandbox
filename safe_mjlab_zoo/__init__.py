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
from .tasks import go2_crawl_twins as _go2_crawl_twins
from .tasks import classic_safety as _classic_safety
from .nominal import classic_dense as _classic_dense
from .nominal import go2_crawl_walker as _go2_crawl_walker
from .nominal import go2_walker as _go2_walker

# safety tasks (margins + safety_sb3 learners)
_go2_gap.register_all()
_go2_crawl.register_all()
_go2_stabilize.register_all()
# Digit's asset still lives in the mjlab FORK (phase-1 compat; vendoring is a
# release-packaging item) — a missing third-party asset must not take down the
# whole zoo on stock mjlab.
try:
  _digit_safety.register_all()
except ImportError as _e:
  import warnings
  warnings.warn(f"digit_safety tasks unavailable (asset not vendored): {_e}")
_classic_safety.register_all()
_go2_crawl_twins.register_all()
# nominal task policies (dense reward + vanilla SB3) — what filters wrap
_go2_walker.register_all()
_go2_crawl_walker.register_all()
_classic_dense.register_all()

__all__ = [
  "MjlabTensorSafetyEnv", "MjlabNumpySafetyEnv", "build_task_cfg",
  "TaskSpec", "register", "spec", "list_tasks", "make_tensor", "make_numpy",
]
