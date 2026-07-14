"""Deprecated alias: safe_mjlab_zoo was renamed to robot_safety_sandbox.

Kept for one transition cycle so existing scripts, checkpoint-adjacent
tooling, and the remote training clones keep working after a pull. Module
identity is preserved for the top level (`sys.modules` alias), so
``from safe_mjlab_zoo import spec, make_tensor`` etc. behave exactly like the
new name. Deep imports (``import safe_mjlab_zoo.sub.mod``) re-execute the
submodule under the alias name — prefer the new name for those.
"""

import sys
import warnings

import robot_safety_sandbox as _rss

warnings.warn(
  "safe_mjlab_zoo has been renamed to robot_safety_sandbox; "
  "update imports (this alias will be removed in a future release).",
  DeprecationWarning, stacklevel=2)

sys.modules[__name__] = _rss
for _name, _mod in list(sys.modules.items()):
  if _name.startswith("robot_safety_sandbox."):
    sys.modules["safe_mjlab_zoo." + _name[len("robot_safety_sandbox."):]] = _mod
