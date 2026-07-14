"""Agility Robotics Digit v3 humanoid (vendored asset, mirrors assets_go2).

MJCF + meshes live in ``xmls/``; robot cfg entry points are re-exported here.
"""

from safe_mjlab_zoo.envs.assets_digit.digit_constants import (
  DIGIT_ACTION_SCALE as DIGIT_ACTION_SCALE,
)
from safe_mjlab_zoo.envs.assets_digit.digit_constants import (
  get_digit_robot_cfg as get_digit_robot_cfg,
)
from safe_mjlab_zoo.envs.assets_digit.digit_constants import (
  get_digit_robot_cfg_calibrated as get_digit_robot_cfg_calibrated,
)
from safe_mjlab_zoo.envs.assets_digit.digit_constants import (
  get_digit_robot_cfg_calibrated_rigidtoe as get_digit_robot_cfg_calibrated_rigidtoe,
)
