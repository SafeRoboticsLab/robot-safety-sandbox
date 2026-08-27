"""Landing sub-task: spawn mid-air over a gap with enough forward velocity to
reach the far platform, and learn to SOFT-LAND into a safe stance.

Avoid-only test, mode="safety" -> SafetyPPO1P (uses g(s) only, ignores l): the
winning-landing
signal is rare and gets buried/explodes at small env counts, so this task is
meant to be run at very large ``num_envs`` (mjlab parallelism) so enough
successful landings appear per iteration to learn from.

Reuses the gaps env (island terrain + height_scan proprioception, depth
dropped) and replaces the reset with a mid-air-over-gap launch.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

from robot_safety_sandbox.envs.terrains.island import ISLAND_CROSSING_TERRAINS_CFG
from robot_safety_sandbox.envs.go2_gap.gap import unitree_go2_gap_reach_avoid_env_cfg

# [x, y, z, roll, pitch, yaw] offset relative to (env_origin + default pose);
# origin is the gap's near edge -> spawn just over the gap, elevated (apex).
_POSE_LOW = [0.05, -0.10, 0.25, -0.10, -0.15, -0.15]
_POSE_HIGH = [0.25, 0.10, 0.45, 0.10, 0.15, 0.15]
# [vx, vy, vz, wx, wy, wz]: strong forward velocity that clears the gap.
_VEL_LOW = [2.50, -0.10, -0.50, -0.20, -0.20, -0.20]
_VEL_HIGH = [3.50, 0.10, 0.30, 0.20, 0.20, 0.20]

# BROAD airborne distribution (S_airborne) for CERTIFICATE HARVESTING, NOT
# for training. Spans near-platform -> over-void -> far, low->high apex, and slow/
# backward -> fast momentum, so the frozen lander SUCCEEDS from the recoverable
# subset and FAILS from the doomed subset -> a balanced, boundary-rich dataset for
# fitting V_safelanding. (The narrow ranges above are ~96% success -> no boundary.)
_POSE_LOW_BROAD = [-0.30, -0.20, 0.10, -0.40, -0.40, -0.60]
_POSE_HIGH_BROAD = [0.80, 0.20, 0.60, 0.40, 0.40, 0.60]
_VEL_LOW_BROAD = [-0.50, -0.60, -1.50, -1.00, -1.00, -1.00]
_VEL_HIGH_BROAD = [4.00, 0.60, 1.00, 1.00, 1.00, 1.00]

# TUBE distribution for the REACH policy: the continuum from S_edge (grounded
# at/near the edge, z offset ~0, rest->momentum) through the launch to S_airborne
# (lifted, launch momentum). z from 0 (grounded) supplies the origination states the
# reverse curriculum used to anneal to; forward-only momentum (no backward/no wild
# spin) keeps it a plausible launch arc rather than the doomed harvest box.
_POSE_LOW_TUBE = [-0.40, -0.15, 0.00, -0.25, -0.25, -0.40]
_POSE_HIGH_TUBE = [0.80, 0.15, 0.50, 0.25, 0.25, 0.40]
_VEL_LOW_TUBE = [0.00, -0.40, -0.50, -0.60, -0.60, -0.60]
_VEL_HIGH_TUBE = [4.00, 0.40, 1.00, 0.60, 0.60, 0.60]
# Reweight the tube toward standstill ORIGINATION: this fraction of envs spawn at
# S_edge (grounded, rest->slow), the rest across the airborne continuum. Uniform
# over the full tube box was airborne-heavy (latched_frac~0.6 in the ablation -> the policy
# under-trained grounded origination and failed 100% from standstill); this biases
# the training distribution toward the states the reach policy must actually learn.
_TUBE_GROUNDED_FRAC = 0.55

# EDGE distribution (S_edge) for the standstill-INITIATION probe / eval video only:
# grounded (z~0) on the near platform up to the edge, from rest to a slow vx sweep.
# vx=0 is the pure standstill-initiation test; the sweep maps the initiation envelope.
_POSE_LOW_EDGE = [-0.35, -0.12, 0.00, -0.15, -0.15, -0.30]
_POSE_HIGH_EDGE = [0.00, 0.12, 0.05, 0.15, 0.15, 0.30]
_VEL_LOW_EDGE = [0.00, -0.20, -0.20, -0.30, -0.30, -0.30]
_VEL_HIGH_EDGE = [1.00, 0.20, 0.20, 0.30, 0.30, 0.30]


def reset_midair_land(env, env_ids, asset_cfg=SceneEntityCfg("robot"),
                      broad=False, tube=False, edge=False):
  """Spawn over the gap. Narrow (default): forward velocity that clears the gap
  (lander training). ``broad``: wide airborne box (S_airborne) for certificate harvesting.
  ``tube``: the S_edge<->S_airborne continuum (reach training). ``edge``:
  grounded near-edge, rest->slow (S_edge) for the standstill-initiation probe/video.
  Precedence: edge > tube > broad > narrow."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return
  asset = env.scene[asset_cfg.name]
  device = env.device
  n = int(len(env_ids))
  root = asset.data.default_root_state[env_ids].clone()
  def _samp(lo, hi):
    return sample_uniform(torch.tensor(lo, device=device),
                          torch.tensor(hi, device=device), (n, 6), device)
  if tube:
    # MIXTURE (reweighted): grounded_frac at S_edge, the rest airborne-continuum.
    grounded = torch.rand(n, 1, device=device) < _TUBE_GROUNDED_FRAC
    pose = torch.where(grounded, _samp(_POSE_LOW_EDGE, _POSE_HIGH_EDGE),
                       _samp(_POSE_LOW_TUBE, _POSE_HIGH_TUBE))
    vel = torch.where(grounded, _samp(_VEL_LOW_EDGE, _VEL_HIGH_EDGE),
                      _samp(_VEL_LOW_TUBE, _VEL_HIGH_TUBE))
  else:
    if edge:
      plo, phi, vlo, vhi = _POSE_LOW_EDGE, _POSE_HIGH_EDGE, _VEL_LOW_EDGE, _VEL_HIGH_EDGE
    elif broad:
      plo, phi, vlo, vhi = _POSE_LOW_BROAD, _POSE_HIGH_BROAD, _VEL_LOW_BROAD, _VEL_HIGH_BROAD
    else:
      plo, phi, vlo, vhi = _POSE_LOW, _POSE_HIGH, _VEL_LOW, _VEL_HIGH
    pose, vel = _samp(plo, phi), _samp(vlo, vhi)
  positions = root[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[env_ids]
  orientations = quat_mul(
    root[:, 3:7], quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
  )
  velocities = root[:, 7:13] + vel
  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def unitree_go2_landing_env_cfg(
  play: bool = False, gap_width: float | None = None,
  broad: bool = False, tube: bool = False, edge: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_go2_gap_reach_avoid_env_cfg(play=play)
  tg = replace(ISLAND_CROSSING_TERRAINS_CFG)
  if gap_width is not None:
    # Collapse the width RANGE to a single value so every patch is the SAME fixed
    # gap (difficulty then has no effect) — RAAS v2 pins gap 0.30. Copy the dict +
    # sub-cfg so the module-level ISLAND_CROSSING_TERRAINS_CFG is never mutated.
    island = replace(tg.sub_terrains["island"], gap_width_range=(gap_width, gap_width))
    tg.sub_terrains = {**tg.sub_terrains, "island": island}
  cfg.scene.terrain.terrain_generator = tg
  cfg.events["reset_base"] = EventTermCfg(func=reset_midair_land, mode="reset",
                                          params={"broad": broad, "tube": tube, "edge": edge})
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
  if not play and "push_robot" in cfg.events:
    cfg.events.pop("push_robot", None)  # avoid-only landing test: no extra disturbance
  return cfg
