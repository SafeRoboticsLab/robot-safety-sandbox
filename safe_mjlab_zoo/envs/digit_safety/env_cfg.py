"""Agility Digit v3 flat-ground standing safety filter — the humanoid analog of
``go2_stabilize`` (ISAACS Tier 2), and a "no-machinery" task:

  * flat terrain, stock spawns from the calibrated safety cfg — no custom spawn
    distribution, no curriculum, no staged pipeline;
  * zero velocity command — the target is a stable stand;
  * boundary experience comes from the ADVERSARY (worst-case torso force), the
    way ``go2_stabilize`` does, not from a widened spawn distribution.

Phase-1 compat: the Digit robot asset + calibrated safety env live in the mjlab
fork (``mjlab.tasks.velocity.config.digit_v3``), imported here rather than
vendored. The cfg is otherwise used UNMODIFIED — the zoo bridge auto-detects the
``actor`` observation group, and drops ``push_robot`` itself (the adversary
replaces it). Action scaling is the bridge's ``ctrl_gain`` (see
tasks/digit_safety), so the policy keeps SB3's natural [-1, 1] action range.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.digit_v3.env_cfgs import (
  digit_v3_flat_safety_box_rigidtoe_env_cfg,
  digit_v3_flat_safety_rigidtoe_env_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


def _pin_twist(cfg: ManagerBasedRlEnvCfg, vx: float) -> None:
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.ranges.lin_vel_x = (vx, vx)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  if hasattr(twist, "rel_standing_envs"):
    twist.rel_standing_envs = 1.0 if vx == 0.0 else 0.0
  if hasattr(twist, "heading_command"):
    twist.heading_command = False


# DEPLOYMENT-FAITHFUL OBS (2026-07-10): the actor keeps ONLY the 92 deployable
# dims. We briefly promoted the critic's foot terms (foot_height/air_time/
# contact/forces) into the actor (92 -> 104); it accelerated learning, but those
# signals don't exist at deployment (no foot F/T sensing in ar-control/LLAPI) —
# the exported policy ran with baked-constant foot obs, was blind to its real
# foot state, and drifted into a tiptoe-crouch in ar-control. Foot flatness is
# FK-recoverable from joint_pos + projected_gravity (toe joints are encoded),
# so the policy can learn to regulate it from proprioception alone; the planted
# constraint lives in the MARGIN (sim state), not the obs, so the training
# pressure is unchanged. (SB3's ActorCriticPolicy feeds pi and vf the same
# input — the env's separate "critic" group never reaches SB3 — so this also
# keeps the value function on the 92 dims. A custom asymmetric policy is the
# fallback if 92-dim value learning stalls.)


def digit_stabilize_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  # RIGIDTOE model (2026-07-10): rigid toe transmission + fitted effective
  # holding gains — the creep-corrected plant whose mirror survives
  # ar-control's full 28 s sway. Obs dims unchanged vs the calibrated model.
  cfg = digit_v3_flat_safety_rigidtoe_env_cfg(play=play)
  _pin_twist(cfg, 0.0)  # zero command: the target is a stable stand
  # Stabilize exemplar: no curriculum. Drop the inherited command_vel curriculum
  # (a no-op under a pinned zero command, and its ang_vel_z_min/_max log keys
  # collide when SB3's human logger truncates them).
  cfg.curriculum = {}
  # The RL reward is g(s) (the zoo hook, added by build_task_cfg); the inherited
  # locomotion reward terms are unused (the bridge reads extras['zoo_g'], not
  # mjlab's reward sum) and their long names (Episode_Reward/joint_deviation_*)
  # collide under SB3's log-key truncation. Drop them; the hook is re-added.
  cfg.rewards = {}
  return cfg


def digit_box_stabilize_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Box-balancing safety task on the rigidtoe plant (the ORIGINAL project
  task): stand in place while keeping the free box on the forearms.

  Same no-machinery treatment as ``digit_stabilize_env_cfg`` (zero command,
  no curriculum, rewards replaced by the zoo hook); box-specific events
  (reset_box_state_to_arms) are inherited untouched. Actor obs gains
  box_pose/box_vel vs the no-box task -> checkpoints are NOT warm-start
  compatible across the two families.
  """
  cfg = digit_v3_flat_safety_box_rigidtoe_env_cfg(play=play)
  _pin_twist(cfg, 0.0)
  cfg.curriculum = {}
  cfg.rewards = {}
  # The legacy box termination hard-terminates on foot height > 0.1 m (the
  # old flat-foot-in-g design). Under the drop-spawn (base z +0.03-0.10) the
  # foot sites START above 0.1 -> instant termination at spawn (reset->die
  # loops, ep_len ~4, NaN'd PPO on 2026-07-10 run ueqe56jt). It also
  # contradicts the dynamic-recovery design: recovery steps must not
  # terminate; stance flatness is enforced by the ANNEALED planted term in
  # the stay margin instead. Disable it; termination = fall + box only,
  # which our g (g_digit_box_stand / g_digit_box_stabilize) mirrors.
  cfg.terminations["fell_over"].params["max_foot_height"] = 10.0
  # Box-contact physics margin: the box task's convex collisions saturate the
  # default CCD budget (637 "increase ccd_iterations" warnings in the crashed
  # run) -> bad contact impulses -> occasional non-finite states -> policy NaN
  # at ~7M (run 5lxb6ey0). Raise the budget, and contain any residual NaNs as
  # per-env terminations (logged as Episode_Termination/nan_term — WATCH this
  # rate; if it is non-negligible or correlates with box events, the physics
  # needs a deeper fix, not the band-aid).
  # ccd_iterations: 2000 AND 1000 both OOM warp's CCD buffer (identical 7.9GB
  # request — allocation is not linear in the setting). 500 is known-good
  # (constructs + trains); the occasional CCD-saturation NaN is contained by
  # nan_term below. WATCH Episode_Termination/nan_term.
  cfg.sim.mujoco.ccd_iterations = 500
  from mjlab.envs import mdp as _envs_mdp
  from mjlab.managers.termination_manager import TerminationTermCfg
  cfg.terminations["nan_term"] = TerminationTermCfg(func=_envs_mdp.nan_detection)
  return cfg
