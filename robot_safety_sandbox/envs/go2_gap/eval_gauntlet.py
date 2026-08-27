"""The gap gauntlet, as an evaluation PRESET rather than a script.

This is everything that used to make ``examples/eval_filter.py`` a gap-terrain
harness end to end, and all of it is gap-specific, which is why it lives here
next to the terrain rather than in ``robot_safety_sandbox/eval/``:

  * a STANDING spawn on the approach, so the walker walks in naturally;
  * a PINNED gap width and gap count (feasible vs infeasible is a CLI knob, not
    a curriculum) and an optional widened island so a walking approach fits;
  * a fixed forward command both policies see;
  * the blind flat walker's exact observation group grafted in as "actor", so
    the nominal reads what it was trained on while the twin reads its own group;
  * the CROSSING and LIVELOCK rates -- the gap's own reading of "did the task
    still get done", which is meaningless on any other terrain.

It reproduces the gap-gauntlet configuration.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import replace

import torch

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...eval.metrics import Metric, StepRecord
from ...eval.presets import EvalPreset, register


# --- env surgery -------------------------------------------------------------

def _reset_standing(env, env_ids, x_lo=0.15, x_hi=0.45,
                    asset_cfg=SceneEntityCfg("robot")):
  """Standing spawn on the approach: the WALKER walks in naturally."""
  from mjlab.utils.lab_api.math import sample_uniform
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  asset = env.scene[asset_cfg.name]
  n = len(env_ids)
  root = asset.data.default_root_state[env_ids].clone()
  pos = root[:, 0:3] + env.scene.env_origins[env_ids]
  pos[:, 0] = env.scene.env_origins[env_ids, 0] + sample_uniform(
    x_lo, x_hi, (n,), env.device)
  asset.write_root_link_pose_to_sim(
    torch.cat([pos, root[:, 3:7]], dim=-1), env_ids=env_ids)
  asset.write_root_link_velocity_to_sim(
    torch.zeros(n, 6, device=env.device), env_ids=env_ids)


def _reset_standing_full(env, env_ids, x_lo=-1.5, x_hi=-1.5,
                         asset_cfg=SceneEntityCfg("robot")):
  """The same, plus the DEFAULT joint pose.

  The brake_or_jump family dropped the joint-reset event, so a spawn that only
  writes the root pose leaves the robot folded on the ground. Used by the
  walk-in variant of the gauntlet (``--gap-full-pose``).
  """
  _reset_standing(env, env_ids, x_lo, x_hi, asset_cfg)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  asset = env.scene[asset_cfg.name]
  jp = asset.data.default_joint_pos[env_ids].clone()
  asset.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)


def gap_cfg_transform(*, gap_width: float, n_gaps: int, episode_s: float,
                      cmd_vx: float, spawn_x=(0.15, 0.45),
                      island_length=None, full_pose: bool = False,
                      full_horizon: bool = False):
  """Build the ``cfg -> cfg`` surgery for a gap gauntlet run."""

  def transform(cfg):
    cfg.episode_length_s = episode_s
    cfg.curriculum = {}
    cfg.events["reset_base"] = EventTermCfg(
      func=_reset_standing_full if full_pose else _reset_standing,
      mode="reset", params={"x_lo": spawn_x[0], "x_hi": spawn_x[1]})
    cfg.events.pop("handover_joints", None)
    cfg.events.pop("randomize_terrain", None)
    cfg.events.pop("push_robot", None)

    # fixed forward command (both policies see the same, in-distribution value)
    twist = cfg.commands["twist"]
    twist.resampling_time_range = (1.0e9, 1.0e9)
    twist.ranges.lin_vel_x = (cmd_vx, cmd_vx)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(twist, "rel_standing_envs"):
      twist.rel_standing_envs = 0.0
    if hasattr(twist, "heading_command"):
      twist.heading_command = False

    # pin the gap width (feasible vs infeasible is a CLI knob, not a curriculum).
    # Terrain cfgs differ across task families -> only set fields that exist.
    gen = cfg.scene.terrain.terrain_generator
    if gen is None:
      raise SystemExit(
        "the 'gap_gauntlet' preset needs a terrain generator, and this task's "
        "terrain is flat (terrain_generator=None). Run it on a gap task "
        "(go2_gap_chain / go2_gap_brake_or_jump_*), or drop --preset.")
    subs = {}
    for k, v in gen.sub_terrains.items():
      names = {f.name for f in dataclasses.fields(v)}
      kw = {}
      if "gap_width_range" in names:
        kw["gap_width_range"] = (gap_width, gap_width)
      if "n_gaps_max" in names:
        kw["n_gaps_max"] = n_gaps
      if island_length is not None and "island_length" in names:
        # walk-in-viable eval variant: the native 0.7 m island cannot host a
        # walking approach; extending it keeps the SAME gap geometry (origin at
        # the gap face) while giving the walker a real approach corridor.
        kw["island_length"] = island_length
      subs[k] = replace(v, **kw) if kw else v
    cfg.scene.terrain.terrain_generator = replace(gen, sub_terrains=subs,
                                                  curriculum=False)

    if full_horizon:
      for k in [k for k in cfg.terminations if k != "time_out"]:
        cfg.terminations.pop(k)

    # graft the walker's exact blind actor group as 'actor'
    from robot_safety_sandbox.envs.velocity.go2 import unitree_go2_flat_env_cfg
    walk_group = copy.deepcopy(
      unitree_go2_flat_env_cfg(play=True).observations["actor"])
    assert "height_scan" not in walk_group.terms
    cfg.observations["actor"] = walk_group
    return cfg

  return transform


# --- the gap's own success reading -------------------------------------------

class GapProgress(Metric):
  """Crossing / livelock / reach, in the gap's own x-coordinate.

  ``crossed`` = the robot got past ``rest_x``; ``livelocked`` = the episode ran
  out (or the window closed) with the robot still short of the gap face, which
  is the avoid twin's signature failure -- safe and stuck. Both are counted per
  episode with the survivors included in the denominator, since a filter whose
  whole purpose is to keep robots alive completes the fewest episodes.
  """

  def __init__(self, num_envs: int, device: str, *, gap_x: float,
               rest_x: float):
    self.gap_x, self.rest_x = float(gap_x), float(rest_x)
    z = lambda dt: torch.zeros(num_envs, dtype=dt, device=device)  # noqa: E731
    self.crossed = z(torch.bool)
    self.max_x = z(torch.float32)
    self.n_env = num_envs
    self.episodes = self.crossings = 0
    self.timeouts_before_gap = self.timeouts_past_gap = 0
    self.finals: list[torch.Tensor] = []
    self._x = None

  def _x_rel(self, env):
    return (env.robot.data.root_link_pos_w[:, 0]
            - env.mj.scene.env_origins[:, 0])

  def update(self, rec: StepRecord) -> None:
    x = self._x_rel(rec.env)
    self._x = x
    self.crossed |= x > self.rest_x
    self.max_x = torch.maximum(self.max_x, x)
    o = rec.out
    d = o.done
    if bool(d.any()):
      self.episodes += int(d.sum())
      self.crossings += int((self.crossed & d).sum())
      before = o.truncated & ~self.crossed & (x < self.gap_x + 0.2)
      self.timeouts_before_gap += int(before.sum())     # livelock signature
      self.timeouts_past_gap += int((o.truncated & self.crossed).sum())
      self.finals.append(self.max_x[d].clone())
      self.crossed &= ~d
      self.max_x[d] = 0.0

  def result(self) -> dict:
    # censored survivors: alive at eval end without completing an episode.
    # With a filter engaged these are LIVELOCKED-SAFE robots; omitting them
    # silently inflates every rate.
    alive = self.max_x > 0.0
    censored = int(alive.sum())
    censored_before = int((alive & (self._x < self.gap_x + 0.2)).sum()) \
        if self._x is not None else 0
    denom = max(self.episodes + censored, 1)
    maxx = torch.cat(self.finals) if self.finals else torch.zeros(1)
    return dict(
      crossings=self.crossings,
      crossing_rate=self.crossings / denom,
      livelock_rate=(self.timeouts_before_gap + censored_before) / denom,
      timeouts_before_gap=self.timeouts_before_gap,
      timeouts_past_gap=self.timeouts_past_gap,
      max_x_mean=float(maxx.mean()),
      max_x_p90=float(maxx.quantile(0.9)),
    )


# --- registration ------------------------------------------------------------

def _add_args(p) -> None:
  g = p.add_argument_group("gap_gauntlet preset")
  g.add_argument("--gap-width", type=float, default=0.35)
  g.add_argument("--n-gaps", type=int, default=1)
  g.add_argument("--gap-x", type=float, default=2.5,
                 help="x_rel of the gap face (chain: 2.5; single-gap: 0.0)")
  g.add_argument("--rest-x", type=float, default=5.0,
                 help="x_rel counted as CROSSED (chain: 5.0; single-gap: ~1.2)")
  g.add_argument("--spawn-x", type=float, nargs=2, default=(0.15, 0.45),
                 help="standing-spawn x_rel range (single-gap: -1.9 -1.5)")
  g.add_argument("--island-length", type=float, default=None,
                 help="override island_length on island-type terrains "
                      "(walk-in-viable eval: 3.0)")
  g.add_argument("--gap-full-pose", action="store_true",
                 help="also write the DEFAULT joint pose at spawn (required on "
                      "the brake_or_jump family, which has no joint reset)")
  g.add_argument("--gap-full-horizon", action="store_true",
                 help="drop every non-timeout termination so each robot plays "
                      "one continuous take to the horizon")


def _cfg_transform(args):
  return gap_cfg_transform(
    gap_width=args.gap_width, n_gaps=args.n_gaps, episode_s=args.episode_s,
    cmd_vx=args.cmd_vx, spawn_x=tuple(args.spawn_x),
    island_length=args.island_length, full_pose=args.gap_full_pose,
    full_horizon=args.gap_full_horizon)


def _metrics(args, env):
  return GapProgress(env.num_envs, env.device, gap_x=args.gap_x,
                     rest_x=args.rest_x)


def register_preset() -> None:
  register(EvalPreset(
    name="gap_gauntlet",
    description="R-CBF claim gauntlet: a blind walker approaching a pinned gap, "
                "filtered by a safety twin (the gap-gauntlet configuration).",
    task="go2_gap_chain",
    defaults=dict(episode_s=20.0, cmd_vx=1.0),
    add_args=_add_args,
    cfg_transform=_cfg_transform,
    metrics=_metrics))
