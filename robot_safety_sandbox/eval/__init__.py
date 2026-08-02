"""Evaluation harness: one rollout, four independently swappable axes.

    from robot_safety_sandbox.eval import (
      build_eval_env, load_nominal, load_twin, safety_modules, build_filter,
      protocol_metrics, run_eval)

    env   = build_eval_env("go2_locomote", 256, "cuda:0", adversary=True)
    twin, norm = load_twin("runs/.../final_model.zip", "cuda:0")
    mods  = safety_modules(twin, env.num_envs, env.device) | {"norm": norm}
    filt  = build_filter("gameplay", mods, env).filt
    nom   = NominalPolicy(*load_nominal("runs/.../final_model.zip", "cuda:0"))
    run_eval(env, nom, filt, protocol_metrics(env, filt), steps=600, norm=norm)

The four axes are chosen independently and none of them knows about the others:

  ENVIRONMENT  envs.build_eval_env      registry-driven, flat or not, with a
                                       real adversary channel
  NOMINAL      policies.load_nominal    the stock-SB3 pi_task being filtered
  FILTER       filters.build_filter     any of the five compositions
  METRICS      metrics.protocol_metrics the CBF-DDP set, + per-task extras

Task-shaped configuration (terrain surgery, a task's own success reading) is an
:class:`~.presets.EvalPreset` registered by the task that owns it; nothing in
this package is specific to any environment.
"""

from .envs import (
  EvalEnv, StepOut, TwistCommandSurgery, build_eval_env, detect_obs_key)
from .filters import (
  FILTERS, FilterBundle, RolloutCfg, SwitchCfg, build_filter, build_shadow,
  reach_avoid_reduction)
from .metrics import (
  ActuatorJerk, DistanceTravelled, Engagement, EpisodeOutcomes,
  InterventionMass, MarginStats, Metric, MetricSet, StepRecord,
  TrajectoryRecorder, WallClock, protocol_metrics)
from .policies import (
  find_obs_stats, load_nominal, load_twin, safety_modules, twin_class_name)
from .presets import EvalPreset, list_presets, preset, register
from .runner import (
  DSTB_SOURCES, NominalPolicy, TwinNominal, VideoRecorder, ZeroNominal,
  make_dstb, run_eval)

__all__ = [
  "EvalEnv", "StepOut", "TwistCommandSurgery", "build_eval_env",
  "detect_obs_key",
  "FILTERS", "FilterBundle", "SwitchCfg", "RolloutCfg", "build_filter",
  "reach_avoid_reduction",
  "build_shadow",
  "Metric", "MetricSet", "StepRecord", "EpisodeOutcomes", "MarginStats",
  "ActuatorJerk", "InterventionMass", "WallClock", "Engagement",
  "DistanceTravelled", "TrajectoryRecorder",
  "protocol_metrics",
  "load_nominal", "load_twin", "twin_class_name", "safety_modules",
  "find_obs_stats",
  "EvalPreset", "register", "preset", "list_presets",
  "NominalPolicy", "TwinNominal", "ZeroNominal", "VideoRecorder", "run_eval",
  "make_dstb", "DSTB_SOURCES",
]
