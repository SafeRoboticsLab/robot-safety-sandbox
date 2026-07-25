"""Evaluation PRESETS: a named gauntlet is a configuration, not a script.

The four axes an evaluation composes (environment / nominal / filter / metrics)
are independently swappable, so what used to be "the gap gauntlet script" is
just a task plus some cfg surgery plus two extra metrics. A preset bundles
exactly that:

  add_args        the CLI knobs the surgery needs (gap width, spawn range, ...)
  cfg_transform   (args) -> (cfg -> cfg), applied AFTER the task's own builder
  metrics         (args, env) -> Metric, the task's own reading of "did it work"

Presets register the same way tasks do -- from the task module that owns them
(see ``tasks/go2_gap.py``) -- so this registry stays free of any particular
task's geometry. Grep test: nothing in ``robot_safety_sandbox/eval/`` may
mention a terrain feature by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

_PRESETS: dict[str, "EvalPreset"] = {}


@dataclass
class EvalPreset:
  """A named evaluation configuration."""

  name: str
  description: str = ""
  #: default ``--task`` when the preset is selected and no task is given.
  task: Optional[str] = None
  #: argparse defaults this preset overrides (dest -> value).
  defaults: dict = field(default_factory=dict)
  #: ``(parser) -> None``: extra CLI knobs. Use a distinct flag namespace.
  add_args: Optional[Callable] = None
  #: ``(args) -> (cfg -> cfg)`` or None. Applied after the task builds the cfg.
  cfg_transform: Optional[Callable] = None
  #: ``(args, env) -> Metric | None``: the task's own success reading.
  metrics: Optional[Callable] = None


def register(preset: EvalPreset) -> None:
  if preset.name in _PRESETS:
    raise ValueError(f"eval preset '{preset.name}' already registered")
  _PRESETS[preset.name] = preset


def list_presets() -> list[str]:
  return sorted(_PRESETS)


def preset(name: str) -> EvalPreset:
  if name not in _PRESETS:
    raise KeyError(f"unknown eval preset '{name}'. Registered: {list_presets()}")
  return _PRESETS[name]
