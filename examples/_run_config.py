"""Shared ``--config`` plumbing for the trainers (train.py / train_sac.py).

A YAML config file sets argparse DEFAULTS, so precedence is:

    argparse defaults  <  --config file  <  explicit CLI flags

i.e. a config is a reusable recipe you can still override one knob at a time on
the command line. Config keys must be argparse ``dest`` names (e.g. ``num_envs``,
``gamma_schedule``) — argparse stays the single source of truth for the schema.

Every run also DUMPS its fully-resolved config to ``<outdir>/config.yaml``, so any
run is exactly reproducible: re-run with ``--config <that file>``.
"""

from __future__ import annotations

import argparse
import os

import yaml


def merge_config(parser):
    """Parse args with an optional ``--config`` YAML applied as defaults.

    Two-phase so a config can supply even REQUIRED args (e.g. ``--task``): a
    throwaway pre-parser extracts ``--config`` from the command line first, its
    keys are validated + applied as defaults on ``parser``, then ``parser`` does
    the strict parse. Precedence: argparse defaults < config < explicit CLI flags.

    Call this INSTEAD of ``parser.parse_args()``. Returns the args namespace.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    path = pre.parse_known_args()[0].config
    if path:
        if not os.path.exists(path):
            raise SystemExit(f"[config] file not found: {path}")
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise SystemExit(
                f"[config] {path} must be a YAML mapping (key: value), got "
                f"{type(cfg).__name__}")
        valid = {a.dest for a in parser._actions if a.dest not in ("help", "config")}
        bad = [k for k in cfg if k not in valid]
        if bad:
            raise SystemExit(
                f"[config] unknown keys in {path}: {sorted(bad)}\n"
                f"  valid keys: {sorted(valid)}")
        parser.set_defaults(**cfg)
        # A `required=True` arg (e.g. --task) ignores a default, so clear the
        # requirement for anything the config supplies.
        for a in parser._actions:
            if a.dest in cfg and getattr(a, "required", False):
                a.required = False
        print(f"[config] loaded {path} ({len(cfg)} keys); CLI flags override it")
    return parser.parse_args()   # strict parse: config = defaults, CLI overrides


def dump_config(outdir, args):
    """Write the fully-resolved run config to ``<outdir>/config.yaml`` (drop the
    ``config`` key itself). Reproduce the run with ``--config <that file>``."""
    d = {k: v for k, v in vars(args).items() if k != "config"}
    path = os.path.join(outdir, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=True, default_flow_style=False)
    print(f"[config] resolved run config -> {path}")
