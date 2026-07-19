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


# A reserved config key: a dict of ENV/TASK params forwarded to the task's
# cfg_builder (overriding the values baked into its registration). It is NOT
# validated against the argparse flags -- it is a passthrough, so an experiment
# can tune the env from YAML without adding a trainer flag. See make_tensor's
# ``cfg_overrides``. On the CLI, ``--env-override KEY=VAL`` (repeatable) sets
# individual entries, overriding the config's dict per-key.
_ENV_OVERRIDES_KEY = "env_overrides"


def _parse_kv(pairs):
    """Parse ``["k=v", ...]`` into a dict, interpreting each value as YAML
    (so ``0.003`` -> float, ``true`` -> bool, ``foo`` -> str)."""
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"[env-override] expected KEY=VAL, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def merge_config(parser):
    """Parse args with an optional ``--config`` YAML applied as defaults.

    Two-phase so a config can supply even REQUIRED args (e.g. ``--task``): a
    throwaway pre-parser extracts ``--config`` from the command line first, its
    keys are validated + applied as defaults on ``parser``, then ``parser`` does
    the strict parse. Precedence: argparse defaults < config < explicit CLI flags.

    The reserved ``env_overrides:`` config key (a dict) is a PASSTHROUGH — not
    validated against the flags — resolved (merged with any ``--env-override``
    CLI entries, CLI winning per-key) onto ``args.env_overrides``.

    Call this INSTEAD of ``parser.parse_args()``. Returns the args namespace.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    path = pre.parse_known_args()[0].config
    cfg_env_overrides = {}
    if path:
        if not os.path.exists(path):
            raise SystemExit(f"[config] file not found: {path}")
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise SystemExit(
                f"[config] {path} must be a YAML mapping (key: value), got "
                f"{type(cfg).__name__}")
        cfg_env_overrides = cfg.pop(_ENV_OVERRIDES_KEY, {}) or {}   # reserved passthrough
        if not isinstance(cfg_env_overrides, dict):
            raise SystemExit(
                f"[config] '{_ENV_OVERRIDES_KEY}' must be a mapping, got "
                f"{type(cfg_env_overrides).__name__}")
        valid = {a.dest for a in parser._actions if a.dest not in ("help", "config")}
        bad = [k for k in cfg if k not in valid]
        if bad:
            raise SystemExit(
                f"[config] unknown keys in {path}: {sorted(bad)}\n"
                f"  valid keys: {sorted(valid)} (env/task params go under "
                f"'{_ENV_OVERRIDES_KEY}:')")
        parser.set_defaults(**cfg)
        # A `required=True` arg (e.g. --task) ignores a default, so clear the
        # requirement for anything the config supplies.
        for a in parser._actions:
            if a.dest in cfg and getattr(a, "required", False):
                a.required = False
        print(f"[config] loaded {path} ({len(cfg)} keys"
              f"{f' + {len(cfg_env_overrides)} env_overrides' if cfg_env_overrides else ''}"
              f"); CLI flags override it")
    args = parser.parse_args()   # strict parse: config = defaults, CLI overrides
    # Resolve env_overrides: config dict, then CLI --env-override entries (win per-key).
    cli_env = _parse_kv(getattr(args, "env_override", None))
    args.env_overrides = {**cfg_env_overrides, **cli_env}
    return args


def dump_config(outdir, args):
    """Write the fully-resolved run config to ``<outdir>/config.yaml``.

    Drops ``config`` and the raw ``env_override`` CLI list, keeping the resolved
    ``env_overrides`` dict, so the dump round-trips via ``--config``. Reproduce
    the run with ``--config <that file>``."""
    drop = {"config", "env_override"}
    d = {k: v for k, v in vars(args).items() if k not in drop}
    if not d.get("env_overrides"):
        d.pop("env_overrides", None)   # omit an empty dict for tidiness
    path = os.path.join(outdir, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=True, default_flow_style=False)
    print(f"[config] resolved run config -> {path}")
