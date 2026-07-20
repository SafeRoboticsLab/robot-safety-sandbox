"""Training entrypoint — routes to the on-policy or off-policy trainer.

The two RL families are PEERS, not a main + a variant:

    --family on_policy   -> train_on_policy.py   (PPO family: Safety/ReachAvoid/Isaacs/Gameplay PPO)
    --family off_policy  -> train_off_policy.py  (SAC family: Safety/ReachAvoid/Isaacs/Gameplay SAC)

The family is REQUIRED (there is no default "main" one): pass ``--family`` on the
CLI (aliases: ``ppo`` -> on_policy, ``sac`` -> off_policy) or set ``family:`` in a
``--config`` YAML. Every OTHER flag is forwarded verbatim to the chosen trainer,
which owns its own argparse (run ``--family <f> --help`` to see it). Both
trainers also remain directly runnable (``python examples/train_off_policy.py ...``).

    python examples/train.py --family off_policy --task go2_stabilize --adversary
    python examples/train.py --family ppo        --task go2_gap_chain
    python examples/train.py --config configs/go2_stabilize_gameplaysac.yaml   # family: in the yaml
"""

import argparse
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_FAMILIES = {
  "on_policy": "train_on_policy.py",    # PPO family
  "off_policy": "train_off_policy.py",  # SAC family
}
_ALIASES = {"on": "on_policy", "ppo": "on_policy",
            "off": "off_policy", "sac": "off_policy"}


def _family_from_config(path):
  if not path or not os.path.exists(path):
    return None
  try:
    import yaml
    with open(path) as f:
      return (yaml.safe_load(f) or {}).get("family")
  except Exception:
    return None


def _usage_error(msg):
  sys.stderr.write(
    f"train.py: {msg}\n"
    "  choose the RL family (there is no default):\n"
    "    --family on_policy   (aliases: ppo)  -> PPO family\n"
    "    --family off_policy  (aliases: sac)  -> SAC family\n"
    "  or set 'family:' in your --config YAML.\n"
    "  run  `--family <f> --help`  to see that trainer's options.\n")
  raise SystemExit(2)


def main():
  # Peel --family / --config without consuming anything else (add_help=False so a
  # bare --help falls through to our guidance below, and --family <f> --help
  # forwards to the chosen trainer's help).
  pre = argparse.ArgumentParser(add_help=False)
  pre.add_argument("--family")
  pre.add_argument("--config")
  known, _ = pre.parse_known_args()

  fam = known.family or _family_from_config(known.config)
  fam = _ALIASES.get(fam, fam)

  if fam not in _FAMILIES:
    if fam is None and any(a in ("-h", "--help") for a in sys.argv[1:]):
      _usage_error("no --family given.")     # help without a family: show guidance
    _usage_error(f"unknown or missing --family {known.family!r}."
                 if known.family or known.config else "the --family is required.")

  target = os.path.join(_HERE, _FAMILIES[fam])
  # Forward everything EXCEPT the --family flag (the sub-trainer doesn't define it;
  # --config is kept and the sub-trainer's config loader ignores the 'family' key).
  fwd, skip = [], False
  for a in sys.argv[1:]:
    if skip:
      skip = False
      continue
    if a == "--family":
      skip = True
      continue
    if a.startswith("--family="):
      continue
    fwd.append(a)

  sys.argv = [target, *fwd]
  print(f"[train] family={fam} -> {_FAMILIES[fam]}", flush=True)
  runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
  main()
