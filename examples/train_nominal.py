"""DEPRECATED shim — forwards to ``examples/train.py --family on_policy``.

The ``kind="nominal"`` axis is gone: a dense-reward task policy is now just a
task with ``mode="cumulative"`` (safety_sb3's third backup, ``reward + gamma *
not_done * V'``), so the ONE on-policy trainer handles it — with the same stock
``stable_baselines3.PPO`` recipe, the same numpy bridge + VecNormalize, and the
same plain SB3 checkpoint zip this script used to write.

Existing commands keep working unchanged:

  python examples/train_nominal.py --task go2_walker_flat --num-envs 4096
  ->  python examples/train.py --family on_policy --task go2_walker_flat --num-envs 4096

Use the second form directly; this file will be removed in a later release.
"""

from __future__ import annotations

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
  target = os.path.join(_HERE, "train.py")
  fwd = ["--family", "on_policy", *sys.argv[1:]]
  sys.stderr.write(
    "[DEPRECATED] examples/train_nominal.py is a shim. The nominal/safety split "
    "was replaced by the task's mode= (registry.MODES); dense-reward task "
    "policies are mode='cumulative' and train through the on-policy trainer.\n"
    f"  forwarding to: python examples/train.py {' '.join(fwd)}\n")
  sys.argv = [target, *fwd]
  runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
  main()
