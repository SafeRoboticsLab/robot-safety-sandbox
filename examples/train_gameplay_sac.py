"""DEPRECATED thin wrapper around the off-policy (SAC) trainer.

``train_off_policy.py`` is the UNIFIED SAC trainer for all four variants
(Safety / ReachAvoid / Isaacs / Gameplay SAC) and it carries the speedups this
file used to lack -- most importantly the on-device **tensor** leaderboard eval
(``GameplaySAC._eval_pair_tensor``) instead of the slow numpy VecEnv eval. Keeping
two full trainers was confusing and let people accidentally run the slow path.

This shim exists only as a syntax convenience so the old command keeps working:

    python examples/train_gameplay_sac.py --num-envs 1024 --steps 100000000 --seed 0

It forces the two-player go2_stabilize reach-avoid game (GameplaySAC) and forwards
every other flag to ``train_off_policy.py``, which accepts a strict superset of
this file's old arguments. Prefer the routed entrypoint (or the trainer directly):

    python examples/train.py --family off_policy --task <task> --adversary [flags]
    python examples/train_off_policy.py --task <task> --adversary [flags]
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN_SAC = os.path.join(_HERE, "train_off_policy.py")


def main() -> None:
  args = sys.argv[1:]
  # Force the two-player disturbance game; default the task to go2_stabilize (this
  # file's historical hardcode) unless the caller overrides it. --adversary is a
  # store_true, so a duplicate from the caller is harmless.
  inject = ["--adversary"]
  if not any(a == "--task" or a.startswith("--task=") for a in args):
    inject = ["--task", "go2_stabilize", *inject]
  sys.argv = [_TRAIN_SAC, *inject, *args]
  print(f"[train_gameplay_sac] deprecated shim -> {os.path.basename(_TRAIN_SAC)} "
        f"{' '.join(inject)} (forwarding {len(args)} more arg(s))", flush=True)
  runpy.run_path(_TRAIN_SAC, run_name="__main__")


if __name__ == "__main__":
  main()
