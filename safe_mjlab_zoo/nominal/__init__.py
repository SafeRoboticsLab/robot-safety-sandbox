"""Nominal TASK policies (kind="nominal") — the policies safety filters wrap.

Everything here is deliberately safety-free: dense env reward, vanilla
stable_baselines3 PPO, no margins (margin_fn=None; envs auto-build in dense
mode). Train with examples/train_nominal.py; compose with a safety twin's
certificate + fallback via examples/eval_filter.py.

A full filter experiment needs BOTH layers of the zoo:
  nominal/  the task policy pi_task (this package)
  tasks/    the safety certificate V(s) + fallback (margins + safety_sb3)
"""
