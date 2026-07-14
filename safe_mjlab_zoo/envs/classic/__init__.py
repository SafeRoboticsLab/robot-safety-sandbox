"""Classic-control safety envs (gym/Robust-Gymnasium lineage), mjlab-native.

GPU ports of the Robust-Gymnasium MuJoCo tasks (hopper, half-cheetah,
humanoid, ...) used for ISAACS training in safe_adaptation_dev — same obs,
actions, healthy semantics and disturbance conventions, ~100x the throughput.
hopper_env_cfg.py is the porting template; see its module docstring for the
recipe and the known fidelity deviations.
"""
