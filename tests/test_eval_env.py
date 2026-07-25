"""Evaluation-harness tests that need a REAL mjlab env (opt-in).

Building a simulator costs tens of seconds per env and a GPU, so these are
skipped unless ``ZOO_SIM_TESTS=1``. They cover exactly the claims a
simulator-free test cannot make:

  * ``build_eval_env`` works on a FLAT task (no terrain generator) and on a
    GAP task through the preset's cfg surgery -- the flat case is what the old
    gap-only harness could not do;
  * obs-group auto-detection picks the right group per role on the real cfgs;
  * a disturbance is actually DELIVERED, and the strength knob actually changes
    the physics.

    ZOO_SIM_TESTS=1 pytest tests/test_eval_env.py -q
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

pytestmark = pytest.mark.skipif(
  os.environ.get("ZOO_SIM_TESTS") != "1",
  reason="builds a real mjlab simulator; set ZOO_SIM_TESTS=1 to run")

from robot_safety_sandbox.eval import build_eval_env, preset  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
N = 4


def _close(env):
  try:
    env.close()
  except Exception:
    pass


def test_flat_task_builds_and_detects_the_actor_group():
  """go2_locomote has NO 'proprioception' group and NO terrain generator --
  both fatal for the old gap-only harness."""
  env = build_eval_env("go2_locomote", N, DEV, adversary=True)
  try:
    assert env.mj.cfg.scene.terrain.terrain_generator is None
    assert env.safety_obs_key == "actor"
    assert env.nominal_obs_key == "actor"
    env.reset(seed=0)
    out = env.step(torch.zeros(N, env.ctrl_dim, device=DEV),
                   torch.zeros(N, env.dstb_dim, device=DEV))
    assert out.g.shape == (N,) and out.l.shape == (N,)
  finally:
    _close(env)


def test_gap_task_builds_through_the_preset_and_emits_both_groups():
  ps = preset("gap_gauntlet")

  class Args:
    gap_width, n_gaps, episode_s, cmd_vx = 0.35, 1, 20.0, 1.0
    spawn_x, island_length = (0.15, 0.45), None
    gap_full_pose = gap_full_horizon = False

  env = build_eval_env("go2_gap_chain", N, DEV,
                       cfg_transform=ps.cfg_transform(Args()))
  try:
    assert env.safety_obs_key == "proprioception"
    assert env.nominal_obs_key == "actor"          # grafted blind walker group
    assert env.nominal_obs().shape[1] != env.safety_obs().shape[1]
    env.reset(seed=0)
    env.step(torch.zeros(N, env.ctrl_dim, device=DEV))
  finally:
    _close(env)


def test_adversary_is_refused_when_the_task_declares_none():
  with pytest.raises(ValueError, match="does not define an adversary"):
    build_eval_env("go2_gap_chain", N, DEV, adversary=True)


def test_cumulative_task_is_refused_as_an_eval_env():
  with pytest.raises(ValueError, match="cumulative"):
    build_eval_env("go2_walker_flat", N, DEV)


def test_env_overrides_reach_the_task_cfg_builder():
  env = build_eval_env("go2_locomote", N, DEV, env_overrides={"cmd_vx": 0.3})
  try:
    assert env.mj.cfg.commands["twist"].ranges.lin_vel_x == (0.3, 0.3)
  finally:
    _close(env)


def test_disturbance_strength_changes_the_physics():
  """The same attack direction at two strengths must move the robot
  differently -- the check that --dstb-scale is wired to the sim and not
  silently dropped (a wrench dstb is unit-normalized, so scaling the ACTION
  alone would be a no-op)."""
  d = torch.zeros(N, 3, device=DEV)
  d[:, 0] = 1.0                                   # push forward, hard
  finals = {}
  for scale in (0.0, 1.0):
    env = build_eval_env("go2_locomote", N, DEV, adversary=True)
    try:
      env.set_dstb_scale(scale)
      env.reset(seed=0)
      for _ in range(20):
        env.step(torch.zeros(N, env.ctrl_dim, device=DEV), d)
      finals[scale] = float(env.robot.data.root_link_lin_vel_w[:, 0].mean())
    finally:
      _close(env)
  assert abs(finals[1.0] - finals[0.0]) > 1e-3, finals
  assert finals[1.0] > finals[0.0], finals        # pushed forward, so faster
