"""Unit tests for the consolidated evaluation harness (simulator-free).

Everything here runs on CPU with the synthetic twin from ``filter_fixtures``:
obs-group detection, the five filter compositions, the CBF-DDP metrics on
hand-built trajectories with known answers, the adversary strength knob, and
the preset registry. The tests that need a real mjlab env live in
``test_eval_env.py`` (opt-in, they build a simulator).
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from filter_fixtures import ACT_DIM, NUM_ENVS, Fixture, ToyShadowSim  # noqa: E402
from robot_safety_sandbox.eval import (  # noqa: E402
  FILTERS, ActuatorJerk, DistanceTravelled, Engagement, EpisodeOutcomes,
  InterventionMass, MarginStats, MetricSet, RolloutCfg, StepRecord, SwitchCfg,
  TrajectoryRecorder, WallClock,
  reach_avoid_reduction,
  build_filter, detect_obs_key, list_presets, preset, protocol_metrics)
from robot_safety_sandbox.eval.envs import (  # noqa: E402
  NOMINAL_OBS_ORDER, SAFETY_OBS_ORDER, StepOut, TwistCommandSurgery)
from robot_safety_sandbox.eval.runner import (  # noqa: E402
  DSTB_SOURCES, ZeroNominal, make_dstb, run_eval)

DEV = "cpu"


# --- obs-group auto-detection ------------------------------------------------

def test_obs_key_detection_prefers_the_role_specific_group():
  """The gap gauntlet's env emits BOTH groups; each role must pick its own."""
  both = {"proprioception": None, "actor": None, "critic": None}
  assert detect_obs_key(both, SAFETY_OBS_ORDER) == "proprioception"
  assert detect_obs_key(both, NOMINAL_OBS_ORDER) == "actor"


def test_obs_key_detection_on_a_flat_env_gives_actor_to_both():
  """go2_stabilize / go2_locomote have no 'proprioception' group at all --
  the bug that made eval_filter.py KeyError on flat ground."""
  flat = {"actor": None, "critic": None}
  assert detect_obs_key(flat, SAFETY_OBS_ORDER) == "actor"
  assert detect_obs_key(flat, NOMINAL_OBS_ORDER) == "actor"


def test_obs_key_detection_falls_back_to_the_first_group():
  odd = {"weird": None}
  assert detect_obs_key(odd, SAFETY_OBS_ORDER) == "weird"


# --- filter construction, all five kinds -------------------------------------

class _FakeEnv:
  """What build_filter reads: the batch, and the task whose mode fixes the
  rollout reduction (go2_locomote is mode='reach-avoid')."""
  num_envs, device = NUM_ENVS, DEV
  task = "go2_locomote"


def _mods(fx: Fixture, *, q: bool = False, dstb: bool = False) -> dict:
  mods = {"fallback_fn": fx.fallback_fn, "value_fn": fx.value_fn,
          "norm": lambda o: o, "twin": "FakeTwin"}
  if q:
    mods["q_fn"] = fx.q_fn
  if dstb:
    mods["dstb_fn"] = lambda s_obs, **_: torch.zeros(NUM_ENVS, 3)
  return mods


@pytest.mark.parametrize("kind", sorted(FILTERS))
def test_build_filter_assembles_every_composition(kind):
  fx = Fixture(DEV)
  shadow = ToyShadowSim(NUM_ENVS, lambda: torch.zeros(NUM_ENVS),
                        device=DEV) if kind in ("rollout", "gameplay") else None
  bundle = build_filter(kind, _mods(fx, q=True, dstb=True), _FakeEnv(),
                        switch=SwitchCfg(), rollout=RolloutCfg(horizon=3),
                        shadow=shadow)
  action, info = bundle.filt(fx.a_nom[0], speed=fx.speed[0], fresh=fx.fresh[0],
                             s_obs=fx.obs[0])
  assert action.shape == (NUM_ENVS, ACT_DIM)
  assert info.engaged.shape == (NUM_ENVS,)
  # a supplied shadow is not owned, so close() must not touch it
  bundle.close()


@pytest.mark.parametrize("kind,missing", [("critic", "q_fn"), ("qcbf", "q_fn"),
                                          ("gameplay", "dstb_fn")])
def test_build_filter_names_the_capability_the_twin_lacks(kind, missing):
  fx = Fixture(DEV)
  with pytest.raises(SystemExit, match=missing):
    build_filter(kind, _mods(fx), _FakeEnv(), rollout=RolloutCfg(horizon=2),
                 shadow=ToyShadowSim(NUM_ENVS, lambda: torch.zeros(NUM_ENVS),
                                     device=DEV))


@pytest.mark.parametrize("kind", ["rollout", "gameplay"])
def test_rollout_reduction_is_derived_from_the_task_mode(kind):
  """The min_t-g defect: a mode='reach-avoid' task certified with min_t g.

  The reduction must follow the task's mode, so a reach-avoid task gets
  max_t min(l, min_s<=t g) without anyone asking for it.
  """
  fx = Fixture(DEV)
  bundle = build_filter(kind, _mods(fx, q=True, dstb=True), _FakeEnv(),
                        rollout=RolloutCfg(horizon=3),
                        shadow=ToyShadowSim(NUM_ENVS,
                                            lambda: torch.zeros(NUM_ENVS),
                                            device=DEV))
  assert bundle.filt.monitor.reach_avoid is True


def test_reach_avoid_reduction_derives_both_modes():
  assert reach_avoid_reduction("go2_locomote") is True       # mode=reach-avoid
  assert reach_avoid_reduction("go2_locomote", True) is True  # may only confirm


@pytest.mark.parametrize("kind", ["rollout", "gameplay"])
def test_avoid_reduction_on_a_reach_avoid_task_is_an_error(kind):
  """Not a warning: it silently downgrades what the certificate proves."""
  fx = Fixture(DEV)
  with pytest.raises(SystemExit, match="contradicts task"):
    build_filter(kind, _mods(fx, q=True, dstb=True), _FakeEnv(),
                 rollout=RolloutCfg(horizon=3, reach_avoid=False),
                 shadow=ToyShadowSim(NUM_ENVS, lambda: torch.zeros(NUM_ENVS),
                                     device=DEV))


def test_build_filter_rejects_an_unknown_kind():
  with pytest.raises(ValueError, match="unknown --filter"):
    build_filter("nope", _mods(Fixture(DEV)), _FakeEnv())


def test_the_five_kinds_are_exactly_the_documented_table():
  assert set(FILTERS) == {"value", "critic", "qcbf", "rollout", "gameplay"}


# --- metrics on synthetic trajectories with known answers --------------------

def _rec(t, *, g, l, term=None, trunc=None, a_nom=None, a_filt=None,
         a_dstb=None, engaged=None, env=None, filter_s=0.0, env_s=0.0):
  n = g.shape[0]
  z = torch.zeros(n, dtype=torch.bool)
  info = type("I", (), {"engaged": z if engaged is None else engaged,
                        "value": torch.zeros(n), "caution": None})()
  out = StepOut(g=g, l=l, terminated=z if term is None else term,
                truncated=z if trunc is None else trunc)
  return StepRecord(t=t, env=env, a_nom=torch.zeros(n, 2) if a_nom is None
                    else a_nom,
                    a_filt=torch.zeros(n, 2) if a_filt is None else a_filt,
                    a_dstb=a_dstb,
                    info=info, out=out, filter_s=filter_s, env_s=env_s)


def test_episode_outcomes_counts_success_safety_and_censoring():
  """3 envs, 4 steps. env0 reaches the target then times out; env1 violates
  and terminates; env2 never finishes (censored, safe, never reached)."""
  m = EpisodeOutcomes(3, DEV)
  T = torch.tensor
  m.update(_rec(0, g=T([1.0, 1.0, 1.0]), l=T([-1.0, -1.0, -1.0])))
  m.update(_rec(1, g=T([1.0, -1.0, 1.0]), l=T([1.0, -1.0, -1.0])))
  m.update(_rec(2, g=T([1.0, -1.0, 1.0]), l=T([-1.0, -1.0, -1.0]),
                term=T([False, True, False]),
                trunc=T([True, False, False])))
  m.update(_rec(3, g=T([1.0, 1.0, 1.0]), l=T([-1.0, -1.0, -1.0])))
  r = m.result()
  assert r["episodes"] == 2 and r["censored_alive"] == 3
  # env0's completed episode reached the target; nothing else did
  assert r["task_success"] == pytest.approx(1 / 5)
  assert r["violation_rate"] == pytest.approx(1 / 5)
  assert r["safe_rate"] == pytest.approx(4 / 5)
  assert r["timeout_rate"] == pytest.approx(1 / 5)
  assert r["termination_rate"] == pytest.approx(1 / 5)
  assert r["ep_len_mean"] == pytest.approx(3.0)


def test_episode_outcomes_clears_flags_on_reset():
  """A violation in one episode must not condemn the NEXT one (the stale-latch
  bug class, in metric form)."""
  m = EpisodeOutcomes(1, DEV)
  T = torch.tensor
  m.update(_rec(0, g=T([-1.0]), l=T([-1.0]), term=T([True])))
  m.update(_rec(1, g=T([1.0]), l=T([1.0]), trunc=T([True])))
  r = m.result()
  assert r["episodes"] == 2 and r["censored_alive"] == 0
  assert r["violation_rate"] == pytest.approx(0.5)
  assert r["task_success"] == pytest.approx(0.5)


def test_margin_stats_takes_the_worst_g_per_episode_and_keeps_survivors():
  """2 envs. env0 dips to -0.5 then its episode ends and a new one starts clean;
  env1 never finishes, so its running worst must still be counted."""
  m = MarginStats(2, DEV)
  T = torch.tensor
  m.update(_rec(0, g=T([0.5, 0.9]), l=T([0.0, 0.0])))
  m.update(_rec(1, g=T([-0.5, 0.8]), l=T([0.0, 0.0]),
                term=T([True, False])))
  m.update(_rec(2, g=T([1.0, 0.7]), l=T([0.0, 0.0])))
  r = m.result()
  # samples: env0's closed episode (-0.5), env0's live one (1.0), env1 (0.7)
  assert r["min_g_worst"] == pytest.approx(-0.5)
  assert r["min_g_mean"] == pytest.approx((-0.5 + 1.0 + 0.7) / 3)


def test_intervention_mass_is_the_l1_gap_to_the_nominal():
  m = InterventionMass()
  a_nom = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
  a_filt = torch.tensor([[0.5, -0.5], [1.0, 1.0]])
  m.update(_rec(0, g=torch.ones(2), l=torch.ones(2), a_nom=a_nom,
                a_filt=a_filt, engaged=torch.tensor([True, False])))
  r = m.result()
  assert r["intervention_mass_total"] == pytest.approx(1.0)
  assert r["intervention_mass_per_step"] == pytest.approx(0.5)
  # only env0 was engaged, and it carries the whole mass
  assert r["intervention_mass_when_engaged"] == pytest.approx(1.0)


def test_intervention_mass_is_zero_for_a_passthrough_filter():
  m = InterventionMass()
  a = torch.randn(4, 3)
  m.update(_rec(0, g=torch.ones(4), l=torch.ones(4), a_nom=a, a_filt=a.clone()))
  assert m.result()["intervention_mass_total"] == pytest.approx(0.0)


class _JerkEnv:
  """An env whose joint accelerations follow a scripted sequence."""

  def __init__(self, accs):
    self.accs, self.i = accs, 0

    class _Data:
      pass

    class _Robot:
      pass

    self._robot, self._robot_data = _Robot(), _Data()
    self._robot.data = self._robot_data

  @property
  def robot(self):
    self._robot_data.joint_acc = self.accs[self.i]
    return self._robot


def test_actuator_jerk_is_the_acceleration_difference_over_dt():
  """2 envs x 2 actuators, dt=0.5, accelerations 0 -> [1,2] -> [1,4].
  Jerks: step1 [2,4] (both envs), step2 [0,4]. Per-actuator mean = [1,4]."""
  accs = [torch.zeros(2, 2), torch.tensor([[1.0, 2.0]] * 2),
          torch.tensor([[1.0, 4.0]] * 2)]
  env = _JerkEnv(accs)
  m = ActuatorJerk(dt=0.5)
  for t in range(3):
    env.i = t
    m.update(_rec(t, g=torch.ones(2), l=torch.ones(2), env=env))
  r = m.result()
  assert r["jerk_per_actuator_mean"] == pytest.approx([1.0, 4.0])
  assert r["jerk_mean"] == pytest.approx(2.5)
  assert r["jerk_max_actuator"] == pytest.approx(4.0)
  # actuator 0 saw {2, 0}, actuator 1 saw {4, 4}: std 1.0 and 0.0
  assert r["jerk_per_actuator_std"] == pytest.approx([1.0, 0.0], abs=1e-4)


def test_actuator_jerk_skips_the_step_after_a_reset():
  """A reset teleports the state; the acceleration jump across it is not a
  control action and must not be charged to the filter."""
  accs = [torch.zeros(1, 1), torch.tensor([[100.0]]), torch.tensor([[100.0]])]
  env = _JerkEnv(accs)
  m = ActuatorJerk(dt=1.0)
  env.i = 0
  m.update(_rec(0, g=torch.ones(1), l=torch.ones(1), env=env,
                term=torch.tensor([True])))       # env resets after this step
  env.i = 1
  m.update(_rec(1, g=torch.ones(1), l=torch.ones(1), env=env))  # skipped
  env.i = 2
  m.update(_rec(2, g=torch.ones(1), l=torch.ones(1), env=env))  # counted, 0
  assert m.result()["jerk_mean"] == pytest.approx(0.0)


# --- distance + trajectory recording -----------------------------------------

class _TrajEnv:
  """An env whose robot state follows a scripted sequence of root positions.

  Mirrors the measured mjlab behaviour: the position read at a DONE step is
  already the next episode's spawn (the auto-reset happens inside step).
  """

  def __init__(self, positions, n=1, ctrl=2, joints=3, dstb=3, yaw=0.0):
    self.positions, self.i = positions, 0
    self.num_envs, self.ctrl_dim, self.dstb_dim = n, ctrl, dstb
    self.device = DEV
    self._data = type("D", (), {})()
    self._robot = type("R", (), {})()
    self._robot.data = self._data
    self._j = joints
    # mujoco order (w, x, y, z); a pure yaw about z
    self._q = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0,
                             math.sin(yaw / 2)]])

  @property
  def robot(self):
    p = self.positions[self.i]
    n, j = self.num_envs, self._j
    self._data.root_link_pos_w = p
    self._data.root_link_quat_w = self._q.expand(n, 4).contiguous()
    self._data.root_link_lin_vel_w = torch.zeros(n, 3) + float(self.i)
    self._data.root_link_ang_vel_w = torch.zeros(n, 3)
    self._data.joint_pos = torch.zeros(n, j) + float(self.i)
    self._data.joint_vel = torch.zeros(n, j)
    self._data.joint_acc = torch.zeros(n, j)
    return self._robot


#: one env: walks +1 m in x, resets (teleport to 100, 100), walks +1 m twice.
_TRAJ_XY = [torch.tensor([[1.0, 0.0, 0.3]]), torch.tensor([[2.0, 0.0, 0.3]]),
            torch.tensor([[100.0, 100.0, 0.3]]),
            torch.tensor([[101.0, 100.0, 0.3]]),
            torch.tensor([[102.0, 100.0, 0.3]])]
_TRAJ_DONE = [False, False, True, False, False]


def _drive(metric, env, *, g=None):
  """Play the scripted stream through a metric, the way run_eval would."""
  for t, dn in enumerate(_TRAJ_DONE):
    env.i = t
    gv = torch.tensor([1.0]) if g is None else g[t]
    metric.update(_rec(t, g=gv, l=-torch.ones(1), env=env,
                       trunc=torch.tensor([dn]),
                       a_nom=torch.full((1, env.ctrl_dim), 0.1 * t),
                       a_filt=torch.full((1, env.ctrl_dim), 0.2 * t),
                       a_dstb=torch.full((1, env.dstb_dim), 0.3 * t)))
  return metric.result()


def test_distance_travelled_drops_the_reset_teleport():
  """Episode 1 walks 1 m; the 140 m teleport into the new spawn is not travel."""
  env = _TrajEnv(_TRAJ_XY)
  r = _drive(DistanceTravelled(1, DEV), env)
  assert r["dist_episodes"] == 1
  assert r["dist_path_mean"] == pytest.approx(1.0)
  assert r["dist_net_mean"] == pytest.approx(1.0)
  assert r["dist_censored"] == 1          # the second episode never finished


def test_distance_separates_path_length_from_net_displacement():
  """A robot that trots in place covers path but goes nowhere."""
  there_and_back = [torch.tensor([[1.0, 0.0, 0.3]]),
                    torch.tensor([[2.0, 0.0, 0.3]]),
                    torch.tensor([[1.0, 0.0, 0.3]]),
                    torch.tensor([[9.0, 9.0, 0.3]]),
                    torch.tensor([[9.0, 9.0, 0.3]])]
  env = _TrajEnv(there_and_back)
  m = DistanceTravelled(1, DEV)
  for t, dn in enumerate([False, False, False, True, False]):
    env.i = t
    m.update(_rec(t, g=torch.ones(1), l=-torch.ones(1), env=env,
                  trunc=torch.tensor([dn])))
  r = m.result()
  # travelled 1 out, 1 back = 2; net displacement from the (1,0) start = 0
  assert r["dist_path_mean"] == pytest.approx(2.0)
  assert r["dist_net_mean"] == pytest.approx(0.0)


def _record(tmp_path, *, traj_envs=1, g=None, env=None, frame="spawn"):
  env = env or _TrajEnv(_TRAJ_XY)
  rec = TrajectoryRecorder(env, str(tmp_path), traj_envs=traj_envs, frame=frame)
  return rec, _drive(rec, env, g=g)


def _pkl_trajs(tmp_path):
  import pickle
  with open(tmp_path / "trajectories.pkl", "rb") as f:
    return pickle.load(f)["trajectories"]


def test_trajectory_recorder_does_not_concatenate_across_an_episode_boundary(
    tmp_path):
  """THE bug this recorder has to not have: mjlab auto-resets inside step, so a
  naive buffer splices the old episode onto the new spawn."""
  _rec_m, summary = _record(tmp_path, frame="world")
  assert summary["traj_episodes"] == 2
  trajs = _pkl_trajs(tmp_path)
  assert len(trajs) == 2
  # episode 1 is the pre-reset walk only; the spawn at (100, 100) is NOT in it
  assert trajs[0][:, 0:2].tolist() == [[1.0, 0.0], [2.0, 0.0]]
  # episode 2 STARTS at the fresh spawn -- the state read on the done step
  assert trajs[1][:, 0:2].tolist() == [[100.0, 100.0], [101.0, 100.0]]
  assert all(t.shape[1] == 3 for t in trajs)      # x, y, z


def test_trajectory_recorder_spawn_frame_puts_every_episode_at_the_origin(
    tmp_path):
  """The coverage plot draws its star at (0, 0) and overlays experiments, but
  mjlab spreads envs over the terrain grid -- so the default frame is each
  episode's own spawn."""
  _record(tmp_path)                                  # frame="spawn" default
  trajs = _pkl_trajs(tmp_path)
  assert trajs[0][:, 0:2].tolist() == [[0.0, 0.0], [1.0, 0.0]]
  assert trajs[1][:, 0:2].tolist() == [[0.0, 0.0], [1.0, 0.0]]
  # z is left in world units -- height above the floor means what it says
  assert trajs[0][:, 2].tolist() == pytest.approx([0.3, 0.3])


def test_trajectory_recorder_spawn_frame_rotates_heading_onto_plus_x(tmp_path):
  """Spawn yaw is uniformly random on go2_locomote (measured std 92 deg); the
  fan only concentrates if the initial heading is rotated onto +x."""
  north = [torch.tensor([[0.0, 1.0, 0.3]]), torch.tensor([[0.0, 2.0, 0.3]]),
           torch.tensor([[50.0, 50.0, 0.3]]),
           torch.tensor([[50.0, 51.0, 0.3]]), torch.tensor([[50.0, 52.0, 0.3]])]
  _record(tmp_path, env=_TrajEnv(north, yaw=math.pi / 2))
  trajs = _pkl_trajs(tmp_path)
  # walked +y in world while facing +y: that is straight FORWARD -> +x
  for t in trajs:
    assert t[:, 0:2].ravel().tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0],
                                                       abs=1e-6)


def test_trajectory_recorder_matches_the_plot_script_contract(tmp_path):
  """plot_trajectory_coverage.py::load_experiment + get_all_xy_points, verbatim.

  If this breaks, the figure the recording exists for cannot be drawn.
  """
  import json
  import os
  import pickle

  import numpy as np
  _record(tmp_path, g=[torch.tensor([v]) for v in
                       (1.0, 1.0, -1.0, 1.0, 1.0)])

  with open(os.path.join(str(tmp_path), 'trajectories.pkl'), 'rb') as f:
    traj_data = pickle.load(f)
  with open(os.path.join(str(tmp_path), 'stats.json')) as f:
    stats = json.load(f)
  trajectories = traj_data['trajectories']
  all_xy = np.vstack([t[:, 0:2] for t in trajectories])
  assert all_xy.shape == (4, 2)
  done_types = stats.get('done_types', ['unknown'] * len(trajectories))
  assert len(done_types) == len(trajectories)
  assert isinstance(stats['avg_distance'], float)
  # episode 1 dipped to g=-1: 'failure' is the only value the script reads
  assert done_types[0] == 'failure' and done_types[1] != 'failure'
  n_success = sum(1 for d in done_types if d != 'failure')
  assert n_success / len(done_types) == pytest.approx(0.5)


def test_trajectory_recorder_captures_state_actions_decision_and_margins(
    tmp_path):
  import numpy as np
  _record(tmp_path)
  z = np.load(tmp_path / "trajectories_full.npz")
  for k in ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel",
            "joint_pos", "joint_vel", "a_nom", "a_filt", "a_dstb", "engaged",
            "value", "caution", "g", "l", "terminated", "truncated", "step",
            "episode_ptr", "episode_env", "episode_done_type"):
    assert k in z, f"missing field {k}"
  assert z["episode_ptr"].tolist() == [0, 2, 4]
  assert z["a_nom"].shape == (4, 2) and z["a_dstb"].shape == (4, 3)
  # row t carries the START-of-step state with the decision made AT step t:
  # first row = state after step 0, actions of step 1
  assert z["a_nom"][0].tolist() == pytest.approx([0.1, 0.1])
  assert z["a_filt"][0].tolist() == pytest.approx([0.2, 0.2])
  assert z["a_dstb"][0].tolist() == pytest.approx([0.3, 0.3, 0.3])
  assert z["step"].tolist() == [1.0, 2.0, 3.0, 4.0]
  assert z["truncated"].tolist() == [0.0, 1.0, 0.0, 0.0]
  assert z["root_pos"][0].tolist() == pytest.approx([1.0, 0.0, 0.3])
  assert z["caution"].tolist() == [0.0] * 4       # qcbf reports no band


def test_trajectory_recorder_records_only_the_requested_subset(tmp_path):
  """512 x 1000 x every field is too big to keep; the other metrics do not
  shrink with it."""
  import numpy as np
  pos = [torch.zeros(8, 3) + float(t) for t in range(5)]
  env = _TrajEnv(pos, n=8)
  rec = TrajectoryRecorder(env, str(tmp_path), traj_envs=3)
  for t, dn in enumerate(_TRAJ_DONE):
    env.i = t
    m = torch.zeros(8, dtype=torch.bool) | dn
    rec.update(_rec(t, g=torch.ones(8), l=-torch.ones(8), env=env, trunc=m,
                    a_nom=torch.zeros(8, 2), a_filt=torch.zeros(8, 2)))
  summary = rec.result()
  assert summary["traj_envs_recorded"] == 3
  assert summary["traj_episodes"] == 6            # 3 envs x 2 episodes
  z = np.load(tmp_path / "trajectories_full.npz")
  assert sorted(set(z["episode_env"].tolist())) == [0, 1, 2]


def test_trajectory_recorder_does_not_influence_the_rollout():
  """The Metric contract. Same seed, same stream, with and without recording."""
  import tempfile

  def actions(with_recorder):
    env = _ToyEvalEnv()
    fx = Fixture(DEV)
    filt = build_filter("value", _mods(fx), _FakeEnv()).filt
    m = MetricSet(InterventionMass())
    if with_recorder:
      env.robot.data.root_link_pos_w = torch.zeros(NUM_ENVS, 3)
      env.robot.data.root_link_quat_w = torch.zeros(NUM_ENVS, 4)
      env.robot.data.root_link_ang_vel_w = torch.zeros(NUM_ENVS, 3)
      env.robot.data.joint_pos = torch.zeros(NUM_ENVS, ACT_DIM)
      env.robot.data.joint_vel = torch.zeros(NUM_ENVS, ACT_DIM)
      m = MetricSet(m, TrajectoryRecorder(env, tempfile.mkdtemp()))
    run_eval(env, ZeroNominal(NUM_ENVS, ACT_DIM, DEV), filt, m, steps=5,
             norm=lambda o: o, dstb_fn=lambda s: torch.full((NUM_ENVS, 3), 0.5),
             dstb_scale=1.0)
    return env.seen_dstb

  a, b = actions(False), actions(True)
  assert len(a) == len(b)
  assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_wall_clock_averages_per_step():
  m = WallClock()
  m.update(_rec(0, g=torch.ones(1), l=torch.ones(1), filter_s=0.01, env_s=0.03))
  m.update(_rec(1, g=torch.ones(1), l=torch.ones(1), filter_s=0.03, env_s=0.05))
  r = m.result()
  assert r["filter_ms_per_step"] == pytest.approx(20.0)
  assert r["env_ms_per_step"] == pytest.approx(40.0)
  assert r["total_ms_per_step"] == pytest.approx(60.0)


def test_metric_set_merges_every_report():
  m = MetricSet(InterventionMass(), WallClock())
  m.update(_rec(0, g=torch.ones(1), l=torch.ones(1)))
  keys = m.result()
  assert "intervention_mass_total" in keys and "total_ms_per_step" in keys


def test_engagement_reads_the_filters_own_telemetry():
  fx = Fixture(DEV)
  filt = build_filter("value", _mods(fx), _FakeEnv()).filt
  m = Engagement(filt)
  for t in range(5):
    _a, info = filt(fx.a_nom[t], speed=fx.speed[t], fresh=fx.fresh[t],
                    s_obs=fx.obs[t])
    m.update(_rec(t, g=torch.ones(NUM_ENVS), l=torch.ones(NUM_ENVS)))
  r = m.result()
  assert 0.0 <= r["intervention_rate"] <= 1.0
  assert r["intervention_rate"] == pytest.approx(
    float(filt.telemetry.engaged_steps.sum() / (NUM_ENVS * 5)))


# --- the adversary channel ---------------------------------------------------

class _FakeBridge:
  def __init__(self, dstb_mode):
    self.dstb_mode = dstb_mode
    self.force_scale = None


class _ScaleEnv:
  """Just enough EvalEnv surface for the strength knob."""
  from robot_safety_sandbox.eval.envs import EvalEnv as _E
  set_dstb_scale = _E.set_dstb_scale
  scale_dstb_action = _E.scale_dstb_action

  def __init__(self, dstb_mode):
    self.bridge = _FakeBridge(dstb_mode)
    self.num_envs, self.device, self.dstb_dim = 4, DEV, 3
    self._dstb_scale = 1.0


def test_dstb_scale_wrench_goes_through_force_scale_not_the_action():
  """A wrench disturbance is unit-normalized before scaling, so scaling the
  ACTION would be a silent no-op; the knob must reach force_scale."""
  env = _ScaleEnv("wrench")
  a = torch.ones(4, 3)
  env.set_dstb_scale(0.25)
  assert torch.allclose(env.bridge.force_scale, torch.full((4,), 0.25))
  assert torch.equal(env.scale_dstb_action(a), a)


def test_dstb_scale_action_mode_scales_the_action():
  env = _ScaleEnv("action")
  a = torch.ones(4, 3)
  env.set_dstb_scale(0.25)
  assert env.bridge.force_scale is None
  assert torch.allclose(env.scale_dstb_action(a), a * 0.25)


def test_make_dstb_sources_and_shapes():
  env = _ScaleEnv("wrench")
  assert set(DSTB_SOURCES) == {"none", "random", "policy"}
  z = make_dstb("none", env)(None)
  assert z.shape == (4, 3) and torch.all(z == 0)
  r = make_dstb("random", env)(None)
  assert r.shape == (4, 3) and float(r.abs().max()) <= 1.0
  mods = {"dstb_fn": lambda s_obs, **_: torch.full((4, 3), 0.5),
          "twin": "FakeTwin"}
  assert torch.allclose(make_dstb("policy", env, mods)(None),
                        torch.full((4, 3), 0.5))


def test_make_dstb_policy_without_a_two_player_twin_is_a_named_error():
  with pytest.raises(SystemExit, match="no disturbance actor"):
    make_dstb("policy", _ScaleEnv("wrench"), {"twin": "ReachAvoidPPO1P"})


def test_make_dstb_rejects_an_unknown_source():
  with pytest.raises(ValueError, match="unknown --dstb"):
    make_dstb("magic", _ScaleEnv("wrench"))


# --- command surgery ---------------------------------------------------------

class _CmdEnv:
  def __init__(self, terms, n=3):
    cmd = torch.full((n, 3), 9.0)
    mgr = type("M", (), {"active_terms": terms,
                         "get_command": lambda self, name: cmd})()
    self.mj = type("E", (), {"command_manager": mgr})()
    self.cmd = cmd


def test_twist_command_surgery_writes_engaged_caution_and_free_values():
  env = _CmdEnv(["twist"])
  s = TwistCommandSurgery(cmd_vx=0.8, engaged_cmd_vx=1.0).bind(env)
  assert s.available
  s(env, engaged=torch.tensor([True, False, False]),
    caution=torch.tensor([False, True, False]))
  assert torch.allclose(env.cmd[:, 0], torch.tensor([1.0, 0.0, 0.8]))


def test_twist_command_surgery_is_a_no_op_without_the_command():
  env = _CmdEnv([])
  s = TwistCommandSurgery().bind(env)
  assert not s.available
  s(env, engaged=torch.ones(3, dtype=torch.bool),
    caution=torch.zeros(3, dtype=torch.bool))
  assert torch.allclose(env.cmd, torch.full((3, 3), 9.0))


# --- presets -----------------------------------------------------------------

def test_gap_gauntlet_preset_is_registered_with_its_hooks():
  import robot_safety_sandbox  # noqa: F401  (registration happens on import)
  assert "gap_gauntlet" in list_presets()
  ps = preset("gap_gauntlet")
  assert ps.task == "go2_gap_chain"
  assert ps.add_args and ps.cfg_transform and ps.metrics


def test_unknown_preset_lists_the_registered_ones():
  import robot_safety_sandbox  # noqa: F401
  with pytest.raises(KeyError, match="gap_gauntlet"):
    preset("no_such_preset")


def test_no_task_geometry_leaks_into_the_eval_core():
  """The core must stay task-agnostic: task-shaped knowledge belongs in a
  preset registered by the task that owns it."""
  import io
  import tokenize

  import robot_safety_sandbox.eval as E
  core = os.path.dirname(os.path.abspath(E.__file__))
  banned = ("gap", "terrain", "island", "crossing", "livelock", "crawl",
            "tunnel", "walker")
  for fname in sorted(os.listdir(core)):
    if not fname.endswith(".py"):
      continue
    src = open(os.path.join(core, fname)).read()
    # CODE only: docstrings and comments may name a preset as an example, so
    # drop every STRING and COMMENT token before looking.
    code = " ".join(
      tok.string.lower() for tok in
      tokenize.generate_tokens(io.StringIO(src).readline)
      if tok.type not in (tokenize.STRING, tokenize.COMMENT))
    for word in banned:
      assert word not in code, (
        f"{fname} mentions '{word}' in CODE -- task geometry does not belong "
        "in the eval core (register an EvalPreset instead)")


# --- the runner --------------------------------------------------------------

class _ToyEvalEnv:
  """A simulator-free EvalEnv: a scripted margin stream and a null robot."""

  def __init__(self, n=NUM_ENVS, ctrl=ACT_DIM, obs_dim=12, steps=8):
    self.num_envs, self.ctrl_dim, self.dstb_dim = n, ctrl, 3
    self.device, self.task, self.adversary = DEV, "toy", True
    self.t = 0
    self.steps = steps
    self._obs = torch.zeros(n, obs_dim)
    self.seen_dstb = []
    self.bridge = _FakeBridge("action")
    root = torch.zeros(n, 3)
    self.robot = type("R", (), {"data": type("D", (), {
      "root_link_lin_vel_w": root, "joint_acc": torch.zeros(n, ctrl)})()})()

  def set_dstb_scale(self, s):
    self._scale = s

  def reset(self, seed=None):
    self.t = 0

  def nominal_obs(self):
    return self._obs

  def safety_obs(self):
    return self._obs

  def step(self, a_ctrl, a_dstb=None):
    self.seen_dstb.append(None if a_dstb is None else a_dstb.clone())
    self.t += 1
    n = self.num_envs
    z = torch.zeros(n, dtype=torch.bool)
    return StepOut(g=torch.ones(n), l=-torch.ones(n), terminated=z, truncated=z)


def test_run_eval_delivers_the_disturbance_every_step():
  env = _ToyEvalEnv()
  fx = Fixture(DEV)
  filt = build_filter("value", _mods(fx), _FakeEnv()).filt
  d = torch.full((NUM_ENVS, 3), 0.5)
  out = run_eval(env, ZeroNominal(NUM_ENVS, ACT_DIM, DEV), filt,
                 MetricSet(InterventionMass()), steps=5, norm=lambda o: o,
                 dstb_fn=lambda s: d, dstb_scale=0.5)
  assert len(env.seen_dstb) == 5
  assert all(torch.equal(x, d) for x in env.seen_dstb)
  assert "intervention_mass_total" in out


def test_run_eval_without_an_adversary_passes_no_disturbance():
  env = _ToyEvalEnv()
  env.adversary = False
  fx = Fixture(DEV)
  filt = build_filter("value", _mods(fx), _FakeEnv()).filt
  run_eval(env, ZeroNominal(NUM_ENVS, ACT_DIM, DEV), filt, MetricSet(),
           steps=3, norm=lambda o: o)
  assert env.seen_dstb == [None, None, None]


def test_run_eval_refuses_an_attack_on_an_env_without_the_channel():
  env = _ToyEvalEnv()
  env.adversary = False
  fx = Fixture(DEV)
  filt = build_filter("value", _mods(fx), _FakeEnv()).filt
  with pytest.raises(ValueError, match="no adversary channel"):
    run_eval(env, ZeroNominal(NUM_ENVS, ACT_DIM, DEV), filt, MetricSet(),
             steps=1, norm=lambda o: o, dstb_fn=lambda s: torch.zeros(1))


def test_no_filter_control_arm_applies_the_nominal_and_reports_zero_engagement():
  env = _ToyEvalEnv()
  fx = Fixture(DEV)
  filt = build_filter("value", _mods(fx), _FakeEnv()).filt
  out = run_eval(env, ZeroNominal(NUM_ENVS, ACT_DIM, DEV), filt,
                 MetricSet(InterventionMass(), Engagement(filt)), steps=6,
                 norm=lambda o: o, no_filter=True)
  assert out["intervention_mass_total"] == pytest.approx(0.0)
  assert out["intervention_rate"] == pytest.approx(0.0)
