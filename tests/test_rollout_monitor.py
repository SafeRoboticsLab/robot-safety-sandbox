"""Rollout / Gameplay monitor: semantics on a toy sim, fidelity on a real one.

Two layers, deliberately separated:

* the SEMANTICS (horizon min, min over parallel rollouts, post-termination
  masking, the reach-avoid reduction, recertification latching) are checked
  against ``ToyShadowSim``, where the right answer is available in closed form
  and a test runs in milliseconds;
* the mjlab SHADOW-SIM path (the state round-trip, margins out of a real task)
  is checked on ``car_goal``, the smallest real env, and skipped without a GPU.
"""

from __future__ import annotations


import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from filter_fixtures import (  # noqa: E402
  ACT_DIM, OBS_DIM, ToyShadowSim, constant_fallback)
from robot_safety_sandbox.filters import (  # noqa: E402
  AdversarialRolloutMonitor, LeastRestrictiveIntervention, PolicyFallback,
  RolloutMonitor, SafetyFilter, rollout_filter)

DEV = "cpu"
H = 10
DT = 0.1


def _live(*xs) -> torch.Tensor:
  return torch.tensor(xs, dtype=torch.float32)


def _monitor(state, *, fallback_speed, rollouts_per_env=1, drift=None,
             horizon=H, reach_avoid=False, recertify_every=1):
  """A RolloutMonitor over a toy sim pinned at ``state`` with a scripted pi^<."""
  n = state.shape[0]
  sim = ToyShadowSim(n, lambda: state, rollouts_per_env=rollouts_per_env,
                     dt=DT, drift=drift)
  mon = RolloutMonitor(n, DEV, sim, horizon, reach_avoid=reach_avoid,
                       recertify_every=recertify_every)
  return mon, sim, PolicyFallback(n, DEV, constant_fallback(fallback_speed),
                                  ACT_DIM)


def _act(n, value):
  return torch.full((n, ACT_DIM), float(value))


# --- the core claim: simulate, don't trust a scalar ---------------------------

def test_doomed_state_is_negative_and_safe_state_is_positive():
  """A fallback that drives INTO the failure set condemns a state that a
  fallback driving away from it certifies — same state, same nominal action,
  same horizon, only pi^< differs. That is the point of a rollout monitor."""
  state = _live(0.5, 0.5)
  doomed, _, fb_down = _monitor(state, fallback_speed=-1.0)
  safe, _, fb_up = _monitor(state, fallback_speed=+1.0)
  a = _act(2, 0.0)                       # the nominal action is neutral

  d, s = doomed(a, fallback=fb_down), safe(a, fallback=fb_up)
  # x0 = 0.5; the nominal step holds, then 9 fallback steps of +-0.1:
  # down -> 0.5 ... -0.4 (min -0.4), up -> 0.5 ... 1.4 (min 0.5).
  assert torch.all(d < 0), d
  assert torch.all(s > 0), s
  assert torch.allclose(d, torch.full((2,), -0.4), atol=1e-5)
  assert torch.allclose(s, torch.full((2,), 0.5), atol=1e-5)


def test_the_nominal_action_is_the_first_step_and_the_fallback_the_rest():
  """H steps total = 1 nominal + (H-1) fallback, and the nominal one counts."""
  state = _live(0.05)
  mon, sim, fb = _monitor(state, fallback_speed=+1.0)
  # A nominal that dives: one step of -1.0 takes x to -0.05 even though the
  # fallback recovers from there. The horizon MIN must still see the dip.
  assert mon(_act(1, -1.0), fallback=fb) < 0
  assert sim.steps == H
  # ... while a neutral nominal at the same state stays certified.
  mon2, _, fb2 = _monitor(state, fallback_speed=+1.0)
  assert mon2(_act(1, 0.0), fallback=fb2) > 0


def test_horizon_is_a_min_not_an_endpoint():
  """A rollout that dips below zero mid-horizon and recovers is NOT certified,
  however healthy it looks at t = H."""
  mon, _, fb = _monitor(_live(0.05), fallback_speed=+1.0, horizon=20)
  m = mon(_act(1, -1.0), fallback=fb)
  assert torch.allclose(m, torch.tensor([-0.05]), atol=1e-5)


def test_post_termination_steps_are_excluded():
  """mjlab auto-resets in place, so margins after an env's done belong to a
  DIFFERENT episode and must not enter the horizon min."""
  mon, _, fb = _monitor(_live(-0.45), fallback_speed=+5.0)
  m = mon(_act(1, -1.0), fallback=fb)
  # dies on step 1 at -0.55; the fallback's +5.0 "recovery" afterwards (a fresh
  # episode, in a real env) must not lift the verdict.
  assert torch.allclose(m, torch.tensor([-0.55]), atol=1e-5)


# --- parallel rollouts: min over R --------------------------------------------

def test_min_over_rollouts_one_failure_condemns_the_action():
  state = _live(0.5, 0.5)
  # rollout 0 drifts up, rollout 1 drifts down hard enough to cross zero.
  mon, sim, fb = _monitor(state, fallback_speed=0.0, rollouts_per_env=2,
                          drift=torch.tensor([+1.0, -2.0]))
  m = mon(_act(2, 0.0), fallback=fb)
  assert sim.num_envs == 4 and mon.rollouts_per_env == 2
  assert torch.all(m < 0), ("one failing rollout out of two must make the "
                            f"verdict unsafe, got {m}")

  # ... and with BOTH rollouts benign the verdict is the (positive) min.
  mon2, _, fb2 = _monitor(state, fallback_speed=0.0, rollouts_per_env=2,
                          drift=torch.tensor([+1.0, +0.5]))
  m2 = mon2(_act(2, 0.0), fallback=fb2)
  assert torch.allclose(m2, torch.full((2,), 0.55), atol=1e-5)


def test_rollouts_map_to_the_right_live_env():
  """Shadow env j mirrors live env j // R. A transposed reshape would mix the
  verdicts of different robots, which reads as noise rather than as a bug."""
  mon, _, fb = _monitor(_live(0.5, -0.2), fallback_speed=0.0,
                        rollouts_per_env=3, drift=torch.zeros(3))
  m = mon(_act(2, 0.0), fallback=fb)
  assert m[0] > 0 and m[1] < 0, m


# --- reach-avoid reduction ----------------------------------------------------

def test_reach_avoid_reduction_credits_reaching_the_target():
  """avoid = min_t g_t; reach-avoid = max_t min(l_t, min_{s<=t} g_s).

  A rollout that touches the target while still safe and only degrades
  afterwards is a WIN under reach-avoid and a near-loss under avoid.
  """
  state = _live(0.9)
  avoid, _, fb = _monitor(state, fallback_speed=-1.0)
  ra, _, fb2 = _monitor(state, fallback_speed=-1.0, reach_avoid=True)
  a = _act(1, +1.0)                       # nominal step lands on the target
  m_avoid, m_ra = avoid(a, fallback=fb), ra(a, fallback=fb2)
  # x runs 1.0 (in target, l=1) then down to 0.1.
  assert torch.allclose(m_avoid, torch.tensor([0.1]), atol=1e-4)
  assert torch.allclose(m_ra, torch.tensor([1.0]), atol=1e-4)
  assert m_ra > m_avoid


# --- recertification latching -------------------------------------------------

def test_recertify_every_latches_between_rollouts():
  live = {"x": _live(1.0, 1.0)}
  sim = ToyShadowSim(2, lambda: live["x"], dt=DT)
  mon = RolloutMonitor(2, DEV, sim, H, recertify_every=3)
  fb = PolicyFallback(2, DEV, constant_fallback(0.0), ACT_DIM)
  a = _act(2, 0.0)                        # x never moves -> margin == live x

  seen = []
  for t in range(9):
    live["x"] = _live(1.0 - 0.1 * t, 1.0 - 0.1 * t)   # the live state moves
    seen.append(mon(a, fallback=fb).clone())

  assert sim.seeds == 3, f"9 steps at k=3 must run 3 rollouts, ran {sim.seeds}"
  for block in (slice(0, 3), slice(3, 6), slice(6, 9)):
    vals = seen[block]
    assert all(torch.equal(v, vals[0]) for v in vals), "verdict must latch"
  assert torch.allclose(seen[0], torch.full((2,), 1.0), atol=1e-5)
  assert torch.allclose(seen[3], torch.full((2,), 0.7), atol=1e-5)
  assert torch.allclose(seen[6], torch.full((2,), 0.4), atol=1e-5)


def test_reset_forces_recertification():
  """A stale latch driving a FRESH episode is the bug class core.py warns about
  (a 77% livelock once came from exactly that), so a done env re-certifies."""
  live = {"x": _live(1.0, 1.0)}
  sim = ToyShadowSim(2, lambda: live["x"], dt=DT)
  mon = RolloutMonitor(2, DEV, sim, H, recertify_every=100)
  fb = PolicyFallback(2, DEV, constant_fallback(0.0), ACT_DIM)
  a = _act(2, 0.0)

  mon(a, fallback=fb)
  mon(a, fallback=fb)
  assert sim.seeds == 1, "no reset -> the verdict stays latched"
  mon.reset(torch.tensor([True, False]))
  live["x"] = _live(0.2, 0.2)
  out = mon(a, fallback=fb)
  assert sim.seeds == 2, "a reset env must force a re-certification"
  assert torch.allclose(out, torch.full((2,), 0.2), atol=1e-5)


def test_seeding_is_idempotent():
  """Evaluating the monitor twice at one state must give one answer — the Def-2
  property test evaluates it several times per state and would otherwise be
  measuring drift rather than validity."""
  mon, _, fb = _monitor(_live(0.3, 0.8), fallback_speed=-0.5)
  a = _act(2, 0.2)
  assert torch.equal(mon(a, fallback=fb), mon(a, fallback=fb))


# --- adversarial variant ------------------------------------------------------

class _DstbDrivenSim(ToyShadowSim):
  """A toy sim whose state is moved by the DSTB channel (the last action dim),
  so the disturbance policy — not the fallback — is what the rollout feels."""

  def step(self, action):
    return super().step(action[:, -1:].expand_as(action))


def test_adversarial_monitor_lets_the_disturbance_condemn_a_safe_rollout():
  """Same state, same fallback: the fallback-only rollout certifies, the game
  against a disturbance policy does not. That IS the gameplay filter."""
  state = _live(0.3, 0.3)
  fb = PolicyFallback(2, DEV, constant_fallback(+0.2), ACT_DIM)
  a = _act(2, 0.0)
  # the nominal step has no shadow observation yet, so it hands the adversary
  # the LIVE ctx (the seeded shadow state IS the live state)
  live_ctx = dict(s_obs=state.unsqueeze(-1).expand(2, OBS_DIM))

  benign = RolloutMonitor(2, DEV, ToyShadowSim(2, lambda: state, dt=DT), H)
  assert torch.all(benign(a, fallback=fb, **live_ctx) > 0)

  game = AdversarialRolloutMonitor(
    2, DEV, _DstbDrivenSim(2, lambda: state, dt=DT), H,
    lambda s_obs, **_: torch.full((s_obs.shape[0], 1), -1.0))
  assert torch.all(game(a, fallback=fb, **live_ctx) < 0)


# --- the architectural claim: the intervention is untouched -------------------

def test_rollout_composes_with_the_unmodified_least_restrictive_intervention():
  """The gameplay filter must be a NEW MONITOR and nothing else. If this ever
  needs an intervention change, the three-module interface is wrong."""
  state = _live(0.5, -0.5)
  sim = ToyShadowSim(2, lambda: state, dt=DT)
  filt = rollout_filter(2, DEV, constant_fallback(0.0), sim, H,
                        action_dim=ACT_DIM)
  assert isinstance(filt, SafetyFilter)
  assert type(filt.intervention) is LeastRestrictiveIntervention

  a_nom = _act(2, 0.9)
  obs = torch.zeros(2, OBS_DIM)
  action, info = filt(a_nom, speed=torch.zeros(2),
                      fresh=torch.ones(2, dtype=torch.bool), s_obs=obs)
  assert not bool(info.engaged[0]) and bool(info.engaged[1])
  assert torch.equal(action[0], a_nom[0])                 # certified: untouched
  assert torch.equal(action[1], torch.zeros(ACT_DIM))     # condemned: pi^<
  filt.reset(torch.tensor([False, True]))                 # reset contract holds


# --- per-env IDENTITY: the domain-randomization draw ---------------------------
# A shadow that copies STATE perfectly but redraws its own domain randomization
# certifies a robot from the same distribution, not THE robot being filtered.
# That gap once invalidated a whole horizon sweep (E051/E052 gameplay rows), so
# it is pinned from both ends: the field list is derived, and an unknown
# randomizer is an error rather than a silent skip.

from types import SimpleNamespace  # noqa: E402

from robot_safety_sandbox.filters import rollout as R  # noqa: E402


def _fake_env(n, *, friction, bias, events, dr_fields=("geom_friction",)):
  """The three attributes the identity copy touches, and nothing else."""
  robot = SimpleNamespace(data=SimpleNamespace(encoder_bias=bias))
  return SimpleNamespace(
    num_envs=n,
    cfg=SimpleNamespace(events=events),
    event_manager=SimpleNamespace(domain_randomization_fields=dr_fields),
    sim=SimpleNamespace(model=SimpleNamespace(geom_friction=friction)),
    scene=SimpleNamespace(entities={"robot": robot}))


def _dr_events():
  friction_fn = lambda *a, **k: None                       # noqa: E731
  friction_fn.model_fields = ("geom_friction",)
  friction_fn.__name__ = "geom_friction"
  bias_fn = lambda *a, **k: None                           # noqa: E731
  bias_fn.__name__ = "encoder_bias"                        # no model_fields
  return {"foot_friction": SimpleNamespace(func=friction_fn, mode="startup"),
          "encoder_bias": SimpleNamespace(func=bias_fn, mode="startup")}


def _pair(n_live=2, r=1):
  """A live env and a shadow with DIFFERENT draws, as construction leaves them."""
  n_sh = n_live * r
  live = _fake_env(n_live, friction=torch.arange(n_live * 3).float().reshape(n_live, 3),
                   bias=torch.arange(n_live * 4).float().reshape(n_live, 4) / 10,
                   events=_dr_events())
  shadow = _fake_env(n_sh, friction=torch.zeros(n_sh, 3),
                     bias=torch.zeros(n_sh, 4), events=_dr_events())
  m = torch.arange(n_live).repeat_interleave(r)
  return live, shadow, m


def test_dr_identity_fields_are_derived_from_the_event_manager():
  live, _shadow, _m = _pair()
  model_fields, entity_fields = R.dr_identity_fields(live)
  assert model_fields == ("geom_friction",)
  assert entity_fields == ("encoder_bias",), (
    "a DR function with no model_fields writes where sim.model cannot be "
    "asked; encoder_bias must come across from entity.data")


def test_an_unrecognized_startup_randomizer_is_an_error_not_a_silent_skip():
  """The failure mode this whole section exists to prevent: a new startup event
  that hands each env a different robot, copied by nobody, noticed by no one."""
  live, _shadow, _m = _pair()
  mystery = lambda *a, **k: None                           # noqa: E731
  mystery.__name__ = "randomize_something_new"
  live.cfg.events["new"] = SimpleNamespace(func=mystery, mode="startup")
  with pytest.raises(ValueError, match="model_fields"):
    R.dr_identity_fields(live)
  # ... with the explicit opt-out it is allowed through, and only then.
  mystery._zoo_shadow_identity_safe = True
  assert R.dr_identity_fields(live)[0] == ("geom_friction",)


def test_sync_copies_the_realized_draw_and_never_resamples():
  live, shadow, m = _pair(n_live=3)
  assert R.dr_identity_diff(live, shadow, m, 3, 3)["geom_friction"] > 0
  copied = R.sync_dr_identity(live, shadow, m, 3, 3)
  assert set(copied) == {"geom_friction", "robot.encoder_bias"}
  assert torch.equal(shadow.sim.model.geom_friction,
                     live.sim.model.geom_friction)
  assert torch.equal(shadow.scene.entities["robot"].data.encoder_bias,
                     live.scene.entities["robot"].data.encoder_bias)
  R.assert_dr_identity(live, shadow, m, 3, 3)


def test_every_rollout_lane_of_one_live_env_gets_that_env_s_values():
  """R > 1 must give all R lanes the SAME robot. R independent draws would be a
  robust certificate over the DR distribution -- a different guarantee, and one
  that must never appear by accident."""
  live, shadow, m = _pair(n_live=2, r=3)
  R.sync_dr_identity(live, shadow, m, 2, 6)
  f = shadow.sim.model.geom_friction
  for j in range(2):
    lanes = f[j * 3:(j + 1) * 3]
    assert torch.equal(lanes, lanes[:1].expand_as(lanes)), lanes
    assert torch.equal(lanes[0], live.sim.model.geom_friction[j])


def test_assert_dr_identity_fails_loudly_on_a_mismatch():
  live, shadow, m = _pair()
  R.sync_dr_identity(live, shadow, m, 2, 2)
  shadow.scene.entities["robot"].data.encoder_bias[1, 2] += 1e-4
  with pytest.raises(RuntimeError, match="not the same robot"):
    R.assert_dr_identity(live, shadow, m, 2, 2)


# --- the real mjlab shadow-sim path -------------------------------------------

mjlab_env = pytest.mark.skipif(
  not torch.cuda.is_available(),
  reason="mjlab (mujoco_warp) needs a CUDA device")


@pytest.fixture(scope="module")
def car_pair():
  """A live car_goal env and a 2-env shadow of it. Module-scoped: the mjlab
  build costs ~1.2 s per env and every test below re-places the car itself."""
  from robot_safety_sandbox import make_tensor
  dev = "cuda:0"
  live = make_tensor("car_goal", num_envs=2, device=dev)
  shadow = make_tensor("car_goal", num_envs=2, device=dev)
  live.reset()
  shadow.reset()
  return live, shadow, dev


def _place_car(env, x, y):
  """Put every car at local (x, y), at rest, in its DEFAULT heading.

  The default spawn quaternion is a +90 deg yaw (the chassis' own forward axis
  is not world +x), and it is the heading that drives toward the goal — so take
  it from the asset rather than hand-writing a quaternion.
  """
  mj = env.mj
  robot, dev, n = mj.scene["robot"], mj.device, mj.num_envs
  pos = mj.scene.env_origins.clone()
  pos[:, 0] += x
  pos[:, 1] += y
  pos[:, 2] = robot.data.default_root_state[:, 2]
  quat = robot.data.default_root_state[:, 3:7]
  robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1))
  robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev))
  robot.write_joint_state_to_sim(torch.zeros(n, 2, device=dev),
                                 torch.zeros(n, 2, device=dev))
  mj.scene.write_data_to_sim()
  mj.sim.forward()


def _rel(env):
  return (env.mj.scene["robot"].data.root_link_pos_w
          - env.mj.scene.env_origins)


@mjlab_env
def test_mjlab_seeding_round_trips_the_live_state(car_pair):
  """The shadow reproduces the live trajectory to float32 on the first step.

  This is the claim the pre-implementation docstring denied. What remains after
  a few steps is the system's own Lyapunov divergence, not missing state.
  """
  from robot_safety_sandbox.filters import MjlabShadowSim
  live, shadow, dev = car_pair
  torch.manual_seed(0)
  for _ in range(15):                     # give the live env a nontrivial state
    live.step_tensor(torch.rand(2, 2, device=dev) * 2 - 1)

  sim = MjlabShadowSim(live, shadow, num_envs=2)
  sim.seed()
  assert torch.equal(_rel(live), _rel(shadow)), "seeded pose must be exact"
  assert torch.equal(live.mj.sim.data.qpos[:, 7:],
                     shadow.mj.sim.data.qpos[:, 7:]), (
    "qpos beyond the free joint must come across too — car_goal's caster is a "
    "BALL joint, which entity.data.joint_pos does not cover")
  assert torch.equal(live.mj.sim.data.qacc_warmstart,
                     shadow.mj.sim.data.qacc_warmstart), (
    "the solver warm-start is state; without it the rollout parts company "
    "with the live env immediately")

  acts = torch.rand(5, 2, 2, device=dev) * 2 - 1
  for t in range(5):
    live.step_tensor(acts[t])
    shadow.step_tensor(acts[t])
  err = (_rel(live) - _rel(shadow)).abs().max().item()
  assert err < 1e-3, f"shadow diverged by {err:.2e} m over 5 steps"


@mjlab_env
def test_mjlab_rollout_monitor_separates_a_doomed_fallback_from_a_safe_one(
    car_pair):
  """car_goal, real task margins: a full-throttle fallback drives the car into
  the keep-out cylinder (g < 0); a braking fallback holds the margin it started
  with. Same state, same horizon — only pi^< differs."""
  from robot_safety_sandbox.filters import MjlabShadowSim
  live, shadow, dev = car_pair
  # obstacle 0 sits at local (0.75, +0.32), r = 0.25; the car's footprint radius
  # is 0.15. Park 0.45 m short of it, pointing straight at it: g = 0.05/0.5.
  _place_car(live, x=0.30, y=0.32)

  sim = MjlabShadowSim(live, shadow, num_envs=2)
  mon = RolloutMonitor(2, dev, sim, horizon=40)
  neutral = torch.zeros(2, 2, device=dev)
  charge = PolicyFallback(2, dev, lambda s_obs, **_: torch.ones_like(s_obs[:, :2]))
  brake = PolicyFallback(2, dev, lambda s_obs, **_: torch.zeros_like(s_obs[:, :2]))

  m_doomed = mon(neutral, fallback=charge)
  mon.reset(torch.ones(2, dtype=torch.bool, device=dev))   # re-certify
  m_safe = mon(neutral, fallback=brake)

  assert torch.all(m_doomed < 0), (
    f"charging into the keep-out cylinder must fail, got {m_doomed}")
  assert torch.all(m_safe > 0), f"braking must stay certified, got {m_safe}"


@pytest.fixture(scope="module")
def go2_pair():
  """go2_stabilize: the env with the state a rollout was said not to survive —
  a contact sensor with a 4-substep force history AND seven observation-history
  buffers on the actor group."""
  from robot_safety_sandbox import make_tensor
  dev = "cuda:0"
  live = make_tensor("go2_stabilize", num_envs=2, device=dev)
  shadow = make_tensor("go2_stabilize", num_envs=2, device=dev)
  live.reset()
  shadow.reset()
  torch.manual_seed(0)
  for _ in range(12):                    # accumulate real history to carry over
    live.step_tensor(torch.rand(2, 12, device=dev) * 2 - 1)
  return live, shadow, dev


@mjlab_env
def test_mjlab_seeding_round_trips_sensor_and_observation_history(go2_pair):
  """The retracted claim, tested.

  "mjlab state save/restore cannot round-trip observation-history buffers" was
  the stated reason a rollout monitor could not exist. It can: they are plain
  torch CircularBuffers on the observation manager, and so are the contact
  sensor's force history and air-time state.
  """
  from robot_safety_sandbox import spec
  from robot_safety_sandbox.filters import MjlabShadowSim
  live, shadow, _dev = go2_pair
  MjlabShadowSim(live, shadow, num_envs=2).seed()
  L, S = live.mj, shadow.mj

  hb_l = L.observation_manager._group_obs_term_history_buffer["actor"]
  hb_s = S.observation_manager._group_obs_term_history_buffer["actor"]
  assert set(hb_l) == set(hb_s) and hb_l, "fixture must have obs history"
  for k, buf in hb_l.items():
    assert torch.equal(buf._buffer, hb_s[k]._buffer), f"{k} history not synced"
    assert buf._pointer == hb_s[k]._pointer, f"{k} write cursor not synced"
    assert torch.equal(buf._num_pushes, hb_s[k]._num_pushes)
  # ... and the check is not vacuous: some of that history is nonzero. (Not
  # ALL of it — go2_stabilize's twist command is identically zero.)
  assert sum(float(b._buffer.abs().sum()) for b in hb_l.values()) > 0

  contact = L.scene.sensors["nonfoot_ground_touch"]
  assert contact.cfg.history_length == 4, "fixture must exercise force history"
  assert torch.equal(contact._history_state["force"],
                     S.scene.sensors["nonfoot_ground_touch"]._history_state["force"])
  feet = L.scene.sensors["feet_ground_contact"]._air_time_state
  assert feet.current_air_time.abs().sum() + feet.last_contact_time.abs().sum() > 0
  assert torch.equal(
    feet.current_contact_time,
    S.scene.sensors["feet_ground_contact"]._air_time_state.current_contact_time)

  assert torch.equal(L.action_manager._prev_action, S.action_manager._prev_action)
  assert torch.equal(L.command_manager.get_command("twist"),
                     S.command_manager.get_command("twist"))
  assert torch.equal(L.episode_length_buf, S.episode_length_buf)

  # ... and the state the margin is a function of round-trips EXACTLY.
  assert (L.sim.data.qpos - S.sim.data.qpos).abs().max() < 1e-6
  assert (L.sim.data.qvel - S.sim.data.qvel).abs().max() < 1e-6
  g_l, l_l = spec("go2_stabilize").margin_fn(L)
  g_s, l_s = spec("go2_stabilize").margin_fn(S)
  assert torch.equal(g_l, g_s), "the seeded shadow must score the same margin"
  assert (l_l - l_s).abs().max() < 1e-5


@mjlab_env
def test_instantaneous_contact_history_switches_the_margin_source(go2_pair):
  """The documented mitigation: when the force history cannot be round-tripped,
  the shadow's margin reads the instantaneous force instead of a half-filled
  window — never a silently wrong peak."""
  from robot_safety_sandbox import spec
  from robot_safety_sandbox.filters import MjlabShadowSim
  live, shadow, _dev = go2_pair
  sim = MjlabShadowSim(live, shadow, num_envs=2,
                       contact_history="instantaneous")
  sim.seed()
  S = shadow.mj
  assert S._zoo_instantaneous_contact is True
  assert not S.scene.sensors["nonfoot_ground_touch"]._history_state["force"].any(), (
    "the un-round-trippable buffer must be zeroed, not left stale")
  g, _l = spec("go2_stabilize").margin_fn(S)
  assert torch.isfinite(g).all()
  del S._zoo_instantaneous_contact              # leave the fixture as found


@mjlab_env
def test_mjlab_seeding_makes_the_shadow_the_same_robot(go2_pair):
  """On a real velocity env the shadow starts as a DIFFERENT robot -- foot
  friction, base COM and encoder bias are drawn per env at construction -- and
  seeding must make it the live one. Without this the rollout certifies a
  sample from the DR distribution and the H = T guarantee is void."""
  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.filters import MjlabShadowSim
  live, _shadow, dev = go2_pair
  fresh = make_tensor("go2_stabilize", num_envs=4, device=dev)   # R = 2
  fresh.reset()
  L, S = live.mj, fresh.mj
  m = torch.arange(2, device=dev).repeat_interleave(2)

  before = R.dr_identity_diff(L, S, m, 2, 4)
  assert {"geom_friction", "body_ipos", "robot.encoder_bias"} <= set(before), (
    f"fixture must exercise startup DR; got fields {sorted(before)}")
  assert max(before.values()) > 1e-3, (
    f"two independently built envs must start as different robots: {before}")

  MjlabShadowSim(live, fresh, num_envs=2,
                 rollouts_per_env=2).seed()          # asserts identity itself
  after = R.dr_identity_diff(L, S, m, 2, 4)
  assert all(v == 0.0 for v in after.values()), after
  # ... and both lanes of one live env got the SAME robot, not two draws.
  fr = S.sim.model.geom_friction[:]
  assert torch.equal(fr[0], fr[1]) and torch.equal(fr[2], fr[3])
  assert not torch.equal(fr[0], fr[2]), (
    "the two LIVE envs must still differ, or this test proves nothing")

  # ... and the same action then gives the same margin, exactly.
  a = torch.zeros(2, 12, device=dev)
  _o, g_l, _d, _t, _l = live.step_tensor(a)
  _o, g_s, _d, _t, _l = fresh.step_tensor(a.repeat_interleave(2, dim=0))
  assert (g_l.repeat_interleave(2) - g_s).abs().max() < 1e-5, (g_l, g_s)
  fresh.close()


@mjlab_env
def test_mjlab_min_over_rollouts_uses_the_worst_rollout(car_pair):
  """R = 2 with a per-rollout scripted fallback: the even shadow env of each
  pair brakes, the odd one charges. The verdict must be the charging one."""
  from robot_safety_sandbox import make_tensor
  from robot_safety_sandbox.filters import MjlabShadowSim
  live, _shadow, dev = car_pair
  shadow4 = make_tensor("car_goal", num_envs=4, device=dev)
  shadow4.reset()
  _place_car(live, x=0.30, y=0.32)

  sim = MjlabShadowSim(live, shadow4, num_envs=2, rollouts_per_env=2)
  mon = RolloutMonitor(2, dev, sim, horizon=40)

  def split_fallback(s_obs, **_):
    a = torch.zeros(s_obs.shape[0], 2, device=s_obs.device)
    a[1::2] = 1.0
    return a

  m = mon(torch.zeros(2, 2, device=dev),
          fallback=PolicyFallback(2, dev, split_fallback))
  assert torch.all(m < 0), (
    f"one charging rollout out of two must condemn the action, got {m}")
  shadow4.close()
