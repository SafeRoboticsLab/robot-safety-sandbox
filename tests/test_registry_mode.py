"""Unit tests for the registry's ``mode`` axis (the retired ``kind`` split).

A task's mode names the safety_sb3 BACKUP it is trained under, and drives four
things: the learner ``algo_name()`` resolves, whether a ``margin_fn`` is
required, whether the bridges build the env in dense-reward mode, and (legacy)
what ``kind=`` maps onto.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from robot_safety_sandbox import registry  # noqa: E402
from robot_safety_sandbox.registry import (  # noqa: E402
  AVOID, CUMULATIVE, MODES, REACH_AVOID, TaskSpec, algo_name, list_tasks,
  make_numpy, make_tensor, register)


def _cfg_builder(play=False):        # never called: no test builds an env
  raise AssertionError("cfg_builder should not run in these tests")


def _margins(env):                   # (env) -> (g, l)
  raise AssertionError("margin_fn should not run in these tests")


def _avoid_only_margins(env):
  raise AssertionError("margin_fn should not run in these tests")


_avoid_only_margins.has_target = False   # what margins.avoid_only() stamps on


@pytest.fixture
def clean_registry():
  """Register throwaway specs without leaking into the real task registry."""
  saved = dict(registry._REGISTRY)
  yield
  registry._REGISTRY.clear()
  registry._REGISTRY.update(saved)


def _reg(task_id, **kw):
  kw.setdefault("cfg_builder", _cfg_builder)
  register(TaskSpec(task_id=task_id, **kw))
  return task_id


# --- the mode strings ARE safety_sb3's backup modes ---------------------------

def test_modes_match_safety_sb3_backups():
  """registry.py defines the mode strings LOCALLY (a cumulative-only install has
  no safety_sb3) — so pin them against the real thing whenever it IS installed."""
  backups = pytest.importorskip("safety_sb3.backups")
  assert AVOID == backups.AVOID
  assert REACH_AVOID == backups.REACH_AVOID
  assert CUMULATIVE == backups.CUMULATIVE
  assert MODES == backups.MODES
  assert registry.SAFETY_MODES == backups.SAFETY_MODES


def test_registry_does_not_import_safety_sb3():
  src = open(os.path.join(os.path.dirname(_HERE),
                          "robot_safety_sandbox", "registry.py")).read()
  assert "import safety_sb3" not in src


# --- mode <-> default_algo ----------------------------------------------------

def test_default_mode_and_algo_are_todays_defaults(clean_registry):
  s = registry.spec(_reg("t_default", margin_fn=_margins))
  assert (s.mode, s.default_algo) == (AVOID, "SafetyPPO")


@pytest.mark.parametrize("mode,algo", [
  (AVOID, "SafetyPPO"), (REACH_AVOID, "ReachAvoidPPO"), (CUMULATIVE, "PPO")])
def test_default_algo_derives_from_mode(clean_registry, mode, algo):
  s = registry.spec(_reg(f"t_derive_{mode}", mode=mode,
                         margin_fn=None if mode == CUMULATIVE else _margins))
  assert s.default_algo == algo


@pytest.mark.parametrize("algo,mode", [
  ("SafetyPPO", AVOID), ("IsaacsPPO", AVOID), ("ReachAvoidPPO", REACH_AVOID),
  ("GameplayPPO", REACH_AVOID), ("PPO", CUMULATIVE)])
def test_explicit_default_algo_fixes_the_mode(clean_registry, algo, mode):
  """Registrations that predate `mode=` name a learner; that still fixes it."""
  s = registry.spec(_reg(f"t_algo_{algo}", default_algo=algo,
                         margin_fn=None if mode == CUMULATIVE else _margins))
  assert s.mode == mode


def test_default_algo_contradicting_mode_is_refused(clean_registry):
  with pytest.raises(ValueError, match="default_algo"):
    TaskSpec(task_id="t_bad", cfg_builder=_cfg_builder, mode=AVOID,
             default_algo="ReachAvoidPPO", margin_fn=_margins)


def test_unknown_mode_and_algo_are_refused(clean_registry):
  with pytest.raises(ValueError, match="mode"):
    TaskSpec(task_id="t_bad", cfg_builder=_cfg_builder, mode="nominal",
             margin_fn=_margins)
  with pytest.raises(ValueError, match="not a known learner"):
    TaskSpec(task_id="t_bad", cfg_builder=_cfg_builder, default_algo="A2C",
             margin_fn=_margins)


# --- mode dispatch (algo_name) ------------------------------------------------

@pytest.mark.parametrize("mode,solo,duo", [
  (AVOID, "SafetyPPO", "IsaacsPPO"),
  (REACH_AVOID, "ReachAvoidPPO", "GameplayPPO")])
def test_algo_name_dispatches_mode_x_players(clean_registry, mode, solo, duo):
  t = _reg(f"t_dispatch_{mode}", mode=mode, margin_fn=_margins,
           supports_adversary=True)
  assert algo_name(t) == solo
  assert algo_name(t, adversary=True) == duo


def test_algo_name_cumulative_is_stock_ppo(clean_registry):
  t = _reg("t_cum", mode=CUMULATIVE)
  assert algo_name(t) == "PPO"


def test_algo_name_refuses_two_player_cumulative(clean_registry):
  t = _reg("t_cum_adv", mode=CUMULATIVE, supports_adversary=True)
  with pytest.raises(ValueError, match="two-player cumulative"):
    algo_name(t, adversary=True)


def test_algo_name_refuses_reach_avoid_on_avoid_only_margins(clean_registry):
  t = _reg("t_no_target", mode=REACH_AVOID, margin_fn=_avoid_only_margins)
  with pytest.raises(ValueError, match="AVOID-ONLY"):
    algo_name(t)


# --- margin_fn requirement ----------------------------------------------------

@pytest.mark.parametrize("mode", [AVOID, REACH_AVOID])
def test_safety_modes_require_a_margin_fn(clean_registry, mode):
  with pytest.raises(ValueError, match="needs a margin_fn"):
    TaskSpec(task_id="t_nomargin", cfg_builder=_cfg_builder, mode=mode)


def test_cumulative_needs_no_margin_fn(clean_registry):
  assert registry.spec(_reg("t_cum2", mode=CUMULATIVE)).margin_fn is None


# --- dense_reward derives from the mode ---------------------------------------

@pytest.mark.parametrize("maker,attr", [
  (make_tensor, "MjlabTensorSafetyEnv"), (make_numpy, "MjlabNumpySafetyEnv")])
@pytest.mark.parametrize("mode,dense", [
  (AVOID, False), (REACH_AVOID, False), (CUMULATIVE, True)])
def test_bridges_build_dense_iff_cumulative(clean_registry, monkeypatch,
                                            maker, attr, mode, dense):
  from robot_safety_sandbox import base

  seen = {}

  def _stub(*a, **kw):
    seen.update(kw)
    return "env"

  monkeypatch.setattr(base, attr, _stub)
  t = _reg(f"t_dense_{mode}_{attr}", mode=mode,
           margin_fn=None if mode == CUMULATIVE else _margins)
  assert maker(t, 4, "cpu") == "env"
  assert seen["dense_reward"] is dense


def test_explicit_dense_reward_still_wins(clean_registry, monkeypatch):
  from robot_safety_sandbox import base

  seen = {}
  monkeypatch.setattr(base, "MjlabTensorSafetyEnv",
                      lambda *a, **kw: seen.update(kw) or "env")
  t = _reg("t_dense_override", mode=AVOID, margin_fn=_margins)
  make_tensor(t, 4, "cpu", dense_reward=True)
  assert seen["dense_reward"] is True


# --- deprecated kind= path ----------------------------------------------------

@pytest.mark.parametrize("kind,mode", [("safety", AVOID),
                                       ("nominal", CUMULATIVE)])
def test_deprecated_kind_kwarg_maps_to_mode(clean_registry, kind, mode):
  with pytest.warns(DeprecationWarning, match="kind"):
    s = TaskSpec(task_id=f"t_kind_{kind}", cfg_builder=_cfg_builder, kind=kind,
                 margin_fn=None if mode == CUMULATIVE else _margins)
  assert s.mode == mode
  assert s.default_algo == registry._MODE_ALGO[mode]


def test_deprecated_kind_conflicting_with_mode_is_refused(clean_registry):
  with pytest.raises(ValueError, match="deprecated kind"):
    with pytest.warns(DeprecationWarning):
      TaskSpec(task_id="t_kind_bad", cfg_builder=_cfg_builder, mode=AVOID,
               kind="nominal")


def test_deprecated_list_tasks_kind_kwarg(clean_registry):
  registry._REGISTRY.clear()
  _reg("t_a", mode=AVOID, margin_fn=_margins)
  _reg("t_c", mode=CUMULATIVE)
  assert list_tasks() == ["t_a", "t_c"]
  assert list_tasks(mode=CUMULATIVE) == ["t_c"]
  with pytest.warns(DeprecationWarning, match="kind"):
    assert list_tasks(kind="nominal") == ["t_c"]
  with pytest.warns(DeprecationWarning, match="kind"):
    assert list_tasks(kind="safety") == ["t_a"]
  with pytest.warns(DeprecationWarning, match="nominal"):
    assert list_tasks("nominal") == ["t_c"]     # retired value, positionally
  with pytest.raises(ValueError, match="unknown mode"):
    list_tasks("bogus")


# --- the shipped registry -----------------------------------------------------

def test_every_registered_task_resolves():
  """Import the real package: every task has a valid mode, and its learner
  resolves (or fails only for the documented avoid-only/reach-avoid clash)."""
  import robot_safety_sandbox  # noqa: F401  (registers every task)

  tasks = list_tasks()
  assert len(tasks) > 20
  for t in tasks:
    s = registry.spec(t)
    assert s.mode in MODES, t
    assert (s.margin_fn is not None) == (s.mode != CUMULATIVE), t
    assert algo_name(t) == registry._LEARNER[(s.mode, 1)], t
    if s.supports_adversary and s.mode != CUMULATIVE:
      assert algo_name(t, adversary=True) == registry._LEARNER[(s.mode, 2)], t
