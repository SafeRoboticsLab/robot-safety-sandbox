"""Unit tests for the registry's ``mode`` axis and the MAP name formula.

A task's mode names the safety_sb3 BACKUP it is trained under, and drives three
things: the learner ``algo_name()`` resolves (the MAP's **M**), whether a
``margin_fn`` is required, and whether the bridges build the env in dense-reward
mode. It is the ONLY thing a registration says about the learner — the **A**
comes from the trainer family and the **P** from ``--adversary``.
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


# --- mode is REQUIRED and validated -------------------------------------------

def test_mode_is_required(clean_registry):
  with pytest.raises(ValueError, match="declares no mode"):
    TaskSpec(task_id="t_nomode", cfg_builder=_cfg_builder, margin_fn=_margins)


def test_unknown_mode_is_refused(clean_registry):
  # "nominal" was the retired kind= value; it is not a mode and buys nothing.
  with pytest.raises(ValueError, match="must be one of"):
    TaskSpec(task_id="t_bad", cfg_builder=_cfg_builder, mode="nominal",
             margin_fn=_margins)


def test_spec_carries_no_learner_field(clean_registry):
  """The learner is DERIVED. A registration must not be able to pin one, and
  pipeline lineage lives in docs/log/experiments.md, not on the spec."""
  s = registry.spec(_reg("t_fields", mode=AVOID, margin_fn=_margins))
  assert not hasattr(s, "default_algo")
  assert not hasattr(s, "warmstart_from")
  with pytest.raises(TypeError):
    TaskSpec(task_id="t_pin", cfg_builder=_cfg_builder, mode=AVOID,
             margin_fn=_margins, default_algo="ReachAvoidPPO1P")


# --- the MAP: algo_name is Mode + Algorithm + Players --------------------------

@pytest.mark.parametrize("family,alg", [("on_policy", "PPO"),
                                        ("off_policy", "SAC")])
@pytest.mark.parametrize("mode,prefix", [(AVOID, "Safety"),
                                         (REACH_AVOID, "ReachAvoid")])
def test_algo_name_is_mode_plus_algorithm_plus_players(
    clean_registry, family, alg, mode, prefix):
  t = _reg(f"t_map_{mode}_{family}", mode=mode, margin_fn=_margins,
           supports_adversary=True)
  assert algo_name(t, family=family) == f"{prefix}{alg}1P"
  assert algo_name(t, adversary=True, family=family) == f"{prefix}{alg}2P"


def test_algo_name_defaults_to_the_on_policy_family(clean_registry):
  t = _reg("t_family_default", mode=REACH_AVOID, margin_fn=_margins)
  assert algo_name(t) == "ReachAvoidPPO1P"


def test_algo_name_refuses_an_unknown_family(clean_registry):
  t = _reg("t_family_bad", mode=AVOID, margin_fn=_margins)
  with pytest.raises(ValueError, match="unknown family"):
    algo_name(t, family="model_based")


@pytest.mark.parametrize("family,alg", [("on_policy", "PPO"),
                                        ("off_policy", "SAC")])
def test_algo_name_cumulative_is_stock_sb3_with_no_player_suffix(
    clean_registry, family, alg):
  t = _reg(f"t_cum_{family}", mode=CUMULATIVE)
  assert algo_name(t, family=family) == alg


def test_algo_name_refuses_two_player_cumulative(clean_registry):
  t = _reg("t_cum_adv", mode=CUMULATIVE, supports_adversary=True)
  with pytest.raises(ValueError, match="two-player cumulative"):
    algo_name(t, adversary=True)


def test_algo_name_refuses_an_adversary_the_task_does_not_define(clean_registry):
  t = _reg("t_no_adv", mode=AVOID, margin_fn=_margins)
  with pytest.raises(ValueError, match="does not define an adversary"):
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


# --- list_tasks ---------------------------------------------------------------

def test_list_tasks_filters_by_mode(clean_registry):
  registry._REGISTRY.clear()
  _reg("t_a", mode=AVOID, margin_fn=_margins)
  _reg("t_c", mode=CUMULATIVE)
  assert list_tasks() == ["t_a", "t_c"]
  assert list_tasks(CUMULATIVE) == ["t_c"]
  assert list_tasks(mode=AVOID) == ["t_a"]
  with pytest.raises(ValueError, match="unknown mode"):
    list_tasks("nominal")            # the retired kind= value is not a mode


# --- the shipped registry -----------------------------------------------------

#: mode -> (1P, 2P) learner name, spelled out INDEPENDENTLY of registry._PREFIX
#: so a typo in the formula can't agree with a typo in the test.
_EXPECTED = {
  AVOID: {"on_policy": ("SafetyPPO1P", "SafetyPPO2P"),
          "off_policy": ("SafetySAC1P", "SafetySAC2P")},
  REACH_AVOID: {"on_policy": ("ReachAvoidPPO1P", "ReachAvoidPPO2P"),
                "off_policy": ("ReachAvoidSAC1P", "ReachAvoidSAC2P")},
  CUMULATIVE: {"on_policy": ("PPO", None), "off_policy": ("SAC", None)},
}


def test_every_registered_task_resolves():
  """Import the real package: every task declares a valid mode and resolves to
  the expected MAP name in BOTH families, at both player counts."""
  import robot_safety_sandbox  # noqa: F401  (registers every task)

  tasks = list_tasks()
  assert len(tasks) > 20
  for t in tasks:
    s = registry.spec(t)
    assert s.mode in MODES, t
    assert (s.margin_fn is not None) == (s.mode != CUMULATIVE), t
    for family, (solo, duo) in _EXPECTED[s.mode].items():
      assert algo_name(t, family=family) == solo, (t, family)
      if s.supports_adversary and s.mode != CUMULATIVE:
        assert algo_name(t, adversary=True, family=family) == duo, (t, family)
