"""PORL ("policy optimization in a filtered environment"): the Go2 velocity
walker trained INSIDE a safety filter, against the same walker trained bare.

The claim under test is the PORL one: a task policy learned in a filtered
environment reaches the same asymptotic return as one learned bare, at the same
rate or slightly faster (fewer catastrophic transitions wasted), while making
near-zero failures DURING training -- and is then very safe when deployed under
the same filter.

Both arms train the SAME task, ``go2_walker_filtered``. The only difference is
whether the env is wrapped by
:class:`~robot_safety_sandbox.filtered_env.FilteredTensorEnv`; nothing about
the task, the reward, the learner or the hyperparameters changes between them,
which is what makes the two curves comparable.

Why this is a separate task from ``go2_walker_flat``:

  * the command space is PINNED to a constant forward ``cmd_vx`` (no lateral,
    no yaw, no standing envs) instead of sampled over the full range. The
    fallback twin is a command-conditioned policy trained at one command, so a
    free command space would put its critic out of distribution and make the
    filter's engagement depend on an axis the experiment is not studying. This
    is the same env ``go2_locomote`` uses, at the same operating point as the
    E051/E054 gauntlets.
  * it carries a ``margin_fn`` despite being mode="cumulative", and asks the
    bridge for ``dense_margins``: the learner's reward stays the env's dense
    stack (that is what "cumulative" means), while (g, l) are computed
    alongside it so the filter's telemetry and the FAILURE COUNTER -- the
    experiment's headline plot -- read the same margin the twin was trained on.

``stance_margins`` is deliberately the twin's own margin, not a new one: g < 0
means a body corner reached the floor, i.e. exactly the failure the fallback
was trained to avoid, so "failures during training" is measured in the units
the filter is supposed to control.
"""

from __future__ import annotations

from functools import partial

from ..registry import CUMULATIVE, TaskSpec, register

#: the pinned forward command, m/s. Matches go2_locomote's default and the
#: E051/E054 operating point.
CMD_VX = 1.0


def register_all() -> None:
  from robot_safety_sandbox.envs.go2_stabilize.env_cfg import (
    go2_locomote_env_cfg, stance_margins)

  register(TaskSpec(
    task_id="go2_walker_filtered",
    cfg_builder=partial(go2_locomote_env_cfg, cmd_vx=CMD_VX),
    margin_fn=stance_margins,
    mode=CUMULATIVE,
    supports_adversary=True,
    kwargs={"dense_margins": True},
    description=f"Go2 velocity walker at a pinned cmd_vx={CMD_VX} m/s, dense "
                "reward + stance margins. The PORL task: train it bare and "
                "inside a safety filter and compare failures / return / "
                "episode length over training."))
