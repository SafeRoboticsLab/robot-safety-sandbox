"""What an evaluation MEASURES -- the protocol set, plus per-task extras.

The core set is the CBF-DDP filter-comparison protocol (Hu et al., "Deployable
Reachability-Guaranteed Safety Filters"-style reporting): a filter is judged on
whether the task still gets done, whether safety held, how HARSH the filter was
on the hardware, how MUCH of the nominal it overwrote, and what it cost per
control step.

  task_success     did the episode reach the task's target set safely (l >= 0
                   while g >= 0)? The definition comes from the TASK's own
                   margins, so it means in eval what it meant in training.
  safe_rate        did the episode avoid the failure set entirely (g >= 0
                   throughout)?
  jerk             per-actuator |d(joint acceleration)/dt|, mean +- std --
                   a filter that is safe by slamming the actuators is not
                   deployable, and this is where that shows.
  intervention     ||pi_task - pi_filtered||_1, total and per step: HOW MUCH
                   nominal behavior was overwritten, not just how often.
  engagement       fraction of env-steps the filter was engaged / in caution
                   (the only metric the old gauntlet had).
  step_ms          wall clock per control step, split filter / env, so the
                   price of a shadow rollout is visible next to its benefit.

Everything here is task-agnostic. A task that has its own reading of "did it
work" (the gap gauntlet's crossing and livelock rates) registers an extra
metric through an :class:`~robot_safety_sandbox.eval.presets.EvalPreset`;
nothing task-shaped is hardcoded below.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class StepRecord:
  """Everything one control step exposes to a metric."""
  t: int
  env: object                    # EvalEnv
  a_nom: torch.Tensor            # (N, ctrl) the nominal's action
  a_filt: torch.Tensor           # (N, ctrl) what was actually applied
  info: object                   # FilterInfo (engaged / value / caution)
  out: object                    # StepOut (g, l, terminated, truncated)
  a_dstb: Optional[torch.Tensor] = None
  filter_s: float = 0.0          # wall clock spent deciding
  env_s: float = 0.0             # wall clock spent stepping the sim


class Metric:
  """Accumulate over a run, then report. Stateless between runs.

  ``update`` is called once per control step with the full
  :class:`StepRecord`; ``result`` returns a flat dict of JSON-serializable
  values. Metrics never influence the rollout.
  """

  def update(self, rec: StepRecord) -> None:
    raise NotImplementedError

  def result(self) -> dict:
    raise NotImplementedError


class MetricSet(Metric):
  """A list of metrics driven as one."""

  def __init__(self, *metrics: Metric):
    self.metrics = [m for m in metrics if m is not None]

  def update(self, rec: StepRecord) -> None:
    for m in self.metrics:
      m.update(rec)

  def result(self) -> dict:
    out: dict = {}
    for m in self.metrics:
      out.update(m.result())
    return out


# --- the protocol set --------------------------------------------------------

class EpisodeOutcomes(Metric):
  """Task success and safety, per EPISODE, from the task's own (g, l).

  Episodes still running when the window closes are CENSORED, not dropped:
  counting only completed episodes silently inflates every rate (a filter whose
  whole point is to keep robots alive completes the fewest episodes). They are
  reported separately and included in the denominator.
  """

  def __init__(self, num_envs: int, device: str):
    z = lambda dt: torch.zeros(num_envs, dtype=dt, device=device)  # noqa: E731
    self.reached, self.violated = z(torch.bool), z(torch.bool)
    self.steps = z(torch.float32)
    self.n_env = num_envs
    self.episodes = self.successes = self.violations = 0
    self.timeouts = self.terminations = 0
    self.lens: list[torch.Tensor] = []
    self._live_reached = self._live_violated = None

  def update(self, rec: StepRecord) -> None:
    o = rec.out
    self.reached |= (o.l >= 0.0) & (o.g >= 0.0)
    self.violated |= o.g < 0.0
    self.steps += 1.0
    d = o.done
    if bool(d.any()):
      self.episodes += int(d.sum())
      self.successes += int((self.reached & d).sum())
      self.violations += int((self.violated & d).sum())
      self.timeouts += int((o.truncated & ~o.terminated & d).sum())
      self.terminations += int((o.terminated & d).sum())
      self.lens.append(self.steps[d].clone())
      self.reached &= ~d
      self.violated &= ~d
      self.steps[d] = 0.0
    self._live_reached, self._live_violated = self.reached, self.violated

  def result(self) -> dict:
    censored = self.steps > 0
    n_cens = int(censored.sum())
    denom = max(self.episodes + n_cens, 1)
    lens = torch.cat(self.lens) if self.lens else torch.zeros(1)
    return dict(
      episodes=self.episodes,
      censored_alive=n_cens,
      task_success=(self.successes + int((self.reached & censored).sum()))
      / denom,
      safe_rate=1.0 - (self.violations + int((self.violated & censored).sum()))
      / denom,
      violation_rate=(self.violations
                      + int((self.violated & censored).sum())) / denom,
      timeout_rate=self.timeouts / denom,
      # episodes ended by a non-timeout DoneTerm. Usually the failure set (and
      # then == violation_rate), but under end_criterion="reach-avoid" the
      # success term terminates too, so the two are reported separately.
      termination_rate=self.terminations / denom,
      ep_len_mean=float(lens.mean()),
    )


class MarginStats(Metric):
  """The distribution of the WORST safety margin reached per episode.

  Rates say how often something failed; this says how CLOSE the near-misses
  came, which is the readout that makes an adversary's effect visible on a task
  whose behavior looks like nothing (a robot standing still): the game lives in
  the margin statistics. ``min_g_p10`` is the tail that moves first.
  """

  def __init__(self, num_envs: int, device: str):
    self.big = 1e9
    self.min_g = torch.full((num_envs,), self.big, device=device)
    self.done_min_g: list[torch.Tensor] = []

  def update(self, rec: StepRecord) -> None:
    self.min_g = torch.minimum(self.min_g, rec.out.g)
    d = rec.out.done
    if bool(d.any()):
      self.done_min_g.append(self.min_g[d].clone())
      self.min_g[d] = self.big

  def result(self) -> dict:
    alive = self.min_g < self.big
    parts = list(self.done_min_g) + ([self.min_g[alive]] if bool(alive.any())
                                     else [])
    if not parts:
      return dict(min_g_mean=float("nan"), min_g_p10=float("nan"))
    v = torch.cat(parts)
    return dict(min_g_mean=float(v.mean()),
                min_g_p10=float(v.quantile(0.10)),
                min_g_worst=float(v.min()))


class ActuatorJerk(Metric):
  """Per-actuator jerk from the SIMULATED joint accelerations.

  Jerk is measured on the robot, not on the action stream: the action is a
  position target and its finite differences confound the controller's own
  filtering with what the joints actually did. j_t = |acc_t - acc_{t-1}| / dt,
  per actuator, accumulated as mean and std over all (step, env) samples via
  running sums (a full history over N envs x T steps x 12 joints is not worth
  holding).

  The first step of each episode is skipped for envs that just reset: a reset
  teleports the state and its acceleration difference is not a control action.
  """

  def __init__(self, dt: float):
    self.dt = float(dt)
    self.prev: Optional[torch.Tensor] = None
    self.n = 0
    self.sum: Optional[torch.Tensor] = None
    self.sq: Optional[torch.Tensor] = None
    self._skip: Optional[torch.Tensor] = None

  def update(self, rec: StepRecord) -> None:
    acc = rec.env.robot.data.joint_acc.detach()
    if self.prev is not None:
      j = (acc - self.prev).abs() / self.dt
      keep = ~self._skip if self._skip is not None else None
      if keep is not None and not bool(keep.all()):
        j = j[keep]
      if j.numel():
        s, q = j.sum(dim=0), (j * j).sum(dim=0)
        self.sum = s if self.sum is None else self.sum + s
        self.sq = q if self.sq is None else self.sq + q
        self.n += int(j.shape[0])
    self.prev = acc.clone()
    self._skip = rec.out.done.clone()      # next step's diff crosses a reset

  def result(self) -> dict:
    if not self.n or self.sum is None:
      return dict(jerk_mean=0.0, jerk_std=0.0, jerk_per_actuator_mean=[],
                  jerk_per_actuator_std=[])
    mean = self.sum / self.n
    var = (self.sq / self.n - mean * mean).clamp_min(0.0)
    std = var.sqrt()
    return dict(
      jerk_mean=float(mean.mean()),
      jerk_std=float(std.mean()),
      jerk_max_actuator=float(mean.max()),
      jerk_per_actuator_mean=[round(float(v), 4) for v in mean],
      jerk_per_actuator_std=[round(float(v), 4) for v in std],
    )


class InterventionMass(Metric):
  """||pi_task - pi_filtered||_1 -- how much nominal behavior was overwritten.

  Engagement RATE says how often the filter acted; this says how far it moved
  the action when it did, which is the axis a minimal-modification filter
  (qcbf) is supposed to win on and a switching filter is not.
  """

  def __init__(self):
    self.total = 0.0
    self.samples = 0
    self.engaged_total = 0.0
    self.engaged_samples = 0

  def update(self, rec: StepRecord) -> None:
    d = (rec.a_filt - rec.a_nom).abs().sum(dim=-1)
    self.total += float(d.sum())
    self.samples += int(d.numel())
    eng = getattr(rec.info, "engaged", None)
    if eng is not None and bool(eng.any()):
      self.engaged_total += float(d[eng].sum())
      self.engaged_samples += int(eng.sum())

  def result(self) -> dict:
    return dict(
      intervention_mass_total=self.total,
      intervention_mass_per_step=self.total / max(self.samples, 1),
      intervention_mass_when_engaged=(self.engaged_total
                                      / max(self.engaged_samples, 1)),
    )


class WallClock(Metric):
  """Per-control-step cost, split into deciding and simulating."""

  def __init__(self):
    self.filter_s = self.env_s = 0.0
    self.steps = 0

  def update(self, rec: StepRecord) -> None:
    self.filter_s += rec.filter_s
    self.env_s += rec.env_s
    self.steps += 1

  def result(self) -> dict:
    n = max(self.steps, 1)
    return dict(
      filter_ms_per_step=1e3 * self.filter_s / n,
      env_ms_per_step=1e3 * self.env_s / n,
      total_ms_per_step=1e3 * (self.filter_s + self.env_s) / n,
    )


class Engagement(Metric):
  """Filter engagement and caution rates, read off the filter's telemetry."""

  def __init__(self, filt):
    self.filt = filt
    self.steps = 0
    self.value_sum = 0.0

  def update(self, rec: StepRecord) -> None:
    self.steps += 1
    v = getattr(rec.info, "value", None)
    if v is not None:
      self.value_sum += float(v.mean())

  def result(self) -> dict:
    tel = self.filt.telemetry
    return dict(
      intervention_rate=tel.intervention_rate(self.steps),
      caution_rate=tel.caution_rate(self.steps),
      monitor_value_mean=self.value_sum / max(self.steps, 1),
    )


def protocol_metrics(env, filt, *, dt: Optional[float] = None) -> MetricSet:
  """The CBF-DDP core set, wired to one env + filter."""
  step_dt = float(dt if dt is not None else env.mj.step_dt)
  return MetricSet(
    EpisodeOutcomes(env.num_envs, env.device),
    MarginStats(env.num_envs, env.device),
    ActuatorJerk(step_dt),
    InterventionMass(),
    WallClock(),
    Engagement(filt),
  )
