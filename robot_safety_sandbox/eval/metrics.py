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
  distance         how far the robot actually GOT: cumulative path length and
                   net displacement per episode. A filter that keeps the robot
                   safe by keeping it still is safe and useless, and this is
                   the axis that says so.

:class:`TrajectoryRecorder` is the odd one out: it reports almost nothing and
instead WRITES the rollout to disk (see its docstring for the two formats).

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


class DistanceTravelled(Metric):
  """How far the robot actually GOT, per episode -- path length and net.

  ``path`` is the cumulative planar path length (sum of per-step ``|dxy|``) and
  ``net`` the straight-line spawn-to-last-state displacement; together they
  separate "walked 10 m forward" from "trotted 10 m in place". A filter that
  buys safety by freezing the robot scores a high ``safe_rate`` and a near-zero
  distance, which is the whole reason this sits next to the safety metrics.

  Episode boundaries (measured, 2026-07-25, see :class:`TrajectoryRecorder`):
  mjlab auto-resets INSIDE ``step``, so on the step whose ``done`` is set the
  robot's ``data`` already holds the NEXT episode's spawn. That step's
  displacement is a teleport (0.86-1.00 m on go2_locomote) and is dropped, and
  the spawn read there is taken as the next episode's origin.
  """

  def __init__(self, num_envs: int, device: str):
    self.path = torch.zeros(num_envs, device=device)
    self.start: Optional[torch.Tensor] = None
    self.prev: Optional[torch.Tensor] = None
    self.done_path: list[torch.Tensor] = []
    self.done_net: list[torch.Tensor] = []

  def update(self, rec: StepRecord) -> None:
    p = rec.env.robot.data.root_link_pos_w[:, :2].detach()
    if self.prev is None:                      # first step of the run
      self.start, self.prev = p.clone(), p.clone()
    d = rec.out.done
    step = (p - self.prev).norm(dim=-1)
    self.path += torch.where(d, torch.zeros_like(step), step)
    if bool(d.any()):
      self.done_path.append(self.path[d].clone())
      self.done_net.append((self.prev - self.start).norm(dim=-1)[d].clone())
      self.path[d] = 0.0
      self.start[d] = p[d]                     # the fresh spawn, read post-reset
    self.prev = p.clone()

  def result(self) -> dict:
    n_live = int((self.path > 0).sum()) if self.prev is not None else 0
    if not self.done_path:
      return dict(dist_episodes=0, dist_censored=n_live,
                  dist_path_mean=float("nan"), dist_path_std=float("nan"),
                  dist_net_mean=float("nan"), dist_net_std=float("nan"))
    path, net = torch.cat(self.done_path), torch.cat(self.done_net)
    sd = lambda v: float(v.std()) if v.numel() > 1 else 0.0  # noqa: E731
    return dict(
      dist_episodes=int(path.numel()),
      dist_censored=n_live,
      dist_path_mean=float(path.mean()), dist_path_std=sd(path),
      dist_net_mean=float(net.mean()), dist_net_std=sd(net),
    )


# --- trajectory recording ----------------------------------------------------

#: Root-state blocks written per step, in order. Joint blocks are appended at
#: construction because their width is the robot's.
_ROOT_BLOCKS = (
  ("root_pos", "root_link_pos_w", 3),
  ("root_quat", "root_link_quat_w", 4),
  ("root_lin_vel", "root_link_lin_vel_w", 3),
  ("root_ang_vel", "root_link_ang_vel_w", 3),
  ("joint_pos", "joint_pos", None),
  ("joint_vel", "joint_vel", None),
)


class TrajectoryRecorder(Metric):
  """Write the rollout itself to disk: state + decision + margins, per episode.

  Every other metric reduces the run to numbers; this one keeps it, so a filter
  comparison can be LOOKED at (coverage hulls, KDE density) and debugged after
  the fact. Recording never touches the rollout -- it reads ``rec`` and the
  robot's ``data``, and writes nothing back.

  **What is captured, per control step, per recorded env**

    state       root xyz, root quaternion, root linear + angular velocity
                (world frame), joint positions, joint velocities
    actions     ``a_nom`` (the task policy's proposal), ``a_dstb`` (the
                adversary's, zeros when there is none), ``a_filt`` (applied)
    decision    ``engaged``, the monitor ``value``, ``caution``
    margins     ``g`` and ``l`` at that step
    flags       ``terminated``, ``truncated``, and the run-global step index

  **Episode alignment -- the thing that is easy to get wrong.** mjlab
  auto-resets INSIDE ``step``, so at metric time on a done step the robot's
  ``data`` already holds the NEXT episode's spawn, while ``g``/``l``/the flags
  still describe the episode that just ended (measured on go2_locomote: at the
  done step ``root_pos`` teleports ~0.9 m and z snaps back to the 0.320 spawn
  height, while ``g`` continues smoothly, 0.500 -> 0.493). The state is
  therefore written one step LATE relative to the scalars: row ``t`` pairs the
  state at the START of step ``t`` with the decision and margins OF step ``t``,
  and a done at ``t`` closes the buffer after row ``t``. Consequences:

    * two episodes are never concatenated, and every episode's first row is its
      true spawn (written by the reset at the previous step);
    * the terminal state itself is unreachable through the ``Metric`` interface
      (it is overwritten before ``update`` is called), so the last row holds the
      last PRE-terminal state together with the terminal margins;
    * the very first control step of a run has no predecessor state and is
      dropped -- one step, once per run, in the first (partial) episode only.

  **On disk** (three files in ``out_dir``):

    ``trajectories.pkl``      ``{"trajectories": [ (T, 3) float32, ... ]}`` --
                              the layout ``plot_trajectory_coverage.py`` reads;
                              columns 0:2 are the XY it consumes, column 2 is z.
                              In the SPAWN frame by default (see ``frame``).
    ``stats.json``            the sidecar that script loads: ``avg_distance``
                              and ``done_types`` (one per trajectory,
                              ``"failure"`` iff the episode ever had ``g < 0``),
                              plus the run's own distance summary.
    ``trajectories_full.npz`` everything above, flat-concatenated over episodes
                              as named arrays (``root_pos``, ``a_nom``, ``g``,
                              ...) with ``episode_ptr`` (E+1 offsets),
                              ``episode_env`` and ``episode_done_type``. Slice
                              episode e as ``arr[ptr[e]:ptr[e + 1]]``.

  :param env: the :class:`~robot_safety_sandbox.eval.envs.EvalEnv` being rolled.
  :param out_dir: directory to write into (created).
  :param traj_envs: record only the FIRST N envs. A full 512x1000 run of every
      field is ~150 MB; the other metrics keep using the whole population, and
      the recorded subset is reported in ``result()`` and in ``stats.json``.
  :param frame: the frame ``trajectories.pkl`` is written in -- ``"spawn"``
      (default) puts every episode's start at the origin with its initial
      heading along +x; ``"world"`` writes raw simulator coordinates. The
      coverage plot's star sits at (0, 0) and its hulls overlay experiments, so
      it needs the spawn frame: mjlab spreads envs across the terrain grid and
      randomizes spawn yaw uniformly (measured on go2_locomote: spawns over
      3 x 15 m, yaw std 92 deg), which turns a world-frame hull into an outline
      of the env grid. In the spawn frame the same episodes concentrate --
      displacement heading std 96 deg -> 24 deg. ``trajectories_full.npz``
      always keeps WORLD coordinates, so nothing is lost either way.
  """

  def __init__(self, env, out_dir: str, *, traj_envs: int = 64,
               frame: str = "spawn"):
    if frame not in ("spawn", "world"):
      raise ValueError(f"frame must be 'spawn' or 'world', got {frame!r}")
    self.frame = frame
    n = min(int(traj_envs), int(env.num_envs))
    if n <= 0:
      raise ValueError(f"traj_envs must be >= 1, got {traj_envs}")
    self.out_dir = out_dir
    self.device = env.device
    self.ids = torch.arange(n, device=env.device)
    self.k = n
    self.dstb_dim = int(getattr(env, "dstb_dim", 0) or 0)
    d = env.robot.data
    self.blocks = [(name, attr, int(w) if w is not None
                    else int(getattr(d, attr).shape[-1]))
                   for name, attr, w in _ROOT_BLOCKS]
    self.blocks += [("a_nom", None, int(env.ctrl_dim)),
                    ("a_filt", None, int(env.ctrl_dim)),
                    ("a_dstb", None, self.dstb_dim),
                    ("engaged", None, 1), ("value", None, 1),
                    ("caution", None, 1), ("g", None, 1), ("l", None, 1),
                    ("terminated", None, 1), ("truncated", None, 1),
                    ("step", None, 1)]
    self._pending: Optional[torch.Tensor] = None
    self._buf: list[list] = [[] for _ in range(n)]
    self._episodes: list = []
    self._ep_env: list[int] = []
    self._ep_done: list[str] = []
    self._col = {}
    c = 0
    for name, _a, w in self.blocks:
      self._col[name], c = c, c + w
    self.width = c

  # --- capture ---------------------------------------------------------------

  def _state(self, env) -> torch.Tensor:
    d = env.robot.data
    return torch.cat([getattr(d, attr)[self.ids].float()
                      for _, attr, _ in self.blocks[:len(_ROOT_BLOCKS)]],
                     dim=-1).detach()

  def _decision(self, rec: StepRecord) -> torch.Tensor:
    ids, k = self.ids, self.k
    col = lambda v: v[ids].reshape(k, 1).float()  # noqa: E731
    zc = torch.zeros(k, 1, device=self.device)
    a_dstb = (rec.a_dstb[ids].float() if rec.a_dstb is not None
              else torch.zeros(k, self.dstb_dim, device=self.device))
    eng = getattr(rec.info, "engaged", None)
    val = getattr(rec.info, "value", None)
    cau = getattr(rec.info, "caution", None)
    o = rec.out
    return torch.cat([
      rec.a_nom[ids].float(), rec.a_filt[ids].float(), a_dstb,
      zc if eng is None else col(eng),
      zc if val is None else col(val),
      zc if cau is None else col(cau),
      col(o.g), col(o.l), col(o.terminated), col(o.truncated),
      torch.full((k, 1), float(rec.t), device=self.device),
    ], dim=-1).detach()

  def update(self, rec: StepRecord) -> None:
    state = self._state(rec.env)
    if self._pending is not None:
      row = torch.cat([self._pending, self._decision(rec)], dim=-1).cpu().numpy()
      for j in range(self.k):
        self._buf[j].append(row[j])          # a view; the block stays alive
      d = rec.out.done[self.ids]
      if bool(d.any()):
        for j in d.nonzero(as_tuple=False).flatten().tolist():
          self._close(int(j), censored=False)
    self._pending = state

  def _close(self, j: int, *, censored: bool) -> None:
    if not self._buf[j]:
      return
    import numpy as np
    ep = np.stack(self._buf[j]).astype(np.float32)
    self._buf[j] = []
    self._episodes.append(ep)
    self._ep_env.append(int(self.ids[j]))
    # 'failure' is the plot script's only special value; it must mean exactly
    # what safe_rate means, so it is "this episode ever left the safe set".
    violated = bool((ep[:, self._col["g"]] < 0.0).any())
    self._ep_done.append("failure" if violated
                         else ("censored" if censored else "timeout"))

  # --- report + write --------------------------------------------------------

  def _plot_xyz(self, ep):
    """(T, 3) xyz for the coverage plot, in ``self.frame``."""
    import numpy as np
    xyz = ep[:, self._col["root_pos"]:self._col["root_pos"] + 3].copy()
    if self.frame == "world" or not len(xyz):
      return xyz
    q = ep[0, self._col["root_quat"]:self._col["root_quat"] + 4]
    w, x, y, z = (float(v) for v in q)      # mujoco order: (w, x, y, z)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    d = xyz[:, 0:2] - xyz[0, 0:2]
    xyz[:, 0] = c * d[:, 0] - s * d[:, 1]
    xyz[:, 1] = s * d[:, 0] + c * d[:, 1]
    return xyz

  def result(self) -> dict:
    import json
    import os
    import pickle

    import numpy as np

    for j in range(self.k):
      self._close(j, censored=True)          # partial episodes are real data
    os.makedirs(self.out_dir, exist_ok=True)
    eps = self._episodes
    xyz = [self._plot_xyz(e) for e in eps]   # cols 0:2 = the XY the plot reads
    p0 = self._col["root_pos"]
    world = [e[:, p0:p0 + 2] for e in eps]   # distances are frame-invariant
    path = [float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum())
            if len(w) > 1 else 0.0 for w in world]
    net = [float(np.linalg.norm(w[-1] - w[0])) if len(w) else 0.0
           for w in world]

    with open(os.path.join(self.out_dir, "trajectories.pkl"), "wb") as f:
      pickle.dump({"trajectories": xyz,
                   "columns": ["x", "y", "z"],
                   "frame": self.frame,
                   "env_ids": self._ep_env}, f)

    n_fail = sum(1 for d in self._ep_done if d == "failure")
    stats = dict(
      avg_distance=float(np.mean(path)) if path else 0.0,
      done_types=self._ep_done,
      n_episodes=len(eps),
      success_rate=1.0 - n_fail / max(len(eps), 1),
      avg_path_length=float(np.mean(path)) if path else 0.0,
      avg_net_displacement=float(np.mean(net)) if net else 0.0,
      recorded_envs=sorted(set(self._ep_env)),
      n_recorded_envs=self.k,
      frame=self.frame,
    )
    with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
      json.dump(stats, f, indent=2)

    flat = (np.concatenate(eps, axis=0) if eps
            else np.zeros((0, self.width), np.float32))
    named = {}
    for name, _attr, w in self.blocks:
      c = self._col[name]
      named[name] = flat[:, c] if w == 1 else flat[:, c:c + w]
    ptr = np.concatenate([[0], np.cumsum([len(e) for e in eps])]).astype(np.int64)
    np.savez_compressed(
      os.path.join(self.out_dir, "trajectories_full.npz"),
      episode_ptr=ptr, episode_env=np.asarray(self._ep_env, np.int32),
      episode_done_type=np.asarray(self._ep_done, dtype="U10"), **named)

    return dict(
      traj_dir=self.out_dir,
      traj_episodes=len(eps),
      traj_envs_recorded=self.k,
      traj_env_ids=f"0..{self.k - 1}",
      traj_steps_total=int(flat.shape[0]),
      traj_frame=self.frame,
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
    DistanceTravelled(env.num_envs, env.device),
  )
