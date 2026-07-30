"""Training INSIDE a safety filter: the env wrapper the PORL experiment needs.

Two things happen here that cannot happen anywhere else in the stack.

**The executed action is the one that gets learned from.** The filter may
replace the policy's proposal with the fallback's action, and it is the
EXECUTED action that must enter the replay buffer -- storing the proposal
against the resulting transition would train the critic on a transition that
never occurred. Off-policy learners make this exactly correct with no
importance correction, which is why PORL is run with SAC. The wrapper publishes
what it executed on ``self.executed_action``; the collector reads it back
(``safety_sb3.sac_base._collect_rollouts_tensor``) instead of trusting the
action it passed in.

**Both arms count failures the same way.** ``filt=None`` gives a pure counting
wrapper, so the bare arm and the filtered arm share one accounting path. If the
control arm counted failures differently from the treatment arm, the headline
plot would be measuring the instrumentation.

The counters are exposed through ``metrics()``, which both trainers already
drain into ``env/*`` logging on a fixed step cadence -- no new callback.

  failures_total       cumulative real terminations (NOT timeouts), all envs
  failures_per_1k      the same, per 1000 env-steps in the window
  failure_rate         fraction of episodes this window ending in failure
  engaged_frac         fraction of env-steps the fallback drove (filtered only)
  margin_mean/_min     the task margin g, for reading engagement against

The filtered arm's ``ep_len_mean`` (logged by the learner, not here) is the
experiment's sanity check: it must sit at the episode limit essentially from
step one. If it does not, the filter is not doing its job and nothing
downstream is worth reading.
"""

from __future__ import annotations

import torch

# obs terms the fallback twin never saw vary, and which therefore have to be
# neutralized before its critic is asked anything. See TwinObsAdapter.
_TWIN_BLIND_TERMS = ("command", "phase")


class TwinObsAdapter:
  """Present the live obs to a twin trained at a DIFFERENT command.

  The fallback here is a stabilizer: it was trained on the same cfg with the
  twist command pinned to zero, so across its whole training set the command
  channel was ``[0, 0, 0]`` and the gait-clock ``phase`` was frozen at
  ``[0, 0]`` (measured, both tasks, 2026-07-25). Feeding it the walker's live
  ``command=[1, 0, 0]`` / ``phase=[0.87, 0.50]`` would ask a critic about a
  region of its input space it has never seen and read the answer as a safety
  certificate.

  Zeroing both restores the twin's own distribution AND states the right
  question: "from this physical state, can the stabilizer still bring the robot
  to a safe stand?" -- which is what a fallback is for. It is emphatically not
  "can it track 1 m/s", which this twin cannot do at all (E053).

  The slices are DERIVED from the observation manager's term table rather than
  hardcoded, so a cfg that reorders or resizes terms cannot silently shift them.
  """

  def __init__(self, env, obs_key: str, blind_terms=_TWIN_BLIND_TERMS):
    om = env.mj.observation_manager
    names = list(om.active_terms[obs_key])
    dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim[obs_key]]
    self.slices, off = [], 0
    for name, width in zip(names, dims):
      if name in blind_terms:
        self.slices.append((off, off + width))
      off += width
    missing = [t for t in blind_terms if t not in names]
    if missing:
      raise ValueError(
        f"obs group '{obs_key}' has no term(s) {missing}; the twin adapter was "
        f"written for the velocity cfg's layout {names}. Update blind_terms "
        "rather than letting the twin see a channel it never trained on.")
    self.total = off

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    out = obs.clone()
    for a, b in self.slices:
      out[:, a:b] = 0.0
    return out

  def __repr__(self) -> str:
    return (f"TwinObsAdapter(zeroing {self.slices} of {self.total} dims)")


class FilteredTensorEnv:
  """A ``MjlabTensorSafetyEnv`` stepped through a safety filter.

  Delegates everything it does not own, so it substitutes for the bridge
  anywhere a tensor env is expected.

  :param env: the bridge, built with ``dense_margins=True`` (the filter needs
      the task margins and so do the counters).
  :param filt: a :class:`~robot_safety_sandbox.filters.SafetyFilter`, or None
      for the CONTROL arm -- the wrapper then only counts, and every transition
      is the policy's own action.
  :param norm: the twin's frozen obs normalizer (from ``load_twin``); required
      when ``filt`` is given.
  :param obs_adapter: raw obs -> the obs the twin should see, applied BEFORE
      ``norm``. Defaults to :class:`TwinObsAdapter`.
  """

  def __init__(self, env, filt=None, *, norm=None, obs_adapter=None):
    if filt is not None and norm is None:
      raise ValueError("a filtered env needs the twin's obs normalizer (norm=)")
    if not getattr(env, "dense_margins", False):
      raise ValueError(
        "FilteredTensorEnv needs an env built with dense_margins=True: the "
        "monitor and the failure counters both read the task margins, and a "
        "plain cumulative env computes none.")
    self.env, self.filt, self.norm = env, filt, norm
    self.obs_adapter = (
      obs_adapter if obs_adapter is not None
      else TwinObsAdapter(env, env.obs_key)) if filt is not None else None
    self.num_envs = int(env.num_envs)
    self.device = str(env.mj.device)
    self.executed_action: torch.Tensor | None = None
    self._fresh = torch.ones(self.num_envs, dtype=torch.bool,
                             device=self.device)
    self._failures_total = 0
    self._zero()
    if filt is not None:
      print(f"[porl] filtered env: {type(filt.intervention).__name__} over "
            f"{type(filt.monitor).__name__}; twin obs via {self.obs_adapter}")
    else:
      print("[porl] CONTROL env: no filter, counters only")

  # --- delegation ------------------------------------------------------------

  def __getattr__(self, name):
    # __getattr__ runs only when normal lookup fails, INCLUDING during __init__
    # before self.env exists -- so miss cleanly instead of raising KeyError.
    try:
      env = self.__dict__["env"]
    except KeyError:
      raise AttributeError(name) from None
    return getattr(env, name)

  # --- counters --------------------------------------------------------------

  def _zero(self) -> None:
    """Reset the WINDOW counters. ``_failures_total`` is cumulative over the
    whole run (it is the headline plot's y axis) and is never zeroed here."""
    self._steps = 0
    self._failures = 0
    self._episodes = 0
    self._engaged = 0.0
    self._g_sum = 0.0
    self._g_min = float("inf")

  def _twin_obs(self, raw: torch.Tensor) -> torch.Tensor:
    return self.norm(self.obs_adapter(raw.float()))

  # --- the env API -----------------------------------------------------------

  def reset(self) -> torch.Tensor:
    obs = self.env.reset()
    self._fresh = torch.ones(self.num_envs, dtype=torch.bool,
                             device=self.device)
    if self.filt is not None:
      self.filt.reset(self._fresh)
    return obs

  def step_tensor(self, a_nom: torch.Tensor):
    if self.filt is None:
      action = a_nom
    else:
      raw = self.env.obs_groups()[self.env.obs_key]
      speed = torch.norm(
        self.env.mj.scene["robot"].data.root_link_lin_vel_w[:, :2], dim=1)
      action, info = self.filt(a_nom, speed=speed, fresh=self._fresh,
                               s_obs=self._twin_obs(raw))
      self._engaged += float(info.engaged.float().sum())
    # THE contract: what was executed is what the learner must store.
    self.executed_action = action

    obs, reward, dones, timeouts, l = self.env.step_tensor(action)

    g, _ = self.env.safety_margins()
    failed = dones.bool() & ~timeouts.bool()
    n_failed = int(failed.sum())
    self._steps += self.num_envs
    self._failures += n_failed
    self._failures_total += n_failed
    self._episodes += int(dones.sum())
    self._g_sum += float(g.sum())
    self._g_min = min(self._g_min, float(g.min()))

    self._fresh = dones.bool()
    if self.filt is not None:
      self.filt.reset(dones.bool())
    return obs, reward, dones, timeouts, l

  def metrics(self) -> dict[str, float]:
    """Drain the window's counters, merged over the bridge's own metrics."""
    out = dict(self.env.metrics() or {})
    out["porl/failures_total"] = float(self._failures_total)
    if self._steps:
      out["porl/failures_per_1k"] = 1000.0 * self._failures / self._steps
      out["porl/failure_rate"] = (self._failures / self._episodes
                                  if self._episodes else 0.0)
      out["porl/episodes"] = float(self._episodes)
      out["porl/margin_mean"] = self._g_sum / self._steps
      out["porl/margin_min"] = self._g_min
      if self.filt is not None:
        out["porl/engaged_frac"] = self._engaged / self._steps
    self._zero()
    return out

  def close(self) -> None:
    self.env.close()
