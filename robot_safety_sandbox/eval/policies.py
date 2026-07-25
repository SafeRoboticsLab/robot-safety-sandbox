"""Loading the two policies an evaluation composes, and nothing else.

An evaluation needs a NOMINAL policy (pi_task -- the thing being filtered) and
a SAFETY TWIN (the certificate + fallback the filter is built from). They live
in different serialization worlds and this module is the only place that knows
it:

  nominal  stock ``stable_baselines3`` zip + a numpy ``VecNormalize`` pickle
           (mode="cumulative" tasks train with stock SB3 on purpose, so the
           checkpoint stays loadable without safety_sb3 -- see train_on_policy)
  twin     a safety_sb3 learner zip -- any of the eight MAP cells -- + a torch
           ``tensornormalize.pt`` of obs statistics

:func:`safety_modules` then exposes ONE loaded twin as the callables the filter
compositions consume (fallback / value_fn / q_fn / dstb_fn), keyed off what the
twin actually has, so a caller asking for something an on-policy 1P twin cannot
supply gets a named error instead of a silently wrong number.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import re
import zipfile

import torch


# --- obs-normalizer discovery ------------------------------------------------

def find_obs_stats(zip_path: str, prefix: str, flat: str) -> str | None:
  """The obs-normalization stats file that belongs to ``zip_path``.

  Finals save ``<flat>`` next to the model; periodic checkpoints save
  ``<prefix>_<step>.<ext>`` in the same directory, so pick the one whose STEP
  is closest to the model's -- taking the newest (what the old gauntlet did)
  pairs a mid-training checkpoint with end-of-training statistics.
  """
  d = os.path.dirname(os.path.abspath(zip_path))
  cand_flat = os.path.join(d, flat)
  if os.path.exists(cand_flat):
    return cand_flat
  cands = sorted(glob.glob(os.path.join(d, prefix + "*")))
  if not cands:
    return None
  m = re.search(r"model_(\d+)_steps", os.path.basename(zip_path))
  if m is None:
    return cands[-1]
  step = int(m.group(1))

  def dist(c):
    n = re.search(r"(\d+)", os.path.basename(c)[len(prefix):])
    return abs(int(n.group(1)) - step) if n else 1 << 62

  return min(cands, key=dist)


# --- the nominal policy (pi_task) --------------------------------------------

def load_nominal(zip_path: str, device: str, quiet: bool = False):
  """A stock SB3 policy + its numpy VecNormalize stats (or None).

  Returns ``(model, vecnormalize_or_None)``. The normalizer is put in
  inference mode; a run directory without one is legal (the nominal simply
  consumes raw observations) and is reported rather than guessed at.
  """
  from stable_baselines3 import PPO
  model = PPO.load(zip_path, device=device)
  vn_path = find_obs_stats(zip_path, "vecnormalize", "vecnormalize.pkl")
  vn = None
  if vn_path:
    with open(vn_path, "rb") as f:
      vn = pickle.load(f)
    vn.training = False
  if not quiet:
    print(f"[nominal] {zip_path} "
          f"(+ {os.path.basename(vn_path) if vn_path else 'NO NORM'})")
  return model, vn


# --- the safety twin ---------------------------------------------------------

def _serialized_class_name(entry) -> str:
  """Class NAME out of one SB3-serialized ``data`` entry (base64 cloudpickle)."""
  if not isinstance(entry, dict) or ":serialized:" not in entry:
    return ""
  try:
    import base64
    import cloudpickle
    obj = cloudpickle.loads(base64.b64decode(entry[":serialized:"]))
    return getattr(obj, "__name__", "")
  except Exception:
    return str(entry.get("__module__", ""))


def _zip_data(zip_path: str) -> dict:
  with zipfile.ZipFile(zip_path) as z:
    return json.loads(z.read("data"))


def twin_class_name(zip_path: str) -> str:
  """Read the MAP cell (Mode + Algorithm + Players) out of the checkpoint.

  The learner class is not stored by SB3, but each of the three letters leaves a
  fingerprint that is, so the name is RECONSTRUCTED rather than guessed:

    A  SAC saves ``actor.optimizer.pth`` (PPO saves ``policy.optimizer.pth``)
    P  a two-player twin saves the disturbance actor's optimizer, and its data
       carries ``ctrl_action_dim`` (where the joint action splits)
    M  the buffer class: {ReachAvoid,Safety}{ReplayBuffer,RolloutBuffer}

  Returns "" when nothing safety_sb3-shaped is found (e.g. a stock SB3 zip), so
  the caller can fall back to trying candidates in order.

  M is the LEAST reliable letter: the buffer class is a cloudpickle reference
  to a module path, so a checkpoint written before a rename (v0.4.0 moved
  ``safety_sb3.tensor_buffers`` / ``isaacs_buffers``) cannot be unpickled and
  the mode reads as empty. Rather than silently defaulting to "Safety", the
  name is returned WITHOUT a mode prefix in that case, which makes
  :func:`load_twin` fall through to its ordered candidate list instead of
  confidently loading a reach-avoid twin as an avoid one.
  """
  with zipfile.ZipFile(zip_path) as z:
    names = set(z.namelist())
  data = _zip_data(zip_path)
  sac = any(n.startswith("actor.optimizer") for n in names)
  alg = "SAC" if sac else "PPO"
  two = ("ctrl_action_dim" in data or any(n.startswith("dstb") for n in names))
  buf = _serialized_class_name(
    data.get("replay_buffer_class" if sac else "rollout_buffer_class"))
  if not sac and not buf.startswith(("ReachAvoid", "Safety", "Tensor")):
    return ""                          # stock stable_baselines3 checkpoint
  if not buf.startswith(("ReachAvoid", "Safety")):
    return ""                          # mode unreadable -> do not guess it
  mode = "ReachAvoid" if "ReachAvoid" in buf else "Safety"
  return f"{mode}{alg}{'2P' if two else '1P'}"


#: candidate learner classes, tried in order when the checkpoint's own MAP cell
#: cannot be reconstructed (pre-rename checkpoints, hand-made zips).
TWIN_CANDIDATES = (
  "ReachAvoidPPO1P", "SafetyPPO1P", "ReachAvoidSAC1P", "SafetySAC1P",
  "ReachAvoidPPO2P", "SafetyPPO2P", "ReachAvoidSAC2P", "SafetySAC2P")


def load_twin(zip_path: str, device: str, quiet: bool = False):
  """Load a safety twin -- any of the eight MAP cells -- plus its obs normalizer.

  Returns ``(model, norm)`` where ``norm(raw_obs) -> normalized_obs`` applies
  the twin's own frozen statistics (clipped to +-10, the training convention).

  The class is resolved from the checkpoint (:func:`twin_class_name`) with an
  ordered candidate chain as the fallback.
  """
  import safety_sb3
  guess = twin_class_name(zip_path)
  data = _zip_data(zip_path)
  candidates = ([guess] if guess else []) + list(TWIN_CANDIDATES)
  # Two-player learners take the joint-action split as a CONSTRUCTOR argument
  # (SB3 restores it into __dict__ too late for _setup_model), and the SAC
  # learners' tensor path builds a device-resident replay buffer sized off
  # self.env -- which a checkpoint loaded for INFERENCE does not have. Override
  # both: no env, no buffer, actors and critic only.
  # ``use_leaderboard`` goes the same way, and for a third reason: the league's
  # ``model_dir`` is serialized as an ABSOLUTE path, so _setup_model tries to
  # mkdir the TRAINING HOST's run directory and a checkpoint copied off that
  # host dies with PermissionError on someone else's /home. Nothing in an
  # evaluation reads the board.
  ctrl_dim = data.get("ctrl_action_dim")
  model, errors = None, []
  for name in dict.fromkeys(candidates):
    cls = getattr(safety_sb3, name, None)
    if cls is None:
      continue
    kw = {"ctrl_action_dim": int(ctrl_dim)} if (
      "2P" in name and ctrl_dim is not None) else {}
    custom = ({"_tensor_path": False, "buffer_size": 1,
               "use_leaderboard": False} if "SAC" in name else None)
    try:
      model = cls.load(zip_path, device=device, custom_objects=custom, **kw)
      break
    except Exception as e:                       # noqa: BLE001 - report them all
      errors.append(f"{name}: {type(e).__name__}: {e}")
  if model is None:
    raise SystemExit(f"could not load safety twin {zip_path}:\n  "
                     + "\n  ".join(errors))
  pt = find_obs_stats(zip_path, "tensornorm", "tensornormalize.pt")
  if not pt:
    raise SystemExit(
      f"safety obs-norm stats not found next to {zip_path} "
      "(expected tensornormalize.pt, or tensornorm_<step>.pt for a checkpoint)")
  st = torch.load(pt, map_location=device, weights_only=True)
  mean, var = st["obs_mean"].to(device), st["obs_var"].to(device)
  if not quiet:
    print(f"[twin] {zip_path} ({type(model).__name__}) "
          f"+ {os.path.basename(pt)}")

  def norm(obs):
    return torch.clamp((obs - mean) / torch.sqrt(var + 1e-8), -10.0, 10.0)

  return model, norm


def safety_modules(model, num_envs: int, device: str) -> dict:
  """Wire a loaded twin into the callables/modules the filter recipes take.

  Every key is keyed off what the twin ACTUALLY has, so an on-policy twin
  simply has no ``q_fn`` and a single-player one no ``dstb_fn`` -- a caller that
  needs one gets a named error, not a silent wrong number.

    fallback   PolicyFallback over pi^<        every twin
    value_fn   V(s)                            *PPO* directly; *SAC* as
                                               Q(s, pi^<(s)) -- safe iff >= 0
    q_fn       Q(s, a), DIFFERENTIABLE in a    *SAC* only (CriticMonitor,
                                               QCBFIntervention)
    dstb_fn    pi_dstb(s)                      *2P* only (AdversarialRollout-
                                               Monitor's disturbance player,
                                               and the LIVE adversary channel)

  All of them take the NORMALIZED observation as ``s_obs`` and tolerate extra
  ctx kwargs, matching the ``**ctx`` convention in robot_safety_sandbox.filters.
  The twin's own bounds are respected: actions are clamped to [-1, 1], and the
  twin critic is reduced by MIN across the ensemble, which is the conservative
  reading under the zoo's "safe iff >= 0" convention (and the same reduction
  the SAC learners use to form their targets).
  """
  from ..filters import PolicyFallback
  policy = model.policy
  policy.set_training_mode(False)
  out: dict = {}

  if hasattr(policy, "predict_values"):                     # on-policy twin
    def value_fn(s_obs, **_):
      with torch.no_grad():
        return policy.predict_values(s_obs).squeeze(-1)
    out["value_fn"] = value_fn

    def fallback_fn(s_obs, **_):
      with torch.no_grad():
        return torch.clamp(policy._predict(s_obs, deterministic=True), -1., 1.)

    # A 2P PPO twin keeps its disturbance player on the LEARNER, not the policy
    # (two independent policies, unlike the SAC twins' one joint policy).
    dstb_policy = getattr(model, "dstb_policy", None)
    if dstb_policy is not None:
      def dstb_fn(s_obs, **_):
        with torch.no_grad():
          return torch.clamp(dstb_policy._predict(s_obs, deterministic=True),
                             -1., 1.)
      out["dstb_fn"] = dstb_fn
  else:                                                     # off-policy twin
    def fallback_fn(s_obs, **_):
      with torch.no_grad():
        return torch.clamp(policy.actor(s_obs, deterministic=True), -1., 1.)

    dstb_actor = getattr(policy, "dstb_actor", None)

    def _joint(action, s_obs):
      """The critic's action argument. A 2P twin's critic is over the FULL
      concatenated [ctrl, dstb] action (TwoPlayerSACPolicy), so the ctrl action
      alone does not index it: append the disturbance the adversary would play.
      pi_dstb depends on s only, so d(Q)/d(a_ctrl) is unaffected."""
      if dstb_actor is None:
        return action
      return torch.cat([action, dstb_actor(s_obs, deterministic=True)], dim=-1)

    def q_fn(action, s_obs, **_):
      qs = policy.critic(s_obs, _joint(action, s_obs))
      return torch.cat(qs, dim=1).min(dim=1).values
    out["q_fn"] = q_fn

    def value_fn(s_obs, **_):
      with torch.no_grad():
        return q_fn(action=fallback_fn(s_obs), s_obs=s_obs)
    out["value_fn"] = value_fn

    if dstb_actor is not None:
      def dstb_fn(s_obs, **_):
        with torch.no_grad():
          return torch.clamp(dstb_actor(s_obs, deterministic=True), -1., 1.)
      out["dstb_fn"] = dstb_fn

  out["fallback_fn"] = fallback_fn
  out["fallback"] = PolicyFallback(num_envs, device, fallback_fn)
  out["twin"] = type(model).__name__
  return out
