"""E037 -- toy gap: a ground-truthable double integrator with the Go2-gap topology.

Purpose: settle, for ~0 GPU, whether the CORRECTED reach-avoid anchor actually
fixes the risk-aversion trap -- before spending days of GPU on the Go2 split
test (E040/E041). If the corrected operator still loiters HERE, the campaign's
blocker is exploration, not semantics, and the Go2 reruns would tell us nothing.

The topology mirrors the gap task exactly:

    [ safe runway: can stop forever ][ PIT: fatal below transit speed ][ target ]
     x < 0                            0 <= x <= W                       x > W+0.2

  * loitering on the runway (v=0) is SAFE FOREVER (g=+3) but off-target (l=-1)
    -- this is the decision state, and the whole campaign's failure mode
  * crossing needs speed v > V_MIN while over the pit, then a stop in the target

Margins use the CAMPAIGN'S clamps (g +/-3, l +/-1), so the risk dial matches:
    p*_new   = (|g_fail| - |l_stop|)+ / (l+ + |g_fail|) = (3-1)/(1+3) = 50%
    p*_buggy = (|g_fail| + g_stop)   / (l+ + |g_fail|) = (3+3)/(1+3) = 150%  (never attempt)

Ground truth is TABULAR value iteration with the same discounted RA operator on
a (x, v) grid -- where the contraction guarantees actually hold. No odp needed.

Deliverables (professor consult, 2026-07-17):
  (i)   corrected ReachAvoidPPO recovers the tabular RA set
  (ii)  buggy vs corrected value on loiter states: V=g>0 vs V=l<0
  (iii) does corrected synthesis INITIATE from standstill where buggy loiters
  (iv)  empirical p* sweep across the p*=0 boundary
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch as th
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from safety_sb3 import ReachAvoidPPO, SafetyPPO
from safety_sb3.safety_buffers import ReachAvoidRolloutBuffer

# --- world ------------------------------------------------------------------
DT, ACC = 0.05, 3.0          # v += ACC * a * DT, a in [-1, 1]
GAP_X0 = 0.0
V_MIN = 1.0                  # min speed to survive the pit
CLAMP_G, CLAMP_L = 3.0, 1.0  # the campaign's clamps
#: `l` must stay informative across the runway (see margins()). The runway is
#: ~2 m, so L_SCALE=2.0 spans it: l runs -1 at x=-1.5 up to +0.25 past the gap,
#: clamping only behind the start.
L_SCALE = 2.0
TIMEOUT = 200                # 10 s


def margins(x, v, width):
  """(g, l) -- g >= 0 safe, l >= 0 in target. Same convention as the zoo.

  `l` is POSITION-ONLY and scaled to stay informative across the whole runway
  (L_SCALE spans it). Two earlier mistakes, both fixed here, both worth naming
  because they produce a fake "the corrected anchor still loiters" result:

  1. `l = (x - t_lo)/0.3` clamps to -1 everywhere on the runway -> the reach
     term is FLAT -> no gradient -> PPO never crosses -> V == l == -1, which is
     the correct value of a policy that never reaches. That is an EXPLORATION
     failure wearing the costume of a semantics failure -- exactly the misread
     that produced E029.
  2. Conjoining a velocity condition (`min(pos, (V_TOL-|v|)/0.25)`) pinned l at
     -1 during ANY crossing, so the target was only reachable by being past the
     gap AND nearly stopped at once -- a conjunction exploration never hits.

  The park-at-the-far-end requirement is the campaign's real target, but it adds
  an exploration problem ORTHOGONAL to the question E037 exists to answer
  ("does the corrected anchor initiate from a standstill?"). Keep it out.
  """
  in_pit = (x > GAP_X0) & (x < GAP_X0 + width)
  g = np.where(in_pit, (v - V_MIN) / 0.3, 3.0)
  t_lo = GAP_X0 + width + 0.2
  l = (x - t_lo) / L_SCALE
  return (np.clip(g, -CLAMP_G, CLAMP_G).astype(np.float32),
          np.clip(l, -CLAMP_L, CLAMP_L).astype(np.float32))


class ToyGap(gym.Env):
  observation_space = gym.spaces.Box(-10, 10, (2,), dtype=np.float32)
  action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

  def __init__(self, width=0.30, x_lo=-2.0, x_hi=-0.5):
    self.width, self.x_lo, self.x_hi = width, x_lo, x_hi

  def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self.x = float(self.np_random.uniform(self.x_lo, self.x_hi))
    self.v = 0.0                       # STANDSTILL: the initiation question
    self.t = 0
    return np.array([self.x, self.v], np.float32), {}

  def step(self, action):
    a = float(np.clip(action[0], -1, 1))
    self.x += self.v * DT
    self.v += ACC * a * DT
    self.t += 1
    g, l = margins(np.array(self.x), np.array(self.v), self.width)
    obs = np.array([self.x, self.v], np.float32)
    return obs, float(g), bool(g < 0), self.t >= TIMEOUT, {"l_x": float(l)}


# --- the <= v0.1.0 bug, reproduced for contrast -----------------------------
class BuggyRABuffer(ReachAvoidRolloutBuffer):
  """The g-anchored backup exactly as it was before v0.2.0."""

  def _target(self, step, v_next, not_done):
    g_t, l_t = self.rewards[step], self.l_x[step]
    v_to_go = np.minimum(g_t, np.maximum(l_t, v_next))
    return (1.0 - self.gamma * not_done) * g_t + self.gamma * not_done * v_to_go


class BuggyRAPPO(ReachAvoidPPO):
  numpy_rollout_buffer_class = BuggyRABuffer


# --- ground truth: tabular VI with the SAME discounted RA operator ----------
def tabular_ra(width, gamma=0.99, nx=201, nv=161, n_act=9, iters=600,
               device="cpu"):
  """Exact-ish RA value on a grid. V = (1-g)min(l,g) + g*max_a min(g,max(l,V')).

  Returns (V, xs, vs). The RA set is {V >= 0}.
  """
  xs = th.linspace(-2.5, GAP_X0 + width + 1.4, nx, device=device)
  vs = th.linspace(-1.5, 4.0, nv, device=device)
  X, V_ = th.meshgrid(xs, vs, indexing="ij")
  g_np, l_np = margins(X.cpu().numpy(), V_.cpu().numpy(), width)
  g = th.tensor(g_np, device=device)
  l = th.tensor(l_np, device=device)
  acts = th.linspace(-1, 1, n_act, device=device)

  # next state per action (semi-implicit, same as the env). x does not depend
  # on the action this step, so it must be broadcast out to n_act explicitly.
  nxt_x = (X.unsqueeze(-1) + V_.unsqueeze(-1) * DT).expand(-1, -1, n_act)
  nxt_v = (V_.unsqueeze(-1) + ACC * acts.view(1, 1, -1) * DT).expand(nx, nv, n_act)
  # -> normalized grid coords for grid_sample (align_corners=True)
  gx = 2 * (nxt_x - xs[0]) / (xs[-1] - xs[0]) - 1
  gv = 2 * (nxt_v - vs[0]) / (vs[-1] - vs[0]) - 1
  grid = th.stack([gv, gx], dim=-1).view(1, nx, nv * n_act, 2).clamp(-1, 1)

  V = th.minimum(l, g).clone()
  anchor = (1 - gamma) * th.minimum(l, g)
  for _ in range(iters):
    Vn = th.nn.functional.grid_sample(
      V.view(1, 1, nx, nv), grid, mode="bilinear",
      padding_mode="border", align_corners=True).view(nx, nv, n_act)
    q = th.minimum(g.unsqueeze(-1), th.maximum(l.unsqueeze(-1), Vn))
    V = anchor + gamma * q.max(dim=-1).values
  return V.cpu().numpy(), xs.cpu().numpy(), vs.cpu().numpy()


# --- probes -----------------------------------------------------------------
def values(model, states):
  obs = th.as_tensor(np.array(states, np.float32), device=model.device)
  with th.no_grad():
    return model.policy.predict_values(obs).cpu().numpy().ravel()


def rollout(model, width, x0, n=64, seed=0):
  """From standstill at x0: crossing rate, death rate, max speed reached."""
  rng = np.random.default_rng(seed)
  crossed = died = 0
  vmax = 0.0
  for i in range(n):
    env = ToyGap(width, x0, x0)
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    for _ in range(TIMEOUT):
      a, _ = model.predict(obs, deterministic=True)
      obs, g, term, trunc, info = env.step(a)
      vmax = max(vmax, env.v)
      if term:
        died += 1
        break
      if info["l_x"] >= 0:
        crossed += 1
        break
      if trunc:
        break
  return crossed / n, died / n, vmax


class ProbeCallback(BaseCallback):
  """Log the two numbers this experiment exists to measure, vs ground truth.

  V(loiter) is the whole question: the buggy operator fixes it at g=+3 (loiter
  beats crossing), the corrected one at the tabular RA value (crossing wins).
  cross_rate is whether the policy acts on that.
  """

  def __init__(self, width, gt_loiter, every=20_000):
    super().__init__()
    self.width, self.gt_loiter, self.every = width, gt_loiter, every
    self._next = 0

  def _on_step(self) -> bool:
    if self.num_timesteps < self._next:
      return True
    self._next = self.num_timesteps + self.every
    v = values(self.model, [[-1.0, 0.0], [-1.5, 0.0], [-0.6, 0.0]])
    cr, dr, vmax = rollout(self.model, self.width, -1.5, n=16)
    wandb.log({
      "probe/V_loiter": float(v[0]), "probe/V_x-1.5": float(v[1]),
      "probe/V_x-0.6": float(v[2]),
      "probe/cross_rate": cr, "probe/die_rate": dr, "probe/v_max": vmax,
      "gt/V_loiter": self.gt_loiter,          # what the correct operator should give
      "gt/buggy_predicts_V_loiter": CLAMP_G,  # what the g anchor gives (= loiter wins)
      "gt/corrected_predicts_loiter_fixedpoint": -CLAMP_L,
    }, step=self.num_timesteps)
    return True


def train(cls, width, steps, seed, tag, gt_loiter, n_envs=16, group=None):
  run = wandb.init(
    project="robot_safety_sandbox", entity="buzinguyen",
    group=group, name=f"{group}_{tag}" if group else tag,
    job_type="E037", reinit=True,
    config=dict(experiment="E037", arm=tag, width=width, steps=steps, seed=seed,
                gamma=0.99, n_envs=n_envs, clamp_g=CLAMP_G, clamp_l=CLAMP_L,
                v_min_transit=V_MIN, gt_V_loiter=gt_loiter, algo=cls.__name__),
  )
  env = make_vec_env(ToyGap, n_envs=n_envs, seed=seed,
                     env_kwargs=dict(width=width))
  m = cls("MlpPolicy", env, n_steps=256, batch_size=1024, seed=seed,
          gamma=0.99, ent_coef=1e-3, verbose=0, device="cpu")
  m.learn(total_timesteps=steps,
          callback=ProbeCallback(width, gt_loiter))
  run.finish()
  return m


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--width", type=float, default=0.30)
  p.add_argument("--steps", type=int, default=400_000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--group", default=None,
                 help="wandb group; default E037-toy-gap-w<width>-s<seed>")
  a = p.parse_args()
  if a.group is None:
    a.group = f"E037-toy-gap-w{a.width:g}-s{a.seed}"

  print(f"=== E037 toy gap | width={a.width} | steps={a.steps} | seed={a.seed} ===\n")

  # --- ground truth
  Vgt, xs, vs = tabular_ra(a.width)
  iv0 = int(np.argmin(np.abs(vs - 0.0)))
  ix_loiter = int(np.argmin(np.abs(xs - (-1.0))))
  gt_loiter = Vgt[ix_loiter, iv0]
  gt_frac = float((Vgt >= 0).mean())
  print(f"[ground truth] tabular RA set = {gt_frac:.1%} of the grid")
  print(f"[ground truth] V(loiter x=-1, v=0) = {gt_loiter:+.3f}"
        f"   (RA-feasible from standstill: {gt_loiter >= 0})")

  gl, ll = margins(np.array(-1.0), np.array(0.0), a.width)
  print(f"[margins]      at loiter: g={gl:+.2f}  l={ll:+.2f}"
        f"   -> buggy predicts V=g={gl:+.2f}, corrected predicts V=l={ll:+.2f}\n")

  # --- the three arms. avoid is the negative control: correct by construction,
  #     and it SHOULD loiter (no reach term) -- if buggy-RA matches it, that is
  #     the finding.
  probes = [[-1.0, 0.0], [-1.5, 0.0], [-0.6, 0.0]]
  arms = (("corrected", ReachAvoidPPO), ("buggy", BuggyRAPPO), ("avoid", SafetyPPO))
  summary = {}
  for name, cls in arms:
    m = train(cls, a.width, a.steps, a.seed, name, gt_loiter, group=a.group)
    v = values(m, probes)
    cr, dr, vmax = rollout(m, a.width, -1.5)
    summary[name] = (v[0], cr, dr, vmax)
    print(f"[{name:10s}] V(loiter)={v[0]:+.3f} V(-1.5,0)={v[1]:+.3f} "
          f"V(-0.6,0)={v[2]:+.3f}")
    print(f"[{name:10s}] from standstill x=-1.5: cross={cr:.0%} die={dr:.0%} "
          f"vmax={vmax:.2f} (need v>{V_MIN})\n")

  print("=== VERDICT ===")
  print(f"  ground truth V(loiter) = {gt_loiter:+.3f} (crossing is optimal)")
  for k, (v0, cr, dr, _) in summary.items():
    print(f"  {k:10s} V(loiter)={v0:+7.3f}  cross={cr:5.0%}  die={dr:4.0%}")
  ra_cr, bug_cr, av_cr = (summary[k][1] for k in ("corrected", "buggy", "avoid"))
  if ra_cr > max(bug_cr, av_cr) + 0.3:
    print("  => corrected RA INITIATES where buggy/avoid loiter. Anchor was the trap.")
  elif ra_cr < 0.3:
    print("  => KILL RULE FIRED: corrected still loiters. Blocker is EXPLORATION,"
          " not semantics. Cancel E040/E041.")
  else:
    print("  => inconclusive; inspect the wandb curves.")


if __name__ == "__main__":
  main()
