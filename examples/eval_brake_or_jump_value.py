"""Split-test value-ordering probe: at near-edge decision states across a
momentum sweep, what does each critic VALUE crossing, and does the policy cross?

For each spawn momentum it reports V(spawn), the realized crossing rate, and the
death (safety-failure) rate. The reach-avoid twin should cross from standstill
with a certificate backed by realized crossings; an avoid-only twin (and a
reach-avoid twin trained under the g-anchor bug) stalls at low momentum while its
critic over-certifies the non-crossing state.

Pass --ra-model (reach-avoid) plus optional --avoid-model / --buggy-model
SafetyPPO/ReachAvoidPPO run dirs to contrast against.

  python examples/eval_split_value_ordering.py --task go2_gap_brake_or_jump_ra_w30 \
      --ra-model runs/<ra_run>/final_model.zip --avoid-model runs/<avoid_run>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch as th

from safety_sb3 import SafetyPPO, ReachAvoidPPO
from robot_safety_sandbox.registry import spec
from robot_safety_sandbox.base import MjlabTensorSafetyEnv
from robot_safety_sandbox.envs.go2_gap import brake_or_jump as S

DEV, N, STEPS = "cuda:0", 512, 130
X0 = -0.10                                   # near edge (minimal runway)
SS = [1.0, 0.7, 0.5, 0.3, 0.15]              # momentum scale -> vx ~ 2.8*s

p = argparse.ArgumentParser()
p.add_argument("--ra-model", required=True, help="corrected-RA checkpoint .zip")
p.add_argument("--task", default="go2_gap_brake_or_jump_ra",
               help="split task (sets gap width): _ra / _ra_w20 / _ra_w30")
p.add_argument("--avoid-model", default=None,
               help="optional SafetyPPO avoid run dir to contrast against")
p.add_argument("--buggy-model", default=None,
               help="optional (g-anchor) ReachAvoidPPO run dir to contrast against")
p.add_argument("--n", type=int, default=N)
args = p.parse_args()
N = args.n

s = spec(args.task)
env = MjlabTensorSafetyEnv(N, DEV, cfg_builder=s.cfg_builder, margin_fn=s.margin_fn,
                           ctrl_dim=12, dstb_dim=0)
mj = env.mj
robot = mj.scene["robot"]
ox = mj.scene.env_origins[:, 0]
S._ensure(mj)
bank = S._bank(mj)
FAR_X = S.far_x(mj)
GAP_W = S._gap_width(mj)
print(f"[probe] task={args.task} N={N} gap_w={GAP_W} far_x={FAR_X:.3f}  bank={bank.shape[0]} states")


def spawn(s_val):
  """Grounded far poses relocated to the near edge with momentum scaled by s_val."""
  env.reset()
  ids = th.arange(N, device=DEV, dtype=th.int)
  rows = S._sample_rows(bank, lambda b: b[:, 0] > 0.35, N, DEV)
  xt = th.full((N,), X0, device=DEV) + th.rand(N, device=DEV) * 0.04 - 0.02
  ms = th.full((N,), s_val, device=DEV)
  S._restore(mj, ids, rows, xt, ms)
  return env.step_tensor(th.zeros(N, 12, device=DEV))[0]


def _find_stats(zp):
  """Locate the obs-normalization stats for a model zip.
  Finals: tensornormalize.pt alongside. Checkpoints: tensornorm_<step>.pt in the
  same dir — pick the one whose step is closest to the model's step."""
  d = os.path.dirname(zp)
  flat = os.path.join(d, "tensornormalize.pt")
  if os.path.exists(flat):
    return flat
  import re, glob
  m = re.search(r"model_(\d+)_steps", os.path.basename(zp))
  step = int(m.group(1)) if m else 0
  cands = glob.glob(os.path.join(d, "tensornorm_*.pt"))
  if not cands:
    raise FileNotFoundError(f"no tensornorm stats near {zp}")
  key = lambda c: abs(int(re.search(r"tensornorm_(\d+)", c).group(1)) - step)
  return min(cands, key=key)


def load(cls, run_dir=None, zip_path=None):
  zp = zip_path or os.path.join(run_dir, "final_model.zip")
  model = cls.load(zp, device=DEV, custom_objects={"tensorboard_log": None})
  st = th.load(_find_stats(zp), map_location=DEV, weights_only=True)
  return model, st["obs_mean"], st["obs_var"]


TW = [("corr-RA ", ReachAvoidPPO, dict(zip_path=args.ra_model))]
if args.avoid_model:
  TW.insert(0, ("avoid   ", SafetyPPO, dict(run_dir=args.avoid_model)))
if args.buggy_model:
  TW.insert(-1, ("buggy-RA", ReachAvoidPPO, dict(run_dir=args.buggy_model)))

for name, Cls, kw in TW:
  try:
    model, mean, var = load(Cls, **kw)
  except Exception as e:
    print(f"\n===== {name}: LOAD FAILED -> {e}"); continue
  print(f"\n===== {name}   V(spawn) / cross  vs momentum =====")
  vs = []
  for sv in SS:
    obs = spawn(sv)
    o = th.clamp((obs.float() - mean) / th.sqrt(var + 1e-8), -10, 10)
    with th.no_grad():
      V = model.policy.predict_values(o).squeeze(-1).mean().item()
    crossed = th.zeros(N, dtype=th.bool, device=DEV)
    died = th.zeros(N, dtype=th.bool, device=DEV)   # safety failure (g<0): charge-and-die
    for _ in range(STEPS):
      o = th.clamp((obs.float() - mean) / th.sqrt(var + 1e-8), -10, 10)
      with th.no_grad():
        a = th.clamp(model.policy._predict(o, deterministic=True), -1, 1)
      obs, g, _term, _trunc, _l = env.step_tensor(a)
      x = robot.data.root_link_pos_w[:, 0] - ox
      up = -robot.data.projected_gravity_b[:, 2]
      crossed |= (x > FAR_X) & (up > 0.7)
      died |= (g < 0) & ~crossed          # fell in the pit / fell over before reaching
    vs.append(V)
    print(f"  vx~{2.8*sv:4.1f} (s={sv:.2f}):  V(spawn) {V:+.3f}   "
          f"cross {crossed.float().mean():.2f}   die {died.float().mean():.2f}")
  print(f"  --> V slope (fast-slow) = {vs[0]-vs[-1]:+.3f}   (RA: positive; avoid/buggy: ~0)")
