"""Value ordering: at near-edge decision states across a momentum sweep, does
each twin's OWN critic value crossing? RA's reach term should make V rise with
feasibility (it banks the reachable far target); avoid's viability value should
be ~flat-high (stopping is always safe), i.e. its critic is indifferent to
crossing. That slope difference = a FORMULATION result, not exploration luck.
Reports V(spawn) and realized cross per momentum bin, for both twins."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/SAFE/safe_mjlab_zoo"))
sys.path.insert(0, os.path.expanduser("~/SAFE/safety-stable-baselines"))
import torch as th
from safety_sb3 import SafetyPPO, ReachAvoidPPO
from safe_mjlab_zoo.registry import spec
from safe_mjlab_zoo.base import MjlabTensorSafetyEnv
from safe_mjlab_zoo.envs.go2_gap import split_v2 as S

DEV, N, STEPS = "cuda:0", 1024, 130
X0 = -0.10                                  # near edge (minimal runway)
SS = [1.0, 0.7, 0.5, 0.3, 0.15]             # momentum scale -> vx ~ 2.8*s
RUNS = os.path.expanduser("~/SAFE/safe_mjlab_zoo/runs_zoo")
TW = [("avoid", SafetyPPO, "go2_gap_split2_avoid"),
      ("RA", ReachAvoidPPO, "go2_gap_split2_ra")]

s = spec("go2_gap_split2_ra")
env = MjlabTensorSafetyEnv(N, DEV, cfg_builder=s.cfg_builder, margin_fn=s.margin_fn,
                           ctrl_dim=12, dstb_dim=0)
mj = env.mj; robot = mj.scene["robot"]; ox = mj.scene.env_origins[:, 0]
S._ensure(mj); bank = S._bank(DEV)


def spawn(s_val):
  env.reset()
  ids = th.arange(N, device=DEV, dtype=th.int)
  rows = S._sample_rows(bank, lambda b: b[:, 0] > 0.35, N, DEV)
  xt = th.full((N,), X0, device=DEV) + th.rand(N, device=DEV) * 0.04 - 0.02
  ms = th.full((N,), s_val, device=DEV)
  S._restore(mj, ids, rows, xt, ms)
  return env.step_tensor(th.zeros(N, 12, device=DEV))[0]


for name, Cls, run in TW:
  rd = os.path.join(RUNS, run)
  model = Cls.load(os.path.join(rd, "final_model.zip"), device=DEV,
                   custom_objects={"tensorboard_log": None})
  st = th.load(os.path.join(rd, "tensornormalize.pt"), map_location=DEV,
               weights_only=True)
  mean, var = st["obs_mean"], st["obs_var"]
  print(f"\n===== {name}  V(spawn) / cross  vs momentum =====")
  for sv in SS:
    obs = spawn(sv)
    o = th.clamp((obs.float() - mean) / th.sqrt(var + 1e-8), -10, 10)
    with th.no_grad():
      V = model.policy.predict_values(o).squeeze(-1).mean().item()
    crossed = th.zeros(N, dtype=th.bool, device=DEV)
    for t in range(STEPS):
      o = th.clamp((obs.float() - mean) / th.sqrt(var + 1e-8), -10, 10)
      with th.no_grad():
        a = th.clamp(model.policy._predict(o, deterministic=True), -1, 1)
      obs, _g, term, trunc, _l = env.step_tensor(a)
      x = robot.data.root_link_pos_w[:, 0] - ox
      up = -robot.data.projected_gravity_b[:, 2]
      crossed |= (x > S.FAR_X) & (up > 0.7)
    print(f"  vx~{2.8*sv:4.1f} (s={sv:.2f}):  V(spawn) {V:+.3f}   cross {crossed.float().mean():.2f}")
