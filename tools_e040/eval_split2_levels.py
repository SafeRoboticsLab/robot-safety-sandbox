"""Decision-band split: pin each twin's 25M checkpoint at levels 8..12 and
measure cross-success. Levels 0-7 are the shared committed band; 8-12 are the
decision band (near edge, momentum scaled down) where the split should appear."""
import os, sys
sys.path.insert(0, os.path.expanduser("~/SAFE/safe_mjlab_zoo"))
sys.path.insert(0, os.path.expanduser("~/SAFE/safety-stable-baselines"))
import torch as th
from safety_sb3 import SafetyPPO, ReachAvoidPPO
from safe_mjlab_zoo.registry import spec
from safe_mjlab_zoo.base import MjlabTensorSafetyEnv
from safe_mjlab_zoo.envs.go2_gap import split_v2 as S

DEV, N, STEPS = "cuda:0", 1024, 130
CKPT = "model_24999936_steps.zip"
NORM = "tensornorm_25001984.pt"
RUNS = "~/SAFE/safe_mjlab_zoo/runs_zoo"
TW = [("avoid", SafetyPPO, "go2_gap_split2_avoid"),
      ("RA", ReachAvoidPPO, "go2_gap_split2_ra")]

s = spec("go2_gap_split2_ra")
env = MjlabTensorSafetyEnv(N, DEV, cfg_builder=s.cfg_builder,
                           margin_fn=s.margin_fn, ctrl_dim=12, dstb_dim=0)
mj = env.mj; robot = mj.scene["robot"]; ox = mj.scene.env_origins[:, 0]
S._ensure(mj)

print(f"{'level':>6} | " + " | ".join(f"{n:>6}" for n, _, _ in TW))
results = {}
for name, Cls, run in TW:
  rd = os.path.expanduser(os.path.join(RUNS, run))
  model = Cls.load(os.path.join(rd, "checkpoints", CKPT), device=DEV,
                   custom_objects={"tensorboard_log": None})
  st = th.load(os.path.join(rd, "checkpoints", NORM), map_location=DEV,
               weights_only=True)
  mean, var = st["obs_mean"], st["obs_var"]
  for L in [8, 9, 10, 11, 12]:
    mj._L = L; mj._peek[:] = False
    obs = env.reset()
    crossed = th.zeros(N, dtype=th.bool, device=DEV)
    for t in range(STEPS):
      o = th.clamp((obs.float() - mean) / th.sqrt(var + 1e-8), -10, 10)
      with th.no_grad():
        a = th.clamp(model.policy._predict(o, deterministic=True), -1, 1)
      obs, _g, term, trunc, _l = env.step_tensor(a)
      x = robot.data.root_link_pos_w[:, 0] - ox
      up = -robot.data.projected_gravity_b[:, 2]
      crossed |= (x > S.FAR_X) & (up > 0.7)
    results[(name, L)] = crossed.float().mean().item()

for L in [8, 9, 10, 11, 12]:
  cells = " | ".join(f"{results[(n, L)]:6.2f}" for n, _, _ in TW)
  tag = {8: "near s0.7", 10: "back s0.45", 12: "A/B/C mix"}.get(L, "")
  print(f"{L:>6} | {cells}   {tag}")
