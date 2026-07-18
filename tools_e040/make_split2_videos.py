"""Scenario eval videos for both split_v2 twins (75M ckpts). Pins the curriculum
level to specific spawn scenarios and records a herd clip per (twin, scenario):
  L7  committed launch      -> both should jump (shared skill)
  L8  decision, mom x0.6     -> RA initiates / avoid stops (the split)
  L9  decision, mom x0.5     -> stronger split
  L12 A/B/C mixture          -> conditional: jump committed, stop far/slow
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/SAFE/safe_mjlab_zoo"))
sys.path.insert(0, os.path.expanduser("~/SAFE/safety-stable-baselines"))
import numpy as np
import torch as th
import imageio
from safety_sb3 import SafetyPPO, ReachAvoidPPO
from safe_mjlab_zoo.registry import spec
from safe_mjlab_zoo.base import MjlabTensorSafetyEnv
from safe_mjlab_zoo.envs.go2_gap import split_v2 as S

DEV, N, LEN = "cuda:0", 6, 200
CKPT, NORM = "model_74999808_steps.zip", "tensornorm_75005952.pt"
RUNS = os.path.expanduser("~/SAFE/safe_mjlab_zoo/runs_zoo")
OUT = os.path.expanduser("~/SAFE/vids"); os.makedirs(OUT, exist_ok=True)
SCEN = [(7, "committed"), (8, "decision_mom0.6"), (9, "decision_mom0.5"),
        (12, "mixtureABC")]
TW = [("avoid", SafetyPPO, "go2_gap_split2_avoid"),
      ("RA", ReachAvoidPPO, "go2_gap_split2_ra")]

s = spec("go2_gap_split2_ra")
env = MjlabTensorSafetyEnv(N, DEV, cfg_builder=s.cfg_builder, margin_fn=s.margin_fn,
                           ctrl_dim=12, dstb_dim=0, render_mode="rgb_array")
mj = env.mj; S._ensure(mj)

for name, Cls, run in TW:
  rd = os.path.join(RUNS, run)
  model = Cls.load(os.path.join(rd, "checkpoints", CKPT), device=DEV,
                   custom_objects={"tensorboard_log": None})
  st = th.load(os.path.join(rd, "checkpoints", NORM), map_location=DEV,
               weights_only=True)
  mean, var = st["obs_mean"], st["obs_var"]
  for L, tag in SCEN:
    mj._L = L; mj._peek[:] = False
    obs = env.reset()
    frames = []
    for t in range(LEN):
      o = th.clamp((obs.float() - mean) / th.sqrt(var + 1e-8), -10, 10)
      with th.no_grad():
        a = th.clamp(model.policy._predict(o, deterministic=True), -1, 1)
      obs, _g, _d, _t, _l = env.step_tensor(a)
      frames.append(np.asarray(env.render()))
    fn = os.path.join(OUT, f"{name}_L{L}_{tag}.mp4")
    imageio.mimwrite(fn, frames, fps=30, macro_block_size=1)
    print(f"[video] {fn}  ({len(frames)} frames, shape {frames[0].shape})",
          flush=True)
print("DONE")
