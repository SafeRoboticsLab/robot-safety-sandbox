# Troubleshooting

Common failures and their fixes. Most are environment/version issues, not bugs in
the package.

## Installation and import

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: ls_parallel was removed in MuJoCo Warp 3.9.1` at env build | unpinned `mujoco-warp` (mjlab 1.2.0 sets an option removed upstream) | install the pinned triple: `mujoco==3.6.0 mujoco-warp==3.6.0` with `mjlab==1.2.0` (see [requirements](requirements.md)) |
| `ModuleNotFoundError: scipy` | mjlab 1.2.0 imports scipy without declaring it | `pip install scipy` |
| `pip install -e .` cannot resolve `safety_sb3` | the pinned git tag is unreachable, or you are offline | ensure network access to GitHub; to develop both repos, `pip install -e` the `safety-stable-baselines` checkout first (it satisfies the requirement) |
| Digit tasks warn-and-skip on import | stock mjlab lacks the entity patch Digit needs to simulate | expected — Digit *imports* fine; simulation needs the custom mjlab fork |

## GPU and rendering

| Symptom | Cause | Fix |
|---|---|---|
| First run hangs for ~1 min with no output | Warp is JIT-compiling kernels for your GPU arch | wait — it is a one-time, per-machine compile; the cache is reused afterward |
| `gladLoadGL error` / OpenGL context error at the first video | no display and no offscreen GL context | `export MUJOCO_GL=egl` (headless NVIDIA EGL) before running |
| `wandb.Video requires moviepy` | video encoder missing | `pip install moviepy imageio` |
| Out-of-memory at env build | `--num-envs` too high for your GPU | lower `--num-envs`; Go2 tasks want ~4–9 GB at 2–3k envs |

## Checkpoints and evaluation

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` / `AttributeError` unpickling a `.zip` | checkpoint trained on an older version (learner class renamed) | retrain on v0.4.0, or pin both repos to the old tag (see [release notes](../release-notes.md)) |
| `argparse` error on an eval command from older docs | retired flags (`--twin`, `--nominal`, `--nominal-from-twin`) | use the v0.4.0 names: `--safety-policy`, `--task-policy`, `--safety-only` (see [evaluation](../guide/evaluation.md)) |
| Repeat eval runs give different numbers | evaluation is **not** bit-reproducible (obs noise + non-deterministic Warp) | compare configurations over several runs, not a single run |
| A run looks "converged" but performs poorly | a stalled curriculum reads like converged training in the reward curve | watch the `env/Curriculum/*` logger keys, not just `ep_rew_mean` |

## Still stuck?

- Re-check the [compatibility matrix](requirements.md) — most issues are a version
  mismatch in the sim stack.
- Re-run the [installation](../installation.md) verification commands (CPU import,
  then the GPU training smoke test) to localize the failure.
