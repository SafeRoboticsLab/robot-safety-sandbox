# Robot Safety Sandbox

**Parallelized mjlab environments for nominal-policy training, safety-policy
synthesis, and safety-filter evaluation.**

Massively-parallel **mjlab** benchmark environments for **safety_sb3**
(safety-stable-baselines): reach-avoid / avoid-only × single-agent /
adversarial (ISAACS), on GPU end-to-end — plus a `filters/` library with the
three deployment styles (value shielding, R-CBF/Q-CBF projection,
rollout shielding).

> Renamed from `safe_mjlab_zoo`; the package is `robot_safety_sandbox`.

📖 **[docs/API.md](docs/API.md)** is the canonical API reference — the `g`/`l` contract,
`TaskSpec`, the registry, `end_criterion`, and the two bridges. It pairs with
safety_sb3's [docs/API.md](https://github.com/SafeRoboticsLab/safety-stable-baselines/blob/main/docs/API.md)
(the algorithm layer).

```python
from robot_safety_sandbox import make_tensor, list_tasks
from safety_sb3 import ReachAvoidPPO

env = make_tensor("go2_gap_chain", num_envs=2048)      # ~50k steps/s on 12 GB
model = ReachAvoidPPO("MlpPolicy", env, normalize_obs=True, adaptive_lr=True,
                      ent_coef=1e-4, n_steps=48, batch_size=24576,
                      policy_kwargs=dict(log_std_init=-1.204))
model.learn(2_000_000_000)
```

## The contract (what every task guarantees)

| channel | meaning |
|---|---|
| reward | `g(s)` — physical safety margin. **Never normalize or reshape it.** |
| `l_x` | `l(s)` — target margin. Avoid-only tasks declare NO target: `compose(g_fn)` emits zeros as an inert placeholder that the avoid learners ignore. |
| dones / timeouts | mjlab auto-resets; timeouts are never value-bootstrapped |
| `metrics()` | curriculum levels + task metrics, forwarded to the logger every rollout |

A task = `cfg_builder(play) -> ManagerBasedRlEnvCfg` (spawn events, curricula,
terrain — plain mjlab, algorithm-agnostic) + `margin_fn(env) -> (g, l)`
(compose from `margins.py`). Register a `TaskSpec` and both bridges
(`make_tensor` for PPO learners, `make_numpy` for the SAC family) work.

## Tasks

| task | objective | learner | warm-starts from |
|---|---|---|---|
| `go2_stabilize` / `go2_locomote` | stand / track a command vs adversarial force (the original task; simplest zoo entry) | ReachAvoidPPO (`--adversary`: GameplayPPO) | — |
| `digit_stabilize` | humanoid stand vs adversarial torso force (Digit analog of go2_stabilize) | ReachAvoidPPO (`--adversary`: GameplayPPO) | — |
| `digit_stabilize_stay` / `_avoid` | humanoid STAY upright forever / don't fall — avoid, no target | SafetyPPO (`--adversary`: IsaacsPPO) | — |
| `digit_box_stabilize_stay` / `_avoid` | as above + keep a box balanced on the forearms | SafetyPPO (`--adversary`: IsaacsPPO) | — |
| `go2_gap_landing` | soft-land from mid-air over a gap | SafetyPPO | — |
| `go2_gap_crossing` | reverse curriculum: landing → launch | SafetyPPO | landing |
| `go2_gap_chain` | takeover momentum → safe rest (brake/jump) | ReachAvoidPPO | crossing |
| `go2_gap_chain_isaacs` | chain + worst-case force adversary | GameplayPPO | chain |
| `go2_crawl` / `_isaacs` | duck under a low bar or stop | ReachAvoidPPO / GameplayPPO | — |
| `go2_crawl_twin_avoid` / `go2_crawl_gate_avoid` | avoid twins of the crawl R-CBF pair (no target) | SafetyPPO | — |
| `go2_crawl_twin_ra` / `go2_crawl_gate_ra` | reach-avoid twins of the crawl R-CBF pair | ReachAvoidPPO | — |

Task structure varies: `go2_stabilize` needs no curriculum or staging at all,
while the gap family only forms its jump through staged warm-starts
(`TaskSpec.warmstart_from`) at real scale (~2B env-steps for the chain). See
PORTING.md for which machinery your task actually needs.

## Training recipe that works (hard-won)

`normalize_obs=True` (obs only — reward normalization is refused by
safety_sb3), `ent_coef=1e-4`, `log_std_init=ln(0.3)`, `adaptive_lr=True`
(`desired_kl=0.01`, lr `5e-4`), `n_steps=48`. Watch the `env/Curriculum/*`
logger keys — a stalled curriculum looks exactly like converged training in
the reward curve.

## Porting a new task

1. Write the mjlab env cfg: terrain, spawn events (takeover-momentum or
   staged spawns), reverse curricula. Study `go2_gap` — especially how the
   landing → crossing → chain pipeline seeds rare-win skills.
2. Compose `margin_fn` from `margins.py` (or add new terms there).
3. `register(TaskSpec(...))` in `tasks/<your_task>.py`.
4. Train with `examples/train.py --task <id>`; verify curricula CLIMB in wandb.

## Extending

`docs/EXTENDING.md` walks the four extension axes with worked examples from
the shipped tasks: margin functions, sensors/observations, terrains
(heightfields, walls, gaps, obstacles), and contacts — plus the
new-robot checklist (go2 and digit are the two reference layouts).

## Repo layout / status

SELF-CONTAINED: env cfgs, terrains, robot assets (Go2, Digit), and
the handover dataset are native under `robot_safety_sandbox/envs/` + `data/`.
`safety_sb3` is a pinned pip dependency
([safety-stable-baselines](https://github.com/SafeRoboticsLab/safety-stable-baselines)
v0.1.0); mjlab is a peer dep with its own pinned sim stack (INSTALL.md).
