# Robot Safety Sandbox

**Parallelized mjlab environments for safety-policy synthesis, task-policy
training, and safety-filter evaluation.**

Massively-parallel **mjlab** benchmark environments for **safety_sb3**
(safety-stable-baselines): reach-avoid / avoid-only × single-player /
zero-sum two-player, on GPU end-to-end — plus a `filters/` library in which a
safety filter is a COMPOSITION of three swappable modules (fallback / monitor /
intervention), not a class per recipe.

> Renamed from `safe_mjlab_zoo`; the package is `robot_safety_sandbox`.

📖 **[docs/API.md](docs/API.md)** is the canonical API reference — the `g`/`l` contract,
`TaskSpec`, the registry, `end_criterion`, and the two bridges. It pairs with
safety_sb3's [docs/API.md](https://github.com/SafeRoboticsLab/safety-stable-baselines/blob/main/docs/API.md)
(the algorithm layer).

```python
from robot_safety_sandbox import make_tensor, list_tasks
from safety_sb3 import ReachAvoidPPO1P

env = make_tensor("go2_gap_chain", num_envs=2048)      # ~50k steps/s on 12 GB
model = ReachAvoidPPO1P("MlpPolicy", env, normalize_obs=True, adaptive_lr=True,
                        ent_coef=1e-4, n_steps=48, batch_size=24576,
                        policy_kwargs=dict(log_std_init=-1.204))
model.learn(2_000_000_000)
```

## The MAP

> **Here's a MAP to navigate the codebase — Mode. Algorithm. Players.**

    M = Mode       Safety | ReachAvoid | Cumulative    the Bellman operator
    A = Algorithm  PPO | SAC | A2C | DQN               the RL update rule
    P = Players    1P | 2P                             single-player | zero-sum

Every learner's name is those three letters concatenated, in that order —
`SafetyPPO1P`, `ReachAvoidSAC2P` — and each letter comes from exactly one place:

| letter | comes from | how you set it |
|---|---|---|
| **M** | the TASK | its `TaskSpec(mode=...)` — a property of its margins |
| **A** | the RUN | `train.py --family on_policy` (PPO) / `--family off_policy` (SAC) |
| **P** | the RUN | `--adversary` |

`registry.algo_name(task_id, adversary, family)` is that concatenation and
nothing else — no lookup table, no per-task override:

```python
>>> from robot_safety_sandbox import algo_name
>>> algo_name("go2_stabilize")                                 # M=ReachAvoid A=PPO P=1P
'ReachAvoidPPO1P'
>>> algo_name("go2_stabilize", adversary=True, family="off_policy")
'ReachAvoidSAC2P'
>>> algo_name("digit_stabilize_avoid", adversary=True)
'SafetyPPO2P'
```

**`*PPO2P` and `*SAC2P` are not interchangeable.** They are different
algorithms, not one game with two optimizers:

| | `*SAC2P` | `*PPO2P` |
|---|---|---|
| critic | ONE shared joint-action critic `Q(s, [a_ctrl, a_dstb])` | two independent `V(s)` nets |
| game | minimax on that shared critic | alternating best response (an *approximation*) |
| machinery | one replay buffer | two rollout buffers + a ctrl/dstb phase machine |

So `--adversary` means something different in each family. Pick the family
deliberately; the E042 result (`go2_stabilize`, best-ever) is `ReachAvoidSAC2P`.

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
(`make_tensor` for the PPO family, `make_numpy` for the SAC family) work.

Each task declares exactly one axis, its **`mode`** (the MAP's M) — the
`safety_sb3` backup it is trained under:

| `mode` | backup | 1P learner | 2P learner (`--adversary`) |
|---|---|---|---|
| `"safety"` | `V = min(g, γV′)` (avoid) | `SafetyPPO1P` / `SafetySAC1P` | `SafetyPPO2P` / `SafetySAC2P` |
| `"reach-avoid"` | `V = min(g, max(l, γV′))` | `ReachAvoidPPO1P` / `ReachAvoidSAC1P` | `ReachAvoidPPO2P` / `ReachAvoidSAC2P` |
| `"cumulative"` | `V = r + γ(1−d)V′` (plain RL) | stock `stable_baselines3` `PPO` / `SAC` | — (no two-player cumulative game) |

`"cumulative"` is the dense-reward **task policy** a safety filter wraps — no
margins, and stock SB3 so its checkpoint stays a vanilla SB3 zip. One trainer
covers all three: `examples/train.py --family on_policy --task <id>`.

## Tasks

The learner column below is what the MAP resolves to in the **on-policy**
family; swap `PPO`→`SAC` for `--family off_policy`.

| task | objective | learner | warm-starts from |
|---|---|---|---|
| `go2_stabilize` / `go2_locomote` | stand / track a command vs adversarial force (the original task; simplest zoo entry) | ReachAvoidPPO1P (`--adversary`: ReachAvoidPPO2P) | — |
| `digit_stabilize` | humanoid stand vs adversarial torso force (Digit analog of go2_stabilize) | ReachAvoidPPO1P (`--adversary`: ReachAvoidPPO2P) | — |
| `digit_stabilize_stay` / `_avoid` | humanoid STAY upright forever / don't fall — avoid, no target | SafetyPPO1P (`--adversary`: SafetyPPO2P) | — |
| `digit_box_stabilize_stay` / `_avoid` | as above + keep a box balanced on the forearms | SafetyPPO1P (`--adversary`: SafetyPPO2P) | — |
| `go2_gap_landing` | soft-land from mid-air over a gap | SafetyPPO1P | — |
| `go2_gap_crossing` | reverse curriculum: landing → launch | SafetyPPO1P | landing |
| `go2_gap_chain` | takeover momentum → safe rest (brake/jump) | ReachAvoidPPO1P | crossing |
| `go2_gap_chain_isaacs` | chain + worst-case force adversary | ReachAvoidPPO2P | chain |
| `go2_crawl` / `_isaacs` | duck under a low bar or stop | ReachAvoidPPO1P / ReachAvoidPPO2P | — |
| `go2_crawl_twin_avoid` / `go2_crawl_gate_avoid` | avoid twins of the crawl R-CBF pair (no target) | SafetyPPO1P | — |
| `go2_crawl_twin_ra` / `go2_crawl_gate_ra` | reach-avoid twins of the crawl R-CBF pair | ReachAvoidPPO1P | — |
| `go2_walker_flat` / `go2_crawl_walk` | dense-reward task policies (π_task) the filters wrap — blind flat walker / low-crawl walker | stock SB3 `PPO` (`mode="cumulative"`) | — |

Task structure varies: `go2_stabilize` needs no curriculum or staging at all,
while the gap family only forms its jump through staged warm-starts at real
scale (~2B env-steps for the chain). Warm-start LINEAGE is a run-level `--load`
choice recorded in the experiment log, not a registry field. See PORTING.md for
which machinery your task actually needs.

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
3. `register(TaskSpec(..., mode=...))` in `tasks/<your_task>.py` — the `mode`
   is the only learner-related thing you declare.
4. Train with `examples/train.py --family on_policy --task <id>`; verify curricula CLIMB in wandb.

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
v0.4.0); mjlab is a peer dep with its own pinned sim stack (INSTALL.md).
