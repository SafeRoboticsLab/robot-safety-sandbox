# Phase-2 migration: DONE (2026-07-05)

The zoo is self-contained (all six tasks build+step without unitree_rl_mjlab
on sys.path). unitree_rl_mjlab remains the rsl_rl workbench + robot deployment
repo. Remaining release step: add this repo as a benchmarks/ submodule of
safety-stable-baselines once it has a GitHub remote.

Original checklist (executed):
  
  Checklist (mechanical; ~20 files):
  1. Copy from unitree_rl_mjlab into `safe_mjlab_zoo/tasks/go2_gap/`:
     - src/isaacs_go2/{island_terrain,safety_filter_terrain,chain_terrain}.py
     - src/tasks/parkour/{parkour_env_cfg.py,terrains.py,mdp/*} (the slice the
       gap cfgs use: observations, terminations, curriculums)
     - src/tasks/parkour/config/go2/env_cfgs.py (base Go2 parkour cfg)
     - src/tasks/go2_safety_filter/{gap,landing,crossing,crossing_chain}/env_cfg.py
     - datasets/walker_handover_states.pt -> zoo data/ (or a download hook)
  2. Rewrite imports `src.tasks...`/`src.isaacs_go2...` -> `safe_mjlab_zoo.tasks.go2_gap...`.
  3. Same for go2_crawl from branch feat/reach-avoid-crawl.
  4. Drop the sys.path compat note from tasks/*.py + README.
  5. Smoke: `make_tensor(task, 16)` builds + steps for every task; short train.
  6. Add the zoo to safety-stable-baselines as submodule `benchmarks/zoo`.
  7. Freeze unitree_rl_mjlab (legacy rsl_rl baselines; not shipped).