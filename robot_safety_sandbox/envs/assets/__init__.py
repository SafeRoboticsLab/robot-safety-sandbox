"""Vendored robot assets: one package per embodiment.

Each subpackage owns its MJCF, meshes and an mjlab ``EntityCfg`` builder:

  car/    a planar car (car_goal tasks)
  digit/  Agility Robotics Digit v3 humanoid
  go2/    Unitree Go2 quadruped

Paths inside each subpackage are resolved RELATIVE TO ITS OWN FILE
(``Path(__file__).resolve().parent / "xmls" / ...``), so a subpackage can be
moved without touching path code.
"""
