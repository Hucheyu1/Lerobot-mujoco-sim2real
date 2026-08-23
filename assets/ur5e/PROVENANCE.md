# UR5e model provenance

The files under `menagerie/` are an unmodified copy of
`universal_robots_ur5e` from Google DeepMind's MuJoCo Menagerie at commit
`da76818e269b82289eba39808e2fb91d679d6994` (retrieved 2026-08-22).

Upstream: <https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/universal_robots_ur5e>

The upstream model is distributed under the BSD-3-Clause license contained in
`menagerie/LICENSE`. It is a simplified MJCF description derived from the
public Universal Robots URDF.

Local derived files:

- `ur5e_torque.xml`: switches the six position servos to direct-drive torque
  motors, keeps the upstream joint torque limits, sets a 0.002 s physics step,
  and adds a fixed payload body whose mass and inertia can be varied at runtime.
- `scene_torque.xml`: loads the torque-controlled model in the upstream scene.

The local model must be described as a dynamics-enabled direct-torque MuJoCo
model unless and until its dynamics have been cross-validated against an
independent reference. It must not be called an official Universal Robots
simulator.
