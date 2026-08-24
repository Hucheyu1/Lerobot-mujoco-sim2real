"""Run a closed-loop UR5e direct-torque tracking demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from control import JointTorquePDController, JointTrajectoryGenerator
from UR5e import UR5eTorqueConfig, UR5eTorqueEnv


def run(steps: int = 300, seed: int = 7, render: bool = False, output: Path | None = None) -> dict[str, float]:
    config = UR5eTorqueConfig(initial_position_span=0.02, initial_velocity_span=0.0)
    env = UR5eTorqueEnv(config, render_mode="human" if render else None)
    controller = JointTorquePDController(
        env.torque_limits,
        feedforward_torque=lambda: env.inverse_dynamics_torque(np.zeros(6)),
    )
    trajectory = JointTrajectoryGenerator()
    state, _ = env.reset(seed=seed)
    states, references, joint_torques = [], [], []
    try:
        for step in range(steps):
            q_ref, dq_ref = trajectory.sample(step * config.control_timestep)
            joint_torque = controller.command(state, q_ref, dq_ref)
            state, _, terminated, _, info = env.step(joint_torque)
            states.append(state.copy())
            references.append(env.reference_state(q_ref, dq_ref))
            joint_torques.append(info["applied_joint_torque"])
            if terminated:
                raise RuntimeError(f"Safety termination at control step {step}")
    finally:
        env.close()

    state_array = np.asarray(states)
    reference_array = np.asarray(references)
    q_rmse = float(np.sqrt(np.mean((state_array[:, 3:9] - reference_array[:, 3:9]) ** 2)))
    metrics = {
        "q_rmse_rad": q_rmse,
        "max_joint_torque_nm": float(np.max(np.abs(joint_torques))),
        "steps": steps,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            states=state_array,
            references=reference_array,
            joint_torques=np.asarray(joint_torques),
            metrics=metrics,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/ur5e_torque/control_demo.npz"))
    args = parser.parse_args()
    print(run(args.steps, args.seed, args.render, args.output))


if __name__ == "__main__":
    main()
