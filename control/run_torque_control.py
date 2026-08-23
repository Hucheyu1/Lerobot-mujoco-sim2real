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
    controller = JointTorquePDController(env.residual_limits)
    trajectory = JointTrajectoryGenerator()
    state, _ = env.reset(seed=seed)
    states, references, residuals, applied = [], [], [], []
    try:
        for step in range(steps):
            q_ref, dq_ref = trajectory.sample(step * config.control_timestep)
            residual = controller.command(state, q_ref, dq_ref)
            state, _, terminated, _, info = env.step(residual)
            states.append(state.copy())
            references.append(np.concatenate((q_ref, dq_ref)))
            residuals.append(info["residual_torque"])
            applied.append(info["applied_torque"])
            if terminated:
                raise RuntimeError(f"Safety termination at control step {step}")
    finally:
        env.close()

    state_array = np.asarray(states)
    reference_array = np.asarray(references)
    q_rmse = float(np.sqrt(np.mean((state_array[:, :6] - reference_array[:, :6]) ** 2)))
    metrics = {
        "q_rmse_rad": q_rmse,
        "max_residual_torque_nm": float(np.max(np.abs(residuals))),
        "max_applied_torque_nm": float(np.max(np.abs(applied))),
        "steps": steps,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            states=state_array,
            references=reference_array,
            residual_torques=np.asarray(residuals),
            applied_torques=np.asarray(applied),
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
