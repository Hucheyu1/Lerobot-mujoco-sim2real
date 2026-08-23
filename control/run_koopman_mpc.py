"""Run a trained Koopman model in UR5e direct residual-torque MPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from args import Args
from control.KoopmanTorqueMPC import KoopmanTorqueMPC
from control.TrajectoryGenerator import JointTrajectoryGenerator
from models.init_model import init_model
from UR5e import UR5eTorqueConfig, UR5eTorqueEnv


def run(
    model_name: str,
    checkpoint: Path,
    steps: int = 300,
    seed: int = 7,
    horizon: int = 8,
    iterations: int = 25,
    device: str = "cpu",
    output: Path | None = None,
) -> dict[str, float]:
    model_args = Args(["--model", model_name, "--device", device])
    model = init_model(model_args)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    config = UR5eTorqueConfig(initial_position_span=0.02, initial_velocity_span=0.0)
    env = UR5eTorqueEnv(config)
    controller = KoopmanTorqueMPC(
        model,
        env.torque_limits,
        env.residual_limits,
        horizon=horizon,
        iterations=iterations,
    )
    trajectory = JointTrajectoryGenerator()
    state, _ = env.reset(seed=seed)
    states, references, residuals = [], [], []
    try:
        for step in range(steps):
            future = []
            for offset in range(1, horizon + 1):
                q_ref, dq_ref = trajectory.sample((step + offset) * config.control_timestep)
                future.append(np.concatenate((q_ref, dq_ref)))
            reference = np.asarray(future)
            residual = controller.command(state, reference)
            state, _, terminated, _, _ = env.step(residual)
            states.append(state.copy())
            references.append(reference[0])
            residuals.append(residual)
            if terminated:
                raise RuntimeError(f"Safety termination at control step {step}")
    finally:
        env.close()
    states = np.asarray(states)
    references = np.asarray(references)
    residuals = np.asarray(residuals)
    metrics = {
        "q_rmse_rad": float(np.sqrt(np.mean((states[:, :6] - references[:, :6]) ** 2))),
        "max_residual_torque_nm": float(np.max(np.abs(residuals))),
        "steps": steps,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, states=states, references=references, residual_torques=residuals, metrics=metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["DKUC", "DBKN", "IKN", "IBKN"], default="IBKN")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/ur5e_torque/koopman_mpc.npz"))
    args = parser.parse_args()
    checkpoint = args.checkpoint or Path("runs/ur5e_torque") / args.model / "best_model.pt"
    metrics = run(
        args.model,
        checkpoint,
        args.steps,
        args.seed,
        args.horizon,
        args.iterations,
        args.device,
        args.output,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
