"""Run a trained Koopman model in UR5e direct residual-torque MPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from args import Args
from control.MPC_Controler import MPCController
from control.TrajectoryGenerator import JointTrajectoryGenerator
from models.init_model import init_model
from UR5e import UR5eTorqueConfig, UR5eTorqueEnv


def run(
    model_name: str,
    checkpoint: Path,
    steps: int = 300,
    seed: int = 7,
    horizon: int = 8,
    mpc_type: str = "delta_mpc",
    device: str = "cpu",
    output: Path | None = None,
) -> dict[str, float | str]:
    model_args = Args(["--model", model_name, "--device", device])
    model = init_model(model_args).double()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    config = UR5eTorqueConfig(initial_position_span=0.02, initial_velocity_span=0.0)
    env = UR5eTorqueEnv(config)
    model_args.args.MPC_type = mpc_type
    model_args.args.mpc_horizon = horizon
    controller = MPCController(
        model,
        model_args,
        env.torque_limits,
        env.residual_limits,
        horizon=horizon,
    )
    trajectory = JointTrajectoryGenerator()
    state, _ = env.reset(seed=seed)
    states, references, residuals = [], [], []
    try:
        for step in range(steps):
            future = []
            for offset in range(1, horizon + 1):
                q_ref, dq_ref = trajectory.sample((step + offset) * config.control_timestep)
                future.append(env.reference_state(q_ref, dq_ref))
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
        "ee_rmse_m": float(np.sqrt(np.mean((states[:, :3] - references[:, :3]) ** 2))),
        "q_rmse_rad": float(np.sqrt(np.mean((states[:, 3:9] - references[:, 3:9]) ** 2))),
        "max_residual_torque_nm": float(np.max(np.abs(residuals))),
        "solver_backend": controller.solver_backend,
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
    parser.add_argument("--MPC-type", choices=["delta_mpc", "mpc"], default="delta_mpc", dest="MPC_type")
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
        args.MPC_type,
        args.device,
        args.output,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
