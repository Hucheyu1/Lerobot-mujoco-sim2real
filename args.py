"""Command-line configuration for the cleaned UR5e torque project."""

from __future__ import annotations

import argparse
from pathlib import Path


class Args:
    """Parse options while preserving the attribute-style API of the old code."""

    def __init__(self, argv: list[str] | None = None):
        parser = argparse.ArgumentParser(description="UR5e torque Koopman training and evaluation")
        parser.add_argument("--model", choices=["DKUC", "DBKN", "IKN", "IBKN", "all"], default="IBKN")
        parser.add_argument("--mode", choices=["collect", "train", "test"], default="train")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--force-data", action="store_true", help="regenerate existing trajectory arrays")

        # Preserve the SOARM101 experiment layout: many short random training
        # trajectories and fewer, longer validation/test trajectories.
        parser.add_argument("--train-samples", type=int, default=50_000)
        parser.add_argument("--train-steps", type=int, default=20)
        parser.add_argument("--val-samples", type=int, default=2_000)
        parser.add_argument("--test-samples", type=int, default=2_000)
        parser.add_argument("--test-steps", type=int, default=200)
        parser.add_argument("--test-type", choices=["random", "sin", "chirp", "all"], default="all")

        parser.add_argument("--physics-timestep", type=float, default=0.002)
        parser.add_argument("--frame-skip", type=int, default=10)
        parser.add_argument("--initial-position-span", type=float, default=0.50)
        parser.add_argument("--initial-velocity-span", type=float, default=0.05)
        parser.add_argument("--waypoint-count", type=int, default=10)
        parser.add_argument("--waypoint-velocity-limit", type=float, default=0.07)
        parser.add_argument("--tracking-kp", type=float, default=16.0)
        parser.add_argument("--tracking-kd", type=float, default=8.0)
        parser.add_argument("--tracking-acceleration-limit", type=float, default=4.0)
        parser.add_argument(
            "--excitation-fraction",
            type=float,
            default=0.01,
            help="maximum identification excitation as a fraction of rated joint torque",
        )
        parser.add_argument("--random-hold-steps", type=int, default=1)

        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--num-epochs", type=int, default=500)
        parser.add_argument("--batch-size", type=int, default=256)
        parser.add_argument("--eval-batch-size", type=int, default=256)
        parser.add_argument("--eval-interval", type=int, default=5)
        parser.add_argument("--pre-length", type=int, default=10)
        parser.add_argument("--loss-name", choices=["mse", "mae", "nmse"], default="mse")
        parser.add_argument("--gamma", type=float, default=0.98)
        parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        parser.add_argument("--use-stable", action="store_true")
        parser.add_argument("--u-z", action="store_true")
        parser.add_argument("--MPC-type", choices=["delta_mpc", "mpc"], default="delta_mpc", dest="MPC_type")
        parser.add_argument("--mpc-horizon", type=int, default=10)
        parser.add_argument("--torque-rate-fraction", type=float, default=0.015)
        parser.add_argument("--smoke", action="store_true", help="use a tiny data/training configuration")
        self.args = parser.parse_args(argv)
        self._derive()

    def _derive(self) -> None:
        project_root = Path(__file__).resolve().parent
        self.args.env = "UR5e"
        self.args.x_dim = 15
        self.args.u_dim = 6
        self.args.xml_path = str(project_root / "assets" / "ur5e" / "scene_torque.xml")
        self.args.dataset_dir = str(project_root / "datasets" / "ur5e_torque")
        self.args.output_root = str(project_root / "runs" / "ur5e_torque")
        self.args.output_dir = str(Path(self.args.output_root) / self.args.model)

        if self.args.smoke:
            self.args.train_samples = 8
            self.args.val_samples = 4
            self.args.test_samples = 4
            self.args.train_steps = 12
            self.args.test_steps = 12
            self.args.num_epochs = 2
            self.args.pre_length = 4
            self.args.batch_size = 4
            self.args.eval_batch_size = 4

        if self.args.pre_length >= self.args.train_steps:
            raise ValueError("pre_length must be smaller than train_steps")
        if self.args.initial_position_span < 0.0 or self.args.initial_velocity_span < 0.0:
            raise ValueError("initial state spans must be non-negative")
        if self.args.waypoint_count < 2 or self.args.waypoint_velocity_limit <= 0.0:
            raise ValueError("waypoint_count must be >=2 and waypoint_velocity_limit must be positive")
        if self.args.tracking_kp < 0.0 or self.args.tracking_kd < 0.0:
            raise ValueError("tracking gains must be non-negative")
        if self.args.tracking_acceleration_limit <= 0.0:
            raise ValueError("tracking_acceleration_limit must be positive")
        if not 0.0 <= self.args.excitation_fraction <= 1.0:
            raise ValueError("excitation_fraction must lie in [0,1]")
        if self.args.random_hold_steps < 1:
            raise ValueError("random_hold_steps must be positive")
        if not 0.0 < self.args.torque_rate_fraction <= 1.0:
            raise ValueError("torque_rate_fraction must lie in (0,1]")
        self.args.layers = [15, 64, 64, 32]
        self.args.x_blocks = [2, 2]
        self.args.x_channels = [12, 16]
        self.args.x_hiddens = [64, 128]
        # Retained for constructor compatibility; torque is passed directly.
        self.args.u_blocks = [1]
        self.args.u_channels = [6]
        self.args.u_hiddens = [32]

    def __getattr__(self, name: str):
        return getattr(self.args, name)
