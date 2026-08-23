"""UR5e torque trajectory generation using the original project data layout.

Each saved row is ``[u_t, x_t]`` where ``u_t`` is the six-dimensional residual
joint torque in N m and ``x_t=[q_t,dq_t]``.  Applying row ``t``'s action produces
row ``t+1``'s state.  The loaders normalize action by the rated joint torques;
the `.npy` files always retain physical units.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from UR5e.UR5e_Env import UR5eTorqueConfig, UR5eTorqueEnv


class TrajectoryCollator:
    """Split trajectory arrays into model state and normalized torque tensors."""

    def __init__(self, x_dim: int, u_dim: int, action_scale: np.ndarray, device: str):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32)
        self.device = device

    def __call__(self, batch_list: list[tuple[torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch = torch.stack([item[0] for item in batch_list])
        u = batch[:, :, : self.u_dim] / self.action_scale
        x = batch[:, :, self.u_dim : self.u_dim + self.x_dim]
        return {"x": x.to(self.device), "u": u.to(self.device)}


class TorqueSignalGenerator:
    """Deterministic bounded random, sinusoidal, or chirp torque excitation."""

    def __init__(self, signal_type: str, trajectories: int, limits: np.ndarray, seed: int):
        if signal_type not in {"random", "sin", "chirp"}:
            raise ValueError(f"Unsupported input type: {signal_type}")
        self.signal_type = signal_type
        self.limits = np.asarray(limits, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.frequency = self.rng.uniform(0.15, 0.75, size=(trajectories, 6))
        self.phase = self.rng.uniform(0.0, 2.0 * np.pi, size=(trajectories, 6))
        self.amplitude = self.rng.uniform(0.35, 0.95, size=(trajectories, 6)) * self.limits
        self.random_hold = np.zeros((trajectories, 6))

    def __call__(self, step: int, trajectory: int, dt: float, total_steps: int) -> np.ndarray:
        if self.signal_type == "random":
            if step % 5 == 0:
                self.random_hold[trajectory] = self.rng.uniform(-self.limits, self.limits)
            return self.random_hold[trajectory].copy()
        time = step * dt
        if self.signal_type == "sin":
            phase = 2.0 * np.pi * self.frequency[trajectory] * time + self.phase[trajectory]
        else:
            ratio = step / max(total_steps, 1)
            frequency = self.frequency[trajectory] + ratio * 0.9
            phase = 2.0 * np.pi * frequency * time + self.phase[trajectory]
        return self.amplitude[trajectory] * np.sin(phase)


class UR5eDataGenerator:
    """Generate/load torque trajectories and expose PyTorch data loaders."""

    def __init__(self, args) -> None:
        self.args = args
        config = UR5eTorqueConfig(
            xml_path=args.xml_path,
            physics_timestep=args.physics_timestep,
            frame_skip=args.frame_skip,
            residual_torque_fraction=args.residual_torque_fraction,
            gravity_compensation_scale=args.gravity_compensation_scale,
        )
        self.env = UR5eTorqueEnv(config)
        self.collate_fn = TrajectoryCollator(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            action_scale=self.env.torque_limits,
            device=args.device,
        )
        self.train_data: np.ndarray | None = None
        self.val_data: np.ndarray | None = None
        self.test_data_dict: dict[str, np.ndarray] = {}

    def generate_trajectories(self, trajectories: int, steps: int, input_type: str, seed: int) -> np.ndarray:
        signal = TorqueSignalGenerator(input_type, trajectories, self.env.residual_limits, seed)
        result = np.empty((trajectories, steps + 1, self.args.u_dim + self.args.x_dim), dtype=np.float32)
        accepted = 0
        attempts = 0
        max_attempts = max(trajectories * 20, 20)
        progress = tqdm(total=trajectories, desc=f"UR5e {input_type}")
        while accepted < trajectories:
            if attempts >= max_attempts:
                raise RuntimeError(f"Only accepted {accepted}/{trajectories} safe trajectories")
            trajectory_seed = seed + attempts
            state, _ = self.env.reset(seed=trajectory_seed)
            candidate = np.empty((steps + 1, self.args.u_dim + self.args.x_dim), dtype=np.float32)
            safe = True
            for step in range(steps + 1):
                action = signal(step, accepted, self.env.config.control_timestep, steps)
                candidate[step] = np.concatenate((action, state))
                if step < steps:
                    state, _, terminated, _, _ = self.env.step(action)
                    if terminated:
                        safe = False
                        break
            attempts += 1
            if not safe:
                continue
            result[accepted] = candidate
            accepted += 1
            progress.update(1)
        progress.close()
        return result

    def generate_and_save_data(self, force: bool = False) -> None:
        output = Path(self.args.dataset_dir)
        output.mkdir(parents=True, exist_ok=True)
        specs = {
            "train": (self.args.train_samples, self.args.train_steps, "random", self.args.seed),
            "val": (self.args.val_samples, self.args.test_steps, "random", self.args.seed + 10_000),
            "test_random": (self.args.test_samples, self.args.test_steps, "random", self.args.seed + 20_000),
            "test_sin": (self.args.test_samples, self.args.test_steps, "sin", self.args.seed + 30_000),
            "test_chirp": (self.args.test_samples, self.args.test_steps, "chirp", self.args.seed + 40_000),
        }
        arrays: dict[str, np.ndarray] = {}
        for name, (count, steps, signal, seed) in specs.items():
            path = output / f"{name}.npy"
            arrays[name] = (
                self.generate_trajectories(count, steps, signal, seed)
                if force or not path.exists()
                else np.load(path)
            )
            if force or not path.exists():
                np.save(path, arrays[name])

        self.train_data = arrays["train"]
        self.val_data = arrays["val"]
        self.test_data_dict = {name.removeprefix("test_"): value for name, value in arrays.items() if name.startswith("test_")}
        manifest = {
            "robot": "UR5e",
            "control_mode": "direct_joint_torque",
            "state": "[q,dq]",
            "state_units": ["rad"] * 6 + ["rad/s"] * 6,
            "action": "residual_joint_torque",
            "action_units": ["N m"] * 6,
            "rated_torque_nm": self.env.torque_limits.tolist(),
            "residual_limits_nm": self.env.residual_limits.tolist(),
            "control_timestep_s": self.env.config.control_timestep,
            "seed": self.args.seed,
            "arrays": {name: list(value.shape) for name, value in arrays.items()},
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _loader(self, array: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(torch.as_tensor(array, dtype=torch.float32))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=shuffle and len(dataset) >= batch_size,
            collate_fn=self.collate_fn,
        )

    def get_train_loader(self) -> tuple[DataLoader, DataLoader]:
        if self.train_data is None or self.val_data is None:
            raise RuntimeError("Call generate_and_save_data() before requesting loaders")
        return (
            self._loader(self.train_data, self.args.batch_size, True),
            self._loader(self.val_data, self.args.eval_batch_size, False),
        )

    def get_test_loader(self, test_type: str) -> DataLoader:
        if test_type not in self.test_data_dict:
            raise KeyError(f"Unknown test type {test_type!r}; choose from {sorted(self.test_data_dict)}")
        return self._loader(self.test_data_dict[test_type], self.args.eval_batch_size, False)

    def close(self) -> None:
        self.env.close()
