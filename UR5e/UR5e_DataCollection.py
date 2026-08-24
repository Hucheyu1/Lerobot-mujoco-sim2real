"""Closed-loop UR5e torque data collection for Koopman model learning.

The reference implementation in ``Adaptive-koopman-main`` generates random
joint-space waypoints, interpolates them with a cubic spline, and follows the
result with a computed-torque controller. This module keeps that structure
while adapting it to the UR5e residual-torque interface: ``random``, ``sin``
and ``chirp`` are bounded identification excitations added to the closed-loop
tracking torque, rather than unsafe open-loop commands.

Each saved row is ``[u_t, x_t]``. Here ``u_t`` is the actual clipped
six-dimensional residual joint torque in N m and
``x_t=[p_ee(t), q_t, dq_t]``. Applying row ``t``'s action produces row
``t+1``'s state. Model loaders normalize action by rated joint torque; the
``.npy`` files retain physical units.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from UR5e.UR5e_Env import UR5eTorqueConfig, UR5eTorqueEnv


DATASET_VERSION = "ur5e_computed_torque_v1"
COLLECTION_MODE = "computed_torque_waypoints_with_excitation"


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


class TorqueExcitationGenerator:
    """Generate a small deterministic excitation for one closed-loop trajectory."""

    def __init__(
        self,
        signal_type: str,
        residual_limits: np.ndarray,
        excitation_fraction: float,
        random_hold_steps: int,
        seed: int,
        control_timestep: float,
        total_steps: int,
    ) -> None:
        if signal_type not in {"random", "sin", "chirp"}:
            raise ValueError(f"Unsupported input type: {signal_type}")
        self.signal_type = signal_type
        self.rng = np.random.default_rng(seed)
        self.max_amplitude = excitation_fraction * np.asarray(residual_limits, dtype=np.float64)
        self.random_hold_steps = random_hold_steps
        self.control_timestep = control_timestep
        self.duration = max(total_steps * control_timestep, control_timestep)
        self.frequency = self.rng.uniform(0.15, 0.75, size=6)
        self.phase = self.rng.uniform(0.0, 2.0 * np.pi, size=6)
        self.amplitude = self.rng.uniform(0.35, 0.95, size=6) * self.max_amplitude
        self.random_value = np.zeros(6, dtype=np.float64)

    def __call__(self, step: int) -> np.ndarray:
        if self.signal_type == "random":
            if step % self.random_hold_steps == 0:
                self.random_value = self.rng.uniform(-self.max_amplitude, self.max_amplitude)
            return self.random_value.copy()

        time = step * self.control_timestep
        if self.signal_type == "sin":
            phase = 2.0 * np.pi * self.frequency * time + self.phase
        else:
            # Integrate a linear frequency sweep instead of multiplying the
            # instantaneous frequency by time, which would double its slope.
            chirp_rate = 0.9 / self.duration
            phase = 2.0 * np.pi * (self.frequency * time + 0.5 * chirp_rate * time**2) + self.phase
        return self.amplitude * np.sin(phase)


class RandomWaypointReference:
    """Safe cubic-spline joint reference following the reference repository."""

    def __init__(
        self,
        env: UR5eTorqueEnv,
        initial_state: np.ndarray,
        steps: int,
        waypoint_count: int,
        waypoint_velocity_limit: float,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        duration = max(steps * env.config.control_timestep, env.config.control_timestep)
        count = max(2, min(waypoint_count, steps + 1))
        waypoint_times = np.linspace(0.0, duration, count)
        segment_durations = np.diff(waypoint_times)

        waypoints = np.empty((count, 6), dtype=np.float64)
        waypoints[0] = initial_state[3:9]
        joint_ranges = env.model.jnt_range[env.joint_ids].astype(np.float64, copy=True)
        limited = env.model.jnt_limited[env.joint_ids].astype(bool)
        lower = np.where(limited, joint_ranges[:, 0] + 0.05, -np.inf)
        upper = np.where(limited, joint_ranges[:, 1] - 0.05, np.inf)
        for index, segment_duration in enumerate(segment_durations, start=1):
            waypoint_velocity = rng.uniform(-waypoint_velocity_limit, waypoint_velocity_limit, size=6)
            waypoints[index] = np.clip(
                waypoints[index - 1] + waypoint_velocity * segment_duration,
                lower,
                upper,
            )

        initial_velocity = initial_state[9:15]
        spline = CubicSpline(
            waypoint_times,
            waypoints,
            axis=0,
            bc_type=((1, initial_velocity), (2, np.zeros(6))),
        )
        sample_times = np.linspace(0.0, duration, steps + 1)
        self.q = spline(sample_times, 0)
        self.dq = spline(sample_times, 1)
        self.ddq = spline(sample_times, 2)


class ComputedTorqueResidualController:
    """Computed-torque tracking adapted to the environment's residual action."""

    def __init__(self, env: UR5eTorqueEnv, kp: float, kd: float, acceleration_limit: float):
        self.env = env
        self.kp = kp
        self.kd = kd
        self.acceleration_limit = acceleration_limit

    def action(
        self,
        state: np.ndarray,
        q_reference: np.ndarray,
        dq_reference: np.ndarray,
        ddq_reference: np.ndarray,
        excitation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = state[3:9]
        dq = state[9:15]
        desired_acceleration = (
            ddq_reference
            + self.kd * (dq_reference - dq)
            + self.kp * (q_reference - q)
        )
        desired_acceleration = np.clip(
            desired_acceleration,
            -self.acceleration_limit,
            self.acceleration_limit,
        )
        target_total_torque = self.env.inverse_dynamics_torque(desired_acceleration)
        gravity_feedforward = self.env.config.gravity_compensation_scale * self.env.gravity_torque()
        raw_residual = target_total_torque - gravity_feedforward + excitation
        clipped_residual = np.clip(
            raw_residual,
            -self.env.residual_limits,
            self.env.residual_limits,
        )
        saturated_components = np.abs(raw_residual - clipped_residual) > 1e-10
        return clipped_residual, saturated_components


class UR5eDataGenerator:
    """Generate/load closed-loop torque trajectories and PyTorch data loaders."""

    def __init__(self, args) -> None:
        self.args = args
        config = UR5eTorqueConfig(
            xml_path=args.xml_path,
            physics_timestep=args.physics_timestep,
            frame_skip=args.frame_skip,
            residual_torque_fraction=args.residual_torque_fraction,
            gravity_compensation_scale=args.gravity_compensation_scale,
            initial_position_span=args.initial_position_span,
            initial_velocity_span=args.initial_velocity_span,
        )
        self.env = UR5eTorqueEnv(config)
        self.controller = ComputedTorqueResidualController(
            self.env,
            kp=args.tracking_kp,
            kd=args.tracking_kd,
            acceleration_limit=args.tracking_acceleration_limit,
        )
        self.collate_fn = TrajectoryCollator(
            x_dim=args.x_dim,
            u_dim=args.u_dim,
            action_scale=self.env.torque_limits,
            device=args.device,
        )
        self.train_data: np.ndarray | None = None
        self.val_data: np.ndarray | None = None
        self.test_data_dict: dict[str, np.ndarray] = {}
        self.last_generation_stats: dict[str, Any] = {}

    def _unsafe_reason(self, state: np.ndarray) -> str:
        if not np.all(np.isfinite(state)):
            return "nonfinite"
        q, dq = state[3:9], state[9:15]
        if np.any(np.abs(dq) > self.env.config.velocity_limit):
            return "velocity_limit"
        limited = self.env.model.jnt_limited[self.env.joint_ids].astype(bool)
        ranges = self.env.model.jnt_range[self.env.joint_ids]
        if np.any(q[limited] < ranges[limited, 0]) or np.any(q[limited] > ranges[limited, 1]):
            return "joint_limit"
        return "unknown"

    def generate_trajectories(
        self,
        trajectories: int,
        steps: int,
        input_type: str,
        seed: int,
    ) -> np.ndarray:
        result = np.empty(
            (trajectories, steps + 1, self.args.u_dim + self.args.x_dim),
            dtype=np.float32,
        )
        accepted = 0
        attempts = 0
        saturated_components = 0
        accepted_action_components = 0
        tracking_squared_error = 0.0
        tracking_samples = 0
        tracking_max_abs_error = 0.0
        rejection_counts = {
            "velocity_limit": 0,
            "joint_limit": 0,
            "nonfinite": 0,
            "unknown": 0,
        }
        max_attempts = max(trajectories * 5, 20)
        progress = tqdm(total=trajectories, desc=f"UR5e closed-loop {input_type}")
        try:
            while accepted < trajectories:
                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"Only accepted {accepted}/{trajectories} safe trajectories after "
                        f"{attempts} attempts; rejections={rejection_counts}"
                    )
                trajectory_seed = seed + attempts
                state, _ = self.env.reset(seed=trajectory_seed)
                reference = RandomWaypointReference(
                    self.env,
                    state,
                    steps,
                    self.args.waypoint_count,
                    self.args.waypoint_velocity_limit,
                    trajectory_seed + 1_000_000,
                )
                excitation = TorqueExcitationGenerator(
                    input_type,
                    self.env.residual_limits,
                    self.args.excitation_fraction,
                    self.args.random_hold_steps,
                    trajectory_seed + 2_000_000,
                    self.env.config.control_timestep,
                    steps,
                )
                candidate = np.empty(
                    (steps + 1, self.args.u_dim + self.args.x_dim),
                    dtype=np.float32,
                )
                candidate_saturated = 0
                candidate_tracking_squared_error = 0.0
                candidate_tracking_max_abs_error = 0.0
                safe = True
                rejection_reason = "unknown"
                for step in range(steps + 1):
                    action, saturation = self.controller.action(
                        state,
                        reference.q[step],
                        reference.dq[step],
                        reference.ddq[step],
                        excitation(step),
                    )
                    candidate[step] = np.concatenate((action, state))
                    candidate_saturated += int(np.count_nonzero(saturation))
                    tracking_error = state[3:9] - reference.q[step]
                    candidate_tracking_squared_error += float(np.sum(tracking_error**2))
                    candidate_tracking_max_abs_error = max(
                        candidate_tracking_max_abs_error,
                        float(np.max(np.abs(tracking_error))),
                    )
                    if step < steps:
                        state, _, terminated, _, _ = self.env.step(action)
                        if terminated:
                            safe = False
                            rejection_reason = self._unsafe_reason(state)
                            break
                attempts += 1
                if not safe:
                    rejection_counts[rejection_reason] += 1
                    continue
                result[accepted] = candidate
                accepted += 1
                saturated_components += candidate_saturated
                accepted_action_components += (steps + 1) * self.args.u_dim
                tracking_squared_error += candidate_tracking_squared_error
                tracking_samples += (steps + 1) * 6
                tracking_max_abs_error = max(
                    tracking_max_abs_error,
                    candidate_tracking_max_abs_error,
                )
                progress.update(1)
        finally:
            progress.close()

        self.last_generation_stats = {
            "requested": trajectories,
            "accepted": accepted,
            "attempts": attempts,
            "rejected": attempts - accepted,
            "acceptance_rate": accepted / attempts,
            "saturation_rate": saturated_components / max(accepted_action_components, 1),
            "tracking_q_rmse_rad": float(
                np.sqrt(tracking_squared_error / max(tracking_samples, 1))
            ),
            "tracking_q_max_abs_rad": tracking_max_abs_error,
            "rejection_counts": rejection_counts,
        }
        return result

    def _specs(self) -> dict[str, tuple[int, int, str, int]]:
        return {
            "train": (self.args.train_samples, self.args.train_steps, "random", self.args.seed),
            "val": (self.args.val_samples, self.args.test_steps, "random", self.args.seed + 10_000),
            "test_random": (self.args.test_samples, self.args.test_steps, "random", self.args.seed + 20_000),
            "test_sin": (self.args.test_samples, self.args.test_steps, "sin", self.args.seed + 30_000),
            "test_chirp": (self.args.test_samples, self.args.test_steps, "chirp", self.args.seed + 40_000),
        }

    def _collection_config(self) -> dict[str, Any]:
        return {
            "dataset_version": DATASET_VERSION,
            "collection_mode": COLLECTION_MODE,
            "train_samples": self.args.train_samples,
            "train_steps": self.args.train_steps,
            "val_samples": self.args.val_samples,
            "test_samples": self.args.test_samples,
            "test_steps": self.args.test_steps,
            "physics_timestep": self.args.physics_timestep,
            "frame_skip": self.args.frame_skip,
            "residual_torque_fraction": self.args.residual_torque_fraction,
            "gravity_compensation_scale": self.args.gravity_compensation_scale,
            "initial_position_span": self.args.initial_position_span,
            "initial_velocity_span": self.args.initial_velocity_span,
            "waypoint_count": self.args.waypoint_count,
            "waypoint_velocity_limit": self.args.waypoint_velocity_limit,
            "tracking_kp": self.args.tracking_kp,
            "tracking_kd": self.args.tracking_kd,
            "tracking_acceleration_limit": self.args.tracking_acceleration_limit,
            "excitation_fraction": self.args.excitation_fraction,
            "random_hold_steps": self.args.random_hold_steps,
            "seed": self.args.seed,
        }

    def _base_manifest(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "robot": "UR5e",
            "control_mode": "direct_joint_torque",
            "state": "[ee_xyz,q,dq]",
            "state_units": ["m"] * 3 + ["rad"] * 6 + ["rad/s"] * 6,
            "action": "clipped_residual_joint_torque",
            "action_units": ["N m"] * 6,
            "rated_torque_nm": self.env.torque_limits.tolist(),
            "residual_limits_nm": self.env.residual_limits.tolist(),
            "control_timestep_s": self.env.config.control_timestep,
            "collection": self._collection_config(),
            "arrays": {},
            "split_statistics": {},
        }

    @staticmethod
    def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _validate_existing_data(
        self,
        output: Path,
        specs: dict[str, tuple[int, int, str, int]],
    ) -> bool:
        manifest_path = output / "manifest.json"
        array_paths = {name: output / f"{name}.npy" for name in specs}
        any_arrays = any(path.exists() for path in array_paths.values())
        if not manifest_path.exists() and not any_arrays:
            return False
        if not manifest_path.exists():
            raise RuntimeError(
                f"Existing arrays in {output} have no dataset manifest and may use the old "
                "open-loop collector. Re-run with --force-data to regenerate all splits."
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise RuntimeError(
                f"Dataset generation in {output} is incomplete. Re-run with --force-data."
            )
        if manifest.get("collection") != self._collection_config():
            raise RuntimeError(
                f"Dataset configuration in {output} does not match the current collector. "
                "Re-run with --force-data."
            )
        for name, (count, steps, _, _) in specs.items():
            path = array_paths[name]
            if not path.exists():
                raise RuntimeError(f"Dataset split {path} is missing. Re-run with --force-data.")
            array = np.load(path, mmap_mode="r")
            expected_shape = (count, steps + 1, self.args.u_dim + self.args.x_dim)
            if array.shape != expected_shape:
                raise RuntimeError(
                    f"Dataset split {path} has shape {array.shape}, expected {expected_shape}. "
                    "Re-run with --force-data."
                )
        return True

    def _array_stats(self, array: np.ndarray) -> dict[str, Any]:
        action = array[:, :, : self.args.u_dim]
        state = array[:, :, self.args.u_dim :]
        return {
            "action_min": np.min(action, axis=(0, 1)).astype(float).tolist(),
            "action_max": np.max(action, axis=(0, 1)).astype(float).tolist(),
            "state_min": np.min(state, axis=(0, 1)).astype(float).tolist(),
            "state_max": np.max(state, axis=(0, 1)).astype(float).tolist(),
        }

    def generate_and_save_data(self, force: bool = False) -> None:
        output = Path(self.args.dataset_dir)
        output.mkdir(parents=True, exist_ok=True)
        specs = self._specs()
        reuse_existing = not force and self._validate_existing_data(output, specs)
        arrays: dict[str, np.ndarray] = {}

        if reuse_existing:
            arrays = {name: np.load(output / f"{name}.npy") for name in specs}
        else:
            manifest_path = output / "manifest.json"
            manifest = self._base_manifest(status="generating")
            self._write_manifest(manifest_path, manifest)
            for name, (count, steps, signal, seed) in specs.items():
                arrays[name] = self.generate_trajectories(count, steps, signal, seed)
                np.save(output / f"{name}.npy", arrays[name])
                manifest["arrays"][name] = list(arrays[name].shape)
                manifest["split_statistics"][name] = {
                    **self.last_generation_stats,
                    **self._array_stats(arrays[name]),
                }
                self._write_manifest(manifest_path, manifest)
            manifest["status"] = "complete"
            self._write_manifest(manifest_path, manifest)

        self.train_data = arrays["train"]
        self.val_data = arrays["val"]
        self.test_data_dict = {
            name.removeprefix("test_"): value
            for name, value in arrays.items()
            if name.startswith("test_")
        }

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
