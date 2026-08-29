"""Versioned normalization statistics shared by data, models, and control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

NORMALIZATION_SCHEMA_VERSION = 1
NORMALIZATION_FILE = "normalization.json"
STATE_NAMES = (
    "ee_x",
    "ee_y",
    "ee_z",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "dq1",
    "dq2",
    "dq3",
    "dq4",
    "dq5",
    "dq6",
)
STATE_UNITS = ("m",) * 3 + ("rad",) * 6 + ("rad/s",) * 6


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without loading a large array twice."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class NormalizationStats:
    """All coordinate transforms required to reproduce a learned model."""

    state_mean: np.ndarray
    state_std: np.ndarray
    state_delta_std: np.ndarray
    rated_torque_nm: np.ndarray
    source_train_shape: tuple[int, ...]
    source_train_sha256: str
    std_floor: float = 1e-6
    state_names: tuple[str, ...] = STATE_NAMES
    state_units: tuple[str, ...] = STATE_UNITS

    def __post_init__(self) -> None:
        arrays = {
            "state_mean": np.asarray(self.state_mean, dtype=np.float64),
            "state_std": np.asarray(self.state_std, dtype=np.float64),
            "state_delta_std": np.asarray(self.state_delta_std, dtype=np.float64),
            "rated_torque_nm": np.asarray(self.rated_torque_nm, dtype=np.float64),
        }
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if arrays["state_mean"].shape != (15,):
            raise ValueError("state_mean must have shape (15,)")
        if arrays["state_std"].shape != (15,) or np.any(arrays["state_std"] <= 0.0):
            raise ValueError("state_std must be positive with shape (15,)")
        if arrays["state_delta_std"].shape != (15,) or np.any(arrays["state_delta_std"] <= 0.0):
            raise ValueError("state_delta_std must be positive with shape (15,)")
        if arrays["rated_torque_nm"].shape != (6,) or np.any(arrays["rated_torque_nm"] <= 0.0):
            raise ValueError("rated_torque_nm must be positive with shape (6,)")
        if tuple(self.state_names) != STATE_NAMES or tuple(self.state_units) != STATE_UNITS:
            raise ValueError("state order or units do not match [ee_xyz,q,dq]")
        if not np.isfinite(self.std_floor) or self.std_floor <= 0.0:
            raise ValueError("std_floor must be positive")
        if not self.source_train_sha256:
            raise ValueError("source_train_sha256 must not be empty")

    @classmethod
    def fit(
        cls,
        train_array: np.ndarray,
        rated_torque_nm: np.ndarray,
        source_train_sha256: str,
        std_floor: float = 1e-6,
    ) -> NormalizationStats:
        """Fit per-coordinate statistics using the training split only."""

        data = np.asarray(train_array)
        if data.ndim != 3 or data.shape[-1] != 21 or data.shape[1] < 2:
            raise ValueError("train_array must have shape (trajectories, steps+1, 21)")
        state = np.asarray(data[:, :, 6:21], dtype=np.float64)
        flat_state = state.reshape(-1, 15)
        flat_delta = np.diff(state, axis=1).reshape(-1, 15)
        return cls(
            state_mean=flat_state.mean(axis=0),
            state_std=np.maximum(flat_state.std(axis=0), std_floor),
            state_delta_std=np.maximum(flat_delta.std(axis=0), std_floor),
            rated_torque_nm=np.asarray(rated_torque_nm, dtype=np.float64),
            source_train_shape=tuple(int(value) for value in data.shape),
            source_train_sha256=source_train_sha256,
            std_floor=float(std_floor),
        )

    @classmethod
    def identity(cls, rated_torque_nm: np.ndarray) -> NormalizationStats:
        """Construct an explicit identity transform for isolated unit tests."""

        return cls(
            state_mean=np.zeros(15),
            state_std=np.ones(15),
            state_delta_std=np.ones(15),
            rated_torque_nm=np.asarray(rated_torque_nm, dtype=np.float64),
            source_train_shape=(0, 0, 21),
            source_train_sha256="identity",
        )

    @property
    def normalized_delta_std(self) -> np.ndarray:
        return np.maximum(self.state_delta_std / self.state_std, self.std_floor)

    def _state_tensor(self, values: torch.Tensor, field: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(field, dtype=values.dtype, device=values.device)

    def normalize_state(self, values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(values, torch.Tensor):
            return (values - self._state_tensor(values, self.state_mean)) / self._state_tensor(values, self.state_std)
        array = np.asarray(values)
        return (array - self.state_mean) / self.state_std

    def denormalize_state(self, values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values * self._state_tensor(values, self.state_std) + self._state_tensor(values, self.state_mean)
        array = np.asarray(values)
        return array * self.state_std + self.state_mean

    def normalize_torque(self, values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(values, torch.Tensor):
            scale = torch.as_tensor(self.rated_torque_nm, dtype=values.dtype, device=values.device)
            return values / scale
        return np.asarray(values) / self.rated_torque_nm

    def denormalize_torque(self, values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(values, torch.Tensor):
            scale = torch.as_tensor(self.rated_torque_nm, dtype=values.dtype, device=values.device)
            return values * scale
        return np.asarray(values) * self.rated_torque_nm

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "state_definition": "[ee_xyz,q,dq]",
            "state_names": list(self.state_names),
            "state_units": list(self.state_units),
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "state_delta_std": self.state_delta_std.tolist(),
            "normalized_delta_std": self.normalized_delta_std.tolist(),
            "std_floor": self.std_floor,
            "action_definition": "tau_applied/tau_rated",
            "rated_torque_nm": self.rated_torque_nm.tolist(),
            "source_split": "train",
            "source_train_shape": list(self.source_train_shape),
            "source_train_sha256": self.source_train_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NormalizationStats:
        if payload.get("schema_version") != NORMALIZATION_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported normalization schema {payload.get('schema_version')!r}; "
                f"expected {NORMALIZATION_SCHEMA_VERSION}"
            )
        return cls(
            state_mean=np.asarray(payload["state_mean"], dtype=np.float64),
            state_std=np.asarray(payload["state_std"], dtype=np.float64),
            state_delta_std=np.asarray(payload["state_delta_std"], dtype=np.float64),
            rated_torque_nm=np.asarray(payload["rated_torque_nm"], dtype=np.float64),
            source_train_shape=tuple(int(value) for value in payload["source_train_shape"]),
            source_train_sha256=str(payload["source_train_sha256"]),
            std_floor=float(payload["std_floor"]),
            state_names=tuple(payload["state_names"]),
            state_units=tuple(payload["state_units"]),
        )

    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        payload = self.to_dict()
        payload["fingerprint"] = self.fingerprint()
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = payload.pop("fingerprint", None)
        stats = cls.from_dict(payload)
        if expected != stats.fingerprint():
            raise RuntimeError(f"Normalization fingerprint mismatch in {path}")
        return stats
