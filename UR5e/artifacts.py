"""Version and validate model artifacts against the control-input semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from UR5e.UR5e_DataCollection import ACTION_DEFINITION, DATASET_VERSION


MODEL_MANIFEST = "model_manifest.json"


def write_model_manifest(
    output_directory: str | Path,
    model_name: str,
    rated_torque_nm: np.ndarray,
) -> Path:
    """Write the physical input definition next to a trained checkpoint."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    torque = np.asarray(rated_torque_nm, dtype=np.float64)
    if torque.shape != (6,) or np.any(torque <= 0.0):
        raise ValueError("rated_torque_nm must be positive with shape (6,)")
    manifest = {
        "schema_version": 1,
        "model": model_name,
        "dataset_version": DATASET_VERSION,
        "action_definition": ACTION_DEFINITION,
        "action_units_before_normalization": "N m",
        "rated_torque_nm": torque.tolist(),
    }
    path = output / MODEL_MANIFEST
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def validate_model_manifest(
    checkpoint: str | Path,
    expected_model: str,
    rated_torque_nm: np.ndarray | None = None,
) -> dict[str, Any]:
    """Reject checkpoints trained with the former residual-torque definition."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    path = checkpoint_path.parent / MODEL_MANIFEST
    if not path.is_file():
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no {MODEL_MANIFEST}; it may have been trained "
            "with the obsolete residual-torque input. Regenerate data and retrain the model."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "model": expected_model,
        "dataset_version": DATASET_VERSION,
        "action_definition": ACTION_DEFINITION,
    }
    mismatches = {
        key: {"expected": value, "found": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if rated_torque_nm is not None:
        expected_torque = np.asarray(rated_torque_nm, dtype=np.float64)
        found_torque = np.asarray(manifest.get("rated_torque_nm", []), dtype=np.float64)
        if expected_torque.shape != (6,) or found_torque.shape != (6,) or not np.allclose(
            expected_torque,
            found_torque,
        ):
            mismatches["rated_torque_nm"] = {
                "expected": expected_torque.tolist(),
                "found": found_torque.tolist(),
            }
    if mismatches:
        raise RuntimeError(
            f"Checkpoint input semantics do not match complete joint-torque control: {mismatches}. "
            "Regenerate data and retrain the model."
        )
    return manifest
