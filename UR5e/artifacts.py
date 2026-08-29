"""Version and validate model weights together with their coordinate system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from UR5e.normalization import NormalizationStats, file_sha256
from UR5e.UR5e_DataCollection import ACTION_DEFINITION, DATASET_VERSION

MODEL_MANIFEST = "model_manifest.json"
MODEL_MANIFEST_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 2


def _model_signature(model: torch.nn.Module) -> dict[str, Any]:
    """Capture architecture switches that tensor shapes alone cannot validate."""

    return {
        "class": type(model).__name__,
        "x_dim": int(model.x_dim),
        "u_dim": int(model.u_dim),
        "lifted_dim": int(model.Nkoopman),
        "use_stable": bool(model.use_stable),
        "use_decoder": bool(model.use_decoder),
        "u_z": bool(getattr(model, "u_z", False)),
    }


def save_model_checkpoint(
    checkpoint: str | Path,
    model: torch.nn.Module,
    model_name: str,
    normalization: NormalizationStats,
    epoch: int,
    best_validation: dict[str, float],
    training_config: dict[str, Any],
) -> Path:
    """Save weights and the exact model coordinates as one portable bundle."""

    path = Path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model_name,
        "model_signature": _model_signature(model),
        "model_state_dict": model.state_dict(),
        "normalization": normalization.to_dict(),
        "normalization_fingerprint": normalization.fingerprint(),
        "epoch": int(epoch),
        "best_validation": {key: float(value) for key, value in best_validation.items()},
        "training_config": training_config,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def write_model_manifest(
    output_directory: str | Path,
    model_name: str,
    rated_torque_nm: np.ndarray,
    normalization: NormalizationStats,
    checkpoint_name: str = "best_model.pt",
) -> Path:
    """Write independently inspectable semantics next to a checkpoint bundle."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Cannot write a manifest before checkpoint exists: {checkpoint}")
    torque = np.asarray(rated_torque_nm, dtype=np.float64)
    if torque.shape != (6,) or np.any(torque <= 0.0):
        raise ValueError("rated_torque_nm must be positive with shape (6,)")
    if not np.allclose(torque, normalization.rated_torque_nm):
        raise ValueError("Manifest torque limits differ from normalization statistics")
    manifest = {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model_name,
        "dataset_version": DATASET_VERSION,
        "state_definition": "[ee_xyz,q,dq] standardized with training split statistics",
        "normalization_fingerprint": normalization.fingerprint(),
        "normalization_source_train_sha256": normalization.source_train_sha256,
        "action_definition": ACTION_DEFINITION,
        "action_units_before_normalization": "N m",
        "rated_torque_nm": torque.tolist(),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
    }
    path = output / MODEL_MANIFEST
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def validate_model_manifest(
    checkpoint: str | Path,
    expected_model: str,
    rated_torque_nm: np.ndarray | None = None,
    expected_normalization: NormalizationStats | None = None,
) -> dict[str, Any]:
    """Reject obsolete weights or any mismatch in state/action coordinates."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    path = checkpoint_path.parent / MODEL_MANIFEST
    if not path.is_file():
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no {MODEL_MANIFEST}; it may use obsolete "
            "residual-torque inputs or lack training-state statistics. Regenerate data and retrain."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": expected_model,
        "dataset_version": DATASET_VERSION,
        "action_definition": ACTION_DEFINITION,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    mismatches = {
        key: {"expected": value, "found": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if rated_torque_nm is not None:
        expected_torque = np.asarray(rated_torque_nm, dtype=np.float64)
        found_torque = np.asarray(manifest.get("rated_torque_nm", []), dtype=np.float64)
        if (
            expected_torque.shape != (6,)
            or found_torque.shape != (6,)
            or not np.allclose(expected_torque, found_torque)
        ):
            mismatches["rated_torque_nm"] = {
                "expected": expected_torque.tolist(),
                "found": found_torque.tolist(),
            }
    if expected_normalization is not None:
        fingerprint = expected_normalization.fingerprint()
        if manifest.get("normalization_fingerprint") != fingerprint:
            mismatches["normalization_fingerprint"] = {
                "expected": fingerprint,
                "found": manifest.get("normalization_fingerprint"),
            }
    if mismatches:
        raise RuntimeError(
            f"Checkpoint semantics do not match the requested model/data/control coordinates: {mismatches}. "
            "Regenerate data and retrain the model."
        )
    return manifest


def load_model_checkpoint(
    checkpoint: str | Path,
    model: torch.nn.Module,
    expected_model: str,
    rated_torque_nm: np.ndarray,
    map_location: str | torch.device,
    expected_normalization: NormalizationStats | None = None,
) -> tuple[NormalizationStats, dict[str, Any]]:
    """Validate, restore weights, and return the checkpoint-owned normalizer."""

    manifest = validate_model_manifest(
        checkpoint,
        expected_model,
        rated_torque_nm,
        expected_normalization=expected_normalization,
    )
    payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Checkpoint {checkpoint} is not a schema-{CHECKPOINT_SCHEMA_VERSION} model bundle; retrain it."
        )
    if payload.get("model") != expected_model:
        raise RuntimeError(f"Checkpoint contains model {payload.get('model')!r}, expected {expected_model!r}")
    expected_signature = _model_signature(model)
    if payload.get("model_signature") != expected_signature:
        raise RuntimeError(
            f"Checkpoint architecture does not match the constructed model: "
            f"expected {expected_signature}, found {payload.get('model_signature')}"
        )
    normalization = NormalizationStats.from_dict(payload["normalization"])
    fingerprint = normalization.fingerprint()
    if payload.get("normalization_fingerprint") != fingerprint:
        raise RuntimeError("Checkpoint normalization payload has an invalid fingerprint")
    if manifest.get("normalization_fingerprint") != fingerprint:
        raise RuntimeError("Checkpoint and model manifest contain different normalization statistics")
    if expected_normalization is not None and expected_normalization.fingerprint() != fingerprint:
        raise RuntimeError("Checkpoint normalization does not match the loaded dataset")
    if not np.allclose(normalization.rated_torque_nm, rated_torque_nm):
        raise RuntimeError("Checkpoint rated torque does not match the control environment")
    model.load_state_dict(payload["model_state_dict"])
    metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    return normalization, metadata
