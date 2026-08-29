"""Collect UR5e torque data, train Koopman models, and evaluate rollouts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from args import Args
from models.init_model import init_model
from models.losses import rollout_loss, rollout_prediction
from UR5e.artifacts import load_model_checkpoint, save_model_checkpoint, write_model_manifest
from UR5e.normalization import NormalizationStats
from UR5e.UR5e_DataCollection import UR5eDataGenerator

CORE_MODELS = ("DKUC", "DBKN", "IKN", "IBKN")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, normalization: NormalizationStats) -> dict[str, float]:
    model.eval()
    physical_sse = np.zeros(4, dtype=np.float64)
    physical_count = np.zeros(4, dtype=np.int64)
    normalized_sse = np.zeros(4, dtype=np.float64)
    normalized_count = np.zeros(4, dtype=np.int64)
    for batch in loader:
        result = rollout_prediction(batch, model, normalization)
        for error, sums, counts in (
            (result["error_phys"], physical_sse, physical_count),
            (result["error_model"], normalized_sse, normalized_count),
        ):
            groups = (error, error[:, :, :3], error[:, :, 3:9], error[:, :, 9:15])
            for index, group in enumerate(groups):
                sums[index] += float(group.square().sum().cpu())
                counts[index] += group.numel()
    physical = np.sqrt(physical_sse / physical_count)
    normalized = np.sqrt(normalized_sse / normalized_count)
    return {
        "rmse": float(physical[0]),
        "ee_rmse_m": float(physical[1]),
        "q_rmse_rad": float(physical[2]),
        "dq_rmse_rad_s": float(physical[3]),
        "normalized_rmse": float(normalized[0]),
        "normalized_ee_rmse": float(normalized[1]),
        "normalized_q_rmse": float(normalized[2]),
        "normalized_dq_rmse": float(normalized[3]),
        "normalized_score": float(normalized[1] + 4.0 * normalized[2] + normalized[3]),
    }


def fit_model(args: Args, data: UR5eDataGenerator, model_name: str) -> dict[str, object]:
    if data.normalization is None:
        raise RuntimeError("Dataset normalization has not been initialized")
    normalization = data.normalization
    args.args.model = model_name
    model = init_model(args)
    train_loader, val_loader = data.get_train_loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    output_dir = Path(args.output_root) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        losses: dict[str, list[float]] = {}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            result = rollout_loss(
                batch,
                model,
                args.loss_name,
                args.gamma,
                args.pre_length,
                normalization=normalization,
                latent_loss_weight=args.latent_loss_weight,
                reconstruction_loss_weight=args.reconstruction_loss_weight,
                delta_loss_weight=args.delta_loss_weight,
                stability_loss_weight=args.stability_loss_weight,
                bilinear_l1_weight=args.bilinear_l1_weight,
            )
            result["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in result.items():
                losses.setdefault(key, []).append(float(value.detach().cpu()))
        record: dict[str, float | int] = {"epoch": epoch}
        record.update({f"train_{key}": float(np.mean(values)) for key, values in losses.items()})
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.num_epochs - 1:
            validation = evaluate(model, val_loader, normalization)
            record.update({f"val_{key}": value for key, value in validation.items()})
            if validation["normalized_score"] < best:
                best = validation["normalized_score"]
                save_model_checkpoint(
                    output_dir / "best_model.pt",
                    model,
                    model_name,
                    normalization,
                    epoch,
                    validation,
                    dict(vars(args.args)),
                )
                write_model_manifest(output_dir, model_name, data.env.torque_limits, normalization)
        history.append(record)
        print({"model": model_name, **record})

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "model": model_name,
        "best_validation_normalized_score": best,
        "parameters": sum(p.numel() for p in model.parameters()),
        "normalization_fingerprint": normalization.fingerprint(),
    }


def test_model(args: Args, data: UR5eDataGenerator, model_name: str) -> dict[str, object]:
    if data.normalization is None:
        raise RuntimeError("Dataset normalization has not been initialized")
    args.args.model = model_name
    model = init_model(args)
    checkpoint = Path(args.output_root) / model_name / "best_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Train {model_name} before testing: {checkpoint}")
    normalization, _ = load_model_checkpoint(
        checkpoint,
        model,
        model_name,
        data.env.torque_limits,
        args.device,
        expected_normalization=data.normalization,
    )
    test_types = tuple(data.test_data_dict) if args.test_type == "all" else (args.test_type,)
    results = {test_type: evaluate(model, data.get_test_loader(test_type), normalization) for test_type in test_types}
    output = Path(args.output_root) / model_name / "test_metrics.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return {"model": model_name, "tests": results}


def main(argv: list[str] | None = None) -> None:
    args = Args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    set_seed(args.seed)
    data = UR5eDataGenerator(args)
    try:
        data.generate_and_save_data(force=args.force_data)
        if args.mode == "collect":
            print({"dataset": args.dataset_dir, "control_mode": "direct_joint_torque"})
            return
        models = CORE_MODELS if args.model == "all" else (args.model,)
        results = []
        for model_name in models:
            results.append(
                fit_model(args, data, model_name) if args.mode == "train" else test_model(args, data, model_name)
            )
        print(json.dumps(results, indent=2))
    finally:
        data.close()


if __name__ == "__main__":
    main()
