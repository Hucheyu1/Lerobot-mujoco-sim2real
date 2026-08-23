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
from UR5e.UR5e_DataCollection import UR5eDataGenerator


CORE_MODELS = ("DKUC", "DBKN", "IKN", "IBKN")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader) -> dict[str, float]:
    model.eval()
    values = {"rmse": [], "q_rmse": [], "dq_rmse": []}
    for batch in loader:
        result = rollout_prediction(batch, model)
        for key in values:
            values[key].append(float(result[key].cpu()))
    return {key: float(np.mean(item)) for key, item in values.items()}


def fit_model(args: Args, data: UR5eDataGenerator, model_name: str) -> dict[str, object]:
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
        losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            result = rollout_loss(batch, model, args.loss_name, args.gamma, args.pre_length)
            result["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(result["total_loss"].detach().cpu()))
        record: dict[str, float | int] = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.num_epochs - 1:
            validation = evaluate(model, val_loader)
            record.update({f"val_{key}": value for key, value in validation.items()})
            if validation["rmse"] < best:
                best = validation["rmse"]
                torch.save(model.state_dict(), output_dir / "best_model.pt")
        history.append(record)
        print({"model": model_name, **record})

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"model": model_name, "best_validation_rmse": best, "parameters": sum(p.numel() for p in model.parameters())}


def test_model(args: Args, data: UR5eDataGenerator, model_name: str) -> dict[str, object]:
    args.args.model = model_name
    model = init_model(args)
    checkpoint = Path(args.output_root) / model_name / "best_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Train {model_name} before testing: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    test_types = tuple(data.test_data_dict) if args.test_type == "all" else (args.test_type,)
    results = {test_type: evaluate(model, data.get_test_loader(test_type)) for test_type in test_types}
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
            results.append(fit_model(args, data, model_name) if args.mode == "train" else test_model(args, data, model_name))
        print(json.dumps(results, indent=2))
    finally:
        data.close()


if __name__ == "__main__":
    main()
