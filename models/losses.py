"""Multi-step losses and evaluation for UR5e state ``[q,dq]``."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .base_model import KoopmanNet


def _metric(prediction: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    if name == "mse":
        return F.mse_loss(prediction, target)
    if name == "mae":
        return F.l1_loss(prediction, target)
    if name == "nmse":
        return F.mse_loss(prediction, target) / target.square().mean().clamp_min(1e-8)
    raise ValueError(f"Unsupported loss {name!r}")


def rollout_loss(
    batch_data: dict[str, torch.Tensor],
    net: KoopmanNet,
    loss_name: str = "mse",
    gamma: float = 0.98,
    pre_length: int = 25,
) -> dict[str, torch.Tensor]:
    x, u = batch_data["x"], batch_data["u"]
    if pre_length >= x.shape[1]:
        raise ValueError("pre_length must be smaller than trajectory length")
    start = random.randint(0, x.shape[1] - pre_length - 1)
    state = x[:, start]
    latent = net.x_encoder(state)
    state_loss = torch.zeros((), device=x.device)
    latent_loss = torch.zeros((), device=x.device)
    q_loss = torch.zeros((), device=x.device)
    dq_loss = torch.zeros((), device=x.device)
    weight_sum = 0.0
    weight = 1.0
    for offset in range(pre_length):
        index = start + offset
        action_latent = net.u_encoder(state, u[:, index])
        latent = net.koopman_operation(latent, action_latent)
        prediction = net.x_decoder(latent)
        target = x[:, index + 1]
        target_latent = net.x_encoder(target)
        state_loss = state_loss + weight * _metric(prediction, target, loss_name)
        latent_loss = latent_loss + weight * _metric(latent, target_latent, loss_name)
        q_loss = q_loss + weight * F.mse_loss(prediction[:, :6], target[:, :6])
        dq_loss = dq_loss + weight * F.mse_loss(prediction[:, 6:], target[:, 6:])
        weight_sum += weight
        weight *= gamma
        state = prediction
    state_loss = state_loss / weight_sum
    latent_loss = latent_loss / weight_sum
    return {
        "total_loss": state_loss + latent_loss,
        "pred_loss": state_loss,
        "koopman_loss": latent_loss,
        "q_loss": q_loss / weight_sum,
        "dq_loss": dq_loss / weight_sum,
    }


def rollout_prediction(batch_data: dict[str, torch.Tensor], net: KoopmanNet) -> dict[str, torch.Tensor]:
    x, u = batch_data["x"], batch_data["u"]
    state = x[:, 0]
    latent = net.x_encoder(state)
    predictions = [state.unsqueeze(1)]
    for index in range(x.shape[1] - 1):
        action_latent = net.u_encoder(state, u[:, index])
        latent = net.koopman_operation(latent, action_latent)
        state = net.x_decoder(latent)
        predictions.append(state.unsqueeze(1))
    prediction = torch.cat(predictions, dim=1)
    error = prediction[:, 1:] - x[:, 1:]
    return {
        "pred": prediction,
        "rmse": error.square().mean().sqrt(),
        "q_rmse": error[:, :, :6].square().mean().sqrt(),
        "dq_rmse": error[:, :, 6:].square().mean().sqrt(),
    }
