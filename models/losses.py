"""Dimensionless multi-step losses for UR5e state ``[ee_xyz,q,dq]``."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .base_model import KoopmanNet

if TYPE_CHECKING:
    from UR5e.normalization import NormalizationStats


def spectral_radius_loss(net: KoopmanNet, margin: float = 0.99) -> torch.Tensor:
    """Penalize eigenvalues outside the requested discrete-time margin."""

    if not hasattr(net, "lA"):
        return next(net.parameters()).new_zeros(())
    eigenvalues = torch.linalg.eigvals(net.lA.weight)
    return torch.clamp(eigenvalues.abs() - margin, min=0.0).sum()


def sparsity_loss(net: KoopmanNet) -> torch.Tensor:
    """Return the bilinear matrix L1 norm when a model has ``H``."""

    if not hasattr(net, "H"):
        return next(net.parameters()).new_zeros(())
    return torch.linalg.vector_norm(net.H.weight, ord=1)


def _metric(prediction: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    if name == "mse":
        return F.mse_loss(prediction, target)
    if name == "mae":
        return F.l1_loss(prediction, target)
    if name == "nmse":
        return F.mse_loss(prediction, target) / target.square().mean().clamp_min(1e-8)
    raise ValueError(f"Unsupported loss {name!r}")


def _physical_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the SOARM group weights in standardized, dimensionless state coordinates."""

    distance = _metric(prediction[:, :3], target[:, :3], name)
    angle = _metric(prediction[:, 3:9], target[:, 3:9], name)
    velocity = _metric(prediction[:, 9:15], target[:, 9:15], name)
    return distance + 4.0 * angle + velocity, distance, angle, velocity


def rollout_loss(
    batch_data: dict[str, torch.Tensor],
    net: KoopmanNet,
    loss_name: str = "mse",
    gamma: float = 0.98,
    pre_length: int = 25,
    normalization: NormalizationStats | None = None,
    latent_loss_weight: float = 0.3,
    reconstruction_loss_weight: float = 1.0,
    delta_loss_weight: float = 0.0,
    stability_loss_weight: float = 1e-3,
    bilinear_l1_weight: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Train free-running rollouts and an increment-whitened one-step auxiliary target."""

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
    ee_loss = torch.zeros((), device=x.device)
    recon_loss = torch.zeros((), device=x.device)
    weight_sum = 0.0
    weight = 1.0
    for offset in range(pre_length):
        index = start + offset
        action_latent = net.u_encoder(state, u[:, index])
        latent = net.koopman_operation(latent, action_latent)
        prediction = net.x_decoder(latent)
        target = x[:, index + 1]
        target_latent = net.x_encoder(target)
        reconstruction = net.x_decoder(target_latent)
        grouped, distance, angle, velocity = _physical_losses(prediction, target, loss_name)
        state_loss = state_loss + weight * grouped
        latent_loss = latent_loss + weight * _metric(latent, target_latent, loss_name)
        ee_loss = ee_loss + weight * distance
        q_loss = q_loss + weight * angle
        dq_loss = dq_loss + weight * velocity
        recon_loss = recon_loss + weight * _metric(reconstruction, target, loss_name)
        weight_sum += weight
        weight *= gamma
        state = prediction
    state_loss = state_loss / weight_sum
    latent_loss = latent_loss / weight_sum
    recon_loss = recon_loss / weight_sum

    # The reference implementation whitens one-step increments. Retain that
    # useful idea as an auxiliary target while keeping multi-step rollout as the
    # primary objective. The scale is always fitted on the training split.
    delta_loss = torch.zeros((), dtype=x.dtype, device=x.device)
    if delta_loss_weight > 0.0:
        teacher_state = x[:, start]
        teacher_latent = net.x_encoder(teacher_state)
        teacher_action = net.u_encoder(teacher_state, u[:, start])
        one_step = net.x_decoder(net.koopman_operation(teacher_latent, teacher_action))
        one_step_target = x[:, start + 1]
        if normalization is None:
            delta_scale = torch.ones(15, dtype=x.dtype, device=x.device)
        else:
            delta_scale = torch.as_tensor(
                normalization.normalized_delta_std,
                dtype=x.dtype,
                device=x.device,
            )
        delta_loss = F.mse_loss((one_step - one_step_target) / delta_scale, torch.zeros_like(one_step))

    # Models that concatenate the measured state reconstruct it exactly through
    # fixed C; invertible models reconstruct by construction.
    use_reconstruction = bool(getattr(net, "use_decoder", False)) and not type(net).__name__.startswith("Invert")
    stable_raw = spectral_radius_loss(net)
    h_raw = sparsity_loss(net)
    stable = stability_loss_weight * stable_raw
    h_sparsity = bilinear_l1_weight * h_raw
    total = (
        state_loss
        + latent_loss_weight * latent_loss
        + reconstruction_loss_weight * int(use_reconstruction) * recon_loss
        + delta_loss_weight * delta_loss
        + stable
        + h_sparsity
    )
    return {
        "total_loss": total,
        "pred_loss": state_loss,
        "koopman_loss": latent_loss_weight * latent_loss,
        "koopman_loss_raw": latent_loss,
        "recon_loss": reconstruction_loss_weight * int(use_reconstruction) * recon_loss,
        "delta_loss": delta_loss_weight * delta_loss,
        "delta_loss_raw": delta_loss,
        "dis_loss": ee_loss / weight_sum,
        "angle_loss": 4.0 * q_loss / weight_sum,
        "velocity_loss": dq_loss / weight_sum,
        "q_loss": q_loss / weight_sum,
        "dq_loss": dq_loss / weight_sum,
        "stable_Loss": stable,
        "H_Loss": h_sparsity,
    }


def rollout_prediction(
    batch_data: dict[str, torch.Tensor],
    net: KoopmanNet,
    normalization: NormalizationStats | None = None,
) -> dict[str, torch.Tensor]:
    """Return free-running predictions and both model/physical-coordinate errors."""

    x, u = batch_data["x"], batch_data["u"]
    state = x[:, 0]
    latent = net.x_encoder(state)
    predictions = [state.unsqueeze(1)]
    for index in range(x.shape[1] - 1):
        action_latent = net.u_encoder(state, u[:, index])
        latent = net.koopman_operation(latent, action_latent)
        state = net.x_decoder(latent)
        predictions.append(state.unsqueeze(1))
    prediction_model = torch.cat(predictions, dim=1)
    error_model = prediction_model[:, 1:] - x[:, 1:]
    if normalization is None:
        prediction_phys = prediction_model
        target_phys = batch_data.get("x_phys", x)
    else:
        prediction_phys = normalization.denormalize_state(prediction_model)
        target_phys = batch_data.get("x_phys")
        if target_phys is None:
            target_phys = normalization.denormalize_state(x)
    error_phys = prediction_phys[:, 1:] - target_phys[:, 1:]
    return {
        "pred": prediction_model,
        "pred_phys": prediction_phys,
        "error_model": error_model,
        "error_phys": error_phys,
        "rmse": error_phys.square().mean().sqrt(),
        "ee_rmse": error_phys[:, :, :3].square().mean().sqrt(),
        "q_rmse": error_phys[:, :, 3:9].square().mean().sqrt(),
        "dq_rmse": error_phys[:, :, 9:15].square().mean().sqrt(),
        "normalized_rmse": error_model.square().mean().sqrt(),
        "normalized_ee_rmse": error_model[:, :, :3].square().mean().sqrt(),
        "normalized_q_rmse": error_model[:, :, 3:9].square().mean().sqrt(),
        "normalized_dq_rmse": error_model[:, :, 9:15].square().mean().sqrt(),
    }
