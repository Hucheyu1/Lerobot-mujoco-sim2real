"""SOARM-style multi-step losses for UR5e state ``[ee_xyz,q,dq]``."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .base_model import KoopmanNet


def spectral_radius_loss(net: KoopmanNet, margin: float = 0.99) -> torch.Tensor:
    """Keep the original soft stability regularizer for unconstrained ``A``."""

    if not hasattr(net, "lA"):
        return next(net.parameters()).new_zeros(())
    eigenvalues = torch.linalg.eigvals(net.lA.weight)
    return torch.clamp(eigenvalues.abs() - margin, min=0.0).sum()


def sparsity_loss(net: KoopmanNet) -> torch.Tensor:
    """Report the original bilinear-H L1 term (diagnostic by default)."""

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
    """Preserve the original distance/angle weighting and add velocity state.

    The SOARM101 loss used end-effector error plus four times joint-angle
    error.  Torque dynamics additionally require joint velocity, so the UR5e
    form is ``L_ee + 4 L_q + L_dq``.
    """

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
        physical, distance, angle, velocity = _physical_losses(prediction, target, loss_name)
        state_loss = state_loss + weight * physical
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
    # This follows the old k_linear_loss: learned non-invertible decoders use
    # reconstruction loss; invertible models reconstruct by construction.
    use_reconstruction = bool(getattr(net, "use_decoder", False)) and not type(net).__name__.startswith("Invert")
    stable = 1e-3 * spectral_radius_loss(net)
    h_sparsity = 1e-9 * sparsity_loss(net)
    return {
        "total_loss": state_loss + latent_loss + int(use_reconstruction) * recon_loss + stable,
        "pred_loss": state_loss,
        "koopman_loss": latent_loss,
        "recon_loss": recon_loss,
        "dis_loss": ee_loss / weight_sum,
        "angle_loss": 4.0 * q_loss / weight_sum,
        "velocity_loss": dq_loss / weight_sum,
        "q_loss": q_loss / weight_sum,
        "dq_loss": dq_loss / weight_sum,
        "stable_Loss": stable,
        "H_Loss": h_sparsity,
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
        "ee_rmse": error[:, :, :3].square().mean().sqrt(),
        "q_rmse": error[:, :, 3:9].square().mean().sqrt(),
        "dq_rmse": error[:, :, 9:15].square().mean().sqrt(),
    }
