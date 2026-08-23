"""Differentiable receding-horizon control using a trained Koopman model."""

from __future__ import annotations

import numpy as np
import torch


class KoopmanTorqueMPC:
    """Optimize normalized residual joint torque through a frozen learned model.

    This controller intentionally favors a compact and auditable implementation
    over a real-time claim.  Every gradient step projects the candidate action
    sequence onto the physical residual-torque box; the returned command is in
    N m and can be passed directly to ``UR5eTorqueEnv.step``.
    """

    def __init__(
        self,
        model,
        rated_torque: np.ndarray,
        residual_limits: np.ndarray,
        horizon: int = 8,
        iterations: int = 25,
        learning_rate: float = 0.08,
    ) -> None:
        if horizon < 1 or iterations < 1 or learning_rate <= 0.0:
            raise ValueError("horizon, iterations, and learning_rate must be positive")
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        first_parameter = next(self.model.parameters())
        self.device = first_parameter.device
        self.dtype = first_parameter.dtype
        self.rated_torque = torch.as_tensor(rated_torque, device=self.device, dtype=self.dtype)
        residual_limits = torch.as_tensor(residual_limits, device=self.device, dtype=self.dtype)
        self.normalized_limit = residual_limits / self.rated_torque
        self.horizon = horizon
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.q_weight = torch.full((6,), 20.0, device=self.device, dtype=self.dtype)
        self.dq_weight = torch.full((6,), 1.5, device=self.device, dtype=self.dtype)
        self.input_weight = 0.05
        self.rate_weight = 0.20
        self.previous_action = torch.zeros(6, device=self.device, dtype=self.dtype)
        self.warm_start = torch.zeros(horizon, 6, device=self.device, dtype=self.dtype)

    def reset(self) -> None:
        self.previous_action.zero_()
        self.warm_start.zero_()

    def command(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(state, device=self.device, dtype=self.dtype)
        reference_tensor = torch.as_tensor(reference, device=self.device, dtype=self.dtype)
        if state_tensor.shape != (12,):
            raise ValueError("state must have shape (12,)")
        if reference_tensor.shape == (self.horizon, 6):
            reference_tensor = torch.cat((reference_tensor, torch.zeros_like(reference_tensor)), dim=1)
        if reference_tensor.shape != (self.horizon, 12):
            raise ValueError(f"reference must have shape ({self.horizon}, 6) or ({self.horizon}, 12)")

        actions = self.warm_start.detach().clone().requires_grad_(True)
        for _ in range(self.iterations):
            predicted_state = state_tensor.unsqueeze(0)
            latent = self.model.x_encoder(predicted_state)
            previous = self.previous_action
            objective = torch.zeros((), device=self.device, dtype=self.dtype)
            for index in range(self.horizon):
                action = actions[index].unsqueeze(0)
                action_latent = self.model.u_encoder(predicted_state, action)
                latent = self.model.koopman_operation(latent, action_latent)
                predicted_state = self.model.x_decoder(latent)
                error = predicted_state.squeeze(0) - reference_tensor[index]
                objective = objective + torch.sum(self.q_weight * error[:6].square())
                objective = objective + torch.sum(self.dq_weight * error[6:].square())
                objective = objective + self.input_weight * torch.sum(actions[index].square())
                objective = objective + self.rate_weight * torch.sum((actions[index] - previous).square())
                previous = actions[index]
            gradient = torch.autograd.grad(objective, actions)[0]
            with torch.no_grad():
                actions = actions - self.learning_rate * gradient
                actions = torch.maximum(torch.minimum(actions, self.normalized_limit), -self.normalized_limit)
            actions.requires_grad_(True)

        optimized = actions.detach()
        command_normalized = optimized[0]
        self.previous_action = command_normalized.clone()
        self.warm_start[:-1] = optimized[1:]
        self.warm_start[-1] = optimized[-1]
        return (command_normalized * self.rated_torque).cpu().numpy()
