"""SOARM-style matrix Koopman MPC for complete UR5e joint torque control."""

from __future__ import annotations

import casadi as ca
import numpy as np
import torch
from scipy.optimize import minimize

from UR5e.normalization import NormalizationStats


class MPCController:
    """CasADi delta-MPC built from the learned Koopman ``A/B/H/C`` matrices.

    As in the original controller, the bilinear term is locally frozen at the
    current lifted state, yielding ``B_total(z0)`` over the prediction horizon.
    The decision variable is normalized complete joint torque (or its
    increment), and both torque and torque-increment bounds are enforced inside
    the optimization. Incremental MPC is the project default.
    """

    def __init__(
        self,
        net,
        args,
        rated_torque: np.ndarray,
        normalization: NormalizationStats,
        horizon: int | None = None,
    ) -> None:
        self.net = net.eval()
        self.device = args.device
        self.args = args
        self.x_dim = args.x_dim
        self.u_dim = args.u_dim
        self.H = horizon or args.mpc_horizon
        self.MPC_type = args.MPC_type

        with torch.no_grad():
            if hasattr(net, "lA"):
                self.Ad = net.lA.weight.detach().cpu().double().numpy()
            else:
                self.Ad = net.get_koopman_matrix_K().detach().cpu().double().numpy()
            self.Bd = net.lB.weight.detach().cpu().double().numpy()
        self.Nkoopman = self.Ad.shape[0]
        self.H_hat_list = net.get_Hi_numpy() if hasattr(net, "H") else None
        self.C = net.lC.weight.detach().cpu().double().numpy() if hasattr(net, "lC") else None
        self.reference_dim = self.x_dim
        self.normalization = normalization

        rated_torque = np.asarray(rated_torque, dtype=np.float64)
        if rated_torque.shape != (6,) or np.any(rated_torque <= 0.0):
            raise ValueError("rated_torque must be positive with shape (6,)")
        if not np.allclose(rated_torque, normalization.rated_torque_nm):
            raise ValueError("MPC rated torque does not match checkpoint normalization")
        self.rated_torque = rated_torque
        self.normalized_limit = np.ones(6, dtype=np.float64)
        self.rate_limit = np.full(6, args.torque_rate_fraction, dtype=np.float64)

        # All models use the same dimensionless state-coordinate cost. Cartesian
        # tracking retains the previous MPC emphasis after per-state scaling.
        self.Q_model = np.diag([50.0] * 3 + [1.0] * 6 + [1.0] * 6)
        self.R = 0.5 * np.eye(self.u_dim)
        self.u_prev = np.zeros(self.u_dim)
        self.warm_start = np.zeros(self.H * self.u_dim)
        self.solver_backend = "casadi-ipopt"
        self.solver = self._setup_solver()

    def linearize_B(self, z0):
        """Return ``B + sum_j z0[j] H_hat_j`` exactly as in the old MPC."""

        total = ca.DM(self.Bd)
        if self.H_hat_list is not None:
            for index, matrix in enumerate(self.H_hat_list):
                total += z0[index] * ca.DM(np.asarray(matrix, dtype=np.float64))
        return total

    def _setup_solver(self):
        decision = ca.SX.sym("delta_u" if self.MPC_type == "delta_mpc" else "u", self.H * self.u_dim)
        reference = ca.SX.sym("reference", self.H, self.reference_dim)
        z0 = ca.SX.sym("z0", self.Nkoopman)
        u_previous = ca.SX.sym("u_previous", self.u_dim)
        output_matrix = ca.SX.sym("output_matrix", self.x_dim, self.Nkoopman)
        output_offset = ca.SX.sym("output_offset", self.x_dim)
        b_total = self.linearize_B(z0)
        z = z0
        u = u_previous
        cost = 0
        constraints = []
        q_matrix = ca.DM(self.Q_model)
        r_matrix = ca.DM(self.R)

        for step in range(self.H):
            variable = decision[step * self.u_dim : (step + 1) * self.u_dim]
            if self.MPC_type == "delta_mpc":
                delta_u = variable
                u = u + delta_u
            else:
                delta_u = variable - u
                u = variable
            z = ca.mtimes(ca.DM(self.Ad), z) + ca.mtimes(b_total, u)
            prediction = ca.mtimes(output_matrix, z) + output_offset
            error = prediction - reference[step, :].T
            cost += ca.mtimes([error.T, q_matrix, error])
            penalized_control = delta_u if self.MPC_type == "delta_mpc" else u
            cost += ca.mtimes([penalized_control.T, r_matrix, penalized_control])
            constraints.append(u)

        parameters = ca.vertcat(
            ca.reshape(ca.transpose(reference), -1, 1),
            z0,
            u_previous,
            ca.reshape(output_matrix, -1, 1),
            output_offset,
        )
        problem = {"x": decision, "f": cost, "g": ca.vertcat(*constraints), "p": parameters}
        options = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 150,
            "ipopt.tol": 1e-5,
            "print_time": False,
            "ipopt.sb": "yes",
        }
        try:
            return ca.nlpsol(f"ur5e_mpc_{id(self)}", "ipopt", problem, options)
        except RuntimeError as error:
            # Some Windows/Conda CasADi builds expose the plugin name but omit
            # one of ipopt's runtime DLLs.  The frozen-bilinear problem is a
            # quadratic program, so SciPy can solve the identical objective
            # and bounds without changing the controller semantics.
            if "Plugin 'ipopt' is not found" not in str(error):
                raise
            self.solver_backend = "scipy-slsqp"
            return None

    def _numeric_b_total(self, lifted_state: np.ndarray) -> np.ndarray:
        total = self.Bd.copy()
        if self.H_hat_list is not None:
            for value, matrix in zip(lifted_state, self.H_hat_list):
                total += value * np.asarray(matrix, dtype=np.float64)
        return total

    def _sequence_from_decision(self, decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        variables = decision.reshape(self.H, self.u_dim)
        commands = np.empty_like(variables)
        increments = np.empty_like(variables)
        previous = self.u_prev.copy()
        for step, variable in enumerate(variables):
            if self.MPC_type == "delta_mpc":
                increments[step] = variable
                commands[step] = previous + variable
            else:
                commands[step] = variable
                increments[step] = variable - previous
            previous = commands[step]
        return commands, increments

    def _solve_with_scipy(
        self,
        lifted_state: np.ndarray,
        reference: np.ndarray,
        output_matrix: np.ndarray,
        output_offset: np.ndarray,
    ) -> np.ndarray:
        """Solve the same frozen-bilinear MPC when CasADi/Ipopt is unavailable."""

        b_total = self._numeric_b_total(lifted_state)
        q_matrix = self.Q_model

        def objective(flat: np.ndarray) -> float:
            commands, increments = self._sequence_from_decision(flat)
            z = lifted_state.copy()
            value = 0.0
            for step in range(self.H):
                z = self.Ad @ z + b_total @ commands[step]
                prediction = output_matrix @ z + output_offset
                error = prediction - reference[step]
                value += float(error @ q_matrix @ error)
                penalized = increments[step] if self.MPC_type == "delta_mpc" else commands[step]
                value += float(penalized @ self.R @ penalized)
            return value

        def torque_margin(flat: np.ndarray) -> np.ndarray:
            commands, _ = self._sequence_from_decision(flat)
            return (self.normalized_limit[None, :] - np.abs(commands)).reshape(-1)

        bound = self.rate_limit if self.MPC_type == "delta_mpc" else self.normalized_limit
        result = minimize(
            objective,
            self.warm_start,
            method="SLSQP",
            bounds=list(zip(np.tile(-bound, self.H), np.tile(bound, self.H))),
            constraints={"type": "ineq", "fun": torque_margin},
            options={"maxiter": 150, "ftol": 1e-7, "disp": False},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"SciPy MPC failed: {result.message}")
        return result.x.reshape(self.H, self.u_dim)

    def Psi_o(self, state: np.ndarray) -> np.ndarray:
        state_model = self.normalization.normalize_state(state)
        tensor = torch.as_tensor(state_model, dtype=next(self.net.parameters()).dtype, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            latent = self.net.x_encoder(tensor).detach().cpu().double().numpy()
        return latent.reshape(-1)

    def _prepare_reference(self, reference: np.ndarray) -> np.ndarray:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.shape != (self.H, self.x_dim):
            raise ValueError(f"reference must have shape ({self.H}, {self.x_dim})")
        return np.asarray(self.normalization.normalize_state(reference), dtype=np.float64)

    def _output_linearization(self, lifted_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map lifted predictions to standardized state for a common MPC cost.

        DKUC/DBKN expose the measured standardized state through a fixed C.
        IKN/IBKN use the Jacobian of their exact inverse decoder at the current
        lifted state, held fixed across one MPC horizon.
        """

        if self.C is not None:
            return self.C, np.zeros(self.x_dim, dtype=np.float64)
        dtype = next(self.net.parameters()).dtype
        z = torch.as_tensor(lifted_state, dtype=dtype, device=self.device).requires_grad_(True)

        def decode(value: torch.Tensor) -> torch.Tensor:
            return self.net.x_decoder(value.unsqueeze(0)).squeeze(0)

        with torch.enable_grad():
            decoded = decode(z)
            jacobian = torch.autograd.functional.jacobian(decode, z, create_graph=False)
        matrix = jacobian.detach().cpu().double().numpy()
        decoded_numpy = decoded.detach().cpu().double().numpy()
        offset = decoded_numpy - matrix @ lifted_state
        if matrix.shape != (self.x_dim, self.Nkoopman) or not np.all(np.isfinite(matrix)):
            raise RuntimeError("Invertible decoder produced an invalid MPC output Jacobian")
        return matrix, offset

    def get_control(self, state: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return physical N m torque and its normalized model input."""

        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.x_dim,):
            raise ValueError(f"state must have shape ({self.x_dim},)")
        lifted_state = self.Psi_o(state)
        prepared_reference = self._prepare_reference(reference)
        output_matrix, output_offset = self._output_linearization(lifted_state)
        parameters = np.concatenate(
            (
                prepared_reference.reshape(-1),
                lifted_state,
                self.u_prev,
                output_matrix.reshape(-1, order="F"),
                output_offset,
            )
        )
        if self.MPC_type == "delta_mpc":
            lower_x = np.tile(-self.rate_limit, self.H)
            upper_x = np.tile(self.rate_limit, self.H)
        else:
            lower_x = np.tile(-self.normalized_limit, self.H)
            upper_x = np.tile(self.normalized_limit, self.H)
        if self.solver is None:
            optimized = self._solve_with_scipy(
                lifted_state,
                prepared_reference,
                output_matrix,
                output_offset,
            )
        else:
            solution = self.solver(
                x0=self.warm_start,
                p=parameters,
                lbx=lower_x,
                ubx=upper_x,
                lbg=np.tile(-self.normalized_limit, self.H),
                ubg=np.tile(self.normalized_limit, self.H),
            )
            optimized = np.asarray(solution["x"]).reshape(self.H, self.u_dim)
        if self.MPC_type == "delta_mpc":
            normalized = np.clip(self.u_prev + optimized[0], -self.normalized_limit, self.normalized_limit)
        else:
            normalized = np.clip(optimized[0], -self.normalized_limit, self.normalized_limit)
        self.u_prev = normalized.copy()
        self.warm_start = np.concatenate((optimized[1:].reshape(-1), optimized[-1]))
        return normalized * self.rated_torque, normalized

    def command(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return self.get_control(state, reference)[0]

    def reset(self, initial_torque: np.ndarray | None = None) -> None:
        """Reset MPC memory, optionally around the current complete torque."""

        if initial_torque is None:
            self.u_prev.fill(0.0)
        else:
            torque = np.asarray(initial_torque, dtype=np.float64)
            if torque.shape != (6,) or not np.all(np.isfinite(torque)):
                raise ValueError("initial_torque must be finite with shape (6,)")
            self.u_prev = np.clip(torque / self.rated_torque, -1.0, 1.0)
        self.warm_start.fill(0.0)
