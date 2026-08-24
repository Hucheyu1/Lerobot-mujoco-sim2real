from __future__ import annotations

import numpy as np
import torch

from args import Args
from control import JointTorquePDController, MPCController
from control.run_torque_control import run
from models.init_model import init_model
from models.losses import _physical_losses, rollout_loss, rollout_prediction
from UR5e.UR5e_DataCollection import UR5eDataGenerator


def test_formal_defaults_match_soarm_experiment_layout() -> None:
    args = Args(["--device", "cpu"])
    assert args.train_samples == 50_000
    assert args.train_steps == 20
    assert args.val_samples == 2_000
    assert args.test_samples == 2_000
    assert args.test_steps == 200
    assert args.pre_length == 10
    assert args.num_epochs == 500
    assert args.batch_size == args.eval_batch_size == 256
    assert args.physics_timestep * args.frame_skip == 0.02
    assert args.mpc_horizon == 10


def test_data_layout_and_action_normalization(tmp_path) -> None:
    args = Args(["--mode", "collect", "--smoke", "--device", "cpu"])
    args.args.dataset_dir = str(tmp_path)
    args.args.train_samples = 2
    args.args.val_samples = 2
    args.args.test_samples = 2
    args.args.train_steps = 5
    args.args.test_steps = 5
    generator = UR5eDataGenerator(args)
    try:
        generator.generate_and_save_data(force=True)
        assert generator.train_data.shape == (2, 6, 21)
        batch = next(iter(generator.get_train_loader()[0]))
        assert batch["x"].shape[-1] == 15
        assert batch["u"].shape[-1] == 6
        assert torch.max(torch.abs(batch["u"])) <= args.residual_torque_fraction + 1e-6
    finally:
        generator.close()


def test_all_core_models_accept_ur5e_state_and_torque() -> None:
    batch = {"x": torch.randn(3, 7, 15), "u": torch.randn(3, 7, 6) * 0.02}
    for name in ("DKUC", "DBKN", "IKN", "IBKN"):
        args = Args(["--model", name, "--smoke", "--device", "cpu"])
        model = init_model(args)
        loss = rollout_loss(batch, model, pre_length=3)
        prediction = rollout_prediction(batch, model)
        assert torch.isfinite(loss["total_loss"])
        assert prediction["pred"].shape == batch["x"].shape


def test_physical_loss_preserves_soarm_weighting_and_adds_velocity() -> None:
    target = torch.zeros(2, 15)
    prediction = torch.ones(2, 15)
    total, ee, q, dq = _physical_losses(prediction, target, "mse")
    assert torch.allclose(ee, torch.tensor(1.0))
    assert torch.allclose(q, torch.tensor(1.0))
    assert torch.allclose(dq, torch.tensor(1.0))
    assert torch.allclose(total, torch.tensor(6.0))


def test_bilinear_matrix_expansion_matches_network_operation() -> None:
    for model_name in ("DBKN", "IBKN"):
        for ordering_flag in ([], ["--u-z"]):
            args = Args(["--model", model_name, "--smoke", "--device", "cpu", *ordering_flag])
            model = init_model(args).double()
            state = torch.randn(1, 15, dtype=torch.float64)
            action = torch.randn(1, 6, dtype=torch.float64) * 0.02
            with torch.no_grad():
                lifted = model.x_encoder(state)
                predicted = model.koopman_operation(lifted, action)
                linear = model.lA(lifted)
                b_total = model.lB.weight.detach().clone()
                for value, matrix in zip(lifted[0], model.get_Hi_numpy()):
                    b_total += value * torch.as_tensor(matrix)
                expanded = linear + action @ b_total.T
            assert torch.allclose(predicted, expanded, atol=1e-10, rtol=1e-8)


def test_pd_controller_clips_residual_torque() -> None:
    limits = np.array([7.5, 7.5, 7.5, 1.4, 1.4, 1.4])
    controller = JointTorquePDController(limits)
    command = controller.command(np.zeros(15), np.ones(6) * 10.0)
    assert np.allclose(command, limits)


def test_koopman_mpc_returns_bounded_physical_torque() -> None:
    args = Args(["--model", "IBKN", "--smoke", "--device", "cpu"])
    args.args.mpc_horizon = 2
    model = init_model(args).double()
    rated = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
    residual = rated * 0.05
    state = np.zeros(15)
    reference = np.zeros((2, 15))
    for mpc_type in ("mpc", "delta_mpc"):
        args.args.MPC_type = mpc_type
        controller = MPCController(model, args, rated, residual, horizon=2)
        command = controller.command(state, reference)
        assert command.shape == (6,)
        assert np.all(np.isfinite(command))
        assert np.all(np.abs(command) <= residual + 1e-6)


def test_closed_loop_direct_torque_smoke(tmp_path) -> None:
    metrics = run(steps=40, seed=4, output=tmp_path / "control.npz")
    assert metrics["steps"] == 40
    assert np.isfinite(metrics["q_rmse_rad"])
    assert metrics["max_residual_torque_nm"] <= 7.5 + 1e-12
    assert (tmp_path / "control.npz").exists()
