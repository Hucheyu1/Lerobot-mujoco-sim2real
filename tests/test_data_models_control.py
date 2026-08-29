from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from args import Args
from control import JointTorquePDController, MPCController
from control.run_torque_control import run
from models.init_model import init_model
from models.losses import _physical_losses, rollout_loss, rollout_prediction
from UR5e.artifacts import (
    load_model_checkpoint,
    save_model_checkpoint,
    validate_model_manifest,
    write_model_manifest,
)
from UR5e.normalization import NORMALIZATION_FILE, NormalizationStats
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
    assert args.MPC_type == "delta_mpc"
    assert args.initial_position_span == 0.50
    assert args.initial_velocity_span == 0.05
    assert args.waypoint_count == 10
    assert args.waypoint_velocity_limit == 0.07
    assert args.tracking_kp == 16.0
    assert args.tracking_kd == 8.0
    assert args.tracking_acceleration_limit == 4.0
    assert args.excitation_fraction == 0.01
    assert args.random_hold_steps == 1
    assert args.state_std_floor == 1e-6
    assert args.latent_loss_weight == 0.3
    assert args.delta_loss_weight == 0.0
    assert args.bilinear_l1_weight == 1e-6


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
        assert batch["x_phys"].shape == batch["x"].shape
        assert batch["tau_nm"].shape == batch["u"].shape
        assert torch.max(torch.abs(batch["u"])) <= 1.0 + 1e-6
        assert generator.normalization is not None
        expected_x = generator.normalization.normalize_state(batch["x_phys"])
        expected_u = generator.normalization.normalize_torque(batch["tau_nm"])
        assert torch.allclose(batch["x"], expected_x)
        assert torch.allclose(batch["u"], expected_u)
        assert (tmp_path / NORMALIZATION_FILE).is_file()
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "complete"
        assert manifest["collection"]["dataset_version"] == "ur5e_full_joint_torque_v2"
        assert manifest["collection"]["collection_mode"] == "computed_full_torque_waypoints_with_excitation"
        assert manifest["collection"]["model_action_definition"] == (
            "applied_complete_joint_torque_normalized_by_rated_torque"
        )
        assert manifest["action"] == "clipped_complete_joint_torque"
        assert manifest["split_statistics"]["train"]["acceptance_rate"] == 1.0
        assert manifest["normalization"]["source_split"] == "train"
        assert manifest["normalization"]["fingerprint"] == generator.normalization.fingerprint()
        train_state = generator.train_data[:, :, 6:21]
        normalized_train_state = generator.normalization.normalize_state(train_state)
        assert np.allclose(np.mean(normalized_train_state, axis=(0, 1)), 0.0, atol=1e-7)
        varying = np.std(train_state, axis=(0, 1)) > args.state_std_floor
        assert np.allclose(np.std(normalized_train_state, axis=(0, 1))[varying], 1.0, atol=1e-7)
    finally:
        generator.close()


def test_old_unversioned_arrays_are_not_silently_reused(tmp_path) -> None:
    args = Args(["--mode", "collect", "--smoke", "--device", "cpu"])
    args.args.dataset_dir = str(tmp_path)
    np.save(tmp_path / "train.npy", np.zeros((1, 2, 21), dtype=np.float32))
    generator = UR5eDataGenerator(args)
    try:
        with pytest.raises(RuntimeError, match="--force-data"):
            generator.generate_and_save_data(force=False)
    finally:
        generator.close()


def test_residual_torque_checkpoint_is_not_silently_reused(tmp_path) -> None:
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"old checkpoint placeholder")
    rated = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
    with pytest.raises(RuntimeError, match="residual-torque"):
        validate_model_manifest(checkpoint, "IBKN", rated)
    args = Args(["--model", "IBKN", "--smoke", "--device", "cpu"])
    model = init_model(args)
    normalization = NormalizationStats.identity(rated)
    save_model_checkpoint(checkpoint, model, "IBKN", normalization, 0, {"normalized_score": 1.0}, {})
    write_model_manifest(tmp_path, "IBKN", rated, normalization)
    manifest = validate_model_manifest(checkpoint, "IBKN", rated, normalization)
    assert manifest["action_definition"] == "applied_complete_joint_torque_normalized_by_rated_torque"
    restored = init_model(args)
    restored_normalization, _ = load_model_checkpoint(
        checkpoint,
        restored,
        "IBKN",
        rated,
        "cpu",
        expected_normalization=normalization,
    )
    assert restored_normalization.fingerprint() == normalization.fingerprint()


def test_closed_loop_collector_keeps_long_trajectories_safe() -> None:
    args = Args(["--mode", "collect", "--smoke", "--device", "cpu"])
    generator = UR5eDataGenerator(args)
    try:
        data = generator.generate_trajectories(3, 200, "random", seed=1234)
        assert data.shape == (3, 201, 21)
        assert generator.last_generation_stats["acceptance_rate"] == 1.0
        assert generator.last_generation_stats["saturation_rate"] < 0.05
        assert np.max(np.abs(data[:, :, 15:21])) < generator.env.config.velocity_limit
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


def test_pd_controller_returns_clipped_complete_torque() -> None:
    limits = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
    controller = JointTorquePDController(limits, feedforward_torque=lambda: np.ones(6))
    command = controller.command(np.zeros(15), np.ones(6) * 10.0)
    assert np.allclose(command, limits)


def test_koopman_mpc_returns_bounded_physical_torque() -> None:
    rated = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
    normalization = NormalizationStats.identity(rated)
    state = np.zeros(15)
    reference = np.zeros((2, 15))
    for model_name in ("DKUC", "IBKN"):
        args = Args(["--model", model_name, "--smoke", "--device", "cpu"])
        args.args.mpc_horizon = 2
        model = init_model(args).double()
        for mpc_type in ("mpc", "delta_mpc"):
            args.args.MPC_type = mpc_type
            controller = MPCController(model, args, rated, normalization, horizon=2)
            if mpc_type == "delta_mpc":
                initial_torque = rated * 0.25
                controller.reset(initial_torque)
            command = controller.command(state, reference)
            assert command.shape == (6,)
            assert np.all(np.isfinite(command))
            assert np.all(np.abs(command) <= rated + 1e-6)
            if mpc_type == "delta_mpc":
                maximum_increment = rated * args.torque_rate_fraction
                assert np.all(np.abs(command - initial_torque) <= maximum_increment + 1e-6)


def test_closed_loop_direct_torque_smoke(tmp_path) -> None:
    metrics = run(steps=40, seed=4, output=tmp_path / "control.npz")
    assert metrics["steps"] == 40
    assert np.isfinite(metrics["q_rmse_rad"])
    assert metrics["max_joint_torque_nm"] <= 150.0 + 1e-12
    assert (tmp_path / "control.npz").exists()
