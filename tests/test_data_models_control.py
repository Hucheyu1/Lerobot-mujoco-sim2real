from __future__ import annotations

import numpy as np
import torch

from args import Args
from control import JointTorquePDController, KoopmanTorqueMPC
from control.run_torque_control import run
from models.init_model import init_model
from models.losses import rollout_loss, rollout_prediction
from UR5e.UR5e_DataCollection import UR5eDataGenerator


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
        assert generator.train_data.shape == (2, 6, 18)
        batch = next(iter(generator.get_train_loader()[0]))
        assert batch["x"].shape[-1] == 12
        assert batch["u"].shape[-1] == 6
        assert torch.max(torch.abs(batch["u"])) <= args.residual_torque_fraction + 1e-6
    finally:
        generator.close()


def test_all_core_models_accept_ur5e_state_and_torque() -> None:
    batch = {"x": torch.randn(3, 7, 12), "u": torch.randn(3, 7, 6) * 0.02}
    for name in ("DKUC", "DBKN", "IKN", "IBKN"):
        args = Args(["--model", name, "--smoke", "--device", "cpu"])
        model = init_model(args)
        loss = rollout_loss(batch, model, pre_length=3)
        prediction = rollout_prediction(batch, model)
        assert torch.isfinite(loss["total_loss"])
        assert prediction["pred"].shape == batch["x"].shape


def test_pd_controller_clips_residual_torque() -> None:
    limits = np.array([7.5, 7.5, 7.5, 1.4, 1.4, 1.4])
    controller = JointTorquePDController(limits)
    command = controller.command(np.zeros(12), np.ones(6) * 10.0)
    assert np.allclose(command, limits)


def test_koopman_mpc_returns_bounded_physical_torque() -> None:
    args = Args(["--model", "IBKN", "--smoke", "--device", "cpu"])
    model = init_model(args)
    rated = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
    residual = rated * 0.05
    controller = KoopmanTorqueMPC(model, rated, residual, horizon=3, iterations=2)
    state = np.zeros(12)
    reference = np.zeros((3, 12))
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
