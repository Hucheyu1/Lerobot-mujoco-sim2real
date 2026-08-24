from __future__ import annotations

import mujoco
import numpy as np

from UR5e import UR5eTorqueConfig, UR5eTorqueEnv


def test_direct_motor_dimensions_and_limits() -> None:
    env = UR5eTorqueEnv()
    try:
        state, info = env.reset(seed=1)
        assert env.observation_space.shape == (15,)
        assert env.action_space.shape == (6,)
        assert state.shape == (15,)
        assert np.allclose(state[:3], info["ee_position"])
        assert env.model.nu == env.model.nq == env.model.nv == 6
        assert np.all(env.model.actuator_biastype[env.actuator_ids] == mujoco.mjtBias.mjBIAS_NONE)
        assert np.all(env.model.actuator_gaintype[env.actuator_ids] == mujoco.mjtGain.mjGAIN_FIXED)
        assert np.allclose(env.torque_limits, [150, 150, 150, 28, 28, 28])
        assert np.allclose(env.action_space.high, env.torque_limits)
    finally:
        env.close()


def test_full_gravity_feedforward_holds_home() -> None:
    config = UR5eTorqueConfig(initial_position_span=0.0, initial_velocity_span=0.0)
    env = UR5eTorqueEnv(config)
    try:
        before, _ = env.reset(seed=3)
        equilibrium_torque = env.inverse_dynamics_torque(np.zeros(6))
        after, _, terminated, _, info = env.step(equilibrium_torque)
        assert not terminated
        assert np.linalg.norm(after[3:9] - before[3:9]) < 1e-6
        assert np.linalg.norm(after[9:15]) < 1e-5
        assert np.allclose(info["applied_joint_torque"], equilibrium_torque)
    finally:
        env.close()


def test_reference_state_contains_ee_q_dq_without_mutating_plant() -> None:
    env = UR5eTorqueEnv()
    try:
        before, _ = env.reset(seed=9)
        q_ref = env.HOME + 0.03
        dq_ref = np.linspace(-0.1, 0.1, 6)
        reference = env.reference_state(q_ref, dq_ref)
        after = env._get_state()
        assert reference.shape == (15,)
        assert np.allclose(reference[3:9], q_ref)
        assert np.allclose(reference[9:15], dq_ref)
        assert np.allclose(after, before)
    finally:
        env.close()


def test_complete_joint_torque_is_clipped_in_physical_units() -> None:
    env = UR5eTorqueEnv()
    try:
        env.reset(seed=5)
        _, _, _, _, info = env.step(np.full(6, 1e6))
        assert np.allclose(info["requested_joint_torque"], 1e6)
        assert np.allclose(info["applied_joint_torque"], env.torque_limits)
        assert np.all(np.linalg.eigvalsh(env.mass_matrix()) > 0.0)
    finally:
        env.close()


def test_inverse_dynamics_torque_is_finite_and_matches_mujoco_equation() -> None:
    env = UR5eTorqueEnv()
    try:
        env.reset(seed=17)
        desired_acceleration = np.linspace(-0.5, 0.5, 6)
        torque = env.inverse_dynamics_torque(desired_acceleration)
        expected = (
            env.mass_matrix() @ desired_acceleration
            + env.data.qfrc_bias[env.dof_ids]
            - env.data.qfrc_passive[env.dof_ids]
        )
        assert torque.shape == (6,)
        assert np.all(np.isfinite(torque))
        assert np.allclose(torque, expected)
    finally:
        env.close()
