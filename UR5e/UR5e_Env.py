"""Gymnasium-style MuJoCo environment with direct UR5e joint torque control.

The public state is ``x = [p_ee, q, dq]`` (15 dimensions), matching the
end-effector-first layout of the original SOARM101 pipeline while retaining
joint velocity for torque-dynamics Markov state.  The public action is a
six-dimensional residual joint torque in N m.  It is added to an optional
gravity feed-forward term and sent to six MuJoCo ``motor`` actuators.  No
position or velocity servo is present in this control path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_ROOT / "assets" / "ur5e" / "scene_torque.xml"


@dataclass(frozen=True)
class UR5eTorqueConfig:
    """Runtime parameters for :class:`UR5eTorqueEnv`."""

    xml_path: str = str(DEFAULT_XML)
    physics_timestep: float = 0.002
    frame_skip: int = 10
    residual_torque_fraction: float = 0.05
    initial_position_span: float = 0.12
    initial_velocity_span: float = 0.05
    velocity_limit: float = 4.0
    gravity_compensation_scale: float = 0.90

    @property
    def control_timestep(self) -> float:
        return self.physics_timestep * self.frame_skip


class UR5eTorqueEnv(gym.Env):
    """Headless-friendly UR5e environment using six direct torque motors."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    JOINT_NAMES = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    ACTUATOR_NAMES = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
    HOME = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
    RATED_TORQUE = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

    def __init__(self, config: UR5eTorqueConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.config = config or UR5eTorqueConfig()
        if render_mode not in (None, "human"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode

        xml_path = Path(self.config.xml_path).expanduser().resolve()
        if not xml_path.is_file():
            raise FileNotFoundError(f"UR5e MJCF not found: {xml_path}")
        if self.config.frame_skip < 1 or self.config.physics_timestep <= 0:
            raise ValueError("frame_skip and physics_timestep must be positive")
        if not 0.0 < self.config.residual_torque_fraction <= 1.0:
            raise ValueError("residual_torque_fraction must lie in (0, 1]")
        if not 0.0 <= self.config.gravity_compensation_scale <= 1.5:
            raise ValueError("gravity_compensation_scale must lie in [0, 1.5]")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = self.config.physics_timestep
        self.data = mujoco.MjData(self.model)
        self._gravity_data = mujoco.MjData(self.model)

        self.joint_ids = self._ids(mujoco.mjtObj.mjOBJ_JOINT, self.JOINT_NAMES)
        self.actuator_ids = self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR, self.ACTUATOR_NAMES)
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.dof_ids = self.model.jnt_dofadr[self.joint_ids]
        self.ee_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.payload_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "payload")
        self.force_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")

        self.torque_limits = np.max(np.abs(self.model.actuator_ctrlrange[self.actuator_ids]), axis=1)
        if not np.allclose(self.torque_limits, self.RATED_TORQUE):
            raise ValueError(f"Unexpected UR5e torque limits: {self.torque_limits}")
        self.residual_limits = self.torque_limits * self.config.residual_torque_fraction

        self.action_space = spaces.Box(-self.residual_limits, self.residual_limits, dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float64)
        self._viewer = None
        self.last_residual_torque = np.zeros(6)
        self.last_gravity_torque = np.zeros(6)
        self.last_applied_torque = np.zeros(6)

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Missing MuJoCo object {name!r}")
        return int(object_id)

    def _ids(self, object_type: mujoco.mjtObj, names: tuple[str, ...]) -> np.ndarray:
        return np.asarray([self._id(object_type, name) for name in names], dtype=np.int32)

    def _get_state(self) -> np.ndarray:
        q = self.data.qpos[self.qpos_ids]
        dq = self.data.qvel[self.dof_ids]
        ee = self.data.site_xpos[self.ee_site_id]
        return np.concatenate((ee, q, dq)).astype(np.float64, copy=True)

    def end_effector_position(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].astype(np.float64, copy=True)

    def reference_state(self, q: np.ndarray, dq: np.ndarray | None = None) -> np.ndarray:
        """Construct ``[p_ee,q,dq]`` for a joint reference without changing the plant."""

        q = np.asarray(q, dtype=np.float64)
        dq = np.zeros(6) if dq is None else np.asarray(dq, dtype=np.float64)
        if q.shape != (6,) or dq.shape != (6,):
            raise ValueError("q and dq must each have shape (6,)")
        self._gravity_data.qpos[:] = 0.0
        self._gravity_data.qvel[:] = 0.0
        self._gravity_data.qpos[self.qpos_ids] = q
        self._gravity_data.qvel[self.dof_ids] = dq
        mujoco.mj_forward(self.model, self._gravity_data)
        ee = self._gravity_data.site_xpos[self.ee_site_id].copy()
        return np.concatenate((ee, q, dq))

    def gravity_torque(self) -> np.ndarray:
        """Return gravity-only generalized force at the current joint position."""

        self._gravity_data.qpos[:] = self.data.qpos
        self._gravity_data.qvel[:] = 0.0
        self._gravity_data.qacc[:] = 0.0
        self._gravity_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self._gravity_data)
        return self._gravity_data.qfrc_bias[self.dof_ids].astype(np.float64, copy=True)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if options and "qpos" in options:
            qpos = np.asarray(options["qpos"], dtype=np.float64)
        else:
            qpos = self.HOME + self.np_random.uniform(
                -self.config.initial_position_span,
                self.config.initial_position_span,
                size=6,
            )
        if options and "qvel" in options:
            qvel = np.asarray(options["qvel"], dtype=np.float64)
        else:
            qvel = self.np_random.uniform(
                -self.config.initial_velocity_span,
                self.config.initial_velocity_span,
                size=6,
            )
        if qpos.shape != (6,) or qvel.shape != (6,):
            raise ValueError("qpos and qvel must each have shape (6,)")

        self.data.qpos[self.qpos_ids] = qpos
        self.data.qvel[self.dof_ids] = qvel
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.last_residual_torque.fill(0.0)
        self.last_gravity_torque.fill(0.0)
        self.last_applied_torque.fill(0.0)
        return self._get_state(), self._info()

    def step(
        self,
        residual_torque: np.ndarray,
        *,
        external_force: np.ndarray | None = None,
        external_joint_torque: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        residual = np.asarray(residual_torque, dtype=np.float64)
        if residual.shape != (6,) or not np.all(np.isfinite(residual)):
            raise ValueError("residual_torque must be finite with shape (6,)")
        residual = np.clip(residual, -self.residual_limits, self.residual_limits)
        gravity = self.config.gravity_compensation_scale * self.gravity_torque()
        applied = np.clip(gravity + residual, -self.torque_limits, self.torque_limits)

        self.data.ctrl[self.actuator_ids] = applied
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        if external_force is not None:
            force = np.asarray(external_force, dtype=np.float64)
            if force.shape != (3,) or not np.all(np.isfinite(force)):
                raise ValueError("external_force must be finite with shape (3,)")
            self.data.xfrc_applied[self.force_body_id, :3] = force
        if external_joint_torque is not None:
            joint_torque = np.asarray(external_joint_torque, dtype=np.float64)
            if joint_torque.shape != (6,) or not np.all(np.isfinite(joint_torque)):
                raise ValueError("external_joint_torque must be finite with shape (6,)")
            self.data.qfrc_applied[self.dof_ids] = joint_torque

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.last_residual_torque = residual.copy()
        self.last_gravity_torque = gravity.copy()
        self.last_applied_torque = applied.copy()
        state = self._get_state()
        terminated = self._unsafe(state)
        if self.render_mode == "human":
            self.render()
        return state, 0.0, terminated, False, self._info()

    def _unsafe(self, state: np.ndarray) -> bool:
        if not np.all(np.isfinite(state)):
            return True
        q, dq = state[3:9], state[9:15]
        limited = self.model.jnt_limited[self.joint_ids].astype(bool)
        ranges = self.model.jnt_range[self.joint_ids]
        joint_violation = np.any(q[limited] < ranges[limited, 0]) or np.any(q[limited] > ranges[limited, 1])
        return bool(joint_violation or np.any(np.abs(dq) > self.config.velocity_limit))

    def _info(self) -> dict[str, Any]:
        return {
            "residual_torque": self.last_residual_torque.copy(),
            "gravity_torque": self.last_gravity_torque.copy(),
            "applied_torque": self.last_applied_torque.copy(),
            "ee_position": self.end_effector_position(),
            "control_timestep": self.config.control_timestep,
        }

    def mass_matrix(self) -> np.ndarray:
        dense = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, dense, self.data.qM)
        return dense[np.ix_(self.dof_ids, self.dof_ids)]

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
