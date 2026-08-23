"""Residual joint-torque controllers for the UR5e environment."""

from __future__ import annotations

import numpy as np


class JointTorquePDController:
    """Compute a clipped residual torque from joint position/velocity error.

    Gravity feed-forward is deliberately owned by :class:`UR5eTorqueEnv`; this
    controller returns only the residual motor torque.  Keeping the two terms
    separate makes learned-model inputs and actuator commands auditable.
    """

    def __init__(
        self,
        torque_limits: np.ndarray,
        kp: np.ndarray | float = 45.0,
        kd: np.ndarray | float = 8.0,
    ) -> None:
        self.torque_limits = np.broadcast_to(np.asarray(torque_limits, dtype=np.float64), (6,)).copy()
        self.kp = np.broadcast_to(np.asarray(kp, dtype=np.float64), (6,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, dtype=np.float64), (6,)).copy()
        if np.any(self.torque_limits <= 0.0) or np.any(self.kp < 0.0) or np.any(self.kd < 0.0):
            raise ValueError("Torque limits must be positive and gains non-negative")

    def command(
        self,
        state: np.ndarray,
        q_reference: np.ndarray,
        dq_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        q_reference = np.asarray(q_reference, dtype=np.float64)
        dq_reference = np.zeros(6) if dq_reference is None else np.asarray(dq_reference, dtype=np.float64)
        if state.shape != (15,) or q_reference.shape != (6,) or dq_reference.shape != (6,):
            raise ValueError("Expected state (15,)=[ee,q,dq], q_reference (6,), dq_reference (6,)")
        q, dq = state[3:9], state[9:15]
        torque = self.kp * (q_reference - q) + self.kd * (dq_reference - dq)
        return np.clip(torque, -self.torque_limits, self.torque_limits)
