"""Safe joint-space references for UR5e torque-control examples."""

from __future__ import annotations

import numpy as np

from UR5e.UR5e_Env import UR5eTorqueEnv


class JointTrajectoryGenerator:
    """Smooth multi-sine reference centered on the standard UR5e home pose."""

    def __init__(self, amplitude: float = 0.08, frequency_hz: float = 0.20):
        if amplitude < 0.0 or frequency_hz <= 0.0:
            raise ValueError("amplitude must be non-negative and frequency_hz positive")
        self.amplitude = amplitude * np.array([1.0, 0.8, 0.8, 0.6, 0.5, 0.5])
        self.frequency = frequency_hz * np.array([1.0, 0.8, 1.2, 1.4, 1.6, 1.8])
        self.phase = np.linspace(0.0, np.pi, 6)

    def sample(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        omega = 2.0 * np.pi * self.frequency
        phase = omega * time_s + self.phase
        q = UR5eTorqueEnv.HOME + self.amplitude * np.sin(phase)
        dq = self.amplitude * omega * np.cos(phase)
        return q, dq
