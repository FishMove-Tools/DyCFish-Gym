"""The supplied CFD target task's single observation/action contract.

No ROM code or optional solver libraries are imported here. The positive
frequency floor is numerical protection, not a validated physical limit.
"""
from __future__ import annotations
import numpy as np


def action_bounds():
    return (np.array([np.finfo(np.float32).eps, 0.0], dtype=np.float32),
            np.array([2.0, np.pi / 4], dtype=np.float32))


def decode_action(action):
    values = np.asarray(action, dtype=np.float64)
    low, high = action_bounds()
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError('Action must contain exactly two finite numbers.')
    if (np.any(values < np.nextafter(low, np.float32(-np.inf))) or
            np.any(values > np.nextafter(high, np.float32(np.inf)))):
        raise ValueError(f'Action {values.tolist()} outside {low.tolist()}..{high.tolist()}')
    frequency, amplitude = values
    if frequency <= 0:
        raise ValueError('Frequency must be positive.')
    return float(amplitude), float(frequency)


def observation(state):
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError('Kinematic observation must have seven finite values.')
    return values.astype(np.float32)


def interface_contract():
    low, high = action_bounds()
    return {'schema_version': 1, 'stage': 'cfd', 'interface_profile': 'legacy',
            'task': 'target',
            'observation_order': ['x_m', 'y_m', 'yaw_rad', 'vx_world_m_s',
                                  'vy_world_m_s', 'yaw_rate_rad_s', 'time_s'],
            'action_order': ['frequency_hz', 'amplitude_rad'],
            'action_low': low.tolist(), 'action_high': high.tolist()}
