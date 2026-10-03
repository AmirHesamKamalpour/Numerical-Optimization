"""Small metrics"""

from __future__ import annotations

import numpy as np


def distance_to_target(xs, target) -> np.ndarray:
    """Euclidean distance from every iterate in ``xs`` to ``target``."""
    xs_array = np.asarray(xs, dtype=float)
    target_array = np.asarray(target, dtype=float)

    if xs_array.ndim == 1 and target_array.ndim == 0:
        return np.abs(xs_array - target_array)
    return np.linalg.norm(xs_array - target_array, axis=1)


def iteration_count(xs) -> int:
    """Number of optimization steps represented by an iterate history."""
    return max(0, len(xs) - 1)
