"""One-dimensional exponential objective from Question 2."""

from __future__ import annotations

import numpy as np


def exp_function(x: float | np.ndarray) -> float | np.ndarray:
    """f(x) = exp(x - 3) - x/2 - 2."""
    return np.exp(x - 3) - x / 2 - 2


def exp_gradient(x: float | np.ndarray) -> float | np.ndarray:
    """First derivative of :func:`exp_function`."""
    return np.exp(x - 3) - 0.5


def exp_hessian(x: float | np.ndarray) -> float | np.ndarray:
    """Second derivative of :func:`exp_function`."""
    return np.exp(x - 3)


def exp_true_minimizer() -> float:
    """Return the analytical minimizer x* = 3 - log(2)."""
    return float(3 - np.log(2))
