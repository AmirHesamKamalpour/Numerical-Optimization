"""Generalized Rosenbrock objective and its analytical derivatives."""

from __future__ import annotations

import numpy as np


def _as_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("Rosenbrock objective expects a 1-D vector with at least two entries.")
    return x


def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """Return the generalized Rosenbrock objective.

    f(x) = sum_i b (x_{i+1} - x_i^2)^2 + (a - x_i)^2.
    """
    x = _as_vector(x)
    x_i = x[:-1]
    x_i_plus_1 = x[1:]
    return float(np.sum(b * (x_i_plus_1 - x_i**2) ** 2 + (a - x_i) ** 2))


def rosenbrock_gradient(
    x: np.ndarray, a: float = 1.0, b: float = 100.0
) -> np.ndarray:
    """Return the analytical gradient of the generalized Rosenbrock objective."""
    x = _as_vector(x)
    grad = np.zeros_like(x, dtype=float)

    grad[0] = -4 * b * x[0] * (x[1] - x[0] ** 2) + 2 * (x[0] - a)

    if x.size > 2:
        x_k = x[1:-1]
        grad[1:-1] = (
            2 * b * (x_k - x[:-2] ** 2)
            - 4 * b * x_k * (x[2:] - x_k**2)
            + 2 * (x_k - a)
        )

    grad[-1] = 2 * b * (x[-1] - x[-2] ** 2)
    return grad


def rosenbrock_hessian(
    x: np.ndarray, a: float = 1.0, b: float = 100.0
) -> np.ndarray:
    """Return the analytical Hessian of the generalized Rosenbrock objective."""
    del a  # The Hessian is independent of a.
    x = _as_vector(x)
    n = x.size
    hessian = np.zeros((n, n), dtype=float)

    diagonal = np.zeros(n, dtype=float)
    diagonal[0] = 12 * b * x[0] ** 2 - 4 * b * x[1] + 2

    if n > 2:
        x_k = x[1:-1]
        diagonal[1:-1] = 12 * b * x_k**2 - 4 * b * x[2:] + 2 * b + 2

    diagonal[-1] = 2 * b
    np.fill_diagonal(hessian, diagonal)

    off_diagonal = -4 * b * x[:-1]
    idx = np.arange(n - 1)
    hessian[idx, idx + 1] = off_diagonal
    hessian[idx + 1, idx] = off_diagonal
    return hessian
