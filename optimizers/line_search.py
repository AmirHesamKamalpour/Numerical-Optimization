"""Backtracking and cubic-interpolation line-search utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _directional_derivative(gradient, direction) -> float:
    g = np.asarray(gradient, dtype=float).reshape(-1)
    p = np.asarray(direction, dtype=float).reshape(-1)
    return float(np.dot(g, p))


def backtracking_line_search(
    f: Callable,
    gradient_f: Callable,
    x,
    p,
    alpha_init: float = 1.0,
    beta: float = 0.5,
    sigma: float = 0.1,
    min_alpha: float = 1e-12,
    raise_on_failure: bool = False,
) -> float:
    """Armijo backtracking line search.

    Parameters match the notebook implementation. The method requires ``p``
    to be a descent direction at ``x``.
    """
    if alpha_init <= 0:
        raise ValueError("alpha_init must be positive.")
    if not 0 < beta < 1:
        raise ValueError("beta must lie in (0, 1).")
    if not 0 < sigma < 1:
        raise ValueError("sigma must lie in (0, 1).")

    alpha = float(alpha_init)
    fx = f(x)
    grad_fx_p = _directional_derivative(gradient_f(x), p)

    if grad_fx_p >= 0:
        raise ValueError("Search direction p is not a descent direction.")

    while f(x + alpha * p) > fx + sigma * alpha * grad_fx_p:
        alpha *= beta
        if alpha < min_alpha:
            if raise_on_failure:
                raise RuntimeError("Backtracking failed: step size became too small.")
            break

    return alpha


def cubic_coefficients(a: float, b: float, fa: float, fpa: float, fb: float, fpb: float):
    """Fit a cubic polynomial matching values and derivatives at a and b."""
    matrix = np.array(
        [
            [a**3, a**2, a, 1.0],
            [3 * a**2, 2 * a, 1.0, 0.0],
            [b**3, b**2, b, 1.0],
            [3 * b**2, 2 * b, 1.0, 0.0],
        ],
        dtype=float,
    )
    y = np.array([fa, fpa, fb, fpb], dtype=float)
    return np.linalg.solve(matrix, y)


def cubic_eval(x: float, c3: float, c2: float, c1: float, c0: float) -> float:
    return float(c3 * x**3 + c2 * x**2 + c1 * x + c0)


def cubic_eval_prime(x: float, c3: float, c2: float, c1: float) -> float:
    return float(3 * c3 * x**2 + 2 * c2 * x + c1)


def cubic_minimizer(a: float, b: float, c3: float, c2: float, c1: float) -> float:
    """Return a stationary point inside (a, b), or the midpoint as fallback."""
    A = 3 * c3
    B = 2 * c2
    C = c1

    if abs(A) < np.finfo(float).eps:
        if abs(B) < np.finfo(float).eps:
            return 0.5 * (a + b)
        root = -C / B
        return float(root if a < root < b else 0.5 * (a + b))

    discriminant = B * B - 4 * A * C
    if discriminant < 0:
        return 0.5 * (a + b)

    sqrt_discriminant = np.sqrt(discriminant)
    r1 = (-B + sqrt_discriminant) / (2 * A)
    r2 = (-B - sqrt_discriminant) / (2 * A)

    for root in (r1, r2):
        if a < root < b:
            return float(root)

    return 0.5 * (a + b)


def cubic_interpolation_minimize(
    f: Callable[[float], float],
    fp: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 50,
):
    """Minimize a 1-D function by repeatedly fitting a cubic interpolant."""
    fa, fpa = f(a), fp(a)
    fb, fpb = f(b), fp(b)

    if not (fpa < 0 and (fpb > 0 or fb > fa)):
        raise ValueError(
            "Bracket must satisfy f'(a) < 0 and either f'(b) > 0 or f(b) > f(a)."
        )

    for _ in range(max_iter):
        c3, c2, c1, c0 = cubic_coefficients(a, b, fa, fpa, fb, fpb)
        c = cubic_minimizer(a, b, c3, c2, c1)
        fc, fpc = f(c), fp(c)

        if abs(fpc) < tol or abs(b - a) < tol:
            return c, fc

        if fpc < 0:
            a, fa, fpa = c, fc, fpc
        else:
            b, fb, fpb = c, fc, fpc

    x_star = 0.5 * (a + b)
    return x_star, f(x_star)
