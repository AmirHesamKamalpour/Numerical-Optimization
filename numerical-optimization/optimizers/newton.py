"""One-dimensional Newton method from Question 2."""

from __future__ import annotations

import numpy as np

from .line_search import backtracking_line_search


def newton_method(
    f,
    grad_f,
    hess_f,
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 1000,
    alpha_init: float = 1.0,
    beta: float = 0.5,
    sigma: float = 0.1,
):
    """Newton's method with Armijo backtracking for a scalar objective."""
    x = float(x0)
    xs = [x]
    alphas = []

    for _ in range(max_iter):
        g = float(grad_f(x))
        if abs(g) < tol:
            break

        H = float(hess_f(x))
        if H == 0:
            raise ZeroDivisionError("Second derivative f''(x) is zero.")

        p = -g / H
        alpha = backtracking_line_search(
            f,
            grad_f,
            x,
            p,
            alpha_init=alpha_init,
            beta=beta,
            sigma=sigma,
            min_alpha=1e-14,
            raise_on_failure=True,
        )
        alphas.append(alpha)
        x = x + alpha * p
        xs.append(x)

    return np.asarray(xs, dtype=float), alphas
