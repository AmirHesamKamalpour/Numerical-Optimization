"""BFGS quasi-Newton method from Question 1."""

from __future__ import annotations

import numpy as np

from .line_search import backtracking_line_search


def bfgs_quasi_newton(
    f,
    grad_f,
    x0,
    tol: float = 1e-6,
    max_iter: int = 500,
    alpha_init: float = 1.0,
    beta: float = 0.5,
    sigma: float = 0.1,
):
    """BFGS using an inverse-Hessian approximation and Armijo backtracking."""
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    H = np.eye(n)

    xs = [x.copy()]
    f_vals = [f(x)]
    alphas = []
    Hs = [H.copy()]

    for _ in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break

        p = -H @ g
        alpha = backtracking_line_search(
            f,
            grad_f,
            x,
            p,
            alpha_init=alpha_init,
            beta=beta,
            sigma=sigma,
        )
        alphas.append(alpha)

        x_new = x + alpha * p
        s = x_new - x
        y = grad_f(x_new) - g

        sy = float(np.dot(s, y))
        if sy > 1e-10:
            rho = 1.0 / sy
            identity = np.eye(n)
            H = (
                (identity - rho * np.outer(s, y))
                @ H
                @ (identity - rho * np.outer(y, s))
                + rho * np.outer(s, s)
            )

        x = x_new
        xs.append(x.copy())
        f_vals.append(f(x))
        Hs.append(H.copy())

    return x, f(x), xs, f_vals, alphas, Hs
