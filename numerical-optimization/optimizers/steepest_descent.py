"""Steepest-descent methods from Question 1."""

from __future__ import annotations

import numpy as np

from .line_search import backtracking_line_search


def steepest_descent(
    f,
    grad_f,
    x0,
    tol: float = 1e-6,
    max_iter: int = 500,
    alpha_init: float = 1.0,
    beta: float = 0.5,
    sigma: float = 0.1,
):
    """Steepest descent with Armijo backtracking."""
    x = np.asarray(x0, dtype=float).copy()
    xs = [x.copy()]
    f_vals = [f(x)]
    alphas = []

    for _ in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break

        p = -g
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
        x = x + alpha * p
        xs.append(x.copy())
        f_vals.append(f(x))

    return x, f(x), xs, f_vals, alphas


def steepest_descent_bb(
    f,
    grad_f,
    x0,
    tol: float = 1e-6,
    max_iter: int = 500,
    beta: float = 0.5,
    sigma: float = 0.1,
):
    """Steepest descent with a Barzilai-Borwein trial step for alpha_0."""
    x = np.asarray(x0, dtype=float).copy()
    xs = [x.copy()]
    f_vals = [f(x)]
    alphas = []

    alpha_prev = 1.0
    g_prev = grad_f(x)
    x_prev = x.copy()

    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break

        p = -g
        if k == 0:
            alpha_init = alpha_prev
        else:
            s = x - x_prev
            y = g - g_prev
            sy = float(np.dot(s, y))
            ss = float(np.dot(s, s))
            alpha_init = ss / sy if sy > 1e-14 else alpha_prev

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

        x_prev = x.copy()
        g_prev = g.copy()
        x = x + alpha * p
        xs.append(x.copy())
        f_vals.append(f(x))
        alpha_prev = alpha

    return x, f(x), xs, f_vals, alphas
