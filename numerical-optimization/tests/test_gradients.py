"""Finite-difference checks for the homework derivatives."""

from __future__ import annotations

import numpy as np

from objectives import (
    exp_function,
    exp_gradient,
    exp_hessian,
    rosenbrock,
    rosenbrock_gradient,
    rosenbrock_hessian,
)


def finite_difference_gradient(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        grad[i] = (f(x + step) - f(x - step)) / (2 * eps)
    return grad


def finite_difference_hessian_from_gradient(grad_f, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    n = x.size
    hessian = np.zeros((n, n))
    for j in range(n):
        step = np.zeros_like(x)
        step[j] = eps
        hessian[:, j] = (grad_f(x + step) - grad_f(x - step)) / (2 * eps)
    return hessian


def test_rosenbrock_gradient_matches_finite_difference():
    x = np.array([0.1, -0.9, 0.4, 1.2])
    numerical = finite_difference_gradient(rosenbrock, x)
    analytical = rosenbrock_gradient(x)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-5)


def test_rosenbrock_hessian_matches_finite_difference():
    x = np.array([0.1, -0.9, 0.4, 1.2])
    numerical = finite_difference_hessian_from_gradient(rosenbrock_gradient, x)
    analytical = rosenbrock_hessian(x)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(analytical, analytical.T)


def test_exponential_derivatives_match_finite_difference():
    x = 2.1
    eps = 1e-6
    numerical_grad = (exp_function(x + eps) - exp_function(x - eps)) / (2 * eps)
    numerical_hess = (exp_gradient(x + eps) - exp_gradient(x - eps)) / (2 * eps)
    np.testing.assert_allclose(exp_gradient(x), numerical_grad, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(exp_hessian(x), numerical_hess, rtol=1e-6, atol=1e-8)
