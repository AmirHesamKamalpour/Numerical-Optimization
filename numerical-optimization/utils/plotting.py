"""Reusable plots for optimization histories."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .metrics import distance_to_target


def plot_function_values(f_values, title: str = "Function value along iterations", logy=False):
    values = np.asarray(f_values, dtype=float)
    if logy:
        plt.semilogy(range(len(values)), values)
    else:
        plt.plot(range(len(values)), values)
    plt.xlabel("Iteration k")
    plt.ylabel("f(x_k)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def plot_errors(xs, target, title: str = "Distance to minimizer", logy=True):
    errors = distance_to_target(xs, target)
    if logy:
        plt.semilogy(range(len(errors)), errors)
    else:
        plt.plot(range(len(errors)), errors)
    plt.xlabel("Iteration k")
    plt.ylabel("||x_k - x*||")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def plot_step_lengths(alphas, title: str = "Step lengths"):
    plt.plot(range(len(alphas)), alphas)
    plt.xlabel("Iteration k")
    plt.ylabel("alpha_k")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
