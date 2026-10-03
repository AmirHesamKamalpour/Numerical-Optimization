"""Run the main experiments from both homework questions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from objectives import (
    exp_function,
    exp_gradient,
    exp_hessian,
    exp_true_minimizer,
    rosenbrock,
    rosenbrock_gradient,
)
from optimizers import (
    bfgs_quasi_newton,
    cubic_interpolation_minimize,
    newton_method,
    steepest_descent,
    steepest_descent_bb,
)
from utils import plot_errors, plot_function_values, plot_step_lengths


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main(show_plots: bool = False):
    cfg = load_config(ROOT / "config" / "config.yaml")

    rb = cfg["rosenbrock"]
    ls = cfg["line_search"]
    x0 = np.asarray(rb["x0"], dtype=float)

    f = lambda x: rosenbrock(x, a=rb["a"], b=rb["b"])
    g = lambda x: rosenbrock_gradient(x, a=rb["a"], b=rb["b"])

    sd = steepest_descent(
        f,
        g,
        x0,
        tol=rb["tol"],
        max_iter=rb["max_iter"],
        alpha_init=ls["alpha_init"],
        beta=ls["beta"],
        sigma=ls["sigma"],
    )
    sd_bb = steepest_descent_bb(
        f,
        g,
        x0,
        tol=rb["tol"],
        max_iter=rb["max_iter"],
        beta=ls["beta"],
        sigma=ls["sigma"],
    )
    bfgs = bfgs_quasi_newton(
        f,
        g,
        x0,
        tol=rb["tol"],
        max_iter=cfg["bfgs"]["max_iter"],
        alpha_init=ls["alpha_init"],
        beta=ls["beta"],
        sigma=ls["sigma"],
    )

    print("Question 1 — Rosenbrock")
    print(f"Steepest descent: iterations={len(sd[2]) - 1}, f*={sd[1]:.6e}, x*={sd[0]}")
    print(f"Steepest descent + BB: iterations={len(sd_bb[2]) - 1}, f*={sd_bb[1]:.6e}, x*={sd_bb[0]}")
    print(f"BFGS: iterations={len(bfgs[2]) - 1}, f*={bfgs[1]:.6e}, x*={bfgs[0]}")

    exp_cfg = cfg["exponential"]
    newton_xs, newton_alphas = newton_method(
        exp_function,
        exp_gradient,
        exp_hessian,
        x0=exp_cfg["x0"],
        tol=exp_cfg["tol"],
        max_iter=exp_cfg["max_iter"],
        alpha_init=ls["alpha_init"],
        beta=ls["beta"],
        sigma=ls["sigma"],
    )
    x_newton = newton_xs[-1]

    cubic_cfg = cfg["cubic_interpolation"]
    x_cubic, f_cubic = cubic_interpolation_minimize(
        exp_function,
        exp_gradient,
        a=cubic_cfg["a"],
        b=cubic_cfg["b"],
        tol=cubic_cfg["tol"],
        max_iter=cubic_cfg["max_iter"],
    )

    print("\nQuestion 2 — Exponential objective")
    print(
        f"Newton: iterations={len(newton_xs) - 1}, x*={x_newton:.12f}, "
        f"f*={float(exp_function(x_newton)):.12f}"
    )
    print(f"Cubic interpolation: x*={x_cubic:.12f}, f*={float(f_cubic):.12f}")
    print(f"Analytical minimizer: x*={exp_true_minimizer():.12f}")

    if show_plots:
        plot_function_values(sd[3], "Steepest descent: function value")
        plt.show()
        plot_errors(sd[2], np.ones_like(x0), "Steepest descent: distance to minimizer")
        plt.show()
        plot_step_lengths(sd[4], "Steepest descent: step lengths")
        plt.show()

        plot_function_values(bfgs[3], "BFGS: function value")
        plt.show()
        plot_errors(bfgs[2], np.ones_like(x0), "BFGS: distance to minimizer")
        plt.show()
        plot_step_lengths(bfgs[4], "BFGS: step lengths")
        plt.show()

        newton_values = exp_function(newton_xs)
        plot_function_values(newton_values, "Newton: function value")
        plt.show()
        plot_errors(newton_xs, exp_true_minimizer(), "Newton: distance to minimizer")
        plt.show()
        plot_step_lengths(newton_alphas, "Newton: step lengths")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Display convergence plots.")
    args = parser.parse_args()
    main(show_plots=args.plot)
