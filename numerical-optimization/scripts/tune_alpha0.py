"""Reproduce the alpha_0 sweeps"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from objectives import exp_function, exp_gradient, exp_hessian, rosenbrock, rosenbrock_gradient
from optimizers import newton_method, steepest_descent

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with (ROOT / "config" / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tune_steepest(cfg):
    rb = cfg["rosenbrock"]
    ls = cfg["line_search"]
    x0 = np.asarray(rb["x0"], dtype=float)
    f = lambda x: rosenbrock(x, a=rb["a"], b=rb["b"])
    g = lambda x: rosenbrock_gradient(x, a=rb["a"], b=rb["b"])

    print("Steepest descent alpha_0 sweep")
    for alpha0 in cfg["alpha0_candidates"]["steepest_descent"]:
        x_star, f_star, xs, _, _ = steepest_descent(
            f,
            g,
            x0,
            tol=rb["tol"],
            max_iter=rb["max_iter"],
            alpha_init=alpha0,
            beta=ls["beta"],
            sigma=ls["sigma"],
        )
        print(
            f"alpha0={alpha0:>4}: iterations={len(xs)-1:>5}, "
            f"f_min={f_star:.6e}, x*={x_star}"
        )


def tune_newton(cfg):
    exp_cfg = cfg["exponential"]
    ls = cfg["line_search"]

    print("Newton alpha_0 sweep")
    for alpha0 in cfg["alpha0_candidates"]["newton"]:
        xs, _ = newton_method(
            exp_function,
            exp_gradient,
            exp_hessian,
            x0=exp_cfg["x0"],
            tol=exp_cfg["tol"],
            max_iter=exp_cfg["max_iter"],
            alpha_init=alpha0,
            beta=ls["beta"],
            sigma=ls["sigma"],
        )
        print(f"alpha0={alpha0:>3}: iterations={len(xs)-1}")


def main(method: str):
    cfg = load_config()
    if method in {"steepest", "all"}:
        tune_steepest(cfg)
    if method == "all":
        print()
    if method in {"newton", "all"}:
        tune_newton(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["steepest", "newton", "all"],
        default="all",
        help="Choose which alpha_0 sweep to run.",
    )
    args = parser.parse_args()
    main(args.method)
