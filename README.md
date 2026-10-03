# Numerical Optimization Homework

A small, reproducible Python project for two numerical-optimization exercises. The original notebook has been refactored into separate objective functions, optimization algorithms, experiment scripts, configuration, plotting helpers, and derivative tests.

## What is implemented

### Question 1 — Generalized Rosenbrock function

The objective is

```math
f(x)=\sum_{i=1}^{n-1}\left[b\left(x_{i+1}-x_i^2\right)^2 + (a-x_i)^2\right],
```

with the homework defaults $a=1$ and $b=100$. For these parameters, the minimizer is

```math
x^*=(1,\ldots,1)^T, \qquad f(x^*)=0.
```

The project includes:

- analytical gradient and Hessian,
- Armijo backtracking line search,
- steepest descent,
- steepest descent with a Barzilai–Borwein trial step for $\alpha_0$,
- BFGS quasi-Newton with an inverse-Hessian approximation,
- convergence plots for objective values, iterate error, and accepted step lengths.

### Question 2 — One-dimensional exponential objective

The second objective is

```math
f_2(x)=e^{x-3}-\frac{x}{2}-2,
```

with

```math
f_2'(x)=e^{x-3}-\frac12,
\qquad
f_2''(x)=e^{x-3}.
```

Setting $f_2'(x)=0$ gives the analytical minimizer

```math
x^*=3-\ln 2 \approx 2.30685281944.
```

The project includes Newton's method with Armijo backtracking and the cubic-interpolation minimization routine from the notebook.

## Setup

```bash
git clone <your-repository-url>
cd numerical-optimization

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the experiments

Run both homework questions:

```bash
python -m scripts.run_comparison
```

Display the convergence plots as well:

```bash
python -m scripts.run_comparison --plot
```

Reproduce the $\alpha_0$ sweeps:

```bash
python -m scripts.tune_alpha0 --method all
```

You can also run only one sweep:

```bash
python -m scripts.tune_alpha0 --method steepest
python -m scripts.tune_alpha0 --method newton
```

All experiment defaults are kept in [`config/config.yaml`](config/config.yaml), so the starting points, tolerances, line-search constants, iteration limits, and candidate $\alpha_0$ values can be changed without editing the algorithms.

## Reproducible results

A fresh run of the project gives the following results for the 4-D Rosenbrock problem with $x_0=(0,0,0,0)^T$ and gradient tolerance $10^{-6}$:

| $\alpha_0$ | Steepest-descent iterations | Final objective |
|---:|---:|---:|
| 0.1 | 11,371 | $8.75\times10^{-13}$ |
| 0.5 | 17,117 | $7.06\times10^{-13}$ |
| 1.0 | 17,117 | $7.06\times10^{-13}$ |
| 2.0 | 17,117 | $7.06\times10^{-13}$ |
| 10.0 | 19,297 | $5.54\times10^{-13}$ |

With the default settings, the Barzilai–Borwein trial-step variant converges in 2,371 iterations with a final objective of about $1.01\times10^{-12}$. BFGS converges in 32 iterations with a final objective of about $2.69\times10^{-16}$.

For the exponential objective starting from $x_0=6$, Newton's method with $\alpha_0=1$ converges in 7 iterations to approximately

```math
x^*=2.3068536485, \qquad f_2(x^*)=-2.6534264097.
```

The Newton $\alpha_0$ sweep gives:

| $\alpha_0$ | Iterations |
|---:|---:|
| 0.1 | 162 |
| 0.5 | 27 |
| 1.0 | 7 |
| 2.0 | 5 |
| 5.0 | 10 |

The cubic-interpolation routine returns $x^*\approx2.30685281950$ and $f_2(x^*)\approx-2.65342640972$, consistent with the analytical minimizer $3-\ln2$.

## Notes on the implementation

- BFGS stores an approximation to the **inverse Hessian**, initialized with the identity matrix.
- Armijo backtracking rejects non-descent directions and reduces the trial step by `beta` until the sufficient-decrease condition is met.
- The Barzilai–Borwein variant changes the initial trial step supplied to backtracking; the accepted step still has to satisfy Armijo.
- The cubic-interpolation routine preserves the bracket logic used in the homework notebook.

## Dependencies

- NumPy
- Matplotlib
- PyYAML
- pytest
