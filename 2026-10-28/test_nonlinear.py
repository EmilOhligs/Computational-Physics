"""Tests for nonlinear.py (exercise 3, part 2): solvers for x' = f(x) with Newton for implicit Euler.

For a linear f(x) = J x the general solvers must reproduce the matrix solvers of ode.py,
and the Newton step must satisfy its defining equation y - x - dt f(y) = 0 for nonlinear f.
"""

import numpy as np
import pytest

from nonlinear import euler_f, heun_f, implicit_euler_f, integrate_grid, rk4_f
from ode import euler_step, heun_step, implicit_euler_step, jacobian, rk4_step

J = jacobian(1.0, 10.0)
X = np.array([0.6, 0.3, 0.1])


@pytest.mark.parametrize("step_f, step", [
    (euler_f, euler_step), (heun_f, heun_step), (rk4_f, rk4_step), (implicit_euler_f, implicit_euler_step),
])
def test_linear_f_matches_matrix_solvers(step_f, step):
    assert np.allclose(step_f(lambda x: J @ x, lambda x: J, X, 0.05), step(J, X, 0.05))


def test_implicit_euler_solves_nonlinear_equation():
    f = lambda x: -x**3
    jac = lambda x: np.diag(-3 * x**2)
    x = np.array([1.0, 2.0])
    y = implicit_euler_f(f, jac, x, 0.5)
    assert np.allclose(y - x - 0.5 * f(y), 0, atol=1e-10)


def test_implicit_euler_raises_without_convergence():
    f = lambda x: -x**3
    jac = lambda x: np.diag(-3 * x**2)
    with pytest.raises(RuntimeError):
        implicit_euler_f(f, jac, np.array([1.0]), 1.0, tol=1e-15, max_iter=1)


def test_integrate_grid_uses_nonuniform_steps():
    x = integrate_grid(euler_f, lambda x: -x, None, np.array([1.0]), np.array([0.0, 0.1, 0.3]))
    assert x.shape == (3, 1)
    assert np.allclose(x[:, 0], [1.0, 0.9, 0.72])
