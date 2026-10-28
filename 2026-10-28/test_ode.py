"""Tests for the ODE solvers of the reaction A -> B -> C (exercise 3, 1a/1b).

Run with:  python -m pytest -v

The reaction is linear, x' = J x with x = (A, B, C). For a linear ODE every
method is a matrix power series in dt J, so one step can be checked exactly:

    explicit Euler   x + dt J x
    Heun             (I + dt J + (dt J)^2 / 2) x
    RK4              (I + dt J + ... + (dt J)^4 / 24) x
    implicit Euler   (I - dt J)^-1 x

Whole runs are checked against the analytical C(t) and the conservation
of A + B + C.
"""

import numpy as np
import pytest

from ode import c_exact, euler_step, heun_step, implicit_euler_step, integrate, jacobian, rk4_step

X0 = np.array([1.0, 0.0, 0.0])
STEPS = [euler_step, heun_step, rk4_step, implicit_euler_step]


# --- 1a: standard form -------------------------------------------------------

def test_jacobian_entries():
    assert np.array_equal(jacobian(1.0, 10.0), [[-1, 0, 0], [1, -10, 0], [0, 10, 0]])


def test_jacobian_columns_sum_to_zero():
    # d/dt (A + B + C) = (1, 1, 1) J x = 0 for every x
    assert np.allclose(jacobian(2.0, 7.0).sum(axis=0), 0)


def test_c_exact_both_branches():
    assert c_exact(0.0, 1.0, 10.0) == pytest.approx(0)
    assert c_exact(1.0, 1.0, 1.0) == pytest.approx(1 - 2 / np.e)
    # k2 -> k1 limit: the general formula approaches the k1 = k2 formula
    assert c_exact(1.0, 1.0, 1.0 + 1e-6) == pytest.approx(1 - 2 / np.e, abs=1e-6)
    assert np.allclose(c_exact(np.array([0.0, 50.0]), 1.0, 10.0), [0, 1])


# --- 1b: one step of each method --------------------------------------------

def test_single_steps_match_series():
    J = jacobian(1.0, 10.0)
    dt = 0.05
    x = np.array([0.6, 0.3, 0.1])
    A = dt * J
    I = np.eye(3)
    A2 = A @ A
    assert np.allclose(euler_step(J, x, dt), (I + A) @ x)
    assert np.allclose(heun_step(J, x, dt), (I + A + A2 / 2) @ x)
    assert np.allclose(rk4_step(J, x, dt), (I + A + A2 / 2 + A2 @ A / 6 + A2 @ A2 / 24) @ x)
    assert np.allclose(implicit_euler_step(J, x, dt), np.linalg.solve(I - A, x))


# --- 1b: whole runs ----------------------------------------------------------

def test_integrate_time_grid():
    t, x = integrate(euler_step, jacobian(1.0, 10.0), X0, 0.1, 10.0)
    assert len(t) == 101 and x.shape == (101, 3)
    assert t[0] == 0 and t[-1] == pytest.approx(10)
    assert np.array_equal(x[0], X0)


@pytest.mark.parametrize("step", STEPS)
def test_total_concentration_conserved(step):
    _, x = integrate(step, jacobian(1.0, 10.0), X0, 0.01, 10.0)
    assert np.allclose(x.sum(axis=1), 1)


@pytest.mark.parametrize("step, dt, tol", [
    (euler_step, 1e-3, 1e-2),
    (implicit_euler_step, 1e-3, 1e-2),
    (heun_step, 1e-2, 1e-3),
    (rk4_step, 1e-2, 1e-6),
])
def test_c_matches_exact(step, dt, tol):
    t, x = integrate(step, jacobian(1.0, 10.0), X0, dt, 10.0)
    assert np.max(np.abs(x[:, 2] - c_exact(t, 1.0, 10.0))) < tol
