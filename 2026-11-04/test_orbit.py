"""Tests for exercise 1. What each test checks: THEORIE.md, sections 1a to 1d."""

import numpy as np
import pytest

from orbit import (
    GM,
    HOUR,
    R_MOON,
    acceleration,
    angular_momentum,
    circular_orbit,
    energy,
    error_at_half_period,
    euler,
    heun,
    implicit_euler,
    implicit_euler_solve,
    integrate,
    jacobian,
    order,
    period,
    rhs,
    rk4,
    symplectic_euler,
    velocity_verlet,
)

METHODS = [euler, implicit_euler, heun, symplectic_euler, velocity_verlet, rk4]


def moon_state():
    r0, v0 = circular_orbit(R_MOON)
    return np.concatenate([r0, v0])


def test_circular_orbit_starts_on_x_axis_with_tangential_speed():
    r0, v0 = circular_orbit(R_MOON)
    assert np.allclose(r0, [R_MOON, 0.0])
    assert v0[0] == 0.0
    # Centripetal acceleration v^2/r equals gravity GM/r^2
    assert np.isclose(v0[1] ** 2 / R_MOON, GM / R_MOON**2)


def test_period_matches_kepler():
    # T must be circumference / speed, and about 27.45 d for the sheet's constants
    r0, v0 = circular_orbit(R_MOON)
    assert np.isclose(period(R_MOON), 2 * np.pi * R_MOON / v0[1])
    assert np.isclose(period(R_MOON) / 86400, 27.45, atol=0.01)


def test_acceleration_points_to_earth_with_inverse_square_magnitude():
    r = np.array([3.0e8, 4.0e8])  # |r| = 5e8
    a = acceleration(r)
    assert np.allclose(a, -GM / 5.0e8**2 * r / 5.0e8)


def test_rhs_of_first_order_system_is_velocity_and_acceleration():
    y = moon_state()
    assert np.allclose(rhs(y), np.concatenate([y[2:], acceleration(y[:2])]))


def test_jacobian_matches_finite_differences():
    y = np.array([3.0e8, -1.0e8, 200.0, 900.0])
    h = np.array([1e3, 1e3, 1e-3, 1e-3])
    numeric = np.column_stack(
        [(rhs(y + h[k] * e) - rhs(y - h[k] * e)) / (2 * h[k]) for k, e in enumerate(np.eye(4))]
    )
    assert np.allclose(jacobian(y), numeric, rtol=1e-6, atol=1e-20)


def test_newton_solves_implicit_equation_in_few_iterations():
    y = moon_state()
    dt = 6 * HOUR
    y_new, iterations = implicit_euler_solve(y, dt)
    assert np.allclose(y_new, y + dt * rhs(y_new), rtol=1e-12, atol=0)
    assert 1 <= iterations <= 5
    assert np.array_equal(implicit_euler(y, dt), y_new)


def test_integrate_returns_all_states_from_start():
    y = moon_state()
    ys = integrate(euler, y, 6 * HOUR, 3)
    assert ys.shape == (4, 4)
    assert np.array_equal(ys[0], y)
    assert np.array_equal(ys[1], euler(y, 6 * HOUR))


def test_integrate_stops_when_implicit_euler_collapses():
    # With dt = 6 h implicit Euler spirals into the Earth within half an orbit;
    # then the implicit equation has no nearby solution and Newton fails.
    n = 1098  # 10 orbits
    ys = integrate(implicit_euler, moon_state(), 6 * HOUR, n)
    assert 50 < len(ys) < n + 1
    assert np.linalg.norm(ys[-1, :2]) < 0.5 * R_MOON


@pytest.mark.parametrize("step", METHODS, ids=lambda s: s.__name__)
def test_all_methods_return_to_start_after_one_orbit_with_small_dt(step):
    # Convergence sanity check: with 20000 steps per orbit every method closes the circle
    # (2000 is too few for the first-order Euler methods: phase error ~0.19 r_M)
    n = 20000
    ys = integrate(step, moon_state(), period(R_MOON) / n, n)
    assert np.linalg.norm(ys[-1, :2] - ys[0, :2]) < 0.05 * R_MOON


# Prediction for 1b, 10 orbits with dt = 6 h. Outward = energy grows (larger semi-major axis)
SPIRAL = {
    "euler": "out",
    "heun": "out",
    "implicit_euler": "in",
    "rk4": "in",
    "symplectic_euler": "stays",
    "velocity_verlet": "stays",
}


@pytest.mark.parametrize("step", METHODS, ids=lambda s: s.__name__)
def test_spiral_direction_after_ten_orbits(step):
    dt = 6 * HOUR
    n = int(round(10 * period(R_MOON) / dt))
    ys = integrate(step, moon_state(), dt, n)
    radius = np.linalg.norm(ys[:, :2], axis=1)
    energy = 0.5 * np.sum(ys[:, 2:] ** 2, axis=1) - GM / radius
    expected = SPIRAL[step.__name__]
    if expected == "out":
        assert energy[-1] > energy[0]
    elif expected == "in":
        assert energy[-1] < energy[0]
    else:
        # symplectic: radius oscillates, but the last orbit looks like the first (no drift)
        per_orbit = n // 10
        first, last = radius[:per_orbit], radius[-per_orbit:]
        assert abs(last.max() - first.max()) < 0.01 * R_MOON
        assert abs(last.min() - first.min()) < 0.01 * R_MOON


def test_energy_and_angular_momentum_of_circular_orbit():
    # Circle: E = v^2/2 - GM/r = -GM/(2r), L = r v = sqrt(GM r)
    ys = moon_state()[None, :]
    assert np.allclose(energy(ys), -GM / (2 * R_MOON))
    assert np.allclose(angular_momentum(ys), np.sqrt(GM * R_MOON))


@pytest.fixture(scope="module")
def hundred_orbits():
    dt = 6 * HOUR
    n = int(round(100 * period(R_MOON) / dt))
    return n, {step.__name__: integrate(step, moon_state(), dt, n) for step in METHODS}


def relative_errors(ys):
    e, l = energy(ys), angular_momentum(ys)
    return np.abs(e / e[0] - 1), np.abs(l / l[0] - 1)


@pytest.mark.parametrize("name", ["symplectic_euler", "velocity_verlet"])
def test_symplectic_methods_conserve_angular_momentum_to_roundoff(hundred_orbits, name):
    _, dl = relative_errors(hundred_orbits[1][name])
    assert dl.max() < 1e-12


@pytest.mark.parametrize("name", ["euler", "implicit_euler", "heun", "rk4"])
def test_other_methods_change_angular_momentum(hundred_orbits, name):
    _, dl = relative_errors(hundred_orbits[1][name])
    assert dl.max() > 1e-8


@pytest.mark.parametrize("name", ["symplectic_euler", "velocity_verlet"])
def test_symplectic_energy_error_stays_bounded(hundred_orbits, name):
    n, runs = hundred_orbits
    de, _ = relative_errors(runs[name])
    ten = n // 10
    assert de[-ten:].max() < 1.1 * de[:ten].max()


@pytest.mark.parametrize("name", ["heun", "rk4"])
def test_nonsymplectic_energy_error_grows(hundred_orbits, name):
    n, runs = hundred_orbits
    de, _ = relative_errors(runs[name])
    ten = n // 10
    assert de[-ten:].max() > 5 * de[:ten].max()


def test_heun_ends_worse_than_symplectic_euler(hundred_orbits):
    de_heun, _ = relative_errors(hundred_orbits[1]["heun"])
    de_symp, _ = relative_errors(hundred_orbits[1]["symplectic_euler"])
    assert de_heun[-1] > 5 * de_symp.max()


def test_error_at_half_period_is_distance_to_opposite_point():
    # Exact solution at t = T/2: r = (-r_M, 0); error relative to r_M
    n = 200
    ys = integrate(euler, moon_state(), period(R_MOON) / n, n // 2)
    expected = np.linalg.norm(ys[-1, :2] - [-R_MOON, 0.0]) / R_MOON
    assert np.isclose(error_at_half_period(euler, n), expected)


def test_order_is_slope_in_log_log():
    dts = np.array([1.0, 0.5, 0.25])
    assert np.isclose(order(dts, 3 * dts**2), 2.0)


ORDERS = {"euler": 1, "implicit_euler": 1, "heun": 2, "symplectic_euler": 1, "velocity_verlet": 2, "rk4": 4}


@pytest.mark.parametrize("step", METHODS, ids=lambda s: s.__name__)
def test_convergence_order_at_half_period(step):
    ns = np.array([200, 400, 800, 1600, 3200, 6400])
    errors = [error_at_half_period(step, n) for n in ns]
    assert abs(order(period(R_MOON) / ns, errors) - ORDERS[step.__name__]) < 0.1
