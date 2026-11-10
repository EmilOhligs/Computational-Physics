"""Tests for exercise 2 (rocket launch). What each test checks: THEORIE.md, sections 2a to 2d."""

import numpy as np
import pytest

from orbit import GM, energy
from rocket import (
    OMEGA_EARTH,
    R_EARTH,
    V_EXHAUST,
    apoapsis,
    burn_time,
    classify,
    fly,
    fly_until,
    gravity_loss,
    launch,
    launch_state,
    mass_after,
    periapsis,
    rocket_rhs,
    thrust_direction,
)

G0 = GM / R_EARTH**2  # surface gravity, about 9.82 m/s^2
M1 = 1 / 19  # 2a: mass after burnout relative to m0


def test_launch_state_corotates_with_earth():
    y = launch_state()
    assert np.allclose(y, [R_EARTH, 0.0, 0.0, OMEGA_EARTH * R_EARTH])


def test_thrust_direction_vertical_east_and_west():
    r = np.array([0.0, R_EARTH])  # on the y axis: e_r = (0, 1), e_phi = (-1, 0)
    assert np.allclose(thrust_direction(r, 90.0), [0.0, 1.0])
    assert np.allclose(thrust_direction(r, 0.0, east=True), [-1.0, 0.0])
    assert np.allclose(thrust_direction(r, 0.0, east=False), [1.0, 0.0])
    e = thrust_direction(r, 30.0)
    assert np.isclose(np.linalg.norm(e), 1.0)
    assert np.isclose(e @ [0.0, 1.0], np.sin(np.radians(30.0)))


def test_burn_time_ends_when_mass_reaches_m1():
    # m(t)/m0 = 1 - (F/m0) t / v_e, since F = mdot v_e
    thrust = 12.0
    tb = burn_time(thrust, M1)
    assert np.isclose(1 - thrust * tb / V_EXHAUST, M1)


def test_rhs_on_launch_pad_and_after_burnout():
    thrust = 12.0
    y = launch_state()
    dy = rocket_rhs(0.0, y, thrust, M1)
    assert np.allclose(dy[:2], y[2:])
    assert np.allclose(dy[2:], [thrust - G0, 0.0])  # vertical launch at (R_E, 0)
    dy = rocket_rhs(burn_time(thrust, M1) + 1.0, y, thrust, M1)
    assert np.allclose(dy[2:], [-G0, 0.0])  # engine off: gravity only


def test_rhs_thrust_grows_as_mass_drops():
    # F/m(t) = (F/m0) / (m(t)/m0): halfway through the burn of m1 = 1/19
    thrust = 12.0
    t = burn_time(thrust, M1) / 2
    dy = rocket_rhs(t, launch_state(), thrust, M1)
    mass = 1 - thrust * t / V_EXHAUST
    assert np.isclose(dy[2] + G0, thrust / mass)


def test_launch_hits_burnout_time_on_grid():
    thrust = 12.0
    t, ys = launch(thrust, M1, t_end=600.0)
    assert t[0] == 0.0 and np.array_equal(ys[0], launch_state())
    assert np.isclose(t[-1], 600.0)
    assert np.any(np.isclose(t, burn_time(thrust, M1)))


def test_launch_stops_when_rocket_hits_the_ground():
    t, ys = launch(10.0, M1, t_end=24 * 3600.0)
    radius = np.linalg.norm(ys[:, :2], axis=1)
    assert t[-1] < 24 * 3600.0
    assert radius[-1] <= R_EARTH
    assert np.all(radius[1:-1] > R_EARTH)


@pytest.mark.parametrize("thrust", [10.0, 12.0, 14.0, 16.0, 18.0])
def test_burnout_speed_is_rocket_equation_minus_gravity_loss(thrust):
    tb = burn_time(thrust, M1)
    t, ys = launch(thrust, M1, t_end=tb)
    r, v = ys[-1, :2], ys[-1, 2:]
    v_radial = v @ r / np.linalg.norm(r)
    loss = V_EXHAUST * np.log(1 / M1) - v_radial
    # gravity loss: integral of g(r) over the burn, a bit less than g0 * tb because g falls with height
    assert 0.8 * G0 * tb < loss < G0 * tb


@pytest.mark.parametrize("thrust, escapes", [(10.0, False), (16.0, False), (18.0, True)])
def test_escape_only_with_high_thrust(thrust, escapes):
    t, ys = launch(thrust, M1, t_end=burn_time(thrust, M1))
    assert (energy(ys[-1:])[0] >= 0) == escapes


def test_periapsis_of_circle_ellipse_and_vertical_launch():
    r = 7000e3
    circle = np.array([r, 0.0, 0.0, np.sqrt(GM / r)])
    assert np.isclose(periapsis(circle), r)
    # Apoapsis r, periapsis rp: vis-viva gives v^2 = GM (2/r - 2/(r + rp)) at the apoapsis
    rp = 6600e3
    ellipse = np.array([r, 0.0, 0.0, np.sqrt(GM * (2 / r - 2 / (r + rp)))])
    assert np.isclose(periapsis(ellipse), rp)
    # Vertical launch: only the Earth's rotation gives angular momentum, the periapsis is deep inside the Earth
    t, ys = launch(16.0, M1, t_end=burn_time(16.0, M1))
    assert periapsis(ys[-1]) < 0.1 * R_EARTH


def test_apoapsis_of_ellipse_and_escape():
    r, rp = 7000e3, 6600e3
    ellipse = np.array([rp, 0.0, 0.0, np.sqrt(GM * (2 / rp - 2 / (r + rp)))])  # start at periapsis
    assert np.isclose(apoapsis(ellipse), r)
    escape = np.array([rp, 0.0, 0.0, 1.01 * np.sqrt(2 * GM / rp)])
    assert apoapsis(escape) == np.inf


def test_fly_coast_keeps_energy_and_hits_duration():
    y = np.array([7000e3, 0.0, 0.0, np.sqrt(GM / 7000e3)])
    t, ys = fly(y, 1000.0)
    assert np.isclose(t[-1], 1000.0) and np.array_equal(ys[0], y)
    assert np.isclose(energy(ys[-1:])[0], energy(y[None, :])[0], rtol=1e-9)


def test_fly_burn_matches_launch():
    # a burn phase from the pad is exactly the burn part of launch
    thrust, tb = 15.0, burn_time(15.0, 0.12)
    theta = lambda t: 90.0 * max(0.0, 1 - t / 175.0)
    t, ys = fly(launch_state(), tb, thrust=thrust, theta=theta)
    t_l, ys_l = launch(thrust, 0.12, t_end=tb, theta=theta)
    assert np.allclose(ys[-1], ys_l[-1])


def test_fly_burn_mass_after():
    # mass fraction left after burning `duration` with relative thrust F/m_start
    assert np.isclose(mass_after(1.0, 15.0, burn_time(15.0, 0.12)), 0.12)
    # the same engine (F = 15 m0) relit on a half-empty rocket burns the same fuel per second
    assert np.isclose(0.5 - mass_after(0.5, 30.0, 10.0), 1.0 - mass_after(1.0, 15.0, 10.0))


def test_fly_stops_on_impact():
    y = np.array([R_EARTH + 1000.0, 0.0, -100.0, 0.0])  # falling straight down from 1 km
    t, ys = fly(y, 600.0)
    assert t[-1] < 600.0
    assert np.linalg.norm(ys[-1, :2]) <= R_EARTH


@pytest.mark.parametrize(
    "tp, east, expected",
    [(100, True, "crash"), (175, True, "orbit"), (250, True, "falls back"), (175, False, "falls back")],
)
def test_classify_pitch_program_trajectories(tp, east, expected):
    # 2b: F/m0 = 15, m1 = 0.12 m0, theta(t) = 90 max(0, 1 - t/tp), 3 h
    thrust, m1 = 15.0, 0.12
    theta = lambda t: 90.0 * max(0.0, 1 - t / tp)
    t, ys = launch(thrust, m1, 3 * 3600.0, theta=theta, east=east)
    assert classify(t, ys, burn_time(thrust, m1)) == expected


def test_classify_escape():
    t, ys = launch(18.0, M1, 3 * 3600.0)
    assert classify(t, ys, burn_time(18.0, M1)) == "escape"


def test_fly_with_start_time_continues_a_burn():
    # theta and the mass formula use the time since ignition, so a split burn equals one burn
    thrust, theta = 15.0, lambda t: 90.0 * max(0.0, 1 - t / 175.0)
    t, ys = fly(launch_state(), 200.0, thrust=thrust, theta=theta, dt=0.05)
    t_a, ys_a = fly(launch_state(), 100.0, thrust=thrust, theta=theta, dt=0.05)
    t_b, ys_b = fly(ys_a[-1], 100.0, thrust=thrust, theta=theta, dt=0.05, t0=100.0)
    assert np.isclose(t_b[0], 100.0) and np.isclose(t_b[-1], 200.0)
    assert np.allclose(ys_b[-1], ys[-1], rtol=1e-9)


def test_fly_until_stops_exactly_at_event():
    # circular orbit from (r, 0): x = 0 after a quarter period
    r = 7000e3
    y = np.array([r, 0.0, 0.0, np.sqrt(GM / r)])
    t, ys = fly_until(y, lambda z: -z[0], 10000.0)
    assert np.isclose(t[-1], 0.5 * np.pi * np.sqrt(r**3 / GM), rtol=1e-8)
    assert abs(ys[-1, 0]) < 1e-3 * r


def test_fly_until_returns_none_if_event_not_reached():
    r = 7000e3
    y = np.array([r, 0.0, 0.0, np.sqrt(GM / r)])
    assert fly_until(y, lambda z: np.linalg.norm(z[:2]) - 2 * r, 10000.0) is None


def test_gravity_loss_is_the_speed_lost_while_coasting():
    # Without thrust d|v|/dt = a_g . v/|v|, so the loss integral equals the speed drop
    y = np.array([R_EARTH, 0.0, 3000.0, 500.0])
    t, ys = fly(y, 200.0, dt=0.1)
    speed = np.linalg.norm(ys[:, 2:], axis=1)
    assert np.isclose(gravity_loss(t, ys), speed[0] - speed[-1], rtol=1e-4)


def test_gravity_loss_vanishes_on_circular_orbit():
    # gravity perpendicular to the velocity: no speed is lost
    r = 7000e3
    t, ys = fly(np.array([r, 0.0, 0.0, np.sqrt(GM / r)]), 3000.0)
    assert abs(gravity_loss(t, ys)) < 1e-6


def test_fly_until_keeps_event_reached_before_impact():
    # climbs above 1200 m, then falls back and hits the ground: the event still counts
    y = np.array([R_EARTH + 1000.0, 0.0, 100.0, 0.0])
    result = fly_until(y, lambda z: np.linalg.norm(z[:2]) - R_EARTH - 1200.0, 600.0)
    assert result is not None
    assert np.isclose(np.linalg.norm(result[1][-1, :2]) - R_EARTH, 1200.0)
