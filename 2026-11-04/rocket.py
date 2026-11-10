"""Rocket launch from the equator, fixed Earth, no atmosphere (xy plane).

r'' = -GM r / |r|^3 + F/m(t) e while burning. Everything is divided by m0:
thrust = F/m0, and with F = mdot v_e: m(t)/m0 = 1 - thrust t / v_e.
Derivations: THEORIE.md, section 2a.
"""

import numpy as np

from orbit import GM, energy

R_EARTH = 6371e3  # m
OMEGA_EARTH = 7.292e-5  # rad/s
V_EXHAUST = 4400.0  # m/s


def vertical(t):
    """Pitch program theta(t) in degrees: straight up."""
    return 90.0


def launch_state():
    """[x, y, vx, vy] on the equator, co-rotating with the Earth: v = omega R."""
    return np.array([R_EARTH, 0.0, 0.0, OMEGA_EARTH * R_EARTH])


def burn_time(thrust, m1):
    """t_b from m(t_b)/m0 = 1 - thrust t_b / v_e = m1  ->  t_b = (1 - m1) v_e / thrust."""
    return (1 - m1) * V_EXHAUST / thrust


def thrust_direction(r, theta_deg, east=True):
    """e = sin(theta) e_r +- cos(theta) e_phi, theta measured from the horizon."""
    e_r = r / np.linalg.norm(r)
    e_phi = np.array([-e_r[1], e_r[0]])
    theta = np.radians(theta_deg)
    return np.sin(theta) * e_r + (1 if east else -1) * np.cos(theta) * e_phi


def rocket_rhs(t, y, thrust, m1, theta=vertical, east=True, burning=None):
    """y' = [v, a] with a = -GM r / |r|^3 + thrust / (1 - thrust t / v_e) e while burning."""
    r = y[:2]
    a = -GM * r / np.linalg.norm(r) ** 3
    # forced flag: RK4 stages at exactly t_b must not switch the engine off mid-step
    if burning is None:
        burning = t < burn_time(thrust, m1)
    if burning:
        a = a + thrust / (1 - thrust * t / V_EXHAUST) * thrust_direction(r, theta(t), east)
    return np.concatenate([y[2:], a])


def rk4_step(f, t, y, h):
    """Classical RK4 for y' = f(t, y)."""
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def mass_after(m, thrust, duration):
    """m (1 - thrust duration / v_e): the mass law of the header, thrust = F/m at ignition."""
    return m * (1 - thrust * duration / V_EXHAUST)


def fly(y, duration, thrust=0.0, theta=vertical, east=True, dt=None, t0=0.0):
    """One flight phase from state y at time t0 to t0 + duration; returns (t, ys).

    thrust > 0 burns the whole phase; thrust = F/m at ignition and t counts from ignition,
    so a = thrust / (1 - thrust t / v_e) and theta(t). t0 > 0 continues a burn started earlier.
    thrust == 0 coasts. n equal RK4 steps put t[-1] on t0 + duration.
    Stops after the first state with |r| <= R_EARTH.
    """
    burning = thrust > 0
    dt = dt or (0.1 if burning else 1.0)
    n = max(1, int(np.ceil(duration / dt - 1e-9)))
    f = lambda s, u: rocket_rhs(s, u, thrust, 0.0, theta, east, burning)
    t, ys = [t0], [y]
    if duration == 0:
        return np.array(t), np.array(ys)
    for i in range(n):
        ys.append(rk4_step(f, t0 + i * duration / n, ys[-1], duration / n))
        t.append(t0 + (i + 1) * duration / n)
        if np.linalg.norm(ys[-1][:2]) <= R_EARTH:
            break
    return np.array(t), np.array(ys)


def fly_until(y, event, max_duration, t0=0.0, **phase):
    """Fly like `fly` until event(state) turns from < 0 to >= 0; returns (t, ys) ending on it.

    The step that crosses is refined by bisection, restarting from the state before it
    (t0 keeps theta and mass right inside a burn). None if the event is not reached
    within max_duration or the rocket hits the ground first.
    """
    t, ys = fly(y, max_duration, t0=t0, **phase)
    hit = np.array([event(z) >= 0 for z in ys])
    if not hit.any():  # fly stops at impact, so any hit lies before it
        return None
    k = int(np.argmax(hit))
    if k == 0:
        return t[:1], ys[:1]
    lo, hi = 0.0, t[k] - t[k - 1]
    for _ in range(60):
        mid = (lo + hi) / 2
        z = fly(ys[k - 1], mid, t0=t[k - 1], dt=mid, **phase)[1][-1]
        lo, hi = (mid, hi) if event(z) < 0 else (lo, mid)
    z = fly(ys[k - 1], hi, t0=t[k - 1], dt=hi, **phase)[1][-1]
    return np.append(t[:k], t[k - 1] + hi), np.vstack([ys[:k], z])


def launch(thrust, m1, t_end, theta=vertical, east=True, dt_burn=0.1, dt_coast=1.0):
    """Integrate to t_end; returns (t, ys) with shapes (N,) and (N, 4).

    Burn phase up to t_b (t_b lies on the grid), then coast for the rest.
    Stops after the first state with |r| <= R_EARTH.
    """
    t_b = burn_time(thrust, m1)
    t, ys = fly(launch_state(), min(t_b, t_end), thrust, theta, east, dt_burn)
    if np.linalg.norm(ys[-1][:2]) > R_EARTH and t_end > t_b:
        t2, ys2 = fly(ys[-1], t_end - t_b, dt=dt_coast)
        t, ys = np.concatenate([t, t_b + t2[1:]]), np.concatenate([ys, ys2[1:]])
    return t, ys


def _conic(y):
    """Angular momentum L and eccentricity e = sqrt(1 + 2 E L^2 / GM^2) of the conic through y."""
    r, v = y[:2], y[2:]
    E = 0.5 * v @ v - GM / np.linalg.norm(r)
    L = r[0] * v[1] - r[1] * v[0]
    return L, np.sqrt(max(0.0, 1 + 2 * E * L**2 / GM**2))


def periapsis(y):
    """Closest distance to the Earth's centre on the conic through state y = [x, y, vx, vy].

    r_p = L^2 / (GM (1 + e)). Holds for ellipses and hyperbolas alike.
    """
    L, e = _conic(y)
    return L**2 / (GM * (1 + e))


def apoapsis(y):
    """Farthest distance r_a = L^2 / (GM (1 - e)); np.inf for e >= 1 (E >= 0, unbound)."""
    L, e = _conic(y)
    return L**2 / (GM * (1 - e)) if e < 1 else np.inf


def classify(t, ys, tb):
    """Fate of a launch with burnout at tb.

    "crash": hits the ground while the engine still burns. Otherwise the state at burnout decides:
    "escape" if E >= 0, "orbit" if its periapsis clears the Earth, else "falls back" (suborbital).
    """
    if np.linalg.norm(ys[-1][:2]) <= R_EARTH and t[-1] <= tb:
        return "crash"
    y = ys[np.argmin(abs(t - tb))]
    if energy(y[None, :])[0] >= 0:
        return "escape"
    return "orbit" if periapsis(y) > R_EARTH else "falls back"


def gravity_loss(t, ys):
    """Speed lost to gravity along a trajectory: -integral of a_g . v/|v| dt (trapezoid rule).

    d|v|/dt = (F/m) e . v/|v| + a_g . v/|v|; the second term is the gravity loss, g sin(gamma)
    with flight path angle gamma. Vertical flight loses the full g, horizontal flight nothing.
    """
    r, v = ys[:, :2], ys[:, 2:]
    a_g = -GM * r / np.linalg.norm(r, axis=1)[:, None] ** 3
    return -np.trapezoid(np.sum(a_g * v, axis=1) / np.linalg.norm(v, axis=1), t)
