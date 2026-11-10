"""Moon orbit around the fixed Earth: r'' = -GM r / |r|^3 (xy plane).

Newton's law of gravity divided by the Moon's mass; only the product GM
appears. Derivations: THEORIE.md, section 1a.
"""

import numpy as np

G = 6.674e-11  # m^3 kg^-1 s^-2
M_EARTH = 5.972e24  # kg
GM = G * M_EARTH
R_MOON = 384_400e3  # m
DAY = 86400.0  # s
HOUR = 3600.0  # s


def circular_orbit(r):
    """Initial conditions r(0), v(0) for a circular orbit of radius r.

    Gravity must supply exactly the centripetal acceleration:
    v^2 / r = GM / r^2  ->  v = sqrt(GM / r).
    Start on the x axis with v along +y (perpendicular to r, counterclockwise).
    """
    return np.array([r, 0.0]), np.array([0.0, np.sqrt(GM / r)])


def period(r):
    """Orbital period T = 2 pi r / v = 2 pi sqrt(r^3 / GM).

    Circumference over speed with v = sqrt(GM / r): Kepler's third law for a
    circle. The Earth is fixed, so T is about 0.13 d longer than the
    sidereal month (see THEORIE.md).
    """
    return 2 * np.pi * np.sqrt(r**3 / GM)


def acceleration(r):
    """a(r) = -GM r / |r|^3: Newton's law with the Moon's mass divided out."""
    return -GM * r / np.linalg.norm(r) ** 3


def rhs(y):
    """First-order system y = [r, v]: y' = f(y) = [v, a(r)]."""
    return np.concatenate([y[2:], acceleration(y[:2])])


def jacobian(y):
    """df/dy = [[0, I], [A, 0]] with A = da/dr = -GM/|r|^3 (I - 3 r r^T / |r|^2).

    Differentiate -GM r |r|^-3: the product rule gives I |r|^-3 and r (-3 |r|^-5 r^T).
    """
    r = y[:2]
    d = np.linalg.norm(r)
    A = -GM / d**3 * (np.eye(2) - 3 * np.outer(r, r) / d**2)
    return np.block([[np.zeros((2, 2)), np.eye(2)], [A, np.zeros((2, 2))]])


def euler(y, dt):
    """Explicit Euler: y + dt f(y), first order."""
    return y + dt * rhs(y)


def heun(y, dt):
    """Heun (explicit trapezoid): y + dt/2 (f(y) + f(y + dt f(y))), second order."""
    k1 = rhs(y)
    k2 = rhs(y + dt * k1)
    return y + dt / 2 * (k1 + k2)


def rk4(y, dt):
    """Classical Runge-Kutta: y + dt/6 (k1 + 2 k2 + 2 k3 + k4), fourth order."""
    k1 = rhs(y)
    k2 = rhs(y + dt / 2 * k1)
    k3 = rhs(y + dt / 2 * k2)
    k4 = rhs(y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def implicit_euler_solve(y, dt):
    """Implicit Euler step: solve g(z) = z - y - dt f(z) = 0 by Newton.

    Newton step: (I - dt J(z)) delta = -g(z), z += delta. Start at the explicit
    Euler step z0 = y + dt f(y): it is O(dt^2) from the solution, so few
    iterations suffice. Stops when each component changes by <= 1e-14 relative
    to its block (r or v), so positions ~1e8 and velocities ~1e3 both stay
    accurate and a component crossing zero cannot block convergence.
    Returns (y_new, iterations).
    """
    z = y + dt * rhs(y)
    for iterations in range(1, 51):
        g = z - y - dt * rhs(z)
        delta = np.linalg.solve(np.eye(4) - dt * jacobian(z), -g)
        z = z + delta
        scale = np.repeat(np.abs(z).reshape(2, 2).max(axis=1), 2)
        if np.all(np.abs(delta) <= 1e-14 * scale):
            return z, iterations
    raise RuntimeError("Newton did not converge")


def implicit_euler(y, dt):
    """Implicit Euler: y_new = y + dt f(y_new), first order, A-stable."""
    return implicit_euler_solve(y, dt)[0]


def symplectic_euler(y, dt):
    """Symplectic Euler: v_new = v + dt a(r), then r_new = r + dt v_new."""
    v = y[2:] + dt * acceleration(y[:2])
    return np.concatenate([y[:2] + dt * v, v])


def velocity_verlet(y, dt):
    """Velocity Verlet: kick dt/2, drift dt, kick dt/2; second order, symplectic."""
    v_half = y[2:] + dt / 2 * acceleration(y[:2])
    r = y[:2] + dt * v_half
    return np.concatenate([r, v_half + dt / 2 * acceleration(r)])


def energy(ys):
    """E = |v|^2 / 2 - GM / |r| per unit mass, for each row of ys (n, 4)."""
    return 0.5 * np.sum(ys[:, 2:] ** 2, axis=1) - GM / np.linalg.norm(ys[:, :2], axis=1)


def angular_momentum(ys):
    """L = r x v per unit mass (z component x vy - y vx), for each row of ys (n, 4)."""
    return ys[:, 0] * ys[:, 3] - ys[:, 1] * ys[:, 2]


def integrate(step, y0, dt, n):
    """Apply step n times; returns array (n+1, 4) with y0 in row 0.

    Stops early (fewer rows) if an implicit step has no solution, e.g. implicit
    Euler after the orbit has collapsed towards the Earth.
    """
    ys = [y0]
    for _ in range(n):
        try:
            ys.append(step(ys[-1], dt))
        except RuntimeError:
            break
    return np.array(ys)


def error_at_half_period(step, n):
    """|r(T/2) - (-r_M, 0)| / r_M for the Moon with dt = T/n (n even, n/2 steps)."""
    ys = integrate(step, np.concatenate(circular_orbit(R_MOON)), period(R_MOON) / n, n // 2)
    return np.linalg.norm(ys[-1, :2] - [-R_MOON, 0.0]) / R_MOON


def order(dts, errors):
    """Slope in log-log plot of errors vs dt."""
    return np.polyfit(np.log(dts), np.log(errors), 1)[0]


if __name__ == "__main__":
    r0, v0 = circular_orbit(R_MOON)
    T = period(R_MOON)
    print(f"r(0) = {r0} m, v(0) = {v0} m/s")
    print(f"T = {T:.4e} s = {T / DAY:.3f} d  (sidereal month 27.32 d)")
