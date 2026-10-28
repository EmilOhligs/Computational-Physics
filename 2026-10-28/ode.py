"""ODE solver for the reaction A -k1-> B -k2-> C.

Linear ODE x' = J x with x = (A, B, C).
"""

import numpy as np

X0 = np.array([1.0, 0.0, 0.0])


def jacobian(k1, k2):
    """Jacobian matrix of the reaction system."""
    return np.array([[-k1, 0, 0], [k1, -k2, 0], [0, k2, 0]], dtype=float)


def c_exact(t, k1, k2):
    """Analytical solution for C(t)."""
    t = np.asarray(t, dtype=float)
    if k1 == k2:
        return 1 - np.exp(-k1 * t) * (1 + k1 * t)
    else:
        return 1 - (k2 * np.exp(-k1 * t) - k1 * np.exp(-k2 * t)) / (k2 - k1)


def euler_step(J, x, dt):
    """Euler step: x + dt * (J @ x)."""
    return x + dt * (J @ x)


def heun_step(J, x, dt):
    """Heun step: x + dt/2 * (k1 + k2) with k1 = J @ x, k2 = J @ (x + dt*k1)."""
    k1 = J @ x
    k2 = J @ (x + dt * k1)
    return x + dt / 2 * (k1 + k2)


def rk4_step(J, x, dt):
    """RK4 step with f(x) = J @ x."""
    k1 = J @ x
    k2 = J @ (x + dt / 2 * k1)
    k3 = J @ (x + dt / 2 * k2)
    k4 = J @ (x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def implicit_euler_step(J, x, dt):
    """Implicit Euler step: solve (I - dt*J) x_new = x."""
    return np.linalg.solve(np.eye(len(x)) - dt * J, x)


def integrate(step, J, x0, dt, t_end):
    """Integrate ODE using given step function from t=0 to t_end."""
    n = int(round(t_end / dt))
    t = np.linspace(0, n * dt, n + 1)
    x = np.empty((n + 1, len(x0)))
    x[0] = x0
    for i in range(n):
        x[i + 1] = step(J, x[i], dt)
    return t, x


def max_error(step, k1, k2, dt, t_end=10.0):
    """Max absolute error in C(t) for given step function."""
    t, x = integrate(step, jacobian(k1, k2), X0, dt, t_end)
    return np.max(np.abs(x[:, 2] - c_exact(t, k1, k2)))


def order(dts, errors):
    """Slope in log-log plot of errors vs dt."""
    return np.polyfit(np.log(dts), np.log(errors), 1)[0]


def stability_function(step, z):
    """R(z) of a step method, computed generically from the step."""
    return np.vectorize(lambda s: step(np.ones((1, 1)), np.ones(1, dtype=complex), s)[0])(z)


def critical_dt(step, k1, k2):
    """Largest dt for which |R(lambda*dt)| <= 1 for eigenvalues -k1, -k2."""
    # Check if A-stable
    if abs(stability_function(step, -1e6)) <= 1:
        return np.inf
    # Bisection on negative real axis
    lo = 0.0
    hi = 1.0
    while abs(stability_function(step, -hi)) <= 1:
        hi *= 2
    # 60 iterations of bisection
    for _ in range(60):
        mid = (lo + hi) / 2
        if abs(stability_function(step, -mid)) <= 1:
            lo = mid
        else:
            hi = mid
    return lo / max(k1, k2)


def steps_for_tolerance(step, k1, k2, tol, t_end=10.0):
    """Smallest n such that max_error with dt = t_end/n is <= tol."""
    # shortcut: assumes error falls monotonically with n, fine for this problem
    def ok(n):
        err = max_error(step, k1, k2, t_end / n, t_end)
        return not np.isnan(err) and err <= tol
    n = 1
    while not ok(n):
        n *= 2
    lo = n // 2
    hi = n
    if lo == hi:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi
