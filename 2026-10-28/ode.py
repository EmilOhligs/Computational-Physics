"""ODE solver for the reaction A -k1-> B -k2-> C.

Linear ODE x' = J x with x = (A, B, C).
"""

import numpy as np


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
