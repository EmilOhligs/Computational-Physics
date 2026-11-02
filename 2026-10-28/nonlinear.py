"""ODE solvers for nonlinear systems x' = f(x) (exercise 3, part 2)."""

import numpy as np


def euler_f(f, jac, x, dt):
    """Euler step: x + dt * f(x)."""
    return x + dt * f(x)


def heun_f(f, jac, x, dt):
    """Heun step: k1 = f(x), k2 = f(x + dt * k1), x + dt / 2 * (k1 + k2)."""
    k1 = f(x)
    k2 = f(x + dt * k1)
    return x + dt / 2 * (k1 + k2)


def rk4_f(f, jac, x, dt):
    """RK4 step with f(x) instead of J @ x."""
    k1 = f(x)
    k2 = f(x + dt / 2 * k1)
    k3 = f(x + dt / 2 * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def implicit_euler_f(f, jac, x, dt, tol=1e-12, max_iter=50):
    """Implicit Euler: solve y - x - dt * f(y) = 0 with Newton."""
    y = x.copy()
    for _ in range(max_iter):
        g = y - x - dt * f(y)
        J = np.eye(len(x)) - dt * jac(y)
        delta = np.linalg.solve(J, -g)
        y = y + delta
        if np.max(np.abs(delta)) <= tol * (1 + np.max(np.abs(y))):
            return y
    raise RuntimeError("Newton did not converge")


def integrate_grid(step, f, jac, x0, t):
    """Integrate ODE using given step function over arbitrary time grid."""
    x = np.empty((len(t), len(x0)))
    x[0] = x0
    for i in range(len(t) - 1):
        x[i + 1] = step(f, jac, x[i], t[i + 1] - t[i])
    return x


def robertson(x, k1=0.04, k2=3e7, k3=1e4):
    """Right-hand side of the Robertson chemical reaction system.

    A, B, C = x; returns the derivatives dA/dt, dB/dt, dC/dt.
    """
    A, B, C = x
    return np.array([-k1 * A + k3 * B * C, k1 * A - k2 * B**2 - k3 * B * C, k2 * B**2], dtype=float)


def robertson_jac(x, k1=0.04, k2=3e7, k3=1e4):
    """Jacobian matrix of the Robertson system.

    Partial derivatives with respect to (A, B, C).
    """
    A, B, C = x
    return np.array([
        [-k1, k3 * C, k3 * B],
        [k1, -2 * k2 * B - k3 * C, -k3 * B],
        [0, 2 * k2 * B, 0]
    ], dtype=float)


def lorenz(x, sigma=10.0, rho=28.0, beta=8 / 3):
    """Right-hand side of the Lorenz system.

    x, y, z = x; returns the derivatives dx/dt, dy/dt, dz/dt.
    """
    u, v, w = x
    return np.array([sigma * (v - u), u * (rho - w) - v, u * v - beta * w], dtype=float)


def lorenz_jac(x, sigma=10.0, rho=28.0, beta=8 / 3):
    """Jacobian matrix of the Lorenz system.

    Partial derivatives with respect to (x, y, z).
    """
    u, v, w = x
    return np.array([
        [-sigma, sigma, 0],
        [rho - w, -1, -u],
        [v, u, -beta]
    ], dtype=float)
