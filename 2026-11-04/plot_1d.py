"""Exercise 4, 1d: error at t = T/2 against dt = T/200 ... T/6400, log-log."""

import matplotlib.pyplot as plt
import numpy as np
from orbit import (
    DAY,
    R_MOON,
    error_at_half_period,
    euler,
    heun,
    implicit_euler,
    order,
    period,
    rk4,
    symplectic_euler,
    velocity_verlet,
)

METHODS = [euler, implicit_euler, heun, symplectic_euler, velocity_verlet, rk4]
NS = np.array([200, 400, 800, 1600, 3200, 6400])

if __name__ == "__main__":
    dts = period(R_MOON) / NS
    fig, ax = plt.subplots(figsize=(8, 6))
    for step in METHODS:
        errors = [error_at_half_period(step, n) for n in NS]
        p = order(dts, errors)
        print(f"{step.__name__}: slope = {p:.2f}, error at dt = T/6400: {errors[-1]:.2e}")
        ax.loglog(dts / 3600, errors, "o-", label=f"{step.__name__} (slope {p:.2f})")
    ax.set_xlabel("dt / h")
    ax.set_ylabel("|r(T/2) - (-r_M, 0)| / r_M")
    ax.set_title(f"Error at t = T/2, dt = T/200 ... T/6400 (T = {period(R_MOON) / DAY:.2f} d)")
    ax.legend()
    ax.grid(True, which="both", lw=0.3)
    fig.tight_layout()
    fig.savefig("convergence_1d.png", dpi=150)
