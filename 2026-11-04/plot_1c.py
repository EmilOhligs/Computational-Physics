"""Exercise 4, 1c: relative errors of energy and angular momentum over 100 orbits, dt = 6 h."""

import matplotlib.pyplot as plt
import numpy as np
from orbit import (
    HOUR,
    R_MOON,
    angular_momentum,
    circular_orbit,
    energy,
    euler,
    heun,
    implicit_euler,
    integrate,
    period,
    rk4,
    symplectic_euler,
    velocity_verlet,
)

DT = 6 * HOUR
METHODS = [euler, implicit_euler, heun, symplectic_euler, velocity_verlet, rk4]
FLOOR = 1e-17  # log axis: exact zeros (L at t = 0) would vanish

if __name__ == "__main__":
    y0 = np.concatenate(circular_orbit(R_MOON))
    T = period(R_MOON)
    n = int(round(100 * T / DT))

    fig, (ax_e, ax_l) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for step in METHODS:
        ys = integrate(step, y0, DT, n)
        t = np.arange(len(ys)) * DT / T
        e, l = energy(ys), angular_momentum(ys)
        de, dl = np.abs(e / e[0] - 1), np.abs(l / l[0] - 1)
        print(f"{step.__name__}: max |dE/E| = {de.max():.2e}, max |dL/L| = {dl.max():.2e}, steps = {len(ys) - 1}")
        ax_e.semilogy(t, de + FLOOR, lw=0.8, label=step.__name__)
        ax_l.semilogy(t, dl + FLOOR, lw=0.8, label=step.__name__)
    ax_e.set_ylabel("|E(t) / E(0) - 1|")
    ax_l.set_ylabel("|L(t) / L(0) - 1|")
    ax_l.set_xlabel("t / T")
    ax_e.legend(fontsize=8, ncol=3)
    fig.suptitle("Moon orbit, 100 orbits, dt = 6 h: relative errors of energy and angular momentum")
    fig.tight_layout()
    fig.savefig("conservation_1c.png", dpi=150)
