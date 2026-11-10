"""Exercise 4, 1b: 10 Moon orbits with dt = 6 h for six ODE methods."""

import matplotlib.pyplot as plt
import numpy as np
from orbit import (
    HOUR,
    R_MOON,
    circular_orbit,
    euler,
    heun,
    implicit_euler,
    implicit_euler_solve,
    integrate,
    period,
    rk4,
    symplectic_euler,
    velocity_verlet,
)

DT = 6 * HOUR
METHODS = [euler, implicit_euler, heun, symplectic_euler, velocity_verlet, rk4]

if __name__ == "__main__":
    y0 = np.concatenate(circular_orbit(R_MOON))
    n = int(round(10 * period(R_MOON) / DT))
    print(f"10 orbits, dt = 6 h: {n} steps")

    # Newton iterations for the first steps of implicit Euler (guess: explicit Euler step)
    y = y0
    for i in range(3):
        y, iterations = implicit_euler_solve(y, DT)
        print(f"implicit Euler step {i + 1}: Newton iterations per step: {iterations}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, step in zip(axes.flat, METHODS):
        ys = integrate(step, y0, DT, n) / R_MOON
        r_end = np.linalg.norm(ys[-1, :2])
        note = "" if len(ys) == n + 1 else f" (Newton fails after {len(ys) - 1} steps)"
        print(f"{step.__name__}: r_end / r_M = {r_end:.6f}{note}")
        ax.plot(ys[:, 0], ys[:, 1], lw=0.7)
        ax.plot(np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200)), "k--", lw=0.5)
        ax.plot(0, 0, "o", color="tab:blue", ms=4)
        ax.set_title(f"{step.__name__}, r_end = {r_end:.3f} r_M")
        ax.set_aspect("equal")
        ax.set_xlabel("x / r_M")
        ax.set_ylabel("y / r_M")
    fig.suptitle("Moon orbit, 10 orbits, dt = 6 h (dashed: exact circle)")
    fig.tight_layout()
    fig.savefig("orbits_1b.png", dpi=150)
