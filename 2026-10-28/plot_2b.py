"""Exercise 3, 2b: Lorenz system, two nearby trajectories with RK4."""

import matplotlib.pyplot as plt
import numpy as np
from nonlinear import integrate_grid, lorenz, lorenz_jac, rk4_f

START_A = np.array([1.0, 1.0, 1.0])
START_B = np.array([1.0, 1.0, 1.0 + 1e-8])
T_END = 40.0

if __name__ == "__main__":
    dts = [1e-3, 5e-4]
    t_1e3 = None
    a_1e3 = None
    b_1e3 = None
    dist_1e3 = None

    for dt in dts:
        t = np.linspace(0.0, T_END, int(round(T_END / dt)) + 1)
        a = integrate_grid(rk4_f, lorenz, lorenz_jac, START_A, t)
        b = integrate_grid(rk4_f, lorenz, lorenz_jac, START_B, t)
        dist = np.linalg.norm(a - b, axis=1)
        idx = np.argmax(dist > 1)
        print(f"RK4, dt = {dt}: trajectories separate (distance > 1) at t = {t[idx]:.2f}")
        if dt == 1e-3:
            t_1e3 = t
            a_1e3 = a
            b_1e3 = b
            dist_1e3 = dist

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(t_1e3, a_1e3[:, 0], label="x from (1, 1, 1)")
    ax1.plot(t_1e3, b_1e3[:, 0], "--", label="x from (1, 1, 1 + 1e-8)")
    ax1.set_ylabel("x")
    ax1.legend()
    ax2.semilogy(t_1e3, dist_1e3)
    ax2.axhline(1, color="gray", lw=0.5)
    ax2.set_ylabel("distance")
    ax2.set_xlabel("t")
    fig.suptitle("Lorenz system, sigma = 10, rho = 28, beta = 8/3, RK4 dt = 1e-3")
    fig.tight_layout()
    fig.savefig("lorenz_2b.png", dpi=150)
    plt.show()
