"""Exercise 3, 1d: critical time step and the solution just above it."""

import matplotlib.pyplot as plt
from ode import X0, c_exact, critical_dt, euler_step, heun_step, integrate, jacobian, max_error, rk4_step


METHODS = [("explicit Euler", euler_step), ("Heun", heun_step), ("RK4", rk4_step)]
RATES = [(1.0, 1.0), (1.0, 10.0), (1.0, 100.0), (10.0, 1.0), (50.0, 10.0)]


if __name__ == "__main__":
    # Part 1: critical dt for each rate and method
    for k1, k2 in RATES:
        for name, step in METHODS:
            dt = critical_dt(step, k1, k2)
            print(f"k1 = {k1:g}, k2 = {k2:g}, {name}: dt_crit = {dt:.5f}, dt_crit * max(k1, k2) = {dt * max(k1, k2):.3f}")

    # Part 2: max error at 0.95 and 1.05 dt_crit for k1=1, k2=10
    for name, step in METHODS:
        dt = critical_dt(step, 1.0, 10.0)
        print(f"{name}: max error at 0.95 dt_crit = {max_error(step, 1.0, 10.0, 0.95 * dt):.3g}, at 1.05 dt_crit = {max_error(step, 1.0, 10.0, 1.05 * dt):.3g}")

    # Part 3: Plot
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for i, (name, step) in enumerate(METHODS):
        dt = critical_dt(step, 1.0, 10.0)
        t, x = integrate(step, jacobian(1.0, 10.0), X0, 1.05 * dt, 10.0)

        ax = axes[i]
        for j, lab in enumerate("ABC"):
            ax.plot(t, x[:, j], "o-", ms=3, label=lab)
        ax.plot(t, c_exact(t, 1.0, 10.0), "k--", label="C exact")
        ax.set_title(f"{name}, dt = 1.05 dt_crit")
        ax.set_xlabel("t")
        ax.legend()

    axes[0].set_ylabel("concentration")
    fig.tight_layout()
    fig.savefig("instability_1d.png", dpi=150)
    plt.show()
