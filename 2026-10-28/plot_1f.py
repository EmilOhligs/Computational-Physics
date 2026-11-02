"""Exercise 3, 1f: stiffness, explicit versus implicit Euler for k2 = 1000."""

import matplotlib.pyplot as plt
from ode import X0, c_exact, critical_dt, euler_step, implicit_euler_step, integrate, jacobian, steps_for_tolerance

if __name__ == "__main__":
    K1, K2, TOL = 1.0, 1000.0, 1e-2

    print(f"dt_crit explicit Euler = {critical_dt(euler_step, K1, K2):.5f}")

    for name, step in [("explicit Euler", euler_step), ("implicit Euler", implicit_euler_step)]:
        n = steps_for_tolerance(step, K1, K2, TOL)
        print(f"{name}: n = {n}, dt = {10.0 / n:.5f}")
        t, x = integrate(step, jacobian(K1, K2), X0, 10.0 / n, 10.0)
        plt.plot(t, x[:, 2], label=f"{name}, n = {n}")

    plt.plot(t, c_exact(t, K1, K2), "k--", label="exact")

    plt.xscale("log")
    plt.xlabel("t")
    plt.ylabel("C")
    plt.title("Stiff case k2 = 1000, error about 1e-2")
    plt.legend()
    plt.savefig("stiff_1f.png", dpi=150, bbox_inches="tight")
    plt.show()
