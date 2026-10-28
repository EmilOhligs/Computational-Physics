"""Exercise 3, 1c: error of C(t) versus dt for k1 = 1, k2 = 10 (log-log)."""

import matplotlib.pyplot as plt
import numpy as np
from ode import euler_step, heun_step, implicit_euler_step, max_error, order, rk4_step

METHODS = [("explicit Euler", euler_step, 1), ("Heun", heun_step, 2), ("RK4", rk4_step, 4), ("implicit Euler", implicit_euler_step, 1)]

DTS = 0.1 / 2.0 ** np.arange(9)

if __name__ == "__main__":
    plt.figure()
    
    for name, step, p in METHODS:
        errors = np.array([max_error(step, 1.0, 10.0, dt) for dt in DTS])
        if name == "RK4":
            p_fit = order(DTS[2:6], errors[2:6])
        else:
            p_fit = order(DTS[-4:], errors[-4:])
        print(f"{name:15s} expected p = {p}, measured p = {p_fit:.2f}")
        plt.loglog(DTS, errors, "o-", label=f"{name} (p = {p_fit:.2f})")
    
    for p in [1, 2, 4]:
        plt.loglog(DTS, DTS**p, "k:", lw=0.8)
        plt.text(DTS[0], DTS[0] ** p, f"$\\Delta t^{p}$")
    
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"$\max_t |C_\mathrm{num} - C_\mathrm{exact}|$")
    plt.title(r"Accuracy, $k_1 = 1$, $k_2 = 10$")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig("accuracy_1c.png", dpi=150, bbox_inches="tight")
    plt.show()
