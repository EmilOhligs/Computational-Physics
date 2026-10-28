"""Exercise 3, 1e: eigenvalues of J, stability regions and eigenmodes."""

import matplotlib.pyplot as plt
import numpy as np
from ode import X0, critical_dt, euler_step, heun_step, implicit_euler_step, jacobian, rk4_step, stability_function


METHODS = [("explicit Euler", euler_step), ("Heun", heun_step), ("RK4", rk4_step), ("implicit Euler", implicit_euler_step)]
DTS = [0.05, 0.15, 0.25]


if __name__ == "__main__":
    k1, k2 = 1.0, 10.0
    J = jacobian(k1, k2)
    lam, vecs = np.linalg.eig(J)
    lam, vecs = lam.real, vecs.real

    print(f"lambda = {lam[0]:g}, eigenvector = {np.round(vecs[:, 0], 3)}")
    print(f"lambda = {lam[1]:g}, eigenvector = {np.round(vecs[:, 1], 3)}")
    print(f"lambda = {lam[2]:g}, eigenvector = {np.round(vecs[:, 2], 3)}")

    for i, (name, step) in enumerate(METHODS[:3]):
        dt = 1.05 * critical_dt(step, k1, k2)
        growth = np.abs(stability_function(step, lam * dt))
        print(f"{name}: first unstable mode lambda = {lam[np.argmax(growth)]:g}, |R| = {growth.max():.3f}")

    # grid avoids z = 1 exactly, where implicit Euler's 1 - z is singular
    X, Y = np.meshgrid(np.linspace(-4, 2, 240), np.linspace(-3, 3, 241))
    Z = X + 1j * Y

    plt.figure(figsize=(7, 6))
    for j, (name, step) in enumerate(METHODS):
        plt.contour(X, Y, np.abs(stability_function(step, Z)), levels=[1], colors=f"C{j}")
        plt.plot([], [], color=f"C{j}", label=name)

    markers = ["o", "s", "^"]
    for i, dt in enumerate(DTS):
        plt.plot(lam * dt, np.zeros(3), marker=markers[i], color="k", ls="none", label=f"z = lambda dt, dt = {dt}")

    plt.axhline(0, color="gray", lw=0.5)
    plt.axvline(0, color="gray", lw=0.5)
    plt.xlabel("Re z")
    plt.ylabel("Im z")
    plt.title("Stability regions |R(z)| <= 1")
    plt.legend(fontsize=8)
    plt.gca().set_aspect("equal")
    plt.savefig("stability_1e.png", dpi=150, bbox_inches="tight")

    coef = np.linalg.solve(vecs, X0)
    t = np.linspace(0, 5, 500)

    plt.figure()
    for i in range(3):
        mode = coef[i] * np.exp(lam[i] * t)[:, None] * vecs[:, i]
        plt.plot(t, mode[:, 2], label=f"lambda = {lam[i]:g}")

    c_sum = sum(coef[i] * np.exp(lam[i] * t) * vecs[2, i] for i in range(3))
    plt.plot(t, c_sum, "k--", label="sum = C(t)")

    plt.xlabel("t")
    plt.ylabel("contribution to C")
    plt.title("Eigenmodes, k1 = 1, k2 = 10")
    plt.legend()
    plt.savefig("modes_1e.png", dpi=150, bbox_inches="tight")
    plt.show()
