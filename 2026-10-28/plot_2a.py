"""Exercise 3, 2a: Robertson model (stiff chemistry) with implicit Euler and Newton."""

import matplotlib.pyplot as plt
import numpy as np
from nonlinear import euler_f, implicit_euler_f, integrate_grid, robertson, robertson_jac

X0 = np.array([1.0, 0.0, 0.0])

if __name__ == "__main__":
    # Implicit Euler on logarithmic time grid
    for n in (400, 1600):
        t = np.concatenate([[0.0], np.logspace(-6, 5, n)])
        x = integrate_grid(implicit_euler_f, robertson, robertson_jac, X0, t)
        print(f"implicit Euler, n = {n}: A, B, C at t = 1e5 = {x[-1, 0]:.5f}, {x[-1, 1]:.3e}, {x[-1, 2]:.5f}, max |A + B + C - 1| = {np.max(np.abs(x.sum(axis=1) - 1)):.1e}")

    # Plot with n = 1600
    plt.figure()
    plt.semilogx(t[1:], x[1:, 0], label="A")
    plt.semilogx(t[1:], 1e4 * x[1:, 1], label="B x 1e4")
    plt.semilogx(t[1:], x[1:, 2], label="C")
    plt.xlabel("t")
    plt.ylabel("concentration")
    plt.title("Robertson model, implicit Euler")
    plt.legend()
    plt.savefig("robertson_2a.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Explicit Euler
    dt = 1e-3
    t = np.arange(0.0, 40.0, dt)
    with np.errstate(all="ignore"):
        x = integrate_grid(euler_f, robertson, robertson_jac, X0, t)
        bad = ~np.isfinite(x).all(axis=1) | (np.abs(x) > 2).any(axis=1)
        if bad.any():
            print(f"explicit Euler, dt = {dt}: blows up at t = {t[np.argmax(bad)]:.2f}")
        else:
            print(f"explicit Euler, dt = {dt}: stable up to t = 40")
