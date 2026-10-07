"""Monte Carlo estimate of pi.

Draw N uniformly distributed points in the unit square [0, 1) x [0, 1).
The quarter circle x^2 + y^2 <= 1 has area pi/4, so the fraction of points
inside it approximates pi/4.

Run as a script to estimate pi for N = 10^2 ... 10^7 and plot the error:

    python monte_carlo_pi.py
"""

import matplotlib.pyplot as plt
import numpy as np


def is_inside(x, y):
    """True for every point (x, y) inside the quarter circle of radius 1."""
    return x**2 + y**2 <= 1.0


def pi_from_hits(hits, n):
    """Estimate of pi if `hits` out of `n` points are inside the quarter circle."""
    return 4.0 * hits / n


def estimate_pi(n, seed=None):
    """Monte Carlo estimate of pi from n random points.

    seed: seed of the random number generator. The same seed gives the same
          result; None gives a different result on every call.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    y = rng.random(n)
    hits = np.count_nonzero(is_inside(x, y))
    return pi_from_hits(hits, n)


def sigma_theory(n):
    """Expected standard deviation of the estimate for n points.

    Every point is a hit with probability p = pi/4, so the number of hits is
    binomially distributed with variance n p (1 - p). With pi_hat = 4 hits/n:
        var(pi_hat) = 16 p (1 - p) / n = pi (4 - pi) / n
    """
    return np.sqrt(np.pi * (4.0 - np.pi) / n)


def main():
    n_values = 10 ** np.arange(2, 8)  # 10^2 ... 10^7
    n_runs = 20  # independent repetitions per N

    # estimates[i, j] = pi_hat of run j with n_values[i] points
    estimates = np.empty((len(n_values), n_runs))
    for i, n in enumerate(n_values):
        for j in range(n_runs):
            # a different seed for every (i, j), so all runs are independent
            seed = i * n_runs + j
            estimates[i, j] = estimate_pi(n, seed=seed)
    errors = np.abs(estimates - np.pi)

    rms_error = np.sqrt(np.mean(errors**2, axis=1))

    # slope of the straight line through log(rms error) versus log(N)
    slope = np.polyfit(np.log10(n_values), np.log10(rms_error), 1)[0]

    print(f"{'N':>10} {'single run':>12} {'RMS error':>12} {'theory':>12}")
    for n, err, rms in zip(n_values, errors[:, 0], rms_error):
        print(f"{n:>10d} {err:>12.2e} {rms:>12.2e} {sigma_theory(n):>12.2e}")
    print(f"fitted exponent: error ~ N^{slope:.3f}  (expected: -0.5)")

    fig, ax = plt.subplots(figsize=(7, 5))
    for j in range(n_runs):
        label = "single runs" if j == 0 else None
        ax.loglog(n_values, errors[:, j], ".", color="lightgray", label=label)
    ax.loglog(n_values, rms_error, "o-", label=f"RMS over {n_runs} runs")
    ax.loglog(
        n_values,
        sigma_theory(n_values),
        "k--",
        label=r"theory $\sqrt{\pi(4-\pi)/N}$",
    )
    ax.set_xlabel("number of points $N$")
    ax.set_ylabel(r"error $|\hat\pi - \pi|$")
    ax.set_title(rf"Monte Carlo estimate of $\pi$, fitted slope {slope:.2f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("error_vs_N.png", dpi=150)
    print("plot saved to error_vs_N.png")

    # final result: first run with the largest N, uncertainty from the theory
    n_max = n_values[-1]
    print(f"\nResult for N = {n_max:.0e}:")
    print(f"  estimate  pi = {estimates[-1, 0]:.6f} +- {sigma_theory(n_max):.6f}")
    print(f"  exact     pi = {np.pi:.6f}")


if __name__ == "__main__":
    main()
