"""Summary figure with all results: both methods, their error and their runtime.

    python make_results_plot.py

Writes results.png with four panels:
  (a) the Monte Carlo method      (b) the polygon method
  (c) error versus effort         (d) error versus runtime
"""

import matplotlib.pyplot as plt
import numpy as np

from compare_methods import correct_digits, format_duration, time_call
from monte_carlo_pi import estimate_pi, is_inside, pi_from_hits, sigma_theory
from polygon_pi import number_of_sides, polygon_bounds


def plot_monte_carlo_method(ax, n=1500, seed=0):
    """Panel (a): random points in the unit square, coloured by inside/outside."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    y = rng.random(n)
    inside = is_inside(x, y)
    hits = np.count_nonzero(inside)
    ax.plot(x[inside], y[inside], ".", color="C0", markersize=3)
    ax.plot(x[~inside], y[~inside], ".", color="C3", markersize=3)
    angle = np.linspace(0, np.pi / 2, 200)
    ax.plot(np.cos(angle), np.sin(angle), "k-")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"(a) Monte Carlo: $N$ = {n}, "
                 rf"$\hat\pi$ = 4$\cdot${hits}/{n} = {pi_from_hits(hits, n):.3f}")


def polygon_corners(n, radius):
    """Corners of a regular n-gon (closed line) for drawing."""
    angle = 2 * np.pi * np.arange(n + 1) / n
    return radius * np.cos(angle), radius * np.sin(angle)


def plot_polygon_method(ax, doublings=1):
    """Panel (b): unit circle between inscribed and circumscribed polygon."""
    n = number_of_sides(doublings)
    lower, upper = polygon_bounds(doublings)
    angle = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(angle), np.sin(angle), "k-", label="circle")
    # the inscribed polygon has its corners on the circle, the circumscribed
    # one touches the circle with the middle of its sides
    ax.plot(*polygon_corners(n, 1.0), "-", color="C0",
            label=f"inscribed: {lower:.4f}")
    ax.plot(*polygon_corners(n, 1.0 / np.cos(np.pi / n)), "-", color="C3",
            label=f"circumscribed: {upper:.4f}")
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.legend(loc="center", title="half perimeter")
    ax.set_title(f"(b) Polygon method: {n} sides, "
                 rf"{lower:.3f} < $\pi$ < {upper:.3f}")


def main():
    n_values = 10 ** np.arange(2, 8)  # Monte Carlo: 10^2 ... 10^7 points
    n_runs = 20
    doublings_values = np.arange(0, 27)  # polygon: 6 ... 6 * 2^26 sides

    # --- measurements -------------------------------------------------------
    mc_estimates = np.empty((len(n_values), n_runs))
    mc_times = np.empty(len(n_values))
    for i, n in enumerate(n_values):
        mc_times[i] = time_call(estimate_pi, n, repeats=3)
        for j in range(n_runs):
            mc_estimates[i, j] = estimate_pi(n, seed=i * n_runs + j)
    mc_errors = np.abs(mc_estimates - np.pi)
    mc_rms = np.sqrt(np.mean(mc_errors**2, axis=1))
    mc_slope = np.polyfit(np.log10(n_values), np.log10(mc_rms), 1)[0]

    sides = np.array([number_of_sides(k) for k in doublings_values])
    poly_times = np.empty(len(doublings_values))
    poly_errors = np.empty(len(doublings_values))
    for i, doublings in enumerate(doublings_values):
        poly_times[i] = time_call(polygon_bounds, doublings, number=1000)
        lower, upper = polygon_bounds(doublings)
        poly_errors[i] = abs(lower - np.pi)
    # slope before rounding errors matter (up to 6 * 2^15 sides)
    poly_slope = np.polyfit(np.log10(sides[:16]), np.log10(poly_errors[:16]), 1)[0]

    seconds_per_point = mc_times[-1] / n_values[-1]
    eps = np.finfo(float).eps

    # --- figure -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    plot_monte_carlo_method(axes[0, 0])
    plot_polygon_method(axes[0, 1])

    # (c) error versus effort
    ax = axes[1, 0]
    for j in range(n_runs):
        label = "Monte Carlo, single runs" if j == 0 else None
        ax.loglog(n_values, mc_errors[:, j], ".", color="lightgray", label=label)
    ax.loglog(n_values, mc_rms, "o-", color="C0",
              label=f"Monte Carlo, RMS of {n_runs} runs: slope {mc_slope:.2f}")
    ax.loglog(n_values, sigma_theory(n_values), "k--",
              label=r"theory $\sqrt{\pi(4-\pi)/N}$")
    ax.loglog(sides, poly_errors, "s-", color="C1",
              label=f"polygon: slope {poly_slope:.2f}")
    ax.axhline(eps, color="gray", linestyle=":", label="machine precision")
    ax.set_xlabel("effort: number of points $N$ / number of sides $n$")
    ax.set_ylabel(r"error $|\hat\pi - \pi|$")
    ax.set_title("(c) Error versus effort")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0.02, 0.07))

    # (d) error versus runtime
    ax = axes[1, 1]
    ax.loglog(mc_times, mc_rms, "o-", color="C0", label="Monte Carlo, measured")
    n_more = 10.0 ** np.arange(7, 13)
    ax.loglog(seconds_per_point * n_more, sigma_theory(n_more), "--", color="C0",
              label="Monte Carlo, extrapolated")
    ax.loglog(poly_times, poly_errors, "s-", color="C1", label="polygon, measured")
    ax.axhline(eps, color="gray", linestyle=":", label="machine precision")
    ax.set_xlabel("runtime in seconds")
    ax.set_ylabel(r"error $|\hat\pi - \pi|$   ($10^{-d}$ = $d$ correct digits)")
    ax.set_title("(d) Error versus runtime")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(r"Two ways to compute $\pi$: Monte Carlo and polygon method",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig("results.png", dpi=150)

    # --- numbers for the results document -----------------------------------
    print("Monte Carlo")
    print(f"{'N':>10} {'RMS error':>10} {'theory':>10} {'digits':>7} {'runtime':>10}")
    for n, rms, t in zip(n_values, mc_rms, mc_times):
        print(f"{n:>10d} {rms:>10.2e} {sigma_theory(n):>10.2e} "
              f"{correct_digits(rms):>7.1f} {format_duration(t):>10}")
    print(f"slope {mc_slope:.3f}, time per point {seconds_per_point * 1e9:.1f} ns")

    print("\nPolygon")
    print(f"{'sides':>10} {'error':>10} {'digits':>7} {'runtime':>10}")
    for n, err, t in zip(sides, poly_errors, poly_times):
        print(f"{n:>10d} {err:>10.2e} {correct_digits(err):>7.1f} {format_duration(t):>10}")
    print(f"slope {poly_slope:.3f}")

    lower, upper = polygon_bounds(doublings_values[-1])
    print("\nResult")
    print(f"  exact        pi = {np.pi:.15f}")
    print(f"  Monte Carlo  pi = {mc_estimates[-1, 0]:.15f}   (N = {n_values[-1]:.0e})")
    print(f"  polygon      pi = {lower:.15f}   ({sides[-1]} sides)")
    print("\nplot saved to results.png")


if __name__ == "__main__":
    main()
