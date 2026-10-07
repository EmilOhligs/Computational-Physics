"""Runtime comparison: Monte Carlo versus polygon method for pi.

Question: how long does each method need to get how many digits of pi right?

    python compare_methods.py

For both methods the script measures runtime and error for increasing effort
(more points / more sides), prints the tables and plots error versus runtime.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from monte_carlo_pi import estimate_pi, sigma_theory
from polygon_pi import number_of_sides, polygon_bounds


def correct_digits(error):
    """Number of correct digits for a given absolute error: error = 10^-digits."""
    return -np.log10(error)


def mc_points_needed(error):
    """Number of Monte Carlo points for which the expected error is `error`.

    Inverse of sigma_theory:  error = sqrt(pi (4 - pi) / n)
    """
    return np.pi * (4.0 - np.pi) / error**2


def time_call(func, *args, number=1, repeats=5):
    """Runtime of one call func(*args) in seconds.

    The call is repeated `number` times in a row, because a single call can be
    too short to be measured. This is done `repeats` times and the fastest
    result is used: other programs can only make a run slower, never faster.
    """
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(number):
            func(*args)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed / number)
    return best


def format_duration(seconds):
    """Human readable duration, from nanoseconds to years."""
    units = [("ns", 1e-9), ("us", 1e-6), ("ms", 1e-3), ("s", 1.0), ("min", 60.0),
             ("h", 3600.0), ("days", 86400.0), ("years", 365.25 * 86400.0)]
    name, size = units[0]
    for next_name, next_size in units:
        if seconds >= next_size:
            name, size = next_name, next_size
    return f"{seconds / size:.3g} {name}"


def measure_monte_carlo(n_values, n_runs=20):
    """Runtime of one estimate and RMS error over n_runs estimates for every n."""
    times = np.empty(len(n_values))
    errors = np.empty(len(n_values))
    for i, n in enumerate(n_values):
        times[i] = time_call(estimate_pi, n, repeats=3)
        estimates = np.array([estimate_pi(n, seed=i * n_runs + j) for j in range(n_runs)])
        errors[i] = np.sqrt(np.mean((estimates - np.pi) ** 2))
    return times, errors


def measure_polygon(doublings_values):
    """Runtime and error of the lower bound for every number of doublings."""
    times = np.empty(len(doublings_values))
    errors = np.empty(len(doublings_values))
    for i, doublings in enumerate(doublings_values):
        # one call takes about a microsecond, so time 1000 calls in a row
        times[i] = time_call(polygon_bounds, doublings, number=1000)
        lower, upper = polygon_bounds(doublings)
        errors[i] = abs(lower - np.pi)
    return times, errors


def main():
    n_values = 10 ** np.arange(2, 8)  # 10^2 ... 10^7
    doublings_values = np.arange(0, 27)  # 6 ... 6 * 2^26 sides

    mc_times, mc_errors = measure_monte_carlo(n_values)
    poly_times, poly_errors = measure_polygon(doublings_values)

    print("Monte Carlo (error = RMS over 20 runs)")
    print(f"{'N':>12} {'error':>10} {'digits':>7} {'runtime':>10}")
    for n, err, t in zip(n_values, mc_errors, mc_times):
        print(f"{n:>12d} {err:>10.2e} {correct_digits(err):>7.1f} {format_duration(t):>10}")

    print("\nPolygon method (error of the lower bound)")
    print(f"{'sides':>12} {'error':>10} {'digits':>7} {'runtime':>10}")
    for k, err, t in zip(doublings_values, poly_errors, poly_times):
        print(f"{number_of_sides(k):>12d} {err:>10.2e} {correct_digits(err):>7.1f} "
              f"{format_duration(t):>10}")

    # Runtime needed for an error below 10^-d.
    # Polygon: measured. Monte Carlo: the number of points follows from
    # sigma_theory. Inside the measured range of N the runtime is interpolated
    # between the measurements. Beyond it (more than 3 digits) it is the
    # measured time per point at the largest N times the number of points:
    # an extrapolation, not a measurement.
    seconds_per_point = mc_times[-1] / n_values[-1]
    print("\nRuntime to reach an error below 10^-d")
    print(f"{'d':>3} {'polygon':>12} {'Monte Carlo':>14} {'MC points':>10}")
    for d in range(1, 16):
        target = 10.0**-d
        reached = np.nonzero(poly_errors < target)[0]
        poly = format_duration(poly_times[reached[0]]) if len(reached) > 0 else "not reached"
        n_needed = mc_points_needed(target)
        if n_needed <= n_values[-1]:
            mc_seconds = np.interp(n_needed, n_values, mc_times)
            note = ""
        else:
            mc_seconds = seconds_per_point * n_needed
            note = "  (extrapolated)"
        print(f"{d:>3d} {poly:>12} {format_duration(mc_seconds):>14} {n_needed:>10.1e}{note}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(mc_times, mc_errors, "o-", label="Monte Carlo, $N = 10^2 \\ldots 10^7$")
    ax.loglog(poly_times, poly_errors, "s-",
              label="polygon, $6 \\ldots 6\\cdot 2^{26}$ sides")
    # where Monte Carlo would continue with more points
    n_more = 10.0 ** np.arange(7, 13)
    ax.loglog(seconds_per_point * n_more, sigma_theory(n_more), "--", color="C0",
              label="Monte Carlo, extrapolated")
    ax.axhline(np.finfo(float).eps, color="gray", linestyle=":",
               label="machine precision")
    ax.set_xlabel("runtime in seconds")
    ax.set_ylabel(r"error $|\hat\pi - \pi|$")
    ax.set_title(r"Runtime needed for a given accuracy of $\pi$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("error_vs_runtime.png", dpi=150)
    print("\nplot saved to error_vs_runtime.png")

    # final result: the best estimate of each method
    n_max = n_values[-1]
    mc_estimate = estimate_pi(n_max, seed=0)
    lower, upper = polygon_bounds(doublings_values[-1])
    print("\nResult")
    print(f"  exact value                    pi = {np.pi:.15f}")
    print(f"  Monte Carlo, N = {n_max:.0e}         pi = {mc_estimate:.15f}"
          f"   error {abs(mc_estimate - np.pi):.1e}, runtime {format_duration(mc_times[-1])}")
    print(f"  polygon, {number_of_sides(doublings_values[-1])} sides  "
          f"    pi = {lower:.15f}"
          f"   error {abs(lower - np.pi):.1e}, runtime {format_duration(poly_times[-1])}")


if __name__ == "__main__":
    main()
