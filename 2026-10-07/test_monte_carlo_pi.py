"""Tests for the Monte Carlo estimator of pi.

Run with:  python -m pytest -v

The estimator is random, so "is the result correct?" cannot be answered by
comparing against pi with a fixed tolerance picked by feeling. Instead the
tests use what we know analytically:

    each point is a hit with probability p = pi/4  (Bernoulli trial)
    pi_hat = 4 * hits / N
    E[pi_hat]   = pi
    std[pi_hat] = sqrt(pi * (4 - pi) / N)  ~  1.64 / sqrt(N)

All random tests use fixed seeds, so they are deterministic and do not fail
sporadically.
"""

import numpy as np
import pytest

from monte_carlo_pi import estimate_pi, is_inside, pi_from_hits


def sigma_theory(n):
    """Standard deviation of the estimator for n points."""
    return np.sqrt(np.pi * (4 - np.pi) / n)


# --- deterministic parts: no randomness involved ---------------------------


def test_is_inside_known_points():
    x = np.array([0.0, 0.5, 0.6, 1.0, 0.9])
    y = np.array([0.0, 0.5, 0.8, 1.0, 0.9])
    # (0.6, 0.8) lies exactly on the circle: 0.36 + 0.64 = 1 -> counted as inside
    expected = np.array([True, True, True, False, False])
    assert np.array_equal(is_inside(x, y), expected)


def test_pi_from_hits_matches_figure_on_exercise_sheet():
    # exercise sheet: N = 1500, 1188 hits -> pi = 4 * 1188 / 1500 = 3.168
    assert pi_from_hits(1188, 1500) == pytest.approx(3.168)


def test_pi_from_hits_extreme_cases():
    assert pi_from_hits(0, 10) == 0.0
    assert pi_from_hits(10, 10) == 4.0


# --- interface -------------------------------------------------------------


def test_same_seed_gives_same_result():
    assert estimate_pi(1000, seed=42) == estimate_pi(1000, seed=42)


def test_different_seeds_give_different_results():
    assert estimate_pi(1000, seed=1) != estimate_pi(1000, seed=2)


def test_result_is_multiple_of_4_over_n():
    # pi_hat = 4 * hits / N with an integer number of hits
    n = 250
    hits = estimate_pi(n, seed=0) * n / 4
    assert hits == pytest.approx(round(hits))
    assert 0 <= hits <= n


@pytest.mark.parametrize("n", [0, -5])
def test_invalid_n_raises(n):
    with pytest.raises(ValueError):
        estimate_pi(n)


# --- statistical correctness -----------------------------------------------


@pytest.mark.parametrize("n", [10**2, 10**3, 10**4, 10**5, 10**6])
def test_estimate_within_5_sigma_of_pi(n):
    # A correct estimator misses a 5 sigma interval with probability ~6e-7.
    # A wrong one (e.g. factor 2 instead of 4, or x + y < 1) is off by
    # hundreds of sigma for large n.
    error = abs(estimate_pi(n, seed=123) - np.pi)
    assert error < 5 * sigma_theory(n)


def test_estimator_is_unbiased():
    # The mean of M independent estimates has standard error sigma / sqrt(M).
    n, m = 1000, 2000
    estimates = np.array([estimate_pi(n, seed=s) for s in range(m)])
    standard_error = sigma_theory(n) / np.sqrt(m)
    assert abs(estimates.mean() - np.pi) < 5 * standard_error


@pytest.mark.parametrize("n", [10**2, 10**4])
def test_spread_matches_theory(n):
    # The RMS error over M runs must equal sigma_theory. The relative
    # statistical uncertainty of an RMS from M samples is ~1/sqrt(2M) = 3.5 %,
    # so 15 % is a tolerance of more than 4 sigma.
    m = 400
    estimates = np.array([estimate_pi(n, seed=s) for s in range(m)])
    rms_error = np.sqrt(np.mean((estimates - np.pi) ** 2))
    assert rms_error == pytest.approx(sigma_theory(n), rel=0.15)


def test_error_scales_like_one_over_sqrt_n():
    # 100 times more points -> error 10 times smaller.
    m = 400
    rms = []
    for n in (10**2, 10**4):
        # different seeds for the two n, so the runs are independent
        estimates = np.array([estimate_pi(n, seed=n + s) for s in range(m)])
        rms.append(np.sqrt(np.mean((estimates - np.pi) ** 2)))
    assert rms[0] / rms[1] == pytest.approx(10, rel=0.2)
