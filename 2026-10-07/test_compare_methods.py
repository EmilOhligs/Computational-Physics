"""Tests for the helper functions of the runtime comparison.

Run with:  python -m pytest -v
"""

import time

import numpy as np
import pytest

from compare_methods import correct_digits, mc_points_needed, time_call
from monte_carlo_pi import sigma_theory


def test_correct_digits():
    # error 10^-d  <->  d correct digits
    assert correct_digits(1e-5) == pytest.approx(5)
    assert correct_digits(1e-12) == pytest.approx(12)
    # 22/7 = 3.1428...: two decimals are right, error 1.3e-3
    assert correct_digits(abs(22 / 7 - np.pi)) == pytest.approx(2.9, abs=0.05)


def test_mc_points_needed_is_inverse_of_sigma_theory():
    for error in (1e-1, 1e-3, 1e-6):
        assert sigma_theory(mc_points_needed(error)) == pytest.approx(error)


def test_mc_two_more_digits_cost_factor_10000():
    assert mc_points_needed(1e-5) / mc_points_needed(1e-3) == pytest.approx(1e4)


def test_time_call_measures_a_known_duration():
    # sleeping 20 ms must be measured as roughly 20 ms
    seconds = time_call(time.sleep, 0.02, number=1, repeats=2)
    assert 0.02 <= seconds < 0.05


def test_time_call_returns_time_per_call():
    # 5 calls of 10 ms each: the result is the time of ONE call
    seconds = time_call(time.sleep, 0.01, number=5, repeats=1)
    assert 0.01 <= seconds < 0.03
