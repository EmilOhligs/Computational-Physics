"""Tests for the polygon method (Archimedes) for pi.

Run with:  python -m pytest -v

The polygon method is deterministic, so unlike the Monte Carlo tests no
statistics is needed. The results are compared with

  * values that can be calculated by hand (hexagon),
  * the closed formulas n sin(pi/n) and n tan(pi/n) for the half perimeter of
    the inscribed and circumscribed regular n-gon of the unit circle,
  * the historical result of Archimedes for the 96-gon.

The closed formulas need pi itself, so they are fine as a reference in a test
but would be cheating inside the method.
"""

import numpy as np
import pytest

from polygon_pi import number_of_sides, polygon_bounds


def test_number_of_sides():
    assert number_of_sides(0) == 6
    assert number_of_sides(1) == 12
    assert number_of_sides(4) == 96


def test_hexagon():
    # Inscribed hexagon: 6 equilateral triangles with side 1 -> perimeter 6.
    # Circumscribed hexagon: side 2 tan(30 deg) = 2/sqrt(3) -> perimeter 4 sqrt(3).
    lower, upper = polygon_bounds(0)
    assert lower == pytest.approx(3.0)
    assert upper == pytest.approx(2 * np.sqrt(3))


@pytest.mark.parametrize("doublings", [1, 2, 5, 10, 15])
def test_matches_closed_formula(doublings):
    n = number_of_sides(doublings)
    lower, upper = polygon_bounds(doublings)
    assert lower == pytest.approx(n * np.sin(np.pi / n), rel=1e-13)
    assert upper == pytest.approx(n * np.tan(np.pi / n), rel=1e-13)


def test_archimedes_96_gon():
    # Archimedes (ca. 250 BC): 223/71 < pi < 22/7 from the 96-gon.
    # His fractions are slightly rounded outwards, so our bounds lie inside.
    lower, upper = polygon_bounds(4)
    assert 223 / 71 < lower < np.pi < upper < 22 / 7


@pytest.mark.parametrize("doublings", range(0, 21))
def test_pi_lies_between_the_bounds(doublings):
    lower, upper = polygon_bounds(doublings)
    assert lower < np.pi < upper


@pytest.mark.parametrize("doublings", range(1, 10))
def test_error_shrinks_by_factor_4_per_doubling(doublings):
    # error ~ 1/n^2, so twice as many sides -> 4 times smaller error.
    # This holds for large n only: from the hexagon to the 12-gon the factor
    # is 4.24, so the hexagon (doublings = 0) is left out.
    lower, upper = polygon_bounds(doublings)
    lower_next, upper_next = polygon_bounds(doublings + 1)
    ratio = (upper - lower) / (upper_next - lower_next)
    assert ratio == pytest.approx(4, rel=0.05)


def test_reaches_machine_precision():
    lower, upper = polygon_bounds(30)
    assert lower == pytest.approx(np.pi, abs=1e-14)
    assert upper == pytest.approx(np.pi, abs=1e-14)


def test_negative_doublings_raise():
    with pytest.raises(ValueError):
        polygon_bounds(-1)
