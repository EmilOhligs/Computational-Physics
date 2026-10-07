"""Polygon method of Archimedes for pi.

A regular n-gon inscribed in the unit circle has a shorter perimeter than the
circle, a circumscribed one a longer perimeter. With half perimeters

    lower = n sin(pi/n)      (inscribed)
    upper = n tan(pi/n)      (circumscribed)

this gives  lower < pi < upper.

sin and tan of pi/n cannot be used to compute pi (they need pi). Instead start
with the hexagon, where both values are known from elementary geometry, and
double the number of sides. From the half-angle formulas follows

    upper_2n = 2 upper_n lower_n / (upper_n + lower_n)    (harmonic mean)
    lower_2n = sqrt(upper_2n lower_n)                     (geometric mean)

so only +, *, / and sqrt are needed.
"""

import math


def number_of_sides(doublings):
    """Number of sides after doubling the hexagon `doublings` times."""
    return 6 * 2**doublings


def polygon_bounds(doublings):
    """Lower and upper bound for pi from the polygon with 6 * 2^doublings sides.

    Returns (lower, upper) with lower < pi < upper.
    """
    if doublings < 0:
        raise ValueError("doublings must not be negative")
    # hexagon: inscribed perimeter 6, circumscribed perimeter 4 sqrt(3)
    lower = 3.0
    upper = 2.0 * math.sqrt(3.0)
    for _ in range(doublings):
        upper = 2.0 * upper * lower / (upper + lower)
        lower = math.sqrt(upper * lower)
    return lower, upper
