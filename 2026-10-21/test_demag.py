"""Tests for the demagnetizing field via FFT convolution.

Run with:  python -m pytest -v

Cells have unit size. The point-dipole kernel is

    K(r) = (3 r r^T / r^2 - I) / (4 pi r^3),   K(0) = -I/3,

where the self term -1/3 is the demagnetizing factor of one cubic cell.
On the x-axis this gives Kxx(x) = 1 / (2 pi |x|^3).

Strategy: the direct sum is checked by hand on tiny cases, then every FFT
variant must reproduce the direct sum to round-off. The charge formulation
is checked against the dipole formulation on a smooth magnetization.
"""

import numpy as np
import pytest

from demag import (
    charge_field_3d,
    demag_direct_1d,
    demag_direct_3d,
    demag_fft_1d,
    demag_fft_1d_fast,
    demag_fft_3d,
    dipole_field,
    kernel_1d,
)


# --- point dipole and 1D kernel --------------------------------------------

def test_dipole_field_on_axis_and_side():
    m = np.array([1.0, 0.0, 0.0])
    # on the axis: h = 2 m / (4 pi r^3)
    assert np.allclose(dipole_field(m, np.array([2.0, 0, 0])), [1 / (16 * np.pi), 0, 0])
    # beside the dipole: h = -m / (4 pi r^3)
    assert np.allclose(dipole_field(m, np.array([0, 2.0, 0])), [-1 / (32 * np.pi), 0, 0])


def test_dipole_field_many_points():
    m = np.array([0.0, 0.0, 1.0])
    r = np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0]])
    h = dipole_field(m, r)
    assert h.shape == (3, 3)
    assert np.allclose(h[:, 2], [1 / (2 * np.pi), 1 / (2 * np.pi), -1 / (4 * np.pi)])


def test_kernel_1d_values():
    assert np.allclose(kernel_1d(np.array([1, -1, 2, -3])),
                       [1 / (2 * np.pi), 1 / (2 * np.pi), 1 / (16 * np.pi), 1 / (54 * np.pi)])
    assert kernel_1d(np.array([0]))[0] == pytest.approx(-1 / 3)


def test_kernel_1d_matches_dipole_field():
    x = np.array([1.0, 3.0, -5.0])
    r = np.stack([x, 0 * x, 0 * x], axis=-1)
    hx = dipole_field(np.array([1.0, 0, 0]), r)[:, 0]
    assert np.allclose(kernel_1d(x), hx)


# --- 1D direct summation ----------------------------------------------------

def test_direct_1d_single_dipole_gives_kernel():
    m = np.array([0, 0, 1.0, 0, 0])
    assert np.allclose(demag_direct_1d(m), kernel_1d(np.arange(5) - 2))


def test_direct_1d_by_hand():
    k1, k2 = 1 / (2 * np.pi), 1 / (16 * np.pi)
    expected = [-1 / 3 + k1 + k2, -1 / 3 + 2 * k1, -1 / 3 + k1 + k2]
    assert np.allclose(demag_direct_1d(np.ones(3)), expected)


# --- 1D FFT convolution -----------------------------------------------------

@pytest.mark.parametrize("fft", [demag_fft_1d, demag_fft_1d_fast])
@pytest.mark.parametrize("n", [1, 2, 37, 64, 101])
def test_fft_1d_equals_direct(fft, n):
    m = np.random.default_rng(n).standard_normal(n)
    h = fft(m)
    assert h.shape == (n,)
    assert np.isrealobj(h)
    assert np.allclose(h, demag_direct_1d(m), rtol=0, atol=1e-13)


@pytest.mark.parametrize("fft", [demag_fft_1d, demag_fft_1d_fast])
def test_fft_1d_is_linear_not_circular(fft):
    # one dipole at the left end: a circular convolution would put its
    # near field (1/2pi) onto the right end as well
    m = np.zeros(8)
    m[0] = 1.0
    assert fft(m)[-1] == pytest.approx(kernel_1d(np.array([7]))[0], abs=1e-15)


# --- 3D ---------------------------------------------------------------------

def test_direct_3d_single_dipole():
    n = 5
    m = np.zeros((3, n, n, n))
    m[:, 2, 2, 2] = [0.3, -0.5, 0.8]
    h = demag_direct_3d(m)
    idx = np.indices((n, n, n)).reshape(3, -1).T
    r = (idx - 2).astype(float)
    away = np.any(r != 0, axis=1)
    expected = dipole_field(m[:, 2, 2, 2], r[away])
    assert np.allclose(h.reshape(3, -1).T[away], expected)
    assert np.allclose(h[:, 2, 2, 2], -m[:, 2, 2, 2] / 3)


def test_direct_3d_line_reproduces_1d():
    m = np.zeros((3, 7, 1, 1))
    m[0, 1:5, 0, 0] = 1.0
    assert np.allclose(demag_direct_3d(m)[0, :, 0, 0], demag_direct_1d(m[0, :, 0, 0]))


@pytest.mark.parametrize("shape", [(4, 5, 3), (6, 6, 6), (1, 1, 9)])
def test_fft_3d_equals_direct(shape):
    m = np.random.default_rng(0).standard_normal((3, *shape))
    h = demag_fft_3d(m)
    assert h.shape == m.shape
    assert np.isrealobj(h)
    assert np.allclose(h, demag_direct_3d(m), rtol=0, atol=1e-13)


# --- magnetic charge formulation ---------------------------------------------

def test_charge_field_zero_magnetization():
    assert np.allclose(charge_field_3d(np.zeros((3, 6, 6, 6))), 0)


def charge_vs_dipole_error(sigma):
    """Relative max difference of both formulations for a Gaussian blob of width sigma."""
    n = int(round(32 * sigma / 3))
    x = np.arange(n) - (n - 1) / 2
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    blob = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))
    m = np.stack([blob, 0.5 * blob, 0 * blob])
    h_dip = demag_fft_3d(m)
    h_chg = charge_field_3d(m)
    assert h_chg.shape == m.shape
    return np.max(np.abs(h_chg - h_dip)) / np.max(np.abs(h_dip))


def test_charge_field_converges_to_dipole_field_for_smooth_m():
    # both formulations approximate the same continuum field; the central
    # differences for div and grad make the difference second order in 1/sigma
    e3, e6 = charge_vs_dipole_error(3.0), charge_vs_dipole_error(6.0)
    assert e6 < 0.03
    assert e3 / e6 > 3.5  # 4 for exact second order
