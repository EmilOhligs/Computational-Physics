"""Demagnetizing field of a magnetization on a grid of unit cells.

Every cell is a point dipole with the kernel

    K(r) = (3 r r^T / r^2 - I) / (4 pi r^3),   K(0) = -I/3,

where the self term -1/3 is the demagnetizing factor of a cubic cell. The
field is h_i = sum_j K(r_i - r_j) m_j: a direct sum is O(N^2), a
zero-padded FFT convolution is O(N log N).
"""

from functools import lru_cache

import numpy as np
import scipy.fft


def dipole_field(m, r):
    """Field of a point dipole m (shape (3,)) at the origin, at points r (..., 3)."""
    r = np.asarray(r, dtype=float)
    dist = np.linalg.norm(r, axis=-1, keepdims=True)
    rhat = r / dist
    return (3 * (rhat @ m)[..., None] * rhat - m) / (4 * np.pi * dist**3)


def kernel_1d(x):
    """Kxx on the x-axis for integer distances x, with Kxx(0) = -1/3."""
    x = np.abs(np.asarray(x, dtype=float))
    out = np.full(x.shape, -1 / 3)
    np.divide(1, 2 * np.pi * x**3, out=out, where=x != 0)
    return out


def demag_direct_1d(m):
    """Direct O(N^2) sum over all cells of a chain."""
    j = np.arange(len(m))
    return np.array([kernel_1d(i - j) @ m for i in range(len(m))])


def demag_fft_1d(m):
    """Linear convolution via FFT, zero-padded to L = 2N."""
    n = len(m)
    length = 2 * n
    k = kernel_1d(np.fft.fftfreq(length, 1 / length))  # wraparound order
    m_pad = np.zeros(length)
    m_pad[:n] = m
    return np.fft.ifft(np.fft.fft(k) * np.fft.fft(m_pad)).real[:n]


@lru_cache(maxsize=None)
def _kernel_1d_hat(length):
    """Real FFT of the wraparound kernel of length `length` (cached)."""
    return np.fft.rfft(kernel_1d(np.fft.fftfreq(length, 1 / length)))


def demag_fft_1d_fast(m):
    """Like demag_fft_1d, with a fast FFT length, real FFTs and a cached kernel."""
    n = len(m)
    length = scipy.fft.next_fast_len(2 * n - 1, real=True)
    return np.fft.irfft(_kernel_1d_hat(length) * np.fft.rfft(m, n=length), n=length)[:n]


def _wrapped_grid(shape):
    """Cell offsets (3, 2nx, 2ny, 2nz) of the zero-padded grid in wraparound order."""
    axes = [np.fft.fftfreq(2 * n, 1 / (2 * n)) for n in shape]
    return np.array(np.meshgrid(*axes, indexing="ij"))


def kernel_3d(shape):
    """Dipole kernel K[i, j] of shape (3, 3, 2nx, 2ny, 2nz) in wraparound order."""
    r = _wrapped_grid(shape)
    r2 = (r**2).sum(axis=0)
    zero = r2 == 0
    r2 = np.where(zero, 1.0, r2)  # avoid 0/0; the origin is set below
    eye = np.eye(3)[:, :, None, None, None]
    k = (3 * r[:, None] * r[None] / r2 - eye) / (4 * np.pi * r2**1.5)
    k[:, :, zero] = -eye[:, :, 0, 0, 0][:, :, None] / 3
    return k


def demag_fft_3d(m):
    """Demagnetizing field of m (3, nx, ny, nz) by zero-padded FFT convolution."""
    shape = m.shape[1:]
    pad = tuple(2 * n for n in shape)
    k_hat = np.fft.rfftn(kernel_3d(shape), s=pad, axes=(2, 3, 4))
    m_hat = np.fft.rfftn(m, s=pad, axes=(1, 2, 3))
    h_hat = np.einsum("ij...,j...->i...", k_hat, m_hat)
    return np.fft.irfftn(h_hat, s=pad, axes=(1, 2, 3))[:, : shape[0], : shape[1], : shape[2]]


def demag_direct_3d(m):
    """Direct O(N^2) sum over all cell pairs, for validating the FFT version."""
    shape = m.shape[1:]
    pos = np.indices(shape).reshape(3, -1).T.astype(float)
    mf = m.reshape(3, -1).T
    h = np.empty_like(mf)
    for i in range(len(pos)):
        other = np.arange(len(pos)) != i
        h[i] = dipole_field_sum(mf[other], pos[i] - pos[other]) - mf[i] / 3
    return h.T.reshape(m.shape)


def dipole_field_sum(m, r):
    """Summed field at the origin of dipoles m (N, 3) at offsets r (N, 3)."""
    dist = np.linalg.norm(r, axis=1, keepdims=True)
    rhat = r / dist
    return ((3 * (rhat * m).sum(axis=1, keepdims=True) * rhat - m) / (4 * np.pi * dist**3)).sum(axis=0)


# Potential at the centre of a unit cube of charge density 1, i.e. the r = 0 value of 1/(4 pi r).
_KC0 = (3 * np.log(2 + np.sqrt(3)) - np.pi / 2) / (4 * np.pi)


def charge_field_3d(m):
    """Field from the magnetic charge rho = -div m, h = -grad phi, phi = Kc * rho."""
    shape = m.shape[1:]
    pad = tuple(2 * n for n in shape)
    rho = -sum(np.gradient(m[a], axis=a) for a in range(3))
    dist = np.sqrt((_wrapped_grid(shape) ** 2).sum(axis=0))
    kc = np.full(dist.shape, _KC0)
    np.divide(1, 4 * np.pi * dist, out=kc, where=dist != 0)
    phi = np.fft.irfftn(np.fft.rfftn(kc) * np.fft.rfftn(rho, s=pad, axes=(0, 1, 2)), s=pad, axes=(0, 1, 2))[: shape[0], : shape[1], : shape[2]]
    return -np.stack(np.gradient(phi))
