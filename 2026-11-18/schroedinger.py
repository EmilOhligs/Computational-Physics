"""Schroedinger equation for exercise 6 (hbar = m = 1). Theory: theory.md."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq


def grid(a, b, n):
    """n interior points of (a, b); psi = 0 at the walls a and b."""
    dx = (b - a) / (n + 1)
    return a + dx * np.arange(1, n + 1), dx


def hamiltonian(V, dx):
    """Sparse H = -1/2 d^2/dx^2 + V (three-point stencil); V may be complex."""
    V = np.asarray(V)
    n = len(V)
    off = np.full(n - 1, -0.5 / dx**2)
    return sp.diags([off, 1 / dx**2 + V, off], [-1, 0, 1], format="csr")


def eigenstates(V, dx, k):
    """k lowest eigenvalues and eigenvectors (columns, sum|psi|^2 dx = 1) for real V."""
    E, psi = eigh_tridiagonal(1 / dx**2 + V, np.full(len(V) - 1, -0.5 / dx**2),
                              select="i", select_range=(0, k - 1))
    return E, psi / np.sqrt(dx)


def well_energies(V0, a):
    """Bound energies of the finite well V = -V0 for |x| < a, else 0."""
    z0 = a * np.sqrt(2 * V0)
    s = lambda z: np.sqrt(max(z0**2 - z**2, 0.0))
    # half-period n: even states (z sin z = s cos z) for n even, odd states (z cos z = -s sin z) for n odd
    even = lambda z: z * np.sin(z) - s(z) * np.cos(z)
    odd = lambda z: z * np.cos(z) + s(z) * np.sin(z)
    zs = []
    for n in range(int(2 * z0 / np.pi) + 1):
        lo, hi = n * np.pi / 2 + 1e-12, min((n + 1) * np.pi / 2 - 1e-12, z0)
        f = even if n % 2 == 0 else odd
        if lo < hi and f(lo) * f(hi) < 0:
            zs.append(brentq(f, lo, hi))
    return np.array(zs) ** 2 / (2 * a**2) - V0


def gaussian(x, x0, sigma, k0):
    """Normalized wave packet; sigma is the standard deviation of |psi|^2."""
    return (2 * np.pi * sigma**2) ** -0.25 * np.exp(-((x - x0) ** 2) / (4 * sigma**2) + 1j * k0 * x)


def moments(x, psi, dx):
    """Norm, mean and standard deviation of x in |psi|^2 / norm^2."""
    p = np.abs(psi) ** 2 * dx
    norm2 = p.sum()
    mean = (x * p).sum() / norm2
    return np.sqrt(norm2), mean, np.sqrt((x**2 * p).sum() / norm2 - mean**2)


def free_packet(t, x0, sigma, k0):
    """Analytic (mean, std) of a free Gaussian packet."""
    return float(x0 + k0 * t), float(sigma * np.sqrt(1 + (t / (2 * sigma**2)) ** 2))


def stepper(H, dt, method):
    """Return step(psi) -> psi_new for i dpsi/dt = H psi."""
    I = sp.identity(H.shape[0], dtype=complex, format="csc")
    H = H.astype(complex)
    if method == "explicit":
        return lambda psi: psi - 1j * dt * (H @ psi)
    if method == "implicit":
        lu = spla.splu((I + 1j * dt * H).tocsc(), permc_spec="MMD_AT_PLUS_A")
        return lu.solve
    if method == "cn":
        lu = spla.splu((I + 0.5j * dt * H).tocsc(), permc_spec="MMD_AT_PLUS_A")
        rhs = I - 0.5j * dt * H
        return lambda psi: lu.solve(rhs @ psi)
    raise ValueError(f"unknown method {method!r}; use 'explicit', 'implicit' or 'cn'")


def transmission(E, V0, b):
    """Transmission through a rectangular barrier of height V0 and width b."""
    E = np.asarray(E, dtype=float)
    kappa = np.sqrt(np.maximum(2 * (V0 - E), 0))
    kp = np.sqrt(np.maximum(2 * (E - V0), 0))
    s2 = np.where(E < V0, np.sinh(kappa * b) ** 2, np.sin(kp * b) ** 2)
    den = np.where(E == V0, 1.0, 4 * E * np.abs(V0 - E))
    return np.where(E == V0, 1 / (1 + V0 * b**2 / 2), 1 / (1 + V0**2 * s2 / den))


def absorber(x, a, b, width, strength):
    """W(x) >= 0, quadratic ramp within `width` of the walls; use V - 1j*W."""
    d = np.minimum(x - a, b - x)
    return np.where(d < width, strength * ((width - d) / width) ** 2, 0.0)


def hamiltonian_2d(x, y, V, A=None):
    """2D H on psi[i, j] at (x[i], y[j]), raveled row-wise; optional Peierls phases for A(X, Y)."""
    nx, ny = len(x), len(y)
    dx = x[1] - x[0]
    if not np.isclose(y[1] - y[0], dx, rtol=1e-12):
        raise ValueError(f"x and y need the same spacing, got {dx} and {y[1] - y[0]}")
    X, Y = np.meshgrid(x, y, indexing="ij")
    idx = np.arange(nx * ny).reshape(nx, ny)
    rows, cols, vals = [], [], []
    # x bonds (i, j)-(i+1, j), then y bonds (i, j)-(i, j+1), A evaluated at the bond midpoint
    for lo, hi, mx, my, comp in (
        (idx[:-1], idx[1:], (X[:-1] + X[1:]) / 2, Y[:-1], 0),
        (idx[:, :-1], idx[:, 1:], X[:, :-1], (Y[:, :-1] + Y[:, 1:]) / 2, 1),
    ):
        t = np.full(lo.shape, -0.5 / dx**2, dtype=complex if A else float)
        if A:
            t = t * np.exp(-1j * A(mx, my)[comp] * dx)
        rows += [lo.ravel(), hi.ravel()]
        cols += [hi.ravel(), lo.ravel()]
        vals += [t.ravel(), t.conj().ravel()]
    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(2 / dx**2 + np.asarray(V).ravel())
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(nx * ny,) * 2).tocsr()


def solenoid(B0, r0, center):
    """Vector potential A(X, Y) of an ideal solenoid (Coulomb gauge), flux B0 pi r0^2."""
    def A(X, Y):
        dx, dy = X - center[0], Y - center[1]
        r = np.hypot(dx, dy)
        r_safe = np.where(r == 0, 1.0, r)
        a = np.where(r < r0, B0 * r / 2, B0 * r0**2 / (2 * r_safe))
        return -a * dy / r_safe, a * dx / r_safe
    return A
