"""Finite differences for exercise 5. Theory: THEORIE.md."""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def grid(N):
    """Interior grid points on (0, 1) and spacing dx."""
    dx = 1 / (N + 1)
    return dx * np.arange(1, N + 1), dx


# Weights of -u'' without the 1/dx^2 factor.
STENCILS = {
    2: np.array([-1, 2, -1]),
    4: np.array([1, -16, 30, -16, 1]) / 12,
    6: np.array([-2, 27, -270, 490, -270, 27, -2]) / 180,
}


def _edge_row(j, N, order, boundary):
    """Row for grid index j (1..N) as {column: weight}; u=0 at j=0 and j=N+1."""
    row = {}
    if boundary == "reduce":  # largest stencil that fits inside the grid
        w = next(STENCILS[s] for s in (6, 4, 2) if s <= order and j - len(STENCILS[s]) // 2 >= 0
                 and j + len(STENCILS[s]) // 2 <= N + 1)
    else:
        w = STENCILS[order]
    for m, wm in zip(range(j - len(w) // 2, j + len(w) // 2 + 1), w):
        if boundary == "mirror" and (m < 0 or m > N + 1):  # odd reflection about the boundary
            m, wm = (-m if m < 0 else 2 * (N + 1) - m), -wm
        if 1 <= m <= N:
            row[m - 1] = row.get(m - 1, 0) + wm
    return row


def laplacian(N, order=2, boundary="reduce"):
    """Matrix of -d^2/dx^2 (stencil `order` 2/4/6, boundary "reduce"/"mirror") with u(0)=u(1)=0."""
    dx = grid(N)[1]
    w = STENCILS[order]
    h = len(w) // 2
    A = scipy.sparse.diags(w, range(-h, h + 1), shape=(N, N), format="lil")
    for j in sorted({*range(1, min(h, N) + 1), *range(max(N - h + 1, 1), N + 1)}):
        A[j - 1] = 0
        for c, v in _edge_row(j, N, order, boundary).items():
            A[j - 1, c] = v
    return A.tocsr() / dx**2


def solve(f, N, order=2, boundary="reduce"):
    """Solve -u'' = f with Dirichlet boundaries; returns (x, u)."""
    x = grid(N)[0]
    return x, scipy.sparse.linalg.spsolve(laplacian(N, order, boundary), f(x))


def laplacian_2d(N):
    """Matrix of -Laplace on the unit square (5-point stencil), N x N interior points, index (i_x, i_y) row-major."""
    A, I = laplacian(N), scipy.sparse.identity(N)
    return (scipy.sparse.kron(A, I) + scipy.sparse.kron(I, A)).tocsr()


def solve_2d(f, N):
    """Solve -Laplace u = f(x, y) with u = 0 on the boundary; returns (X, Y, U) as N x N arrays."""
    X, Y = np.meshgrid(grid(N)[0], grid(N)[0], indexing="ij")
    return X, Y, scipy.sparse.linalg.spsolve(laplacian_2d(N), f(X, Y).ravel()).reshape(N, N)


def solve_neumann(f, N, bc="ghost"):
    """Solve -u'' = f with u(0) = 0 and u'(1) = 0 ("first": one-sided, "ghost": ghost point); returns (x, u)."""
    dx = grid(N)[1]
    x = dx * np.arange(1, N + 2)
    A = scipy.sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(N + 1, N + 1), format="lil")
    b = np.array(f(x), dtype=float)
    A[N, N - 1], A[N, N] = -1, 1
    # first: (u_{N+1} - u_N)/dx = 0; ghost: halve the row so A stays symmetric
    b[N] = 0 if bc == "first" else b[N] / 2
    return x, scipy.sparse.linalg.spsolve(A.tocsr() / dx**2, b)


# 2: method of lines u' = J u with J = -A; same step signature as 2026-10-28/ode.py
def implicit_euler_step(J, x, dt):
    """Solve (I - dt J) x_new = x with a sparse J."""
    return scipy.sparse.linalg.spsolve(scipy.sparse.identity(len(x), format="csc") - dt * J, x)


def crank_nicolson_step(J, x, dt):
    """Trapezoidal rule: (I - dt/2 J) x_new = (I + dt/2 J) x."""
    return implicit_euler_step(J, x + dt / 2 * (J @ x), dt / 2)


def pulse_series(x, t, kmax):
    """Exact PDE solution for u(x,0) = 1 on (0.25, 0.75): sine series up to mode kmax."""
    k = np.arange(1, kmax + 1)[:, None]
    b = 2 / (k * np.pi) * (np.cos(k * np.pi / 4) - np.cos(3 * k * np.pi / 4))
    return np.sum(b * np.exp(-(k * np.pi) ** 2 * t) * np.sin(k * np.pi * np.asarray(x)), axis=0)
