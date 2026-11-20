"""Tests for exercise 5, part 1a.

What each test checks: THEORIE.md, section 1a.
"""
import numpy as np
import scipy.sparse as sp

from fd import grid, laplacian, solve


def test_grid_three_points_have_spacing_quarter():
    x, dx = grid(3)
    assert dx == 0.25
    assert np.allclose(x, [0.25, 0.5, 0.75])


def test_grid_has_n_interior_points_without_boundaries():
    x, _ = grid(10)
    assert len(x) == 10
    assert x.min() > 0 and x.max() < 1


def test_laplacian_is_sparse_square_and_symmetric():
    A = laplacian(8)
    assert sp.issparse(A)
    assert A.shape == (8, 8)
    assert abs(A - A.T).max() == 0


def test_laplacian_n3_is_scaled_tridiagonal_stencil():
    _, dx = grid(3)
    expected = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
    assert np.array_equal(laplacian(3).toarray() * dx**2, expected)


def test_laplacian_row_sums_vanish_inside_and_equal_boundary_value_at_ends():
    # Dirichlet ends drop one neighbour, so only the first/last row keep a rest
    _, dx = grid(10)
    sums = np.asarray(laplacian(10).sum(axis=1)).ravel()
    assert np.allclose(sums[1:-1], 0, atol=1e-9 / dx**2)
    assert np.allclose([sums[0], sums[-1]], 1 / dx**2)


def test_laplacian_is_exact_for_quadratic_with_zero_boundary():
    # u'''' = 0, so the 3-point stencil has no truncation error here
    x, _ = grid(20)
    assert np.allclose(laplacian(20) @ (x * (1 - x)), 2, rtol=1e-10)


def test_laplacian_eigenvalues_match_analytic_formula():
    N = 10
    _, dx = grid(N)
    k = np.arange(1, N + 1)
    expected = 4 / dx**2 * np.sin(k * np.pi * dx / 2) ** 2
    got = np.sort(np.linalg.eigvalsh(laplacian(N).toarray()))
    assert np.allclose(got, expected, rtol=1e-12)


def test_solve_sine_source_gives_exact_discrete_eigenvector_solution():
    # sin(pi x_j) is an eigenvector of A, so u = pi^2/lambda_1 * sin(pi x)
    N = 15
    x, u = solve(lambda x: np.pi**2 * np.sin(np.pi * x), N)
    dx = 1 / (N + 1)
    lam1 = 4 / dx**2 * np.sin(np.pi * dx / 2) ** 2
    assert np.allclose(u, np.pi**2 / lam1 * np.sin(np.pi * x), rtol=1e-12)


def test_solve_sine_source_is_close_to_continuous_solution():
    x, u = solve(lambda x: np.pi**2 * np.sin(np.pi * x), 15)
    assert np.max(np.abs(u - np.sin(np.pi * x))) < 1e-2


# 1c: higher-order stencils
import pytest  # noqa: E402


def _rows(N, order, boundary="reduce"):
    _, dx = grid(N)
    scale = {2: 1, 4: 12, 6: 180}[order] * dx**2
    return laplacian(N, order, boundary).toarray() * scale


def test_order4_interior_row_is_five_point_stencil():
    assert np.allclose(_rows(9, 4)[4], [0, 0, 1, -16, 30, -16, 1, 0, 0], atol=1e-12)


def test_order6_interior_row_is_seven_point_stencil():
    expected = [0, 0, -2, 27, -270, 490, -270, 27, -2, 0, 0]
    assert np.allclose(_rows(11, 6)[5], expected, atol=1e-12)


def test_reduce_boundary_rows_of_order4_use_three_point_stencil():
    _, dx = grid(9)
    A, A2 = laplacian(9, 4).toarray(), laplacian(9).toarray()
    assert np.allclose(A[[0, -1]] * dx**2, A2[[0, -1]] * dx**2, atol=1e-12)


def test_reduce_second_row_of_order6_uses_five_point_stencil():
    expected = [-16, 30, -16, 1, 0, 0, 0, 0, 0, 0, 0]
    A = laplacian(11, 6).toarray() * 12 * grid(11)[1] ** 2
    assert np.allclose(A[1], expected, atol=1e-12)
    assert np.allclose(A[-2], expected[::-1], atol=1e-12)


def test_mirror_first_row_of_order4_folds_ghost_point_onto_diagonal():
    # ghost u_-1 = -u_1 with weight +1 lowers the diagonal 30 -> 29
    assert np.allclose(_rows(9, 4, "mirror")[0][:4], [29, -16, 1, 0], atol=1e-12)


@pytest.mark.parametrize("order", [4, 6])
def test_mirror_laplacian_is_symmetric(order):
    A = laplacian(20, order, "mirror").toarray()
    assert np.allclose(A, A.T)


@pytest.mark.parametrize("order, power, lo", [(4, 4, 2), (6, 6, 3)])
def test_higher_order_laplacian_is_exact_on_interior_rows(order, power, lo):
    x, _ = grid(20)
    got = (laplacian(20, order) @ x**power)[lo:-lo]
    want = -power * (power - 1) * x[lo:-lo] ** (power - 2)
    assert np.allclose(got, want, rtol=1e-8, atol=1e-6)


def _rate(order, boundary):
    err = []
    for N in (31, 63):
        x, u = solve(lambda x: np.pi**2 * np.sin(np.pi * x), N, order, boundary)
        err.append(np.max(np.abs(u - np.sin(np.pi * x))))
    return np.log2(err[0] / err[1])


@pytest.mark.parametrize("order", [4, 6])
def test_mirror_solve_converges_with_stencil_order(order):
    assert abs(_rate(order, "mirror") - order) < 0.2


def test_reduce_solve_order4_converges_faster_than_third_order():
    assert _rate(4, "reduce") > 3.8


# 1d: Neumann boundary u'(1) = 0
def _neumann_error(bc, N):
    from fd import solve_neumann

    x, u = solve_neumann(lambda x: np.pi**2 / 4 * np.sin(np.pi * x / 2), N, bc)
    return x, u - np.sin(np.pi * x / 2)


def test_neumann_grid_includes_right_endpoint():
    from fd import solve_neumann

    x, _ = solve_neumann(lambda x: 0 * x, 7)
    assert len(x) == 8
    assert x[-1] == 1.0
    assert x[0] == 0.125


def test_neumann_ghost_is_exact_for_quadratic_solution():
    from fd import solve_neumann

    x, u = solve_neumann(lambda x: 2 + 0 * x, 15, "ghost")
    assert np.allclose(u, x * (2 - x), atol=1e-10)


@pytest.mark.parametrize("bc, order", [("first", 1), ("ghost", 2)])
def test_neumann_converges_with_expected_order(bc, order):
    e = [np.max(np.abs(_neumann_error(bc, N)[1])) for N in (31, 63)]
    assert abs(np.log2(e[0] / e[1]) - order) < 0.1


def test_neumann_first_order_error_is_linear_over_whole_interval():
    # One-sided difference enforces u'(1) = (dx/2) u''(1) = -pi^2 dx/8; the perturbation
    # is the linear solution of -u'' = 0 with u(0) = 0.
    x, err = _neumann_error("first", 63)
    dx = 1 / 64
    assert np.allclose(err, -np.pi**2 / 8 * dx * x, rtol=0.02, atol=0)


# 2: time-dependent heat equation, method of lines u' = J u with J = -A
def _mode1(N):
    x, dx = grid(N)
    return x, -laplacian(N), 4 / dx**2 * np.sin(np.pi * dx / 2) ** 2


def test_implicit_euler_step_matches_dense_solve_with_sparse_matrix():
    from fd import implicit_euler_step

    x, J, _ = _mode1(12)
    u0 = x * (1 - x) ** 2
    want = np.linalg.solve(np.eye(12) - 0.01 * J.toarray(), u0)
    assert np.allclose(implicit_euler_step(J, u0, 0.01), want, rtol=1e-12)


@pytest.mark.parametrize("name, R", [
    ("implicit_euler_step", lambda z: 1 / (1 - z)),
    ("crank_nicolson_step", lambda z: (1 + z / 2) / (1 - z / 2)),
])
def test_implicit_steps_scale_eigenvector_by_stability_function(name, R):
    import fd

    x, J, lam1 = _mode1(20)
    u0, dt = np.sin(np.pi * x), 0.003
    assert np.allclose(getattr(fd, name)(J, u0, dt), R(-lam1 * dt) * u0, rtol=1e-12)


def test_pulse_series_at_start_is_one_inside_and_zero_outside():
    from fd import pulse_series

    u = pulse_series(np.array([0.1, 0.5, 0.9]), 0.0, kmax=4001)
    assert np.allclose(u, [0, 1, 0], atol=1e-3)


def test_pulse_series_first_mode_decays_with_pi_squared():
    from fd import pulse_series

    x, t = np.linspace(0, 1, 7), 0.05
    b1 = 2 / np.pi * (np.cos(np.pi / 4) - np.cos(3 * np.pi / 4))
    assert np.allclose(pulse_series(x, t, kmax=1), b1 * np.exp(-np.pi**2 * t) * np.sin(np.pi * x))


# 1e: 2D Poisson on the unit square
def test_laplacian_2d_center_row_is_five_point_stencil():
    from fd import laplacian_2d

    A = laplacian_2d(3)
    assert sp.issparse(A) and A.shape == (9, 9)
    assert np.array_equal(A.toarray()[4] * grid(3)[1] ** 2, [0, -1, 0, -1, 4, -1, 0, -1, 0])


def test_laplacian_2d_eigenvalues_are_sums_of_1d_eigenvalues():
    from fd import laplacian_2d

    lam = np.linalg.eigvalsh(laplacian(6).toarray())
    want = np.sort((lam[:, None] + lam[None, :]).ravel())
    assert np.allclose(np.linalg.eigvalsh(laplacian_2d(6).toarray()), want, rtol=1e-12)


def test_solve_2d_keeps_x_and_y_apart_for_eigenvector_source():
    # sin(pi x) sin(2 pi y) is an eigenvector; swapping x and y would give a different field
    from fd import solve_2d

    N = 15
    X, Y, U = solve_2d(lambda x, y: 5 * np.pi**2 * np.sin(np.pi * x) * np.sin(2 * np.pi * y), N)
    dx = grid(N)[1]
    lam = lambda k: 4 / dx**2 * np.sin(k * np.pi * dx / 2) ** 2
    assert U.shape == (N, N) and np.allclose(X[:, 0], grid(N)[0]) and np.allclose(Y[0], grid(N)[0])
    assert np.allclose(U, 5 * np.pi**2 / (lam(1) + lam(2)) * np.sin(np.pi * X) * np.sin(2 * np.pi * Y), rtol=1e-10)
