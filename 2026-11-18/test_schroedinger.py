"""Tests for exercise 6 (Schroedinger equation). Units hbar = m = 1.

What each test checks: theory.md, section of the same task.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from schroedinger import (absorber, eigenstates, free_packet, gaussian, grid, hamiltonian, hamiltonian_2d,
                          moments, solenoid, stepper, transmission, well_energies)


# --- grid and Hamiltonian (1) ---

def test_grid_has_interior_points_only():
    x, dx = grid(0, 1, 3)
    assert dx == 0.25
    assert np.allclose(x, [0.25, 0.5, 0.75])


def test_hamiltonian_n3_is_kinetic_stencil_plus_potential():
    V = np.array([1.0, 2.0, 3.0])
    H = hamiltonian(V, 0.5)
    assert sp.issparse(H)
    expected = -0.5 * np.array([[-2, 1, 0], [1, -2, 1], [0, 1, -2]]) / 0.25 + np.diag(V)
    assert np.allclose(H.toarray(), expected)


def test_hamiltonian_keeps_complex_potential():
    H = hamiltonian(np.array([0, -1j, 0]), 1.0)
    assert H[1, 1] == 1 - 1j


def test_eigenstates_infinite_well_match_discrete_formula_and_are_normalized():
    x, dx = grid(0, 1, 200)
    E, psi = eigenstates(np.zeros_like(x), dx, 4)
    n = np.arange(1, 5)
    assert np.allclose(E, 2 / dx**2 * np.sin(n * np.pi * dx / 2) ** 2, rtol=1e-10)
    assert psi.shape == (200, 4)
    assert np.allclose(np.sum(np.abs(psi) ** 2, axis=0) * dx, 1)


def test_eigenstates_infinite_well_converge_to_n2_pi2_over_2():
    x, dx = grid(0, 1, 1000)
    E, _ = eigenstates(np.zeros_like(x), dx, 3)
    assert np.allclose(E, np.arange(1, 4) ** 2 * np.pi**2 / 2, rtol=1e-4)


def test_eigenstates_harmonic_oscillator_give_n_plus_half():
    x, dx = grid(-10, 10, 2000)
    E, _ = eigenstates(x**2 / 2, dx, 5)
    assert np.allclose(E, np.arange(5) + 0.5, atol=1e-3)


def test_well_energies_v0_10_a_1_has_three_bound_states():
    E = well_energies(10, 1)
    assert len(E) == 3
    assert np.all(np.diff(E) > 0)
    assert np.all((E > -10) & (E < 0))


def test_well_energies_solve_matching_conditions():
    E = well_energies(10, 1)
    k, kappa = np.sqrt(2 * (E + 10)), np.sqrt(-2 * E)
    assert np.isclose(k[0] * np.tan(k[0]), kappa[0])  # even ground state
    assert np.isclose(-k[1] / np.tan(k[1]), kappa[1])  # odd first excited state
    assert np.isclose(k[2] * np.tan(k[2]), kappa[2])


def test_well_energies_agree_with_finite_differences():
    x, dx = grid(-8, 8, 4000)
    E_fd, _ = eigenstates(np.where(np.abs(x) < 1, -10.0, 0.0), dx, 3)
    assert np.allclose(E_fd, well_energies(10, 1), atol=2e-2)


# --- wave packets (2) ---

def test_gaussian_is_normalized_with_mean_x0_and_width_sigma():
    x, dx = grid(-30, 30, 6000)
    psi = gaussian(x, 2.0, 1.5, 3.0)
    norm, mean, std = moments(x, psi, dx)
    assert np.isclose(norm, 1, atol=1e-10)
    assert np.isclose(mean, 2.0, atol=1e-10)
    assert np.isclose(std, 1.5, atol=1e-8)


def test_gaussian_carries_plane_wave_phase():
    x = np.array([0.0, 1.0])
    psi = gaussian(x, 0.0, 1e6, 2.0)  # sigma huge: only the phase is left
    assert np.isclose(psi[1] / psi[0], np.exp(2j))


def test_moments_report_norm_not_squared_norm():
    x, dx = grid(-30, 30, 6000)
    norm, _, _ = moments(x, 3 * gaussian(x, 0, 1, 0), dx)
    assert np.isclose(norm, 3)


def test_free_packet_starts_at_x0_and_spreads():
    assert free_packet(0, -5, 1, 2) == pytest.approx((-5, 1))
    mean, std = free_packet(4, -5, 1, 2)
    assert mean == pytest.approx(3)
    assert std == pytest.approx(np.sqrt(1 + (4 / 2) ** 2))  # sigma sqrt(1 + (t / 2 sigma^2)^2)


def _well(n=100):
    x, dx = grid(0, 1, n)
    return x, dx, hamiltonian(np.zeros_like(x), dx)


def test_stepper_crank_nicolson_conserves_norm():
    x, dx, H = _well()
    step = stepper(H, 1e-3, "cn")
    psi = gaussian(x, 0.4, 0.05, 20)
    for _ in range(200):
        psi = step(psi)
    assert np.isclose(moments(x, psi, dx)[0], 1, atol=1e-12)


def test_stepper_implicit_euler_loses_norm_explicit_euler_gains_norm():
    x, dx, H = _well()
    psi0 = gaussian(x, 0.4, 0.05, 20)
    norms = {}
    for method in ("explicit", "implicit"):
        step, psi = stepper(H, 1e-4, method), psi0
        for _ in range(20):
            psi = step(psi)
        norms[method] = moments(x, psi, dx)[0]
    assert norms["implicit"] < 1 - 1e-4
    assert norms["explicit"] > 1 + 1e-4


def test_stepper_crank_nicolson_turns_eigenstate_by_cayley_phase():
    x, dx, H = _well()
    E, psi = eigenstates(np.zeros_like(x), dx, 1)
    dt = 1e-3
    new = stepper(H, dt, "cn")(psi[:, 0].astype(complex))
    assert np.allclose(new, psi[:, 0] * (1 - 0.5j * dt * E[0]) / (1 + 0.5j * dt * E[0]))


def test_stepper_rejects_unknown_method():
    _, _, H = _well(5)
    with pytest.raises(ValueError):
        stepper(H, 0.1, "rk4")


def test_crank_nicolson_free_packet_matches_analytic_moments():
    x, dx = grid(-20, 40, 2400)
    H = hamiltonian(np.zeros_like(x), dx)
    step, psi = stepper(H, 0.01, "cn"), gaussian(x, -5, 1, 2)
    for _ in range(300):
        psi = step(psi)
    _, mean, std = moments(x, psi, dx)
    assert mean == pytest.approx(free_packet(3, -5, 1, 2)[0], abs=2e-2)
    assert std == pytest.approx(free_packet(3, -5, 1, 2)[1], abs=2e-2)


def test_transmission_is_continuous_at_barrier_top():
    V0, b = 1.0, 1.0
    below, above = transmission(np.array([V0 - 1e-7, V0 + 1e-7]), V0, b)
    assert below == pytest.approx(1 / (1 + V0 * b**2 / 2), rel=1e-5)
    assert above == pytest.approx(below, rel=1e-5)


def test_transmission_is_one_at_over_barrier_resonance_and_small_deep_below():
    V0, b = 1.0, 1.0
    assert transmission(V0 + np.pi**2 / 2, V0, b) == pytest.approx(1)
    assert transmission(0.01, 10.0, 3.0) < 1e-10
    E = np.linspace(0.05, 3, 50)
    T = transmission(E, V0, b)
    assert np.all((T > 0) & (T <= 1 + 1e-12))


def test_absorber_is_zero_inside_and_grows_towards_both_ends():
    x, _ = grid(0, 10, 99)
    W = absorber(x, 0, 10, 2, 5)
    assert np.all(W[(x > 2) & (x < 8)] == 0)
    assert np.all(W >= 0)
    assert W[0] > W[5] > 0 and W[-1] > W[-6] > 0
    assert W.max() <= 5


# --- 2D (3) ---

def test_hamiltonian_2d_without_field_is_kronecker_sum_of_1d():
    x, dx = grid(0, 1, 4)
    V = np.arange(16.0).reshape(4, 4)
    H1 = hamiltonian(np.zeros(4), dx)
    I = sp.identity(4)
    expected = sp.kron(H1, I) + sp.kron(I, H1) + sp.diags(V.ravel())
    assert np.allclose(hamiltonian_2d(x, x, V).toarray(), expected.toarray())


def test_hamiltonian_2d_with_constant_vector_potential_has_same_spectrum():
    # constant A is a pure gauge: A = grad(A.r); the Peierls phases are exact here
    x, _ = grid(-1, 1, 7)
    V = np.zeros((7, 7))
    H0 = hamiltonian_2d(x, x, V).toarray()
    HA = hamiltonian_2d(x, x, V, A=lambda X, Y: (0.7 + 0 * X, -0.3 + 0 * Y)).toarray()
    assert not np.allclose(H0, HA)
    assert np.allclose(np.linalg.eigvalsh(HA), np.linalg.eigvalsh(H0))


def test_hamiltonian_2d_with_solenoid_is_hermitian():
    x, _ = grid(-3, 3, 12)
    H = hamiltonian_2d(x, x, np.zeros((12, 12)), A=solenoid(2.0, 1.0, (0.3, -0.2)))
    assert abs(H - H.conj().T).max() < 1e-14
    assert np.iscomplexobj(H.toarray())


def test_solenoid_circulation_equals_flux_outside_and_vanishes_at_centre():
    B0, r0, c = 2.0, 1.0, (1.0, -0.5)
    A = solenoid(B0, r0, c)
    phi = np.linspace(0, 2 * np.pi, 4001)[:-1]
    for R in (1.5, 4.0):
        X, Y = c[0] + R * np.cos(phi), c[1] + R * np.sin(phi)
        Ax, Ay = A(X, Y)
        circulation = np.sum(-Ax * np.sin(phi) + Ay * np.cos(phi)) * R * (phi[1] - phi[0])
        assert circulation == pytest.approx(B0 * np.pi * r0**2, rel=1e-9)
    assert np.allclose(A(np.array([1.0]), np.array([-0.5])), 0)


def test_solenoid_is_continuous_at_r0_and_inside_is_uniform_field():
    A = solenoid(3.0, 1.0, (0, 0))
    inside, outside = A(np.array([0.0]), np.array([1 - 1e-9])), A(np.array([0.0]), np.array([1 + 1e-9]))
    assert np.allclose(inside, outside, atol=1e-6)
    # B0 r / 2 e_theta at (0.5, 0): e_theta = e_y
    assert np.allclose(A(np.array([0.5]), np.array([0.0])), ([0.0], [0.75]))


def test_hamiltonian_2d_peierls_sign_gives_kinetic_momentum_k_minus_a():
    # <H> of a packet with momentum k0 under constant A_x is (k0 - A)^2 / 2 + spread terms
    x, dx = grid(-15, 15, 299)
    X, Y = np.meshgrid(x, x, indexing="ij")
    k0, s, Ax = 1.0, 2.0, 0.5
    psi = (gaussian(X, 0, s, k0) * gaussian(Y, 0, s, 0)).ravel()
    H = hamiltonian_2d(x, x, np.zeros_like(X), A=lambda X, Y: (Ax + 0 * X, 0 * Y))
    energy = np.vdot(psi, H @ psi).real * dx**2
    assert energy == pytest.approx((k0 - Ax) ** 2 / 2 + 2 / (8 * s**2), abs=5e-3)


def test_hamiltonian_2d_rejects_different_spacing_in_x_and_y():
    x, _ = grid(0, 1, 9)
    y, _ = grid(0, 1, 7)
    with pytest.raises(ValueError):
        hamiltonian_2d(x, y, np.zeros((9, 7)))
