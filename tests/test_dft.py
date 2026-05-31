"""Validation tests for the DFT helium project.

Convention (see pdocs/project_spec.md, "Testing"): every milestone gets a pytest
test that mirrors its validation gate. `pytest -v` must be green before a
milestone is marked done in the devlog. Tolerances are ~1e-3, matching the
precision the spec/PDF quote for the target numbers.

The grid is built once and shared across tests (it is read-only here).
"""

import os
import sys

import numpy as np
import pytest

# Make the project root importable so `import dft` works no matter where pytest
# is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dft

TOL = 1e-3  # numerical tolerance for the validation gates


@pytest.fixture(scope="module")
def grid():
    """The shared radial grid (default h, r_max from dft.py)."""
    return dft.make_grid()


# --- Milestone 0: the reusable integrator must be trustworthy ----------------
def test_integrator_known_value(grid):
    """integral of e^{-r} over [0, inf) is 1; on our finite grid it is ~1."""
    val = dft.integrate(np.exp(-grid))
    assert abs(val - 1.0) < 2e-3


# --- Milestone 1: hydrogen radial solver -------------------------------------
def test_m1_hydrogen_energy(grid):
    """Z=1 ground-state energy must equal -0.5 Hartree."""
    E, _ = dft.find_eigenvalue(Z=1.0, r=grid)
    assert abs(E - (-0.5)) < TOL


def test_m1_hydrogen_wavefunction(grid):
    """Z=1 wavefunction must match the analytic 1s, u(r) = 2 r e^{-r}."""
    _, u = dft.find_eigenvalue(Z=1.0, r=grid)
    u_exact = 2.0 * grid * np.exp(-grid)
    assert np.max(np.abs(u - u_exact)) < TOL


def test_m1_helium_bare_nucleus_energy(grid):
    """Z=2 bare nucleus (no Hartree) energy must equal -2.0 Hartree (-Z^2/2)."""
    E, _ = dft.find_eigenvalue(Z=2.0, r=grid)
    assert abs(E - (-2.0)) < TOL


# --- Milestone 2: Hartree potential via Poisson ------------------------------
def test_m2_hartree_hydrogen_closed_form(grid):
    """Feeding the hydrogen 1s density must reproduce U(r) = -(r+1)e^{-2r} + 1.

    Also checks the enclosed charge q_max -> 1 and that V_H matches its closed
    form V_H(r) = 1/r - (1 + 1/r) e^{-2r}.
    """
    _, u = dft.find_eigenvalue(Z=1.0, r=grid)
    V_H, U, q_max = dft.hartree_potential(u, grid)

    U_exact = -(grid + 1.0) * np.exp(-2.0 * grid) + 1.0
    VH_exact = 1.0 / grid - (1.0 + 1.0 / grid) * np.exp(-2.0 * grid)

    assert abs(q_max - 1.0) < TOL
    assert np.max(np.abs(U - U_exact)) < TOL
    assert np.max(np.abs(V_H - VH_exact)) < TOL


# --- Milestone 3: self-consistent helium, Hartree only (no XC) ---------------
def test_m3_scf_no_xc_total_energy(grid):
    """SCF with the self-interaction-removed Hartree must give E = -2.86 Ha.

    This is the restricted Hartree-Fock value, and it is the CORRECT M3 answer:
    for a two-electron closed-shell singlet the SIC-Hartree potential equals the
    RHF potential exactly (exchange only cancels the self-interaction). The -2.72
    figure belongs to M4 (full Hartree + Slater exchange, no correlation).
    """
    res = dft.scf_no_xc(Z=2.0, r=grid, verbose=False)
    assert abs(res["E"] - (-2.86)) < 1e-2


# --- Milestone 4: full Hartree + Slater LDA exchange (no correlation) --------
def test_m4_scf_lda_exchange_total_energy(grid):
    """SCF with full Hartree + Slater exchange must give E = -2.72 Ha.

    Less bound than the M3 (-2.86, exact-exchange/HF) result because LDA Slater
    exchange removes the self-interaction only approximately.
    """
    res = dft.scf_lda_x(Z=2.0, r=grid, verbose=False)
    assert abs(res["E"] - (-2.72)) < 1e-2
