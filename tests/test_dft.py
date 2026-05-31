"""Validation tests for the DFT helium project.

Convention (see pdocs/project_spec.md, "Testing"): each stage of the build gets a
pytest test that mirrors its validation target; `pytest -v` must be green before
that stage is recorded as done in the devlog. Tolerances are ~1e-3, matching the
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


# --- The reusable integrator must be trustworthy before anything relies on it -
def test_integrator_known_value(grid):
    """integral of e^{-r} over [0, inf) is 1; on our finite grid it is ~1."""
    val = dft.integrate(np.exp(-grid))
    assert abs(val - 1.0) < 2e-3


# --- One electron in a bare nucleus: the hydrogen / He+ radial solver --------
def test_hydrogen_ground_state_energy(grid):
    """Z=1 ground-state energy must equal -0.5 Hartree."""
    E, _ = dft.find_eigenvalue(Z=1.0, r=grid)
    assert abs(E - (-0.5)) < TOL


def test_hydrogen_ground_state_wavefunction(grid):
    """Z=1 wavefunction must match the analytic 1s, u(r) = 2 r e^{-r}."""
    _, u = dft.find_eigenvalue(Z=1.0, r=grid)
    u_exact = 2.0 * grid * np.exp(-grid)
    assert np.max(np.abs(u - u_exact)) < TOL


def test_helium_bare_nucleus_energy(grid):
    """Z=2 bare nucleus (no Hartree) energy must equal -2.0 Hartree (-Z^2/2).

    This is He+ (one electron on the helium nucleus), the exact -Z^2/2 result.
    """
    E, _ = dft.find_eigenvalue(Z=2.0, r=grid)
    assert abs(E - (-2.0)) < TOL


# --- Hartree potential from the radial Poisson equation ----------------------
def test_hartree_potential_hydrogen_analytic(grid):
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


# --- Self-consistent Hartree (equals Hartree-Fock for the helium singlet) ----
def test_scf_hartree_equals_hf(grid):
    """SCF with the self-interaction-removed Hartree must give E = -2.86 Ha.

    This is the restricted Hartree-Fock value, and it is the CORRECT Hartree-only
    answer: for a two-electron closed-shell singlet the SIC-Hartree potential
    equals the RHF potential exactly (exchange only cancels the self-interaction).
    The -2.72 figure requires Slater exchange (see the next test).
    """
    res = dft.scf_no_xc(Z=2.0, r=grid, verbose=False)
    assert abs(res["E"] - (-2.86)) < 1e-2


# --- Self-consistent LDA with Slater exchange (no correlation) ---------------
def test_scf_lda_slater_exchange_energy(grid):
    """SCF with full Hartree + Slater exchange must give E = -2.72 Ha.

    Less bound than the Hartree-only (-2.86, exact-exchange/HF) result because
    LDA Slater exchange removes the self-interaction only approximately.
    """
    res = dft.scf_lda_x(Z=2.0, r=grid, verbose=False)
    assert abs(res["E"] - (-2.72)) < 1e-2


# --- Full LDA: Slater exchange + Perdew-Zunger correlation -------------------
def test_scf_full_lda_energy(grid):
    """Full LDA (Hartree + Slater X + PZ correlation) must give E = -2.83 Ha.

    Correlation is a small attractive correction that deepens the exchange-only
    -2.72 toward the exact -2.90; the LDA lands at about -2.83.
    """
    res = dft.scf_lda_xc(Z=2.0, r=grid, verbose=False)
    assert abs(res["E"] - (-2.83)) < 1e-2
