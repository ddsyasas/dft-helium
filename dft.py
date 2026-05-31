"""dft.py - reusable physics engine for the DFT helium project.

This module holds the numerics so that both the notebook (helium_dft.ipynb) and
the test suite (tests/test_dft.py) import the *same* code. The notebook keeps the
narrative, demo calls, and plots; the validated physics lives here.

Atomic units throughout: hbar = m_e = e = 4*pi*eps0 = 1. Length in Bohr, energy
in Hartree. The radial wavefunction is written u(r) = r * R(r); the orbital is
psi = R(r) * Y_00, and normalisation is integral u(r)^2 dr = 1.

Everything (integration, the Numerov solve, root finding, the Poisson solve) is
hand-written on purpose so every line can be explained to an examiner.

Milestones implemented here:
  M0  make_grid, integrate          (scaffolding: grid + reusable integrator)
  M1  solve_radial, shoot,          (one electron in a bare nucleus -Z/r)
      find_eigenvalue, normalise
  M2  hartree_potential             (electron-cloud repulsion via Poisson)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Default numerical parameters (the notebook documents what each one means).
# H_STEP : grid spacing in Bohr. 0.001 resolves the steep e^{-Zr} cusp of the
#          Z=2 orbital well enough to hit E=-2.0 (at 0.005 it is off by ~1.5e-3).
# R_MAX  : outer edge of the grid in Bohr; the 1s orbital is negligible by ~25.
# ---------------------------------------------------------------------------
H_STEP = 0.001   # Bohr
R_MAX = 25.0     # Bohr


# ===========================================================================
# Milestone 0: scaffolding -- the grid and the one reusable integrator.
# ===========================================================================
def make_grid(h=H_STEP, r_max=R_MAX):
    """Build the uniform radial grid r = h, 2h, ..., r_max.

    The grid starts at r = h (NOT 0): the nuclear potential -Z/r and the Hartree
    potential both diverge at the origin, so stepping one point out keeps every
    evaluation finite. Returns the array r (length N = r_max / h).
    """
    n = int(round(r_max / h))
    return (np.arange(n) + 1) * h


def integrate(y, h=H_STEP):
    """Approximate integral of y over the grid via composite Simpson's rule.

    Simpson fits a parabola through each consecutive triple of points, far more
    accurate than the trapezoid rule for smooth integrands like our wavefunctions
    For a uniform step h:
        integral ~ (h/3)[ y0 + 4(y1+y3+...) + 2(y2+y4+...) + y_{N-1} ].
    Simpson needs an even number of intervals (odd number of points); if handed
    an even number of points we Simpson the first N-1 and close the last interval
    with a single trapezoid step, so the routine accepts any length.

    The [0, h] sliver is dropped (the grid starts at r=h). For our integrands
    (u^2 ~ r^2 near 0) that piece is O(h^3) and negligible, and it keeps us away
    from the 1/r singularity at the origin.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 1:
        return 0.0
    if n % 2 == 1:
        s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])
        return s * h / 3.0
    else:
        s = y[0] + y[-2] + 4.0 * np.sum(y[1:-2:2]) + 2.0 * np.sum(y[2:-2:2])
        simpson_part = s * h / 3.0
        trapz_tail = 0.5 * (y[-2] + y[-1]) * h
        return simpson_part + trapz_tail


# ===========================================================================
# Milestone 1: hydrogen radial solver (the core engine).
# Solve u''(r) = 2[V(r) - E] u(r) for the bound state, V(r) = -Z/r.
# ===========================================================================
def solve_radial(E, V, r, h=H_STEP):
    """Integrate the radial Schrodinger equation inward for a given energy E.

    Inputs: E (trial energy, < 0 for a bound state), V (potential array on the
    grid), r (the grid), h (step). Returns the un-normalised wavefunction u(r).

    Method: Numerov for u'' = g u with g(r) = 2[V(r) - E]. We march from the
    outer edge inward; solving the Numerov recurrence for the next-inner point,
        f_n     = 1 - (h^2/12) g_n
        u_{n-1} = [ 2(1 + 5h^2/12 g_n) u_n - f_{n+1} u_{n+1} ] / f_{n-1}.

    Why inward: far out the physical state decays as e^{-kappa r}; stepping
    inward this grows while the unphysical e^{+kappa r} solution shrinks away, so
    inward integration stays locked onto the physical solution.
    """
    g = 2.0 * (V - E)
    f = 1.0 - (h * h / 12.0) * g
    u = np.zeros_like(r)

    # Seed the two outermost points with the decaying tail u ~ r e^{-kappa r},
    # kappa = sqrt(-2E). The overall scale is arbitrary (the ODE is linear).
    kappa = np.sqrt(-2.0 * E)
    u[-1] = r[-1] * np.exp(-kappa * r[-1])
    u[-2] = r[-2] * np.exp(-kappa * r[-2])

    for n in range(len(r) - 2, 0, -1):
        u[n - 1] = (2.0 * (1.0 + 5.0 * (h * h / 12.0) * g[n]) * u[n]
                    - f[n + 1] * u[n + 1]) / f[n - 1]
    return u


def shoot(E, Z, r, h=H_STEP):
    """Return (u(0), u-array) for trial energy E in the bare potential -Z/r.

    A bound state requires u(0) = 0. Our grid starts at r=h, so we linearly
    extrapolate the two innermost points (r=h, r=2h) back to r=0:
        u(0) ~ 2 u(h) - u(2h).
    The sign of this value flips as E sweeps through an eigenvalue, which is what
    the root finder brackets.
    """
    V = -Z / r
    u = solve_radial(E, V, r, h)
    u0 = 2.0 * u[0] - u[1]
    return u0, u


def normalise(u, h=H_STEP):
    """Scale u so integral u^2 dr = 1, and fix the sign so the main lobe is +.

    Quantum mechanics requires total probability 1: integral |u|^2 dr = 1. The
    radial solve gives u up to a constant, so we divide by sqrt(that integral).
    We also flip the sign if needed so the nodeless ground state comes out
    positive, for a clean comparison with the analytic form.
    """
    norm = np.sqrt(integrate(u * u, h))
    u = u / norm
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    return u


def _u0(E, V, r, h):
    """Inward-solve at energy E for an arbitrary potential V(r); return
    (extrapolated u(0), full u array). The sign of u(0) is the bisection signal.
    """
    u = solve_radial(E, V, r, h)
    return 2.0 * u[0] - u[1], u


def eigenstate(V, r, h=H_STEP, e_lo=-3.0, e_hi=-0.01, n_scan=200, tol=1e-10):
    """Find the GROUND-STATE energy E and normalised u(r) for ANY potential V(r).

    This is the generalised version of the M1 solver: M1 used the fixed bare
    potential -Z/r, but the SCF loop (M3+) needs the eigenstate of
    V(r) = -Z/r + V_H(r) + ... , so the potential is passed in as an array.

    Two stages (identical algorithm to M1):
      1. Coarse scan of E over [e_lo, e_hi] for the lowest (most negative) sign
         change of the extrapolated u(0). The lowest crossing is the nodeless
         ground state; higher crossings are excited s-states (2s, ...).
      2. Bisection inside that bracket until the energy is pinned to 'tol'.
    """
    # Stage 1: scan for the first sign change as E increases from e_lo.
    energies = np.linspace(e_lo, e_hi, n_scan)
    e_prev = energies[0]
    prev_val, _ = _u0(e_prev, V, r, h)
    bracket = None
    for E in energies[1:]:
        val, _ = _u0(E, V, r, h)
        if np.sign(val) != np.sign(prev_val):
            bracket = (e_prev, E)
            break
        e_prev, prev_val = E, val
    if bracket is None:
        raise RuntimeError("No bound state found in the scan window; "
                           "widen [e_lo, e_hi].")

    # Stage 2: bisection on E within the bracket.
    a, b = bracket
    fa, _ = _u0(a, V, r, h)
    while (b - a) > tol:
        m = 0.5 * (a + b)
        fm, _ = _u0(m, V, r, h)
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
    E = 0.5 * (a + b)
    _, u = _u0(E, V, r, h)
    return E, normalise(u, h)


def find_eigenvalue(Z, r, h=H_STEP, e_lo=None, e_hi=None, n_scan=200, tol=1e-10):
    """Ground-state E and normalised u(r) for the bare nuclear potential -Z/r.

    Thin wrapper around `eigenstate` with V = -Z/r and a search window straddling
    the He-like scale -Z^2/2 (from -0.6 Z^2, below the ground state, up to ~0).
    Kept as the M1 entry point so the M1 demos/tests read naturally.
    """
    if e_lo is None:
        e_lo = -0.6 * Z * Z
    if e_hi is None:
        e_hi = -0.01
    return eigenstate(-Z / r, r, h, e_lo=e_lo, e_hi=e_hi,
                      n_scan=n_scan, tol=tol)


# ===========================================================================
# Milestone 2: Hartree potential via the radial Poisson equation.
# ===========================================================================
def hartree_potential(u, r, h=H_STEP):
    """Compute the Hartree potential V_H(r) from a normalised orbital u(r).

    The Hartree potential is the electrostatic repulsion produced by the electron
    cloud itself. With radial symmetry, defining U(r) = r * V_H(r), the radial
    Poisson equation (PDF Eqs. 4-6) reduces -- after the 4*pi cancels against the
    normalisation -- to:
        U''(r) = - u(r)^2 / r.
    The right-hand side is the (spherically averaged) charge source.

    Boundary conditions (PDF Eqs. 7-10):
      * U(0) = 0          (the potential times r vanishes at the origin)
      * U(r_max) = q_max  where q_max = integral_0^{r_max} u^2 dr is the charge
        enclosed (-> 1 for a normalised orbital).

    Method:
      1. Integrate U'' = s(r), s = -u^2/r, OUTWARD with the Verlet/second-
         difference scheme  U_{n+1} = 2 U_n - U_{n-1} + h^2 s_n, starting from the
         virtual origin U(r=0) = 0. The starting slope is arbitrary because...
      2. ...any two particular solutions of this linear ODE that both satisfy
         U(0)=0 differ only by a multiple of the homogeneous solution U = alpha*r
         (the other homogeneous solution, a constant, is killed by U(0)=0). So we
         add alpha*r, choosing alpha = (q_max - U(r_max)) / r_max, to pin the
         outer boundary exactly. This does not disturb U(0)=0.
      3. V_H(r) = U(r) / r.

    Inputs: u (MUST already be normalised), r (the grid), h (step).
    Returns (V_H, U, q_max).
    """
    # Charge source. Near r=0, u ~ r so u^2/r ~ r is finite; no singularity.
    s = -(u * u) / r

    # --- Step 1: outward Verlet integration from the virtual origin U(0)=0. ---
    n = len(r)
    U = np.zeros(n)
    # The recurrence at the first grid point (r=h) needs U at r=0 and r=h.
    # U(r=0) = 0 (virtual point). We pick the slope freely; U(h) = h means slope
    # 1, an arbitrary choice that the alpha*r correction in step 2 reabsorbs.
    U[0] = h
    # U[1] from the recurrence with U(r=0)=0:  U[1] = 2 U[0] - U(0) + h^2 s[0].
    U[1] = 2.0 * U[0] - 0.0 + h * h * s[0]
    for i in range(1, n - 1):
        U[i + 1] = 2.0 * U[i] - U[i - 1] + h * h * s[i]

    # --- Step 2: add the homogeneous alpha*r to satisfy U(r_max) = q_max. ---
    q_max = integrate(u * u, h)
    alpha = (q_max - U[-1]) / r[-1]
    U = U + alpha * r

    # --- Step 3: the Hartree potential itself. ---
    V_H = U / r
    return V_H, U, q_max


# ===========================================================================
# Milestone 3: self-consistent DFT for helium, no exchange-correlation (Z=2).
# ===========================================================================
def scf_no_xc(Z=2.0, r=None, h=H_STEP, w=0.3, tol=1e-7, max_iter=200,
              n_scan=120, eig_tol=1e-9, verbose=True):
    """Self-consistent helium ground state with the Hartree term only (M3).

    The physics. Each of helium's two electrons sits in the same 1s orbital and
    feels two things: the nuclear pull -Z/r AND the electrostatic repulsion of
    the *other* electron's cloud, the Hartree potential V_H(r). But V_H depends
    on the orbital, which depends on V_H -- a circular problem. We break the
    circle by iterating to self-consistency (SCF): guess -> solve -> rebuild the
    potential -> repeat until the eigenvalue stops moving.

    Expected result: E = -2.86 Ha. That is the restricted Hartree-Fock value,
    and getting it here is CORRECT, not a bug. For a two-electron closed-shell
    singlet (both electrons in one spatial 1s orbital) the self-interaction-
    removed Hartree potential coincides exactly with the RHF potential, because
    HF exchange does nothing more than cancel the self-interaction: with one
    occupied orbital (2J - K)phi = (2J - J)phi = J phi. So SIC-Hartree == RHF for
    helium, and -2.86 is the right M3 number. The -2.72 figure is a later
    milestone (M4: full Hartree + Slater exchange, no correlation).

    The factor-of-2 bookkeeping (the classic trap, see spec section 7). With two
    electrons the naive Hartree would double-count and would also include each
    electron interacting with itself. In this no-XC milestone the factor 2 from
    spin (two electrons in one orbital) exactly cancels the 1/2 that removes the
    self-interaction, so the effective V_H is simply the potential of a single
    NORMALISED orbital density -- which is exactly what hartree_potential()
    already returns (it solves U'' = -u^2/r with q_max -> 1). So no extra factor
    is applied here; we use hartree_potential(u) directly.

    The algorithm:
      0. Initial guess: the bare-nucleus orbital (V = -Z/r), giving its V_H.
      1. Build V(r) = -Z/r + V_H(r) and solve for the ground state (eps, u).
      2. Rebuild the Hartree potential from the new u.
      3. Mix:  V_H <- (1-w) V_H_old + w V_H_new   (w ~ 0.3) to damp oscillation
         (PDF/spec recommendation; pure replacement, w=1, can oscillate).
      4. Repeat until |eps - eps_prev| < tol.

    Total energy (PDF Eq. 13 / spec Eq. SCF):
        E = 2 eps - integral V_H(r) u(r)^2 dr.
    The 2 eps adds both electrons' single-particle energies; each already
    contains the full electron-electron interaction once, so summing them
    double-counts it. Subtracting integral V_H u^2 removes that one extra count.

    Inputs:  Z (nuclear charge), r (grid; built if None), h, mixing w,
             convergence tol on eps, max_iter, verbose (print E per iteration).
    Returns: dict with E (total energy), eps (eigenvalue), u, V_H, r,
             iterations, and the per-iteration history of (eps, E).
    """
    if r is None:
        r = make_grid(h)
    V_nuc = -Z / r

    # --- Step 0: initial guess from the bare nucleus (no repulsion yet). ---
    eps, u = eigenstate(V_nuc, r, h, e_lo=-0.6 * Z * Z, e_hi=-0.01,
                        n_scan=n_scan, tol=eig_tol)
    V_H, _, _ = hartree_potential(u, r, h)

    if verbose:
        print(f"{'iter':>4}  {'eps (Ha)':>12}  {'E_total (Ha)':>14}  "
              f"{'d eps':>10}")

    history = []
    eps_prev = None
    iterations = 0
    for it in range(1, max_iter + 1):
        iterations = it
        # Step 1: solve the radial SE in the current effective potential.
        # The repulsive V_H raises the level above the bare -Z^2/2, so search a
        # window from below the bare ground state up to the ionisation edge.
        V = V_nuc + V_H
        eps, u = eigenstate(V, r, h, e_lo=-0.6 * Z * Z, e_hi=-0.01,
                            n_scan=n_scan, tol=eig_tol)

        # Step 2: Hartree potential of the *current* density (self-consistent
        # piece used in the energy), plus the energy itself.
        V_H_new, _, _ = hartree_potential(u, r, h)
        E_tot = 2.0 * eps - integrate(V_H_new * u * u, h)

        d_eps = float("nan") if eps_prev is None else abs(eps - eps_prev)
        history.append((eps, E_tot))
        if verbose:
            print(f"{it:>4}  {eps:>12.6f}  {E_tot:>14.6f}  {d_eps:>10.2e}")

        # Step 4: convergence test on the eigenvalue.
        if eps_prev is not None and d_eps < tol:
            break
        eps_prev = eps

        # Step 3: mix the Hartree potential for the next iteration.
        V_H = (1.0 - w) * V_H + w * V_H_new

    return {
        "E": E_tot,
        "eps": eps,
        "u": u,
        "V_H": V_H_new,
        "r": r,
        "iterations": iterations,
        "history": history,
    }
