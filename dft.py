"""dft.py - reusable physics engine for the DFT helium project.

This module holds the numerics so that both the notebook (helium_dft.ipynb) and
the test suite (tests/test_dft.py) import the *same* code. The notebook keeps the
narrative, demo calls, and plots; the validated physics lives here.

Atomic units throughout: hbar = m_e = e = 4*pi*eps0 = 1. Length in Bohr, energy
in Hartree. The radial wavefunction is written u(r) = r * R(r); the orbital is
psi = R(r) * Y_00, and normalisation is integral u(r)^2 dr = 1.

Everything (integration, the Numerov solve, root finding, the Poisson solve) is
hand-written on purpose so every line can be explained to an examiner.

What lives here, building up the physics one layer at a time:
  - make_grid, integrate          radial grid + reusable Simpson integrator
  - solve_radial, shoot,          one electron in a bare nucleus -Z/r
    eigenstate, find_eigenvalue,  (the radial Schrodinger solver + root finder)
    normalise
  - hartree_potential             electron-cloud electrostatic repulsion (Poisson)
  - scf_no_xc                     self-consistent Hartree (= Hartree-Fock for He)
  - exchange_potential, scf_lda_x self-consistent LDA with Slater exchange
  - correlation_pz, scf_lda_xc    full LDA: Slater exchange + PZ correlation
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
# Scaffolding: the radial grid and the one reusable integrator.
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
# Hydrogen-like radial solver (the core engine reused by every later step).
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


def eigenstate(V, r, h=H_STEP, e_lo=-3.0, e_hi=-0.01, n_scan=200, tol=1e-10,
               e_guess=None):
    """Find the GROUND-STATE energy E and normalised u(r) for ANY potential V(r).

    This is the generalised version of the bare-nucleus solver: find_eigenvalue
    used the fixed potential -Z/r, but the self-consistent loops need the
    eigenstate of V(r) = -Z/r + V_H(r) + V_x(r) + ... , so the potential is
    passed in as an array.

    Two stages (identical algorithm to the bare-nucleus case):
      1. Bracket the ground state. By default we coarse-scan E over [e_lo, e_hi]
         for the lowest (most negative) sign change of the extrapolated u(0); the
         lowest crossing is the nodeless ground state (higher crossings are 2s,
         ...). If `e_guess` is given (an SCF warm start), we instead bracket
         LOCALLY by expanding a small window around the guess until the sign
         changes -- far cheaper than a full rescan when we already know roughly
         where the eigenvalue sits. Either way stage 2 bisects the SAME isolated
         root to the SAME tolerance, so the answer is identical; the guess only
         changes how we find the bracket, not the bracketed root.
      2. Bisection inside that bracket until the energy is pinned to 'tol'.
    """
    bracket = None
    if e_guess is not None:
        # Local bracketing around the warm start. Expand a symmetric window
        # (clamped to [e_lo, e_hi]) until u(0) changes sign across it. Starting
        # near the true root, this brackets it long before reaching a neighbour.
        lo, hi = e_guess - 0.02, e_guess + 0.02
        f_lo, _ = _u0(max(lo, e_lo), V, r, h)
        f_hi, _ = _u0(min(hi, e_hi), V, r, h)
        for _ in range(60):
            if np.sign(f_lo) != np.sign(f_hi):
                bracket = (max(lo, e_lo), min(hi, e_hi))
                break
            lo, hi = lo - 0.05, hi + 0.05
            if lo <= e_lo or hi >= e_hi:   # fell off the window: scan instead
                break
            f_lo, _ = _u0(lo, V, r, h)
            f_hi, _ = _u0(hi, V, r, h)
        # If the local search failed, fall through to the robust full scan.

    if bracket is None:
        # Stage 1 (default): scan for the first sign change as E rises from e_lo.
        energies = np.linspace(e_lo, e_hi, n_scan)
        e_prev = energies[0]
        prev_val, _ = _u0(e_prev, V, r, h)
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
    the hydrogen-like scale -Z^2/2 (from -0.6 Z^2, below the ground state, up to
    ~0). This is the convenience entry point for the bare-nucleus (hydrogen and
    He+) checks, where the potential is just -Z/r and reads naturally as such.
    """
    if e_lo is None:
        e_lo = -0.6 * Z * Z
    if e_hi is None:
        e_hi = -0.01
    return eigenstate(-Z / r, r, h, e_lo=e_lo, e_hi=e_hi,
                      n_scan=n_scan, tol=tol)


# ===========================================================================
# Hartree potential: the electron cloud's electrostatic repulsion, found by
# solving the radial Poisson equation.
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
# Self-consistent Hartree for helium (no exchange-correlation). For the
# two-electron singlet this coincides with restricted Hartree-Fock.
# ===========================================================================
def scf_no_xc(Z=2.0, r=None, h=H_STEP, w=0.3, tol=1e-7, max_iter=200,
              n_scan=120, eig_tol=1e-9, verbose=True):
    """Self-consistent helium ground state with the Hartree term only.

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
    helium, and -2.86 is the right Hartree-only number. The -2.72 figure comes
    later, once Slater exchange is added (see scf_lda_x).

    The factor-of-2 bookkeeping (the classic trap). With two electrons the naive
    Hartree would double-count and would also include each electron interacting
    with itself. In this Hartree-only (no exchange-correlation) treatment the
    factor 2 from spin (two electrons in one orbital) exactly cancels the 1/2
    that removes the self-interaction, so the effective V_H is simply the
    potential of a single
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


# ===========================================================================
# Self-consistent LDA with Slater exchange: add the exchange potential to the
# full-Hartree SCF loop (Z=2).
# ===========================================================================
def exchange_potential(u, r):
    """Slater LDA exchange potential from the radial orbital u(r).

    LDA borrows the exchange energy of the uniform electron gas and applies it
    point by point: V_x(r) = -(3/pi)^{1/3} n(r)^{1/3}. Here n is the FULL density
    of BOTH electrons. In terms of u (normalised so integral u^2 dr = 1), the
    full density is n(r) = u(r)^2 / (2 pi r^2) (twice the single-orbital
    density), and the potential becomes (PDF Eq. 17):
        V_x(r) = - ( 3 u(r)^2 / (2 pi^2 r^2) )^{1/3}.
    It is negative (exchange lowers the energy) and depends only on r for an
    s-state. Where u -> 0 the density and V_x vanish smoothly; r > 0 on our grid
    so there is no singularity.
    """
    return -(3.0 * u * u / (2.0 * np.pi**2 * r * r))**(1.0 / 3.0)


def scf_lda_x(Z=2.0, r=None, h=H_STEP, w=0.3, tol=1e-7, max_iter=300,
              n_scan=120, eig_tol=1e-9, verbose=True):
    """Self-consistent helium with full Hartree + Slater LDA exchange.

    This is a SEPARATE SCF variant from the Hartree-only scf_no_xc; that path is
    left untouched so its -2.86 result keeps passing. Two things change meaning
    relative to the Hartree-only treatment (the second classic bug site, so be
    deliberate):

      1. The density is now the FULL two-electron density n = u^2/(2 pi r^2)
         (the Hartree-only run used the single-orbital density).
      2. The Hartree is now the FULL Hartree, i.e. the potential of that doubled
         density, which is exactly TWICE the single-orbital Hartree that
         hartree_potential() returns:  V_H_full = 2 * V_H_single. We no longer
         halve it for self-interaction -- because that job now belongs to the
         exchange term.

    The effective potential inside the loop is
        V(r) = -Z/r + V_H_full(r) + V_x(r).

    Total energy (PDF Eq. 18 / spec Eq. X-E), with the FULL V_H:
        E = 2 eps - integral V_H_full u^2 dr - (1/2) integral u^2 V_x dr.
    The Hartree double-count is removed as before; the -(1/2) integral u^2 V_x
    converts the exchange potential back into the exchange ENERGY (the energy is
    half the potential's first moment for this n^{1/3} form).

    Why this is LESS bound than the Hartree-only run (a good report point). That
    run (SIC-Hartree) equals restricted Hartree-Fock and uses EXACT exchange,
    which removes the self-interaction perfectly, giving -2.86. LDA Slater
    exchange is only an
    APPROXIMATE, averaged exchange (borrowed from the uniform electron gas), so
    it removes the self-interaction only approximately and leaves a residual
    self-repulsion. That extra repulsion pushes the energy up to about -2.72,
    i.e. less bound than the exact-exchange HF value.

    Initialisation. We ramp the interaction in from zero (V_H = V_x = 0 on the
    first step) rather than starting from the compact bare orbital: the full
    (un-halved) Hartree built from the bare He+ orbital is so repulsive it can
    unbind the electron. Mixing (w ~ 0.3) then grows the potentials gently to
    self-consistency.

    Returns: dict with E, eps, u, V_H (full), V_x, r, iterations, history.
    """
    if r is None:
        r = make_grid(h)
    V_nuc = -Z / r

    # Ramp the interaction in from zero (see docstring) for a stable start.
    V_H = np.zeros_like(r)   # full Hartree, grown in via mixing
    V_x = np.zeros_like(r)   # Slater exchange, grown in via mixing

    if verbose:
        print(f"{'iter':>4}  {'eps (Ha)':>12}  {'E_total (Ha)':>14}  "
              f"{'d eps':>10}")

    history = []
    eps_prev = None
    iterations = 0
    for it in range(1, max_iter + 1):
        iterations = it
        # Step 1: solve the radial SE in the current effective potential.
        V = V_nuc + V_H + V_x
        eps, u = eigenstate(V, r, h, e_lo=-0.6 * Z * Z, e_hi=-0.01,
                            n_scan=n_scan, tol=eig_tol)

        # Step 2: rebuild the FULL Hartree (= 2 x single-orbital) and exchange
        # from the current orbital, then form the total energy.
        V_H1, _, _ = hartree_potential(u, r, h)
        V_H_new = 2.0 * V_H1                      # full two-electron Hartree
        V_x_new = exchange_potential(u, r)        # Slater LDA exchange
        E_tot = (2.0 * eps
                 - integrate(V_H_new * u * u, h)
                 - 0.5 * integrate(u * u * V_x_new, h))

        d_eps = float("nan") if eps_prev is None else abs(eps - eps_prev)
        history.append((eps, E_tot))
        if verbose:
            print(f"{it:>4}  {eps:>12.6f}  {E_tot:>14.6f}  {d_eps:>10.2e}")

        # Step 4: convergence test on the eigenvalue.
        if eps_prev is not None and d_eps < tol:
            break
        eps_prev = eps

        # Step 3: mix both potentials for the next iteration.
        V_H = (1.0 - w) * V_H + w * V_H_new
        V_x = (1.0 - w) * V_x + w * V_x_new

    return {
        "E": E_tot,
        "eps": eps,
        "u": u,
        "V_H": V_H_new,
        "V_x": V_x_new,
        "r": r,
        "iterations": iterations,
        "history": history,
    }


# ===========================================================================
# Full LDA: add Ceperley-Alder / Perdew-Zunger correlation on top of the
# Slater-exchange SCF loop (Z=2).
# ===========================================================================
# Unpolarised (spin-paired) Perdew-Zunger parameters for the CA correlation
# energy of the uniform electron gas (PDF table, p.5 / spec Eq. C).
_PZ_A, _PZ_B, _PZ_C, _PZ_D = 0.0311, -0.048, 0.0020, -0.0116
_PZ_GAMMA, _PZ_BETA1, _PZ_BETA2 = -0.1423, 1.0529, 0.3334


def correlation_pz(u, r):
    """Perdew-Zunger LDA correlation: return (V_c, e_c) on the grid.

    Correlation is the last piece of the quantum correction: electrons avoid one
    another a little more than plain electrostatics (Hartree) and exchange
    account for. LDA again borrows the uniform-electron-gas result and applies it
    pointwise, as a function of the local density through the Wigner-Seitz radius
        r_s = (3 / (4 pi n))^{1/3},   n(r) = u^2 / (2 pi r^2)  (full density).
    Small r_s = dense, large r_s = dilute.

    The CA energy is parametrised in two regimes (the branch boundary r_s = 1 is
    the known pitfall -- a wrong cutoff gives a small but visible energy error):

      r_s >= 1 (low density):
        e_c = gamma / (1 + beta1 sqrt(r_s) + beta2 r_s)
        V_c = e_c (1 + (7/6) beta1 sqrt(r_s) + (4/3) beta2 r_s)
                  / (1 + beta1 sqrt(r_s) + beta2 r_s)
      r_s < 1 (high density):
        e_c = A ln r_s + B + C r_s ln r_s + D r_s
        V_c = A ln r_s + B - A/3 + (2/3) C r_s ln r_s + (2D - C) r_s / 3

    TWO DISTINCT OBJECTS (do not mix them up):
      * e_c is the correlation energy PER ELECTRON (used in the total energy).
      * V_c = d(n e_c)/dn is the correlation POTENTIAL (used inside the SCF SE).

    Where the density is negligible (u -> 0 at large r) r_s -> infinity; there
    both e_c and V_c tend to 0, so we simply leave those points at 0 and only
    evaluate the formulas where n is appreciable (avoids 0-division / log(inf)).
    """
    n = u * u / (2.0 * np.pi * r * r)        # full two-electron density
    V_c = np.zeros_like(r)
    e_c = np.zeros_like(r)

    # Only evaluate where there is real density; elsewhere e_c = V_c = 0.
    has_n = n > 1e-30
    r_s = np.full_like(r, np.inf)
    r_s[has_n] = (3.0 / (4.0 * np.pi * n[has_n]))**(1.0 / 3.0)

    # --- Branch 1: r_s >= 1 (and finite) --------------------------------------
    lo = has_n & (r_s >= 1.0)
    sq = np.sqrt(r_s[lo])
    denom = 1.0 + _PZ_BETA1 * sq + _PZ_BETA2 * r_s[lo]
    e_c[lo] = _PZ_GAMMA / denom
    V_c[lo] = e_c[lo] * (1.0 + (7.0 / 6.0) * _PZ_BETA1 * sq
                         + (4.0 / 3.0) * _PZ_BETA2 * r_s[lo]) / denom

    # --- Branch 2: r_s < 1 ----------------------------------------------------
    hi = has_n & (r_s < 1.0)
    ln = np.log(r_s[hi])
    e_c[hi] = (_PZ_A * ln + _PZ_B + _PZ_C * r_s[hi] * ln + _PZ_D * r_s[hi])
    V_c[hi] = (_PZ_A * ln + _PZ_B - _PZ_A / 3.0
               + (2.0 / 3.0) * _PZ_C * r_s[hi] * ln
               + (2.0 * _PZ_D - _PZ_C) * r_s[hi] / 3.0)

    return V_c, e_c


def scf_lda_xc(Z=2.0, r=None, h=H_STEP, w=0.3, tol=1e-7, max_iter=300,
               n_scan=120, eig_tol=1e-9, verbose=True):
    """Self-consistent helium with full Hartree + Slater exchange + PZ
    correlation -- the full LDA, and the headline number.

    A SEPARATE SCF variant: the Hartree-only (scf_no_xc) and exchange-only
    (scf_lda_x) paths are untouched, so their -2.86 and -2.72 results keep
    passing. This just adds the correlation potential V_c on top of the
    Slater-exchange setup:
        V(r) = -Z/r + V_H_full(r) + V_x(r) + V_c(r).

    Total energy: the exchange-only energy form plus the correlation
    contribution. The
    eigenvalue already contains integral V_c n (V_c is in the SE), so to recover
    the correlation ENERGY integral e_c n we add integral (e_c - V_c) n. In the
    radial convention integral f n d^3r = 2 integral f u^2 dr, so:
        E = 2 eps - integral V_H_full u^2 dr
                  - (1/2) integral u^2 V_x dr
                  + 2 integral (e_c - V_c) u^2 dr.
    Note e_c (energy per electron) and V_c (potential) are different objects: the
    SCF uses V_c, the energy correction uses (e_c - V_c).

    Correlation is a small, attractive correction; it deepens the exchange-only
    result from about -2.72 toward -2.83 (closer to the exact -2.90).

    Initialisation ramps all of V_H, V_x, V_c in from zero with mixing w ~ 0.3,
    exactly as in the exchange-only solver.

    Returns: dict with E, eps, u, V_H, V_x, V_c, r, iterations, history.
    """
    if r is None:
        r = make_grid(h)
    V_nuc = -Z / r

    # Ramp every interaction term in from zero for a stable start (same reason
    # as the Slater-exchange solver: the full Hartree from a compact starting
    # orbital is repulsive enough to unbind the electron if applied all at once).
    V_H = np.zeros_like(r)
    V_x = np.zeros_like(r)
    V_c = np.zeros_like(r)

    if verbose:
        print(f"{'iter':>4}  {'eps (Ha)':>12}  {'E_total (Ha)':>14}  "
              f"{'d eps':>10}")

    history = []
    eps_prev = None
    iterations = 0
    for it in range(1, max_iter + 1):
        iterations = it
        # Step 1: solve the radial SE in the current effective potential. After
        # the first iteration we warm-start the eigenvalue search at the previous
        # eps (the eigenvalue barely moves between iterations), which brackets
        # locally instead of rescanning the whole window -- same root, far faster.
        V = V_nuc + V_H + V_x + V_c
        eps, u = eigenstate(V, r, h, e_lo=-0.6 * Z * Z, e_hi=-0.01,
                            n_scan=n_scan, tol=eig_tol, e_guess=eps_prev)

        # Step 2: rebuild full Hartree, exchange, and correlation from the new
        # orbital, then form the total energy.
        V_H1, _, _ = hartree_potential(u, r, h)
        V_H_new = 2.0 * V_H1                       # full two-electron Hartree
        V_x_new = exchange_potential(u, r)         # Slater LDA exchange
        V_c_new, e_c_new = correlation_pz(u, r)    # PZ LDA correlation
        E_tot = (2.0 * eps
                 - integrate(V_H_new * u * u, h)
                 - 0.5 * integrate(u * u * V_x_new, h)
                 + 2.0 * integrate((e_c_new - V_c_new) * u * u, h))

        d_eps = float("nan") if eps_prev is None else abs(eps - eps_prev)
        history.append((eps, E_tot))
        if verbose:
            print(f"{it:>4}  {eps:>12.6f}  {E_tot:>14.6f}  {d_eps:>10.2e}")

        # Step 4: convergence test on the eigenvalue.
        if eps_prev is not None and d_eps < tol:
            break
        eps_prev = eps

        # Step 3: mix all three potentials for the next iteration.
        V_H = (1.0 - w) * V_H + w * V_H_new
        V_x = (1.0 - w) * V_x + w * V_x_new
        V_c = (1.0 - w) * V_c + w * V_c_new

    return {
        "E": E_tot,
        "eps": eps,
        "u": u,
        "V_H": V_H_new,
        "V_x": V_x_new,
        "V_c": V_c_new,
        "r": r,
        "iterations": iterations,
        "history": history,
    }
