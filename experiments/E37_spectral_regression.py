#!/usr/bin/env python3
"""
E37: Spectral Regression — The Linear Mix Test

THE KEY IDENTITY (from prior experiments + trig identity):
  λ_n(A) = (1-A)·λ_n^(0) + A/2
  where λ_n^(0) = cos(nπ/L)/2 are the Markov eigenvalues.

This means: K_eff = (1-A)·K_Markov + (A/2)·I

TESTS:
  Part A: Does S(A*, L) match exact R(K) at finite K?
          If so, the E37 spectral model IS the carry arithmetic model.
  Part B: Gram matrix of Walsh functions on Ω — origin of the A/2·I term.
  Part C: Derive A from the Gram matrix and compare with A*.
  Part D: The shifted resolvent z = (1-A*/2)/(1-A*).
"""
import sys
import time
import numpy as np
from fractions import Fraction
from collections import defaultdict
from mpmath import (mp, mpf, log, pi as mpi, nstr, sin, cos, sqrt,
                    atan, tan, quad, mpf as MP)

mp.dps = 30


def pr(s=""):
    print(s, flush=True)


def S_sum(A, L):
    """E37 spectral sum: S(A,L) = Σ_{n=1}^{L-1} sin(nπ/2)/(1-λ_n(A))."""
    A = mpf(A)
    total = mpf(0)
    for n in range(1, L):
        sn = sin(n * mpi / 2)
        if abs(sn) < mpf('1e-25'):
            continue
        lam = cos(n * mpi / L) / 2 + A * sin(n * mpi / (2 * L)) ** 2
        denom = 1 - lam
        if abs(denom) < mpf('1e-35'):
            continue
        total += sn / denom
    return total


def S_formula(A):
    """Closed form: S(A) = 2(1-A)/(3-2A) as L→∞."""
    return 2 * (1 - mpf(A)) / (3 - 2 * mpf(A))


def fast_R(K):
    """Compute exact R(K) = Σval_casc(10)/Σval_casc(00) for K-bit numbers."""
    D = 2 * K - 1
    lo = 1 << (K - 1)
    hi = 1 << K

    sums = {'00': 0, '10': 0}

    Y_all = np.arange(lo, hi, dtype=np.int64)
    y_bits = np.zeros((len(Y_all), K), dtype=np.int8)
    for i in range(K):
        y_bits[:, i] = (Y_all >> i) & 1
    b_bits = (Y_all >> (K - 2)) & 1

    for X in range(lo, hi):
        a = (X >> (K - 2)) & 1
        xb = [(X >> i) & 1 for i in range(K)]

        P = np.int64(X) * Y_all
        dodd = (P >> (D - 1) == 1) & (P >> D == 0)
        idx = np.where(dodd)[0]
        if len(idx) == 0:
            continue

        yb = y_bits[idx]
        b_sec = b_bits[idx]
        n = len(idx)

        c = np.zeros(n, dtype=np.int32)
        carries_arr = np.zeros((D + 1, n), dtype=np.int32)

        for j in range(D):
            conv = np.zeros(n, dtype=np.int32)
            i_lo = max(0, j - K + 1)
            i_hi = min(j, K - 1)
            for i in range(i_lo, i_hi + 1):
                conv += xb[i] * yb[:, j - i].astype(np.int32)
            c = (conv + c) >> 1
            carries_arr[j + 1] = c

        M_vals = np.full(n, -1, dtype=np.int32)
        for j in range(D - 1, 0, -1):
            update = (carries_arr[j] > 0) & (M_vals == -1)
            M_vals[update] = j

        val_c = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            m = M_vals[i]
            if m > 0:
                val_c[i] = carries_arr[m - 1, i] - 1

        for b_val in [0]:
            mask = b_sec == b_val
            if not np.any(mask):
                continue
            sums[f"{a}{b_val}"] += int(np.sum(val_c[mask]))

    if sums['00'] == 0:
        return None
    return Fraction(sums['10'], sums['00'])


# ═══════════════════════════════════════════════════════════════════
pr("=" * 78)
pr("E37: Spectral Regression — The Linear Mix Test")
pr("=" * 78)
pr()

Astar = (2 + 3 * mpi) / (2 * (1 + mpi))
pr(f"  A* = (2+3π)/(2(1+π)) = {nstr(Astar, 20)}")
pr(f"  1-A* = {nstr(1 - Astar, 20)}")
pr(f"  A*/2 = {nstr(Astar / 2, 20)}")
pr()

pr("  LINEAR MIX IDENTITY:")
pr("    λ_n(A) = (1-A)·cos(nπ/L)/2 + A/2")
pr("    K_eff = (1-A)·K_Markov + (A/2)·I")
pr()
pr("  At A=A*: dominant eigenvalue (n→0) = 1/2 for ALL A.")
pr(f"  Shifted resolvent point: z = (1-A*/2)/(1-A*) = {nstr((1 - Astar / 2) / (1 - Astar), 15)}")
pr()

# ── PART A: S(A*, L) vs exact R(K) ──────────────────────────
pr("=" * 78)
pr("PART A: S(A*, L) vs exact R(K) — the crucial comparison")
pr("-" * 60)
pr()

pr("  If the linear mix is EXACTLY the carry arithmetic model,")
pr("  then R(K) = S(A*, L) for some L = L(K).")
pr()
pr("  Candidates: L = D = 2K-1, or L = D-1 = 2K-2, or L = K, etc.")
pr()

pr("Computing exact R(K) and S(A*, L) for K=3..13...")
pr()

K_max = 13
results = []

for K in range(3, K_max + 1):
    t0 = time.time()
    R_frac = fast_R(K)
    elapsed = time.time() - t0
    if R_frac is None:
        continue

    R_val = mpf(R_frac.numerator) / mpf(R_frac.denominator)
    D = 2 * K - 1

    S_D = S_sum(Astar, D)
    S_Dm1 = S_sum(Astar, D - 1)
    S_Dp1 = S_sum(Astar, D + 1)
    S_K = S_sum(Astar, K)
    S_2K = S_sum(Astar, 2 * K)

    results.append({
        'K': K, 'D': D, 'R': R_val,
        'S_D': S_D, 'S_Dm1': S_Dm1, 'S_Dp1': S_Dp1,
        'S_K': S_K, 'S_2K': S_2K,
        'time': elapsed
    })
    sys.stdout.write(f"  K={K} ({elapsed:.1f}s) ")
    sys.stdout.flush()

pr("\n")

pr(f"  {'K':>3s}  {'R(K)':>16s}  {'S(A*,D)':>16s}  {'R-S(D)':>14s}  "
   f"{'S(A*,D-1)':>16s}  {'R-S(D-1)':>14s}  {'S(A*,K)':>16s}  {'R-S(K)':>14s}")

for r in results:
    pr(f"  {r['K']:3d}  {nstr(r['R'], 12):>16s}  "
       f"{nstr(r['S_D'], 12):>16s}  {nstr(r['R'] - r['S_D'], 8):>14s}  "
       f"{nstr(r['S_Dm1'], 12):>16s}  {nstr(r['R'] - r['S_Dm1'], 8):>14s}  "
       f"{nstr(r['S_K'], 12):>16s}  {nstr(r['R'] - r['S_K'], 8):>14s}")

pr()

pr("  Gap convergence — does R(K) - S(A*, L) → 0?")
pr()
pr(f"  {'K':>3s}  {'R(K)+π':>14s}  {'S(A*,D)+π':>14s}  {'S(A*,D-1)+π':>14s}  "
   f"{'|gap R-S(D)|':>14s}  {'ratio':>10s}")

prev_gap = None
for r in results:
    gap_R = r['R'] + mpi
    gap_SD = r['S_D'] + mpi
    gap_SDm1 = r['S_Dm1'] + mpi
    gap_RS = abs(r['R'] - r['S_D'])
    ratio_str = ""
    if prev_gap is not None and abs(gap_RS) > mpf('1e-25'):
        ratio_str = nstr(prev_gap / gap_RS, 6)
    prev_gap = gap_RS
    pr(f"  {r['K']:3d}  {nstr(gap_R, 8):>14s}  {nstr(gap_SD, 8):>14s}  "
       f"{nstr(gap_SDm1, 8):>14s}  {nstr(gap_RS, 8):>14s}  {ratio_str:>10s}")

pr()

# ── PART B: Gram matrix of Walsh functions on Ω ──────────────
pr("=" * 78)
pr("PART B: Gram matrix — Walsh function non-orthogonality on Ω")
pr("-" * 60)
pr()

pr("  On [0,1]²: ∫∫ W_i(u)·W_j(v) du dv = δ_{i0}·δ_{j0}")
pr("  On Ω = {v < (1-u)/(1+u)}: the boundary creates cross-terms.")
pr()

pr("  Rademacher functions: r_k(x) = (-1)^⌊2^k·x⌋ = 1-2·bit_k(x)")
pr("  Walsh functions: W_n(x) = Π_{k: bit_k(n)=1} r_k(x)")
pr()

def bit_j_float(x, j):
    """j-th binary digit of x ∈ [0,1)."""
    return int(np.floor(x * 2**j)) % 2

def rademacher(x, k):
    """r_k(x) = (-1)^{bit_k(x)}."""
    return 1 - 2 * bit_j_float(x, k)

def walsh(x, n):
    """W_n(x) = product of r_k(x) for bits set in n."""
    result = 1
    k = 0
    m = n
    while m > 0:
        if m & 1:
            result *= rademacher(x, k + 1)
        m >>= 1
        k += 1
    return result

pr("  Computing Gram matrix G_{m,n} = ∫∫_Ω W_m(u)·W_n(v) du dv")
pr("  for m, n = 0, 1, ..., 7")
pr()

N_WALSH = 8

def gram_element(m, n, npts=2000):
    """Compute ∫∫_Ω W_m(u)·W_n(v) du dv numerically."""
    total = mpf(0)
    du = mpf(1) / npts
    for i in range(npts):
        u = (mpf(i) + mpf('0.5')) / npts
        v_max = min(mpf('0.5'), (1 - u) / (1 + u))
        if v_max <= 0:
            continue
        wm_u = walsh(float(u), m)
        inner = mpf(0)
        nv = max(1, int(float(v_max * npts)))
        dv = v_max / nv
        for j in range(nv):
            v = (mpf(j) + mpf('0.5')) * dv
            wn_v = walsh(float(v), n)
            inner += wn_v * dv
        total += wm_u * inner * du
    return total

pr("  Using mpmath quadrature for precision...")
pr()

def gram_quad(m, n):
    """High-precision Gram element via mpmath quadrature."""
    alpha_s = atan(mpf('0.5'))

    def integrand_full(u):
        v_max = min(mpf('0.5'), (1 - u) / (1 + u))
        if v_max <= 0:
            return mpf(0)
        wm = mpf(walsh(float(u), m))

        bps = [mpf(0)]
        for k in range(1, 10):
            bp = mpf(k) / mpf(2 ** ((n.bit_length() if n > 0 else 1)))
            if 0 < bp < v_max:
                bps.append(bp)
        bps.append(v_max)
        bps = sorted(set(bps))

        inner = mpf(0)
        for idx in range(len(bps) - 1):
            va, vb = bps[idx], bps[idx + 1]
            if vb <= va:
                continue
            v_mid = (va + vb) / 2
            wn = mpf(walsh(float(v_mid), n))
            inner += wn * (vb - va)

        return wm * inner

    u_bps = [mpf(0)]
    for k in range(1, 10):
        bp = mpf(k) / mpf(2 ** ((m.bit_length() if m > 0 else 1)))
        if 0 < bp < 1:
            u_bps.append(bp)
    u_bps.append(mpf(1))
    u_bps = sorted(set(u_bps))

    total = mpf(0)
    for idx in range(len(u_bps) - 1):
        ua, ub = u_bps[idx], u_bps[idx + 1]
        if ub <= ua:
            continue
        total += quad(integrand_full, [ua, ub], error=True)[0]

    return total

pr("  Computing Gram matrix (numerical quadrature)...")
t0 = time.time()

G = {}
for m in range(N_WALSH):
    for n in range(N_WALSH):
        G[(m, n)] = gram_quad(m, n)

pr(f"  Done in {time.time()-t0:.1f}s")
pr()

N_area = G[(0, 0)]
pr(f"  G(0,0) = N_total = {nstr(N_area, 15)} (should be 2ln2-1 = {nstr(2*log(mpf(2))-1, 15)})")
pr()

pr(f"  Full Gram matrix G(m,n) for m,n = 0..{N_WALSH-1}:")
pr()
header = "     " + "".join(f"  n={n:2d}     " for n in range(N_WALSH))
pr(header)
for m in range(N_WALSH):
    row = f"m={m:2d} "
    for n in range(N_WALSH):
        row += f"{nstr(G[(m, n)], 6):>10s} "
    pr(row)

pr()

pr("  Normalized Gram matrix G(m,n)/G(0,0):")
pr()
header = "     " + "".join(f"  n={n:2d}     " for n in range(min(N_WALSH, 5)))
pr(header)
for m in range(min(N_WALSH, 5)):
    row = f"m={m:2d} "
    for n in range(min(N_WALSH, 5)):
        val = G[(m, n)] / N_area if abs(N_area) > mpf('1e-20') else mpf(0)
        row += f"{nstr(val, 6):>10s} "
    pr(row)

pr()

# ── PART C: Extract A from Gram matrix ─────────────────────────
pr("=" * 78)
pr("PART C: Extracting the correlation parameter A from the Gram matrix")
pr("-" * 60)
pr()

pr("  On the full [0,1]², the 1-step correlation is 0 (Markov).")
pr("  On Ω, the 1-step correlation C₁ = G(1,1)/G(0,0) is:")
if abs(N_area) > mpf('1e-20'):
    C1 = G[(1, 1)] / N_area
    pr(f"    C₁ = {nstr(C1, 15)}")
    pr()

    C1_sector00 = mpf(0)
    C1_sector10 = mpf(0)

    pr("  The sector-specific correlations:")
    pr("    Sector (0,0): u ∈ [0, 1/2), v ∈ [0, 1/2)")
    pr("    Sector (1,0): u ∈ [1/2, 1), v ∈ [0, 1/2)")
    pr()

    pr("  The A parameter should satisfy:")
    pr("    A/2 = correlation shift from Markov to non-Markov")
    pr()

    pr(f"  If A = 2·C₁: A_from_Gram = {nstr(2 * C1, 15)}")
    pr(f"  A* = {nstr(Astar, 15)}")
    pr(f"  Difference: {nstr(2 * C1 - Astar, 12)}")
    pr()

    pr("  Off-diagonal elements G(i,j)/G(0,0) for i≠j:")
    for i in range(min(4, N_WALSH)):
        for j in range(i + 1, min(4, N_WALSH)):
            val = G[(i, j)] / N_area
            pr(f"    G({i},{j})/N = {nstr(val, 12)}")

pr()

# ── PART D: The shifted resolvent ──────────────────────────────
pr("=" * 78)
pr("PART D: The shifted resolvent z = (1-A*/2)/(1-A*)")
pr("-" * 60)
pr()

z_shift = (1 - Astar / 2) / (1 - Astar)
pr(f"  z = (1-A*/2)/(1-A*) = {nstr(z_shift, 20)}")
pr(f"  z = {nstr(z_shift, 8)} (should this be recognizable?)")
pr()

pr("  The resolvent identity:")
pr("    (I - K_eff)^{-1} = (I - (1-A)K_M - (A/2)I)^{-1}")
pr("                     = ((1-A/2)I - (1-A)K_M)^{-1}")
pr("                     = 1/(1-A) · (z·I - K_M)^{-1}")
pr(f"    where z = (1-A/2)/(1-A) = {nstr(z_shift, 12)}")
pr()

pr("  So: ΔR = Tr[W · (I-K_eff)^{-1}]")
pr("        = 1/(1-A*) · Tr[W · (z·I-K_M)^{-1}]")
pr()

pr("  The Markov resolvent (z·I-K_M)^{-1} has eigenvalues:")
pr("    1/(z - λ_n^(0)) = 1/(z - cos(nπ/L)/2)")
pr()

pr("  The trace in the L→∞ limit becomes an INTEGRAL:")
pr("    Tr = Σ_n sin(nπ/2)/(z - cos(nπ/L)/2)")
pr("        → ∫_0^π sin(θ/2)/(z - cos(θ)/2) dθ/π  (Riemann sum)")
pr()

I_res = quad(lambda t: sin(t / 2) / (z_shift - cos(t) / 2),
             [0, mpi], error=True)[0] / mpi
pr(f"  Numerical integral: {nstr(I_res, 15)}")
pr(f"  1/(1-A*) · integral = {nstr(I_res / (1 - Astar), 15)}")
pr(f"  S(A*) formula = {nstr(S_formula(Astar), 15)}")
pr(f"  -π = {nstr(-mpi, 15)}")
pr()

pr("  CONSISTENCY CHECK:")
S_check = S_formula(Astar)
pr(f"    S(A*) = 2(1-A*)/(3-2A*) = {nstr(S_check, 20)}")
pr(f"    -π =                       {nstr(-mpi, 20)}")
pr(f"    Match: {abs(S_check + mpi) < mpf('1e-20')}")
pr()

# ── PART E: Summary ──────────────────────────────────────────
pr("=" * 78)
pr("PART E: Summary and implications")
pr("-" * 60)
pr()

if results:
    last = results[-1]
    pr(f"  At K={last['K']}:")
    pr(f"    R(K)    = {nstr(last['R'], 15)}")
    pr(f"    S(A*,D) = {nstr(last['S_D'], 15)}")
    pr(f"    gap R-S = {nstr(last['R'] - last['S_D'], 10)}")
    pr()

pr("  THE LINEAR MIX HYPOTHESIS:")
pr("    K_eff = (1-A*)·K_Markov + (A*/2)·I")
pr()
pr("  If R(K) ≠ S(A*, D):")
pr("    → The E09 spectral model with parametric eigenvalue shifts is an")
pr("      APPROXIMATION, not the exact carry arithmetic model.")
pr("    → The actual eigenvalue structure may be more complex.")
pr()
pr("  If R(K) ≈ S(A*, D) with matching convergence rate:")
pr("    → The linear mix is exact in the K→∞ limit.")
pr("    → Proving A = A* from the Gram matrix completes the proof.")
pr()

pr("=" * 78)
pr("E37 complete.")
pr("=" * 78)
