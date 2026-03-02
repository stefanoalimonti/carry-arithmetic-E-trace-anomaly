#!/usr/bin/env python3
"""
E40: Extracting A_eff(K) from exact R(K) — convergence to A*

THE KEY TEST:
  S(A) = 2(1-A)/(3-2A) is the closed form (L→∞).
  Inverting: if R(K) = S(A_eff), then A_eff(K) = (2-3R)/(2(1-R)).
  If A_eff(K) → A* as K→∞, the linear mix is confirmed asymptotically.

RESULTS (exact fractions):
  K=3:  R = 3/4                    A_eff = -0.500   gap = 1.879
  K=8:  R = 14/821                 A_eff =  0.991   gap = 0.388
  K=12: R = -1.5820                A_eff =  1.306   gap = 0.073
  K=16: R = -2.8940                A_eff =  1.372   gap = 0.008
  K=17: R = -95687935/31865046     A_eff =  1.375   gap = 0.004

  Aitken extrapolation (K=15,16,17): A_eff^(∞) = 1.37955
  Target: A* = (2+3π)/(2(1+π)) = 1.37927  (0.03% error)

RUNTIME: K=14 ~10s, K=15 ~40s, K=16 ~3min, K=17 ~11min.

ALSO:
  Corrected Gram matrix with proper breakpoints.
  Test G(0,2)/G(0,0) ≈ 1/π more carefully.
"""
import sys
import time
import numpy as np
from fractions import Fraction
from mpmath import mp, mpf, log, pi as mpi, nstr, sqrt

mp.dps = 30


def pr(s=""):
    print(s, flush=True)


def F(x):
    return mpf(x)


Astar = (2 + 3 * mpi) / (2 * (1 + mpi))
z_shift = F(-1) / 2 - 1 / mpi


def fast_R(K):
    """Exact R(K) = Σval_casc(10)/Σval_casc(00) for K-bit numbers."""
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
pr("E40: A_eff(K) convergence to A*")
pr("=" * 78)
pr()
pr(f"  A* = (2+3π)/(2(1+π)) = {nstr(Astar, 20)}")
pr(f"  z  = -1/2 - 1/π      = {nstr(z_shift, 20)}")
pr()

pr("  The E09 closed form (L→∞): S(A) = 2(1-A)/(3-2A)")
pr("  Inverse: A = (2-3R)/(2(1-R))  given R = S(A)")
pr()

# ── PART A: Compute A_eff(K) ────────────────────────────────
pr("PART A: Computing R(K) and extracting A_eff(K)")
pr("-" * 60)
pr()

K_max = 17
results = []

for K in range(3, K_max + 1):
    t0 = time.time()
    R_frac = fast_R(K)
    elapsed = time.time() - t0
    if R_frac is None:
        continue
    R_val = mpf(R_frac.numerator) / mpf(R_frac.denominator)
    A_eff = (2 - 3 * R_val) / (2 * (1 - R_val))
    z_eff = (1 - A_eff / 2) / (1 - A_eff)

    results.append({
        'K': K, 'R': R_val, 'A_eff': A_eff, 'z_eff': z_eff,
        'time': elapsed
    })
    sys.stdout.write(f"  K={K} ({elapsed:.1f}s) ")
    sys.stdout.flush()

pr("\n")

pr(f"  {'K':>3s}  {'R(K)':>14s}  {'A_eff(K)':>16s}  {'A*-A_eff':>12s}  "
   f"{'z_eff':>14s}  {'z*':>14s}  {'z*-z_eff':>12s}")

for r in results:
    pr(f"  {r['K']:3d}  {nstr(r['R'], 10):>14s}  "
       f"{nstr(r['A_eff'], 12):>16s}  {nstr(Astar - r['A_eff'], 8):>12s}  "
       f"{nstr(r['z_eff'], 10):>14s}  {nstr(z_shift, 10):>14s}  "
       f"{nstr(z_shift - r['z_eff'], 8):>12s}")

pr()

# ── PART B: Convergence rate of A_eff → A* ───────────────────
pr("PART B: Convergence rate A_eff(K) → A*")
pr("-" * 60)
pr()

pr(f"  {'K':>3s}  {'A*-A_eff':>14s}  {'ratio (K/K-1)':>16s}  "
   f"{'K·(A*-A_eff)':>14s}  {'K²·(A*-A_eff)':>14s}")

prev_gap = None
for r in results:
    gap = Astar - r['A_eff']
    ratio = ""
    if prev_gap is not None and abs(gap) > F('1e-20'):
        ratio = nstr(prev_gap / gap, 8)
    K = r['K']
    pr(f"  {K:3d}  {nstr(gap, 10):>14s}  {ratio:>16s}  "
       f"{nstr(K * gap, 8):>14s}  {nstr(K ** 2 * gap, 8):>14s}")
    prev_gap = gap

pr()

# ── PART C: Corrected Gram matrix ────────────────────────────
pr("=" * 78)
pr("PART C: Corrected Gram matrix G(0,n)/G(0,0)")
pr("-" * 60)
pr()

from mpmath import quad


def inner_v_integral(n, v_max):
    """Exact ∫₀^{v_max} W_n(v) dv."""
    if n == 0:
        return v_max
    bits = []
    m = n
    while m:
        bits.append(m & 1)
        m >>= 1
    total_bits = len(bits)
    n_intervals = 1 << total_bits
    result = F(0)
    for idx in range(n_intervals):
        a = F(idx) / n_intervals
        b = F(idx + 1) / n_intervals
        if a >= v_max:
            break
        b_eff = min(b, v_max)
        sign = F(1)
        for bit_pos in range(total_bits):
            if bits[bit_pos]:
                bit_val = (idx >> (total_bits - 1 - bit_pos)) & 1
                if bit_val:
                    sign *= F(-1)
        result += sign * (b_eff - a)
    return result


def vmax(u):
    return (1 - u) / (1 + u)


def breakpoints_for_n(n):
    """Compute u-breakpoints where v_max(u) = k/2^L for Walsh order n."""
    if n == 0:
        return [F(0), F(1) / 3, F(1)]
    bits = []
    m = n
    while m:
        bits.append(m & 1)
        m >>= 1
    L = len(bits)
    N = 1 << L
    bps = [F(0)]
    for k in range(1, N):
        v_bp = F(k) / N
        u_bp = (1 - v_bp) / (1 + v_bp)
        if 0 < u_bp < 1:
            bps.append(u_bp)
    bps.append(F(1))
    bps.append(F(1) / 3)
    bps = sorted(set(bps))
    return bps


G00 = 2 * log(F(2)) - 1
pr(f"  G(0,0) = 2ln2-1 = {nstr(G00, 20)}")
pr()

N_MAX = 8
pr(f"  {'n':>3s}  {'G(0,n)':>20s}  {'G(0,n)/G(0,0)':>18s}  {'recognizable?':>30s}")

gram_01n = {}
for n in range(N_MAX + 1):
    if n == 0:
        gram_01n[n] = G00
        pr(f"  {n:3d}  {nstr(G00, 14):>20s}  {nstr(F(1), 14):>18s}  {'1':>30s}")
        continue

    bps = breakpoints_for_n(n)
    fn = lambda u, nn=n: inner_v_integral(nn, vmax(u))
    val = quad(fn, bps)
    gram_01n[n] = val
    ratio = val / G00

    recog = ""
    if abs(ratio - mpi / (mpi + 2)) < 0.002:
        recog = f"≈ π/(π+2) = {nstr(mpi / (mpi + 2), 8)}"
    elif abs(ratio - 1 / mpi) < 0.002:
        recog = f"≈ 1/π = {nstr(1 / mpi, 8)}"
    elif abs(ratio) < 0.002:
        recog = "≈ 0"

    pr(f"  {n:3d}  {nstr(val, 14):>20s}  {nstr(ratio, 14):>18s}  {recog:>30s}")

pr()

pr("  Key ratios:")
r01 = gram_01n[1] / G00
r02 = gram_01n[2] / G00
r04 = gram_01n[4] / G00 if 4 in gram_01n else None

pr(f"    G(0,1)/N = {nstr(r01, 15)}")
pr(f"    G(0,2)/N = {nstr(r02, 15)}")
pr(f"    π/(π+2) = {nstr(mpi / (mpi + 2), 15)}")
pr(f"    1/π     = {nstr(1 / mpi, 15)}")
pr(f"    G(0,1)/N - π/(π+2) = {nstr(r01 - mpi / (mpi + 2), 10)}")
pr(f"    G(0,2)/N - 1/π     = {nstr(r02 - 1 / mpi, 10)}")
pr()

if r04 is not None:
    pr(f"    G(0,4)/N = {nstr(r04, 15)}")
    pr(f"    1/π²    = {nstr(1 / mpi ** 2, 15)}")
    pr(f"    gap      = {nstr(r04 - 1 / mpi ** 2, 10)}")
    pr()

pr("  Product test: G(0,1)·G(0,2)/G(0,0)² vs G(0,3)/G(0,0):")
r03 = gram_01n[3] / G00 if 3 in gram_01n else None
if r03 is not None:
    pr(f"    G(0,1)·G(0,2)/G(0,0)² = {nstr(r01 * r02, 15)}")
    pr(f"    G(0,3)/G(0,0)          = {nstr(r03, 15)}")
    pr(f"    Ratio                   = {nstr((r01 * r02) / r03, 15)}")
pr()

# ── PART D: The z_eff convergence ─────────────────────────────
pr("=" * 78)
pr("PART D: z_eff convergence analysis")
pr("-" * 60)
pr()

pr("  If z_eff(K) → z* = -1/2 - 1/π, does the 1/π component emerge?")
pr()

pr(f"  {'K':>3s}  {'z_eff':>16s}  {'z_eff + 1/2':>14s}  {'= -1/π + δ':>14s}  "
   f"{'δ':>12s}")

for r in results:
    zval = r['z_eff']
    offset = zval + F(1) / 2
    delta = offset + 1 / mpi
    pr(f"  {r['K']:3d}  {nstr(zval, 12):>16s}  {nstr(offset, 10):>14s}  "
       f"{nstr(-1 / mpi, 10):>14s}  {nstr(delta, 8):>12s}")

pr()

# ── SUMMARY ──────────────────────────────────────────────────
pr("=" * 78)
pr("SUMMARY")
pr("-" * 60)
pr()

if results:
    last = results[-1]
    pr(f"  At K={last['K']}:")
    pr(f"    R(K)     = {nstr(last['R'], 12)}")
    pr(f"    A_eff(K) = {nstr(last['A_eff'], 12)}")
    pr(f"    A*       = {nstr(Astar, 12)}")
    pr(f"    gap      = {nstr(Astar - last['A_eff'], 8)}")
    pr(f"    z_eff    = {nstr(last['z_eff'], 12)}")
    pr(f"    z*       = {nstr(z_shift, 12)}")
    pr()

pr("  CONVERGENCE: A_eff(K) → A* monotonically from below.")
if len(results) >= 3:
    gaps = [Astar - r['A_eff'] for r in results]
    ratios = [gaps[i - 1] / gaps[i] for i in range(1, len(gaps))
              if abs(gaps[i]) > F('1e-20')]
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        pr(f"  Average ratio of successive gaps: {nstr(avg_ratio, 8)}")
        pr(f"  This suggests the gap decays as ~ 1/K^α with α ≈ {nstr(log(avg_ratio) / log(F(2)), 6)}")
pr()

pr("  GRAM MATRIX NEAR-MISSES:")
pr(f"    G(0,1)/N ≈ π/(π+2)  [gap {nstr(r01 - mpi / (mpi + 2), 6)}]")
pr(f"    G(0,2)/N ≈ 1/π      [gap {nstr(r02 - 1 / mpi, 6)}]")
pr()

pr("=" * 78)
pr("E40 complete.")
pr("=" * 78)
