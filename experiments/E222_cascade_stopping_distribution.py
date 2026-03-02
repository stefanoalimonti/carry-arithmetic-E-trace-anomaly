#!/usr/bin/env python3
"""E222: Cascade stopping position and diffusive limit.

ATTACK 2: Does the cascade stopping rule produce a distribution
connected to π through the diffusive (continuum) limit?

For each D-odd pair, the cascade stopping position is:
  M = max{m <= L : carry[m] > 0}  (scanning from top)
If no carry is nonzero, the pair is excluded.

TESTS:
A. Distribution of M/L (normalized stopping position)
   → Does it converge to a known distribution as K → ∞?
   → Arc-sine? Beta? Related to Brownian bridge last-excursion?

B. K-1 coupling: spectral analysis in sine basis
   → Construct the "reflection operator" T_R that couples d to d+K-1
   → Project onto sine basis: is it off-diagonal?
   → If diagonal elements ≈ 0, first-order perturbation vanishes

C. Joint (M, val) analysis by sector
   → How do cascade contributions depend on stopping position?
   → Does the sector ratio R(K) have a position-dependent structure?

D. Diffusive scaling
   → Does the carry profile scale as √K (diffusive) or K (ballistic)?
   → Does the stopping position distribution sharpen as K → ∞?

Usage: python3 E222_cascade_stopping_distribution.py [K_max]
"""

import numpy as np
import sys
import time
from math import pi, sin, cos, sqrt, log


def compute_carry_chain(p, q, K):
    D = 2 * K - 1
    carries = [0] * (D + 1)
    for j in range(D):
        cv = 0
        i_lo = max(0, j - K + 1)
        i_hi = min(j, K - 1)
        for i in range(i_lo, i_hi + 1):
            cv += ((p >> i) & 1) * ((q >> (j - i)) & 1)
        carries[j + 1] = (cv + int(carries[j])) >> 1
    return carries


def cascade_val_and_pos(carries, D):
    """Find cascade stopping position M and val.
    M = max{m: carry[m]>0}, val = carry[M-1] - 1.
    Returns (M, val) or (None, None) if no nonzero carry."""
    for m in range(D, 0, -1):
        if carries[m] > 0:
            return m, carries[m - 1] - 1
    return None, None


def reflection_operator_sine_basis(K):
    """Compute the K-1 reflection coupling in the sine basis.

    The "reflection operator" T_R connects position d to d+K-1:
      T_R[d, d+K-1] = 1 for d=1,...,K
      (and its transpose)

    We normalize by the number of positions and project onto sine basis.
    """
    L = 2 * K - 2
    L_plus_1 = L + 1

    def phi(n, d):
        return sin(n * pi * d / L_plus_1)

    norm_sq = L_plus_1 / 2.0

    M = np.zeros((L, L))
    for n in range(1, L + 1):
        for m in range(1, L + 1):
            val = 0.0
            for d in range(1, K + 1):
                d_prime = d + K - 1
                if 1 <= d_prime <= L:
                    val += phi(n, d) * phi(m, d_prime)
                    val += phi(n, d_prime) * phi(m, d)
            M[n - 1, m - 1] = val / norm_sq
    return M


def run_one_K(K, verbose=True):
    D = 2 * K - 1
    L = D - 1
    lo = 1 << (K - 1)
    hi = 1 << K

    t0 = time.time()

    n_dodd = 0
    n_cascade = 0
    hist_M = np.zeros(D + 1)
    sum_val_00 = np.zeros(D + 1)
    sum_val_10 = np.zeros(D + 1)
    count_00 = np.zeros(D + 1)
    count_10 = np.zeros(D + 1)
    max_carry_values = np.zeros(D + 1, dtype=int)

    for p in range(lo, hi):
        for q in range(lo, hi):
            prod = p * q
            if not ((prod >> (D - 1)) & 1) or ((prod >> D) & 1):
                continue

            n_dodd += 1
            carries = compute_carry_chain(p, q, K)

            M, val = cascade_val_and_pos(carries, D)
            if M is None:
                continue

            n_cascade += 1
            hist_M[M] += 1

            sector_a = (p >> (K - 1)) & 1
            sector_b = (q >> (K - 1)) & 1
            sec = sector_a * 10 + sector_b

            if sec == 0:
                sum_val_00[M] += val
                count_00[M] += 1
            elif sec == 10:
                sum_val_10[M] += val
                count_10[M] += 1

            mc = max(carries)
            if mc > max_carry_values[M]:
                max_carry_values[M] = mc

    elapsed = time.time() - t0

    dist_M = hist_M / max(n_cascade, 1)
    x_M = np.arange(D + 1) / D

    R_refl = reflection_operator_sine_basis(K)
    diag_R = np.diag(R_refl)

    if verbose:
        print(f"\n{'='*72}")
        print(f"K = {K}   D = {D}   L = {L}   D-odd = {n_dodd}   cascade = {n_cascade}")
        print(f"Elapsed: {elapsed:.1f}s")

        print(f"\n--- A: Cascade stopping position distribution ---")
        print(f"  Exclusion rate (no nonzero carry): {1 - n_cascade/max(n_dodd,1):.4f}")
        print(f"  {'M':>4s}  {'M/D':>6s}  {'P(M)':>10s}  {'cumul':>10s}")
        cumul = 0
        for m in range(D, 0, -1):
            if hist_M[m] > 0:
                cumul += dist_M[m]
                print(f"  {m:4d}  {m/D:6.3f}  {dist_M[m]:10.6f}  {cumul:10.6f}")
                if cumul > 0.999:
                    break

        print(f"\n  Statistics of M/D:")
        valid_M = []
        for m in range(1, D + 1):
            valid_M.extend([m / D] * int(hist_M[m]))
        valid_M = np.array(valid_M)
        if len(valid_M) > 0:
            print(f"    mean = {np.mean(valid_M):.4f}")
            print(f"    std  = {np.std(valid_M):.4f}")
            print(f"    median = {np.median(valid_M):.4f}")
            q25 = np.percentile(valid_M, 25)
            q75 = np.percentile(valid_M, 75)
            print(f"    Q25  = {q25:.4f}   Q75 = {q75:.4f}")

        print(f"\n--- B: Reflection operator T_R in sine basis ---")
        print(f"  L = {L}   T_R couples d to d+{K-1}")
        print(f"  {'mode n':>7s}  {'M[n,n]':>10s}  {'|M[n,n]|':>10s}  {'max|M[n,m!=n]|':>16s}")
        n_show = min(10, L)
        for n in range(n_show):
            off_diag = [abs(R_refl[n, m]) for m in range(L) if m != n]
            max_off = max(off_diag) if off_diag else 0
            print(f"  {n+1:7d}  {diag_R[n]:10.6f}  {abs(diag_R[n]):10.6f}  {max_off:16.6f}")

        diag_norm = np.linalg.norm(diag_R)
        off_diag_norm = np.sqrt(np.linalg.norm(R_refl, 'fro')**2 - diag_norm**2)
        print(f"\n  ||diag||  = {diag_norm:.6f}")
        print(f"  ||off-diag|| = {off_diag_norm:.6f}")
        print(f"  Ratio diag/total = {diag_norm / max(np.linalg.norm(R_refl, 'fro'), 1e-30):.6f}")
        if diag_norm < 0.1 * off_diag_norm:
            print(f"  *** T_R is PREDOMINANTLY OFF-DIAGONAL in sine basis ***")
            print(f"  → First-order perturbation to eigenvalues ≈ 0")
            print(f"  → Correction is SECOND ORDER in coupling strength")

        print(f"\n--- C: Sector-resolved cascade contributions by stopping position ---")
        print(f"  {'M':>4s}  {'n_00':>8s}  {'mean_v00':>10s}  {'n_10':>8s}  {'mean_v10':>10s}  {'local_R':>10s}")
        for m in range(D, 0, -1):
            if count_00[m] + count_10[m] > 0:
                mv00 = sum_val_00[m] / max(count_00[m], 1) if count_00[m] > 0 else float('nan')
                mv10 = sum_val_10[m] / max(count_10[m], 1) if count_10[m] > 0 else float('nan')
                lr = sum_val_10[m] / max(sum_val_00[m], 1e-30) if sum_val_00[m] != 0 else float('nan')
                print(f"  {m:4d}  {int(count_00[m]):8d}  {mv00:10.4f}  {int(count_10[m]):8d}  {mv10:10.4f}  {lr:10.4f}")
                if m < D - 8:
                    break

        total_v00 = sum_val_00.sum()
        total_v10 = sum_val_10.sum()
        R_K = total_v10 / max(total_v00, 1e-30) if total_v00 != 0 else float('nan')
        print(f"\n  Total R(K) = {R_K:.6f}   (target: -π = {-pi:.6f})")

        print(f"\n--- D: Diffusive scaling ---")
        mean_carry = sum([np.mean([carries for carries in [[0]]])
                         for _ in range(0)])  # placeholder
        carry_profile_peak = 0
        for m in range(D + 1):
            if max_carry_values[m] > carry_profile_peak:
                carry_profile_peak = max_carry_values[m]
        print(f"  Max carry value encountered: {carry_profile_peak}")
        print(f"  Expected for diffusive: O(√K) = {sqrt(K):.2f}")
        print(f"  Expected for ballistic: O(K) = {K}")

    return {
        'K': K, 'D': D, 'n_dodd': n_dodd, 'n_cascade': n_cascade,
        'dist_M': dist_M, 'x_M': x_M,
        'mean_M_over_D': np.mean(valid_M) if len(valid_M) > 0 else 0,
        'std_M_over_D': np.std(valid_M) if len(valid_M) > 0 else 0,
        'R_K': R_K if 'R_K' in dir() else 0,
        'R_refl': R_refl,
        'diag_R': diag_R,
    }


def main():
    K_max = int(sys.argv[1]) if len(sys.argv) > 1 else 11

    print("E222: CASCADE STOPPING DISTRIBUTION & DIFFUSIVE LIMIT")
    print("=" * 72)
    print(f"K range: 5 .. {K_max}")
    print(f"Target R(∞) = -π = {-pi:.8f}")

    results = []
    for K in range(5, K_max + 1):
        r = run_one_K(K)
        results.append(r)

    print(f"\n\n{'='*72}")
    print("SUMMARY")
    print("=" * 72)

    print(f"\n  Cascade stopping position statistics:")
    print(f"  {'K':>3s}  {'mean(M/D)':>10s}  {'std(M/D)':>10s}  {'excl_rate':>10s}  {'R(K)':>10s}")
    for r in results:
        excl = 1 - r['n_cascade'] / max(r['n_dodd'], 1)
        print(f"  {r['K']:3d}  {r['mean_M_over_D']:10.4f}  {r['std_M_over_D']:10.4f}  {excl:10.4f}  {r.get('R_K', 0):10.6f}")

    print(f"\n  Reflection operator diagonal (sine basis) — should be ≈0 if off-diagonal:")
    print(f"  {'K':>3s}  {'|M[1,1]|':>10s}  {'|M[2,2]|':>10s}  {'|M[3,3]|':>10s}  {'diag/total':>12s}")
    for r in results:
        d = r['diag_R']
        R = r['R_refl']
        L = r['D'] - 1
        diag_norm = np.linalg.norm(d)
        total_norm = np.linalg.norm(R, 'fro')
        ratio = diag_norm / max(total_norm, 1e-30)
        print(f"  {r['K']:3d}  {abs(d[0]):10.6f}  {abs(d[1]) if len(d)>1 else 0:10.6f}  {abs(d[2]) if len(d)>2 else 0:10.6f}  {ratio:12.6f}")

    print(f"\n  KEY TEST: Is T_R off-diagonal in sine basis?")
    final = results[-1]
    d = final['diag_R']
    R = final['R_refl']
    diag_norm = np.linalg.norm(d)
    total_norm = np.linalg.norm(R, 'fro')
    ratio = diag_norm / max(total_norm, 1e-30)
    if ratio < 0.1:
        print(f"  YES (diag/total = {ratio:.4f}) — first-order perturbation VANISHES")
        print(f"  → The bit-sharing correction to the LMH is SECOND ORDER")
        print(f"  → This explains why the path-graph model works so well")
    elif ratio < 0.3:
        print(f"  PARTIALLY (diag/total = {ratio:.4f}) — small first-order contribution")
    else:
        print(f"  NO (diag/total = {ratio:.4f}) — significant diagonal elements")
        print(f"  → First-order correction is non-negligible")


if __name__ == '__main__':
    main()
