#!/usr/bin/env python3
"""E221: Precision matrix and spatial Markov test for D-odd carries.

CORE TEST: Is Cov(carry)^{-1} approximately tridiagonal?

For a spatially Markov process (like the Holte bridge), the precision matrix
Ω = Cov^{-1} is tridiagonal: Ω[d,d'] = 0 for |d-d'| > 1.
This means carry[d] ⊥ carry[d+k] | carry[d±1] for k > 1.

For multiplication carries (bit-sharing), deviations from tridiagonal
measure the RANGE of spatial correlations. If Ω is approximately tridiagonal,
the LMH path-graph model is justified from below.

ANALYSIS:
  A. Precision matrix bandwidth — tridiagonal ratio
  B. Normalized precision: Ω[d,d']/max vs distance |d-d'| (decay profile)
  C. Mean carry profile: D-odd vs unconditioned, sine projection
  D. Excess profile from D-odd conditioning → shape analysis

Usage: python3 E221_precision_matrix.py [K_max]
"""

import numpy as np
import sys
import time
from math import pi, sin, cos


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


def holte_bridge_covariance(D):
    """Analytical covariance for the Holte (addition) carry bridge."""
    KM = np.array([[3/4, 1/4], [1/4, 3/4]])

    def h(c, d):
        return 0.5 + ((-1)**c) * 2.0**(-(D - d + 1))

    def forward(c, d):
        return 0.5 + ((-1)**c) * 2.0**(-(d + 1))

    def joint_bridge(c1, d1, c2, d2):
        f_c1 = forward(c1, d1)
        mid = np.linalg.matrix_power(KM, d2 - d1)[c1, c2]
        h_c2 = h(c2, d2)
        return f_c1 * mid * h_c2 / h(0, 0)

    def bridge_prob(c, d):
        return forward(c, d) * h(c, d) / h(0, 0)

    mean = np.zeros(D + 1)
    cov = np.zeros((D + 1, D + 1))
    for d in range(D + 1):
        mean[d] = bridge_prob(1, d)
    for d1 in range(D + 1):
        for d2 in range(d1, D + 1):
            e_cc = joint_bridge(1, d1, 1, d2) if d1 != d2 else bridge_prob(1, d1)
            cov[d1, d2] = e_cc - mean[d1] * mean[d2]
            cov[d2, d1] = cov[d1, d2]
    return mean, cov


def tridiag_ratio(M):
    """Fraction of Frobenius norm in the tridiagonal band."""
    n = M.shape[0]
    tri_norm_sq = 0.0
    full_norm_sq = 0.0
    for i in range(n):
        for j in range(n):
            v = M[i, j]**2
            full_norm_sq += v
            if abs(i - j) <= 1:
                tri_norm_sq += v
    if full_norm_sq < 1e-30:
        return 1.0
    return np.sqrt(tri_norm_sq / full_norm_sq)


def band_decay(M):
    """Average |M[i,j]| as a function of distance |i-j|, normalized by diagonal."""
    n = M.shape[0]
    max_dist = n - 1
    sums = np.zeros(max_dist + 1)
    counts = np.zeros(max_dist + 1)
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            sums[d] += abs(M[i, j])
            counts[d] += 1
    avg = sums / np.maximum(counts, 1)
    if avg[0] > 1e-30:
        avg /= avg[0]
    return avg


def sine_projection(profile, L_eff):
    """Project a profile (length L_eff) onto the sine basis sin(nπd/(L+1))."""
    coeffs = np.zeros(L_eff)
    for n in range(L_eff):
        phi = np.array([sin((n + 1) * pi * (d + 1) / (L_eff + 1)) for d in range(L_eff)])
        phi /= np.linalg.norm(phi)
        coeffs[n] = np.dot(profile, phi)
    return coeffs


def run_one_K(K, verbose=True):
    D = 2 * K - 1
    lo = 1 << (K - 1)
    hi = 1 << K
    L_eff = D - 1

    t0 = time.time()

    n_dodd = 0
    n_total = 0
    sum_c_dodd = np.zeros(D + 1)
    sum_cc_dodd = np.zeros((D + 1, D + 1))
    sum_c_all = np.zeros(D + 1)

    for p in range(lo, hi):
        for q in range(lo, hi):
            prod = p * q
            carries = compute_carry_chain(p, q, K)
            c_arr = np.array(carries, dtype=np.float64)

            n_total += 1
            sum_c_all += c_arr

            if not ((prod >> (D - 1)) & 1) or ((prod >> D) & 1):
                continue

            n_dodd += 1
            sum_c_dodd += c_arr
            sum_cc_dodd += np.outer(c_arr, c_arr)

    elapsed = time.time() - t0

    mean_all = sum_c_all / n_total
    mean_dodd = sum_c_dodd / n_dodd
    cov_dodd = sum_cc_dodd / n_dodd - np.outer(mean_dodd, mean_dodd)
    excess = mean_dodd - mean_all

    cov_int = cov_dodd[1:D, 1:D]

    eigvals = np.linalg.eigvalsh(cov_int)
    min_eig = eigvals.min()
    if min_eig < 1e-12:
        cov_int_reg = cov_int + np.eye(L_eff) * max(1e-12, -min_eig + 1e-12)
    else:
        cov_int_reg = cov_int
    precision = np.linalg.inv(cov_int_reg)

    holte_mean, holte_cov = holte_bridge_covariance(D)
    holte_cov_int = holte_cov[1:D, 1:D]
    holte_prec = np.linalg.inv(holte_cov_int)

    tri_mult = tridiag_ratio(precision)
    tri_holte = tridiag_ratio(holte_prec)

    decay_mult = band_decay(precision)
    decay_holte = band_decay(holte_prec)

    excess_int = excess[1:D]
    excess_norm = excess_int / (np.linalg.norm(excess_int) + 1e-30)
    excess_coeffs = sine_projection(excess_norm, L_eff)
    mean_dodd_int = mean_dodd[1:D]
    mean_dodd_norm = mean_dodd_int / (np.linalg.norm(mean_dodd_int) + 1e-30)
    mean_coeffs = sine_projection(mean_dodd_norm, L_eff)

    if verbose:
        print(f"\n{'='*72}")
        print(f"K = {K}   D = {D}   L_eff = {L_eff}   D-odd = {n_dodd}/{n_total}")
        print(f"Elapsed: {elapsed:.1f}s")

        print(f"\n--- A: Precision matrix tridiagonal ratio ---")
        print(f"  Multiplication: {tri_mult:.6f}")
        print(f"  Holte bridge:   {tri_holte:.6f}")
        print(f"  (1.0 = perfectly tridiagonal)")

        print(f"\n--- B: Precision off-diagonal decay (normalized by diagonal) ---")
        n_show = min(8, len(decay_mult))
        print(f"  {'|d-d|':>6s}  {'mult':>10s}  {'holte':>10s}")
        for k in range(n_show):
            print(f"  {k:6d}  {decay_mult[k]:10.6f}  {decay_holte[k]:10.6f}")

        print(f"\n--- C: Mean carry profile ---")
        print(f"  {'pos d':>6s}  {'uncond':>10s}  {'D-odd':>10s}  {'excess':>10s}  {'Holte':>10s}")
        step = max(1, L_eff // 10)
        for d in range(0, D + 1, step):
            mc_a = mean_all[d]
            mc_d = mean_dodd[d]
            ex = excess[d]
            hm = holte_mean[d]
            print(f"  {d:6d}  {mc_a:10.4f}  {mc_d:10.4f}  {ex:10.4f}  {hm:10.4f}")

        print(f"\n--- D: Sine projection of D-odd mean carry (normalized) ---")
        print(f"  {'n':>4s}  {'mean coeff':>12s}  {'excess coeff':>12s}")
        for n in range(min(8, L_eff)):
            print(f"  {n+1:4d}  {mean_coeffs[n]:12.6f}  {excess_coeffs[n]:12.6f}")
        mean_mode1 = abs(mean_coeffs[0])
        print(f"\n  Mode-1 dominance (D-odd mean): |c_1|/||c|| = {mean_mode1:.4f}")
        if mean_mode1 > 0.95:
            print(f"  *** D-odd carry profile IS a sine arch ***")
        elif mean_mode1 > 0.85:
            print(f"  *** Nearly sine, with corrections ***")
        else:
            print(f"  *** Not a simple sine ***")

        print(f"\n--- E: Precision matrix — largest off-tridiagonal entries ---")
        off_tri = []
        for i in range(L_eff):
            for j in range(L_eff):
                if abs(i - j) > 1:
                    off_tri.append((abs(precision[i, j]), i + 1, j + 1, precision[i, j]))
        off_tri.sort(reverse=True)
        print(f"  {'|Ω[i,j]|':>12s}  {'i':>4s}  {'j':>4s}  {'Ω[i,j]':>12s}  {'|i-j|':>6s}")
        for val, i, j, raw in off_tri[:8]:
            print(f"  {val:12.4f}  {i:4d}  {j:4d}  {raw:12.4f}  {abs(i-j):6d}")

    return {
        'K': K, 'D': D, 'n_dodd': n_dodd,
        'tri_mult': tri_mult, 'tri_holte': tri_holte,
        'decay_mult': decay_mult, 'decay_holte': decay_holte,
        'mean_mode1': abs(mean_coeffs[0]),
        'excess_coeffs': excess_coeffs,
        'mean_dodd': mean_dodd,
        'excess': excess,
    }


def main():
    K_max = int(sys.argv[1]) if len(sys.argv) > 1 else 11

    print("E221: PRECISION MATRIX AND SPATIAL MARKOV TEST")
    print("=" * 72)
    print(f"K range: 5 .. {K_max}")
    print(f"Question: Is Cov(carry)^{{-1}} tridiagonal (spatial Markov property)?")

    results = []
    for K in range(5, K_max + 1):
        r = run_one_K(K)
        results.append(r)

    print(f"\n\n{'='*72}")
    print("SUMMARY")
    print("=" * 72)
    print(f"\n  Tridiagonal ratio (1.0 = perfectly tridiagonal):")
    print(f"  {'K':>3s}  {'mult prec':>12s}  {'holte prec':>12s}  {'mean mode1':>12s}")
    for r in results:
        print(f"  {r['K']:3d}  {r['tri_mult']:12.6f}  {r['tri_holte']:12.6f}  {r['mean_mode1']:12.6f}")

    print(f"\n  Off-diagonal decay (distance 2 vs distance 1):")
    print(f"  {'K':>3s}  {'mult d=2/d=1':>14s}  {'holte d=2/d=1':>14s}")
    for r in results:
        if len(r['decay_mult']) > 2 and r['decay_mult'][1] > 1e-10:
            rm = r['decay_mult'][2] / r['decay_mult'][1]
        else:
            rm = 0
        if len(r['decay_holte']) > 2 and r['decay_holte'][1] > 1e-10:
            rh = r['decay_holte'][2] / r['decay_holte'][1]
        else:
            rh = 0
        print(f"  {r['K']:3d}  {rm:14.6f}  {rh:14.6f}")

    final = results[-1]
    print(f"\n\nVERDICT (K = {final['K']}):")
    if final['tri_mult'] > 0.99:
        print(f"  PRECISION IS TRIDIAGONAL (ratio = {final['tri_mult']:.4f})")
        print(f"  → Carry chain IS spatially Markov under D-odd conditioning!")
        print(f"  → LMH path-graph model VALIDATED from below.")
    elif final['tri_mult'] > 0.95:
        print(f"  PRECISION IS APPROXIMATELY TRIDIAGONAL (ratio = {final['tri_mult']:.4f})")
        print(f"  → Nearly spatially Markov, small long-range corrections.")
        print(f"  → LMH is a good effective description.")
    elif final['tri_mult'] > 0.85:
        print(f"  PRECISION HAS SIGNIFICANT OFF-TRIDIAGONAL ENTRIES (ratio = {final['tri_mult']:.4f})")
        print(f"  → Spatial Markov property is approximate.")
        print(f"  → LMH captures the dominant structure but misses corrections.")
    else:
        print(f"  PRECISION IS NOT TRIDIAGONAL (ratio = {final['tri_mult']:.4f})")
        print(f"  → Carries have long-range spatial correlations.")
        print(f"  → The path-graph model is not a good description.")

    if final['mean_mode1'] > 0.95:
        print(f"\n  CARRY PROFILE IS A SINE ARCH (mode-1 = {final['mean_mode1']:.4f})")
    elif final['mean_mode1'] > 0.85:
        print(f"\n  CARRY PROFILE IS APPROXIMATELY SINE (mode-1 = {final['mean_mode1']:.4f})")
    else:
        print(f"\n  CARRY PROFILE IS NOT A SIMPLE SINE (mode-1 = {final['mean_mode1']:.4f})")


if __name__ == '__main__':
    main()
