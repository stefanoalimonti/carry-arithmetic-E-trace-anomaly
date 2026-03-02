#!/usr/bin/env python3
"""E221b: Partial correlation analysis — the CORRECT spatial Markov test.

The precision matrix Ω = Cov^{-1} has misleading raw entries because
boundary positions have near-zero variance → huge diagonal.

The PARTIAL CORRELATION:
  ρ_partial[i,j] = -Ω[i,j] / sqrt(Ω[i,i] * Ω[j,j])

is properly normalized ∈ [-1, 1] and gives the correct test:
  - For spatial Markov: ρ_partial[i,j] = 0 for |i-j| > 1
  - Non-zero partial corr at distance > 1 = long-range coupling

We compute this for both Holte bridge (should be exactly tridiagonal)
and multiplication D-odd (question: how much non-tridiagonal structure?).

Also: restrict analysis to BULK positions (d=3..D-3) to avoid
boundary artifacts.
"""

import numpy as np
import sys
import time
from math import pi, sin


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


def partial_corr_from_cov(C, reg=1e-10):
    """Compute partial correlation matrix from covariance."""
    n = C.shape[0]
    C_reg = C + reg * np.eye(n)
    Omega = np.linalg.inv(C_reg)
    P = np.zeros_like(Omega)
    for i in range(n):
        for j in range(n):
            denom = np.sqrt(abs(Omega[i, i] * Omega[j, j]))
            if denom > 1e-30:
                P[i, j] = -Omega[i, j] / denom
            P[i, i] = 1.0
    return P, Omega


def analyze_partial_corr(P, label, n_show_dist=10):
    """Analyze partial correlation decay with distance."""
    n = P.shape[0]
    print(f"\n  {label}")

    avg_abs_by_dist = {}
    max_abs_by_dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            v = abs(P[i, j])
            if d not in avg_abs_by_dist:
                avg_abs_by_dist[d] = []
                max_abs_by_dist[d] = 0.0
            avg_abs_by_dist[d].append(v)
            max_abs_by_dist[d] = max(max_abs_by_dist[d], v)

    print(f"  {'dist':>5s}  {'mean|ρ|':>10s}  {'max|ρ|':>10s}  {'n_pairs':>8s}")
    for d in range(1, min(n_show_dist + 1, n)):
        if d in avg_abs_by_dist:
            vals = avg_abs_by_dist[d]
            print(f"  {d:5d}  {np.mean(vals):10.6f}  {max_abs_by_dist[d]:10.6f}  {len(vals):8d}")

    tri_norm = 0.0
    full_off_norm = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            v = P[i, j]**2
            full_off_norm += v
            if j - i == 1:
                tri_norm += v
    if full_off_norm > 1e-30:
        tri_frac = np.sqrt(tri_norm / full_off_norm)
    else:
        tri_frac = 1.0
    print(f"  Tridiagonal fraction of off-diag: {tri_frac:.6f}")
    return tri_frac


def run_one_K(K, verbose=True):
    D = 2 * K - 1
    lo = 1 << (K - 1)
    hi = 1 << K

    t0 = time.time()
    n_dodd = 0
    sum_c = np.zeros(D + 1)
    sum_cc = np.zeros((D + 1, D + 1))

    for p in range(lo, hi):
        for q in range(lo, hi):
            prod = p * q
            if not ((prod >> (D - 1)) & 1) or ((prod >> D) & 1):
                continue
            carries = compute_carry_chain(p, q, K)
            n_dodd += 1
            c_arr = np.array(carries, dtype=np.float64)
            sum_c += c_arr
            sum_cc += np.outer(c_arr, c_arr)

    elapsed = time.time() - t0
    mean_c = sum_c / n_dodd
    cov_full = sum_cc / n_dodd - np.outer(mean_c, mean_c)
    cov_int = cov_full[1:D, 1:D]
    L_eff = D - 1

    holte_mean, holte_cov = holte_bridge_covariance(D)
    holte_int = holte_cov[1:D, 1:D]

    P_mult, Omega_mult = partial_corr_from_cov(cov_int)
    P_holte, Omega_holte = partial_corr_from_cov(holte_int)

    margin = min(3, L_eff // 4)
    bulk_slice = slice(margin, L_eff - margin)
    P_mult_bulk = P_mult[bulk_slice, bulk_slice]
    P_holte_bulk = P_holte[bulk_slice, bulk_slice]

    if verbose:
        print(f"\n{'='*72}")
        print(f"K = {K}   D = {D}   L_eff = {L_eff}   D-odd = {n_dodd}   ({elapsed:.1f}s)")

        print(f"\n--- Full interior ({L_eff}×{L_eff}) ---")
        tf_mult = analyze_partial_corr(P_mult, "Multiplication D-odd", n_show_dist=12)
        tf_holte = analyze_partial_corr(P_holte, "Holte bridge", n_show_dist=12)

        bulk_size = L_eff - 2 * margin
        print(f"\n--- Bulk only (positions {margin+1}..{L_eff-margin}, {bulk_size}×{bulk_size}) ---")
        tb_mult = analyze_partial_corr(P_mult_bulk, "Multiplication D-odd (bulk)", n_show_dist=8)
        tb_holte = analyze_partial_corr(P_holte_bulk, "Holte bridge (bulk)", n_show_dist=8)

        print(f"\n--- Precision matrix diagonal range ---")
        diag_mult = np.diag(Omega_mult)
        diag_holte = np.diag(Omega_holte)
        print(f"  Mult:  min={diag_mult.min():.2f}  max={diag_mult.max():.2f}  ratio={diag_mult.max()/max(diag_mult.min(),1e-30):.1f}")
        print(f"  Holte: min={diag_holte.min():.2f}  max={diag_holte.max():.2f}  ratio={diag_holte.max()/max(diag_holte.min(),1e-30):.1f}")

    return {
        'K': K, 'D': D, 'n_dodd': n_dodd,
        'P_mult': P_mult, 'P_holte': P_holte,
        'P_mult_bulk': P_mult_bulk, 'P_holte_bulk': P_holte_bulk,
    }


def main():
    K_max = int(sys.argv[1]) if len(sys.argv) > 1 else 11

    print("E221b: PARTIAL CORRELATION ANALYSIS — SPATIAL MARKOV TEST")
    print("=" * 72)
    print(f"K range: 5 .. {K_max}")
    print(f"For spatial Markov process: ρ_partial(d,d')=0 for |d-d'|>1")

    results = []
    for K in range(5, K_max + 1):
        r = run_one_K(K)
        results.append(r)

    print(f"\n\n{'='*72}")
    print("SUMMARY: PARTIAL CORRELATION DECAY")
    print("=" * 72)
    print(f"\n  Full matrix — mean |ρ_partial| by distance:")
    header = f"  {'K':>3s}  {'d=1':>8s}  {'d=2':>8s}  {'d=3':>8s}  {'d=4':>8s}  {'d=5':>8s}  {'d=K-1':>8s}"
    print(header)
    for r in results:
        P = r['P_mult']
        n = P.shape[0]
        avgs = []
        for dist in [1, 2, 3, 4, 5]:
            vals = [abs(P[i, j]) for i in range(n) for j in range(i + 1, n) if j - i == dist]
            avgs.append(np.mean(vals) if vals else 0)
        d_km1 = r['K'] - 1
        vals_km1 = [abs(P[i, j]) for i in range(n) for j in range(i + 1, n) if j - i == d_km1]
        avg_km1 = np.mean(vals_km1) if vals_km1 else 0
        print(f"  {r['K']:3d}  {avgs[0]:8.4f}  {avgs[1]:8.4f}  {avgs[2]:8.4f}  {avgs[3]:8.4f}  {avgs[4]:8.4f}  {avg_km1:8.4f}")

    print(f"\n  Holte bridge (should be zero for d>1):")
    print(header)
    for r in results:
        P = r['P_holte']
        n = P.shape[0]
        avgs = []
        for dist in [1, 2, 3, 4, 5]:
            vals = [abs(P[i, j]) for i in range(n) for j in range(i + 1, n) if j - i == dist]
            avgs.append(np.mean(vals) if vals else 0)
        d_km1 = r['K'] - 1
        vals_km1 = [abs(P[i, j]) for i in range(n) for j in range(i + 1, n) if j - i == d_km1]
        avg_km1 = np.mean(vals_km1) if vals_km1 else 0
        print(f"  {r['K']:3d}  {avgs[0]:8.4f}  {avgs[1]:8.4f}  {avgs[2]:8.4f}  {avgs[3]:8.4f}  {avgs[4]:8.4f}  {avg_km1:8.4f}")

    print(f"\n  Bulk — mean |ρ_partial| by distance:")
    header2 = f"  {'K':>3s}  {'d=1':>8s}  {'d=2':>8s}  {'d=3':>8s}  {'d=4':>8s}"
    print(header2)
    for r in results:
        P = r['P_mult_bulk']
        n = P.shape[0]
        avgs = []
        for dist in [1, 2, 3, 4]:
            vals = [abs(P[i, j]) for i in range(n) for j in range(i + 1, n) if j - i == dist]
            avgs.append(np.mean(vals) if vals else 0)
        print(f"  {r['K']:3d}  {avgs[0]:8.4f}  {avgs[1]:8.4f}  {avgs[2]:8.4f}  {avgs[3]:8.4f}")

    final = results[-1]
    P = final['P_mult']
    n = P.shape[0]
    d1_vals = [abs(P[i, j]) for i in range(n) for j in range(i + 1, n) if j - i == 1]
    d2_vals = [abs(P[i, j]) for i in range(n) for j in range(i + 1, n) if j - i == 2]
    ratio = np.mean(d2_vals) / max(np.mean(d1_vals), 1e-30)

    print(f"\n\nVERDICT (K = {final['K']}):")
    print(f"  mean |ρ_partial| at d=1: {np.mean(d1_vals):.6f}")
    print(f"  mean |ρ_partial| at d=2: {np.mean(d2_vals):.6f}")
    print(f"  Ratio d=2/d=1: {ratio:.4f}")
    if ratio < 0.05:
        print(f"  → SPATIAL MARKOV: partial correlations decay sharply at d>1")
        print(f"  → LMH path-graph structure CONFIRMED")
    elif ratio < 0.2:
        print(f"  → APPROXIMATELY SPATIAL MARKOV: small but nonzero d=2 correlations")
        print(f"  → LMH is a good approximation with perturbative corrections")
    else:
        print(f"  → NOT SPATIAL MARKOV: significant long-range partial correlations")
        print(f"  → The path-graph model misses important spatial structure")


if __name__ == '__main__':
    main()
