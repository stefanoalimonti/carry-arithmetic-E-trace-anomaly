#!/usr/bin/env python3
"""E220: Spatial covariance structure of D-odd carry chains.

CORE QUESTION: Do the eigenvectors of the carry-carry covariance matrix
under D-odd conditioning match the sine basis sin(n*pi*d/(L+1))?

If YES → the spatial model underlying the LMH is validated from below.
The eigenvalues then constrain the effective operator spectrum.

If NO → the LMH spatial model doesn't emerge from covariance structure
and the derivation requires a different approach.

METHODOLOGY:
  Part A: Carry covariance matrix and eigenvector analysis
    - Exact enumeration for K = 5..12
    - Cov(carry[d], carry[d']) for D-odd products
    - Eigendecompose interior covariance
    - Compare eigenvectors to sine basis via overlap matrix

  Part B: Holte bridge baseline (analytical)
    - For the addition carry chain (binary Markov) conditioned as a bridge
      (carry[0]=0, carry[L]=0), compute covariance analytically
    - Compare to multiplication carry covariance

  Part C: Bit-sharing correction isolation
    - ΔCov = Cov_multiplication - Cov_Holte_bridge
    - Does the correction have LMH structure?

  Part D: LMH eigenvalue extraction
    - From empirical covariance eigenvalues, extract implied operator parameters
    - Compare to A* = (2+3π)/(2(1+π))

Usage: python3 E220_doob_spatial_covariance.py [K_max]
Default K_max = 11 (feasible in ~15 min on a laptop).
"""

import numpy as np
import sys
import time
from math import pi, sqrt, cos, sin, log10


def compute_carry_chain(p, q, K):
    """Carry chain for p*q with K-bit operands. Returns carries[0..D]."""
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
    """Analytical covariance of the Holte (addition) carry bridge.

    The Holte chain on {0,1} with K_M = [[3/4, 1/4], [1/4, 3/4]]
    conditioned on carry[0]=0, carry[D]=0 (bridge).

    Uses h-transform: h(c, d) = P(carry[D]=0 | carry[d]=c)
    = [K_M^{D-d}]_{c,0} = 1/2 + (-1)^c * 2^{-(D-d+1)}

    The bridge distribution at position d:
    P(carry[d]=c | bridge) = P(carry[d]=c | carry[0]=0) * h(c,d) / h(0,0)

    Forward: P(carry[d]=c | carry[0]=0) = [K_M^d]_{0,c}
    K_M^n[0,0] = 1/2 + 2^{-(n+1)}, K_M^n[0,1] = 1/2 - 2^{-(n+1)}
    """
    KM = np.array([[3/4, 1/4], [1/4, 3/4]])

    def h(c, d):
        return 0.5 + ((-1)**c) * 2.0**(-(D - d + 1))

    def forward(c, d):
        """P(carry[d]=c | carry[0]=0)"""
        return 0.5 + ((-1)**c) * 2.0**(-(d + 1))

    def bridge_prob(c, d):
        """P(carry[d]=c | carry[0]=0, carry[D]=0)"""
        return forward(c, d) * h(c, d) / h(0, 0)

    def joint_bridge(c1, d1, c2, d2):
        """P(carry[d1]=c1, carry[d2]=c2 | bridge), d1 < d2."""
        f_c1 = forward(c1, d1)
        mid = np.linalg.matrix_power(KM, d2 - d1)[c1, c2]
        h_c2 = h(c2, d2)
        return f_c1 * mid * h_c2 / h(0, 0)

    mean = np.zeros(D + 1)
    cov = np.zeros((D + 1, D + 1))

    for d in range(D + 1):
        mean[d] = bridge_prob(1, d)

    for d1 in range(D + 1):
        for d2 in range(d1, D + 1):
            if d1 == d2:
                e_cc = bridge_prob(1, d1)
            else:
                e_cc = joint_bridge(1, d1, 1, d2)
            cov[d1, d2] = e_cc - mean[d1] * mean[d2]
            cov[d2, d1] = cov[d1, d2]

    return mean, cov


def sine_basis(L_eff):
    """Normalized sine basis vectors on L_eff interior points.
    phi_n(d) = sin(n*pi*d/(L_eff+1)), d=1..L_eff, n=1..L_eff
    """
    basis = np.zeros((L_eff, L_eff))
    for n in range(L_eff):
        for d in range(L_eff):
            basis[d, n] = sin((n + 1) * pi * (d + 1) / (L_eff + 1))
    norms = np.linalg.norm(basis, axis=0)
    basis /= norms[np.newaxis, :]
    return basis


def eigvec_sine_overlap(eigvecs, L_eff):
    """Compute |<eigvec_i, sine_j>| overlap matrix.
    Returns max overlap for each eigenvector and its best-matching sine index.
    """
    basis = sine_basis(L_eff)
    overlap = np.abs(eigvecs.T @ basis)
    best_sine = np.argmax(overlap, axis=1)
    best_overlap = np.max(overlap, axis=1)
    return overlap, best_sine, best_overlap


def run_one_K(K, verbose=True):
    """Run full covariance analysis for one K value."""
    D = 2 * K - 1
    lo = 1 << (K - 1)
    hi = 1 << K
    L_eff = D - 1

    t0 = time.time()
    n_dodd = 0
    sum_c = np.zeros(D + 1)
    sum_cc = np.zeros((D + 1, D + 1))
    max_carry = 0

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
            mc = max(carries)
            if mc > max_carry:
                max_carry = mc

    elapsed = time.time() - t0
    mean_c = sum_c / n_dodd
    cov_full = sum_cc / n_dodd - np.outer(mean_c, mean_c)

    cov_int = cov_full[1:D, 1:D]
    eigvals_raw, eigvecs_raw = np.linalg.eigh(cov_int)
    idx = np.argsort(eigvals_raw)[::-1]
    eigvals = eigvals_raw[idx]
    eigvecs = eigvecs_raw[:, idx]

    overlap_mat, best_sine, best_overlap = eigvec_sine_overlap(eigvecs, L_eff)

    holte_mean, holte_cov = holte_bridge_covariance(D)
    holte_cov_int = holte_cov[1:D, 1:D]
    h_eigvals_raw, h_eigvecs_raw = np.linalg.eigh(holte_cov_int)
    h_idx = np.argsort(h_eigvals_raw)[::-1]
    h_eigvals = h_eigvals_raw[h_idx]
    h_eigvecs = h_eigvecs_raw[:, h_idx]
    _, h_best_sine, h_best_overlap = eigvec_sine_overlap(h_eigvecs, L_eff)

    delta_cov = cov_int - holte_cov_int
    d_eigvals_raw, d_eigvecs_raw = np.linalg.eigh(delta_cov)
    d_idx = np.argsort(np.abs(d_eigvals_raw))[::-1]
    d_eigvals = d_eigvals_raw[d_idx]
    d_eigvecs = d_eigvecs_raw[:, d_idx]
    _, d_best_sine, d_best_overlap = eigvec_sine_overlap(d_eigvecs, L_eff)

    if verbose:
        print(f"\n{'='*72}")
        print(f"K = {K}   D = {D}   L_eff = {L_eff}   D-odd pairs = {n_dodd}")
        print(f"Elapsed: {elapsed:.1f}s   Max carry value: {max_carry}")

        print(f"\n--- Part A: Multiplication carry covariance eigenvectors ---")
        print(f"  {'mode':>4s}  {'eigenvalue':>12s}  {'best sine n':>12s}  {'|overlap|':>10s}  {'match?':>6s}")
        n_show = min(8, L_eff)
        for i in range(n_show):
            match = "YES" if best_overlap[i] > 0.95 else ("~" if best_overlap[i] > 0.80 else "NO")
            print(f"  {i+1:4d}  {eigvals[i]:12.6e}  {best_sine[i]+1:12d}  {best_overlap[i]:10.6f}  {match:>6s}")

        total_overlap = np.mean(best_overlap[:min(6, L_eff)])
        print(f"\n  Mean |overlap| (top 6 modes): {total_overlap:.4f}")
        if total_overlap > 0.95:
            print(f"  *** SINE BASIS MATCH: STRONG ***")
        elif total_overlap > 0.85:
            print(f"  *** SINE BASIS MATCH: MODERATE ***")
        else:
            print(f"  *** SINE BASIS MATCH: WEAK ***")

        print(f"\n--- Part B: Holte bridge covariance eigenvectors ---")
        print(f"  {'mode':>4s}  {'eigenvalue':>12s}  {'best sine n':>12s}  {'|overlap|':>10s}")
        for i in range(n_show):
            print(f"  {i+1:4d}  {h_eigvals[i]:12.6e}  {h_best_sine[i]+1:12d}  {h_best_overlap[i]:10.6f}")

        print(f"\n--- Part C: Bit-sharing correction ΔCov = Cov_mult - Cov_Holte ---")
        print(f"  Frobenius norms: ||Cov_mult|| = {np.linalg.norm(cov_int, 'fro'):.6f}")
        print(f"                   ||Cov_Holte|| = {np.linalg.norm(holte_cov_int, 'fro'):.6f}")
        print(f"                   ||ΔCov||      = {np.linalg.norm(delta_cov, 'fro'):.6f}")
        print(f"  ΔCov / Cov_mult ratio: {np.linalg.norm(delta_cov, 'fro') / np.linalg.norm(cov_int, 'fro'):.4f}")
        print(f"\n  ΔCov eigenvectors (by |eigenvalue|):")
        print(f"  {'mode':>4s}  {'eigenvalue':>12s}  {'best sine n':>12s}  {'|overlap|':>10s}")
        for i in range(min(6, L_eff)):
            print(f"  {i+1:4d}  {d_eigvals[i]:12.6e}  {d_best_sine[i]+1:12d}  {d_best_overlap[i]:10.6f}")

        print(f"\n--- Part D: LMH eigenvalue comparison ---")
        A_star = (2 + 3*pi) / (2*(1 + pi))
        print(f"  A* = {A_star:.6f}")
        print(f"\n  Covariance eigenvalue spectrum (empirical vs LMH-predicted resolvent):")
        print(f"  The LMH predicts spatial eigenvalues lambda_n = (1-A)*cos(n*pi/(L+1))/2 + A/2")
        print(f"  Resolvent eigenvalues: 1/(1 - lambda_n)")
        print(f"\n  {'n':>4s}  {'cov_eigval':>12s}  {'lambda_LMH':>12s}  {'1/(1-lam)':>12s}  {'ratio':>10s}")
        for n in range(1, min(7, L_eff + 1)):
            lam_lmh = (1 - A_star) * cos(n * pi / (L_eff + 1)) / 2 + A_star / 2
            resolvent = 1.0 / (1.0 - lam_lmh)
            ev = eigvals[n - 1] if n - 1 < len(eigvals) else 0
            ratio = ev / resolvent if abs(resolvent) > 1e-15 else float('nan')
            print(f"  {n:4d}  {ev:12.6e}  {lam_lmh:12.6f}  {resolvent:12.6f}  {ratio:10.6f}")

    return {
        'K': K, 'D': D, 'n_dodd': n_dodd, 'max_carry': max_carry,
        'mean_carry': mean_c,
        'cov_int': cov_int,
        'eigvals': eigvals, 'eigvecs': eigvecs,
        'best_sine': best_sine, 'best_overlap': best_overlap,
        'h_eigvals': h_eigvals, 'h_best_overlap': h_best_overlap,
        'delta_norm_ratio': np.linalg.norm(delta_cov, 'fro') / max(np.linalg.norm(cov_int, 'fro'), 1e-15),
        'd_eigvals': d_eigvals, 'd_best_overlap': d_best_overlap,
    }


def main():
    K_max = int(sys.argv[1]) if len(sys.argv) > 1 else 11

    print("E220: SPATIAL COVARIANCE STRUCTURE OF D-ODD CARRY CHAINS")
    print("=" * 72)
    print(f"K range: 5 .. {K_max}")
    print(f"Target: A* = (2+3π)/(2(1+π)) = {(2+3*pi)/(2*(1+pi)):.8f}")
    print(f"Question: Do covariance eigenvectors match sin(nπd/(L+1))?")

    results = []
    for K in range(5, K_max + 1):
        r = run_one_K(K)
        results.append(r)

    print(f"\n\n{'='*72}")
    print("SUMMARY: EIGENVECTOR-SINE OVERLAP ACROSS K")
    print("=" * 72)
    print(f"  {'K':>3s}  {'D':>3s}  {'n_dodd':>10s}  {'<|overlap|>':>12s}  {'mode1':>8s}  {'mode2':>8s}  {'mode3':>8s}  {'ΔCov/Cov':>10s}")
    for r in results:
        n6 = min(6, len(r['best_overlap']))
        mean_ov = np.mean(r['best_overlap'][:n6])
        ov1 = r['best_overlap'][0] if len(r['best_overlap']) > 0 else 0
        ov2 = r['best_overlap'][1] if len(r['best_overlap']) > 1 else 0
        ov3 = r['best_overlap'][2] if len(r['best_overlap']) > 2 else 0
        print(f"  {r['K']:3d}  {r['D']:3d}  {r['n_dodd']:10d}  {mean_ov:12.6f}  {ov1:8.4f}  {ov2:8.4f}  {ov3:8.4f}  {r['delta_norm_ratio']:10.4f}")

    print(f"\nHolte bridge baseline (same metric):")
    print(f"  {'K':>3s}  {'mode1':>8s}  {'mode2':>8s}  {'mode3':>8s}")
    for r in results:
        ov1 = r['h_best_overlap'][0] if len(r['h_best_overlap']) > 0 else 0
        ov2 = r['h_best_overlap'][1] if len(r['h_best_overlap']) > 1 else 0
        ov3 = r['h_best_overlap'][2] if len(r['h_best_overlap']) > 2 else 0
        print(f"  {r['K']:3d}  {ov1:8.4f}  {ov2:8.4f}  {ov3:8.4f}")

    print(f"\nVERDICT:")
    final_overlaps = results[-1]['best_overlap'][:min(6, len(results[-1]['best_overlap']))]
    mean_final = np.mean(final_overlaps)
    if mean_final > 0.95:
        print(f"  SINE EIGENVECTORS CONFIRMED (mean overlap {mean_final:.4f})")
        print(f"  → The LMH spatial model is VALIDATED at the covariance level.")
        print(f"  → The carry chain under D-odd conditioning has path-graph structure.")
        print(f"  → Next: extract A from eigenvalue ratios to test A = A*.")
    elif mean_final > 0.85:
        print(f"  PARTIAL SINE STRUCTURE (mean overlap {mean_final:.4f})")
        print(f"  → Low modes match sine basis; high modes deviate.")
        print(f"  → The LMH is an effective low-energy description, not exact.")
    else:
        print(f"  SINE STRUCTURE NOT FOUND (mean overlap {mean_final:.4f})")
        print(f"  → The LMH spatial model does NOT emerge from covariance analysis.")
        print(f"  → A different approach to deriving the LMH is needed.")


if __name__ == '__main__':
    main()
