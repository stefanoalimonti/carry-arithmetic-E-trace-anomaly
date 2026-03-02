#!/usr/bin/env python3
"""E223: Boundary layer analysis of the cascade.

KEY INSIGHT from E222: The cascade is a boundary-layer phenomenon.
D-M ∈ {2,...,7} with a converging distribution. The cascade reads only
the top few positions of the carry chain.

SECTORS (§2.3): a_1 = (p >> (K-2)) & 1, c_1 = (q >> (K-2)) & 1.
Sector (1,0): a_1=1, c_1=0.  Sector (0,0): a_1=0, c_1=0.
R(K) = Σ val_{10} / Σ val_{00}.

ANALYSIS:
A. Correct R(K) computation → verify against Table 1
B. Per-depth cascade decomposition:
   R(K) = Σ_j P(cascade at depth j) * R_j(K)
   where j = D - M (depth from top)
C. Carry distribution entering the boundary (carry at position D-j_0)
D. Convergence of per-depth contributions R_j(K) as K → ∞
E. Boundary layer limit: can we compute R(∞) from the boundary alone?

Usage: python3 E223_boundary_layer.py [K_max]
"""

import numpy as np
import sys
import time
from math import pi


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


def run_one_K(K, verbose=True):
    D = 2 * K - 1
    lo = 1 << (K - 1)
    hi = 1 << K

    t0 = time.time()

    max_depth = 15
    n_dodd = 0
    n_cascade = 0
    n_excluded = 0

    # Per-depth, per-sector: val sums and counts
    # depth j = D - M (distance from top)
    val_by_depth = {}   # depth -> {sector: [sum_val, count]}
    total_val = {}      # sector -> sum_val
    total_count = {}    # sector -> count

    # Carry distribution at depth j from top: carry[D-j] values
    carry_at_depth = {}  # depth -> list of carry values (for distribution)

    for sec in ['00', '01', '10', '11']:
        total_val[sec] = 0.0
        total_count[sec] = 0

    for p in range(lo, hi):
        a1 = (p >> (K - 2)) & 1
        for q in range(lo, hi):
            prod = p * q
            if not ((prod >> (D - 1)) & 1) or ((prod >> D) & 1):
                continue

            n_dodd += 1
            c1 = (q >> (K - 2)) & 1
            sec = f"{a1}{c1}"

            carries = compute_carry_chain(p, q, K)

            # Find cascade stopping position M
            M = None
            for m in range(D, 0, -1):
                if carries[m] > 0:
                    M = m
                    break

            if M is None:
                n_excluded += 1
                continue

            n_cascade += 1
            val = carries[M - 1] - 1
            depth = D - M

            total_val[sec] += val
            total_count[sec] += 1

            if depth not in val_by_depth:
                val_by_depth[depth] = {}
            if sec not in val_by_depth[depth]:
                val_by_depth[depth][sec] = [0.0, 0]
            val_by_depth[depth][sec][0] += val
            val_by_depth[depth][sec][1] += 1

            # Record carry values at various depths
            for j in range(1, min(max_depth + 1, D)):
                pos = D - j
                if pos >= 0:
                    if j not in carry_at_depth:
                        carry_at_depth[j] = []
                    carry_at_depth[j].append(carries[pos])

    elapsed = time.time() - t0

    # Compute R(K)
    s00 = total_val.get('00', 0)
    s10 = total_val.get('10', 0)
    n00 = total_count.get('00', 0)
    n10 = total_count.get('10', 0)
    R_K = s10 / s00 if s00 != 0 else float('nan')

    if verbose:
        print(f"\n{'='*72}")
        print(f"K = {K}   D = {D}   D-odd = {n_dodd}   cascade = {n_cascade}")
        print(f"Excluded (no nonzero carry): {n_excluded} ({100*n_excluded/max(n_dodd,1):.1f}%)")
        print(f"Elapsed: {elapsed:.1f}s")

        print(f"\n--- A: Sector sums and R(K) ---")
        print(f"  Sector counts: 00={n00}  01={total_count.get('01',0)}  10={n10}  11={total_count.get('11',0)}")
        print(f"  Σval_00 = {s00:.4f}   Σval_10 = {s10:.4f}")
        print(f"  R(K) = {R_K:.6f}   (target: -π = {-pi:.6f})")
        print(f"  Gap = {R_K - (-pi):.6f}")

        print(f"\n--- B: Per-depth cascade decomposition ---")
        print(f"  j=D-M: cascade depth from the top")
        print(f"  {'j':>3s}  {'P(j)':>8s}  {'n_00':>6s}  {'Σv_00':>10s}  {'n_10':>6s}  {'Σv_10':>10s}  {'R_j':>10s}")
        depths = sorted(val_by_depth.keys())
        for j in depths:
            d = val_by_depth[j]
            sv00, nc00 = d.get('00', [0, 0])
            sv10, nc10 = d.get('10', [0, 0])
            ntot = sum(v[1] for v in d.values())
            pj = ntot / max(n_cascade, 1)
            rj = sv10 / sv00 if sv00 != 0 else float('nan')
            print(f"  {j:3d}  {pj:8.4f}  {nc00:6d}  {sv00:10.2f}  {nc10:6d}  {sv10:10.2f}  {rj:10.4f}")

        print(f"\n--- C: Carry distribution at boundary depths ---")
        for j in [1, 2, 3, 4, 5, 6]:
            if j in carry_at_depth:
                vals = carry_at_depth[j]
                unique, counts = np.unique(vals, return_counts=True)
                total = len(vals)
                print(f"  Depth j={j} (position D-{j}={D-j}):")
                for u, c in zip(unique, counts):
                    print(f"    carry={u}: {c}/{total} = {c/total:.4f}")

        print(f"\n--- D: Per-depth R_j convergence ---")
        print(f"  (R_j = ratio of cascade contributions at each depth)")
        print(f"  If R_j converges to a fixed value as K → ∞, the boundary")
        print(f"  layer determines R(∞).")

    return {
        'K': K, 'D': D, 'R_K': R_K,
        'val_by_depth': val_by_depth,
        'n_cascade': n_cascade,
        'n_dodd': n_dodd,
        'n_excluded': n_excluded,
    }


def main():
    K_max = int(sys.argv[1]) if len(sys.argv) > 1 else 11

    print("E223: BOUNDARY LAYER ANALYSIS OF THE CASCADE")
    print("=" * 72)
    print(f"K range: 5 .. {K_max}")
    print(f"Sectors: a_1=(p>>(K-2))&1, c_1=(q>>(K-2))&1")
    print(f"R(K) = Σval_10 / Σval_00")

    results = []
    for K in range(5, K_max + 1):
        r = run_one_K(K)
        results.append(r)

    print(f"\n\n{'='*72}")
    print("SUMMARY")
    print("=" * 72)

    print(f"\n  R(K) verification:")
    print(f"  {'K':>3s}  {'R(K)':>12s}  {'gap':>12s}  {'excl%':>8s}")
    for r in results:
        gap = r['R_K'] - (-pi) if not np.isnan(r['R_K']) else float('nan')
        excl = 100 * r['n_excluded'] / max(r['n_dodd'], 1)
        print(f"  {r['K']:3d}  {r['R_K']:12.6f}  {gap:12.6f}  {excl:8.1f}")

    print(f"\n  Per-depth cascade probability P(j) across K:")
    header = f"  {'K':>3s}"
    for j in range(2, 9):
        header += f"  {'j='+str(j):>8s}"
    print(header)
    for r in results:
        line = f"  {r['K']:3d}"
        for j in range(2, 9):
            d = r['val_by_depth']
            if j in d:
                ntot = sum(v[1] for v in d[j].values())
                pj = ntot / max(r['n_cascade'], 1)
                line += f"  {pj:8.4f}"
            else:
                line += f"  {'---':>8s}"
        print(line)

    print(f"\n  Per-depth sector ratio R_j across K:")
    header = f"  {'K':>3s}"
    for j in range(2, 9):
        header += f"  {'j='+str(j):>8s}"
    print(header)
    for r in results:
        line = f"  {r['K']:3d}"
        for j in range(2, 9):
            d = r['val_by_depth']
            if j in d and '00' in d[j] and d[j]['00'][0] != 0:
                rj = d[j].get('10', [0, 0])[0] / d[j]['00'][0]
                line += f"  {rj:8.4f}"
            else:
                line += f"  {'---':>8s}"
        print(line)

    print(f"\n  BOUNDARY LAYER TEST:")
    print(f"  If P(j) and R_j both converge as K → ∞, then:")
    print(f"  R(∞) = Σ_j P_∞(j) * R_∞(j)")
    print(f"  and the boundary layer FULLY determines R(∞) = -π.")


if __name__ == '__main__':
    main()
