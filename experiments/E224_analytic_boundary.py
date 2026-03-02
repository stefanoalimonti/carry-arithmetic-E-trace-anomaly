#!/usr/bin/env python3
"""E224: Analytical boundary layer calculation.

GOAL: Compute EXACT cascade probabilities in the top j0 positions,
parameterized by the incoming carry c_in. Then weight by the empirical
carry distribution from E223 to predict R(K) and compare.

STRUCTURE:
  For each (sector, c_in):
    Enumerate all 2^{2(j0-2)} free-digit configurations
    Propagate carry chain, impose D-odd (carry[D]=0)
    Record cascade depth j and val

  KEY IDENTITIES (proven analytically in E223):
    - carry[D-1] = 0 ALWAYS (forced by conv[D-1] = a_{K-1}*b_{K-1} = 1)
    - Sector 10: cascade at j=2 IMPOSSIBLE (conv[D-2] = 1 forces carry[D-1]>0)
    - Sector 00: cascade at j=2 possible (conv[D-2] = 0)

  Then: average over c_in distribution to get P(stop at j | sector) and R_pred.

Usage: python3 E224_analytic_boundary.py [j0]
Default j0 = 10.
"""

import numpy as np
from math import pi
import sys
import time
from fractions import Fraction


def boundary_exact(j0, sector_a, sector_b, c_in):
    """Exact boundary calculation for given sector and incoming carry.

    Digits: a[0]=1 (MSB), a[1]=sector_a, a[2..j0-1]=free
            b[0]=1 (MSB), b[1]=sector_b, b[2..j0-1]=free

    conv at depth j (position D-j) = sum_{k=0}^{j-1} a[k]*b[j-1-k]

    Carry propagation (i=0 is entry, i=j0 is top):
      carry[0] = c_in
      carry[i+1] = floor((conv_at_depth(j0-i) + carry[i]) / 2)

    D-odd: carry[j0] = 0.
    Cascade: largest i in {1..j0-2} with carry[i]>0, depth = j0-i.
    Val: carry[i-1] - 1.
    """
    n_free = j0 - 2
    n_configs = 1 << (2 * n_free)

    depth_val = {}   # depth -> (sum_val, count)
    n_dodd = 0
    n_no_cascade = 0

    for cfg in range(n_configs):
        a = [0] * j0
        b = [0] * j0
        a[0] = 1
        b[0] = 1
        a[1] = sector_a
        b[1] = sector_b
        for k in range(n_free):
            a[2 + k] = (cfg >> k) & 1
            b[2 + k] = (cfg >> (n_free + k)) & 1

        conv = [0] * (j0 + 1)
        for j in range(1, j0 + 1):
            s = 0
            for k in range(j):
                if k < j0 and (j - 1 - k) < j0:
                    s += a[k] * b[j - 1 - k]
            conv[j] = s

        carry = [0] * (j0 + 1)
        carry[0] = c_in
        for i in range(j0):
            carry[i + 1] = (conv[j0 - i] + carry[i]) >> 1

        if carry[j0] != 0:
            continue

        n_dodd += 1

        cascade_i = None
        for i in range(j0 - 1, 0, -1):
            if carry[i] > 0:
                cascade_i = i
                break

        if cascade_i is None:
            n_no_cascade += 1
            continue

        depth = j0 - cascade_i
        if cascade_i >= 1:
            val = carry[cascade_i - 1] - 1
        else:
            n_no_cascade += 1
            continue

        if depth not in depth_val:
            depth_val[depth] = [0, 0]
        depth_val[depth][0] += val
        depth_val[depth][1] += 1

    return n_configs, n_dodd, n_no_cascade, depth_val


def compute_carry_chain_full(p, q, K):
    """Full carry chain for exact comparison."""
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


def get_empirical_carry_dist(K, j0):
    """Get empirical carry distribution at boundary entry from exact enumeration."""
    D = 2 * K - 1
    lo = 1 << (K - 1)
    hi = 1 << K
    entry_pos = D - j0

    dist = {}
    n_dodd = 0

    for p in range(lo, hi):
        for q in range(lo, hi):
            prod = p * q
            if not ((prod >> (D - 1)) & 1) or ((prod >> D) & 1):
                continue
            n_dodd += 1
            carries = compute_carry_chain_full(p, q, K)
            c_entry = carries[entry_pos]

            a1 = (p >> (K - 2)) & 1
            c1 = (q >> (K - 2)) & 1
            sec = f"{a1}{c1}"

            key = (sec, c_entry)
            dist[key] = dist.get(key, 0) + 1

    return dist, n_dodd


def main():
    j0 = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print("E224: ANALYTICAL BOUNDARY LAYER CALCULATION")
    print("=" * 72)
    print(f"Boundary depth j0 = {j0}")
    print(f"Free bits per operand: {j0-2}")
    print(f"Configurations per (sector, c_in): {1 << (2*(j0-2))}")

    sectors = {'00': (0, 0), '01': (0, 1), '10': (1, 0), '11': (1, 1)}
    max_c_in = 25

    print(f"\n{'='*72}")
    print(f"PART 1: Boundary probabilities P(depth j | sector, c_in)")
    print(f"{'='*72}")

    all_results = {}  # (sec, c_in) -> (n_total, n_dodd, n_no_cascade, depth_val)

    for sec_name, (sa, sb) in sectors.items():
        if sec_name not in ('00', '10'):
            continue
        print(f"\n--- Sector {sec_name} ---")

        for c_in in range(max_c_in + 1):
            n_total, n_dodd, n_no_cas, dv = boundary_exact(j0, sa, sb, c_in)
            all_results[(sec_name, c_in)] = (n_total, n_dodd, n_no_cas, dv)

            if n_dodd == 0:
                if c_in > 5:
                    break
                continue

            p_dodd = n_dodd / n_total

            line = f"  c_in={c_in:2d}  D-odd={n_dodd:6d}/{n_total}={p_dodd:.4f}  "
            depths = sorted(dv.keys())
            parts = []
            for d in depths:
                sv, cnt = dv[d]
                pd = cnt / n_dodd
                parts.append(f"j={d}:{pd:.3f}({sv:+.0f})")
            no_cas_frac = n_no_cas / n_dodd if n_dodd > 0 else 0
            parts.append(f"none:{no_cas_frac:.3f}")
            line += "  ".join(parts)
            print(line)

    print(f"\n\n{'='*72}")
    print(f"PART 2: Weighted predictions using empirical carry distribution")
    print(f"{'='*72}")

    for K_test in [8, 9, 10, 11, 12]:
        t0 = time.time()
        edist, n_dodd_total = get_empirical_carry_dist(K_test, j0)
        elapsed = time.time() - t0

        print(f"\n--- K = {K_test}  (D-odd = {n_dodd_total}, {elapsed:.1f}s) ---")

        # Carry entry distribution per sector
        for sec_name in ['00', '10']:
            total_sec = sum(v for (s, c), v in edist.items() if s == sec_name)
            print(f"\n  Sector {sec_name} (n={total_sec}):")
            print(f"    Carry entry distribution:")
            for c_in in range(max_c_in + 1):
                cnt = edist.get((sec_name, c_in), 0)
                if cnt > 0:
                    print(f"      c_in={c_in}: {cnt}/{total_sec} = {cnt/total_sec:.4f}")

            total_val_sum = 0
            total_cascade_count = 0
            depth_totals = {}

            for c_in in range(max_c_in + 1):
                n_pairs_at_cin = edist.get((sec_name, c_in), 0)
                if n_pairs_at_cin == 0:
                    continue

                key = (sec_name, c_in)
                if key not in all_results:
                    continue

                n_total, n_dodd, n_no_cas, dv = all_results[key]
                if n_dodd == 0:
                    continue

                weight = n_pairs_at_cin / total_sec

                for depth, (sv, cnt) in dv.items():
                    p_depth_given_cin_dodd = cnt / n_dodd
                    mean_val = sv / cnt if cnt > 0 else 0

                    weighted_count = weight * p_depth_given_cin_dodd * total_sec
                    weighted_val = weight * (sv / n_dodd) * total_sec

                    if depth not in depth_totals:
                        depth_totals[depth] = [0.0, 0.0]
                    depth_totals[depth][0] += weighted_val
                    depth_totals[depth][1] += weighted_count

            print(f"\n    Predicted per-depth contributions:")
            print(f"    {'j':>3s}  {'pred_count':>10s}  {'pred_Σval':>12s}  {'pred_R_j':>10s}")
            for d in sorted(depth_totals.keys()):
                sv, cnt = depth_totals[d]
                rj = "---" if cnt < 0.5 else f"{sv/cnt:.4f}"
                print(f"    {d:3d}  {cnt:10.1f}  {sv:12.1f}  {rj:>10s}")

            sec_total_val = sum(sv for sv, cnt in depth_totals.values())
            sec_total_cnt = sum(cnt for sv, cnt in depth_totals.values())
            print(f"    Total: count={sec_total_cnt:.0f}  Σval={sec_total_val:.1f}")

        # Compute predicted R(K)
        total_v00 = 0
        total_v10 = 0
        for sec_name in ['00', '10']:
            total_sec = sum(v for (s, c), v in edist.items() if s == sec_name)
            for c_in in range(max_c_in + 1):
                n_pairs = edist.get((sec_name, c_in), 0)
                if n_pairs == 0:
                    continue
                key = (sec_name, c_in)
                if key not in all_results:
                    continue
                n_total, n_dodd, n_no_cas, dv = all_results[key]
                if n_dodd == 0:
                    continue
                for depth, (sv, cnt) in dv.items():
                    # Scale: each of n_pairs empirical pairs contributes
                    # (sv/n_dodd) * n_pairs as val sum
                    # But actually, each empirical pair with c_in has
                    # (sv/n_dodd) expected val contribution from the boundary
                    # Wait, I need to be more careful.
                    # n_pairs pairs have c_in at the entry.
                    # Of these, fraction (n_dodd/n_total) survive D-odd.
                    # But they already survived D-odd (they're D-odd empirical pairs).
                    # The boundary D-odd fraction is conditional on the digit configs.
                    # Since the free digits in the boundary are a SUBSET of the full
                    # digit configuration, and we're conditioning on the full D-odd,
                    # the boundary D-odd fraction gives us the probability.
                    pass

        # Simpler approach: direct weighted sum
        pred_v00 = 0.0
        pred_v10 = 0.0
        for sec_name in ['00', '10']:
            total_sec = sum(v for (s, c), v in edist.items() if s == sec_name)
            for c_in in range(max_c_in + 1):
                n_pairs = edist.get((sec_name, c_in), 0)
                if n_pairs == 0:
                    continue
                key = (sec_name, c_in)
                if key not in all_results:
                    continue
                n_total, n_dodd, n_no_cas, dv = all_results[key]
                if n_dodd == 0:
                    continue
                for depth, (sv, cnt) in dv.items():
                    contrib = n_pairs * sv / n_dodd
                    if sec_name == '00':
                        pred_v00 += contrib
                    else:
                        pred_v10 += contrib

        pred_R = pred_v10 / pred_v00 if pred_v00 != 0 else float('nan')

        # Exact R(K) for comparison
        D = 2 * K_test - 1
        lo = 1 << (K_test - 1)
        hi = 1 << K_test
        exact_v00 = 0
        exact_v10 = 0
        for p in range(lo, hi):
            a1 = (p >> (K_test - 2)) & 1
            for q in range(lo, hi):
                prod = p * q
                if not ((prod >> (D - 1)) & 1) or ((prod >> D) & 1):
                    continue
                c1 = (q >> (K_test - 2)) & 1
                carries = compute_carry_chain_full(p, q, K_test)
                M = None
                for m in range(D, 0, -1):
                    if carries[m] > 0:
                        M = m
                        break
                if M is None:
                    continue
                val = carries[M - 1] - 1
                if a1 == 0 and c1 == 0:
                    exact_v00 += val
                elif a1 == 1 and c1 == 0:
                    exact_v10 += val

        exact_R = exact_v10 / exact_v00 if exact_v00 != 0 else float('nan')
        print(f"\n  K={K_test}: Predicted R = {pred_R:.6f}   Exact R = {exact_R:.6f}   "
              f"Diff = {abs(pred_R - exact_R):.6f}")
        print(f"  Target: -π = {-pi:.6f}")

    print(f"\n\n{'='*72}")
    print(f"PART 3: Limiting boundary probabilities (c_in → distribution)")
    print(f"{'='*72}")
    print(f"\n  Exact P(D-odd | c_in, sector) and P(cascade at j | c_in, sector, D-odd):")
    print(f"  These are UNIVERSAL constants, independent of K.")
    for sec_name in ['00', '10']:
        sa, sb = sectors[sec_name]
        print(f"\n  Sector {sec_name}:")
        print(f"  {'c_in':>5s}  {'P(Dodd)':>8s}  {'j=2':>8s}  {'j=3':>8s}  {'j=4':>8s}  {'j=5':>8s}  {'j=6':>8s}  {'j=7':>8s}  {'j=8':>8s}")
        for c_in in range(16):
            key = (sec_name, c_in)
            if key not in all_results:
                continue
            n_total, n_dodd, n_no_cas, dv = all_results[key]
            if n_dodd == 0:
                continue
            p_dodd = n_dodd / n_total
            parts = [f"{c_in:5d}", f"{p_dodd:8.4f}"]
            for j in range(2, 9):
                if j in dv:
                    parts.append(f"{dv[j][1]/n_dodd:8.4f}")
                else:
                    parts.append(f"{'0':>8s}")
            print("  " + "  ".join(parts))


if __name__ == '__main__':
    main()
