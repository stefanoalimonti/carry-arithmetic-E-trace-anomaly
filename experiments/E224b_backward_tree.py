#!/usr/bin/env python3
"""E224b: Backward tree DP + forward exact analysis of cascade boundary layer.

PART A: Backward tree from carry[D-1]=0 — reveals structural exclusions
PART B: Forward exact enumeration — gives convergent P(j) limits
PART C: Convergence analysis — extract universal boundary constants

KEY STRUCTURAL FINDINGS:
  - j=2 cascade EXCLUSIVELY in sector 00 (conv[D-2]=0)
  - Sector 10/01: j=2 blocked (conv[D-2]=1 forces carry=0)
  - Sector 11: D-odd impossible (conv[D-2]=2, no valid carry)

KEY QUANTITATIVE FINDING:
  - P(j|sector) converge as K → ∞ to universal constants
  - The product-bit-1/2 approximation in the tree is ~20% off
  - Exact limits require bulk carry distribution

Usage: python3 E224b_backward_tree.py [j_max]
"""

import sys
import time
from math import pi
from collections import defaultdict


def compute_conv(a_bits, b_bits, j):
    """Column sum conv[D-j] = sum_{k=0}^{j-1} a[k]*b[j-1-k]."""
    conv = 0
    for k in range(j):
        conv += ((a_bits >> (j - 1 - k)) & 1) * ((b_bits >> k) & 1)
    return conv


def backward_tree_dp(j_max, sector_a, sector_b):
    """Backward tree DP from carry[D-1]=0 downward."""
    ongoing = {(1, 1): 1.0}
    cascade = {}
    awaiting_val = []
    total_pruned = 0.0

    for j in range(2, j_max + 2):
        d_opts = [(sector_a, sector_b)] if j == 2 else [(0, 0), (0, 1), (1, 0), (1, 1)]
        d_w = 1.0 if j == 2 else 0.25

        new_ongoing = defaultdict(float)
        new_awaiting = []

        if j <= j_max:
            for (ab, bb), w in ongoing.items():
                for (an, bn) in d_opts:
                    ae = (ab << 1) | an
                    be = (bb << 1) | bn
                    conv = compute_conv(ae, be, j)
                    valids = [(bit, bit - conv) for bit in (0, 1) if bit - conv >= 0]
                    if not valids:
                        total_pruned += w * d_w
                        continue
                    bw = 1.0 / len(valids)
                    for (_, nc) in valids:
                        sw = w * d_w * bw
                        if nc > 0:
                            cascade.setdefault(j, {'count': 0.0, 'val_sum': 0.0})
                            cascade[j]['count'] += sw
                            new_awaiting.append((ae, be, nc, sw))
                        else:
                            new_ongoing[(ae, be)] += sw

        for (ab_s, bb_s, c_s, w_s) in awaiting_val:
            val_acc = 0.0
            for (an, bn) in d_opts:
                ae = (ab_s << 1) | an
                be = (bb_s << 1) | bn
                conv = compute_conv(ae, be, j)
                v_valids = [(bit, 2 * c_s + bit - conv) for bit in (0, 1)
                            if 2 * c_s + bit - conv >= 0]
                if not v_valids:
                    continue
                vbw = d_w / len(v_valids)
                for (_, bc) in v_valids:
                    val_acc += vbw * (bc - 1)
            cascade[j - 1]['val_sum'] += w_s * val_acc

        awaiting_val = new_awaiting
        ongoing = dict(new_ongoing)

    for (_, _, c_s, w_s) in awaiting_val:
        if j_max in cascade:
            cascade[j_max]['val_sum'] += w_s * (c_s - 1)

    return cascade, sum(ongoing.values()), total_pruned


def forward_exact(K):
    """Full exact enumeration of D-odd cascade."""
    D = 2 * K - 1
    lo = 1 << (K - 1)
    hi = 1 << K
    per_depth = {}
    n_dodd = 0
    n_cascade = 0

    for p in range(lo, hi):
        a1 = (p >> (K - 2)) & 1
        for q in range(lo, hi):
            prod = p * q
            if prod.bit_length() != D:
                continue
            c1 = (q >> (K - 2)) & 1
            sec = f"{a1}{c1}"
            n_dodd += 1

            carries = [0] * (D + 1)
            for d in range(D):
                cv = 0
                for i in range(max(0, d - K + 1), min(d, K - 1) + 1):
                    cv += ((p >> i) & 1) * ((q >> (d - i)) & 1)
                carries[d + 1] = (cv + carries[d]) >> 1

            M = None
            for m in range(D, 0, -1):
                if carries[m] > 0:
                    M = m
                    break
            if M is None:
                continue

            n_cascade += 1
            depth = D - M
            val = carries[M - 1] - 1
            key = (sec, depth)
            per_depth.setdefault(key, [0, 0.0])
            per_depth[key][0] += 1
            per_depth[key][1] += val

    return per_depth, n_dodd, n_cascade


def main():
    j_max = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print("E224b: BACKWARD TREE + FORWARD EXACT — BOUNDARY LAYER ANALYSIS")
    print("=" * 72)
    print(f"Boundary depth j_max = {j_max}")
    print()

    # ==================================================================
    # PART A: Backward tree structure
    # ==================================================================
    print("=" * 72)
    print("PART A: BACKWARD TREE — STRUCTURAL ANALYSIS")
    print("=" * 72)
    print("""
  Carry inversion: carry[D-j] = 2*carry[D-j+1] + product_bit - conv[D-j]
  Starting from carry[D-1] = 0 (D-odd), expand downward.
  Free product bits weighted 1/2 each (approximation).

  STRUCTURAL EXCLUSIONS at j=2 (conv[D-2] = a[1]+b[1]):
    Sector 00: conv=0 → carry ∈ {0,1} → CASCADE POSSIBLE
    Sector 10: conv=1 → only carry=0  → j=2 BLOCKED
    Sector 01: conv=1 → only carry=0  → j=2 BLOCKED
    Sector 11: conv=2 → no valid carry → D-ODD IMPOSSIBLE
""")

    tree_data = {}
    for sec_name, sa, sb in [('00', 0, 0), ('10', 1, 0)]:
        cascade, ongoing, pruned = backward_tree_dp(j_max, sa, sb)
        total_casc = sum(c['count'] for c in cascade.values())
        surviving = total_casc + ongoing

        print(f"  Sector {sec_name}: cascade={total_casc:.4f}  "
              f"ongoing={ongoing:.4f}  pruned={pruned:.4f}")

        tree_data[sec_name] = {}
        for j in sorted(cascade.keys()):
            cnt = cascade[j]['count']
            vs = cascade[j]['val_sum']
            p_norm = cnt / surviving if surviving > 0 else 0
            e_val = vs / cnt if cnt > 1e-15 else float('nan')
            tree_data[sec_name][j] = (p_norm, e_val)

    # ==================================================================
    # PART B: Forward exact + convergence
    # ==================================================================
    print(f"\n{'='*72}")
    print("PART B: FORWARD EXACT — P(j|sector) CONVERGENCE")
    print("=" * 72)

    K_values = [7, 8, 9, 10, 11, 12]
    all_exact = {}

    for K in K_values:
        t0 = time.time()
        per_depth, n_dodd, n_cascade = forward_exact(K)
        elapsed = time.time() - t0
        all_exact[K] = (per_depth, n_dodd, n_cascade)

        v00 = sum(v[1] for (s, d), v in per_depth.items() if s == '00')
        v10 = sum(v[1] for (s, d), v in per_depth.items() if s == '10')
        R = v10 / v00 if abs(v00) > 1e-10 else float('nan')

        n00 = sum(v[0] for (s, d), v in per_depth.items() if s == '00')
        n10 = sum(v[0] for (s, d), v in per_depth.items() if s == '10')

        print(f"\n  K={K:2d}  D={2*K-1:2d}  n_dodd={n_dodd:>8d}  "
              f"n_00={n00:>7d}  n_10={n10:>7d}  "
              f"R={R:8.4f}  ({elapsed:.1f}s)")

    # Convergence table: P(j | sector 00, cascade)
    print(f"\n  --- P(j | sector 00, cascade) ---")
    header = f"  {'K':>3s}"
    for j in range(2, 11):
        header += f"  {'j=' + str(j):>8s}"
    print(header)
    for K in K_values:
        per_depth, _, _ = all_exact[K]
        n_sec = sum(v[0] for (s, d), v in per_depth.items() if s == '00')
        line = f"  {K:3d}"
        for j in range(2, 11):
            key = ('00', j)
            if key in per_depth:
                line += f"  {per_depth[key][0] / n_sec:8.4f}"
            else:
                line += f"  {'---':>8s}"
        print(line)
    # Tree line
    line = f"  {'T':>3s}"
    for j in range(2, 11):
        td = tree_data.get('00', {}).get(j, (0, 0))
        if td[0] > 0.001:
            line += f"  {td[0]:8.4f}"
        else:
            line += f"  {'---':>8s}"
    print(line + "  ← tree approx")

    # Convergence table: P(j | sector 10, cascade)
    print(f"\n  --- P(j | sector 10, cascade) ---")
    header = f"  {'K':>3s}"
    for j in range(2, 11):
        header += f"  {'j=' + str(j):>8s}"
    print(header)
    for K in K_values:
        per_depth, _, _ = all_exact[K]
        n_sec = sum(v[0] for (s, d), v in per_depth.items() if s == '10')
        if n_sec == 0:
            continue
        line = f"  {K:3d}"
        for j in range(2, 11):
            key = ('10', j)
            if key in per_depth:
                line += f"  {per_depth[key][0] / n_sec:8.4f}"
            else:
                line += f"  {'---':>8s}"
        print(line)
    line = f"  {'T':>3s}"
    for j in range(2, 11):
        td = tree_data.get('10', {}).get(j, (0, 0))
        if td[0] > 0.001:
            line += f"  {td[0]:8.4f}"
        else:
            line += f"  {'---':>8s}"
    print(line + "  ← tree approx")

    # Convergence table: E[val | j, sector, cascade]
    print(f"\n  --- E[val | j, sector 00, cascade] ---")
    header = f"  {'K':>3s}"
    for j in range(2, 11):
        header += f"  {'j=' + str(j):>8s}"
    print(header)
    for K in K_values:
        per_depth, _, _ = all_exact[K]
        line = f"  {K:3d}"
        for j in range(2, 11):
            key = ('00', j)
            if key in per_depth and per_depth[key][0] > 0:
                line += f"  {per_depth[key][1] / per_depth[key][0]:8.4f}"
            else:
                line += f"  {'---':>8s}"
        print(line)

    print(f"\n  --- E[val | j, sector 10, cascade] ---")
    header = f"  {'K':>3s}"
    for j in range(2, 11):
        header += f"  {'j=' + str(j):>8s}"
    print(header)
    for K in K_values:
        per_depth, _, _ = all_exact[K]
        line = f"  {K:3d}"
        for j in range(2, 11):
            key = ('10', j)
            if key in per_depth and per_depth[key][0] > 0:
                line += f"  {per_depth[key][1] / per_depth[key][0]:8.4f}"
            else:
                line += f"  {'---':>8s}"
        print(line)

    # ==================================================================
    # PART C: Sector ratio and limiting constants
    # ==================================================================
    print(f"\n\n{'='*72}")
    print("PART C: SECTOR RATIO R(K) = Σval_10 / Σval_00")
    print("=" * 72)

    print(f"\n  {'K':>3s}  {'R(K)':>10s}  {'gap':>10s}  {'n00/n10':>8s}")
    for K in K_values:
        per_depth, n_dodd, n_cascade = all_exact[K]
        v00 = sum(v[1] for (s, d), v in per_depth.items() if s == '00')
        v10 = sum(v[1] for (s, d), v in per_depth.items() if s == '10')
        n00 = sum(v[0] for (s, d), v in per_depth.items() if s == '00')
        n10 = sum(v[0] for (s, d), v in per_depth.items() if s == '10')
        R = v10 / v00 if abs(v00) > 1e-10 else float('nan')
        ratio = n00 / n10 if n10 > 0 else float('nan')
        print(f"  {K:3d}  {R:10.6f}  {R + pi:10.6f}  {ratio:8.3f}")

    # Per-depth R_j(K) = Σval_10(j) / Σval_00(j)
    print(f"\n  --- Per-depth R_j(K) = Σval_10(j) / Σval_00(j) ---")
    header = f"  {'K':>3s}"
    for j in range(2, 11):
        header += f"  {'j=' + str(j):>8s}"
    print(header)
    for K in K_values:
        per_depth, _, _ = all_exact[K]
        line = f"  {K:3d}"
        for j in range(2, 11):
            k00 = ('00', j)
            k10 = ('10', j)
            if k00 in per_depth and per_depth[k00][1] != 0:
                v00j = per_depth[k00][1]
                v10j = per_depth.get(k10, [0, 0.0])[1]
                rj = v10j / v00j
                line += f"  {rj:8.4f}"
            else:
                line += f"  {'---':>8s}"
        print(line)

    # Per-depth weighted contribution: w_j = Σval_00(j) / Σval_00_total
    print(f"\n  --- Weight w_j(K) = Σval_00(j) / Σval_00 ---")
    header = f"  {'K':>3s}"
    for j in range(2, 11):
        header += f"  {'j=' + str(j):>8s}"
    print(header)
    for K in K_values:
        per_depth, _, _ = all_exact[K]
        v00_total = sum(v[1] for (s, d), v in per_depth.items() if s == '00')
        if abs(v00_total) < 1e-10:
            continue
        line = f"  {K:3d}"
        for j in range(2, 11):
            key = ('00', j)
            if key in per_depth and per_depth[key][1] != 0:
                wj = per_depth[key][1] / v00_total
                line += f"  {wj:8.4f}"
            else:
                line += f"  {'---':>8s}"
        print(line)

    print(f"\n  R(K) = Σ_j w_j * R_j = Σ_j [Σval_00(j)/Σval_00] * [Σval_10(j)/Σval_00(j)]")
    print(f"       = Σ_j Σval_10(j) / Σval_00 = Σval_10 / Σval_00  ✓")
    print(f"\n  Target: R(∞) = -π/2 = {-pi/2:.6f}  (if v10=v01 and R=v10/v00)")
    print(f"  or R(∞) = -π = {-pi:.6f}  (if R includes both non-00 sectors)")


if __name__ == '__main__':
    main()
