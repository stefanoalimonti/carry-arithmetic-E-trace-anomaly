#!/usr/bin/env python3
"""E19: Identify closed form for N₁₀/N₀₀ (count ratio of D-odd pairs by sector).

From prior experiments: N₁₀/N₀₀ → ~0.320 as K → ∞.
Candidates: 1/π, (π-2)/π², 1/e, (3-π)/π, ln(2)/2, ...

Strategy:
  Part A: Compute exact N₁₀, N₀₀ for K = 3..19
  Part B: Richardson extrapolation to get 6+ digits of the limit
  Part C: PSLQ / constant recognition to identify closed form
"""
import sys
import time
import numpy as np
from mpmath import mp, mpf, pi as MPI, identify, pslq, log, e as ME, euler

mp.dps = 50


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def count_sectors(K):
    """Count D-odd pairs by sector. Returns (N00, N10, N01, N11)."""
    lo = 1 << (K - 1)
    hi = 1 << K
    D = 2 * K - 1
    mask_D = (1 << D) - 1
    bit_D1 = 1 << (D - 1)
    bit_D = 1 << D

    N = [[0, 0], [0, 0]]

    for p in range(lo, hi):
        a = (p >> (K - 2)) & 1
        for q in range(lo, hi):
            c = (q >> (K - 2)) & 1
            prod = p * q
            if prod >= bit_D or prod < bit_D1:
                continue
            N[a][c] += 1

    return N[0][0], N[1][0], N[0][1], N[1][1]


def count_sectors_fast(K):
    """Faster: use numpy vectorization for inner loop."""
    lo = 1 << (K - 1)
    hi = 1 << K
    D = 2 * K - 1
    bit_D1 = 1 << (D - 1)
    bit_D = 1 << D

    qs = np.arange(lo, hi, dtype=np.int64)
    c_bits = (qs >> (K - 2)) & 1

    N00 = 0
    N10 = 0
    N01 = 0
    N11 = 0

    for p in range(lo, hi):
        a = (p >> (K - 2)) & 1
        prods = p * qs
        dodd = (prods >= bit_D1) & (prods < bit_D)

        if a == 0:
            N00 += int(np.sum(dodd & (c_bits == 0)))
            N01 += int(np.sum(dodd & (c_bits == 1)))
        else:
            N10 += int(np.sum(dodd & (c_bits == 0)))
            N11 += int(np.sum(dodd & (c_bits == 1)))

    return N00, N10, N01, N11


def richardson(vals, Ks, order=None):
    """Richardson extrapolation assuming f(K) = L + a₁/K² + a₂/K⁴ + ..."""
    n = len(vals)
    if order is None:
        order = n - 1
    order = min(order, n - 1)

    tab = [list(vals)]
    ks = list(Ks)

    for j in range(1, order + 1):
        prev = tab[j - 1]
        new_row = []
        for i in range(len(prev) - 1):
            h1 = 1.0 / ks[i] ** 2
            h2 = 1.0 / ks[i + j] ** 2
            val = (h1 * prev[i + 1] - h2 * prev[i]) / (h1 - h2)
            new_row.append(val)
        tab.append(new_row)

    return tab


def main():
    t0 = time.time()
    pr("=" * 76)
    pr("E19: COUNT RATIO N₁₀/N₀₀ — CLOSED FORM IDENTIFICATION")
    pr("=" * 76)

    # ═══════ Part A: Compute exact counts ═════════════════════════════
    pr(f"\n{'═' * 76}")
    pr("A. EXACT SECTOR COUNTS")
    pr(f"{'═' * 76}\n")

    data = []
    K_max = 19
    for K in range(3, K_max + 1):
        ts = time.time()
        if K <= 14:
            N00, N10, N01, N11 = count_sectors_fast(K)
        else:
            N00, N10, N01, N11 = count_sectors_fast(K)

        N_total = N00 + N10 + N01 + N11
        ratio = N10 / N00 if N00 > 0 else 0
        elapsed = time.time() - ts

        data.append({'K': K, 'N00': N00, 'N10': N10, 'N01': N01, 'N11': N11,
                     'ratio': ratio})

        pr(f"  K={K:2d}: N00={N00:12d}  N10={N10:12d}  N01={N01:12d}  N11={N11:12d}"
           f"  N10/N00={ratio:.10f}  ({elapsed:.1f}s)")

        if elapsed > 600:
            pr(f"  [Stopping at K={K}, next would be too slow]")
            K_max = K
            break

    # ═══════ Part B: Richardson extrapolation ══════════════════════════
    pr(f"\n{'═' * 76}")
    pr("B. RICHARDSON EXTRAPOLATION")
    pr(f"{'═' * 76}\n")

    Ks_odd = [d['K'] for d in data if d['K'] >= 5]
    vals_odd = [d['ratio'] for d in data if d['K'] >= 5]

    if len(vals_odd) >= 4:
        tab = richardson(vals_odd, Ks_odd)
        pr(f"  Raw sequence (K ≥ 5):")
        for i, (K, v) in enumerate(zip(Ks_odd, vals_odd)):
            pr(f"    K={K:2d}: N10/N00 = {v:.12f}")

        pr(f"\n  Richardson table (diagonal):")
        for j in range(len(tab)):
            if tab[j]:
                pr(f"    Order {j}: {tab[j][0]:.14f}"
                   + (f"  (last: {tab[j][-1]:.14f})" if len(tab[j]) > 1 else ""))

        best = tab[-1][0] if tab[-1] else tab[-2][0]
        pr(f"\n  Best Richardson estimate: {best:.14f}")
    else:
        best = vals_odd[-1] if vals_odd else 0.32
        pr(f"  Not enough data for Richardson, using last value: {best:.10f}")

    # Also try 1/K and 1/K ansatz
    pr(f"\n  Trying ansatz f(K) = L + a/K + b/K² (polynomial in 1/K):")
    if len(vals_odd) >= 3:
        tab2 = [list(vals_odd)]
        ks2 = list(Ks_odd)
        for j in range(1, min(len(vals_odd), 6)):
            prev = tab2[j - 1]
            new_row = []
            for i in range(len(prev) - 1):
                h1 = 1.0 / ks2[i]
                h2 = 1.0 / ks2[i + j]
                val = (h1 * prev[i + 1] - h2 * prev[i]) / (h1 - h2)
                new_row.append(val)
            tab2.append(new_row)

        for j in range(len(tab2)):
            if tab2[j]:
                pr(f"    Order {j} (1/K ansatz): {tab2[j][0]:.14f}"
                   + (f"  (last: {tab2[j][-1]:.14f})" if len(tab2[j]) > 1 else ""))

        best2 = tab2[-1][0] if tab2[-1] else tab2[-2][0]
        pr(f"  Best (1/K ansatz): {best2:.14f}")
    else:
        best2 = best

    # ═══════ Part C: Constant recognition ═════════════════════════════
    pr(f"\n{'═' * 76}")
    pr("C. CONSTANT RECOGNITION")
    pr(f"{'═' * 76}\n")

    L = mpf(str(best))
    L2 = mpf(str(best2))

    candidates = [
        ("1/π", 1 / MPI),
        ("1/e", 1 / ME),
        ("(π-2)/π²", (MPI - 2) / MPI**2),
        ("(3-π)/π", (3 - MPI) / MPI),
        ("ln(2)/2", log(2) / 2),
        ("1/3", mpf(1) / 3),
        ("2/3 - 1/π", mpf(2)/3 - 1/MPI),
        ("π/(3π+1)", MPI / (3*MPI + 1)),
        ("(4-π)/(4+π)", (4 - MPI) / (4 + MPI)),
        ("1/(1+π)", 1 / (1 + MPI)),
        ("(π-2)/(2π)", (MPI - 2) / (2 * MPI)),
        ("π/(π²+1)", MPI / (MPI**2 + 1)),
        ("2/(π+4)", 2 / (MPI + 4)),
        ("(π-3+1)/(π)", (MPI - 2) / MPI),
        ("ln(2)", log(2)),
        ("1-ln(2)", 1 - log(2)),
        ("π/10", MPI / 10),
        ("1/(π+1/6)", 1 / (MPI + mpf(1)/6)),
        ("2·ln(2)-1", 2*log(2) - 1),
        ("(e-2)/(e-1)", (ME - 2)/(ME - 1)),
        ("3/(3+π)", 3 / (3 + MPI)),
        ("π/(3(π+1))", MPI / (3*(MPI+1))),
        ("1/(2+ln2)", 1 / (2 + log(2))),
        ("(4-π)/4", (4 - MPI)/4),
        ("2/(2π-1)", 2 / (2*MPI - 1)),
    ]

    pr(f"  Target (1/K² ansatz): {L}")
    pr(f"  Target (1/K ansatz):  {L2}")
    pr()

    for label in ["1/K² Richardson", "1/K Richardson"]:
        target = L if "1/K²" in label else L2
        pr(f"  --- Candidates vs {label} = {float(target):.14f} ---")
        results = []
        for name, val in candidates:
            diff = abs(float(target) - float(val))
            results.append((diff, name, float(val)))
        results.sort()
        for diff, name, val in results[:10]:
            digits = -np.log10(max(diff, 1e-20))
            pr(f"    {name:20s} = {val:.14f}  diff={diff:.2e}  ({digits:.1f} digits)")
        pr()

    # PSLQ: try to express L as integer linear combination of constants
    pr(f"  --- PSLQ analysis ---")
    for label, target in [("1/K² Richardson", L), ("1/K Richardson", L2)]:
        pr(f"\n  {label} = {target}")

        constants_list = [
            ("1, π, π²", [mpf(1), MPI, MPI**2]),
            ("1, π, ln2", [mpf(1), MPI, log(2)]),
            ("1, 1/π, 1/π²", [mpf(1), 1/MPI, 1/MPI**2]),
            ("1, π, e", [mpf(1), MPI, ME]),
            ("1, π, 1/π", [mpf(1), MPI, 1/MPI]),
        ]

        for desc, consts in constants_list:
            vec = [target] + consts
            try:
                rel = pslq(vec)
                if rel is not None and rel[0] != 0:
                    expr_parts = []
                    for i, (coef, name) in enumerate(zip(rel[1:], desc.split(", "))):
                        if coef != 0:
                            expr_parts.append(f"{coef}·{name}")
                    rhs = " + ".join(expr_parts)
                    val_check = sum(c * v for c, v in zip(rel[1:], consts))
                    residual = float(abs(target + val_check / rel[0]))
                    pr(f"    PSLQ({desc}): {rel[0]}·L = {rhs}")
                    pr(f"      → L = {float(-val_check/rel[0]):.14f}  "
                       f"residual={residual:.2e}")
            except Exception as ex:
                pr(f"    PSLQ({desc}): failed — {ex}")

    # ═══════ Part D: Even/odd K subsequences ══════════════════════════
    pr(f"\n{'═' * 76}")
    pr("D. EVEN vs ODD K SUBSEQUENCES")
    pr(f"{'═' * 76}\n")

    for parity, label in [(0, "even"), (1, "odd")]:
        sub = [d for d in data if d['K'] % 2 == parity and d['K'] >= 5]
        if len(sub) >= 3:
            Ks_sub = [d['K'] for d in sub]
            vals_sub = [d['ratio'] for d in sub]
            pr(f"  {label}-K: {[(K, f'{v:.10f}') for K, v in zip(Ks_sub, vals_sub)]}")

            tab_sub = richardson(vals_sub, Ks_sub)
            if tab_sub[-1]:
                pr(f"    Richardson limit: {tab_sub[-1][0]:.14f}")
            elif len(tab_sub) >= 2 and tab_sub[-2]:
                pr(f"    Richardson limit: {tab_sub[-2][0]:.14f}")

    # ═══════ Part E: Differences and rates ════════════════════════════
    pr(f"\n{'═' * 76}")
    pr("E. CONVERGENCE RATE ANALYSIS")
    pr(f"{'═' * 76}\n")

    pr(f"  {'K':>3s}  {'N10/N00':>14s}  {'Δ(K)':>14s}  {'Δ·K':>10s}  "
       f"{'Δ·K²':>10s}  {'Δ·2^K':>14s}")

    for i in range(1, len(data)):
        K = data[i]['K']
        r = data[i]['ratio']
        r_prev = data[i-1]['ratio']
        delta = r - r_prev
        pr(f"  {K:3d}  {r:14.10f}  {delta:+14.10f}  {delta*K:+10.4f}  "
           f"{delta*K**2:+10.2f}  {delta*(2**K):+14.2f}")

    # ═══════ Part F: N00, N10 as fractions of total ═══════════════════
    pr(f"\n{'═' * 76}")
    pr("F. SECTOR FRACTIONS OF TOTAL D-ODD PAIRS")
    pr(f"{'═' * 76}\n")

    pr(f"  {'K':>3s}  {'N00/Ntot':>12s}  {'N10/Ntot':>12s}  {'N01/Ntot':>12s}  "
       f"{'N11/Ntot':>12s}  {'N10/N00':>12s}")
    for d in data:
        Nt = d['N00'] + d['N10'] + d['N01'] + d['N11']
        if Nt == 0:
            continue
        pr(f"  {d['K']:3d}  {d['N00']/Nt:12.8f}  {d['N10']/Nt:12.8f}  "
           f"{d['N01']/Nt:12.8f}  {d['N11']/Nt:12.8f}  {d['ratio']:12.8f}")

    pr(f"\n  Total runtime: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
