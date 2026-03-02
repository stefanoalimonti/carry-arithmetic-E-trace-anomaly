#!/usr/bin/env python3
"""E01: Base Universality — Does σ_10/σ_00 → -π in ALL bases?

CRITICAL TEST:
  In base 2: σ_10/σ_00 → -π (4.3 digits, confirmed to K=20).
  If this holds for base b ≠ 2, it's a UNIVERSAL theorem.
  If it holds ONLY for base 2, it confirms Angular Uniqueness.
  If the ratio depends on b, the b-dependence reveals the mechanism.

CONNECTION TO ζ(2):
  If σ_10/σ_00 = -π for ALL bases, and α = 1/6 for ALL bases,
  then α·π² = π²/6 = ζ(2) is a STRUCTURAL relation, not coincidence.

COMPUTATION:
  For base b, K-digit factors p, q ∈ [b^{K-1}, b^K):
  - Product: p·q ∈ [b^{2K-2}, b^{2K})
  - D-odd: p·q < b^{2K-1} (product has 2K-1 digits)
  - Carry chain: c_{j+1} = ⌊(conv_j + c_j)/b⌋
  - Sector: determined by digit K-2 of factors
  - σ_ac = Σ (c_{M-1} - 1) / b^{2(K-1)} over D-odd pairs in sector (a,c)
"""
import sys
import time
from collections import defaultdict

import mpmath
from mpmath import mpf, mp, log, pi

mp.dps = 30


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def digits_base(n, b, K):
    """Extract K digits of n in base b (LSB first)."""
    d = []
    for _ in range(K):
        d.append(n % b)
        n //= b
    return d


def sector_ratio_base(b, K):
    """Compute sector decomposition for base b, K-digit factors.

    Returns dict with sector sums, counts, and the ratio σ_10/σ_00.
    """
    lo = b ** (K - 1)
    hi = b ** K
    D = 2 * K - 1  # D-odd products have exactly D digits

    S = defaultdict(int)      # (a, c) → sum of (c_{M-1} - 1)
    cnt = defaultdict(int)    # (a, c) → count
    S_by_carry = defaultdict(int)  # (a, c, c_at_top) → sum
    cnt_by_carry = defaultdict(int)

    for p in range(lo, hi):
        p_digits = digits_base(p, b, K)
        a = p_digits[K - 2]  # second-most-significant digit

        for q in range(lo, hi):
            prod = p * q
            # D-odd: product has exactly D digits in base b
            if prod < b ** (D - 1) or prod >= b ** D:
                continue

            q_digits = digits_base(q, b, K)
            c_digit = q_digits[K - 2]

            # Compute carry chain
            carries = [0] * (D + 2)
            for j in range(D):
                conv_j = 0
                for i in range(max(0, j - K + 1), min(j + 1, K)):
                    conv_j += p_digits[i] * q_digits[j - i]
                carries[j + 1] = (conv_j + carries[j]) // b

            # Find M (top nonzero carry) and c_{M-1}
            M = 0
            cm1 = 0
            for j in range(D, 0, -1):
                if carries[j] > 0:
                    M = j
                    cm1 = carries[M - 1]
                    break

            if M == 0:
                continue

            val = cm1 - 1
            sec = (a, c_digit)
            S[sec] += val
            cnt[sec] += 1

            c_top = carries[D - 2]  # carry at penultimate position
            S_by_carry[(a, c_digit, c_top)] += val
            cnt_by_carry[(a, c_digit, c_top)] += 1

    Nt = b ** (2 * (K - 1))
    return {
        'S': dict(S), 'cnt': dict(cnt),
        'S_by_carry': dict(S_by_carry), 'cnt_by_carry': dict(cnt_by_carry),
        'Nt': Nt, 'b': b, 'K': K,
    }


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("E01: BASE UNIVERSALITY — DOES σ_10/σ_00 → -π IN ALL BASES?")
    pr("=" * 72)

    # ── Base 2 reference ───────────────────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("BASE 2 (reference)")
    pr(f"{'═' * 72}\n")

    b2_ratios = []
    for K in range(3, 14):
        t1 = time.time()
        r = sector_ratio_base(2, K)
        dt = time.time() - t1
        Nt = r['Nt']

        s00 = r['S'].get((0, 0), 0)
        s10 = r['S'].get((1, 0), 0)
        n00 = r['cnt'].get((0, 0), 0)
        n10 = r['cnt'].get((1, 0), 0)

        sig00 = mpf(s00) / Nt
        sig10 = mpf(s10) / Nt

        ratio = float(sig10 / sig00) if sig00 != 0 else float('nan')
        b2_ratios.append((K, ratio))

        # α = σ_c1/σ_c0 (carry at D-2)
        sc0 = r['S_by_carry'].get((0, 0, 0), 0)
        sc1 = sum(r['S_by_carry'].get((0, 0, cv), 0) for cv in range(1, 2))
        alpha = sc1 / sc0 if sc0 != 0 else float('nan')

        pr(f"  K={K:2d}: σ10/σ00 = {ratio:8.4f}  (n00={n00:8d}, n10={n10:8d}) "
           f" α={float(alpha):.4f}  [{dt:.2f}s]")

    pr(f"  Target: σ10/σ00 → -π = {float(-pi):.6f}")

    # ── Base 3 ─────────────────────────────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("BASE 3")
    pr(f"{'═' * 72}\n")

    b3_ratios = []
    K_max_3 = 9
    for K in range(3, K_max_3 + 1):
        t1 = time.time()
        r = sector_ratio_base(3, K)
        dt = time.time() - t1
        Nt = r['Nt']

        # In base 3, sectors (a,c) where a,c ∈ {0,1,2}
        # Analogue of (0,0) and (1,0):
        s00 = r['S'].get((0, 0), 0)
        s10 = r['S'].get((1, 0), 0)
        s20 = r['S'].get((2, 0), 0)
        n00 = r['cnt'].get((0, 0), 0)
        n10 = r['cnt'].get((1, 0), 0)
        n20 = r['cnt'].get((2, 0), 0)

        sig00 = mpf(s00) / Nt
        sig10 = mpf(s10) / Nt
        sig20 = mpf(s20) / Nt

        r10 = float(sig10 / sig00) if sig00 != 0 else float('nan')
        r20 = float(sig20 / sig00) if sig00 != 0 else float('nan')
        b3_ratios.append((K, r10, r20))

        # α for base 3
        sc0 = r['S_by_carry'].get((0, 0, 0), 0)
        sc_pos = sum(r['S_by_carry'].get((0, 0, cv), 0)
                     for cv in range(1, 3))
        alpha = sc_pos / sc0 if sc0 != 0 else float('nan')

        pr(f"  K={K:2d}: σ10/σ00={r10:8.4f}  σ20/σ00={r20:8.4f}  "
           f"(n00={n00:6d}) α={float(alpha):.4f}  [{dt:.1f}s]")

        if dt > 120:
            pr(f"  (stopping base 3 at K={K}, too slow)")
            break

    pr(f"  For reference: -π = {float(-pi):.6f}")

    # ── Base 5 ─────────────────────────────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("BASE 5")
    pr(f"{'═' * 72}\n")

    for K in range(3, 8):
        t1 = time.time()
        r = sector_ratio_base(5, K)
        dt = time.time() - t1
        Nt = r['Nt']

        s00 = r['S'].get((0, 0), 0)
        s10 = r['S'].get((1, 0), 0)
        s20 = r['S'].get((2, 0), 0)
        n00 = r['cnt'].get((0, 0), 0)

        sig00 = mpf(s00) / Nt
        sig10 = mpf(s10) / Nt
        sig20 = mpf(s20) / Nt

        r10 = float(sig10 / sig00) if sig00 != 0 else float('nan')
        r20 = float(sig20 / sig00) if sig00 != 0 else float('nan')

        sc0 = r['S_by_carry'].get((0, 0, 0), 0)
        sc_pos = sum(r['S_by_carry'].get((0, 0, cv), 0)
                     for cv in range(1, 5))
        alpha = sc_pos / sc0 if sc0 != 0 else float('nan')

        pr(f"  K={K:2d}: σ10/σ00={r10:8.4f}  σ20/σ00={r20:8.4f}  "
           f"(n00={n00:6d}) α={float(alpha):.4f}  [{dt:.1f}s]")

        if dt > 120:
            break

    # ── Base 7 ─────────────────────────────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("BASE 7")
    pr(f"{'═' * 72}\n")

    for K in range(3, 7):
        t1 = time.time()
        r = sector_ratio_base(7, K)
        dt = time.time() - t1
        Nt = r['Nt']

        s00 = r['S'].get((0, 0), 0)
        s10 = r['S'].get((1, 0), 0)
        n00 = r['cnt'].get((0, 0), 0)

        sig00 = mpf(s00) / Nt
        sig10 = mpf(s10) / Nt

        r10 = float(sig10 / sig00) if sig00 != 0 else float('nan')

        sc0 = r['S_by_carry'].get((0, 0, 0), 0)
        sc_pos = sum(r['S_by_carry'].get((0, 0, cv), 0)
                     for cv in range(1, 7))
        alpha = sc_pos / sc0 if sc0 != 0 else float('nan')

        pr(f"  K={K:2d}: σ10/σ00={r10:8.4f}  (n00={n00:6d}) "
           f"α={float(alpha):.4f}  [{dt:.1f}s]")

        if dt > 120:
            break

    # ── Base 10 ────────────────────────────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("BASE 10")
    pr(f"{'═' * 72}\n")

    for K in range(3, 6):
        t1 = time.time()
        r = sector_ratio_base(10, K)
        dt = time.time() - t1
        Nt = r['Nt']

        s00 = r['S'].get((0, 0), 0)
        s10 = r['S'].get((1, 0), 0)
        n00 = r['cnt'].get((0, 0), 0)

        sig00 = mpf(s00) / Nt
        sig10 = mpf(s10) / Nt

        r10 = float(sig10 / sig00) if sig00 != 0 else float('nan')

        sc0 = r['S_by_carry'].get((0, 0, 0), 0)
        sc_pos = sum(r['S_by_carry'].get((0, 0, cv), 0)
                     for cv in range(1, 10))
        alpha = sc_pos / sc0 if sc0 != 0 else float('nan')

        pr(f"  K={K:2d}: σ10/σ00={r10:8.4f}  (n00={n00:6d}) "
           f"α={float(alpha):.4f}  [{dt:.1f}s]")

        if dt > 120:
            break

    # ── Synthesis ──────────────────────────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("SYNTHESIS: IS σ_10/σ_00 → -π UNIVERSAL?")
    pr(f"{'═' * 72}\n")

    pr(f"  Base 2 (K=13): σ10/σ00 = {b2_ratios[-1][1]:.4f} → -π = {float(-pi):.4f}")
    if b3_ratios:
        pr(f"  Base 3 (K={b3_ratios[-1][0]}): σ10/σ00 = {b3_ratios[-1][1]:.4f}")
    pr()
    pr("  KEY QUESTION: Do the ratios converge to the same value (-π)?")
    pr("  Or do they depend on the base b?")
    pr()
    pr("  If UNIVERSAL: suggests deep connection to ζ(2) = π²/6")
    pr("  If BASE-SPECIFIC: confirms Angular Uniqueness of base 2")

    pr(f"\n  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
