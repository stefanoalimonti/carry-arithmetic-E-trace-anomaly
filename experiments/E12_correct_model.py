#!/usr/bin/env python3
"""E12: Correct model — build transfer matrix matching the ACTUAL problem.

The ACTUAL problem (from exact enumeration) is:
  - D-odd pairs: K-bit numbers p, q with product p*q having exactly D=2K-1 bits
  - Carry chain: carries[0] = 0, carries[j+1] = (conv_j + carries[j]) >> 1
    where conv_j = Σ_i p_i * q_{j-i}
  - M = top non-zero carry position
  - val = c_{M-1} - 1  (the "carry weight" / "trace anomaly")
  - Sectors: a = p_{K-2}, c = q_{K-2} (second MSB of each factor)
  - S_{ac}(K) = Σ val over D-odd pairs in sector (a,c)
  - R(K) = S10(K) / S00(K) → -π as K → ∞

KEY INSIGHT FROM E11-E12:
  The previous transfer matrices  modeled a GENERIC
  carry chain, NOT the specific D-odd multiplication problem. The sectors
  were misidentified as carry boundaries; they're actually INPUT digit sectors.

THIS EXPERIMENT:
  Part A: Python verification matching reference data for K=3..7
  Part B: Analyze the D-odd + carry weight structure
  Part C: Express R(K) in terms of carry chain properties
  Part D: Identify the bridge / Green's function connection
"""
import sys
import time
import numpy as np
from mpmath import mp, mpf, pi, log

mp.dps = 30


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


E12_DATA = {
    3:  {'S00_c0': 0, 'S00_c1': -1, 'S10': 0},
    4:  {'S00_c0': -1, 'S00_c1': -3, 'S10': -1},
    5:  {'S00_c0': -6, 'S00_c1': -6, 'S10': -3},
    6:  {'S00_c0': -32, 'S00_c1': -11, 'S10': -4},
    7:  {'S00_c0': -114, 'S00_c1': -28, 'S10': 13},
    8:  {'S00_c0': -373, 'S00_c1': -61, 'S10': 167},
    9:  {'S00_c0': -1155, 'S00_c1': -171, 'S10': 963},
    10: {'S00_c0': -3684, 'S00_c1': -472, 'S10': 4648},
}


def compute_sectors(K):
    """Replicate the reference computation in Python for verification."""
    lo = 1 << (K - 1)
    hi = 1 << K
    D = 2 * K - 1

    S00_c0, S00_c1, S10, S01 = 0, 0, 0, 0
    n00_c0, n00_c1, n10, n01 = 0, 0, 0, 0

    for p in range(lo, hi):
        a = (p >> (K - 2)) & 1
        for q in range(lo, hi):
            c = (q >> (K - 2)) & 1
            if a & c:
                continue

            prod = p * q
            if (prod >> (D - 1)) != 1 or (prod >> D) != 0:
                continue

            carries = [0] * (D + 1)
            for j in range(D):
                conv_j = 0
                i_lo = max(0, j - K + 1)
                i_hi = min(j, K - 1)
                for i in range(i_lo, i_hi + 1):
                    conv_j += ((p >> i) & 1) * ((q >> (j - i)) & 1)
                carries[j + 1] = (conv_j + carries[j]) >> 1

            M = 0
            cm1 = 0
            for j in range(D, 0, -1):
                if carries[j] > 0:
                    M = j
                    cm1 = carries[j - 1]
                    break
            if M == 0:
                continue

            val = cm1 - 1
            c_top = carries[D - 2]

            if a == 0 and c == 0:
                if c_top == 0:
                    S00_c0 += val
                    n00_c0 += 1
                else:
                    S00_c1 += val
                    n00_c1 += 1
            elif a == 1 and c == 0:
                S10 += val
                n10 += 1
            else:
                S01 += val
                n01 += 1

    return {
        'S00_c0': S00_c0, 'S00_c1': S00_c1, 'S10': S10, 'S01': S01,
        'n00_c0': n00_c0, 'n00_c1': n00_c1, 'n10': n10, 'n01': n01,
    }


def main():
    t0 = time.time()

    pr("=" * 76)
    pr("E12: CORRECT MODEL — MATCHING THE ACTUAL D-ODD SECTOR PROBLEM")
    pr("=" * 76)

    # ═══════ Part A: Verify against reference data ═══════════════════════════
    pr(f"\n{'═' * 76}")
    pr("A. VERIFICATION: Python vs reference data C code")
    pr(f"{'═' * 76}\n")

    for K in range(3, 9):
        tb = time.time()
        result = compute_sectors(K)
        dt = time.time() - tb

        S00 = result['S00_c0'] + result['S00_c1']
        S10 = result['S10']
        S01 = result['S01']
        Nt = 4 ** (K - 1)
        R = S10 / S00 if S00 != 0 else float('inf')

        expected = E12_DATA.get(K, {})
        match_c0 = result['S00_c0'] == expected.get('S00_c0', '?')
        match_c1 = result['S00_c1'] == expected.get('S00_c1', '?')
        match_10 = result['S10'] == expected.get('S10', '?')

        pr(f"  K={K}: S00_c0={result['S00_c0']:8d} {'✓' if match_c0 else '✗'} "
           f"S00_c1={result['S00_c1']:8d} {'✓' if match_c1 else '✗'} "
           f"S10={result['S10']:8d} {'✓' if match_10 else '✗'}  "
           f"S00={S00:8d} R={R:+.6f} ({dt:.2f}s)")

    # ═══════ Part B: D-odd statistics ══════════════════════════════
    pr(f"\n{'═' * 76}")
    pr("B. D-ODD PAIR STATISTICS")
    pr(f"{'═' * 76}\n")

    for K in range(3, 8):
        lo = 1 << (K - 1)
        hi = 1 << K
        D = 2 * K - 1

        n_pairs = 0
        n_dodd = 0
        carry_weights = {(a, c): [] for a in range(2) for c in range(2) if not (a & c)}
        carry_chains = {(a, c): [] for a in range(2) for c in range(2) if not (a & c)}

        for p in range(lo, hi):
            a = (p >> (K - 2)) & 1
            for q in range(lo, hi):
                c = (q >> (K - 2)) & 1
                if a & c:
                    continue
                n_pairs += 1

                prod = p * q
                if (prod >> (D - 1)) != 1 or (prod >> D) != 0:
                    continue
                n_dodd += 1

                carries = [0] * (D + 1)
                for j in range(D):
                    conv_j = 0
                    i_lo = max(0, j - K + 1)
                    i_hi = min(j, K - 1)
                    for i in range(i_lo, i_hi + 1):
                        conv_j += ((p >> i) & 1) * ((q >> (j - i)) & 1)
                    carries[j + 1] = (conv_j + carries[j]) >> 1

                M = 0
                cm1 = 0
                for j in range(D, 0, -1):
                    if carries[j] > 0:
                        M = j
                        cm1 = carries[j - 1]
                        break
                if M == 0:
                    continue

                carry_weights[(a, c)].append(cm1 - 1)
                carry_chains[(a, c)].append(carries[:D + 1])

        pr(f"  K={K}: D={D}, pairs={n_pairs}, D-odd={n_dodd} "
           f"({100*n_dodd/n_pairs:.1f}%)")

        for (a, c) in [(0, 0), (1, 0), (0, 1)]:
            ws = carry_weights[(a, c)]
            if not ws:
                continue
            ws = np.array(ws)
            mean_w = np.mean(ws)
            sum_w = np.sum(ws)
            pr(f"    ({a},{c}): count={len(ws):5d}, Σval={sum_w:6d}, "
               f"⟨val⟩={mean_w:+.4f}")
        pr()

    # ═══════ Part C: Carry chain structure by sector ═══════════════
    pr(f"\n{'═' * 76}")
    pr("C. CARRY CHAIN PROFILE BY SECTOR")
    pr(f"{'═' * 76}\n")

    for K in [5, 6, 7]:
        lo = 1 << (K - 1)
        hi = 1 << K
        D = 2 * K - 1

        profiles = {(0, 0): [], (1, 0): []}

        for p in range(lo, hi):
            a = (p >> (K - 2)) & 1
            for q in range(lo, hi):
                c = (q >> (K - 2)) & 1
                if a & c:
                    continue

                prod = p * q
                if (prod >> (D - 1)) != 1 or (prod >> D) != 0:
                    continue

                carries = [0] * (D + 1)
                for j in range(D):
                    conv_j = 0
                    i_lo = max(0, j - K + 1)
                    i_hi = min(j, K - 1)
                    for i in range(i_lo, i_hi + 1):
                        conv_j += ((p >> i) & 1) * ((q >> (j - i)) & 1)
                    carries[j + 1] = (conv_j + carries[j]) >> 1

                M = 0
                for j in range(D, 0, -1):
                    if carries[j] > 0:
                        M = j
                        break
                if M == 0:
                    continue

                sector = (a, c)
                if sector in profiles:
                    profiles[sector].append(carries[:D + 1])

        pr(f"  K={K} (D={D}):")
        for sector in [(0, 0), (1, 0)]:
            chains = profiles[sector]
            if not chains:
                continue
            arr = np.array(chains, dtype=float)
            mean_profile = arr.mean(axis=0)
            pr(f"    Sector {sector}: {len(chains)} chains")
            pr(f"    Mean carry profile: ", end='')
            for j in range(D + 1):
                if j > 0 and j % 10 == 0:
                    pr(f"\n{'':>27}", end='')
                pr(f"{mean_profile[j]:.3f} ", end='')
            pr()

        diff = (np.array(profiles[(1, 0)], dtype=float).mean(axis=0) -
                np.array(profiles[(0, 0)], dtype=float).mean(axis=0))
        pr(f"    Δ(carry) = profile(1,0) - profile(0,0):")
        pr(f"      ", end='')
        for j in range(D + 1):
            if j > 0 and j % 10 == 0:
                pr(f"\n      ", end='')
            pr(f"{diff[j]:+.4f} ", end='')
        pr()
        pr()

    # ═══════ Part D: The bridge interpretation ═════════════════════
    pr(f"\n{'═' * 76}")
    pr("D. BRIDGE INTERPRETATION: WHERE IS THE SIN² SHIFT?")
    pr(f"{'═' * 76}\n")

    pr("  The D-odd carry chain is NOT a simple bridge with c₀=0, c_L=0.")
    pr("  carries[0] = 0 always (no initial carry in multiplication).")
    pr("  M = top non-zero carry, val = c_{M-1} - 1.")
    pr()
    pr("  The SECTORS are determined by input digits (a = p_{K-2}, c = q_{K-2}),")
    pr("  not by carry boundary conditions.")
    pr()
    pr("  The sector difference a=1 vs a=0 changes bit p_{K-2} from 0 to 1.")
    pr("  This adds 1 to conv_j for j around K-2 (specifically j = K-2 + i")
    pr("  for all i where q_i = 1).")
    pr()
    pr("  This is a LOCALIZED PERTURBATION at position K-2 in the carry chain.")
    pr("  The perturbation propagates through the carry chain, and its")
    pr("  cumulative effect on val = c_{M-1} - 1 determines S10 - S00.")
    pr()
    pr("  KEY QUESTION: can we express the carry chain propagator")
    pr("  as a Green's function with the bridge eigenvalue structure?")

    # ═══════ Part E: Perturbation analysis ═════════════════════════
    pr(f"\n{'═' * 76}")
    pr("E. PERTURBATION: EFFECT OF FLIPPING a=0 → a=1")
    pr(f"{'═' * 76}\n")

    pr("  For each D-odd pair in sector (0,0), what happens if we flip")
    pr("  p_{K-2} from 0 to 1 (moving to sector (1,0))?")
    pr()

    K = 6
    lo = 1 << (K - 1)
    hi = 1 << K
    D = 2 * K - 1

    flip_effects = []
    for p in range(lo, hi):
        a = (p >> (K - 2)) & 1
        if a != 0:
            continue
        for q in range(lo, hi):
            c = (q >> (K - 2)) & 1
            if c != 0:
                continue

            p_flip = p | (1 << (K - 2))
            a_flip = (p_flip >> (K - 2)) & 1
            assert a_flip == 1

            prod0 = p * q
            prod1 = p_flip * q

            dodd0 = (prod0 >> (D - 1)) == 1 and (prod0 >> D) == 0
            dodd1 = (prod1 >> (D - 1)) == 1 and (prod1 >> D) == 0

            def get_val(pp, qq):
                carries = [0] * (D + 1)
                for j in range(D):
                    conv_j = 0
                    i_lo = max(0, j - K + 1)
                    i_hi = min(j, K - 1)
                    for i in range(i_lo, i_hi + 1):
                        conv_j += ((pp >> i) & 1) * ((qq >> (j - i)) & 1)
                    carries[j + 1] = (conv_j + carries[j]) >> 1
                M = 0
                cm1 = 0
                for j in range(D, 0, -1):
                    if carries[j] > 0:
                        M = j
                        cm1 = carries[j - 1]
                        break
                return cm1 - 1 if M > 0 else None, carries

            val0, chain0 = get_val(p, q)
            val1, chain1 = get_val(p_flip, q)

            if val0 is not None or val1 is not None:
                flip_effects.append({
                    'p': p, 'q': q,
                    'dodd0': dodd0, 'dodd1': dodd1,
                    'val0': val0, 'val1': val1,
                    'chain_diff': [chain1[j] - chain0[j] for j in range(D + 1)]
                })

    both_dodd = [e for e in flip_effects if e['dodd0'] and e['dodd1']]
    only0 = [e for e in flip_effects if e['dodd0'] and not e['dodd1']]
    only1 = [e for e in flip_effects if not e['dodd0'] and e['dodd1']]
    neither = [e for e in flip_effects if not e['dodd0'] and not e['dodd1']]

    pr(f"  K={K}: {len(flip_effects)} pairs analyzed")
    pr(f"    Both D-odd: {len(both_dodd)}")
    pr(f"    Only original D-odd: {len(only0)}")
    pr(f"    Only flipped D-odd: {len(only1)}")
    pr(f"    Neither D-odd: {len(neither)}")

    if both_dodd:
        val_diffs = [e['val1'] - e['val0'] for e in both_dodd
                     if e['val0'] is not None and e['val1'] is not None]
        pr(f"\n    For pairs where BOTH are D-odd:")
        pr(f"    Mean Δval = {np.mean(val_diffs):+.4f} "
           f"(std = {np.std(val_diffs):.4f})")
        chain_diffs = np.array([e['chain_diff'] for e in both_dodd], dtype=float)
        pr(f"    Mean carry chain difference:")
        pr(f"      ", end='')
        mean_cd = chain_diffs.mean(axis=0)
        for j in range(D + 1):
            pr(f"{mean_cd[j]:+.3f} ", end='')
        pr()

    pr(f"\n    The bit flip at position K-2={K-2} creates a LOCALIZED")
    pr(f"    carry perturbation that propagates through the chain.")

    pr(f"\n  Total runtime: {time.time() - t0:.1f}s")
    pr("=" * 76)


if __name__ == '__main__':
    main()
