#!/usr/bin/env python3
"""
E218: D-Even Control — The Rosetta Stone Test

QUESTION: What is R(K) for D-even products (D = 2K bits)?

If the Dirichlet boundary condition (D-odd → carry[D-1] = 0) is
the SOURCE of π, then changing the boundary condition should change
the constant.  D-even products have a different constraint structure:
the product has 2K bits, with carry[2K-1] = 1 (the leading bit IS
the carry).  This imposes a NEUMANN-like boundary at the top
instead of Dirichlet.

WHAT WE MEASURE:
  1. R_sb(K) and R_cas(K) for D-even products
  2. Convergence behavior — does R_cas converge? To what constant?
  3. If R_cas → c ≠ −π, what is c?
     - Another rational multiple of π?
     - A logarithmic constant (like R₀)?
     - Something with √2, ln2, etc.?

The D-even cascade readout:
  For D-even, carry[D-1] = 1 (not 0). The cascade searches from the
  top for the first carry that deviates from this running value.
  Specifically: cascade_val = carry[M-1] where M is the topmost
  position where carry transitions away from the terminal pattern.

ALSO MEASURED: D-even vs D-odd R₀ (schoolbook) to verify they differ.
"""

import numpy as np
import math
import time

PI = math.pi


def pr(s=""):
    print(s, flush=True)


def enumerate_both_parities(K):
    """
    Exact K-bit multiplication, separated by D-parity.

    D-odd:  product P has 2K-1 bits  (2^{2K-2} ≤ P < 2^{2K-1})
    D-even: product P has 2K bits    (2^{2K-1} ≤ P < 2^{2K})

    For both: compute schoolbook val (carry at position D-2)
    and cascade val (carry at topmost nonzero position, minus 1).

    We use the FULL Y range [2^{K-1}, 2^K) for both parities,
    since D-even products need not be restricted to c1=0 sector.
    But for consistency with E215 D-odd convention, we restrict
    Y to the c1=0 half: Y in [2^{K-1}, 2^{K-1} + 2^{K-2}).
    """
    D_odd = 2 * K - 1
    D_even = 2 * K

    X_lo, X_hi = 1 << (K - 1), 1 << K
    Y_lo = X_lo
    Y_hi = X_lo + (1 << (K - 2))

    Dodd_lo, Dodd_hi = 1 << (D_odd - 1), 1 << D_odd
    Deven_lo, Deven_hi = 1 << (D_even - 1), 1 << D_even

    stats = {}
    for parity in ['odd', 'even']:
        stats[parity] = {
            'n00': 0, 'n10': 0,
            'sb_00': 0, 'sb_10': 0,
            'cas_00_tot': 0, 'cas_10_tot': 0,
            'D': D_odd if parity == 'odd' else D_even,
        }

    Y_arr = np.arange(Y_lo, Y_hi, dtype=np.int64)

    for X in range(X_lo, X_hi):
        a1 = (X >> (K - 2)) & 1
        P_arr = X * Y_arr

        is_dodd  = (P_arr >= Dodd_lo)  & (P_arr < Dodd_hi)
        is_deven = (P_arr >= Deven_lo) & (P_arr < Deven_hi)

        for parity, mask in [('odd', is_dodd), ('even', is_deven)]:
            n_valid = int(mask.sum())
            if n_valid == 0:
                continue

            D = stats[parity]['D']
            valid_Y = Y_arr[mask]

            carries = np.zeros((D + 2, n_valid), dtype=np.int32)
            for j in range(D):
                conv_j = np.zeros(n_valid, dtype=np.int32)
                lo, hi = max(0, j - K + 1), min(j, K - 1)
                for i in range(lo, hi + 1):
                    if (X >> i) & 1:
                        conv_j += ((valid_Y >> (j - i)) & 1).astype(np.int32)
                carries[j + 1] = (conv_j + carries[j]) >> 1

            sb_pos = D - 2
            if sb_pos >= 0:
                sb_val = 2 * carries[sb_pos] - 1
            else:
                sb_val = np.zeros(n_valid, dtype=np.int32)

            M_arr = np.zeros(n_valid, dtype=np.int32)
            for j in range(D, 0, -1):
                update = (carries[j] > 0) & (M_arr == 0)
                M_arr[update] = j

            cas_val = np.zeros(n_valid, dtype=np.int32)
            vi = np.arange(n_valid)
            mask_M = M_arr > 0
            if mask_M.any():
                cas_val[mask_M] = carries[M_arr[mask_M] - 1, vi[mask_M]] - 1

            s = stats[parity]
            if a1 == 0:
                s['n00'] += n_valid
                s['sb_00'] += int(sb_val.sum())
                s['cas_00_tot'] += int(cas_val[mask_M].sum())
            else:
                s['n10'] += n_valid
                s['sb_10'] += int(sb_val.sum())
                s['cas_10_tot'] += int(cas_val[mask_M].sum())

    return stats


def main():
    pr()
    pr("E218: D-EVEN CONTROL — THE ROSETTA STONE")
    pr("=" * 72)
    pr()
    pr("  D-odd:  product has 2K-1 bits, carry[D-1] = 0 (Dirichlet top BC)")
    pr("  D-even: product has 2K bits,   carry[D-1] = 1 (Neumann-like top)")
    pr()
    pr("  If π comes from Dirichlet BCs, D-even should give a DIFFERENT constant.")
    pr()

    # ─── Phase 1: Enumeration ──────────────────────────────────
    pr("PHASE 1: ENUMERATION")
    pr("─" * 72)
    pr()
    pr(f"  {'K':>3s}  {'n_odd':>10s}  {'R_sb_odd':>10s}  {'R_cas_odd':>10s}  "
       f"{'gap_odd':>10s}  │  {'n_even':>10s}  {'R_sb_even':>10s}  "
       f"{'R_cas_even':>10s}")

    results = []
    for K in range(4, 16):
        t0 = time.time()
        stats = enumerate_both_parities(K)
        dt = time.time() - t0

        row = f"  {K:3d}"

        for parity in ['odd', 'even']:
            s = stats[parity]
            n_total = s['n00'] + s['n10']
            R_sb = s['sb_10'] / s['sb_00'] if s['sb_00'] != 0 else float('nan')
            R_cas = (s['cas_10_tot'] / s['cas_00_tot']
                     if s['cas_00_tot'] != 0 else float('nan'))

            if parity == 'odd':
                gap = R_cas + PI
                row += (f"  {n_total:10d}  {R_sb:+10.4f}  {R_cas:+10.4f}  "
                        f"{gap:+10.4f}  │")
                stats[parity]['R_sb'] = R_sb
                stats[parity]['R_cas'] = R_cas
            else:
                row += (f"  {n_total:10d}  {R_sb:+10.4f}  "
                        f"{R_cas:+10.4f}")
                stats[parity]['R_sb'] = R_sb
                stats[parity]['R_cas'] = R_cas

        pr(row + f"  [{dt:.1f}s]")
        results.append(stats)

        if dt > 600:
            pr("  [time limit]")
            break

    # ─── Phase 2: Convergence analysis ─────────────────────────
    pr()
    pr("PHASE 2: CONVERGENCE ANALYSIS")
    pr("─" * 72)
    pr()

    pr("  D-ODD cascade (should → −π):")
    pr(f"  {'K':>3s}  {'R_cas':>12s}  {'R+π':>12s}  {'ratio':>10s}")
    prev_gap = None
    for i, s in enumerate(results):
        K = i + 4
        R = s['odd']['R_cas']
        gap = R + PI
        ratio = gap / prev_gap if prev_gap and abs(prev_gap) > 1e-15 else float('nan')
        ratio_str = f"{ratio:.4f}" if not math.isnan(ratio) else "—"
        pr(f"  {K:3d}  {R:+12.6f}  {gap:+12.6f}  {ratio_str:>10s}")
        prev_gap = gap

    pr()
    pr("  D-EVEN cascade:")
    pr(f"  {'K':>3s}  {'R_cas':>12s}  {'ΔR(K)':>12s}  {'ratio':>10s}")
    prev_gap_e = None
    even_R_values = []
    for i, s in enumerate(results):
        K = i + 4
        R = s['even']['R_cas']
        even_R_values.append(R)
        if len(even_R_values) >= 2:
            delta = R - even_R_values[-2]
            ratio = delta / prev_gap_e if prev_gap_e and abs(prev_gap_e) > 1e-15 else float('nan')
            ratio_str = f"{ratio:.4f}" if not math.isnan(ratio) else "—"
        else:
            delta = float('nan')
            ratio_str = "—"
        pr(f"  {K:3d}  {R:+12.6f}  {delta:+12.6f}  {ratio_str:>10s}")
        prev_gap_e = delta if not math.isnan(delta) else None

    # ─── Phase 3: Identify D-even limit ────────────────────────
    pr()
    pr("PHASE 3: WHAT IS THE D-EVEN LIMIT?")
    pr("─" * 72)
    pr()

    if len(even_R_values) >= 3:
        R_last = even_R_values[-1]
        R_prev = even_R_values[-2]
        R_pprev = even_R_values[-3]

        d1 = R_last - R_prev
        d2 = R_prev - R_pprev

        if abs(d2) > 1e-15 and abs(d1) > 1e-15:
            rho_est = d1 / d2
            if abs(1 - rho_est) > 1e-10:
                R_extrap = R_last + d1 / (1 - rho_est)
                pr(f"  Aitken extrapolation: R_∞ ≈ {R_extrap:.8f}")
                pr()

                candidates = [
                    ("−π",        -PI),
                    ("−π/2",      -PI/2),
                    ("−2π",       -2*PI),
                    ("−3π/2",     -3*PI/2),
                    ("−π²/3",     -PI**2/3),
                    ("−e",        -math.e),
                    ("−3",        -3.0),
                    ("−4",        -4.0),
                    ("−2",        -2.0),
                    ("−ln2",      -math.log(2)),
                    ("−2ln2",     -2*math.log(2)),
                    ("−3ln2",     -3*math.log(2)),
                    ("−ln3",      -math.log(3)),
                    ("−4ln2",     -4*math.log(2)),
                    ("−π·ln2",    -PI*math.log(2)),
                    ("0",          0.0),
                    ("+1",        +1.0),
                    ("+π",        +PI),
                    ("1−4ln2",    1 - 4*math.log(2)),
                    ("2−4ln2",    2 - 4*math.log(2)),
                    ("−1−2ln2",   -1 - 2*math.log(2)),
                    ("1/2−2ln2",  0.5 - 2*math.log(2)),
                ]

                pr("  Candidate matching:")
                for name, val in sorted(candidates, key=lambda x: abs(x[1] - R_extrap)):
                    gap = R_extrap - val
                    pr(f"    {name:>12s} = {val:+12.8f}  gap = {gap:+.8f}")
                    if abs(gap) < 0.01:
                        break
            else:
                pr(f"  Convergence ratio ≈ 1: series may not converge geometrically.")
        else:
            pr("  Insufficient convergence data for extrapolation.")
    else:
        pr("  Need more K values for extrapolation.")

    # ─── Phase 4: Schoolbook comparison ────────────────────────
    pr()
    pr("PHASE 4: SCHOOLBOOK LIMITS (R₀)")
    pr("─" * 72)
    pr()
    pr("  R₀ = schoolbook val ratio. Both D-odd and D-even should converge")
    pr("  to specific logarithmic constants (no π).")
    pr()
    pr(f"  {'K':>3s}  {'R₀(odd)':>12s}  {'R₀(even)':>12s}  {'diff':>12s}")
    for i, s in enumerate(results):
        K = i + 4
        R_odd = s['odd']['R_sb']
        R_even = s['even']['R_sb']
        pr(f"  {K:3d}  {R_odd:+12.6f}  {R_even:+12.6f}  "
           f"{R_even - R_odd:+12.6f}")

    # ─── Verdict ───────────────────────────────────────────────
    pr()
    pr("=" * 72)
    pr("VERDICT")
    pr("=" * 72)
    pr()

    R_odd_last = results[-1]['odd']['R_cas']
    R_even_last = results[-1]['even']['R_cas']

    pr(f"  R_cas(odd,  K={len(results)+3}) = {R_odd_last:+.8f}  "
       f"(→ −π = {-PI:.8f})")
    pr(f"  R_cas(even, K={len(results)+3}) = {R_even_last:+.8f}")
    pr()

    if abs(R_even_last - R_odd_last) < 0.1:
        pr("  D-even and D-odd cascade ratios are SIMILAR.")
        pr("  The boundary condition may NOT be the source of π.")
    else:
        pr("  D-even and D-odd cascade ratios DIFFER SIGNIFICANTLY.")
        pr("  The boundary condition IS a key ingredient.")
        if abs(R_even_last + PI) < 0.5:
            pr("  But D-even also converges near −π — same universality class?")
        else:
            pr("  D-even converges to a DIFFERENT constant.")
            pr("  This is strong evidence for boundary-condition dependence.")

    pr()


if __name__ == "__main__":
    main()
