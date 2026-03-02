#!/usr/bin/env python3
"""
E215: Reward Function Projection — Critical Test for Paper E §7

CRITICAL TEST: Does the cascade readout project onto χ₄(n) = sin(nπ/2)?

Paper E's Theorem 2 proof claims w_n = sin(nπ/2), equivalent to the reward
function being a delta at the midpoint of the Dirichlet chain. This is never
derived from the cascade stopping rule. We test it directly.

FOUR PARTS:
  1. Markov model: cascade stopping distribution → NOT proportional to χ₄
  2. Actual K-bit multiplication: per-position cascade profiles
  3. χ₄ projection test on the actual profiles (THE CRITICAL TEST)
  4. Phase Transition Formula convergence (finite-L verification)

THREE POSSIBLE OUTCOMES:
  A (Confirmed):  Projections converge to χ₄ → midpoint model justified
  B (Asymptotic): Projections approach χ₄ as K→∞ → limit conjecture needed
  C (Falsified):  Projections do NOT converge → midpoint model is wrong

RESULT: [FILLED AFTER RUNNING]
"""

import numpy as np
import math
import time

PI = math.pi


def pr(s=""):
    print(s, flush=True)


def project_sine(f, D):
    """
    Project f[k] (k=0..D) onto the Dirichlet sine basis:
        w_n = Σ_{k=1}^{D-1} f[k] · sin(nπk/D)
    for n = 1, ..., D-1.

    Convention: Dirichlet walls at k=0 and k=D.
    Interior points: k=1,...,D-1.
    Eigenvectors: φ_n(k) = sin(nπk/D), eigenvalues: cos(nπ/D)/2.
    """
    if D < 3:
        return np.array([])
    k = np.arange(1, D, dtype=np.float64)
    result = np.zeros(D - 1)
    for n in range(1, D):
        result[n - 1] = np.sum(f[1:D] * np.sin(n * PI * k / D))
    return result


# ═══════════════════════════════════════════════════════════════
# PART 1: MARKOV CARRY CHAIN
# ═══════════════════════════════════════════════════════════════

def part1():
    pr("=" * 72)
    pr("PART 1: MARKOV MODEL — Where does the cascade stop?")
    pr("=" * 72)
    pr()
    pr("  Carry chain: K = [[3/4, 1/4], [1/4, 3/4]] (base-2 addition)")
    pr("  Dirichlet BCs: c_0 = c_L = 0")
    pr("  M = max{d : c_d > 0} (cascade stopping = topmost nonzero carry)")
    pr()
    pr("  Analytical: P(M=k) = [(2^k-1)/2^{k+1}]·(1/4)·(3/4)^{L-k-1} / Z")
    pr()

    for L in [9, 13, 17, 25, 33, 65]:
        W = np.zeros(L + 1)
        for k in range(1, L):
            W[k] = ((2**k - 1) / 2**(k + 1)) * 0.25 * 0.75**(L - k - 1)
        W[0] = 0.75**L
        Z = (2**L + 1) / 2**(L + 1)
        W /= Z

        W_active = W.copy()
        W_active[0] = 0
        total = W_active.sum()
        if total < 1e-15:
            continue
        W_active /= total

        peak = np.argmax(W_active)
        midpoint = L / 2.0

        w = project_sine(W_active, L)
        chi4 = np.array([np.sin(n * PI / 2) for n in range(1, L)])

        ratios = []
        for n in [1, 3, 5, 7]:
            idx = n - 1
            if idx < len(w) and abs(chi4[idx]) > 1e-12:
                ratios.append(w[idx] / chi4[idx])

        cv = (np.std(ratios) / abs(np.mean(ratios))
              if len(ratios) >= 2 and abs(np.mean(ratios)) > 1e-15
              else float('inf'))

        pr(f"  L={L:3d}: peak at k={peak} (mid={midpoint:.1f}), P(all-zero)={W[0]:.3f}")
        for i, n in enumerate([1, 3, 5, 7][:len(ratios)]):
            pr(f"    w_{n}/χ₄({n}) = {ratios[i]:+.6f}")
        pr(f"    CoV = {cv:.4f} {'← NOT χ₄' if cv > 0.1 else ''}")
        pr()


# ═══════════════════════════════════════════════════════════════
# PART 2: ACTUAL K-BIT MULTIPLICATION
# ═══════════════════════════════════════════════════════════════

def enumerate_cascade(K):
    """
    Exact enumeration of K-bit multiplication.

    For each D-odd pair (X,Y) with X in [2^{K-1}, 2^K), Y in [2^{K-1}, 2^{K-1}+2^{K-2}):
      - Compute carry sequence from schoolbook multiplication
      - Find cascade stopping position M = topmost nonzero carry
      - Record per-position cascade val = carries[M-1] - 1 per sector

    Y restricted to first half → c1 = 0 (sectors (0,0) and (1,0) only).
    """
    D = 2 * K - 1
    X_lo = 1 << (K - 1)
    X_hi = 1 << K
    Y_lo = X_lo
    Y_hi = X_lo + (1 << (K - 2))  # c1 = 0 sector only

    D_lo = 1 << (D - 1)
    D_hi = 1 << D

    cas_00 = np.zeros(D + 2)
    cas_10 = np.zeros(D + 2)
    stop_00 = np.zeros(D + 2, dtype=np.int64)
    stop_10 = np.zeros(D + 2, dtype=np.int64)
    n00 = n10 = 0
    sb_00 = sb_10 = 0
    cas_00_tot = cas_10_tot = 0

    n_Y = Y_hi - Y_lo
    Y_arr = np.arange(Y_lo, Y_hi, dtype=np.int64)

    for X in range(X_lo, X_hi):
        a1 = (X >> (K - 2)) & 1

        P_arr = X * Y_arr
        dodd = (P_arr >= D_lo) & (P_arr < D_hi)
        n_valid = int(dodd.sum())
        if n_valid == 0:
            continue

        valid_Y = Y_arr[dodd]

        carries = np.zeros((D + 2, n_valid), dtype=np.int32)
        for j in range(D):
            conv_j = np.zeros(n_valid, dtype=np.int32)
            lo = max(0, j - K + 1)
            hi = min(j, K - 1)
            for i in range(lo, hi + 1):
                if (X >> i) & 1:
                    conv_j += ((valid_Y >> (j - i)) & 1).astype(np.int32)
            carries[j + 1] = (conv_j + carries[j]) >> 1

        sb_val = 2 * carries[D - 2] - 1

        M_arr = np.zeros(n_valid, dtype=np.int32)
        for j in range(D, 0, -1):
            update = (carries[j] > 0) & (M_arr == 0)
            M_arr[update] = j

        cas_val = np.zeros(n_valid, dtype=np.int32)
        valid_idx = np.arange(n_valid)
        mask_M = M_arr > 0
        if mask_M.any():
            cas_val[mask_M] = carries[M_arr[mask_M] - 1, valid_idx[mask_M]] - 1

        if a1 == 0:
            n00 += n_valid
            sb_00 += int(sb_val.sum())
            cas_00_tot += int(cas_val[mask_M].sum())
            if mask_M.any():
                np.add.at(cas_00, M_arr[mask_M], cas_val[mask_M].astype(np.float64))
                np.add.at(stop_00, M_arr[mask_M], 1)
        else:
            n10 += n_valid
            sb_10 += int(sb_val.sum())
            cas_10_tot += int(cas_val[mask_M].sum())
            if mask_M.any():
                np.add.at(cas_10, M_arr[mask_M], cas_val[mask_M].astype(np.float64))
                np.add.at(stop_10, M_arr[mask_M], 1)

    return {
        'D': D, 'K': K,
        'n00': n00, 'n10': n10,
        'cas_00': cas_00, 'cas_10': cas_10,
        'stop_00': stop_00, 'stop_10': stop_10,
        'sb_00': sb_00, 'sb_10': sb_10,
        'cas_00_tot': cas_00_tot, 'cas_10_tot': cas_10_tot,
    }


def part2():
    pr("\n" + "=" * 72)
    pr("PART 2: ACTUAL K-BIT MULTIPLICATION — CASCADE PROFILES")
    pr("=" * 72)
    pr()
    pr("  For each D-odd pair, cascade_val = carries[M-1] - 1 at position M.")
    pr("  σ_{ij}(d) = sum of cascade_val for sector (i,j) at position d.")
    pr()

    results = []

    for K in range(5, 16):
        t0 = time.time()
        data = enumerate_cascade(K)
        dt = time.time() - t0

        D = data['D']
        R_sb = data['sb_10'] / data['sb_00'] if data['sb_00'] != 0 else float('nan')
        R_cas = (data['cas_10_tot'] / data['cas_00_tot']
                 if data['cas_00_tot'] != 0 else float('nan'))

        stop_all = data['stop_00'] + data['stop_10']
        total_stops = stop_all.sum()
        if total_stops > 0:
            mean_stop = np.sum(np.arange(len(stop_all)) * stop_all) / total_stops
            peak_stop = np.argmax(stop_all[1:]) + 1
        else:
            mean_stop = peak_stop = 0

        pr(f"  K={K:2d} D={D:2d}: n00={data['n00']:9d} n10={data['n10']:9d}  "
           f"R_sb={R_sb:+.4f}  R_cas={R_cas:+.4f}  "
           f"gap={R_cas + PI:+.4f}  [{dt:.1f}s]")
        pr(f"    stop: peak d={peak_stop}, mean={mean_stop:.1f}, "
           f"mid={D / 2:.1f}")

        results.append(data)

        if dt > 300:
            pr("    [Time limit — stopping enumeration]")
            break

    return results


# ═══════════════════════════════════════════════════════════════
# PART 3: THE χ₄ PROJECTION TEST
# ═══════════════════════════════════════════════════════════════

def part3(results):
    pr("\n" + "=" * 72)
    pr("PART 3: χ₄ PROJECTION TEST (THE CRITICAL MEASUREMENT)")
    pr("=" * 72)
    pr()
    pr("  Project per-position cascade profiles onto Dirichlet sine basis:")
    pr("    φ_n(k) = sin(nπk/D),  k = 1,...,D-1,  n = 1,...,D-1")
    pr()
    pr("  Test: w_n / sin(nπ/2) constant for odd n?")
    pr("  If CoV → 0: the cascade projects onto χ₄ (Scenarios A or B).")
    pr("  If CoV stays large: midpoint model falsified (Scenario C).")
    pr()

    for data in results:
        K = data['K']
        D = data['D']
        if D < 8:
            continue

        chi4 = np.array([np.sin(n * PI / 2) for n in range(1, D)])

        profiles = {
            'σ₀₀': data['cas_00'][:D + 1],
            'σ₁₀': data['cas_10'][:D + 1],
            'f(d)': data['cas_10'][:D + 1] + PI * data['cas_00'][:D + 1],
        }

        # Stopping distribution (normalized)
        stop_all = (data['stop_00'] + data['stop_10'])[:D + 1]
        s = stop_all.sum()
        profiles['stop'] = stop_all / s if s > 0 else stop_all

        pr(f"  K={K} (D={D}, mid={D / 2:.1f}):")

        for label, prof in profiles.items():
            w = project_sine(prof, D)
            ratios = []
            for n in [1, 3, 5, 7, 9]:
                idx = n - 1
                if (idx < len(w) and abs(chi4[idx]) > 1e-15
                        and abs(w[idx]) > 1e-15):
                    ratios.append(w[idx] / chi4[idx])

            if len(ratios) >= 2:
                cv = (np.std(ratios) / abs(np.mean(ratios))
                      if abs(np.mean(ratios)) > 1e-15 else float('inf'))
                rstr = ', '.join(f'{r:+.4f}' for r in ratios)
                if cv < 0.05:
                    verd = 'χ₄ ✓'
                elif cv > 0.3:
                    verd = 'NOT χ₄ ✗'
                else:
                    verd = '~χ₄ ?'
                pr(f"    {label:4s}: [{rstr}]  CoV={cv:.3f} {verd}")
            elif len(ratios) == 1:
                pr(f"    {label:4s}: w_1/χ₄ = {ratios[0]:+.6f} (1 mode only)")

        pr()

    # Trend analysis
    pr("  TREND ANALYSIS: CoV of σ₀₀ ratios across K")
    pr(f"  {'K':>4s}  {'CoV(σ₀₀)':>10s}  {'CoV(σ₁₀)':>10s}  "
       f"{'CoV(f)':>10s}  {'CoV(stop)':>10s}")

    for data in results:
        K = data['K']
        D = data['D']
        if D < 8:
            continue

        chi4 = np.array([np.sin(n * PI / 2) for n in range(1, D)])
        covs = []

        for prof in [data['cas_00'][:D + 1], data['cas_10'][:D + 1],
                     data['cas_10'][:D + 1] + PI * data['cas_00'][:D + 1],
                     (data['stop_00'] + data['stop_10'])[:D + 1]]:
            w = project_sine(prof, D)
            ratios = []
            for n in [1, 3, 5, 7]:
                idx = n - 1
                if (idx < len(w) and abs(chi4[idx]) > 1e-15
                        and abs(w[idx]) > 1e-15):
                    ratios.append(w[idx] / chi4[idx])
            if len(ratios) >= 2 and abs(np.mean(ratios)) > 1e-15:
                covs.append(np.std(ratios) / abs(np.mean(ratios)))
            else:
                covs.append(float('nan'))

        pr(f"  {K:4d}  {covs[0]:10.4f}  {covs[1]:10.4f}  "
           f"{covs[2]:10.4f}  {covs[3]:10.4f}")


# ═══════════════════════════════════════════════════════════════
# PART 4: PHASE TRANSITION FORMULA VERIFICATION
# ═══════════════════════════════════════════════════════════════

def part4():
    pr("\n" + "=" * 72)
    pr("PART 4: PHASE TRANSITION FORMULA — FINITE-L CONVERGENCE")
    pr("=" * 72)
    pr()

    A_star = (2 + 3 * PI) / (2 * (1 + PI))
    beta = 1 - A_star
    S_th = 2 * beta / (1 + 2 * beta)

    pr(f"  A* = {A_star:.10f},  β = {beta:.10f}")
    pr(f"  S = 2β/(1+2β) = {S_th:.10f},  -π = {-PI:.10f}")
    pr(f"  |S+π| = {abs(S_th + PI):.2e}")
    pr()

    pr(f"  {'L':>6s}  {'S(L)':>14s}  {'|S+π|':>12s}  {'mid int?':>10s}")
    for L in [9, 15, 16, 31, 32, 63, 64, 127, 255, 511]:
        lam = np.array([(1 - A_star) * np.cos(n * PI / (L + 1)) / 2 + A_star / 2
                        for n in range(1, L + 1)])
        chi4 = np.array([np.sin(n * PI / 2) for n in range(1, L + 1)])
        S_L = float(np.sum(chi4 / (1 - lam)))
        mid_int = ((L + 1) % 2 == 0)
        pr(f"  {L:6d}  {S_L:+14.8f}  {abs(S_L + PI):12.2e}  "
           f"{'YES' if mid_int else 'no':>10s}")

    pr()
    pr("  The formula converges to -π regardless of L parity.")
    pr("  For L odd (L+1 even): midpoint (L+1)/2 is exact integer → δ_mid ≡ χ₄.")
    pr("  For L even (L+1 odd): midpoint is half-integer → O(1/L) correction.")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pr()
    pr("E215: REWARD FUNCTION PROJECTION — CRITICAL TEST FOR PAPER E §7")
    pr("=" * 72)
    pr()
    pr("QUESTION: Does the cascade readout project onto χ₄(n) = sin(nπ/2)?")
    pr()

    part1()
    results = part2()
    if results:
        part3(results)
    part4()

    pr("\n" + "=" * 72)
    pr("DIAGNOSIS")
    pr("=" * 72)
    pr()
    pr("  Look at the TREND TABLE in Part 3.")
    pr("  If CoV DECREASES with K → Scenario B (asymptotic χ₄)")
    pr("  If CoV is SMALL for all K → Scenario A (exact χ₄)")
    pr("  If CoV STAYS LARGE → Scenario C (midpoint model falsified)")
    pr()
    pr("  In Scenario C, Paper E §7 needs reformulation:")
    pr("    - Downgrade Theorem 2 to 'exact for symmetric toy model'")
    pr("    - Argue actual system is in same universality class")
    pr("    - OR find the actual w_n and redo the spectral sum")
