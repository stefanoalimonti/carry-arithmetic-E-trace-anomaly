#!/usr/bin/env python3
"""
E216: Parity Decay Test — Universality Hypothesis for Paper E §7

QUESTION: Does η(K) = E_even/E_odd → 0 as K → ∞?

η measures the fraction of spectral energy in the even sine harmonics
relative to odd.  χ₄(n) = sin(nπ/2) has zero even-mode content, so
η → 0 means the cascade readout becomes spectrally equivalent to a
midpoint delta in the thermodynamic limit.

PROFILES TESTED:
  - stop(d):  normalised stopping distribution
  - σ₀₀(d):  per-position cascade contribution, sector (0,0)
  - σ₁₀(d):  per-position cascade contribution, sector (1,0)
  - f(d) = σ₁₀ + π·σ₀₀: combined profile (vanishes globally if R = -π)

SECONDARY TESTS:
  - η decay rate:   η(K)/η(K-1) — exponential, power-law, or stagnant?
  - χ₄ consistency: within odd harmonics, is w_{2k+1}/(-1)^k constant?
  - Spectral portrait of the largest K available
"""

import numpy as np
import math
import time

PI = math.pi


def pr(s=""):
    print(s, flush=True)


# ═══════════════════════════════════════════════════════════════
# PROJECTION TOOLS
# ═══════════════════════════════════════════════════════════════

def project_sine_full(f, D):
    """
    Project f[0..D] onto full Dirichlet sine basis.
    Returns c_n for n = 1, ..., D-1.
    Dirichlet walls at k=0 and k=D; interior points k=1,...,D-1.
    """
    if D < 3:
        return np.array([])
    k = np.arange(1, D, dtype=np.float64)
    interior = f[1:D].astype(np.float64)
    n_arr = np.arange(1, D, dtype=np.float64)
    sin_matrix = np.sin(np.outer(n_arr, k) * PI / D)
    return sin_matrix @ interior


def compute_eta(c):
    """η = Σ|c_n|² (even n) / Σ|c_n|² (odd n), 1-indexed."""
    even_energy = np.sum(c[1::2]**2)   # n = 2, 4, 6, ...
    odd_energy  = np.sum(c[0::2]**2)   # n = 1, 3, 5, ...
    if odd_energy < 1e-30:
        return float('inf')
    return even_energy / odd_energy


def chi4_cov(c, max_modes=10):
    """
    Within odd harmonics, check χ₄ proportionality.
    χ₄(2k+1) = (-1)^k, so w_{2k+1}/(-1)^k should be constant.
    Returns (CoV, array of signed ratios).
    """
    odd_c = c[0::2]  # c_1, c_3, c_5, ...
    signs = np.array([(-1)**k for k in range(len(odd_c))])
    signed = odd_c * signs

    mask = np.abs(signed) > 1e-15 * np.max(np.abs(signed) + 1e-30)
    vals = signed[mask][:max_modes]
    if len(vals) < 2:
        return float('inf'), vals
    return np.std(vals) / (abs(np.mean(vals)) + 1e-30), vals


# ═══════════════════════════════════════════════════════════════
# ENUMERATION (from E215, self-contained)
# ═══════════════════════════════════════════════════════════════

def enumerate_cascade(K):
    """Exact K-bit D-odd multiplication: carry sequences, cascade stopping."""
    D = 2 * K - 1
    X_lo, X_hi = 1 << (K - 1), 1 << K
    Y_lo = X_lo
    Y_hi = X_lo + (1 << (K - 2))
    D_lo, D_hi = 1 << (D - 1), 1 << D

    cas_00  = np.zeros(D + 2)
    cas_10  = np.zeros(D + 2)
    stop_00 = np.zeros(D + 2, dtype=np.int64)
    stop_10 = np.zeros(D + 2, dtype=np.int64)
    n00 = n10 = sb_00 = sb_10 = cas_00_tot = cas_10_tot = 0

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
            lo, hi = max(0, j - K + 1), min(j, K - 1)
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
        vi = np.arange(n_valid)
        mask_M = M_arr > 0
        if mask_M.any():
            cas_val[mask_M] = carries[M_arr[mask_M] - 1, vi[mask_M]] - 1

        if a1 == 0:
            n00 += n_valid; sb_00 += int(sb_val.sum())
            cas_00_tot += int(cas_val[mask_M].sum())
            if mask_M.any():
                np.add.at(cas_00, M_arr[mask_M], cas_val[mask_M].astype(np.float64))
                np.add.at(stop_00, M_arr[mask_M], 1)
        else:
            n10 += n_valid; sb_10 += int(sb_val.sum())
            cas_10_tot += int(cas_val[mask_M].sum())
            if mask_M.any():
                np.add.at(cas_10, M_arr[mask_M], cas_val[mask_M].astype(np.float64))
                np.add.at(stop_10, M_arr[mask_M], 1)

    return dict(D=D, K=K, n00=n00, n10=n10,
                cas_00=cas_00, cas_10=cas_10,
                stop_00=stop_00, stop_10=stop_10,
                sb_00=sb_00, sb_10=sb_10,
                cas_00_tot=cas_00_tot, cas_10_tot=cas_10_tot)


# ═══════════════════════════════════════════════════════════════
# BUILD PROFILE DICT
# ═══════════════════════════════════════════════════════════════

LABELS = ['stop', 'σ₀₀', 'σ₁₀', 'f(d)']

def make_profiles(data):
    D = data['D']
    stop_all = (data['stop_00'] + data['stop_10'])[:D + 1].astype(np.float64)
    s = stop_all.sum()
    return {
        'stop': stop_all / s if s > 0 else stop_all,
        'σ₀₀':  data['cas_00'][:D + 1],
        'σ₁₀':  data['cas_10'][:D + 1],
        'f(d)': data['cas_10'][:D + 1] + PI * data['cas_00'][:D + 1],
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    pr()
    pr("E216: PARITY DECAY TEST — UNIVERSALITY HYPOTHESIS")
    pr("=" * 72)
    pr()
    pr("  η(K) = Σ|c_n|² (even n) / Σ|c_n|² (odd n)")
    pr("  If η → 0:  even modes die out → cascade readout flows to χ₄")
    pr("  If η ↛ 0:  the mechanism producing -π is not simple χ₄ projection")
    pr()

    # ─── Phase 1: Enumerate ─────────────────────────────────────
    pr("PHASE 1: ENUMERATION")
    pr("─" * 72)

    all_data = []
    for K in range(5, 16):
        t0 = time.time()
        data = enumerate_cascade(K)
        dt = time.time() - t0

        D = data['D']
        R_cas = (data['cas_10_tot'] / data['cas_00_tot']
                 if data['cas_00_tot'] != 0 else float('nan'))
        pr(f"  K={K:2d}  D={D:2d}  n={data['n00']+data['n10']:10d}  "
           f"R_cas={R_cas:+.6f}  gap={R_cas+PI:+.6f}  [{dt:.1f}s]")
        all_data.append(data)
        if dt > 600:
            pr("  [time limit]")
            break

    # ─── Phase 2: η(K) table ────────────────────────────────────
    pr()
    pr("PHASE 2: PARITY ENERGY RATIO η(K)")
    pr("─" * 72)
    pr()
    pr(f"  {'K':>3s} {'D':>3s}  {'η(stop)':>12s}  {'η(σ₀₀)':>12s}  "
       f"{'η(σ₁₀)':>12s}  {'η(f)':>12s}")
    pr(f"  {'─'*3} {'─'*3}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")

    eta_history = {lb: [] for lb in LABELS}

    for data in all_data:
        K, D = data['K'], data['D']
        profs = make_profiles(data)
        row = f"  {K:3d} {D:3d}"
        for lb in LABELS:
            c = project_sine_full(profs[lb], D)
            eta = compute_eta(c)
            eta_history[lb].append(eta)
            row += f"  {eta:12.6f}"
        pr(row)

    # ─── Phase 3: η ratio (consecutive) ────────────────────────
    pr()
    pr("PHASE 3: η DECAY RATE — η(K)/η(K-1)")
    pr("─" * 72)
    pr()
    pr("  Ratio < 1 consistently → η decays.  Ratio ≈ const → exponential.")
    pr()
    pr(f"  {'K':>3s}  {'r(stop)':>10s}  {'r(σ₀₀)':>10s}  "
       f"{'r(σ₁₀)':>10s}  {'r(f)':>10s}")
    pr(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for i in range(1, len(all_data)):
        K = all_data[i]['K']
        row = f"  {K:3d}"
        for lb in LABELS:
            prev, curr = eta_history[lb][i-1], eta_history[lb][i]
            if prev > 1e-30:
                row += f"  {curr/prev:10.4f}"
            else:
                row += f"  {'—':>10s}"
        pr(row)

    # ─── Phase 4: χ₄ consistency within odd harmonics ──────────
    pr()
    pr("PHASE 4: χ₄ CONSISTENCY WITHIN ODD HARMONICS")
    pr("─" * 72)
    pr()
    pr("  CoV of w_{2k+1}·(-1)^k for first 10 odd modes.")
    pr("  CoV → 0: odd harmonics have exact χ₄ alternation.")
    pr()
    pr(f"  {'K':>3s}  {'CoV(stop)':>10s}  {'CoV(σ₀₀)':>10s}  "
       f"{'CoV(σ₁₀)':>10s}  {'CoV(f)':>10s}")
    pr(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for data in all_data:
        K, D = data['K'], data['D']
        profs = make_profiles(data)
        row = f"  {K:3d}"
        for lb in LABELS:
            c = project_sine_full(profs[lb], D)
            cov, _ = chi4_cov(c)
            row += f"  {cov:10.4f}"
        pr(row)

    # ─── Phase 5: Spectral portrait, largest K ──────────────────
    pr()
    pr("PHASE 5: SPECTRAL PORTRAIT (largest K)")
    pr("─" * 72)

    data = all_data[-1]
    K, D = data['K'], data['D']
    profs = make_profiles(data)

    c_stop = project_sine_full(profs['stop'], D)
    c_f    = project_sine_full(profs['f(d)'], D)

    pr(f"\n  K={K}, D={D}: first 24 sine coefficients")
    pr(f"  {'n':>4s}  {'c_n(stop)':>14s}  {'c_n(f)':>14s}  "
       f"{'|c_n|²(stop)':>14s}  {'par':>5s}  {'χ₄':>5s}")

    for n in range(1, min(25, D)):
        chi4_val = math.sin(n * PI / 2)
        par = "ODD" if n % 2 == 1 else "even"
        pr(f"  {n:4d}  {c_stop[n-1]:+14.8f}  {c_f[n-1]:+14.8f}  "
           f"{c_stop[n-1]**2:14.8f}  {par:>5s}  {chi4_val:+5.1f}")

    # ─── Phase 6: Resolvent-weighted η ──────────────────────────
    pr()
    pr("PHASE 6: RESOLVENT-WEIGHTED η (with LMH eigenvalues)")
    pr("─" * 72)
    pr()
    pr("  The spectral sum S = Σ w_n/(1-λ_n) weights each mode by the")
    pr("  resolvent.  η_R = Σ|w_n/(1-λ_n)|² (even) / (odd) tests whether")
    pr("  the RESOLVENT-AMPLIFIED profile is χ₄-like, even if raw w_n is not.")
    pr()

    A_star = (2 + 3*PI) / (2*(1 + PI))

    pr(f"  {'K':>3s} {'D':>3s}  {'η_R(stop)':>12s}  {'η_R(σ₀₀)':>12s}  "
       f"{'η_R(σ₁₀)':>12s}  {'η_R(f)':>12s}")
    pr(f"  {'─'*3} {'─'*3}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}")

    for data in all_data:
        K, D = data['K'], data['D']
        profs = make_profiles(data)

        lam = np.array([(1 - A_star)*np.cos(n*PI/(D))/2 + A_star/2
                        for n in range(1, D)])
        resolvent_weight = 1.0 / (1.0 - lam + 1e-30)

        row = f"  {K:3d} {D:3d}"
        for lb in LABELS:
            c = project_sine_full(profs[lb], D)
            weighted = c * resolvent_weight[:len(c)]
            even_e = np.sum(weighted[1::2]**2)
            odd_e  = np.sum(weighted[0::2]**2)
            eta_r = even_e / odd_e if odd_e > 1e-30 else float('inf')
            row += f"  {eta_r:12.6f}"
        pr(row)

    # ─── Verdict ────────────────────────────────────────────────
    pr()
    pr("=" * 72)
    pr("VERDICT")
    pr("=" * 72)
    pr()

    eta_first = eta_history['stop'][0]
    eta_last  = eta_history['stop'][-1]
    K_first   = all_data[0]['K']
    K_last    = all_data[-1]['K']

    pr(f"  η(stop): K={K_first} → K={K_last}: {eta_first:.6f} → {eta_last:.6f}")
    pr(f"  η(f):    K={K_first} → K={K_last}: "
       f"{eta_history['f(d)'][0]:.6f} → {eta_history['f(d)'][-1]:.6f}")
    pr()

    if eta_last < eta_first * 0.3:
        pr("  ═══ η DECAYS STRONGLY — UNIVERSALITY HYPOTHESIS SUPPORTED ═══")
        pr("  Even-harmonic energy is dying out. The cascade readout flows")
        pr("  toward the χ₄ parity class in the thermodynamic limit.")
        pr("  This validates Part II of Conjecture 1.")
    elif eta_last < eta_first * 0.7:
        pr("  ─── η decays moderately — SUGGESTIVE but inconclusive ───")
        pr("  Trend is downward.  Larger K needed for definitive answer.")
    elif abs(eta_last - eta_first) / (eta_first + 1e-30) < 0.3:
        pr("  ─── η is approximately CONSTANT — χ₄ parity NOT emerging ───")
        pr("  Even and odd energies scale together.")
        pr("  The universality mechanism (if any) is NOT simple parity decay.")
    else:
        pr("  ═══ η GROWS — UNIVERSALITY HYPOTHESIS FALSIFIED ═══")
        pr("  Even-harmonic energy grows relative to odd.")
        pr("  The cascade readout diverges FROM χ₄, not toward it.")

    pr()
    pr("  NOTE: Phase 6 (resolvent-weighted η) may tell a different story.")
    pr("  The resolvent (I - K_eff)⁻¹ amplifies modes near eigenvalue 1")
    pr("  and could suppress even-mode contributions even if raw η ≠ 0.")
    pr()


if __name__ == "__main__":
    main()
