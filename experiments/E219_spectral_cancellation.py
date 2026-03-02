#!/usr/bin/env python3
"""
E219: Spectral Cancellation Test — The Moment of Truth

QUESTION: Does R(K) → −π emerge from mode-by-mode proportionality
between the two sectors' spectral profiles?

HYPOTHESIS: σ₁₀(d) / σ₀₀(d) ≈ c(K) for all active positions d.
Equivalently, w_n^{10} / w_n^{00} ≈ c(K) for all modes n.
If c(K) → −π, the cancellation is in the RATIO, not in χ₄.

KEY MATHEMATICAL FACT: In the spatial sum Σ_d σ(d) = Σ_n w_n · G_n,
the spatial integration weight G_n = Σ_d sin(nπd/D) = 0 for even n.
Even modes are AUTOMATICALLY invisible in the total, regardless of w_n.
Only odd modes contribute. So the real question is:
does w_n^{10}/w_n^{00} ≈ constant across ODD modes n?

PHASES:
  1. Direct spatial ratio: σ₁₀(d)/σ₀₀(d) across positions d
  2. Stopping distribution ratio: stop₁₀(d)/stop₀₀(d)
  3. Mode-by-mode spectral ratio: w_n^{10}/w_n^{00} for odd n
  4. Resolvent reconstruction vs direct R_cas
  5. Convergence of the mode ratio to −π
"""

import numpy as np
import math
import time

PI = math.pi


def pr(s=""):
    print(s, flush=True)


def project_sine(f, D):
    """Project f[0..D] onto Dirichlet sine basis. Returns c_n, n=1..D-1."""
    if D < 3:
        return np.array([])
    k = np.arange(1, D, dtype=np.float64)
    interior = f[1:D].astype(np.float64)
    n_arr = np.arange(1, D, dtype=np.float64)
    sin_mat = np.sin(np.outer(n_arr, k) * PI / D)
    return sin_mat @ interior


def enumerate_cascade(K):
    """Exact K-bit D-odd multiplication with per-sector cascade profiles."""
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


def main():
    pr()
    pr("E219: SPECTRAL CANCELLATION TEST — THE MOMENT OF TRUTH")
    pr("=" * 72)
    pr()
    pr("  If w_n^{10}/w_n^{00} ≈ c(K) for all odd n, and c(K) → −π,")
    pr("  then −π emerges from SECTOR PROPORTIONALITY, not from χ₄.")
    pr()

    A_star = (2 + 3*PI) / (2*(1 + PI))

    # ─── Phase 0: Enumerate ────────────────────────────────────
    pr("PHASE 0: ENUMERATION")
    pr("─" * 72)

    all_data = []
    for K in range(5, 16):
        t0 = time.time()
        data = enumerate_cascade(K)
        dt = time.time() - t0
        D = data['D']
        R_cas = (data['cas_10_tot'] / data['cas_00_tot']
                 if data['cas_00_tot'] != 0 else float('nan'))
        pr(f"  K={K:2d}  D={D:2d}  R_cas={R_cas:+.6f}  "
           f"gap={R_cas+PI:+.6f}  [{dt:.1f}s]")
        all_data.append(data)
        if dt > 600:
            break

    # ─── Phase 1: Direct spatial ratio ─────────────────────────
    pr()
    pr("PHASE 1: DIRECT SPATIAL RATIO σ₁₀(d)/σ₀₀(d)")
    pr("─" * 72)
    pr()
    pr("  If this ratio is ≈ constant across d, the sectors are spatially")
    pr("  proportional and R_cas = that constant automatically.")
    pr()

    for data in all_data:
        K, D = data['K'], data['D']
        cas00 = data['cas_00'][:D + 1]
        cas10 = data['cas_10'][:D + 1]

        active = np.abs(cas00) > 0.5
        if not active.any():
            pr(f"  K={K:2d}: no active positions")
            continue

        ratios = cas10[active] / cas00[active]
        R_cas = data['cas_10_tot'] / data['cas_00_tot']

        mean_r = np.mean(ratios)
        std_r = np.std(ratios)
        cov_r = std_r / abs(mean_r) if abs(mean_r) > 1e-15 else float('inf')
        n_active = int(active.sum())

        pr(f"  K={K:2d} D={D:2d}: {n_active:2d} active pos.  "
           f"ratio mean={mean_r:+.4f} std={std_r:.4f} CoV={cov_r:.4f}  "
           f"R_cas={R_cas:+.4f}")

    # ─── Phase 2: Stopping distribution ratio ──────────────────
    pr()
    pr("PHASE 2: STOPPING RATIO stop₁₀(d)/stop₀₀(d)")
    pr("─" * 72)
    pr()

    for data in all_data:
        K, D = data['K'], data['D']
        s00 = data['stop_00'][:D + 1].astype(np.float64)
        s10 = data['stop_10'][:D + 1].astype(np.float64)

        active = s00 > 0.5
        if not active.any():
            continue

        ratios = s10[active] / s00[active]
        mean_r = np.mean(ratios)
        std_r = np.std(ratios)
        cov_r = std_r / abs(mean_r) if abs(mean_r) > 1e-15 else float('inf')

        pr(f"  K={K:2d}: stop ratio mean={mean_r:+.4f} std={std_r:.4f} "
           f"CoV={cov_r:.4f}  (n10/n00={data['n10']/data['n00']:.4f})")

    # ─── Phase 3: Mode-by-mode spectral ratio ──────────────────
    pr()
    pr("PHASE 3: MODE-BY-MODE RATIO w_n^{10}/w_n^{00} (ODD n only)")
    pr("─" * 72)
    pr()
    pr("  Only odd modes contribute to the spatial integral.")
    pr("  If the ratio is constant across odd n, sectors are spectrally")
    pr("  proportional and R_cas = that constant.")
    pr()

    ratio_tables = []

    for data in all_data:
        K, D = data['K'], data['D']
        if D < 8:
            continue

        w00 = project_sine(data['cas_00'][:D + 1], D)
        w10 = project_sine(data['cas_10'][:D + 1], D)

        odd_indices = list(range(0, len(w00), 2))  # n=1,3,5,...

        ratios_odd = []
        for idx in odd_indices:
            if abs(w00[idx]) > 1e-10 * max(abs(w00.max()), 1):
                ratios_odd.append(w10[idx] / w00[idx])
            else:
                ratios_odd.append(float('nan'))

        valid = [r for r in ratios_odd if not math.isnan(r)]
        if len(valid) >= 2:
            mean_r = np.mean(valid)
            std_r = np.std(valid)
            cov_r = std_r / abs(mean_r) if abs(mean_r) > 1e-15 else float('inf')
        else:
            mean_r = float('nan')
            cov_r = float('inf')

        R_cas = (data['cas_10_tot'] / data['cas_00_tot']
                 if data['cas_00_tot'] != 0 else float('nan'))

        pr(f"  K={K:2d}: mean ratio={mean_r:+.6f}  CoV={cov_r:.4f}  "
           f"R_cas={R_cas:+.6f}  gap(ratio,R_cas)={mean_r-R_cas:+.6f}")

        first_few = [(odd_indices[i]+1, ratios_odd[i])
                     for i in range(min(6, len(ratios_odd)))]
        ff_str = "  ".join(f"n={n}: {r:+.4f}" for n, r in first_few
                           if not math.isnan(r))
        pr(f"         [{ff_str}]")

        ratio_tables.append({
            'K': K, 'D': D, 'mean': mean_r, 'cov': cov_r,
            'R_cas': R_cas, 'ratios_odd': ratios_odd,
            'w00': w00, 'w10': w10,
        })

    # ─── Phase 4: Resolvent reconstruction ─────────────────────
    pr()
    pr("PHASE 4: RESOLVENT RECONSTRUCTION WITH REAL w_n")
    pr("─" * 72)
    pr()
    pr("  S_ij = Σ_{n odd} w_n^{ij} · G_n / (1-λ_n)")
    pr("  G_n = Σ_d sin(nπd/D) = 0 for even n, nonzero for odd n")
    pr("  Using LMH eigenvalues λ_n(A*) = (1-A*)cos(nπ/D)/2 + A*/2")
    pr()

    pr(f"  {'K':>3s}  {'R_cas':>10s}  {'R_recon':>10s}  {'gap':>10s}  "
       f"{'R_recon+π':>10s}")
    pr(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for rt in ratio_tables:
        K, D = rt['K'], rt['D']
        w00, w10 = rt['w00'], rt['w10']

        G_n = np.zeros(D - 1)
        for n in range(1, D):
            G_n[n-1] = sum(math.sin(n * PI * d / D) for d in range(1, D))

        lam = np.array([(1-A_star)*math.cos(n*PI/D)/2 + A_star/2
                        for n in range(1, D)])

        weight = G_n / (1.0 - lam)

        S00 = np.sum(w00 * weight)
        S10 = np.sum(w10 * weight)

        R_recon = S10 / S00 if abs(S00) > 1e-15 else float('nan')

        pr(f"  {K:3d}  {rt['R_cas']:+10.6f}  {R_recon:+10.6f}  "
           f"{R_recon - rt['R_cas']:+10.6f}  {R_recon+PI:+10.6f}")

    # ─── Phase 5: Even vs Odd mode contribution to ratio ───────
    pr()
    pr("PHASE 5: EVEN vs ODD MODE CONTRIBUTIONS")
    pr("─" * 72)
    pr()
    pr("  G_n = 0 for even n ⟹ even modes AUTOMATICALLY vanish")
    pr("  in the spatial sum, regardless of w_n.")
    pr()
    pr("  Verify: Σ_{n even} |w_n · G_n| vs Σ_{n odd} |w_n · G_n|")
    pr()

    for rt in ratio_tables:
        K, D = rt['K'], rt['D']
        w00 = rt['w00']

        G_n = np.zeros(D - 1)
        for n in range(1, D):
            G_n[n-1] = sum(math.sin(n * PI * d / D) for d in range(1, D))

        even_contrib = np.sum(np.abs(w00[1::2] * G_n[1::2]))
        odd_contrib  = np.sum(np.abs(w00[0::2] * G_n[0::2]))

        pr(f"  K={K:2d}: |even·G|={even_contrib:.2e}  "
           f"|odd·G|={odd_contrib:.2e}  "
           f"ratio={even_contrib/odd_contrib:.2e}" if odd_contrib > 0 else
           f"  K={K:2d}: |even·G|={even_contrib:.2e}  |odd·G|=0")

    # ─── Phase 6: Convergence of mode ratio to −π ──────────────
    pr()
    pr("PHASE 6: CONVERGENCE OF MODE RATIO → −π")
    pr("─" * 72)
    pr()
    pr(f"  {'K':>3s}  {'⟨w10/w00⟩_odd':>14s}  {'R_cas':>10s}  "
       f"{'ratio+π':>10s}  {'R_cas+π':>10s}  {'CoV':>8s}")
    pr(f"  {'─'*3}  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

    for rt in ratio_tables:
        K = rt['K']
        pr(f"  {K:3d}  {rt['mean']:+14.6f}  {rt['R_cas']:+10.6f}  "
           f"{rt['mean']+PI:+10.6f}  {rt['R_cas']+PI:+10.6f}  "
           f"{rt['cov']:8.4f}")

    # ─── Phase 7: Detailed spectral portrait for largest K ─────
    pr()
    pr("PHASE 7: SPECTRAL PORTRAIT (largest K)")
    pr("─" * 72)

    rt = ratio_tables[-1]
    K, D = rt['K'], rt['D']
    w00, w10 = rt['w00'], rt['w10']

    G_n = np.zeros(D - 1)
    for n in range(1, D):
        G_n[n-1] = sum(math.sin(n * PI * d / D) for d in range(1, D))

    pr(f"\n  K={K}, D={D}: per-mode analysis")
    pr(f"  {'n':>4s}  {'w_n^00':>14s}  {'w_n^10':>14s}  {'ratio':>10s}  "
       f"{'G_n':>10s}  {'w00·G':>14s}  {'par':>5s}")

    for n in range(1, min(D, 21)):
        par = "ODD" if n % 2 == 1 else "even"
        ratio = w10[n-1]/w00[n-1] if abs(w00[n-1]) > 1e-10 else float('nan')
        wg = w00[n-1] * G_n[n-1]
        pr(f"  {n:4d}  {w00[n-1]:+14.4f}  {w10[n-1]:+14.4f}  "
           f"{ratio:+10.4f}  {G_n[n-1]:+10.4f}  {wg:+14.4f}  {par:>5s}")

    # ─── Phase 8: Best-fit A from real w_n ─────────────────────
    pr()
    pr("PHASE 8: BEST-FIT A FROM RESOLVENT WITH REAL w_n")
    pr("─" * 72)
    pr()
    pr("  Vary A and find which A makes S10/S00 = R_cas.")
    pr("  Compare to A* = (2+3π)/(2(1+π)).")
    pr()

    for rt in ratio_tables[-4:]:
        K, D = rt['K'], rt['D']
        w00, w10 = rt['w00'], rt['w10']
        R_cas = rt['R_cas']

        G_n = np.zeros(D - 1)
        for n in range(1, D):
            G_n[n-1] = sum(math.sin(n * PI * d / D) for d in range(1, D))

        def resolvent_ratio(A):
            lam = np.array([(1-A)*math.cos(n*PI/D)/2 + A/2
                            for n in range(1, D)])
            denom = 1.0 - lam
            mask = np.abs(denom) > 1e-15
            weight = np.zeros(D - 1)
            weight[mask] = G_n[mask] / denom[mask]
            S00 = np.sum(w00 * weight)
            S10 = np.sum(w10 * weight)
            return S10 / S00 if abs(S00) > 1e-15 else float('nan')

        best_A = None
        best_err = float('inf')
        for A_try in np.linspace(0.5, 2.5, 2000):
            r = resolvent_ratio(A_try)
            if not math.isnan(r):
                err = abs(r - R_cas)
                if err < best_err:
                    best_err = err
                    best_A = A_try

        if best_A is not None:
            R_fit = resolvent_ratio(best_A)
            R_astar = resolvent_ratio(A_star)
            pr(f"  K={K:2d}: A_fit={best_A:.4f}  R_fit={R_fit:+.6f}  "
               f"R_cas={R_cas:+.6f}  |err|={best_err:.2e}")
            pr(f"         A*={A_star:.4f}    R(A*)={R_astar:+.6f}  "
               f"|R(A*)+π|={abs(R_astar+PI):.2e}")

    # ─── Verdict ───────────────────────────────────────────────
    pr()
    pr("=" * 72)
    pr("VERDICT")
    pr("=" * 72)
    pr()

    if len(ratio_tables) >= 3:
        last3_covs = [rt['cov'] for rt in ratio_tables[-3:]]
        last_mean  = ratio_tables[-1]['mean']
        first_mean = ratio_tables[0]['mean']

        pr(f"  Mode ratio ⟨w10/w00⟩_odd:")
        pr(f"    K={ratio_tables[0]['K']}: {ratio_tables[0]['mean']:+.6f}")
        pr(f"    K={ratio_tables[-1]['K']}: {ratio_tables[-1]['mean']:+.6f}")
        pr(f"    −π = {-PI:.6f}")
        pr()

        if all(c < 0.15 for c in last3_covs):
            pr("  ═══ SECTORS ARE SPECTRALLY PROPORTIONAL ═══")
            pr("  w_n^{10}/w_n^{00} ≈ constant across odd modes.")
            pr("  The cascade ratio R = this constant.")
            if abs(last_mean + PI) < abs(ratio_tables[-1]['R_cas'] + PI) * 2:
                pr("  AND the constant converges toward −π!")
                pr("  ═══ THE CANCELLATION MECHANISM IS IDENTIFIED ═══")
            else:
                pr("  But the constant does NOT converge to −π.")
        elif all(c < 0.5 for c in last3_covs):
            pr("  ─── Approximate proportionality ───")
            pr("  Sectors are roughly proportional. Refinement needed.")
        else:
            pr("  ═══ SECTORS ARE NOT PROPORTIONAL ═══")
            pr("  w_n^{10}/w_n^{00} varies significantly across modes.")
            pr("  The cancellation mechanism is NOT simple proportionality.")

    pr()


if __name__ == "__main__":
    main()
