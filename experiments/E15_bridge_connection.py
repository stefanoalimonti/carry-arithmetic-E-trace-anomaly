#!/usr/bin/env python3
"""E15: The bridge connection — verify the spectral sum reproduces R(K).

KEY INSIGHT: The bridge eigenfunctions sin(nπj/L) at j=L/2 give
sin(nπ/2) = χ₄(n) for odd n. This is exactly the character that
produces L(1,χ₄) = π/4, hence -π = -4L(1,χ₄).

THE CRITICAL TEST:
  From the EXACT carry chain data, compute:
  1. The Dirichlet decomposition of the carry perturbation Δc(j)
  2. The effective eigenvalue structure
  3. The resulting spectral sum
  4. Compare with R(K)

  If the spectral sum reproduces R(K), we have the bridge connection.

  The remaining proof step would then be:
  SHOW that the exact carry chain produces the specific eigenvalue
  shift that transforms the bridge sum from 2/3 to -π.
"""
import sys
import time
import numpy as np
import mpmath
from mpmath import mp, mpf, pi, cos as mcos, sin as msin, nstr

mp.dps = 40


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def enumerate_dodd(K):
    lo, hi = 1 << (K - 1), 1 << K
    D = 2 * K - 1
    results = {'00': [], '10': []}
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
            sector = f"{a}{c}"
            if sector in results:
                results[sector].append({'val': val, 'carries': carries, 'M': M})
    return results, D


def main():
    t0 = time.time()

    pr("=" * 78)
    pr("E15: THE BRIDGE CONNECTION — SPECTRAL SUM = R(K)?")
    pr("=" * 78)

    # ═══════ Part A: Carry-weighted bridge decomposition ═══════════
    pr(f"\n{'═' * 78}")
    pr("A. VAL-WEIGHTED CARRY PROFILE AND ITS BRIDGE DECOMPOSITION")
    pr(f"{'═' * 78}\n")

    pr("  Instead of mean carry ⟨c_j⟩, compute VAL-WEIGHTED carry:")
    pr("  V_ac(j) = Σ_{pairs in sector ac} val · c_j / Σ val")
    pr("  This gives the carry profile WEIGHTED by the anomaly.")
    pr()

    for K in [7, 8, 9, 10]:
        data, D = enumerate_dodd(K)
        L = D - 1

        for sector in ['00', '10']:
            entries = data[sector]
            if not entries:
                continue
            vals = np.array([e['val'] for e in entries], dtype=float)
            carries_arr = np.array([e['carries'] for e in entries], dtype=float)

            total_val = vals.sum()
            if abs(total_val) < 1e-10:
                continue

            weighted_profile = (carries_arr.T * vals).sum(axis=1) / total_val

            if sector == '00':
                prof_00 = weighted_profile
            else:
                prof_10 = weighted_profile

        delta_weighted = prof_10 - prof_00

        pr(f"  K={K} (D={D}, L={L}):")
        pr(f"    Val-weighted Δprofile = V_10(j) - V_00(j):")

        j_arr = np.arange(D + 1, dtype=float)
        pr(f"    {'n':>4} {'a_n (Dirichlet)':>16} {'sin(nπ/2)':>12} "
           f"{'a_n·sin(nπ/2)':>16} {'cumul':>16}")

        bridge = delta_weighted[1:D]
        j_bridge = np.arange(1, D, dtype=float)
        cumul = 0.0
        coeffs = []
        for n in range(1, min(L, 12)):
            phi = np.sin(n * np.pi * j_bridge / L)
            an = np.dot(bridge, phi) * 2 / L
            sn = np.sin(n * np.pi / 2)
            contrib = an * sn
            cumul += contrib
            coeffs.append(an)
            if abs(sn) > 0.01:
                pr(f"    {n:4d} {an:16.8f} {sn:12.4f} {contrib:16.8f} {cumul:16.8f}")

        pr()

    # ═══════ Part B: The spectral sum from exact data ══════════════
    pr(f"\n{'═' * 78}")
    pr("B. SPECTRAL SUM FROM EXACT CARRY DATA")
    pr(f"{'═' * 78}\n")

    pr("  For each K, compute the effective bridge eigenvalues")
    pr("  from the EXACT carry chain and evaluate the bridge sum.")
    pr()
    pr("  The bridge sum S = Σ sin(nπ/2) / (1 - λ_eff_n)")
    pr("  where λ_eff_n are extracted from the carry chain's")
    pr("  position-resolved propagator.")
    pr()

    pr("  ALTERNATIVE: compute R directly from the spectral decomposition")
    pr("  of the carry weight, using Dirichlet modes.")
    pr()

    for K in [7, 8, 9, 10]:
        data, D = enumerate_dodd(K)
        L = D - 1

        S00 = sum(e['val'] for e in data['00'])
        S10 = sum(e['val'] for e in data['10'])
        R = S10 / S00 if S00 != 0 else float('inf')

        pr(f"  K={K}: S00={S00}, S10={S10}, R={R:+.6f}")

        # Decompose S00 and S10 by Dirichlet mode of val(j) where j=M
        # σ_ac = Σ val_i = Σ_j Σ_{i: M_i=j} val_i
        for sector, label in [('00', 'σ₀₀'), ('10', 'σ₁₀')]:
            entries = data[sector]
            sigma_by_j = np.zeros(D + 1)
            count_by_j = np.zeros(D + 1)
            for e in entries:
                sigma_by_j[e['M']] += e['val']
                count_by_j[e['M']] += 1

            bridge_sigma = sigma_by_j[1:D]
            j_bridge = np.arange(1, D, dtype=float)

            modes = []
            for n in range(1, L):
                phi = np.sin(n * np.pi * j_bridge / L)
                fn = np.dot(bridge_sigma, phi) * 2 / L
                modes.append(fn)

            reconstructed = sum(modes[n-1] * np.sin(n*np.pi*j_bridge/L)
                                for n in range(1, L)) * L / 2
            err = np.max(np.abs(reconstructed - bridge_sigma))

            pr(f"    {label} by mode (top 8):")
            pr(f"      {'n':>4} {'f_n':>14} {'sin(nπ/2)':>10} "
               f"{'f_n·sin(nπ/2)':>16}")
            for n in range(1, min(9, L)):
                sn = np.sin(n * np.pi / 2)
                pr(f"      {n:4d} {modes[n-1]:14.4f} {sn:10.4f} "
                   f"{modes[n-1]*sn:16.4f}")

            total_from_modes = sum(modes)
            pr(f"      Total from modes: {total_from_modes:.4f} "
               f"(actual: {sum(bridge_sigma):.4f})")
            pr()

    # ═══════ Part C: The R ratio in the modal basis ════════════════
    pr(f"\n{'═' * 78}")
    pr("C. R(K) IN THE MODAL BASIS: MODE-BY-MODE RATIO")
    pr(f"{'═' * 78}\n")

    pr("  R(K) = Σ_n f_n^{10} / Σ_n f_n^{00}")
    pr("  where f_n are the Dirichlet mode amplitudes.")
    pr("  If f_n^{10}/f_n^{00} → -π for all contributing modes,")
    pr("  then R → -π trivially.")
    pr()

    for K in [7, 8, 9, 10]:
        data, D = enumerate_dodd(K)
        L = D - 1

        modes_00 = []
        modes_10 = []

        for sector, mode_list in [('00', modes_00), ('10', modes_10)]:
            entries = data[sector]
            sigma_by_j = np.zeros(D + 1)
            for e in entries:
                sigma_by_j[e['M']] += e['val']
            bridge_sigma = sigma_by_j[1:D]
            j_bridge = np.arange(1, D, dtype=float)
            for n in range(1, L):
                phi = np.sin(n * np.pi * j_bridge / L)
                fn = np.dot(bridge_sigma, phi) * 2 / L
                mode_list.append(fn)

        S00 = sum(e['val'] for e in data['00'])
        S10 = sum(e['val'] for e in data['10'])
        R_actual = S10 / S00 if S00 != 0 else float('inf')

        pr(f"  K={K}: R = {R_actual:+.6f}")
        pr(f"    {'n':>4} {'f_n^00':>12} {'f_n^10':>12} {'ratio':>12} "
           f"{'sin(nπ/2)':>10}")

        cumul_00 = 0
        cumul_10 = 0
        for n in range(1, min(L, 12)):
            fn00 = modes_00[n-1]
            fn10 = modes_10[n-1]
            ratio = fn10 / fn00 if abs(fn00) > 1e-10 else float('inf')
            sn = np.sin(n * np.pi / 2)
            cumul_00 += fn00
            cumul_10 += fn10

            if abs(sn) > 0.01 or abs(fn00) > 1:
                pr(f"    {n:4d} {fn00:12.4f} {fn10:12.4f} {ratio:+12.4f} "
                   f"{sn:10.4f}")

        pr(f"    Cumul: 00={cumul_00:.4f} 10={cumul_10:.4f} "
           f"ratio={cumul_10/cumul_00:+.6f}")
        pr()

    # ═══════ Part D: What determines the mode ratio? ═══════════════
    pr(f"\n{'═' * 78}")
    pr("D. THE MODE RATIO f_n^10 / f_n^00 — WHAT DETERMINES IT?")
    pr(f"{'═' * 78}\n")

    pr("  If f_n^10 / f_n^00 = const for ALL n, then R = const.")
    pr("  If it varies with n, the ratio R is a WEIGHTED AVERAGE.")
    pr()
    pr("  From Part C, the ratio f_n^10/f_n^00 is NOT constant.")
    pr("  The dominant mode n=1 has a DIFFERENT ratio than higher modes.")
    pr()
    pr("  This means R is determined by the COMPETITION between modes,")
    pr("  and the eigenvalue structure (through λ_n) determines the weights.")
    pr()

    pr("  The ALTERNATING BRIDGE SUM framework says:")
    pr("  R ≈ S(A) = Σ sin(nπ/2) / (1 - λ_n - A·sin²(nπ/(2L)))")
    pr("  This predicts a SPECIFIC ratio structure for each A.")
    pr()
    pr("  Can we FIT A from the exact mode ratios?")
    pr()

    for K in [8, 9, 10]:
        data, D = enumerate_dodd(K)
        L = D - 1

        modes_00 = []
        modes_10 = []
        for sector, mode_list in [('00', modes_00), ('10', modes_10)]:
            entries = data[sector]
            sigma_by_j = np.zeros(D + 1)
            for e in entries:
                sigma_by_j[e['M']] += e['val']
            bridge_sigma = sigma_by_j[1:D]
            j_bridge = np.arange(1, D, dtype=float)
            for n in range(1, L):
                phi = np.sin(n * np.pi * j_bridge / L)
                fn = np.dot(bridge_sigma, phi) * 2 / L
                mode_list.append(fn)

        S00 = sum(e['val'] for e in data['00'])
        S10 = sum(e['val'] for e in data['10'])
        R_actual = S10 / S00 if S00 != 0 else float('inf')

        Astar = float((2 + 3 * pi) / (2 * (1 + pi)))

        def bridge_sum(A, L_val):
            S = 0.0
            for n in range(1, L_val):
                sn = np.sin(n * np.pi / 2)
                if abs(sn) < 1e-15:
                    continue
                lam = np.cos(n * np.pi / L_val) / 2
                delta = A * np.sin(n * np.pi / (2 * L_val))**2
                denom = 1 - lam - delta
                if abs(denom) < 1e-15:
                    return float('inf')
                S += sn / denom
            return S

        # Binary search for best A
        A_lo, A_hi = -2.0, 3.0
        for _ in range(100):
            A_mid = (A_lo + A_hi) / 2
            S_mid = bridge_sum(A_mid, L)
            if S_mid > R_actual:
                A_lo = A_mid
            else:
                A_hi = A_mid
        A_fit = (A_lo + A_hi) / 2
        S_fit = bridge_sum(A_fit, L)

        pr(f"  K={K} (L={L}): R = {R_actual:+.8f}")
        pr(f"    Best-fit A = {A_fit:.8f} (A* = {Astar:.8f})")
        pr(f"    Bridge sum S(A_fit) = {S_fit:+.8f}")
        pr(f"    A_fit - A* = {A_fit - Astar:+.8f}")
        pr()

    # ═══════ Part E: Convergence of A_fit to A* ═══════════════════
    pr(f"\n{'═' * 78}")
    pr("E. CONVERGENCE: A_fit(K) → A*?")
    pr(f"{'═' * 78}\n")

    Astar = float((2 + 3 * pi) / (2 * (1 + pi)))

    SECTOR_FULL = {
        5: (-12, -3), 6: (-43, -4), 7: (-142, 13),
        8: (-434, 167), 9: (-1326, 963), 10: (-4156, 4648),
        11: (-13254, 20590), 12: (-44302, 87256),
        13: (-154150, 360533), 14: (-560006, 1468632),
        15: (-2101426, 5934740), 16: (-8081892, 23873440),
        17: (-31564097, 95791014), 18: (-124527003, 383811312),
        19: (-494176763, 1536648367),
        20: (-1967962747, 6149608524),
    }

    pr(f"  A* = {Astar:.10f}")
    pr()
    pr(f"  {'K':>3} {'R(K)':>14} {'A_fit(K)':>14} {'A*-A_fit':>14} {'digits':>8}")

    for K in sorted(SECTOR_FULL):
        S00, S10 = SECTOR_FULL[K]
        if S00 == 0:
            continue
        R = S10 / S00
        L = 2 * K - 2

        def bs(A):
            S = 0.0
            for n in range(1, L):
                sn = np.sin(n * np.pi / 2)
                if abs(sn) < 1e-15:
                    continue
                lam = np.cos(n * np.pi / L) / 2
                delta = A * np.sin(n * np.pi / (2 * L))**2
                denom = 1 - lam - delta
                if abs(denom) < 1e-15:
                    return float('inf')
                S += sn / denom
            return S

        A_lo, A_hi = -2.0, 3.0
        for _ in range(200):
            A_mid = (A_lo + A_hi) / 2
            if bs(A_mid) > R:
                A_lo = A_mid
            else:
                A_hi = A_mid
        A_fit = (A_lo + A_hi) / 2
        delta_A = Astar - A_fit
        dig = -np.log10(abs(delta_A / Astar)) if abs(delta_A) > 1e-15 else 99

        pr(f"  {K:3d} {R:+14.8f} {A_fit:14.10f} {delta_A:+14.8f} {dig:8.2f}")

    pr()
    pr("  If A_fit(K) → A* as K→∞, the bridge framework is EXACT")
    pr("  and the proof reduces to: carry correlations → A = A*.")

    pr(f"\n  Total runtime: {time.time() - t0:.1f}s")
    pr("=" * 78)


if __name__ == '__main__':
    main()
