#!/usr/bin/env python3
"""E17: Structural decomposition of σ_{ac} via top-carry constraint.

CRITICAL STRUCTURAL INSIGHT (from prior experiments, Part G):
  conv_{D-2} = a + c  (second MSBs of factors)
  D-odd requires carries[D-1] = 0, hence (conv_{D-2} + carries[D-2]) < 2.

  Sector (0,0): conv_{D-2} = 0, so carries[D-2] ∈ {0,1}
    → M can be D-2 (when carries[D-2]=1) or M < D-2 (when carries[D-2]=0)

  Sector (1,0): conv_{D-2} = 1, so carries[D-2] MUST be 0
    → M < D-2 always

  Therefore: σ_{00} = σ_{00}^{top} + σ_{00}^{low}
             σ_{10} = σ_{10}^{low}  (no "top" contribution)

  where "top" = pairs with M = D-2, "low" = pairs with M < D-2.

  This experiment:
    Part A: Decompose σ_{00} = σ_{00}^{top} + σ_{00}^{low}
    Part B: Compare σ_{10}^{low} with σ_{00}^{low}
    Part C: Analyze σ_{00}^{top} structure
    Part D: Test if the top contribution determines R = -π
    Part E: Deeper: decompose by position of M
    Part F: Recursive structure — what happens one level down?
"""
import sys
import time
import numpy as np
from mpmath import mp, mpf, pi as MPI

mp.dps = 30


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def enumerate_dodd_decomp(K):
    """Enumerate D-odd pairs, decompose by M position and carry values."""
    lo = 1 << (K - 1)
    hi = 1 << K
    D = 2 * K - 1

    results = []
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
            results.append({
                'a': a, 'c': c, 'val': val, 'M': M,
                'carries': carries, 'p': p, 'q': q,
                'cD2': carries[D - 2], 'cD3': carries[D - 3],
            })

    return results


def main():
    t0 = time.time()

    pr("=" * 76)
    pr("E17: STRUCTURAL DECOMPOSITION — TOP CARRY CONSTRAINT")
    pr("=" * 76)

    pr("\n  KEY IDENTITY: conv_{D-2} = a + c")
    pr("  D-odd → carries[D-1] = 0 → (a + c + carries[D-2]) < 2")
    pr("  Sector (1,0): carries[D-2] = 0 forced → M ≤ D-3")
    pr("  Sector (0,0): carries[D-2] ∈ {0,1} → M can be D-2")

    # ═══════ Part A: Top/Low decomposition ═══════════════════════════
    pr(f"\n{'═' * 76}")
    pr("A. σ_{00} = σ_{00}^{top} + σ_{00}^{low}, σ_{10} = σ_{10}^{low}")
    pr(f"{'═' * 76}\n")

    convergence_data = []

    for K in range(4, 14):
        tb = time.time()
        data = enumerate_dodd_decomp(K)
        dt = time.time() - tb
        D = 2 * K - 1

        s00_top_val, s00_top_cnt = 0, 0
        s00_low_val, s00_low_cnt = 0, 0
        s10_low_val, s10_low_cnt = 0, 0
        s01_low_val, s01_low_cnt = 0, 0

        for d in data:
            a, c, val, M = d['a'], d['c'], d['val'], d['M']
            if a == 0 and c == 0:
                if M == D - 2:
                    s00_top_val += val
                    s00_top_cnt += 1
                else:
                    s00_low_val += val
                    s00_low_cnt += 1
            elif a == 1 and c == 0:
                s10_low_val += val
                s10_low_cnt += 1
            elif a == 0 and c == 1:
                s01_low_val += val
                s01_low_cnt += 1

        S00 = s00_top_val + s00_low_val
        S10 = s10_low_val
        R = S10 / S00 if S00 != 0 else float('inf')

        R_low = s10_low_val / s00_low_val if s00_low_val != 0 else float('inf')
        frac_top_cnt = s00_top_cnt / (s00_top_cnt + s00_low_cnt) if (s00_top_cnt + s00_low_cnt) else 0
        frac_top_val = s00_top_val / S00 if S00 != 0 else 0

        convergence_data.append({
            'K': K, 'D': D,
            'S00': S00, 'S10': S10, 'R': R,
            's00_top_val': s00_top_val, 's00_top_cnt': s00_top_cnt,
            's00_low_val': s00_low_val, 's00_low_cnt': s00_low_cnt,
            's10_low_val': s10_low_val, 's10_low_cnt': s10_low_cnt,
            's01_low_val': s01_low_val, 's01_low_cnt': s01_low_cnt,
            'R_low': R_low,
            'frac_top_cnt': frac_top_cnt,
            'frac_top_val': frac_top_val,
            'dt': dt,
        })

        pr(f"  K={K:2d}: S00={S00:10d} = {s00_top_val:10d}(top) + {s00_low_val:10d}(low)  "
           f"S10={S10:10d}  R={R:+.8f}  ({dt:.2f}s)")
        pr(f"        N00={s00_top_cnt+s00_low_cnt:8d} = {s00_top_cnt:8d}(top) + {s00_low_cnt:8d}(low)  "
           f"N10={s10_low_cnt:8d}")
        pr(f"        frac_top(count)={frac_top_cnt:.6f}  frac_top(val)={frac_top_val:.6f}  "
           f"R_low=σ10/σ00^low={R_low:+.8f}")

    # ═══════ Part B: Convergence analysis ════════════════════════════
    pr(f"\n{'═' * 76}")
    pr("B. CONVERGENCE: R, R_low, frac_top")
    pr(f"{'═' * 76}\n")

    pr(f"  {'K':>3s}  {'R':>12s}  {'R+π':>14s}  "
       f"{'R_low':>12s}  {'R_low+π':>14s}  "
       f"{'f_top(cnt)':>10s}  {'f_top(val)':>10s}")

    for cd in convergence_data:
        K = cd['K']
        R = cd['R']
        R_low = cd['R_low']
        err_R = R + float(MPI)
        err_Rlow = R_low + float(MPI) if R_low != float('inf') else float('inf')
        pr(f"  {K:3d}  {R:+12.8f}  {err_R:+14.10f}  "
           f"{R_low:+12.8f}  {err_Rlow:+14.10f}  "
           f"{cd['frac_top_cnt']:10.6f}  {cd['frac_top_val']:10.6f}")

    pr(f"\n  -π = {-float(MPI):.10f}")

    # ═══════ Part C: The top contribution ═════════════════════════════
    pr(f"\n{'═' * 76}")
    pr("C. σ_{00}^{top}: PAIRS WITH M = D-2")
    pr(f"{'═' * 76}\n")

    pr("  For M = D-2: val = carries[D-3] - 1")
    pr("  carries[D-2] = 1 (defining property of top)")
    pr("  carries[D-2] = (conv_{D-3} + carries[D-3]) >> 1 = 1")
    pr("  → conv_{D-3} + carries[D-3] ≥ 2")
    pr()

    for cd in convergence_data:
        K = cd['K']
        if K < 6 or K > 12:
            continue
        mean_top = cd['s00_top_val'] / cd['s00_top_cnt'] if cd['s00_top_cnt'] else 0
        mean_low = cd['s00_low_val'] / cd['s00_low_cnt'] if cd['s00_low_cnt'] else 0
        mean_10 = cd['s10_low_val'] / cd['s10_low_cnt'] if cd['s10_low_cnt'] else 0
        pr(f"  K={K:2d}: ⟨val⟩_top={mean_top:+.6f}  ⟨val⟩_00_low={mean_low:+.6f}  "
           f"⟨val⟩_10={mean_10:+.6f}")

    # ═══════ Part D: R as function of top/low ═════════════════════════
    pr(f"\n{'═' * 76}")
    pr("D. R EXPRESSED VIA TOP/LOW DECOMPOSITION")
    pr(f"{'═' * 76}\n")

    pr("  R = σ10 / σ00 = σ10_low / (σ00_top + σ00_low)")
    pr("  = R_low · (σ00_low / σ00)")
    pr("  = R_low · (1 - f_top)")
    pr()
    pr("  where f_top = σ00_top / σ00")
    pr()
    pr("  If R → -π, then R_low · (1 - f_top) → -π")
    pr()

    pr(f"  {'K':>3s}  {'R_low':>12s}  {'1-f_top':>12s}  "
       f"{'R_low*(1-f_top)':>16s}  {'actual R':>12s}")

    for cd in convergence_data:
        K = cd['K']
        R_low = cd['R_low']
        f_top = cd['frac_top_val']
        computed = R_low * (1 - f_top) if R_low != float('inf') else float('inf')
        pr(f"  {K:3d}  {R_low:+12.8f}  {1-f_top:12.8f}  "
           f"{computed:+16.8f}  {cd['R']:+12.8f}")

    # ═══════ Part E: Symmetry — (1,0) vs (0,1) ═══════════════════════
    pr(f"\n{'═' * 76}")
    pr("E. SYMMETRY: σ_{10} vs σ_{01}")
    pr(f"{'═' * 76}\n")

    pr("  Both sectors (1,0) and (0,1) have conv_{D-2} = 1, hence M < D-2.")
    pr("  They differ in which factor has the perturbed bit.")
    pr()

    for cd in convergence_data:
        K = cd['K']
        if K < 5:
            continue
        ratio_sym = cd['s01_low_val'] / cd['s10_low_val'] if cd['s10_low_val'] else float('inf')
        pr(f"  K={K:2d}: σ10={cd['s10_low_val']:10d}  σ01={cd['s01_low_val']:10d}  "
           f"σ01/σ10={ratio_sym:+.8f}")

    # ═══════ Part F: Decompose by M position ═════════════════════════
    pr(f"\n{'═' * 76}")
    pr("F. σ_{ac} DECOMPOSED BY M POSITION")
    pr(f"{'═' * 76}\n")

    for K_target in [8, 10, 12]:
        found = False
        for cd in convergence_data:
            if cd['K'] == K_target:
                found = True
                break
        if not found:
            continue

        K = K_target
        D = 2 * K - 1
        data = enumerate_dodd_decomp(K)

        sigma_by_M = {}
        count_by_M = {}
        for d in data:
            key = (d['a'], d['c'], d['M'])
            sigma_by_M[key] = sigma_by_M.get(key, 0) + d['val']
            count_by_M[key] = count_by_M.get(key, 0) + 1

        pr(f"  K={K} (D={D}):")
        for sector in [(0, 0), (1, 0)]:
            pr(f"    Sector {sector}:")
            for M in range(1, D + 1):
                key = (sector[0], sector[1], M)
                s = sigma_by_M.get(key, 0)
                n = count_by_M.get(key, 0)
                if n > 0:
                    pr(f"      M={M:2d}: Σval={s:10d}  N={n:8d}  ⟨val⟩={s/n:+.6f}")
        pr()

    # ═══════ Part G: The ratio R(M) by carry position ═════════════════
    pr(f"\n{'═' * 76}")
    pr("G. R(M) = σ10(M) / σ00(M) BY CARRY POSITION")
    pr(f"{'═' * 76}\n")

    pr("  Does R(M) depend on M? Or is R ≈ -π for each M separately?")
    pr()

    for K_target in [10, 12]:
        found = False
        for cd in convergence_data:
            if cd['K'] == K_target:
                found = True
                break
        if not found:
            continue

        K = K_target
        D = 2 * K - 1
        data = enumerate_dodd_decomp(K)

        sigma_by_M = {}
        count_by_M = {}
        for d in data:
            key = (d['a'], d['c'], d['M'])
            sigma_by_M[key] = sigma_by_M.get(key, 0) + d['val']
            count_by_M[key] = count_by_M.get(key, 0) + 1

        pr(f"  K={K} (D={D}):")
        for M in range(1, D):
            s00 = sigma_by_M.get((0, 0, M), 0)
            s10 = sigma_by_M.get((1, 0, M), 0)
            n00 = count_by_M.get((0, 0, M), 0)
            n10 = count_by_M.get((1, 0, M), 0)
            if n00 > 0 and n10 > 0:
                R_M = s10 / s00 if s00 != 0 else float('inf')
                pr(f"    M={M:2d}: σ00={s00:10d} (N={n00:6d})  "
                   f"σ10={s10:10d} (N={n10:6d})  R(M)={R_M:+.6f}")
            elif n00 > 0 or n10 > 0:
                pr(f"    M={M:2d}: σ00={s00:10d} (N={n00:6d})  "
                   f"σ10={s10:10d} (N={n10:6d})  [only one sector]")
        pr()

    # ═══════ Part H: carries[D-3] distribution ═══════════════════════
    pr(f"\n{'═' * 76}")
    pr("H. carries[D-3] DISTRIBUTION — THE VAL DETERMINANT")
    pr(f"{'═' * 76}\n")

    pr("  val = carries[M-1] - 1")
    pr("  For M = D-2 (top): val = carries[D-3] - 1")
    pr("  For M = D-3: val = carries[D-4] - 1")
    pr("  What is the distribution of carries at M-1?")
    pr()

    for K_target in [10, 12]:
        K = K_target
        D = 2 * K - 1
        data = enumerate_dodd_decomp(K)

        for sector in [(0, 0), (1, 0)]:
            cm1_dist = {}
            for d in data:
                if (d['a'], d['c']) != sector:
                    continue
                M = d['M']
                cm1 = d['carries'][M - 1] if M > 0 else 0
                cm1_dist[cm1] = cm1_dist.get(cm1, 0) + 1

            total = sum(cm1_dist.values())
            pr(f"  K={K}, sector {sector}: c_{{M-1}} distribution:")
            for cm1_val in sorted(cm1_dist.keys()):
                cnt = cm1_dist[cm1_val]
                pr(f"    c_{{M-1}}={cm1_val}: {cnt:8d} ({100*cnt/total:.2f}%)  "
                   f"val={cm1_val-1:+d}")
        pr()

    # ═══════ Part I: The "renormalized" ratio ═════════════════════════
    pr(f"\n{'═' * 76}")
    pr("I. NORMALIZED RATIOS: σ_{ac}/N_{ac}")
    pr(f"{'═' * 76}\n")

    pr("  ⟨val⟩_{ac} = σ_{ac} / N_{ac}")
    pr("  R = σ10/σ00 = (⟨val⟩_10 · N10) / (⟨val⟩_00 · N00)")
    pr("  = (⟨val⟩_10 / ⟨val⟩_00) · (N10 / N00)")
    pr()

    pr(f"  {'K':>3s}  {'⟨val⟩_00':>10s}  {'⟨val⟩_10':>10s}  "
       f"{'N10/N00':>10s}  {'⟨val⟩10/⟨val⟩00':>16s}  {'R':>12s}")

    for cd in convergence_data:
        K = cd['K']
        N00 = cd['s00_top_cnt'] + cd['s00_low_cnt']
        N10 = cd['s10_low_cnt']
        mean00 = cd['S00'] / N00 if N00 else 0
        mean10 = cd['S10'] / N10 if N10 else 0
        ratio_N = N10 / N00 if N00 else 0
        ratio_mean = mean10 / mean00 if mean00 else 0
        pr(f"  {K:3d}  {mean00:+10.6f}  {mean10:+10.6f}  "
           f"{ratio_N:10.6f}  {ratio_mean:+16.8f}  {cd['R']:+12.8f}")

    pr()
    pr(f"  R = (⟨val⟩10/⟨val⟩00) · (N10/N00)")
    pr(f"  If N10/N00 → 1/3 and ⟨val⟩10/⟨val⟩00 → -3π:")
    pr(f"    then R → (1/3)·(-3π) = -π ✓")

    # ═══════ Part J: Count ratio ═════════════════════════════════════
    pr(f"\n{'═' * 76}")
    pr("J. COUNT RATIO: N10/N00 CONVERGENCE")
    pr(f"{'═' * 76}\n")

    for cd in convergence_data:
        K = cd['K']
        N00 = cd['s00_top_cnt'] + cd['s00_low_cnt']
        N10 = cd['s10_low_cnt']
        ratio = N10 / N00 if N00 else 0
        pr(f"  K={K:2d}: N00={N00:10d}  N10={N10:10d}  "
           f"N10/N00={ratio:.8f}  1/3={1/3:.8f}  "
           f"diff={ratio - 1/3:+.8f}")

    # ═══════ Summary ══════════════════════════════════════════════════
    pr(f"\n{'═' * 76}")
    pr("SUMMARY")
    pr(f"{'═' * 76}\n")

    pr(f"  Total runtime: {time.time() - t0:.1f}s")
    pr()
    pr("  KEY STRUCTURAL DECOMPOSITION:")
    pr("  σ00 = σ00_top (M=D-2) + σ00_low (M<D-2)")
    pr("  σ10 = σ10_low only (M<D-2 forced by D-odd + conv_{D-2}=1)")
    pr()
    pr("  CONVERGENCE QUESTIONS:")
    pr("  1. Does R_low = σ10_low/σ00_low → some limit?")
    pr("  2. Does f_top = σ00_top/σ00 → some limit?")
    pr("  3. R = R_low · (1 - f_top) → -π")
    pr("  4. Does R(M) (ratio at fixed M) depend on M or approach -π uniformly?")


if __name__ == '__main__':
    main()
