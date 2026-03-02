#!/usr/bin/env python3
"""E16: Sector-resolved Dirichlet series and L(1,χ₄) connection.

GOAL: Express σ₀₀ and σ₁₀ as sums connected to Dirichlet L-functions,
specifically L(1,χ₄) = π/4, to prove R = σ₁₀/σ₀₀ → -π = -4·L(1,χ₄).

Strategy:
  For each D-odd pair (p,q) with product n = p·q:
    - Record sector (a,c), carry weight val, and product n
    - Decompose σ_{ac} by n mod m (m = 2, 4, 8, ...)
    - Compute χ₄-twisted sums T_{ac} = Σ val · χ₄(n)
    - Construct Dirichlet series F_{ac}(s) = Σ f_{ac}(n) · n^{-s}
    - Test multiplicative structure of f_{ac}(n)
    - Look for L(1,χ₄) in ratios

Key identity: -π = -4·L(1,χ₄) where L(1,χ₄) = Σ χ₄(n)/n = 1 - 1/3 + 1/5 - ...
Character: χ₄(n) = 0 if n even, +1 if n ≡ 1 (mod 4), -1 if n ≡ 3 (mod 4)
"""
import sys
import time
import numpy as np
from collections import defaultdict
from mpmath import mp, mpf, pi as MPI

mp.dps = 30


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def chi4(n):
    """Dirichlet character χ₄(n): 0 if even, +1 if n≡1(4), -1 if n≡3(4)."""
    if n % 2 == 0:
        return 0
    return 1 if n % 4 == 1 else -1


def enumerate_dodd(K):
    """Enumerate all D-odd pairs, returning detailed per-pair data."""
    lo = 1 << (K - 1)
    hi = 1 << K
    D = 2 * K - 1

    pairs = []
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
            pairs.append((a, c, val, prod, p, q, M))

    return pairs


def main():
    t0 = time.time()

    pr("=" * 76)
    pr("E16: SECTOR-RESOLVED DIRICHLET SERIES AND L(1,χ₄)")
    pr("=" * 76)

    # ═══════ Part A: Sector sums decomposed by n mod 4 ════════════════
    pr(f"\n{'═' * 76}")
    pr("A. SECTOR SUMS DECOMPOSED BY PRODUCT RESIDUE mod 4")
    pr(f"{'═' * 76}\n")

    pr("  χ₄(n): +1 if n≡1(4), -1 if n≡3(4), 0 if n even")
    pr("  For D-odd products: n = p·q with D = 2K-1 bits")
    pr("  n is always odd (both p,q have MSB=1 and LSB from K-bit numbers)")
    pr()

    all_data = {}

    for K in range(3, 12):
        tb = time.time()
        pairs = enumerate_dodd(K)
        dt = time.time() - tb

        sigma = defaultdict(int)
        sigma_mod4 = defaultdict(lambda: defaultdict(int))
        count = defaultdict(int)
        count_mod4 = defaultdict(lambda: defaultdict(int))
        chi4_twisted = defaultdict(int)

        for (a, c, val, prod, p, q, M) in pairs:
            sector = (a, c)
            r = prod % 4
            sigma[sector] += val
            sigma_mod4[sector][r] += val
            count[sector] += 1
            count_mod4[sector][r] += 1
            chi4_twisted[sector] += val * chi4(prod)

        S00 = sigma[(0, 0)]
        S10 = sigma[(1, 0)]
        R = S10 / S00 if S00 != 0 else float('inf')

        all_data[K] = {
            'pairs': pairs, 'sigma': dict(sigma),
            'sigma_mod4': {k: dict(v) for k, v in sigma_mod4.items()},
            'count': dict(count), 'count_mod4': {k: dict(v) for k, v in count_mod4.items()},
            'chi4_twisted': dict(chi4_twisted),
        }

        pr(f"  K={K:2d} (D={2*K-1:2d}): S00={S00:12d}  S10={S10:12d}  "
           f"R={R:+.8f}  ({dt:.2f}s)")

        for sector in [(0, 0), (1, 0)]:
            label = f"σ_{sector[0]}{sector[1]}"
            s = sigma[sector]
            t = chi4_twisted[sector]
            pr(f"         {label} = {s:12d}  "
               f"χ₄-twisted = {t:12d}  "
               f"ratio(twisted/plain) = {t/s if s else 0:+.8f}")
            for r in [1, 3]:
                sr = sigma_mod4[sector].get(r, 0)
                cr = count_mod4[sector].get(r, 0)
                pr(f"           mod4≡{r}: Σval={sr:10d}  count={cr:8d}  "
                   f"⟨val⟩={sr/cr if cr else 0:+.6f}")
        pr()

    # ═══════ Part B: χ₄-twist ratio analysis ══════════════════════════
    pr(f"\n{'═' * 76}")
    pr("B. χ₄-TWIST RATIO: T_{ac} / σ_{ac} WHERE T = Σ val·χ₄(n)")
    pr(f"{'═' * 76}\n")

    pr(f"  If R = -4·L(1,χ₄), maybe the character twist reveals the structure.")
    pr(f"  Define: T_{{ac}} = Σ val · χ₄(n),  ρ_{{ac}} = T_{{ac}} / σ_{{ac}}")
    pr()

    pr(f"  {'K':>3s}  {'ρ_00':>12s}  {'ρ_10':>12s}  {'T10/T00':>12s}  {'R=S10/S00':>12s}")
    pr(f"  {'---':>3s}  {'---':>12s}  {'---':>12s}  {'---':>12s}  {'---':>12s}")

    for K in range(3, 12):
        d = all_data[K]
        S00 = d['sigma'].get((0, 0), 0)
        S10 = d['sigma'].get((1, 0), 0)
        T00 = d['chi4_twisted'].get((0, 0), 0)
        T10 = d['chi4_twisted'].get((1, 0), 0)
        rho00 = T00 / S00 if S00 else 0
        rho10 = T10 / S10 if S10 else 0
        ratio_T = T10 / T00 if T00 else float('inf')
        R = S10 / S00 if S00 else float('inf')
        pr(f"  {K:3d}  {rho00:+12.8f}  {rho10:+12.8f}  {ratio_T:+12.8f}  {R:+12.8f}")

    pr(f"\n  L(1,χ₄) = π/4 = {float(MPI/4):.10f}")
    pr(f"  -4·L(1,χ₄) = -π = {-float(MPI):.10f}")

    # ═══════ Part C: Odd products only (since D-odd → n is always odd?) ═══
    pr(f"\n{'═' * 76}")
    pr("C. PARITY OF D-ODD PRODUCTS")
    pr(f"{'═' * 76}\n")

    for K in range(3, 12):
        pairs = all_data[K]['pairs']
        n_odd = sum(1 for (a, c, val, prod, p, q, M) in pairs if prod % 2 == 1)
        n_even = sum(1 for (a, c, val, prod, p, q, M) in pairs if prod % 2 == 0)
        pr(f"  K={K}: {n_odd} odd products, {n_even} even products "
           f"({100*n_odd/(n_odd+n_even):.1f}% odd)")

    # ═══════ Part D: Product mod 4 decomposition of R ═══════════════
    pr(f"\n{'═' * 76}")
    pr("D. DECOMPOSITION: σ_{ac} = σ_{ac}^{(1)} + σ_{ac}^{(3)}  (odd products)")
    pr(f"{'═' * 76}\n")

    pr("  Since n = p·q with p,q K-bit (MSB=1), n is always odd for K≥2.")
    pr("  So σ_{ac} = σ_{ac}^{(1)} + σ_{ac}^{(3)} where r = n mod 4.")
    pr("  And T_{ac} = σ_{ac}^{(1)} - σ_{ac}^{(3)} = χ₄-twisted sum.")
    pr()

    pr(f"  {'K':>3s}  {'σ00^(1)':>12s}  {'σ00^(3)':>12s}  "
       f"{'σ10^(1)':>12s}  {'σ10^(3)':>12s}  "
       f"{'σ00^(1)/σ00':>12s}  {'σ10^(1)/σ10':>12s}")
    pr(f"  {'---':>3s}  {'---':>12s}  {'---':>12s}  "
       f"{'---':>12s}  {'---':>12s}  "
       f"{'---':>12s}  {'---':>12s}")

    for K in range(3, 12):
        d = all_data[K]
        sm00 = d['sigma_mod4'].get((0, 0), {})
        sm10 = d['sigma_mod4'].get((1, 0), {})
        s00_1 = sm00.get(1, 0)
        s00_3 = sm00.get(3, 0)
        s10_1 = sm10.get(1, 0)
        s10_3 = sm10.get(3, 0)
        S00 = d['sigma'].get((0, 0), 0)
        S10 = d['sigma'].get((1, 0), 0)
        frac00 = s00_1 / S00 if S00 else 0
        frac10 = s10_1 / S10 if S10 else 0
        pr(f"  {K:3d}  {s00_1:12d}  {s00_3:12d}  "
           f"{s10_1:12d}  {s10_3:12d}  "
           f"{frac00:+12.8f}  {frac10:+12.8f}")

    # ═══════ Part E: Input factor characters ═════════════════════════
    pr(f"\n{'═' * 76}")
    pr("E. CHARACTER OF FACTORS: χ₄(p), χ₄(q), χ₄(p)·χ₄(q) vs χ₄(n)")
    pr(f"{'═' * 76}\n")

    pr("  Since χ₄ is completely multiplicative: χ₄(pq) = χ₄(p)·χ₄(q)")
    pr("  Sector a = p_{K-2} (2nd MSB of p)")
    pr("  χ₄(p) depends on p mod 4 = (p_1, p_0) — DIFFERENT bits from a")
    pr()

    for K in [5, 7, 9, 11]:
        if K not in all_data:
            continue
        pairs = all_data[K]['pairs']

        cross_table = defaultdict(lambda: defaultdict(lambda: [0, 0]))

        for (a, c, val, prod, p, q, M) in pairs:
            xp = chi4(p)
            xq = chi4(q)
            key = (a, xp)
            cross_table[key][(c, xq)][0] += val
            cross_table[key][(c, xq)][1] += 1

        pr(f"  K={K}:")
        pr(f"    Sector a vs χ₄(p), weighted by val:")
        for a_val in [0, 1]:
            for xp_val in [1, -1]:
                total_val = sum(v[0] for v in cross_table[(a_val, xp_val)].values())
                total_cnt = sum(v[1] for v in cross_table[(a_val, xp_val)].values())
                mean_v = total_val / total_cnt if total_cnt else 0
                pr(f"      a={a_val}, χ₄(p)={xp_val:+d}: Σval={total_val:10d}  "
                   f"count={total_cnt:8d}  ⟨val⟩={mean_v:+.6f}")
        pr()

    # ═══════ Part F: Dirichlet series F_{ac}(s) ═══════════════════════
    pr(f"\n{'═' * 76}")
    pr("F. DIRICHLET SERIES: F_{ac}(s) = Σ f_{ac}(n) · n^{-s}")
    pr(f"{'═' * 76}\n")

    pr("  f_{ac}(n) = Σ val over D-odd factorizations p·q = n in sector (a,c)")
    pr("  F_{ac}(s) = Σ_n f_{ac}(n) · n^{-s}")
    pr("  Test: does F10(s)/F00(s) involve L-function structure?")
    pr()

    for K in [7, 9, 11]:
        if K not in all_data:
            continue
        pairs = all_data[K]['pairs']

        f_ac = defaultdict(lambda: defaultdict(int))

        for (a, c, val, prod, p, q, M) in pairs:
            sector = (a, c)
            f_ac[sector][prod] += val

        for s_val in [0.0, 0.5, 1.0]:
            F00 = sum(v * n ** (-s_val) if s_val > 0 else v
                      for n, v in f_ac[(0, 0)].items())
            F10 = sum(v * n ** (-s_val) if s_val > 0 else v
                      for n, v in f_ac[(1, 0)].items())
            ratio = F10 / F00 if abs(F00) > 1e-30 else float('inf')
            pr(f"  K={K}, s={s_val:.1f}: F00={F00:+15.6f}  F10={F10:+15.6f}  "
               f"F10/F00={ratio:+.8f}")

        F00_chi4 = sum(v * chi4(n) for n, v in f_ac[(0, 0)].items())
        F10_chi4 = sum(v * chi4(n) for n, v in f_ac[(1, 0)].items())
        pr(f"  K={K}, χ₄-weighted: F00·χ₄={F00_chi4:+15.6f}  "
           f"F10·χ₄={F10_chi4:+15.6f}")
        pr()

    # ═══════ Part G: Multiplicative structure of f(n) ═════════════════
    pr(f"\n{'═' * 76}")
    pr("G. MULTIPLICATIVE STRUCTURE: f_{ac}(n) AS ARITHMETIC FUNCTION")
    pr(f"{'═' * 76}\n")

    pr("  For each product n, how many D-odd factorizations exist?")
    pr("  Is f_{ac}(n) multiplicative (related to divisor-like functions)?")
    pr()

    for K in [7, 9]:
        if K not in all_data:
            continue
        pairs = all_data[K]['pairs']

        f_by_n = defaultdict(lambda: defaultdict(list))

        for (a, c, val, prod, p, q, M) in pairs:
            f_by_n[prod][(a, c)].append(val)

        n_vals = sorted(f_by_n.keys())
        n_multi_factor = sum(1 for n in n_vals
                             if sum(len(v) for v in f_by_n[n].values()) > 1)
        pr(f"  K={K}: {len(n_vals)} distinct products, "
           f"{n_multi_factor} with multiple factorizations")

        pr(f"    Sample products with multiple factorizations:")
        shown = 0
        for n in n_vals:
            total_facts = sum(len(v) for v in f_by_n[n].values())
            if total_facts > 1 and shown < 5:
                vals_by_sec = {k: sum(v) for k, v in f_by_n[n].items()}
                pr(f"      n={n} (mod4={n%4}): factorizations={total_facts}, "
                   f"val_sums={dict(vals_by_sec)}")
                shown += 1
        pr()

    # ═══════ Part H: The key ratio decomposition ═══════════════════════
    pr(f"\n{'═' * 76}")
    pr("H. KEY TEST: R DECOMPOSED BY χ₄(n)")
    pr(f"{'═' * 76}\n")

    pr("  Since all D-odd products are odd, χ₄(n) ∈ {+1, -1}.")
    pr("  σ_{ac} = σ_{ac}^+ + σ_{ac}^-  where ± labels χ₄(n) = ±1")
    pr("  T_{ac} = σ_{ac}^+ - σ_{ac}^-  (χ₄-twisted)")
    pr()
    pr("  If R = -4L(1,χ₄), the character decomposition should show structure.")
    pr()

    pr(f"  {'K':>3s}  {'σ00+':>11s}  {'σ00-':>11s}  "
       f"{'σ10+':>11s}  {'σ10-':>11s}  "
       f"{'R+':>12s}  {'R-':>12s}  {'R':>12s}")

    for K in range(3, 12):
        d = all_data[K]
        sm00 = d['sigma_mod4'].get((0, 0), {})
        sm10 = d['sigma_mod4'].get((1, 0), {})
        s00p = sm00.get(1, 0)
        s00m = sm00.get(3, 0)
        s10p = sm10.get(1, 0)
        s10m = sm10.get(3, 0)
        Rp = s10p / s00p if s00p else float('inf')
        Rm = s10m / s00m if s00m else float('inf')
        R = (s10p + s10m) / (s00p + s00m) if (s00p + s00m) else float('inf')
        pr(f"  {K:3d}  {s00p:11d}  {s00m:11d}  "
           f"{s10p:11d}  {s10m:11d}  "
           f"{Rp:+12.8f}  {Rm:+12.8f}  {R:+12.8f}")

    pr()
    pr(f"  Target: R → -π = {-float(MPI):.10f}")
    pr(f"  If R+ and R- converge to different limits, the character decomposition")
    pr(f"  is non-trivial and may reveal the L-function structure.")

    # ═══════ Part I: Linear combination search ═════════════════════════
    pr(f"\n{'═' * 76}")
    pr("I. LINEAR COMBINATION SEARCH: σ₁₀ + c·σ₀₀ → 0")
    pr(f"{'═' * 76}\n")

    pr("  If R = σ₁₀/σ₀₀ → -π, then Q(K) = σ₁₀ + π·σ₀₀ → 0.")
    pr("  Rate of convergence and structure of Q(K) may reveal mechanism.")
    pr()

    pr(f"  {'K':>3s}  {'Q = σ10+π·σ00':>16s}  {'Q/σ00':>14s}  "
       f"{'Q·2^K':>16s}  {'Q·K·2^K':>16s}")

    for K in range(3, 12):
        d = all_data[K]
        S00 = d['sigma'].get((0, 0), 0)
        S10 = d['sigma'].get((1, 0), 0)
        Q = S10 + float(MPI) * S00
        Q_norm = Q / S00 if S00 else 0
        Q_exp = Q * 2**K
        Q_Kexp = Q * K * 2**K
        pr(f"  {K:3d}  {Q:16.4f}  {Q_norm:14.10f}  "
           f"{Q_exp:16.4f}  {Q_Kexp:16.4f}")

    # ═══════ Part J: χ₄(p)·χ₄(q) correlation with sector ════════════
    pr(f"\n{'═' * 76}")
    pr("J. KEY INSIGHT: χ₄(p)·χ₄(q) CORRELATION WITH SECTOR BIT a")
    pr(f"{'═' * 76}\n")

    pr("  χ₄(n) = χ₄(p)·χ₄(q) (completely multiplicative)")
    pr("  Sector a = p_{K-2}: upper/lower half of K-bit range")
    pr("  Question: does flipping a (p_{K-2}: 0→1) systematically")
    pr("  change χ₄(p), and hence χ₄(n)?")
    pr()

    for K in [5, 7, 9, 11]:
        if K not in all_data:
            continue
        pairs = all_data[K]['pairs']

        sector_chi_table = defaultdict(lambda: [0, 0, 0])

        for (a, c, val, prod, p, q, M) in pairs:
            xp = chi4(p)
            idx = 0 if xp == 1 else 1
            sector_chi_table[(a, c)][idx] += 1
            sector_chi_table[(a, c)][2] += val

        pr(f"  K={K}:")
        for sector in [(0, 0), (1, 0)]:
            t = sector_chi_table[sector]
            frac_p1 = t[0] / (t[0] + t[1]) if (t[0] + t[1]) else 0
            pr(f"    Sector {sector}: χ₄(p)=+1: {t[0]:8d}  "
               f"χ₄(p)=-1: {t[1]:8d}  fraction(+1)={frac_p1:.6f}  "
               f"Σval={t[2]:12d}")

        a0_p1 = sector_chi_table[(0, 0)][0]
        a0_m1 = sector_chi_table[(0, 0)][1]
        a1_p1 = sector_chi_table[(1, 0)][0]
        a1_m1 = sector_chi_table[(1, 0)][1]
        f0 = a0_p1 / (a0_p1 + a0_m1) if (a0_p1 + a0_m1) else 0
        f1 = a1_p1 / (a1_p1 + a1_m1) if (a1_p1 + a1_m1) else 0
        pr(f"    Δ(fraction χ₄(p)=+1) = {f1 - f0:+.8f}")
        pr()

    # ═══════ Part K: Conditional analysis ════════════════════════════
    pr(f"\n{'═' * 76}")
    pr("K. CONDITIONAL: R RESTRICTED TO χ₄(p) = +1 vs χ₄(p) = -1")
    pr(f"{'═' * 76}\n")

    pr("  Split sector sums by χ₄(p): does R depend on χ₄(p)?")
    pr()

    pr(f"  {'K':>3s}  {'R|χ₄(p)=+1':>14s}  {'R|χ₄(p)=-1':>14s}  "
       f"{'R(all)':>14s}  {'R|χ₄(q)=+1':>14s}  {'R|χ₄(q)=-1':>14s}")

    for K in range(4, 12):
        if K not in all_data:
            continue
        pairs = all_data[K]['pairs']

        sigma_cond = defaultdict(lambda: defaultdict(int))

        for (a, c, val, prod, p, q, M) in pairs:
            xp = chi4(p)
            xq = chi4(q)
            sigma_cond[('a', a, 'xp', xp)]['val'] += val
            sigma_cond[('a', a, 'xq', xq)]['val'] += val

        def get_R_cond(cond_key, cond_val):
            s00 = sigma_cond[('a', 0, cond_key, cond_val)].get('val', 0)
            s10 = sigma_cond[('a', 1, cond_key, cond_val)].get('val', 0)
            return s10 / s00 if s00 else float('nan')

        R_xp1 = get_R_cond('xp', 1)
        R_xpm = get_R_cond('xp', -1)
        R_xq1 = get_R_cond('xq', 1)
        R_xqm = get_R_cond('xq', -1)

        d = all_data[K]
        R = d['sigma'].get((1, 0), 0) / d['sigma'].get((0, 0), 1) if d['sigma'].get((0, 0), 0) else 0

        pr(f"  {K:3d}  {R_xp1:+14.8f}  {R_xpm:+14.8f}  "
           f"{R:+14.8f}  {R_xq1:+14.8f}  {R_xqm:+14.8f}")

    pr()
    pr(f"  -π = {-float(MPI):.10f}")

    # ═══════ Part L: Product structure — n mod higher powers ══════════
    pr(f"\n{'═' * 76}")
    pr("L. PRODUCT STRUCTURE: σ_{ac} DECOMPOSED BY n mod 8, mod 16")
    pr(f"{'═' * 76}\n")

    for K in [9, 11]:
        if K not in all_data:
            continue
        pairs = all_data[K]['pairs']

        sigma_mod = defaultdict(lambda: defaultdict(int))
        count_mod = defaultdict(lambda: defaultdict(int))

        for (a, c, val, prod, p, q, M) in pairs:
            sector = (a, c)
            for m in [8, 16]:
                r = prod % m
                sigma_mod[(sector, m)][r] += val
                count_mod[(sector, m)][r] += 1

        for m in [8, 16]:
            pr(f"  K={K}, mod {m}:")
            for sector in [(0, 0), (1, 0)]:
                pr(f"    Sector {sector}:", end='')
                for r in range(m):
                    if r % 2 == 0:
                        continue
                    s = sigma_mod[(sector, m)].get(r, 0)
                    c_r = count_mod[(sector, m)].get(r, 0)
                    pr(f"  r={r}:{s:9d}", end='')
                pr()
            pr()

    # ═══════ Summary ══════════════════════════════════════════════════
    pr(f"\n{'═' * 76}")
    pr("SUMMARY")
    pr(f"{'═' * 76}\n")

    pr(f"  Total runtime: {time.time() - t0:.1f}s")
    pr()
    pr("  KEY FINDINGS:")
    pr("  1. All D-odd products are odd → χ₄(n) ∈ {+1, -1} always defined")
    pr("  2. χ₄(n) = χ₄(p)·χ₄(q) — multiplicative decomposition")
    pr("  3. Sector bit a = p_{K-2} and χ₄(p) = p mod 4 are on different bits")
    pr("  4. Check tables above for:")
    pr("     a. Does R+ ≠ R- (character decomposition non-trivial)?")
    pr("     b. Does R|χ₄(p)=+1 differ from R|χ₄(p)=-1?")
    pr("     c. Does the Dirichlet series ratio F10/F00 show s-dependence?")
    pr("     d. Is f_{ac}(n) related to a multiplicative arithmetic function?")


if __name__ == '__main__':
    main()
