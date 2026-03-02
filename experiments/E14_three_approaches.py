#!/usr/bin/env python3
"""E14: Three approaches to prove R = σ₁₀/σ₀₀ → -π.

Incorporates K=20 data and tests three proof strategies:
  A. CLT + Brownian bridge (continuous limit)
  B. Character sum → L(1,χ₄) connection
  C. Bootstrap self-consistency (α = 1/6, R = -π)
"""
import sys
import time
import numpy as np
import mpmath
from mpmath import mp, mpf, pi, log, cos as mcos, sin as msin, nstr, zeta, euler

mp.dps = 40


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


SECTOR_DATA = {
    3:  {'S00_c0': 0, 'S00_c1': -1, 'S10': 0, 'Nt': 4**2},
    4:  {'S00_c0': -1, 'S00_c1': -3, 'S10': -1, 'Nt': 4**3},
    5:  {'S00_c0': -6, 'S00_c1': -6, 'S10': -3, 'Nt': 4**4},
    6:  {'S00_c0': -32, 'S00_c1': -11, 'S10': -4, 'Nt': 4**5},
    7:  {'S00_c0': -114, 'S00_c1': -28, 'S10': 13, 'Nt': 4**6},
    8:  {'S00_c0': -373, 'S00_c1': -61, 'S10': 167, 'Nt': 4**7},
    9:  {'S00_c0': -1155, 'S00_c1': -171, 'S10': 963, 'Nt': 4**8},
    10: {'S00_c0': -3684, 'S00_c1': -472, 'S10': 4648, 'Nt': 4**9},
    11: {'S00_c0': -11768, 'S00_c1': -1486, 'S10': 20590, 'Nt': 4**10},
    12: {'S00_c0': -39195, 'S00_c1': -5107, 'S10': 87256, 'Nt': 4**11},
    13: {'S00_c0': -135206, 'S00_c1': -18944, 'S10': 360533, 'Nt': 4**12},
    14: {'S00_c0': -487465, 'S00_c1': -72541, 'S10': 1468632, 'Nt': 4**13},
    15: {'S00_c0': -1817747, 'S00_c1': -283679, 'S10': 5934740, 'Nt': 4**14},
    16: {'S00_c0': -6959865, 'S00_c1': -1122027, 'S10': 23873440, 'Nt': 4**15},
    17: {'S00_c0': -27101298, 'S00_c1': -4462799, 'S10': 95791014, 'Nt': 4**16},
    18: {'S00_c0': -106725895, 'S00_c1': -17801108, 'S10': 383811312, 'Nt': 4**17},
    19: {'S00_c0': -424591756, 'S00_c1': -69585007, 'S10': 1536648367, 'Nt': 4**18},
    20: {'S00_c0': -1683746513, 'S00_c1': -284216234, 'S10': 6149608524, 'Nt': 4**19},
}


def main():
    t0 = time.time()

    pr("=" * 78)
    pr("E14: THREE APPROACHES TO PROVE R → -π")
    pr("=" * 78)

    # ═══════ Prelude: Updated convergence with K=20 ════════════════
    pr(f"\n{'═' * 78}")
    pr("0. CONVERGENCE TABLE WITH K=20 DATA")
    pr(f"{'═' * 78}\n")

    pr(f"  {'K':>3} {'R(K)':>14} {'R+π':>14} {'digits':>8} {'ratio':>8}")
    prev_dr = None
    for K in sorted(SECTOR_DATA):
        d = SECTOR_DATA[K]
        S00 = d['S00_c0'] + d['S00_c1']
        S10 = d['S10']
        if S00 == 0:
            continue
        R = mpf(S10) / mpf(S00)
        dr = R + pi
        digits = float(-mpmath.log10(abs(dr))) if dr != 0 else 99
        ratio = ""
        if prev_dr is not None and prev_dr != 0:
            ratio = f"{float(abs(dr / prev_dr)):.4f}"
        pr(f"  {K:3d} {float(R):+14.8f} {float(dr):+14.8f} {digits:8.2f} {ratio:>8}")
        prev_dr = dr

    # ══════════════════════════════════════════════════════════════
    # APPROACH A: CLT + BROWNIAN BRIDGE
    # ══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 78}")
    pr("A. APPROACH A: CLT + BROWNIAN BRIDGE CONTINUOUS LIMIT")
    pr(f"{'═' * 78}\n")

    pr("  MODEL: carry recursion c_{j+1} = (conv_j + c_j) / 2")
    pr("  conv_j has n_j terms, each Bernoulli(1/4)")
    pr("  E[conv_j] = n_j/4, Var[conv_j] = 3n_j/16")
    pr()
    pr("  In the bulk (j ≈ K), n_j ≈ K, so conv_j ≈ K/4 + √(3K/16)·Z_j")
    pr("  The carry becomes: c_j = Σ_{k<j} (1/2)^{j-k} · conv_k")
    pr("  This is an AR(1)/Ornstein-Uhlenbeck process.")
    pr()

    pr("  SECTOR PERTURBATION: flipping a = p_{K-2} changes conv_j for j ∈ [K-2, 2K-3]")
    pr("  Δconv_j = q_{j-(K-2)} ∈ {0,1} (Bernoulli(1/2))")
    pr()
    pr("  Expected carry change:")
    pr("  E[Δc_j] = (1/2) · Σ_{k=K-2}^{min(j-1,2K-3)} (1/2)^{j-k}")
    pr()

    pr(f"  {'K':>4} {'E[Δc] at j=D-2':>16} {'E[Δc] at j=K':>16} "
       f"{'Max E[Δc]':>12}")
    for K in [10, 20, 50, 100, 200, 500]:
        D = 2 * K - 1
        delta_c = np.zeros(D + 1)
        for j in range(1, D + 1):
            for k in range(max(0, K - 2), min(j, 2 * K - 2)):
                delta_c[j] += 0.5 * (0.5) ** (j - k)
        pr(f"  {K:4d} {delta_c[D-2]:16.8f} {delta_c[K]:16.8f} "
           f"{np.max(delta_c):12.8f}")

    pr()
    pr("  OBSERVATION: E[Δc] at all positions converges to finite values.")
    pr("  The sector perturbation creates a DETERMINISTIC shift in mean carry.")
    pr("  The RATIO of perturbed/unperturbed carry weights gives R.")
    pr()

    pr("  But: val = c_{M-1} - 1 depends on the FLUCTUATIONS, not just the mean.")
    pr("  The CLT gives the fluctuation distribution, and the bridge condition")
    pr("  (carries must reach 0 at both ends) constrains the path.")
    pr()

    pr("  Brownian bridge Green's function:")
    pr("  G(x,y) = min(x,y)(L-max(x,y))/L on [0,L]")
    pr("  With source at x = L/2 (midpoint = sector zone center)")
    pr("  G(L/2, y) = y(L-L/2)/L = y/2  for y < L/2")
    pr("  G(L/2, y) = (L/2)(L-y)/L = (L-y)/2  for y > L/2")
    pr()
    pr("  At the cascade y = L-1: G(L/2, L-1) = 1/2")
    pr("  At the midpoint y = L/2: G(L/2, L/2) = L/4")
    pr()
    pr("  The Markov bridge response is POSITIVE and O(1).")
    pr("  For R → -π, the non-Markov corrections must be O(4^K) larger.")
    pr("  This means the CLT/bridge model is NOT sufficient alone —")
    pr("  the discrete arithmetic structure is essential.")

    # ══════════════════════════════════════════════════════════════
    # APPROACH B: CHARACTER SUMS AND L-FUNCTIONS
    # ══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 78}")
    pr("B. APPROACH B: CHARACTER SUMS AND L-FUNCTIONS")
    pr(f"{'═' * 78}\n")

    pr("  KEY IDENTITY: -π = -4·L(1,χ₄) where L(1,χ₄) = 1-1/3+1/5-1/7+...")
    pr()
    pr("  QUESTION: can we express R = σ₁₀/σ₀₀ as a ratio involving L(1,χ₄)?")
    pr()
    pr("  The D-odd condition constrains p·q to have exactly D=2K-1 bits.")
    pr("  For K-bit p,q: p ∈ [2^{K-1}, 2^K-1], product ∈ [2^{D-1}, 2^D-1]")
    pr("  This is a condition on the PRODUCT, not the sum → multiplicative structure.")
    pr()

    pr("  The carry weight val = c_{M-1} - 1 is related to the BINARY EXPANSION")
    pr("  of the product p·q. Specifically:")
    pr("  - M = position of highest non-zero carry")
    pr("  - c_{M-1} = the carry that 'fed' the highest bit")
    pr()

    pr("  Let f(n) = Σ_{p·q = n, K-bit} (c_{M-1}(p,q) - 1)")
    pr("  Then σ₀₀ = Σ_{D-odd n} f₀₀(n), σ₁₀ = Σ_{D-odd n} f₁₀(n)")
    pr("  where f₀₀, f₁₀ restrict to sectors.")
    pr()

    pr("  The D-odd sum has the structure of a SELBERG-DELANGE sum:")
    pr("  Σ_{n ∈ [2^{D-1}, 2^D-1]} r₂(n) · g(n)")
    pr("  where r₂(n) = #{(p,q): p·q = n, K-bit} and g captures the carry weight.")
    pr()

    pr("  CONNECTION TO χ₄:")
    pr("  The character χ₄(n) = (n mod 4: 0→0, 1→1, 2→0, 3→-1)")
    pr("  The sector bit a = p_{K-2} is related to p mod 4:")
    pr("  p = 2^{K-1} + a·2^{K-2} + (lower bits)")
    pr("  For K≥3: p mod 4 depends on p₀ and p₁ (two lowest bits)")
    pr("  So a = p_{K-2} is NOT directly a character mod 4.")
    pr()

    pr("  However, the PRODUCT p·q mod 4 depends on (p mod 4)(q mod 4),")
    pr("  and this IS related to Dirichlet characters.")
    pr()

    pr("  TEST: correlation between sector and product mod 4:")
    for K in [5, 6, 7, 8]:
        lo, hi = 1 << (K - 1), 1 << K
        D = 2 * K - 1
        counts = np.zeros((3, 4), dtype=int)  # sector × (n mod 4)
        for p in range(lo, hi):
            a = (p >> (K - 2)) & 1
            for q in range(lo, hi):
                c = (q >> (K - 2)) & 1
                if a & c:
                    continue
                prod = p * q
                if (prod >> (D - 1)) != 1 or (prod >> D) != 0:
                    continue
                sector = 0 if (a == 0 and c == 0) else (1 if a == 1 else 2)
                counts[sector, prod % 4] += 1

        pr(f"\n    K={K}: D-odd product mod 4 by sector")
        pr(f"    {'sector':>8} {'n≡0':>8} {'n≡1':>8} {'n≡2':>8} {'n≡3':>8}")
        for s, label in enumerate(["(0,0)", "(1,0)", "(0,1)"]):
            if counts[s].sum() == 0:
                continue
            pcts = counts[s] / counts[s].sum() * 100
            pr(f"    {label:>8} {pcts[0]:7.1f}% {pcts[1]:7.1f}% "
               f"{pcts[2]:7.1f}% {pcts[3]:7.1f}%")

    pr()
    pr("  If sectors correlate with product mod 4, the ratio involves χ₄.")
    pr("  The emergence of π via L(1,χ₄) = π/4 would then be natural.")

    # ══════════════════════════════════════════════════════════════
    # APPROACH C: BOOTSTRAP SELF-CONSISTENCY
    # ══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 78}")
    pr("C. APPROACH C: BOOTSTRAP SELF-CONSISTENCY")
    pr(f"{'═' * 78}\n")

    C_odd = pi / 18 - 1 + 6 * log(2) - 3 * log(3)
    pr(f"  Known: Σ_odd → C_odd · 4^{{K-1}} where C_odd = {nstr(C_odd, 16)}")
    pr(f"  Known: α = σ_{{00,c1}}/σ_{{00,c0}} → 1/6")
    pr(f"  Claim: R = σ₁₀/σ₀₀ → -π")
    pr()
    pr("  IF all three are exact, THEN:")
    pr("    σ₀₀ = σ_{00,c0} + σ_{00,c1} = σ_{00,c0}(1 + 1/6) = 7σ_{00,c0}/6")
    pr("    σ₁₀ = -π · σ₀₀ = -7π·σ_{00,c0}/6")
    pr("    Σ_odd = σ₀₀ + 2σ₁₀ = σ_{00,c0}(7/6 - 14π/6) = σ_{00,c0}·(7-14π)/6")
    pr("    → σ_{00,c0} = 6·C_odd/(7-14π) · 4^{K-1}")
    pr()

    sigma_c0_pred = 6 * C_odd / (7 - 14 * pi)
    sigma_c1_pred = sigma_c0_pred / 6
    sigma_00_pred = 7 * sigma_c0_pred / 6
    sigma_10_pred = -pi * sigma_00_pred
    So_pred = sigma_00_pred + 2 * sigma_10_pred

    pr(f"  Predicted densities (σ/Nt = σ/4^{{K-1}}):")
    pr(f"    σ_{{00,c0}}/Nt = {nstr(sigma_c0_pred, 16)}")
    pr(f"    σ_{{00,c1}}/Nt = {nstr(sigma_c1_pred, 16)}")
    pr(f"    σ₀₀/Nt       = {nstr(sigma_00_pred, 16)}")
    pr(f"    σ₁₀/Nt       = {nstr(sigma_10_pred, 16)}")
    pr(f"    Σ_odd/Nt      = {nstr(So_pred, 16)} (should be {nstr(C_odd, 16)})")
    pr()

    pr(f"  {'K':>3} {'σc0/Nt':>14} {'pred':>14} {'err':>10} "
       f"{'σc1/Nt':>14} {'pred':>14} {'err':>10}")
    for K in sorted(SECTOR_DATA):
        d = SECTOR_DATA[K]
        Nt = mpf(d['Nt'])
        c0 = mpf(d['S00_c0']) / Nt
        c1 = mpf(d['S00_c1']) / Nt
        err_c0 = float(c0 - sigma_c0_pred)
        err_c1 = float(c1 - sigma_c1_pred)
        pr(f"  {K:3d} {float(c0):14.10f} {float(sigma_c0_pred):14.10f} "
           f"{err_c0:+10.2e} {float(c1):14.10f} {float(sigma_c1_pred):14.10f} "
           f"{err_c1:+10.2e}")

    # Richardson extrapolation on sector densities
    pr(f"\n  Richardson extrapolation of sector densities:")
    Ks = sorted(SECTOR_DATA.keys())
    for label, get_val, target in [
        ("σ_c0/Nt", lambda K: mpf(SECTOR_DATA[K]['S00_c0'])/mpf(SECTOR_DATA[K]['Nt']),
         sigma_c0_pred),
        ("σ_c1/Nt", lambda K: mpf(SECTOR_DATA[K]['S00_c1'])/mpf(SECTOR_DATA[K]['Nt']),
         sigma_c1_pred),
        ("σ_10/Nt", lambda K: mpf(SECTOR_DATA[K]['S10'])/mpf(SECTOR_DATA[K]['Nt']),
         sigma_10_pred),
    ]:
        vals = [get_val(K) for K in Ks]
        # 3-term Richardson with rate 1/2
        if len(Ks) >= 3:
            K1, K2, K3 = Ks[-3], Ks[-2], Ks[-1]
            v1, v2, v3 = get_val(K1), get_val(K2), get_val(K3)
            r = mpf(1) / 2
            ext = (v3 - r * v2) / (1 - r)
            err = float(ext - target)
            dig = float(-mpmath.log10(abs(ext - target))) if ext != target else 99
            pr(f"    {label}: Richardson({K1},{K2},{K3}) = {nstr(ext, 14)}, "
               f"err = {err:+.4e} ({dig:.1f} digits vs prediction)")

    # ═══════ Part C2: The three-equation system ═══════════════════
    pr(f"\n  THREE-EQUATION SYSTEM:")
    pr("  (1) Σ_odd/Nt → C_odd (analytically known)")
    pr("  (2) α = σ_{c1}/σ_{c0} → 1/6 (B₂)")
    pr("  (3) R = σ₁₀/σ₀₀ → -π")
    pr()
    pr("  These are THREE constraints on THREE unknowns (σ_c0, σ_c1, σ_10).")
    pr("  Any TWO determine the third.")
    pr()
    pr("  If we can prove (1) and (2) independently, then (3) follows IF")
    pr("  the system has a unique solution. But (1)+(2) do NOT determine (3):")
    pr("  they give σ₀₀ = (7/6)σ_c0 and Σ_odd = σ₀₀ + 2σ₁₀,")
    pr("  but σ₁₀ is still free.")
    pr()
    pr("  We need an INDEPENDENT equation for σ₁₀.")

    # ═══════ Part C3: Test alternative formulation ═════════════════
    pr(f"\n  ALTERNATIVE: Express R in terms of known quantities.")
    pr()
    pr("  From prior experiments: C_odd = π/18 - 1 + 6ln(2) - 3ln(3)")
    pr("  From the Bernoulli hierarchy: α = B₂ = 1/6")
    pr("  These involve π (via π/18), ln(2), ln(3).")
    pr()
    pr("  If R = -π, the sector decomposition gives:")
    pr("    σ₀₀/Nt = C_odd/(1 - 2π)")
    pr("    σ₁₀/Nt = -π·C_odd/(1 - 2π)")
    pr()

    sigma_00_from_R = C_odd / (1 - 2 * pi)
    sigma_10_from_R = -pi * sigma_00_from_R

    pr(f"  σ₀₀/Nt predicted = {nstr(sigma_00_from_R, 16)}")
    pr(f"  σ₁₀/Nt predicted = {nstr(sigma_10_from_R, 16)}")
    pr()

    pr(f"  {'K':>3} {'σ₀₀/Nt actual':>16} {'predicted':>16} "
       f"{'err':>12} {'digits':>7}")
    for K in sorted(SECTOR_DATA):
        d = SECTOR_DATA[K]
        Nt = mpf(d['Nt'])
        s00 = mpf(d['S00_c0'] + d['S00_c1']) / Nt
        err = float(s00 - sigma_00_from_R)
        dig = float(-mpmath.log10(abs(s00 - sigma_00_from_R))) if err != 0 else 99
        pr(f"  {K:3d} {float(s00):16.12f} {float(sigma_00_from_R):16.12f} "
           f"{err:+12.4e} {dig:7.2f}")

    # ══════════════════════════════════════════════════════════════
    # APPROACH D: DEEPER — THE GENERATING FUNCTION IDENTITY
    # ══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 78}")
    pr("D. GENERATING FUNCTION: WHAT IDENTITY WOULD PROVE R = -π?")
    pr(f"{'═' * 78}\n")

    pr("  We need to show:")
    pr("    σ₁₀(K) = -π · σ₀₀(K) + O(σ₀₀ · K²/2^K)")
    pr()
    pr("  Equivalently: σ₁₀(K) + π·σ₀₀(K) = o(σ₀₀(K))")
    pr()
    pr("  Define Q(K) = S10(K) + π·S00(K) = Nt·(σ₁₀ + π·σ₀₀)")
    pr("  We need Q(K)/S00(K) → 0.")
    pr()

    pr(f"  {'K':>3} {'Q(K)':>18} {'Q/S00':>14} {'Q/S00·2^K':>14} "
       f"{'Q/S00·K·2^K':>14}")
    for K in sorted(SECTOR_DATA):
        d = SECTOR_DATA[K]
        S00 = d['S00_c0'] + d['S00_c1']
        S10 = d['S10']
        if S00 == 0:
            continue
        Q = mpf(S10) + pi * mpf(S00)
        ratio = float(Q / S00)
        pr(f"  {K:3d} {float(Q):+18.2f} {ratio:+14.8f} "
           f"{ratio * 2**K:+14.2f} {ratio * K * 2**K:+14.2f}")

    pr()
    pr("  Q(K)/S00(K) = R(K) + π → 0 with rate O(K²/2^K).")
    pr("  The quantity Q(K) itself grows as 4^K but Q/S00 vanishes.")

    # ══════════════════════════════════════════════════════════════
    # SYNTHESIS: WHAT DOES EACH APPROACH TELL US?
    # ══════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 78}")
    pr("SYNTHESIS: ASSESSMENT OF PROOF STRATEGIES")
    pr(f"{'═' * 78}\n")

    pr("  APPROACH A (CLT + Brownian Bridge):")
    pr("  ┌─ The continuous limit DOES give a well-defined bridge process")
    pr("  ├─ The sector perturbation has finite expected carry change")
    pr("  ├─ BUT: the Markov bridge predicts R → 0, not -π")
    pr("  ├─ The non-Markov corrections are the entire signal")
    pr("  └─ VERDICT: insufficient alone. The discrete structure is essential.")
    pr("     The CLT smooths away the very features that produce π.")
    pr()
    pr("  APPROACH B (Character Sums):")
    pr("  ┌─ R = -π = -4L(1,χ₄) strongly suggests Dirichlet characters")
    pr("  ├─ The D-odd condition is multiplicative (condition on product)")
    pr("  ├─ Product mod 4 distribution differs by sector")
    pr("  ├─ BUT: the sector bit a = p_{K-2} is NOT a character")
    pr("  └─ VERDICT: promising but requires finding the hidden character")
    pr("     structure linking sectors to L(1,χ₄).")
    pr()
    pr("  APPROACH C (Bootstrap):")
    pr("  ┌─ Three equations (Σ_odd = C_odd, α = 1/6, R = -π) determine")
    pr("  │  all sector densities uniquely")
    pr("  ├─ The predictions match data with increasing precision")
    pr("  ├─ BUT: it's circular — uses R = -π as input")
    pr("  └─ VERDICT: strong consistency check, not a proof.")
    pr("     However, any independent proof of σ₀₀ or σ₁₀ would close the loop.")
    pr()
    pr("  RECOMMENDED PATH:")
    pr("  1. Prove the individual sector densities σ₀₀/Nt → C_odd/(1-2π)")
    pr("     and σ₁₀/Nt → -πC_odd/(1-2π) INDEPENDENTLY")
    pr("  2. This requires a GENERATING FUNCTION for the sector-resolved")
    pr("     carry weight sum, decomposed over D-odd products")
    pr("  3. The generating function should naturally produce the factor")
    pr("     L(1,χ₄) = π/4, which would give the -π in the ratio")
    pr()
    pr("  CONCRETE NEXT STEP:")
    pr("  Compute the generating function F(s) = Σ_{n D-odd} f_ac(n) · n^{-s}")
    pr("  for each sector. If F_10(s)/F_00(s) → -π at some critical s,")
    pr("  we have a path to an analytic proof via Tauberian theorems.")

    pr(f"\n  Total runtime: {time.time() - t0:.1f}s")
    pr("=" * 78)


if __name__ == '__main__':
    main()
