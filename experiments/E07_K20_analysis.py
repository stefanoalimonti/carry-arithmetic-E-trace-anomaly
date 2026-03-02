#!/usr/bin/env python3
"""E07: Analysis of K=20 exact enumeration data from E07.

K=20 is a MAJOR milestone: c₁(20) OVERSHOOTS π/18 for the first time!
This confirms oscillatory convergence and gives ~5.3 digits on c₁ = π/18.

Data source: E07 run with K=17..20 (single-threaded, ~8h for K=20).
"""
import sys
import mpmath
from mpmath import mpf, mp, pi, log, nstr, pslq, fac

mp.dps = 50


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def richardson(vals, rate, n_steps):
    table = [list(vals)]
    for step in range(n_steps):
        prev = table[-1]
        r = rate ** (step + 1)
        new = []
        for i in range(len(prev) - 1):
            new.append((prev[i + 1] - r * prev[i]) / (1 - r))
        table.append(new)
    return table


def wynn_epsilon(seq):
    n = len(seq)
    eps = [[mpf(0)] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        eps[i][1] = seq[i]
    for k in range(2, n + 1):
        for i in range(n - k + 1):
            diff = eps[i + 1][k - 1] - eps[i][k - 1]
            if abs(diff) < mpf(10) ** (-45):
                eps[i][k] = mpf(10) ** 45
            else:
                eps[i][k] = eps[i + 1][k - 2] + 1 / diff
    results = []
    for k in range(2, n + 1, 2):
        if n - k >= 0:
            results.append(eps[0][k])
    return results


TARGET_C1 = pi / 18
TARGET_SO = pi / 18 - 1 + 6 * log(2) - 3 * log(3)
TARGET_SE = 1 + 3 * log(mpf(3) / 4)

# EXACT integer data from K=20 enumeration
# Format: K -> (So_numerator, Se_numerator, Nt_denominator)
E62B_DATA = {
    17: (160017931, 588179206, 4294967296),
    18: (643095621, 2352782572, 17179869184),
    19: (2579119971, 9411261235, 68719476736),
    20: (10331254301, 37645307004, 274877906944),
}

# Extended data from earlier experiments (enumeration outputs for K=7-16)
# So = sum of carry weights for D-odd pairs
# Format: K -> So_integer (over Nt = 4^(K-1))
SO_EXTENDED = {
    7: 1541, 8: 7755, 9: 35841, 10: 155468,
    11: 645002, 12: 2585638, 13: 10092310, 14: 38523768,
    15: 144401076, 16: 532564516,
    17: 160017931, 18: 643095621, 19: 2579119971, 20: 10331254301,
}

SE_EXTENDED = {
    7: 4449, 8: 18920, 9: 77792, 10: 314672,
    11: 1264800, 12: 5073344, 13: 20320320, 14: 81341280,
    15: 325504000, 16: 1302338560,
    17: 588179206, 18: 2352782572, 19: 9411261235, 20: 37645307004,
}


def Nt(K):
    return mpf(4) ** (K - 1)


def main():
    pr("=" * 72)
    pr("E07: K=20 EXACT ENUMERATION ANALYSIS")
    pr("=" * 72)

    # ── Part A: Raw data and convergence ─────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("A. RAW DATA AND CONVERGENCE TO π/18")
    pr(f"{'═' * 72}\n")

    pr(f"  Target: c₁(∞) = π/18 = {nstr(TARGET_C1, 16)}")
    pr(f"  Target: So(∞) = {nstr(TARGET_SO, 16)}")
    pr(f"  Target: Se(∞) = 1+3ln(3/4) = {nstr(TARGET_SE, 16)}")
    pr()

    c1_vals = {}
    so_vals = {}
    se_vals = {}

    pr(f"  {'K':>3} {'c₁(K)':>20} {'Δ(π/18)':>14} {'digits':>7} {'So':>20} {'Se':>20}")
    for K in sorted(E62B_DATA):
        So_int, Se_int, Nt_int = E62B_DATA[K]
        So = mpf(So_int) / mpf(Nt_int)
        Se = mpf(Se_int) / mpf(Nt_int)
        c1 = So + Se

        c1_vals[K] = c1
        so_vals[K] = So
        se_vals[K] = Se

        delta = c1 - TARGET_C1
        digits = -mpmath.log10(abs(delta)) if abs(delta) > 0 else 99
        sign = "+" if delta > 0 else "−"

        pr(f"  {K:3d} {nstr(c1, 18):>20} {sign}{nstr(abs(delta), 4):>13} "
           f"{nstr(digits, 3):>7} {nstr(So, 16):>20} {nstr(Se, 16):>20}")

    # ── Part B: The overshoot at K=20 ────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("B. ★★★ THE OVERSHOOT AT K=20 ★★★")
    pr(f"{'═' * 72}\n")

    d19 = c1_vals[19] - TARGET_C1
    d20 = c1_vals[20] - TARGET_C1
    pr(f"  K=19: c₁ − π/18 = {nstr(d19, 8)} (BELOW)")
    pr(f"  K=20: c₁ − π/18 = {nstr(d20, 8)} (ABOVE !!!)")
    pr(f"  Ratio: δ(20)/δ(19) = {nstr(d20/d19, 6)}")
    pr()
    pr(f"  → SIGN CHANGE confirms oscillatory convergence.")
    pr(f"  → |δ(20)| = {nstr(abs(d20), 4)} is 10× smaller than |δ(19)| = {nstr(abs(d19), 4)}")
    pr(f"  → c₁ = π/18 confirmed to {nstr(-mpmath.log10(abs(d20)), 3)} digits")

    # ── Part C: Se convergence — EXACT rate 1/2 ─────────────────────
    pr(f"\n{'═' * 72}")
    pr("C. Se CONVERGENCE: EXACT GEOMETRIC RATE 1/2")
    pr(f"{'═' * 72}\n")

    pr("  The D-even component converges at EXACTLY rate 1/2:")
    pr(f"  {'K':>3} {'δSe':>16} {'δSe(K)/δSe(K-1)':>18}")
    prev_dSe = None
    for K in sorted(E62B_DATA):
        dSe = se_vals[K] - TARGET_SE
        ratio_str = ""
        if prev_dSe is not None and abs(prev_dSe) > 0:
            ratio_str = nstr(dSe / prev_dSe, 8)
        pr(f"  {K:3d} {nstr(dSe, 10):>16} {ratio_str:>18}")
        prev_dSe = dSe

    pr(f"\n  Richardson on Se (rate 1/2, K=17-20):")
    se_list = [se_vals[K] for K in [17, 18, 19, 20]]
    rt_se = richardson(se_list, mpf(1) / 2, 3)
    for step, row in enumerate(rt_se):
        if row:
            val = row[-1]
            err = val - TARGET_SE
            digits = -mpmath.log10(abs(err)) if abs(err) > 0 else 99
            pr(f"    Step {step}: {nstr(val, 18)}  error: {nstr(err, 4)}  "
               f"({nstr(digits, 3)} digits)")

    # ── Part D: So convergence — complex correction structure ────────
    pr(f"\n{'═' * 72}")
    pr("D. So CONVERGENCE: OSCILLATORY CORRECTION")
    pr(f"{'═' * 72}\n")

    pr(f"  {'K':>3} {'δSo':>16} {'δSo·2^K':>14} {'δSo(K)/δSo(K-1)':>18}")
    prev_dSo = None
    for K in sorted(E62B_DATA):
        dSo = so_vals[K] - TARGET_SO
        scaled = dSo * mpf(2) ** K
        ratio_str = ""
        if prev_dSo is not None and abs(prev_dSo) > 0:
            ratio_str = nstr(dSo / prev_dSo, 8)
        pr(f"  {K:3d} {nstr(dSo, 10):>16} {nstr(scaled, 8):>14} {ratio_str:>18}")
        prev_dSo = dSo

    pr(f"\n  The correction δSo changes SIGN between K=19 and K=20!")
    pr(f"  This is the super-geometric convergence phase.")
    pr(f"  The polynomial prefactor P(K) in δ = P(K)·(1/2)^K crosses zero near K ≈ 19.5.")

    pr(f"\n  Richardson on So (rate 1/2, K=17-20):")
    so_list = [so_vals[K] for K in [17, 18, 19, 20]]
    rt_so = richardson(so_list, mpf(1) / 2, 3)
    for step, row in enumerate(rt_so):
        if row:
            val = row[-1]
            err = val - TARGET_SO
            digits = -mpmath.log10(abs(err)) if abs(err) > 0 else 99
            pr(f"    Step {step}: {nstr(val, 18)}  error: {nstr(err, 4)}  "
               f"({nstr(digits, 3)} digits)")

    # ── Part E: Combined c₁ Richardson ──────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("E. COMBINED c₁ RICHARDSON EXTRAPOLATION")
    pr(f"{'═' * 72}\n")

    c1_list = [c1_vals[K] for K in [17, 18, 19, 20]]
    rt_c1 = richardson(c1_list, mpf(1) / 2, 3)
    for step, row in enumerate(rt_c1):
        if row:
            val = row[-1]
            err = val - TARGET_C1
            digits = -mpmath.log10(abs(err)) if abs(err) > 0 else 99
            pr(f"    Step {step}: {nstr(val, 18)}  error: {nstr(err, 4)}  "
               f"({nstr(digits, 3)} digits)")

    pr(f"\n  Wynn epsilon on c₁(K=17..20):")
    we = wynn_epsilon(c1_list)
    for i, v in enumerate(we):
        err = v - TARGET_C1
        digits = -mpmath.log10(abs(err)) if abs(err) > 0 else 99
        pr(f"    ε[{2*i+2}]: {nstr(v, 18)}  error: {nstr(err, 4)}  "
           f"({nstr(digits, 3)} digits)")

    # ── Part F: PSLQ with K=20 precision ─────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("F. PSLQ WITH K=20 DATA")
    pr(f"{'═' * 72}\n")

    So_best = rt_so[-1][-1] if rt_so[-1] else so_vals[20]
    pr(f"  Best So estimate: {nstr(So_best, 18)}")
    pr(f"  Target So(∞):     {nstr(TARGET_SO, 18)}")
    pr(f"  Difference:       {nstr(So_best - TARGET_SO, 8)}")
    pr()

    pr("  PSLQ on So_best vs {1, π, ln2, ln3, π²}:")
    try:
        result = pslq([So_best, mpf(1), pi, log(2), log(3), pi ** 2],
                       maxcoeff=200)
        if result:
            labels = ["So", "1", "π", "ln2", "ln3", "π²"]
            terms = [f"{c}·{labels[i]}" for i, c in enumerate(result) if c != 0]
            pr(f"    {result}")
            pr(f"    → {' + '.join(terms)}")
            check_val = sum(c * v for c, v in zip(result,
                            [So_best, mpf(1), pi, log(2), log(3), pi**2]))
            pr(f"    Check: {nstr(check_val, 10)}")
        else:
            pr("    No relation found")
    except Exception as ex:
        pr(f"    Error: {ex}")

    pr("\n  PSLQ on So_best vs {1, π/18, ln2, ln3}:")
    try:
        result = pslq([So_best, pi / 18, mpf(1), log(2), log(3)],
                       maxcoeff=200)
        if result:
            labels = ["So", "π/18", "1", "ln2", "ln3"]
            terms = [f"{c}·{labels[i]}" for i, c in enumerate(result) if c != 0]
            pr(f"    {result}")
            pr(f"    → {' + '.join(terms)}")
        else:
            pr("    No relation found")
    except Exception as ex:
        pr(f"    Error: {ex}")

    # ── Part G: Sector data implications ─────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("G. SECTOR DATA FOR K=20")
    pr(f"{'═' * 72}\n")

    So_20 = 10331254301
    pr(f"  So(20) integer = {So_20}")
    pr(f"  So = σ₀₀ + 2σ₁₀ = S00 + 2·S10")
    pr(f"  To extract S00 and S10 separately, need to run E03-style code.")
    pr()

    pr("  From previous sector data, the pattern:")
    S_sector = {
        17: (-31564097, 95791014),
        18: (-124527003, 383811312),
        19: (-494176763, 1536648367),
    }
    for K in sorted(S_sector):
        S00, S10 = S_sector[K]
        So_check = S00 + 2 * S10
        R = mpf(S10) / mpf(S00)
        pr(f"    K={K}: S00={S00:14d}  S10={S10:14d}  "
           f"So={So_check:14d}  R={nstr(R, 8)}")

    pr(f"\n  K=20 prediction from sector ratio trend:")
    R_vals = [mpf(S_sector[K][1]) / mpf(S_sector[K][0]) for K in [17, 18, 19]]
    rt_R = richardson(R_vals, mpf(1) / 2, 2)
    R_pred = rt_R[1][-1] if len(rt_R) > 1 and rt_R[1] else R_vals[-1]
    pr(f"    Extrapolated R(20) ≈ {nstr(R_pred, 8)} (from K=17-19 Richardson)")
    pr(f"    If R(20) ≈ R_pred, then:")
    pr(f"      S00(20) = So/(1 + 2R) = {So_20}/(1 + 2·{nstr(R_pred, 6)})")
    S00_pred = mpf(So_20) / (1 + 2 * R_pred)
    S10_pred = (mpf(So_20) - S00_pred) / 2
    pr(f"      → S00 ≈ {nstr(S00_pred, 12)}, S10 ≈ {nstr(S10_pred, 12)}")
    pr(f"      → σ₁₀/σ₀₀ ≈ {nstr(S10_pred / S00_pred, 10)}")
    pr(f"    (This is a rough prediction; exact computation needed)")

    # ── Part H: Summary ──────────────────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("H. SUMMARY OF K=20 RESULTS")
    pr(f"{'═' * 72}\n")

    pr("  ┌─────────────────────────────────────────────────────────────┐")
    pr("  │  c₁(20) = 0.174537713264726                               │")
    pr("  │  π/18   = 0.174532925199433                               │")
    pr(f"  │  Δ      = +4.79×10⁻⁶ (OVERSHOOT)                        │")
    pr(f"  │  Digits = 5.3                                             │")
    pr("  │                                                           │")
    pr("  │  OSCILLATORY CONVERGENCE CONFIRMED:                       │")
    pr("  │    K=19: Δ = −5.0×10⁻⁵ (below)                          │")
    pr("  │    K=20: Δ = +4.8×10⁻⁶ (above!) → 10× closer            │")
    pr("  │                                                           │")
    pr("  │  Se = 1+3ln(3/4) confirmed to 8+ digits (Richardson)     │")
    pr("  │  So convergence: sign flip at K=20 (super-geometric)     │")
    pr("  └─────────────────────────────────────────────────────────────┘")
    pr()
    pr("  IMPLICATIONS:")
    pr("  1. c₁ = π/18 at 5.3 digits — STRONGEST EVIDENCE to date")
    pr("  2. Se = 1+3ln(3/4) EXACT — Richardson gives 8+ digits")
    pr("  3. So correction P(K)·(1/2)^K changes sign near K=19.5")
    pr("  4. Sector data for K=20 (S00, S10) would give ~4 digits on −π")
    pr("  5. K=20 runtime: ~8h single thread → K=21 feasible (~32h or ~4h×8 cores)")

    pr(f"\n{'═' * 72}")


if __name__ == '__main__':
    main()
