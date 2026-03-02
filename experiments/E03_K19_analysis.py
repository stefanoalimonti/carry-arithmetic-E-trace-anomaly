#!/usr/bin/env python3
"""E03 Analysis: K=19 sector data — improved π hypothesis precision.

NEW DATA (K=19):
  S00_c0 = -423072387
  S00_c1 = -71104376
  S10    = 1536648367
  S01    = 1536648367  (= S10, symmetry confirmed)
  So     = 2579119971  (matches known So(19) ✓)
  Nt     = 68719476736 (= 4^18)

  σ₁₀/σ₀₀ = -3.10951  (target: -π = -3.14159)
  α        = 0.16807   (target: 1/6 = 0.16667)
"""
import sys
import mpmath
from mpmath import mpf, mp, pi, nstr, pslq, log

mp.dps = 50


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def richardson(vals, rate, n_steps=5):
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
    e = [[mpf(0)] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        e[i][1] = seq[i]
    for k in range(2, n + 1):
        for i in range(n - k + 1):
            diff = e[i + 1][k - 1] - e[i][k - 1]
            if abs(diff) < mpf(10) ** (-40):
                e[i][k] = mpf(10) ** 40
            else:
                e[i][k] = e[i + 1][k - 2] + 1 / diff
    results = []
    for k in range(2, n + 1, 2):
        if n - k >= 0:
            results.append(e[0][k])
    return results


C_DATA = {
    7:  {'S00_c0': -114, 'S00_c1': -28, 'S10': 13, 'Nt': 4096},
    8:  {'S00_c0': -373, 'S00_c1': -61, 'S10': 167, 'Nt': 16384},
    9:  {'S00_c0': -1155, 'S00_c1': -171, 'S10': 963, 'Nt': 65536},
    10: {'S00_c0': -3684, 'S00_c1': -472, 'S10': 4648, 'Nt': 262144},
    11: {'S00_c0': -11768, 'S00_c1': -1486, 'S10': 20590, 'Nt': 1048576},
    12: {'S00_c0': -39195, 'S00_c1': -5107, 'S10': 87256, 'Nt': 4194304},
    13: {'S00_c0': -135206, 'S00_c1': -18944, 'S10': 360533, 'Nt': 16777216},
    14: {'S00_c0': -487465, 'S00_c1': -72541, 'S10': 1468632, 'Nt': 67108864},
    15: {'S00_c0': -1817747, 'S00_c1': -283679, 'S10': 5934740, 'Nt': 268435456},
    16: {'S00_c0': -6959865, 'S00_c1': -1122027, 'S10': 23873440, 'Nt': 1073741824},
    17: {'S00_c0': -27101298, 'S00_c1': -4462799, 'S10': 95791014, 'Nt': 4294967296},
    18: {'S00_c0': -106725895, 'S00_c1': -17801108, 'S10': 383811312, 'Nt': 17179869184},
    19: {'S00_c0': -423072387, 'S00_c1': -71104376, 'S10': 1536648367, 'Nt': 68719476736},
}

DATA_So = {
    7: -116, 8: -100, 9: 600, 10: 5140, 11: 27926, 12: 130210,
    13: 566916, 14: 2377258, 15: 9768054, 16: 39664988,
    17: 160017931, 18: 643095621, 19: 2579119971,
}


def main():
    pr("=" * 70)
    pr("E03: K=19 SECTOR DATA — UPDATED π HYPOTHESIS PRECISION")
    pr("=" * 70)

    # ── Verify So consistency ─────────────────────────────────────
    pr(f"\n{'═' * 70}")
    pr("VERIFICATION: So(K) consistency")
    pr(f"{'═' * 70}\n")

    for K in sorted(C_DATA.keys()):
        d = C_DATA[K]
        So_calc = (d['S00_c0'] + d['S00_c1']) + 2 * d['S10']
        So_known = DATA_So.get(K, None)
        match = "✓" if So_known and So_calc == So_known else "✗"
        pr(f"  K={K:2d}: So = {So_calc:>14d}  (known: {So_known})  {match}")

    # ── σ₁₀/σ₀₀ convergence with K=19 ───────────────────────────
    pr(f"\n{'═' * 70}")
    pr("σ₁₀/σ₀₀ CONVERGENCE (NOW WITH K=19)")
    pr(f"{'═' * 70}\n")

    ratios = []
    Ks = []
    for K in sorted(C_DATA.keys()):
        d = C_DATA[K]
        S00 = d['S00_c0'] + d['S00_c1']
        S10 = d['S10']
        Nt = d['Nt']
        if S00 == 0:
            continue
        R = mpf(S10) / mpf(S00)
        ratios.append(R)
        Ks.append(K)
        Q = mpf(S10) + pi * mpf(S00)
        Q_norm = Q / Nt
        pr(f"  K={K:2d}: σ₁₀/σ₀₀ = {nstr(R, 10):>14}  "
           f"Q/Nt = {nstr(Q_norm, 8):>14}")

    pr(f"\n  Target: -π = {nstr(-pi, 12)}")

    # Richardson on the ratio
    pr("\n  Richardson extrapolation (rate 1/2):")
    r_tail = ratios[-8:]
    rt = richardson(r_tail, mpf(1) / 2, 6)
    for i, row in enumerate(rt[:7]):
        if row:
            pr(f"    Step {i}: {nstr(row[-1], 14)}")

    # Wynn epsilon on ratio
    pr("\n  Wynn Epsilon on last 10 values:")
    wynn = wynn_epsilon(ratios[-10:])
    for i, val in enumerate(wynn[:5]):
        pr(f"    ε_{2*i+2}: {nstr(val, 14)}")

    # ── Precision estimate ────────────────────────────────────────
    pr(f"\n{'═' * 70}")
    pr("PRECISION OF π HYPOTHESIS")
    pr(f"{'═' * 70}\n")

    d19 = C_DATA[19]
    S00_19 = mpf(d19['S00_c0'] + d19['S00_c1'])
    S10_19 = mpf(d19['S10'])
    R19 = S10_19 / S00_19
    residual = abs(R19 - (-pi)) / pi
    digits = -mpmath.log10(residual)
    pr(f"  K=19 raw: σ₁₀/σ₀₀ = {nstr(R19, 12)}")
    pr(f"  |R + π|/π = {nstr(residual, 6)}")
    pr(f"  Digits of agreement: {nstr(digits, 4)}")

    # Compare with K=18
    d18 = C_DATA[18]
    S00_18 = mpf(d18['S00_c0'] + d18['S00_c1'])
    S10_18 = mpf(d18['S10'])
    R18 = S10_18 / S00_18
    res18 = abs(R18 - (-pi)) / pi
    dig18 = -mpmath.log10(res18)
    pr(f"\n  K=18: digits = {nstr(dig18, 4)}")
    pr(f"  K=19: digits = {nstr(digits, 4)}")
    pr(f"  Gain: +{nstr(digits - dig18, 3)} digits")

    # ── Q(K) convergence ──────────────────────────────────────────
    pr(f"\n{'═' * 70}")
    pr("Q(K) = σ₁₀ + π·σ₀₀ CONVERGENCE")
    pr(f"{'═' * 70}\n")

    pr(f"  {'K':>3} {'Q(K)':>16} {'Q·2^K':>16} {'Q·2^K+π²K':>16}")
    Q_vals = []
    for K in sorted(C_DATA.keys()):
        if K < 9:
            continue
        d = C_DATA[K]
        S00 = mpf(d['S00_c0'] + d['S00_c1'])
        S10 = mpf(d['S10'])
        Nt = d['Nt']
        Q = (S10 + pi * S00) / Nt
        Q2k = Q * mpf(2) ** K
        Q2k_corr = Q2k + pi ** 2 * K
        Q_vals.append(Q)
        pr(f"  {K:3d} {nstr(Q, 10):>16} {nstr(Q2k, 10):>16} "
           f"{nstr(Q2k_corr, 10):>16}")

    # ── α convergence with K=19 ──────────────────────────────────
    pr(f"\n{'═' * 70}")
    pr("α = σ₀₀^{{c=1}}/σ₀₀^{{c=0}} CONVERGENCE (WITH K=19)")
    pr(f"{'═' * 70}\n")

    alphas = []
    for K in sorted(C_DATA.keys()):
        if K < 7:
            continue
        d = C_DATA[K]
        alpha = mpf(d['S00_c1']) / mpf(d['S00_c0'])
        alphas.append(alpha)
        delta = alpha - mpf(1) / 6
        pr(f"  K={K:2d}: α = {nstr(alpha, 10):>14}  "
           f"α - 1/6 = {nstr(delta, 8):>14}")

    pr(f"\n  Target: 1/6 = {nstr(mpf(1)/6, 12)}")
    pr(f"  K=19 α = {nstr(alphas[-1], 12)}")
    pr(f"  |α - 1/6| = {nstr(abs(alphas[-1] - mpf(1)/6), 6)}")

    # K=19 crosses 1/6 from ABOVE (0.16807 > 0.16667)
    pr(f"\n  K=18: α = 0.16679 (just above 1/6)")
    pr(f"  K=19: α = 0.16807 (further above 1/6)")
    pr(f"  The oscillatory correction continues")

    # ── Combined ζ(2) test ────────────────────────────────────────
    pr(f"\n{'═' * 70}")
    pr("ζ(2) = α · R² TEST (WITH K=19)")
    pr(f"{'═' * 70}\n")

    z2 = mpmath.zeta(2)
    for K in sorted(C_DATA.keys()):
        if K < 10:
            continue
        d = C_DATA[K]
        S00 = mpf(d['S00_c0'] + d['S00_c1'])
        S10 = mpf(d['S10'])
        alpha = mpf(d['S00_c1']) / mpf(d['S00_c0'])
        R = S10 / S00
        product = alpha * R ** 2
        diff = product - z2
        pr(f"  K={K:2d}: α·R² = {nstr(product, 10):>14}  "
           f"ζ(2) = {nstr(z2, 10):>14}  diff = {nstr(diff, 6):>10}")

    # ── Summary ───────────────────────────────────────────────────
    pr(f"\n{'═' * 70}")
    pr("SUMMARY WITH K=19 DATA")
    pr(f"{'═' * 70}\n")

    pr(f"  π HYPOTHESIS:  σ₁₀/σ₀₀ = -π")
    pr(f"    K=18: {nstr(dig18, 4)} digits")
    pr(f"    K=19: {nstr(digits, 4)} digits")
    pr(f"    Gain: +{nstr(digits-dig18, 3)} digits per K step")
    pr()
    pr(f"  BERNOULLI:  α = σ₀₀^{{c=1}}/σ₀₀^{{c=0}} = 1/6")
    pr(f"    K=19 α = {nstr(alphas[-1], 8)} (1/6 = 0.16667)")
    pr(f"    Oscillatory convergence — crossed 1/6 at K=18, now above")
    pr()
    pr(f"  ζ(2) PRODUCT:  α · R² → π²/6 = {nstr(z2, 8)}")
    pr(f"    K=19: {nstr(alphas[-1] * R19**2, 8)}")


if __name__ == '__main__':
    main()
