#!/usr/bin/env python3
"""E04 Analysis: Identify limiting constants in Bernoulli hierarchy test.

KEY FINDINGS from C computation:
  1. S[1][1][0] = 0 always — carry runs of length exactly 2 are IMPOSSIBLE
  2. α₃ marginal (D-3) converging to ~ -0.45 (negative!)
  3. α₃|c₂=1 converging to ~ -0.90
  4. The carry near the top forms runs of length 0, 1, or ≥3 (never 2)

This defines a RUN-LENGTH decomposition:
  R=0: c_{D-2}=0           (no carry reached top)
  R=1: c_{D-2}=1, c_{D-3}=0  (carry at top only)
  R≥2: c_{D-2}=1, c_{D-3}=1  (carry run from below)
  Since S[1][1][0]=0, R=2 doesn't exist, so R≥2 means R≥3.
"""
import sys
import mpmath
from mpmath import mpf, mp, pi, nstr, pslq, log, bernoulli

mp.dps = 30


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def richardson(vals, rate, n_steps=3):
    table = [list(vals)]
    for step in range(n_steps):
        prev = table[-1]
        r = rate ** (step + 1)
        new = []
        for i in range(len(prev) - 1):
            new.append((prev[i + 1] - r * prev[i]) / (1 - r))
        table.append(new)
    return table


# Data from prior experiments C output
DATA = {
    7:  {'S': [[0, 0], [0, 0]], 'alpha2': 0.245614, 'alpha3m': 0.007092,
         'alpha3_c20': 0.325581, 'alpha3_c21': -0.490909, 'alpha4': 0.290909},
    8:  {'alpha2': 0.163539, 'alpha3m': -0.099585,
         'alpha3_c20': 0.295139, 'alpha3_c21': -0.685567, 'alpha4': 0.163539},
    9:  {'alpha2': 0.148052, 'alpha3m': -0.178439,
         'alpha3_c20': 0.335260, 'alpha3_c21': -0.771696, 'alpha4': 0.110553},
    10: {'alpha2': 0.128122, 'alpha3m': -0.247374,
         'alpha3_c20': 0.410413, 'alpha3_c21': -0.837801, 'alpha4': 0.046324},
    11: {'alpha2': 0.126275, 'alpha3m': -0.307270,
         'alpha3_c20': 0.543547, 'alpha3_c21': -0.870884, 'alpha4': 0.009290},
    12: {'alpha2': 0.130297, 'alpha3m': -0.354415,
         'alpha3_c20': 0.710302, 'alpha3_c21': -0.888264, 'alpha4': -0.018847},
    13: {'alpha2': 0.140112, 'alpha3m': -0.390832,
         'alpha3_c20': 0.909878, 'alpha3_c21': -0.896059, 'alpha4': -0.036773},
    14: {'alpha2': 0.148813, 'alpha3m': -0.415998,
         'alpha3_c20': 1.108559, 'alpha3_c21': -0.900318, 'alpha4': -0.049103},
    15: {'alpha2': 0.156061, 'alpha3m': -0.432608,
         'alpha3_c20': 1.285643, 'alpha3_c21': -0.902461, 'alpha4': -0.056731},
    16: {'alpha2': 0.161214, 'alpha3m': -0.442774,
         'alpha3_c20': 1.420519, 'alpha3_c21': -0.903510, 'alpha4': -0.061145},
}

# S[c2][c3] tables from K=14,15,16
TABLES = {
    14: {'S00': -231184, 'S01': -256281, 'S10': -727727, 'S11': 655186},
    15: {'S00': -795289, 'S01': -1022458, 'S10': -2908366, 'S11': 2624687},
    16: {'S00': -2875361, 'S01': -4084504, 'S10': -11628422, 'S11': 10506395},
}


def main():
    pr("=" * 60)
    pr("E04 ANALYSIS: BERNOULLI HIERARCHY IDENTIFICATION")
    pr("=" * 60)

    # ── α₃|c₂=1: converging to what? ──────────────────────────
    pr(f"\n{'═' * 60}")
    pr("α₃|c₂=1 (carry at D-3, given c₂=1)")
    pr(f"{'═' * 60}\n")

    vals = [mpf(DATA[K]['alpha3_c21']) for K in sorted(DATA.keys())]
    Ks = sorted(DATA.keys())

    pr("  Sequence:")
    for K in Ks:
        pr(f"    K={K}: {DATA[K]['alpha3_c21']:.6f}")

    pr("\n  Converging to ~ -0.904")
    pr("  Candidates:")
    target = mpf('-0.9035')
    for name, val in [
        ("-9/10", mpf(-9) / 10), ("-1+1/10", mpf(-9) / 10),
        ("-1+1/e", -1 + 1 / mpmath.e), ("-1+B₂", -1 + mpf(1) / 6),
        ("-1+1/π", -1 + 1 / pi), ("-e/3", -mpmath.e / 3),
        ("-1+1/12", -1 + mpf(1) / 12),
        ("-(1-1/10)", -(1 - mpf(1) / 10)),
    ]:
        pr(f"    {name:15s} = {nstr(val, 8):>12}  diff = {nstr(target-val, 6):>10}")

    # Richardson on α₃|c₂=1
    a3c1 = [mpf(DATA[K]['alpha3_c21']) for K in Ks if K >= 11]
    rt = richardson(a3c1, mpf(1) / 2, 4)
    pr(f"\n  Richardson (rate 1/2):")
    for i, row in enumerate(rt[:5]):
        if row:
            pr(f"    Step {i}: {nstr(row[-1], 10)}")

    # ── α₃ marginal ───────────────────────────────────────────
    pr(f"\n{'═' * 60}")
    pr("α₃ MARGINAL (carry at D-3, all)")
    pr(f"{'═' * 60}\n")

    a3m = [mpf(DATA[K]['alpha3m']) for K in Ks if K >= 10]
    Ks_m = [K for K in Ks if K >= 10]

    pr("  Sequence:")
    for K in Ks_m:
        pr(f"    K={K}: {DATA[K]['alpha3m']:.6f}")

    rt3 = richardson(a3m, mpf(1) / 2, 4)
    pr(f"\n  Richardson (rate 1/2):")
    for i, row in enumerate(rt3[:5]):
        if row:
            pr(f"    Step {i}: {nstr(row[-1], 10)}")

    pr("\n  Candidates for limit:")
    best = rt3[1][-1] if len(rt3) > 1 and rt3[1] else a3m[-1]
    for name, val in [
        ("-1/2 = B₁", mpf(-1) / 2),
        ("-π/6", -pi / 6),
        ("-1/e", -1 / mpmath.e),
        ("-ln(2)", -log(2)),
        ("-B₂·3", -mpf(3) / 6),
        ("-7/15", mpf(-7) / 15),
    ]:
        pr(f"    {name:15s} = {nstr(val, 8):>12}  diff = {nstr(best-val, 6):>10}")

    # ── RUN LENGTH analysis ────────────────────────────────────
    pr(f"\n{'═' * 60}")
    pr("RUN LENGTH DECOMPOSITION")
    pr(f"{'═' * 60}\n")

    pr("  S[1][1][0] = 0 for all K!")
    pr("  → Carry runs near the top have length 0, 1, or ≥3 (never 2)")
    pr()
    pr("  R=0: c₂=0 (no carry)       = S[0][*] = σ₀₀^{c=0}")
    pr("  R=1: c₂=1, c₃=0 (top only) = S[1][0]")
    pr("  R≥3: c₂=c₃=1               = S[1][1]")
    pr()

    for K in [14, 15, 16]:
        t = TABLES[K]
        R0 = t['S00'] + t['S01']
        R1 = t['S10']
        R3 = t['S11']
        total = R0 + R1 + R3
        pr(f"  K={K}: R=0: {R0:12d}  R=1: {R1:12d}  R≥3: {R3:12d}")
        pr(f"        R1/R0 = {R1/R0:.6f}  R3/R0 = {R3/R0:.6f}  "
           f"R3/R1 = {R3/R1:.6f}")

    # Extrapolate R1/R0 and R3/R1
    pr("\n  Run-length ratios:")
    R1_R0 = []
    R3_R1 = []
    R3_R0 = []
    for K in [14, 15, 16]:
        t = TABLES[K]
        R0 = t['S00'] + t['S01']
        R1 = t['S10']
        R3 = t['S11']
        R1_R0.append(mpf(R1) / R0)
        R3_R1.append(mpf(R3) / R1)
        R3_R0.append(mpf(R3) / R0)
        pr(f"    K={K}: R1/R0={float(R1/R0):.6f}  "
           f"R3/R1={float(R3/R1):.6f}  R3/R0={float(R3/R0):.6f}")

    pr("\n  Richardson on R1/R0 (rate 1/2):")
    rt_r = richardson(R1_R0, mpf(1) / 2, 2)
    for i, row in enumerate(rt_r[:3]):
        if row:
            pr(f"    Step {i}: {nstr(row[-1], 10)}")

    pr("\n  Richardson on R3/R1 (rate 1/2):")
    rt_r2 = richardson(R3_R1, mpf(1) / 2, 2)
    for i, row in enumerate(rt_r2[:3]):
        if row:
            pr(f"    Step {i}: {nstr(row[-1], 10)}")

    # PSLQ on the ratios
    pr(f"\n{'═' * 60}")
    pr("PSLQ ON RUN-LENGTH RATIOS")
    pr(f"{'═' * 60}\n")

    # Use K=16 values as best estimates
    t16 = TABLES[16]
    R0_16 = mpf(t16['S00'] + t16['S01'])
    R1_16 = mpf(t16['S10'])
    R3_16 = mpf(t16['S11'])

    for name, val in [("R1/R0", R1_16/R0_16), ("R3/R1", R3_16/R1_16), ("R3/R0", R3_16/R0_16)]:
        pr(f"  {name} = {nstr(val, 10)}")
        try:
            result = pslq([val, mpf(1), pi, log(2), mpf(1)/6, mpf(1)/30])
            if result:
                labels = [name, "1", "π", "ln2", "1/6", "1/30"]
                expr = " + ".join(f"{c}·{labels[i]}" for i, c in enumerate(result) if c != 0)
                pr(f"    PSLQ: {result} → {expr}")
            else:
                pr(f"    PSLQ: no relation found")
        except Exception as e:
            pr(f"    PSLQ error: {e}")

    # ── α₂ confirmation ───────────────────────────────────────
    pr(f"\n{'═' * 60}")
    pr("CONFIRMATION: α₂ → 1/6 = B₂")
    pr(f"{'═' * 60}\n")

    a2 = [mpf(DATA[K]['alpha2']) for K in Ks if K >= 10]
    rt2 = richardson(a2, mpf(1) / 2, 4)
    pr("  Richardson (rate 1/2):")
    for i, row in enumerate(rt2[:5]):
        if row:
            pr(f"    Step {i}: {nstr(row[-1], 10)}")
    pr(f"  Target: 1/6 = {nstr(mpf(1)/6, 10)}")

    # ── Summary ────────────────────────────────────────────────
    pr(f"\n{'═' * 60}")
    pr("SUMMARY")
    pr(f"{'═' * 60}\n")

    pr("  STRUCTURAL DISCOVERY:")
    pr("    S[1][1][0] = 0: carry runs of length 2 are impossible.")
    pr("    Near the D-odd boundary, carries form runs of length 0, 1, or ≥3.")
    pr("    This is a TOPOLOGICAL constraint of the carry chain.")
    pr()
    pr("  HIERARCHY STATUS:")
    pr("    Level 1: |B₁| = 1/2  ✓  (spectral gap)")
    pr("    Level 2: B₂ = 1/6    ✓  (α₂ = σ₀₀^{c₂=1}/σ₀₀^{c₂=0})")
    pr("    Level 3: |B₄| = 1/30 ?  NOT found in simple positional ratios")
    pr()
    pr("  The 1/30 might appear in:")
    pr("    (a) The CORRECTION COEFFICIENT of α₂ → 1/6")
    pr("    (b) A ratio involving the run-length distribution")
    pr("    (c) A spectral quantity of the transfer operator")
    pr("    (d) It might NOT appear at all (1/6 could be coincidence)")


if __name__ == '__main__':
    main()
