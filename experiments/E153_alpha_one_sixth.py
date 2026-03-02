#!/usr/bin/env python3
"""E153: The α = 1/6 Hypothesis and the Analytical Proof of σ_10/σ_00 = -π.

DISCOVERY (from E152):
  α(K) = σ_00^{c=1}(K) / σ_00^{c=0}(K) → 0.16679 at K=18
  This is tantalizingly close to 1/6 = 0.16667.

IF α → 1/6, then:
  σ_00 = σ_00^{c=0} · (1 + 1/6) = 7/6 · σ_00^{c=0}
  σ_10/σ_00 = β/(1+α) where β = σ_10/σ_00^{c=0}

  For σ_10/σ_00 = -π: need β = -7π/6

  This decomposes the π hypothesis into TWO independent statements:
    (I)  α = σ_00^{c=1}/σ_00^{c=0} → 1/6
    (II) β = σ_10/σ_00^{c=0} → -7π/6

PROOF STRATEGY:
  Statement (I) relates the c=0 and c=1 channels of the (0,0) sector.
  The c=1 channel has pairs with c_{D-2} = 1 (rescued D-even pairs).
  The ratio 1/6 should come from the top-level carry structure:
    P(c_{D-2}=1 | (0,0)) weighted by carry weight.

  Statement (II) relates the (1,0) sector to the (0,0,c=0) channel.
  These differ by one bit flip + boundary condition.
  The factor 7π/6 should come from the spectral structure.

Parts:
  A — Test α → 1/6 with Richardson extrapolation
  B — Test β → -7π/6 with Richardson extrapolation
  C — Carry weight decomposition of the c=1 channel
  D — The per-pair weight analysis
  E — Testing alternative limits for α
  F — Physical meaning: why 1/6?
"""
import sys
import time

import mpmath
from mpmath import mpf, mp, log, pi, pslq, sqrt

mp.dps = 50


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


C_DATA = {
    7:  {'S00_c0': -114, 'S00_c1': -28, 'S10': 13,
         'n00_c0': 306, 'n00_c1': 504, 'n10': 253, 'Nt': 4096},
    8:  {'S00_c0': -373, 'S00_c1': -61, 'S10': 167,
         'n00_c0': 1450, 'n00_c1': 2049, 'n10': 1109, 'Nt': 16384},
    9:  {'S00_c0': -1155, 'S00_c1': -171, 'S10': 963,
         'n00_c0': 6315, 'n00_c1': 8263, 'n10': 4660, 'Nt': 65536},
    10: {'S00_c0': -3684, 'S00_c1': -472, 'S10': 4648,
         'n00_c0': 26689, 'n00_c1': 33219, 'n10': 19173, 'Nt': 262144},
    11: {'S00_c0': -11768, 'S00_c1': -1486, 'S10': 20590,
         'n00_c0': 109744, 'n00_c1': 133210, 'n10': 77815, 'Nt': 1048576},
    12: {'S00_c0': -39195, 'S00_c1': -5107, 'S10': 87256,
         'n00_c0': 446078, 'n00_c1': 533536, 'n10': 313641, 'Nt': 4194304},
    13: {'S00_c0': -135206, 'S00_c1': -18944, 'S10': 360533,
         'n00_c0': 1798934, 'n00_c1': 2135512, 'n10': 1259418, 'Nt': 16777216},
    14: {'S00_c0': -487465, 'S00_c1': -72541, 'S10': 1468632,
         'n00_c0': 7228083, 'n00_c1': 8544782, 'n10': 5047708, 'Nt': 67108864},
    15: {'S00_c0': -1817747, 'S00_c1': -283679, 'S10': 5934740,
         'n00_c0': 28976492, 'n00_c1': 34184758, 'n10': 20210714, 'Nt': 268435456},
    16: {'S00_c0': -6959865, 'S00_c1': -1122027, 'S10': 23873440,
         'n00_c0': 116043286, 'n00_c1': 136749864, 'n10': 80883209, 'Nt': 1073741824},
    17: {'S00_c0': -27101298, 'S00_c1': -4462799, 'S10': 95791014,
         'n00_c0': 464437202, 'n00_c1': 547021297, 'n10': 323610813, 'Nt': 4294967296},
    18: {'S00_c0': -106725895, 'S00_c1': -17801108, 'S10': 383811312,
         'n00_c0': 1858297109, 'n00_c1': 2188128902, 'n10': 1294598851, 'Nt': 17179869184},
}

DATA_So = {
    3: -1, 4: -6, 5: -18, 6: -51, 7: -116, 8: -100,
    9: 600, 10: 5140, 11: 27926, 12: 130210, 13: 566916,
    14: 2377258, 15: 9768054, 16: 39664988, 17: 160017931,
    18: 643095621, 19: 2579119971,
}


def richardson_extrap(K_vals, f_vals, rate=mpf(1)/2, n_steps=3):
    """Richardson extrapolation with given rate.
    Assumes f(K) = f_∞ + C₁·rate^K + C₂·rate^{2K} + ...
    """
    table = [list(f_vals)]
    for step in range(n_steps):
        prev = table[-1]
        r = rate ** (step + 1)
        new = []
        for i in range(len(prev) - 1):
            new.append((prev[i + 1] - r * prev[i]) / (1 - r))
        table.append(new)
    return table


def wynn_epsilon(seq):
    """Wynn's epsilon algorithm for sequence acceleration."""
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


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("E153: THE α = 1/6 HYPOTHESIS AND PROOF OF σ_10/σ_00 = -π")
    pr("=" * 72)

    # ── Part A: Test α → 1/6 ──────────────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("PART A: DOES α = σ_00^{{c=1}}/σ_00^{{c=0}} → 1/6?")
    pr(f"{'═' * 72}\n")

    alpha_vals = {}
    pr(f"  {'K':>3} {'α(K)':>14} {'α - 1/6':>14} {'(α-1/6)·2^K':>14} "
       f"{'(α-1/6)·4^K':>14}")
    for K in sorted(C_DATA.keys()):
        d = C_DATA[K]
        alpha = mpf(d['S00_c1']) / mpf(d['S00_c0'])
        delta = alpha - mpf(1) / 6
        delta_2k = delta * mpf(2) ** K
        delta_4k = delta * mpf(4) ** K
        alpha_vals[K] = alpha
        pr(f"  {K:3d} {float(alpha):14.10f} {float(delta):+14.10e} "
           f"{float(delta_2k):+14.6f} {float(delta_4k):+14.1f}")

    # Richardson extrapolation of α
    pr("\n  Richardson extrapolation of α(K):")
    K_range = list(range(12, 19))
    alpha_seq = [alpha_vals[K] for K in K_range]

    for rate_name, rate in [("1/2", mpf(1)/2), ("-1/2", mpf(-1)/2),
                            ("1/4", mpf(1)/4)]:
        table = richardson_extrap(K_range, alpha_seq, rate=rate, n_steps=4)
        pr(f"\n    Rate = {rate_name}:")
        for step in range(min(5, len(table))):
            vals = table[step]
            if vals:
                pr(f"      Step {step}: {' '.join(f'{float(v):.10f}' for v in vals[-3:])}")
        last = table[-1]
        if last:
            pr(f"      → Extrapolated: {float(last[-1]):.12f}")
            pr(f"        1/6 = {float(mpf(1)/6):.12f}")
            pr(f"        Δ = {float(last[-1] - mpf(1)/6):.6e}")

    # Wynn epsilon on α
    pr("\n  Wynn epsilon on α(K) for K=10..18:")
    alpha_seq_wynn = [alpha_vals[K] for K in range(10, 19)]
    eps = wynn_epsilon(alpha_seq_wynn)
    for i, e in enumerate(eps):
        pr(f"    ε_{2*(i+1)} = {float(e):.12f}, Δ from 1/6 = {float(e - mpf(1)/6):.6e}")

    # ── Part B: Test β → -7π/6 ────────────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("PART B: DOES β = σ_10/σ_00^{{c=0}} → -7π/6?")
    pr(f"{'═' * 72}\n")

    target_beta = -7 * pi / 6
    pr(f"  Target: -7π/6 = {float(target_beta):.12f}")
    pr()

    beta_vals = {}
    pr(f"  {'K':>3} {'β(K)':>14} {'β + 7π/6':>14} {'(β+7π/6)·2^K':>14} "
       f"{'Δβ':>12}")
    prev_beta = None
    for K in sorted(C_DATA.keys()):
        d = C_DATA[K]
        beta = mpf(d['S10']) / mpf(d['S00_c0'])
        delta_b = beta - target_beta
        delta_b_2k = delta_b * mpf(2) ** K
        beta_vals[K] = beta
        db = beta - prev_beta if prev_beta is not None else mpf(0)
        pr(f"  {K:3d} {float(beta):14.8f} {float(delta_b):+14.8e} "
           f"{float(delta_b_2k):+14.4f} {float(db):+12.6e}")
        prev_beta = beta

    pr("\n  Richardson extrapolation of β(K):")
    K_range_b = list(range(12, 19))
    beta_seq = [beta_vals[K] for K in K_range_b]

    for rate_name, rate in [("1/2", mpf(1)/2), ("-1/2", mpf(-1)/2),
                            ("1/4", mpf(1)/4)]:
        table = richardson_extrap(K_range_b, beta_seq, rate=rate, n_steps=4)
        pr(f"\n    Rate = {rate_name}:")
        for step in range(min(5, len(table))):
            vals = table[step]
            if vals:
                pr(f"      Step {step}: {' '.join(f'{float(v):.10f}' for v in vals[-3:])}")
        last = table[-1]
        if last:
            pr(f"      → Extrapolated: {float(last[-1]):.12f}")
            pr(f"        -7π/6 = {float(target_beta):.12f}")
            pr(f"        Δ = {float(last[-1] - target_beta):.6e}")

    # Wynn epsilon on β
    pr("\n  Wynn epsilon on β(K) for K=10..18:")
    beta_seq_wynn = [beta_vals[K] for K in range(10, 19)]
    eps_b = wynn_epsilon(beta_seq_wynn)
    for i, e in enumerate(eps_b):
        pr(f"    ε_{2*(i+1)} = {float(e):.12f}, "
           f"Δ from -7π/6 = {float(e - target_beta):.6e}")

    # ── Part C: PSLQ on α and β limits ─────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("PART C: PSLQ ANALYSIS OF α AND β LIMITS")
    pr(f"{'═' * 72}\n")

    # Use best Wynn estimates
    if eps:
        alpha_est = eps[-1]
    else:
        alpha_est = alpha_vals[18]

    if eps_b:
        beta_est = eps_b[-1]
    else:
        beta_est = beta_vals[18]

    pr(f"  Best α estimate: {float(alpha_est):.14f}")
    pr(f"  Best β estimate: {float(beta_est):.14f}")
    pr()

    # PSLQ on α
    pr("  PSLQ on α with basis [1, α, π, ln2, ln3]:")
    res_a = pslq([1, alpha_est, pi, log(2), log(3)], maxcoeff=200)
    if res_a:
        pr(f"    Result: {res_a}")
        a0, a1, a2, a3, a4 = res_a
        pr(f"    Meaning: {a0} + {a1}·α + {a2}·π + {a3}·ln2 + {a4}·ln3 = 0")
        if a1 != 0:
            val = -(a0 + a2 * float(pi) + a3 * float(log(2)) + a4 * float(log(3))) / a1
            pr(f"    → α = {val:.12f}")
    else:
        pr("    No PSLQ relation found with maxcoeff=200")

    # PSLQ on α with rational basis
    pr("\n  PSLQ on α with basis [1, α] (testing rational limit):")
    res_rat = pslq([1, alpha_est], maxcoeff=1000)
    if res_rat:
        pr(f"    Result: {res_rat}")
        if res_rat[1] != 0:
            pr(f"    → α = {-res_rat[0]}/{res_rat[1]}")

    # PSLQ on β
    pr("\n  PSLQ on β with basis [1, β, π, ln2, ln3]:")
    res_b = pslq([1, beta_est, pi, log(2), log(3)], maxcoeff=200)
    if res_b:
        pr(f"    Result: {res_b}")
        b0, b1, b2, b3, b4 = res_b
        pr(f"    Meaning: {b0} + {b1}·β + {b2}·π + {b3}·ln2 + {b4}·ln3 = 0")
        if b1 != 0:
            val = -(b0 + b2 * float(pi) + b3 * float(log(2)) + b4 * float(log(3))) / b1
            pr(f"    → β = {val:.12f}")
    else:
        pr("    No PSLQ relation found with maxcoeff=200")

    # ── Part D: Alternative hypotheses for α ───────────────────────────
    pr(f"\n{'═' * 72}")
    pr("PART D: TESTING ALTERNATIVE HYPOTHESES FOR α")
    pr(f"{'═' * 72}\n")

    candidates = [
        ("1/6", mpf(1) / 6),
        ("(π-3)/1", pi - 3),
        ("1/(2π)", 1 / (2 * pi)),
        ("ln(9/8)/ln(4/3)", log(mpf(9)/8) / log(mpf(4)/3)),
        ("A_10/A_00", (4*log(2)-2*log(3)-mpf(1)/2) / (2*log(mpf(9)/8))),
        ("1/(2π-1)", 1 / (2 * pi - 1)),
        ("(9-2π)/(9·2π-9)", (9 - 2*pi) / (9*2*pi - 9)),
        ("ln2 - 1/2", log(2) - mpf(1)/2),
    ]

    pr(f"  {'Candidate':>25} {'Value':>14} {'α_18-cand':>14} "
       f"{'|Δ|·2^18':>14}")
    for name, val in candidates:
        diff = alpha_vals[18] - val
        diff_2k = abs(diff) * mpf(2) ** 18
        pr(f"  {name:>25} {float(val):14.10f} {float(diff):+14.10e} "
           f"{float(diff_2k):14.4f}")

    # Test convergence rate for each candidate
    pr(f"\n  Convergence rate test (looking for geometric Δ·2^K):")
    pr()
    for name, target_val in [("1/6", mpf(1)/6),
                              ("A_10/A_00", (4*log(2)-2*log(3)-mpf(1)/2)/(2*log(mpf(9)/8)))]:
        pr(f"  Target = {name}:")
        pr(f"    {'K':>3} {'Δ(K)':>14} {'Δ·2^K':>14} {'Δ·4^K':>14} "
           f"{'Δ·2^K ratio':>14}")
        prev = None
        for K in sorted(C_DATA.keys()):
            delta = alpha_vals[K] - target_val
            d2k = delta * mpf(2) ** K
            d4k = delta * mpf(4) ** K
            ratio_str = ""
            if prev is not None and abs(prev) > 1e-20:
                r = d2k / prev
                ratio_str = f"{float(r):14.6f}"
            pr(f"    {K:3d} {float(delta):+14.10e} {float(d2k):+14.6f} "
               f"{float(d4k):+14.2f} {ratio_str}")
            prev = d2k
        pr()

    # ── Part E: Per-pair weight structure ──────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("PART E: PER-PAIR WEIGHT STRUCTURE")
    pr(f"{'═' * 72}\n")

    A_00 = 2 * log(mpf(9) / 8)
    A_10 = 4 * log(2) - 2 * log(3) - mpf(1) / 2

    pr("  The per-pair weights h̄_00^{c=0}, h̄_00^{c=1}, h̄_10:")
    pr()
    pr(f"  {'K':>3} {'h̄_00c0':>12} {'h̄_00c1':>12} {'h̄_10':>12} "
       f"{'h̄_10/h̄_00c0':>14} {'h̄_00c1/h̄_00c0':>14}")
    for K in sorted(C_DATA.keys()):
        d = C_DATA[K]
        Nt = mpf(d['Nt'])
        sc0 = mpf(d['S00_c0']) / Nt
        sc1 = mpf(d['S00_c1']) / Nt
        s10 = mpf(d['S10']) / Nt
        nc0 = mpf(d['n00_c0']) / Nt
        nc1 = mpf(d['n00_c1']) / Nt
        n10 = mpf(d['n10']) / Nt

        hc0 = sc0 / nc0 if nc0 > 0 else mpf(0)
        hc1 = sc1 / nc1 if nc1 > 0 else mpf(0)
        h10 = s10 / n10 if n10 > 0 else mpf(0)

        rr = h10 / hc0 if abs(hc0) > 1e-20 else mpf(0)
        ra = hc1 / hc0 if abs(hc0) > 1e-20 else mpf(0)

        pr(f"  {K:3d} {float(hc0):12.6f} {float(hc1):12.6f} {float(h10):12.6f} "
           f"{float(rr):14.6f} {float(ra):14.6f}")

    # ── Part F: The count ratios ───────────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("PART F: COUNT RATIOS (AREA FRACTIONS)")
    pr(f"{'═' * 72}\n")

    pr("  A_00^{c=0} = n_00^{c=0}/Nt, A_00^{c=1} = n_00^{c=1}/Nt")
    pr()
    pr(f"  {'K':>3} {'A_c0':>12} {'A_c1':>12} {'A_10':>12} "
       f"{'A_c1/A_c0':>12} {'A_c0+A_c1':>12}")

    a_c1_over_c0 = []
    for K in sorted(C_DATA.keys()):
        d = C_DATA[K]
        Nt = mpf(d['Nt'])
        ac0 = mpf(d['n00_c0']) / Nt
        ac1 = mpf(d['n00_c1']) / Nt
        a10 = mpf(d['n10']) / Nt
        r = ac1 / ac0 if ac0 > 0 else mpf(0)
        a_c1_over_c0.append((K, r))
        pr(f"  {K:3d} {float(ac0):12.8f} {float(ac1):12.8f} {float(a10):12.8f} "
           f"{float(r):12.8f} {float(ac0+ac1):12.8f}")

    pr()
    pr("  The ratio A_c1/A_c0 → limit:")
    prev = None
    for K, r in a_c1_over_c0:
        delta_str = ""
        if prev is not None:
            delta_str = f"  Δ={float(r-prev):+.8e}"
        pr(f"    K={K}: {float(r):.10f}{delta_str}")
        prev = r

    pr()
    lim_area_ratio = a_c1_over_c0[-1][1]
    pr(f"  Limit estimate: {float(lim_area_ratio):.10f}")
    pr(f"  Known value: n_c1/n_c0 → A_00^{{c=1}}/A_00^{{c=0}} =")
    pr(f"    = (area of (0,0) ∩ D-even) / (area of (0,0) ∩ D-odd)")

    from mpmath import quad as mpquad

    # Area of (0,0)∩D-even: X,Y ∈ [1/2,3/4), XY ≥ 1/2
    # Boundary: Y = 1/(2X), crosses [1/2,3/4)² at X ∈ [2/3, 3/4]
    # D-even part: ∫_{2/3}^{3/4} (3/4 - 1/(2X)) dX
    area_c1 = mpquad(lambda X: mpf(3)/4 - 1/(2*X), [mpf(2)/3, mpf(3)/4])
    area_total_00 = mpf(1)/4 * mpf(1)/4  # = 1/16, area of [1/2,3/4)²

    pr(f"  Area (0,0)∩D-even (XY≥1/2): {float(area_c1):.10f}")
    pr(f"  Total area (0,0): 1/16 = {float(area_total_00):.10f}")

    area_00_dodd = area_total_00 - area_c1
    pr(f"  Area (0,0)∩D-odd: {float(area_00_dodd):.10f}")
    pr(f"  Expected A_00/4 = {float(area_00_dodd):.10f} vs "
       f"ln(9/8)/2 = {float(log(mpf(9)/8)/2):.10f}")
    pr(f"  Ratio A_c1_geom/A_c0_geom: {float(area_c1/area_00_dodd):.10f}")

    # But the data ratio A_c1/A_c0 counts pairs by c_{D-2}, not by geometry.
    # Pairs with c_{D-2}=1 can come from BOTH D-even and D-odd sub-regions.
    # The c_{D-2}=1 condition depends on the carry chain, not just XY.
    pr(f"\n  WARNING: A_c1/A_c0 is NOT purely geometric.")
    pr(f"  It depends on P(c_{{D-2}}=1 | pair), which involves the")
    pr(f"  carry chain structure, not just the product XY.")
    pr(f"  Data at K=18: A_c1/A_c0 = {float(a_c1_over_c0[-1][1]):.10f}")

    # ── Part G: Testing the PRODUCT α × (A_c0/A_c1) ───────────────────
    pr(f"\n{'═' * 72}")
    pr("PART G: THE WEIGHT RATIO σ_c1/σ_c0 = α = (h̄_c1/h̄_c0)·(A_c1/A_c0)")
    pr(f"{'═' * 72}\n")

    pr("  α = σ_c1/σ_c0 = (h̄_c1·A_c1)/(h̄_c0·A_c0)")
    pr("  If α → 1/6, and A_c1/A_c0 → some known ratio,")
    pr("  then h̄_c1/h̄_c0 → (1/6)·(A_c0/A_c1)")
    pr()

    for K in [14, 15, 16, 17, 18]:
        d = C_DATA[K]
        Nt = mpf(d['Nt'])
        nc0 = mpf(d['n00_c0']) / Nt
        nc1 = mpf(d['n00_c1']) / Nt
        sc0 = mpf(d['S00_c0']) / Nt
        sc1 = mpf(d['S00_c1']) / Nt

        hc0 = sc0 / nc0
        hc1 = sc1 / nc1
        area_r = nc1 / nc0
        weight_r = hc1 / hc0
        alpha = sc1 / sc0

        pr(f"  K={K}: A_c1/A_c0={float(area_r):.8f}  h̄_c1/h̄_c0={float(weight_r):.8f}  "
           f"α={float(alpha):.8f}")

    # ── Part H: Synthesis ──────────────────────────────────────────────
    pr(f"\n{'═' * 72}")
    pr("PART H: SYNTHESIS — THE COMPLETE DECOMPOSITION")
    pr(f"{'═' * 72}\n")

    TARGET_SO = pi / 18 - 1 + 6 * log(2) - 3 * log(3)

    pr("  IF σ_10/σ_00 = -π AND Σ_odd = σ_00 + 2σ_10:")
    pr(f"    σ_00 = -Σ_odd/(2π-1) = {float(-TARGET_SO/(2*pi-1)):.12f}")
    pr(f"    σ_10 = π·Σ_odd/(2π-1) = {float(pi*TARGET_SO/(2*pi-1)):.12f}")
    pr()

    sigma_00_pred = -TARGET_SO / (2 * pi - 1)
    sigma_10_pred = pi * TARGET_SO / (2 * pi - 1)

    if True:  # compute predicted α assuming 1/6
        sig_c0_pred = sigma_00_pred * 6 / 7
        sig_c1_pred = sigma_00_pred * 1 / 7
        pr(f"  If α = 1/6:")
        pr(f"    σ_00^{{c=0}} = 6σ_00/7 = {float(sig_c0_pred):.12f}")
        pr(f"    σ_00^{{c=1}} = σ_00/7  = {float(sig_c1_pred):.12f}")
        pr(f"    β = σ_10/σ_00^{{c=0}} = -7π/6 = {float(-7*pi/6):.12f}")
        pr()

        pr(f"  Verification at K=18:")
        d = C_DATA[18]
        Nt = mpf(d['Nt'])
        sc0_18 = mpf(d['S00_c0']) / Nt
        sc1_18 = mpf(d['S00_c1']) / Nt
        s10_18 = mpf(d['S10']) / Nt

        pr(f"    σ_00^{{c=0}}: data={float(sc0_18):.10e}, pred={float(sig_c0_pred):.10e}, "
           f"diff={float(sc0_18-sig_c0_pred):.4e}")
        pr(f"    σ_00^{{c=1}}: data={float(sc1_18):.10e}, pred={float(sig_c1_pred):.10e}, "
           f"diff={float(sc1_18-sig_c1_pred):.4e}")
        pr(f"    σ_10:       data={float(s10_18):.10e}, pred={float(sigma_10_pred):.10e}, "
           f"diff={float(s10_18-sigma_10_pred):.4e}")

    # Final check: Q(K) = (2π-1)·σ_10 - π·Σ_odd for the combined test
    pr(f"\n  Master test: Q(K) = σ_10 + π·σ_00 → 0")
    pr(f"  {'K':>3} {'Q(K)':>14} {'Q·2^K':>12} {'Q·K·2^K':>12} "
       f"{'Q·K²·2^K':>12}")
    for K in sorted(C_DATA.keys()):
        d = C_DATA[K]
        Nt = mpf(d['Nt'])
        s10 = mpf(d['S10']) / Nt
        s00 = mpf(d['S00_c0'] + d['S00_c1']) / Nt
        Q = s10 + pi * s00
        Q2K = Q * mpf(2) ** K
        pr(f"  {K:3d} {float(Q):+14.8e} {float(Q2K):+12.4f} "
           f"{float(Q2K*K):+12.2f} {float(Q2K*K**2):+12.1f}")

    pr(f"\n  Runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
