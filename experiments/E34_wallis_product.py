#!/usr/bin/env python3
"""
E34: Wallis Product Investigation

Key observation: Surv∞(d) = Π_{d'=1}^{d-1} Q∞(d') can be decomposed as
  Surv∞(d) = (1/2)^{d-1} · Π_{d'=1}^{d-1} (1 + 2ε(d'))
where ε(d') = Q∞(d') - 1/2.

If the infinite product Π(1+2ε(d)) converges to a known constant
(e.g., √3, π/2, etc.), that's the key to the proof.

Also: the full resolvent series with extrapolated Q∞ and r∞.
"""
import math
import numpy as np

PI = math.pi
L2 = math.log(2)
L3 = math.log(3)
R0 = (0.5 - 4 * L2 + 2 * L3) / (1 - 3 * L2 + L3)
DELTA_R_TARGET = -PI - R0


def pr(s=""):
    print(s, flush=True)


pr("=" * 80)
pr("E34: Wallis Product Investigation")
pr("=" * 80)
pr()

# Q∞(d) from prior experiments Aitken extrapolation (K=14,15,16)
Q_inf_10 = {
    1: 0.5427627062,
    2: 0.5785615350,
    3: 0.5414929630,
    4: 0.5384328782,
    5: 0.5294932953,
    6: 0.5229364791,
    7: 0.5142573934,
    8: 0.5105631277,
    9: 0.5074799532,
    10: 0.5018459997,
}

# r∞(d) from K=16 data (best available)
r_inf_10 = {
    1: 1.27037,
    2: 1.31036,
    3: 1.33719,
    4: 1.30286,
    5: 1.31363,
    6: 1.30914,
    7: 1.28943,
    8: 1.24171,
    9: 1.15540,
    10: 0.97190,
}

# ═══════════════════════════════════════════════════════════════════════
pr("SECTION 1: The Correction Product Π(1 + 2ε)")
pr("=" * 80)
pr()

pr(f"  {'d':>3s}  {'Q∞(d)':>10s}  {'ε(d)':>10s}  {'1+2ε':>10s}  "
   f"{'Π(1+2ε)':>12s}  {'Surv(d)':>12s}  {'(1/2)^d':>12s}")

product = 1.0
for d in range(1, 11):
    q = Q_inf_10[d]
    eps = q - 0.5
    factor = 1 + 2 * eps
    product *= factor
    surv = product / (2 ** d)
    half_d = 1.0 / (2 ** d)

    pr(f"  {d:3d}  {q:10.6f}  {eps:+10.6f}  {factor:10.6f}  "
       f"{product:12.8f}  {surv:12.8f}  {half_d:12.8f}")

pr()
pr(f"  Π(1+2ε) for d=1..10 = {product:.10f}")
pr()

known = {
    '√3': math.sqrt(3),
    'π/2': PI / 2,
    'e^(1/2)': math.exp(0.5),
    '4/3': 4.0 / 3,
    '7/4': 7.0 / 4,
    '√e': math.sqrt(math.e),
    '√(eπ/3)': math.sqrt(math.e * PI / 3),
    'ln(2)^(-1)': 1 / L2,
    '2ln(3)-1': 2 * L3 - 1,
    '3/√π': 3 / math.sqrt(PI),
    'Γ(1/2)/Γ(1)': math.sqrt(PI),
    '2√(2/π)': 2 * math.sqrt(2 / PI),
}

pr("  Testing known constants:")
for name, val in sorted(known.items(), key=lambda x: abs(x[1] - product)):
    diff = product - val
    pr(f"    {name:20s} = {val:.10f}  diff = {diff:+.6f}")
    if abs(diff) < 0.05:
        pr(f"      *** CLOSE MATCH ***")
pr()

# Estimate remaining product d > 10
tail_eps = sum(Q_inf_10.get(d, 0.5) - 0.5 for d in range(11, 20))
pr(f"  Estimated tail correction (d=11..∞): ≈ exp(2·{tail_eps:.6f}) = {math.exp(2*tail_eps):.6f}")
pr(f"  Full product estimate: {product * math.exp(2 * tail_eps):.10f}")
pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 2: Full Resolvent Series with Q∞ and r∞")
pr("=" * 80)
pr()

pr("  Δ⟨val⟩₁₀ = Σ_d Surv(d) · (1-Q(d)) · r(d)")
pr()

surv = 1.0
cum_series = 0.0
pr(f"  {'d':>3s}  {'Q∞':>10s}  {'r∞':>10s}  {'Surv':>12s}  "
   f"{'term':>14s}  {'cum':>14s}")

for d in range(1, 11):
    q = Q_inf_10[d]
    r = r_inf_10[d]
    term = surv * (1 - q) * r
    cum_series += term

    pr(f"  {d:3d}  {q:10.6f}  {r:10.6f}  {surv:12.8f}  "
       f"{term:+14.10f}  {cum_series:+14.10f}")

    surv *= q

pr()
pr(f"  Δ⟨val⟩₁₀ (10 terms) = {cum_series:.10f}")

# Tail: approximate with Q=1/2, r≈1 (reward decreases to 0 deep in the chain)
tail = surv * 1.0
cum_with_tail = cum_series + tail
pr(f"  Surv(11) = {surv:.10f}")
pr(f"  Tail estimate (Q=1/2, r=1): {tail:.10f}")
pr(f"  Total Δ⟨val⟩₁₀ ≈ {cum_with_tail:.10f}")
pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 3: What Δ⟨val⟩₁₀ and ⟨val⟩₀₀ Must Be for R = -π")
pr("=" * 80)
pr()

N10_cont = 2 * math.log(4 / 3) - 0.5
N00_cont = 2 * math.log(9 / 8)
nr = N10_cont / N00_cont

pr(f"  R = (N₁₀/N₀₀) · (⟨val⟩₁₀ / ⟨val⟩₀₀) = -π")
pr(f"  N₁₀/N₀₀ = {nr:.10f}")
pr(f"  ⟨val⟩₁₀ = -1 + Δ⟨val⟩₁₀")
pr(f"  ⟨val⟩₁₀ / ⟨val⟩₀₀ = -π / (N₁₀/N₀₀) = {-PI / nr:.10f}")
pr()

val10_over_val00 = -PI / nr
pr(f"  Required: ⟨val⟩₁₀ / ⟨val⟩₀₀ = {val10_over_val00:.10f}")
pr()

# If Δ⟨val⟩₁₀∞ ≈ 1.297:
delta_10_est = 1.2971
val10_est = -1 + delta_10_est
val00_est = val10_est / val10_over_val00
pr(f"  If Δ⟨val⟩₁₀∞ = {delta_10_est:.4f}:")
pr(f"    ⟨val⟩₁₀ = {val10_est:+.4f}")
pr(f"    ⟨val⟩₀₀ must = {val00_est:.10f}")
pr(f"    R = {nr:.6f} · {val10_est:.6f} / {val00_est:.6f} = {nr * val10_est / val00_est:.10f}")
pr()

# Check with directly extrapolated ⟨val⟩₀₀
val00_extr = -0.0301
pr(f"  Extrapolated ⟨val⟩₀₀∞ = {val00_extr:.6f}")
pr(f"  R∞ = {nr:.6f} · {val10_est:.6f} / {val00_extr:.6f} = {nr * val10_est / val00_extr:.6f}")
pr(f"  Gap from -π: {nr * val10_est / val00_extr + PI:.6f}")
pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 4: Logarithmic Structure of the Product")
pr("=" * 80)
pr()

pr("  ln(Π(1+2ε)) = Σ ln(1+2ε(d))")
pr("  ≈ Σ 2ε(d) - Σ 2ε²(d) + ...")
pr()

sum_eps = sum(Q_inf_10[d] - 0.5 for d in range(1, 11))
sum_eps2 = sum((Q_inf_10[d] - 0.5) ** 2 for d in range(1, 11))
ln_product = math.log(product)

pr(f"  Σε(d) for d=1..10 = {sum_eps:.10f}")
pr(f"  2·Σε(d) = {2 * sum_eps:.10f}")
pr(f"  2·Σε²(d) = {2 * sum_eps2:.10f}")
pr(f"  ln(Π) exact = {ln_product:.10f}")
pr(f"  ln(Π) ≈ 2Σε - 2Σε² = {2 * sum_eps - 2 * sum_eps2:.10f}")
pr()

pr("  Testing ln(Π) against known constants:")
known_ln = {
    'ln(√3) = ln(3)/2': L3 / 2,
    'ln(π/2)': math.log(PI / 2),
    'ln(e^0.5) = 1/2': 0.5,
    'ln(7/4)': math.log(7 / 4),
    'ln(4/3)': math.log(4 / 3),
    'π/6': PI / 6,
    '2ln(4/3)': 2 * math.log(4 / 3),
    '1-ln(2)': 1 - L2,
    '2-2ln(2)': 2 - 2 * L2,
}

for name, val in sorted(known_ln.items(), key=lambda x: abs(x[1] - ln_product)):
    diff = ln_product - val
    pr(f"    {name:25s} = {val:.10f}  diff = {diff:+.8f}")
    if abs(diff) < 0.02:
        pr(f"      *** CLOSE MATCH ***")
pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 5: The ε(d) Series — Is It Summable?")
pr("=" * 80)
pr()

pr("  ε(d) = Q∞(d) - 1/2:")
pr(f"  {'d':>3s}  {'ε(d)':>12s}  {'d·ε(d)':>12s}  {'ε(d)/ε(d-1)':>12s}")

for d in range(1, 11):
    eps = Q_inf_10[d] - 0.5
    ratio = ""
    if d > 1:
        eps_prev = Q_inf_10[d - 1] - 0.5
        if abs(eps_prev) > 1e-10:
            ratio = f"{eps / eps_prev:.6f}"
    pr(f"  {d:3d}  {eps:+12.8f}  {d * eps:+12.6f}  {ratio:>12s}")

pr()
pr(f"  Σε = {sum_eps:.10f}")
pr(f"  If Σε → ln(√3)/2 = {L3/4:.10f}: diff = {sum_eps - L3/4:.6f}")
pr(f"  If Σε → π/8 = {PI/8:.10f}: diff = {sum_eps - PI/8:.6f}")
pr(f"  If Σε → ln(2)/2 = {L2/2:.10f}: diff = {sum_eps - L2/2:.6f}")
pr(f"  If Σε → 1/π = {1/PI:.10f}: diff = {sum_eps - 1/PI:.6f}")
pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 6: Cross-Sector Product Comparison")
pr("=" * 80)
pr()

Q_inf_00 = {
    1: 0.70175846,
    2: 0.66136199,
    3: 0.59904072,
    4: 0.60793166,
    5: 0.57190495,
    6: 0.58078265,
    7: 0.54549026,
    8: 0.56801139,
}

pr("  Sector (0,0): Q∞(d)")
product_00 = 1.0
for d in range(1, 9):
    if d not in Q_inf_00:
        break
    q = Q_inf_00[d]
    eps = q - 0.5
    product_00 *= (1 + 2 * eps)
    pr(f"    d={d}: Q∞={q:.6f}, ε={eps:+.6f}, Π(1+2ε)={product_00:.8f}")

pr()

product_10_8 = 1.0
for d in range(1, 9):
    eps = Q_inf_10[d] - 0.5
    product_10_8 *= (1 + 2 * eps)

ratio_products = product_10_8 / product_00 if abs(product_00) > 1e-10 else float('nan')
pr(f"  Π₁₀(d=1..8) = {product_10_8:.10f}")
pr(f"  Π₀₀(d=1..8) = {product_00:.10f}")
pr(f"  Π₁₀ / Π₀₀  = {ratio_products:.10f}")
pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 7: Alternative Product — Q(0)₀₀ contribution")
pr("=" * 80)
pr()

pr("  In sector (0,0), Q(0) ≈ 0.459 (not 1.0 as in sector 10).")
pr("  Q₀₀(0)∞ ≈ 0.4595")
pr("  This is the 'starting survival' for sector 00.")
pr()

Q00_0 = 0.4595
pr(f"  Surv₀₀(d) = Q₀₀(0) · Π₁(d) Q₀₀")
pr(f"  Surv₁₀(d) = 1.0 · Π₁(d) Q₁₀")
pr()
pr(f"  The sector asymmetry at d=0: Q₁₀(0)/Q₀₀(0) = 1.0/{Q00_0:.4f} = {1.0/Q00_0:.6f}")
pr(f"  2/(Q₀₀(0)) would be {2/Q00_0:.6f}")
pr(f"  If Q₀₀(0) = π/2·(N₀₀/N₁₀) × something?")
pr(f"  Q₀₀(0)∞ should equal P(carry[D-3]>0 | D-odd, sector 00)")
pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SUMMARY")
pr("=" * 80)
pr()

pr(f"  The correction product Π(1+2ε) over 10 terms = {product:.10f}")
pr(f"  ln(Π) = {ln_product:.10f}")
pr()
pr(f"  Closest matches:")
pr(f"    √e = {math.sqrt(math.e):.10f} (diff = {product - math.sqrt(math.e):+.6f})")
pr(f"    √3 = {math.sqrt(3):.10f} (diff = {product - math.sqrt(3):+.6f})")
pr(f"    7/4 = {7/4:.10f} (diff = {product - 7/4:+.6f})")
pr()
pr("  The infinite product is approximately 1.74 ± 0.04.")
pr("  Insufficient precision to identify. Need Q∞(d) to 4+ significant digits.")
pr()
pr(f"  R₀ = {R0:+.10f}")
pr(f"  ΔR target = {DELTA_R_TARGET:+.10f}")
pr(f"  -π = {-PI:.10f}")
pr("=" * 80)
