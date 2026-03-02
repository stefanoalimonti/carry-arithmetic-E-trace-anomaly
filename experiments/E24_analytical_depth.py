#!/usr/bin/env python3
"""
E24: Analytical computation of σ(d₀) at each depth level

For each depth d₀, enumerate ALL valid bit patterns (a₂,...,a_{d₀+1}, c₂,...,c_{d₀+1})
that lead to first-carry at depth d₀, compute val and the exact integral.

Each sub-region is the intersection of:
  - u-rectangle: u ∈ [u_lo, u_hi) from bit prefix of u
  - v-rectangle: v ∈ [v_lo, v_hi) from bit prefix of v
  - t-band: t ∈ [t_lo, t_hi) from product bits b₁,...,b_{d₀+1}
  - D-odd: t < 2

The integral is always: Σ of (α · Δu + β · ln((1+u₂)/(1+u₁))) terms.
"""

import math
from fractions import Fraction
from collections import defaultdict

def pr(s=""):
    print(s)

E24_theory = 2 * math.log(9/8)
E24_theory = 2 * math.log(4/3) - 0.5


def exact_area(u_lo, u_hi, v_lo, v_hi, t_lo, t_hi):
    """Exact ∫∫ du dv over {u∈[u_lo,u_hi], v∈[v_lo,v_hi], t_lo≤(1+u)(1+v)<t_hi}.
    All integrals reduce to u·const and ln(1+u) terms."""
    if u_hi <= u_lo or v_hi <= v_lo or t_hi <= t_lo:
        return 0.0

    breakpoints = sorted(set([u_lo, u_hi] + [
        t / (1 + v) - 1
        for t in [t_lo, t_hi] for v in [v_lo, v_hi]
        if u_lo < t / (1 + v) - 1 < u_hi
    ]))

    total = 0.0
    for i in range(len(breakpoints) - 1):
        ua, ub = breakpoints[i], breakpoints[i + 1]
        if ub - ua < 1e-18:
            continue
        u_mid = (ua + ub) / 2

        v_from_t_hi = t_hi / (1 + u_mid) - 1
        v_from_t_lo = t_lo / (1 + u_mid) - 1
        v_upper = min(v_hi, v_from_t_hi)
        v_lower = max(v_lo, v_from_t_lo)

        if v_upper <= v_lower + 1e-18:
            continue

        upper_hyp = (v_from_t_hi < v_hi - 1e-15)
        lower_hyp = (v_from_t_lo > v_lo + 1e-15)
        log_ratio = math.log((1 + ub) / (1 + ua))
        du = ub - ua

        if upper_hyp and lower_hyp:
            total += (t_hi - t_lo) * log_ratio
        elif upper_hyp:
            total += t_hi * log_ratio - (1 + v_lo) * du
        elif lower_hyp:
            total += (1 + v_hi) * du - t_lo * log_ratio
        else:
            total += (v_hi - v_lo) * du

    return total


def compute_sigma_analytical(sector, max_depth=12):
    """Compute σ(d₀) analytically for each depth d₀ in the given sector.
    sector: '00' or '10'."""

    a1 = int(sector[0])
    c1 = int(sector[1])
    results = {}

    for d0 in range(1, max_depth + 1):
        n_bits = d0 + 1  # need bits a_2..a_{d0+1}, c_2..c_{d0+1}
        sigma_d = 0.0
        n_valid = 0
        n_total = 0

        for pattern in range(1 << (2 * d0)):
            a_extra = [(pattern >> (2 * d0 - 1 - 2 * j)) & 1 for j in range(d0)]
            c_extra = [(pattern >> (2 * d0 - 2 - 2 * j)) & 1 for j in range(d0)]

            a_bits = [1, a1] + a_extra  # a[0]=1, a[1]=a1, a[2]..a[d0+1]
            c_bits = [1, c1] + c_extra

            n_total += 1
            valid = True
            b_bits = [1]  # b[0] = 1 always (t ∈ [1,2))

            for k in range(1, d0):
                C_k = sum(a_bits[i] * c_bits[k - i] for i in range(k + 1))
                if C_k < 0 or C_k > 1:
                    valid = False
                    break
                b_bits.append(C_k)

            if not valid:
                continue

            C_d0 = sum(a_bits[i] * c_bits[d0 - i] for i in range(d0 + 1))
            if C_d0 != 0:
                continue
            b_bits.append(1)  # b[d0] = 1

            C_d0p1 = sum(a_bits[i] * c_bits[d0 + 1 - i] for i in range(d0 + 2))

            for b_next in [0, 1]:
                val = 1 + b_next - C_d0p1
                b_full = b_bits + [b_next]

                t_lo = sum(b_full[j] * 2**(-j) for j in range(len(b_full)))
                t_hi = t_lo + 2**(-(d0 + 1))
                t_hi = min(t_hi, 2.0)  # D-odd constraint

                if t_lo >= 2.0 or t_hi <= t_lo:
                    continue

                u_lo = a1 * 0.5 + sum(a_extra[j] * 2**(-(j + 2)) for j in range(d0))
                u_hi = u_lo + 2**(-(d0 + 1))

                v_lo = c1 * 0.5 + sum(c_extra[j] * 2**(-(j + 2)) for j in range(d0))
                v_hi = v_lo + 2**(-(d0 + 1))

                area = exact_area(u_lo, u_hi, v_lo, v_hi, t_lo, t_hi)
                if area > 1e-20:
                    sigma_d += val * area
                    n_valid += 1

        results[d0] = {'sigma': sigma_d, 'n_valid': n_valid, 'n_total': n_total}

    return results


# ---------- PART A: Sector (0,0) ----------
pr("=" * 70)
pr("PART A: Analytical σ₀₀(d₀) for sector (0,0)")
pr("=" * 70)
pr()

max_d = 10
res_00 = compute_sigma_analytical('00', max_depth=max_d)

cum_sigma = 0.0
pr(f"{'d₀':>4s} {'σ₀₀(d)':>16s} {'cum σ₀₀':>16s} {'n_valid':>8s} {'n_patterns':>12s}")
for d0 in range(1, max_d + 1):
    r = res_00[d0]
    cum_sigma += r['sigma']
    pr(f"{d0:4d} {r['sigma']:+16.12f} {cum_sigma:+16.12f} {r['n_valid']:8d} {r['n_total']:12d}")

pr()
pr(f"Cumulative σ₀₀ = {cum_sigma:+.12f}")
pr()

# ---------- PART B: Sector (1,0) ----------
pr("=" * 70)
pr("PART B: Analytical σ₁₀(d₀) for sector (1,0)")
pr("=" * 70)
pr()

res_10 = compute_sigma_analytical('10', max_depth=max_d)

cum_sigma_10 = 0.0
pr(f"{'d₀':>4s} {'σ₁₀(d)':>16s} {'cum σ₁₀':>16s} {'n_valid':>8s}")
for d0 in range(1, max_d + 1):
    r = res_10[d0]
    cum_sigma_10 += r['sigma']
    pr(f"{d0:4d} {r['sigma']:+16.12f} {cum_sigma_10:+16.12f} {r['n_valid']:8d}")

pr()
pr(f"Cumulative σ₁₀ = {cum_sigma_10:+.12f}")
pr()

# ---------- PART C: Ratio and convergence ----------
pr("=" * 70)
pr("PART C: Ratio σ₁₀/σ₀₀ convergence")
pr("=" * 70)
pr()

cum_00 = 0.0
cum_10 = 0.0
pr(f"{'d_max':>6s} {'σ₀₀':>16s} {'σ₁₀':>16s} {'σ₁₀/σ₀₀':>14s} {'|R+π|':>12s}")
for d0 in range(1, max_d + 1):
    cum_00 += res_00[d0]['sigma']
    cum_10 += res_10[d0]['sigma']
    if abs(cum_00) > 1e-16:
        R = cum_10 / cum_00
        pr(f"{d0:6d} {cum_00:+16.12f} {cum_10:+16.12f} {R:14.8f} {abs(R + math.pi):12.8f}")

pr()
pr(f"Target: σ₁₀/σ₀₀ = -π = {-math.pi:.12f}")
pr()

# ---------- PART D: Decompose σ into Area + ∫b - ∫C ----------
pr("=" * 70)
pr("PART D: Decompose σ = Area + ∫b - ∫C at each depth")
pr("=" * 70)
pr()

def compute_sigma_decomposed(sector, max_depth=8):
    a1 = int(sector[0])
    c1 = int(sector[1])
    results = {}

    for d0 in range(1, max_depth + 1):
        area_total = 0.0
        int_b_total = 0.0
        int_C_total = 0.0
        sigma_total = 0.0

        for pattern in range(1 << (2 * d0)):
            a_extra = [(pattern >> (2 * d0 - 1 - 2 * j)) & 1 for j in range(d0)]
            c_extra = [(pattern >> (2 * d0 - 2 - 2 * j)) & 1 for j in range(d0)]

            a_bits = [1, a1] + a_extra
            c_bits = [1, c1] + c_extra

            b_bits = [1]
            valid = True
            for k in range(1, d0):
                C_k = sum(a_bits[i] * c_bits[k - i] for i in range(k + 1))
                if C_k < 0 or C_k > 1:
                    valid = False
                    break
                b_bits.append(C_k)

            if not valid:
                continue

            C_d0 = sum(a_bits[i] * c_bits[d0 - i] for i in range(d0 + 1))
            if C_d0 != 0:
                continue
            b_bits.append(1)

            C_d0p1 = sum(a_bits[i] * c_bits[d0 + 1 - i] for i in range(d0 + 2))

            for b_next in [0, 1]:
                b_full = b_bits + [b_next]
                t_lo = sum(b_full[j] * 2**(-j) for j in range(len(b_full)))
                t_hi = t_lo + 2**(-(d0 + 1))
                t_hi = min(t_hi, 2.0)

                if t_lo >= 2.0 or t_hi <= t_lo:
                    continue

                u_lo = a1 * 0.5 + sum(a_extra[j] * 2**(-(j + 2)) for j in range(d0))
                u_hi = u_lo + 2**(-(d0 + 1))
                v_lo = c1 * 0.5 + sum(c_extra[j] * 2**(-(j + 2)) for j in range(d0))
                v_hi = v_lo + 2**(-(d0 + 1))

                area = exact_area(u_lo, u_hi, v_lo, v_hi, t_lo, t_hi)
                if area > 1e-20:
                    area_total += area
                    int_b_total += b_next * area
                    int_C_total += C_d0p1 * area
                    sigma_total += (1 + b_next - C_d0p1) * area

        results[d0] = {
            'area': area_total,
            'int_b': int_b_total,
            'int_C': int_C_total,
            'sigma': sigma_total,
        }
    return results

for sector_label in ['00', '10']:
    res = compute_sigma_decomposed(sector_label, max_depth=8)
    pr(f"Sector ({sector_label[0]},{sector_label[1]}):")
    pr(f"{'d₀':>4s} {'Area':>14s} {'∫b':>14s} {'∫C':>14s} {'σ=A+∫b-∫C':>14s}")
    for d0 in sorted(res.keys()):
        r = res[d0]
        pr(f"{d0:4d} {r['area']:+14.10f} {r['int_b']:+14.10f} "
           f"{r['int_C']:+14.10f} {r['sigma']:+14.10f}")
    pr()


# ---------- PART E: Individual logarithmic terms ----------
pr("=" * 70)
pr("PART E: Structure of individual integrals")
pr("=" * 70)
pr()
pr("Each sub-integral is a combination of:")
pr("  α·Δu + β·ln((1+u₂)/(1+u₁))")
pr("for specific constants α, β from the hyperbolic boundaries.")
pr()

pr("Checking: does σ₀₀ involve π explicitly?")
pr()
pr(f"σ₀₀(cumulative d=1..{max_d}) = {cum_sigma:+.14f}")
pr(f"σ₁₀(cumulative d=1..{max_d}) = {cum_sigma_10:+.14f}")

sigma_00_exact = cum_sigma
sigma_10_exact = cum_sigma_10
if abs(sigma_00_exact) > 1e-16:
    R_exact = sigma_10_exact / sigma_00_exact
    pr(f"\nσ₁₀/σ₀₀ = {R_exact:.14f}")
    pr(f"-π       = {-math.pi:.14f}")
    pr(f"|R+π|    = {abs(R_exact + math.pi):.10f}")
    pr(f"\nResidual σ₁₀ + π·σ₀₀ = {sigma_10_exact + math.pi * sigma_00_exact:+.14f}")
    pr(f"σ₁₀ - (-π)·σ₀₀ = {sigma_10_exact + math.pi * sigma_00_exact:+.14f}")

pr()

# ---------- PART F: Test closed forms ----------
pr("=" * 70)
pr("PART F: Test known closed forms for σ₀₀ and σ₁₀")
pr("=" * 70)
pr()

candidates_sigma00 = [
    ("-ln(2)²/2", -math.log(2)**2 / 2),
    ("1/4 - ln(2)/2", 0.25 - math.log(2)/2),
    ("-7/960", -7/960),
    ("-(1 - ln 2)²", -(1 - math.log(2))**2),
    ("ln(3)/4 - ln(2)/2", math.log(3)/4 - math.log(2)/2),
    ("-π/432", -math.pi/432),
    ("-1/140", -1/140),
    ("2·ln(9/8) - 1/4", 2*math.log(9/8) - 0.25),
    ("ln(9/8) - 1/8", math.log(9/8) - 1/8),
    ("2·(ln(9/8)-1/8)", 2*(math.log(9/8) - 1/8)),
]

pr(f"σ₀₀ = {sigma_00_exact:+.14f}")
for name, val in candidates_sigma00:
    if abs(val) > 1e-16:
        pr(f"  {name:30s} = {val:+.14f}  diff = {abs(val - sigma_00_exact):.6e}")

pr()
pr(f"σ₁₀ = {sigma_10_exact:+.14f}")
candidates_sigma10 = [
    ("π/140", math.pi/140),
    ("ln(4/3)/8", math.log(4/3)/8),
    ("(2ln(4/3)-1/2)/8", (2*math.log(4/3)-0.5)/8),
    ("π·f₁₀/9", math.pi * f10_theory / 9),
    ("(1-ln2)/32", (1 - math.log(2))/32),
]
for name, val in candidates_sigma10:
    if abs(val) > 1e-16:
        pr(f"  {name:30s} = {val:+.14f}  diff = {abs(val - sigma_10_exact):.6e}")

pr()

# ---------- PART G: Synthesis ----------
pr("=" * 70)
pr("PART G: Synthesis")
pr("=" * 70)
pr()
pr("The analytical depth series gives EXACT values for σ₀₀(d₀) and σ₁₀(d₀).")
pr("Each value is a linear combination of terms of the form:")
pr("  c₁·Δu + c₂·ln(r)  for rational r")
pr()
pr("The cumulative sum should converge to values whose ratio is -π.")
pr("The series truncated at d₀=10 gives the current best estimate.")
pr("Deeper computation (d₀=12-15) would improve precision but")
pr("requires exponentially more patterns (4^d₀).")
