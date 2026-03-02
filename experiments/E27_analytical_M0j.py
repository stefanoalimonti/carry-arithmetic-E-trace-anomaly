#!/usr/bin/env python3
"""
E27: Analytical M_{0,j} — The Arctangent Path to π

Computes the UNCONSTRAINED weight integrals M_{0,j} for both sectors
EXACTLY as rational numbers + rational multiples of logarithms.

KEY FORMULA:
  M_{0,j}^{sector} = ∫_{U_sector} G_j(v_top(u)) du

where v_top(u) = min(1/2, (1-u)/(1+u)) and G_j is the bit mass function.

G_j has breakpoints at v_k = k/2^j. In u-coordinates:
  u_k = (2^j - k) / (2^j + k)

Each sub-integral gives EXACT terms:
  rational · Δu  +  2 · rational · ln((2^j + k_a) / (2^j + k_b))

The prime decomposition reveals the structure connecting to π.
"""

from fractions import Fraction
from collections import defaultdict
import math

def pr(s=""):
    print(s, flush=True)


class SymVal:
    """Exact symbolic value: rational + Σ c_i · ln(n_i / d_i)"""
    def __init__(self, rat=Fraction(0), logs=None):
        self.rat = Fraction(rat)
        self.logs = logs or []

    def __add__(self, other):
        return SymVal(self.rat + other.rat, self.logs + other.logs)

    def __sub__(self, other):
        neg_logs = [(-c, n, d) for c, n, d in other.logs]
        return SymVal(self.rat - other.rat, self.logs + neg_logs)

    def __mul__(self, s):
        s = Fraction(s)
        return SymVal(self.rat * s, [(c * s, n, d) for c, n, d in self.logs])

    def __rmul__(self, s):
        return self.__mul__(s)

    def numerical(self):
        v = float(self.rat)
        for c, n, d in self.logs:
            v += float(c) * math.log(n / d)
        return v

    def simplify(self):
        combined = defaultdict(Fraction)
        for c, n, d in self.logs:
            if n < d:
                combined[(d, n)] -= c
            elif n > d:
                combined[(n, d)] += c
        new_logs = [(c, n, d) for (n, d), c in sorted(combined.items()) if c != 0]
        return SymVal(self.rat, new_logs)

    def prime_decomp(self):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                  53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                  127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
                  193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257]
        coeffs = defaultdict(Fraction)
        for c, n, d in self.logs:
            for x, sign in [(n, +1), (d, -1)]:
                tmp = x
                for p in primes:
                    while tmp % p == 0:
                        coeffs[p] += c * sign
                        tmp //= p
                if tmp > 1:
                    coeffs[tmp] += c * sign
        return dict(coeffs)

    def __repr__(self):
        s = f"{self.rat}"
        for c, n, d in self.logs:
            s += f" + ({c})*ln({n}/{d})"
        return s


def compute_M0j(j, sector):
    """Compute M_{0,j} exactly.

    Returns SymVal = rational + Σ c_i * ln(n_i/d_i)

    The integral is:
      ∫_{U_sector} G_j(v_top(u)) du
    where v_top(u) = min(1/2, (1-u)/(1+u)).
    """
    result = SymVal()
    pow2j = 2**j

    if sector == '00':
        u_lo = Fraction(0)
        u_hi = Fraction(1, 2)
        u_cross = Fraction(1, 3)  # where v_top = 1/2

        # Region 1: u in [0, 1/3), v_top = 1/2
        if j == 1:
            g_half = Fraction(0)  # c_1(v)=0 for v<1/2
        else:
            g_half = Fraction(1, 4)  # G_j(1/2) = 1/4 for j>=2
        result = result + SymVal(g_half * (u_cross - u_lo))

        # Region 2: u in [1/3, 1/2), v_top = (1-u)/(1+u) ∈ (1/3, 1/2]
        reg_lo = u_cross
        reg_hi = u_hi
    else:
        # Sector (1,0): u in [1/2, 1), v_top = (1-u)/(1+u) ∈ [0, 1/3)
        reg_lo = Fraction(1, 2)
        reg_hi = Fraction(1)

    # Determine v-range in the hyperbolic region
    v_at_lo = (1 - reg_lo) / (1 + reg_lo)
    v_at_hi = (1 - reg_hi) / (1 + reg_hi)
    # v decreases as u increases: v_at_lo > v_at_hi

    # Breakpoints: v_k = k/2^j in [v_at_hi, v_at_lo]
    k_lo_f = float(v_at_hi) * pow2j
    k_hi_f = float(v_at_lo) * pow2j
    k_lo = math.ceil(k_lo_f - 1e-12)
    k_hi = math.floor(k_hi_f + 1e-12)

    # Map v-breakpoints to u-breakpoints
    bps = [reg_lo, reg_hi]
    for k in range(max(0, k_lo), k_hi + 1):
        u_k = Fraction(pow2j - k, pow2j + k)
        if reg_lo < u_k < reg_hi:
            bps.append(u_k)
    bps = sorted(set(bps))

    for idx in range(len(bps) - 1):
        u_a = bps[idx]
        u_b = bps[idx + 1]
        if u_b <= u_a:
            continue

        # Determine which v-interval the midpoint falls in
        u_mid_f = float((u_a + u_b) / 2)
        v_mid_f = (1 - u_mid_f) / (1 + u_mid_f)
        k = int(math.floor(v_mid_f * pow2j))
        if k < 0:
            k = 0

        # G_j on [k/2^j, (k+1)/2^j): G_j(v) = floor(k/2)/2^j + [k odd]·(v - k/2^j)
        g_base = Fraction(k // 2, pow2j)
        v_base = Fraction(k, pow2j)
        c_j_val = k % 2

        du = u_b - u_a

        if c_j_val == 0:
            result = result + SymVal(g_base * du)
        else:
            # G_j(v) = g_base + (v - v_base)
            # ∫ [g_base - v_base + (1-u)/(1+u)] du
            # = (g_base - v_base - 1)·Δu + [2·ln(1+u)]_{u_a}^{u_b}
            coeff = g_base - v_base - 1
            result = result + SymVal(coeff * du)

            # Log term: 2·ln((1+u_b)/(1+u_a))
            one_ub = 1 + u_b
            one_ua = 1 + u_a
            num = one_ub.numerator * one_ua.denominator
            den = one_ua.numerator * one_ub.denominator
            g = math.gcd(abs(num), abs(den))
            num //= g
            den //= g
            if num > 0 and den > 0:
                result = result + SymVal(Fraction(0), [(Fraction(2), num, den)])

    return result.simplify()


# ═══════════════════════════════════════════════════════════════
pr("=" * 78)
pr("E27: Analytical M_{0,j} — The Path to pi")
pr("=" * 78)
pr()

MAX_J = 12

# ── Part 1: Compute exact M_{0,j} ────────────────────────────
pr("PART 1: Exact M_{0,j} for j=1..%d" % MAX_J)
pr("-" * 50)
pr()

m00 = {}
m10 = {}

for j in range(1, MAX_J + 1):
    m00[j] = compute_M0j(j, '00')
    m10[j] = compute_M0j(j, '10')

    n00 = m00[j].numerical()
    n10 = m10[j].numerical()
    ratio = n10 / n00 if abs(n00) > 1e-15 else float('nan')

    pr(f"  j={j:2d}: M_00 = {n00:+.12f}  M_10 = {n10:+.12f}  "
       f"ratio = {ratio:.8f}")
pr()

# ── Part 2: Exact symbolic forms ─────────────────────────────
pr("PART 2: Exact Symbolic Forms")
pr("-" * 50)
pr()

for j in range(1, min(MAX_J, 5) + 1):
    sv00 = m00[j]
    sv10 = m10[j]
    pr(f"  j={j}:")
    pr(f"    M_00 = {sv00.rat}")
    for c, n, d in sv00.logs:
        pr(f"           + ({c}) * ln({n}/{d})")
    pr(f"    M_10 = {sv10.rat}")
    for c, n, d in sv10.logs:
        pr(f"           + ({c}) * ln({n}/{d})")
    pr()

# ── Part 3: Prime decomposition ──────────────────────────────
pr("PART 3: Prime Decomposition")
pr("-" * 50)
pr()

pr("  M_{0,j} = rational + Σ β_p · ln(p)")
pr()

for j in range(1, min(MAX_J, 8) + 1):
    pd00 = m00[j].prime_decomp()
    pd10 = m10[j].prime_decomp()
    pr(f"  j={j}:")
    all_primes = sorted(set(pd00.keys()) | set(pd10.keys()))
    for p in all_primes:
        c00 = pd00.get(p, Fraction(0))
        c10 = pd10.get(p, Fraction(0))
        pr(f"    ln({p:3d}): coeff_00 = {str(c00):>15s}  "
           f"coeff_10 = {str(c10):>15s}")
    pr(f"    rational: {str(m00[j].rat):>15s}  {str(m10[j].rat):>15s}")
    pr()

# ── Part 4: Cumulative sums and ratio ────────────────────────
pr("=" * 78)
pr("PART 4: Cumulative Sums")
pr("-" * 50)
pr()

cum_00 = SymVal()
cum_10 = SymVal()

pr(f"  {'j':>3s}  {'Σ M_00':>15s}  {'Σ M_10':>15s}  {'ratio':>12s}  {'Σ10+π·Σ00':>15s}")

for j in range(1, MAX_J + 1):
    cum_00 = cum_00 + m00[j]
    cum_10 = cum_10 + m10[j]
    s00 = cum_00.numerical()
    s10 = cum_10.numerical()
    ratio = s10 / s00 if abs(s00) > 1e-15 else 0
    combined = s10 + math.pi * s00
    pr(f"  {j:3d}  {s00:>15.10f}  {s10:>15.10f}  {ratio:>12.8f}  {combined:>15.10f}")

pr()

# ── Part 5: The Difference M_00 - M_10 ───────────────────────
pr("=" * 78)
pr("PART 5: Sector Difference Δ_j = M_{0,j}^{00} - M_{0,j}^{10}")
pr("-" * 50)
pr()

pr("  The asymmetry that drives -π lives in these differences.")
pr()

cum_diff = SymVal()
for j in range(1, MAX_J + 1):
    diff = m00[j] - m10[j]
    diff = diff.simplify()
    cum_diff = cum_diff + diff
    nd = diff.numerical()
    ncd = cum_diff.numerical()
    pr(f"  j={j:2d}: Δ_j = {nd:+.12f}  Σ Δ = {ncd:+.12f}")

pr()

# ── Part 6: Prime structure of cumulative difference ─────────
pr("=" * 78)
pr("PART 6: Prime Structure of Cumulative Δ = Σ(M_00 - M_10)")
pr("-" * 50)
pr()

cd = cum_diff.simplify()
pd_diff = cd.prime_decomp()
pr(f"  Rational part: {cd.rat} = {float(cd.rat):.15f}")
all_p = sorted(pd_diff.keys())
for p in all_p:
    pr(f"  ln({p:3d}): {str(pd_diff[p]):>25s}  = {float(pd_diff[p]):+.15f}")

pr(f"\n  Numerical: {cd.numerical():.15f}")
pr()

# ── Part 7: The breakpoint structure ─────────────────────────
pr("=" * 78)
pr("PART 7: Breakpoints u_k = (2^j - k)/(2^j + k)")
pr("-" * 50)
pr()
pr("  These are the DISCONTINUITIES where G_j changes slope.")
pr("  In angular coords: α_k = arctan(u_k)")
pr()

for j in range(1, 5):
    pow2j = 2**j
    pr(f"  j={j}: breakpoints (in D-odd region):")
    for k in range(0, pow2j):
        u_k = Fraction(pow2j - k, pow2j + k)
        alpha_k = math.atan(float(u_k))
        if 0 < float(u_k) < 1:
            sector = "(0,0)" if float(u_k) < 0.5 else "(1,0)"
            pr(f"    k={k}: u = {u_k} = {float(u_k):.6f}, "
               f"arctan(u) = {alpha_k:.6f} = {alpha_k/math.pi:.6f}π  [{sector}]")
    pr()

# ── Part 8: The arctangent connection ─────────────────────────
pr("=" * 78)
pr("PART 8: Connection to Arctangent Series")
pr("-" * 50)
pr()

pr("  Each ln((2^j+k_a)/(2^j+k_b)) in M_{0,j} corresponds to")
pr("  evaluating the area element between angular breakpoints:")
pr("    arctan((2^j-k_a)/(2^j+k_a)) and arctan((2^j-k_b)/(2^j+k_b))")
pr()

pr("  The sector split at u=1/2 (α=arctan(1/2)) divides the")
pr("  arctangent evaluations into two groups.")
pr()

# Compute the arctan values at all breakpoints for j=1..6
for j in range(1, 7):
    pow2j = 2**j
    arctan_sum_00 = 0.0
    arctan_sum_10 = 0.0
    for k in range(0, pow2j):
        u_k = (pow2j - k) / (pow2j + k)
        at = math.atan(u_k)
        if u_k < 0.5:
            arctan_sum_00 += at * ((-1)**k)
        elif u_k >= 0.5:
            arctan_sum_10 += at * ((-1)**k)
    pr(f"  j={j}: Σ(-1)^k arctan(u_k): "
       f"sector00={arctan_sum_00:+.8f}, sector10={arctan_sum_10:+.8f}, "
       f"total={arctan_sum_00+arctan_sum_10:+.8f}")

pr()
pr(f"  π/4 = {math.pi/4:.10f}")
pr(f"  arctan(1/2) = {math.atan(0.5):.10f}")
pr(f"  arctan(1/3) = {math.atan(1/3):.10f}")
pr()

# ── Part 9: Summary ──────────────────────────────────────────
pr("=" * 78)
pr("PART 9: Summary")
pr("-" * 50)
pr()
pr("  RESULTS:")
pr(f"  1. M_{{0,j}} computed EXACTLY for j=1..{MAX_J} (both sectors)")
pr(f"  2. Each M_{{0,j}} = Fraction + Σ (Fraction)·ln(integer ratio)")
pr(f"  3. The breakpoints u_k = (2^j-k)/(2^j+k) are arctangent arguments")
pr(f"  4. The sector difference Σ Δ_j involves logarithms of ratios")
pr(f"     of integers 2^j+k, which factor into primes {2,3,5,7,...}")
pr()
pr("  NEXT: Apply carry constraints (masks) to get M_{0,j}^{constrained}")
pr("  and connect the prime-log structure to the Bridge operator of E09.")
pr()
pr("=" * 78)
pr("E27 complete.")
pr("=" * 78)
