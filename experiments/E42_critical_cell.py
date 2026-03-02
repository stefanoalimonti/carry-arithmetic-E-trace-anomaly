#!/usr/bin/env python3
"""
E42: Anatomical Analysis of the Critical Cell at d=1

The single boundary cell that flips R from positive to negative:
  Sector (0,0), depth d₀=1, bits (a₂=1, c₂=1, b₂=0), val = -1

This cell is the "keystone" — removing it makes R > 0 instead of → -π.
We compute its exact geometry, areas, and search for connections to A*.
"""

from fractions import Fraction
from mpmath import (mp, mpf, log, pi as mpi, nstr, atan, quad, sqrt,
                    asin, acos, nprint)

mp.dps = 50

ZERO = mpf(0)
ONE = mpf(1)
TWO = mpf(2)
HALF = ONE / 2


def pr(s=""):
    print(s, flush=True)


def frac_to_mpf(f):
    return mpf(f.numerator) / mpf(f.denominator)


# ═══════════════════════════════════════════════════════════════════
pr("=" * 78)
pr("E42: Anatomy of the Critical Cell at d=1")
pr("=" * 78)
pr()

# ── PART A: Enumerate ALL d=1 cells ──────────────────────────────
pr("PART A: All d=1 cells (sector (0,0))")
pr("-" * 60)
pr()
pr("  DFS at d₀=1, sector (0,0): a₁=0, c₁=0")
pr("  C(1) = a₀·c₁ + a₁·c₀ = 0 → process d₀=1")
pr("  b[1] = 1 (override), cross = a₁·c₁ = 0")
pr("  n = d₀+1 = 2, cells are 1/4-wide in u,v")
pr()

pr("  Sector (1,0): a₁=1, c₁=0")
pr("  C(1) = a₀·c₁ + a₁·c₀ = 0 + 1 = 1 → carry=1, recurse deeper")
pr("  → NO d=1 cells in sector (1,0)")
pr()

cells = []
a1, c1 = 0, 0

for an in range(2):
    for cn in range(2):
        Cv = cn + 0 + an
        for bn in range(2):
            val = 1 + bn - Cv
            if val == 0:
                continue
            b_bits = [1, 1, bn]
            t_int = 4 + 2 + bn
            tlo = Fraction(t_int, 4)
            thi = Fraction(t_int + 1, 4)
            if thi > 2:
                thi = Fraction(2)

            ulo = Fraction(an, 4)
            uhi = Fraction(an + 1, 4) if an == 0 else Fraction(2, 4)
            ulo = Fraction(a1 * 2 + an, 4)
            uhi = ulo + Fraction(1, 4)
            vlo = Fraction(c1 * 2 + cn, 4)
            vhi = vlo + Fraction(1, 4)

            ulo_m = frac_to_mpf(ulo)
            uhi_m = frac_to_mpf(uhi)
            vlo_m = frac_to_mpf(vlo)
            vhi_m = frac_to_mpf(vhi)
            tlo_m = frac_to_mpf(tlo)
            thi_m = frac_to_mpf(thi)

            max_t_in_rect = (ONE + uhi_m) * (ONE + vhi_m)
            min_t_in_rect = (ONE + ulo_m) * (ONE + vlo_m)

            if max_t_in_rect < tlo_m or min_t_in_rect >= thi_m:
                continue

            v_max_at_uhi = (ONE - uhi_m) / (ONE + uhi_m)
            is_bdy = (vhi_m > v_max_at_uhi + mpf('1e-40'))

            cells.append({
                'an': an, 'cn': cn, 'bn': bn, 'val': val, 'Cv': Cv,
                'ulo': ulo, 'uhi': uhi, 'vlo': vlo, 'vhi': vhi,
                'tlo': tlo, 'thi': thi, 'is_bdy': is_bdy,
                'ulo_m': ulo_m, 'uhi_m': uhi_m,
                'vlo_m': vlo_m, 'vhi_m': vhi_m,
                'tlo_m': tlo_m, 'thi_m': thi_m,
            })

pr(f"  Found {len(cells)} cells at d=1:")
pr()

for i, c in enumerate(cells):
    tag = " ← CRITICAL CELL" if c['is_bdy'] else ""
    pr(f"  Cell {i+1}: a₂={c['an']}, c₂={c['cn']}, b₂={c['bn']}")
    pr(f"    val = 1 + {c['bn']} - {c['Cv']} = {c['val']}")
    pr(f"    u ∈ [{c['ulo']}, {c['uhi']}]  =  [{float(c['ulo']):.4f}, {float(c['uhi']):.4f}]")
    pr(f"    v ∈ [{c['vlo']}, {c['vhi']}]  =  [{float(c['vlo']):.4f}, {float(c['vhi']):.4f}]")
    pr(f"    t ∈ [{c['tlo']}, {c['thi']}]  =  [{float(c['tlo']):.4f}, {float(c['thi']):.4f}]")
    pr(f"    Boundary: {c['is_bdy']}{tag}")
    pr()

# ── PART B: Exact area of the critical cell ───────────────────────
pr("=" * 78)
pr("PART B: Exact Area of the Critical Cell")
pr("-" * 60)
pr()

crit = [c for c in cells if c['is_bdy']][0]
pr(f"  Critical cell: u ∈ [{crit['ulo']}, {crit['uhi']}] × v ∈ [{crit['vlo']}, {crit['vhi']}]")
pr(f"  t-strip: [{crit['tlo']}, {crit['thi']})")
pr(f"  val = {crit['val']}")
pr()

pr("  STEP 1: Determine effective region in (u,v)")
pr()
pr("  Lower t-bound: (1+u)(1+v) = 3/2")
pr("    At u=1/4: v = 3/(2·5/4) - 1 = 1/5 < v_lo=1/4")
pr("    At u=1/2: v = 3/(2·3/2) - 1 = 0 < v_lo=1/4")
pr("  → Lower bound is BELOW rectangle. No restriction from t_lo.")
pr()

pr("  Upper t-bound: (1+u)(1+v) = 7/4")
pr("    At u=1/4: v = 7/(4·5/4) - 1 = 2/5")
pr("    At u=1/2: v = 7/(4·3/2) - 1 = 1/6 < v_lo=1/4")
u_exit = Fraction(2, 5)
pr(f"    Curve exits rectangle bottom (v=1/4) at u = {u_exit} = {float(u_exit):.4f}")
pr(f"  → Cell occupies u ∈ [1/4, 2/5], with v_max(u) = (3-4u)/(4(1+u))")
pr()

pr("  STEP 2: Compute flat area ∫∫ du dv")
pr()
pr("  Area = ∫_{1/4}^{2/5} [(3-4u)/(4(1+u)) - 1/4] du")
pr("       = ∫_{1/4}^{2/5} (2-5u)/(4(1+u)) du")
pr("  Decompose: (2-5u)/(1+u) = -5 + 7/(1+u)")
pr("  So: Area = ∫_{1/4}^{2/5} [-5/4 + 7/(4(1+u))] du")
pr("           = [-5u/4 + 7/4·ln(1+u)]_{1/4}^{2/5}")
pr()

val_upper = -mpf(5) * mpf(2) / (4 * 5) + mpf(7) / 4 * log(mpf(7) / 5)
val_lower = -mpf(5) / 16 + mpf(7) / 4 * log(mpf(5) / 4)
area_exact = val_upper - val_lower
pr(f"  At u=2/5: -5·(2/5)/4 + 7/4·ln(7/5) = -1/2 + 7/4·ln(7/5)")
pr(f"  At u=1/4: -5·(1/4)/4 + 7/4·ln(5/4) = -5/16 + 7/4·ln(5/4)")
pr(f"  Area = (-1/2 + 5/16) + 7/4·[ln(7/5) - ln(5/4)]")
pr(f"       = -3/16 + 7/4·ln(28/25)")
pr()

area_symbolic_rat = Fraction(-3, 16)
area_symbolic_log = (Fraction(7, 4), 28, 25)
area_num = mpf(-3) / 16 + mpf(7) / 4 * log(mpf(28) / 25)

pr(f"  EXACT: Area = -3/16 + (7/4)·ln(28/25)")
pr(f"  NUMERIC: Area = {nstr(area_num, 30)}")
print("  CHECK via quad: ", end="", flush=True)
area_quad = quad(
    lambda u: (mpf(3) - 4 * u) / (4 * (1 + u)) - mpf(1) / 4,
    [mpf(1) / 4, mpf(2) / 5]
)
pr(f"{nstr(area_quad, 30)}")
pr(f"  Match: {abs(area_num - area_quad) < mpf('1e-40')}")
pr()

sigma_00_bdy_1 = crit['val'] * area_num
pr(f"  σ₀₀_bdy(d=1) = val × area = ({crit['val']}) × {nstr(area_num, 15)}")
pr(f"                = {nstr(sigma_00_bdy_1, 20)}")
pr()

# ── PART C: Angular area ──────────────────────────────────────────
pr("=" * 78)
pr("PART C: Angular Area of the Critical Cell")
pr("-" * 60)
pr()

pr("  Angular area = ∫∫_{cell} dα dβ = ∫∫_{cell} du dv / ((1+u²)(1+v²))")
pr()

alpha_lo = atan(mpf(1) / 4)
alpha_hi = atan(mpf(1) / 2)
beta_lo = atan(mpf(1) / 4)
beta_hi = atan(mpf(1) / 2)

pr(f"  Rectangle corners in angular coords:")
pr(f"    α ∈ [arctan(1/4), arctan(1/2)] = [{nstr(alpha_lo, 12)}, {nstr(alpha_hi, 12)}]")
pr(f"    β ∈ [arctan(1/4), arctan(1/2)] = [{nstr(beta_lo, 12)}, {nstr(beta_hi, 12)}]")
pr(f"    Δα = {nstr(alpha_hi - alpha_lo, 12)}")
pr(f"    Δβ = {nstr(beta_hi - beta_lo, 12)}")
pr()

pr("  Full rectangle angular area (no t-strip restriction):")
full_rect_ang = (alpha_hi - alpha_lo) * (beta_hi - beta_lo)
pr(f"    Δα · Δβ = {nstr(full_rect_ang, 15)}")
pr()

pr("  Cell angular area (with t-strip [3/2, 7/4)):")
pr("    = ∫_{1/4}^{2/5} [arctan(v_max(u)) - arctan(1/4)] / (1+u²) du")
pr("    where v_max(u) = (3-4u)/(4(1+u))")
pr()

def ang_integrand(u):
    v_max = (mpf(3) - 4 * u) / (4 * (1 + u))
    return (atan(v_max) - atan(mpf(1) / 4)) / (1 + u ** 2)

ang_area = quad(ang_integrand, [mpf(1) / 4, mpf(2) / 5])
pr(f"  Angular area = {nstr(ang_area, 20)}")
pr()

pr("  Comparison:")
pr(f"    Flat area (du dv)    = {nstr(area_num, 15)}")
pr(f"    Angular area (dα dβ) = {nstr(ang_area, 15)}")
pr(f"    Ratio flat/angular   = {nstr(area_num / ang_area, 15)}")
pr(f"    Average Jacobian      = {nstr(area_num / ang_area, 15)}")
pr()

# ── PART D: D-odd boundary interaction ────────────────────────────
pr("=" * 78)
pr("PART D: D-odd Boundary Interaction")
pr("-" * 60)
pr()

pr("  The D-odd boundary is v = (1-u)/(1+u), i.e., α + β = π/4")
pr(f"  v_max(u=1/4) = {nstr((ONE - mpf(1)/4)/(ONE + mpf(1)/4), 8)} = 3/5")
pr(f"  v_max(u=2/5) = {nstr((ONE - mpf(2)/5)/(ONE + mpf(2)/5), 8)} = 3/7")
pr(f"  v_max(u=1/2) = {nstr((ONE - HALF)/(ONE + HALF), 8)} = 1/3")
pr()

pr("  Cell's t-strip upper bound: (1+u)(1+v) = 7/4")
pr(f"  D-odd boundary: (1+u)(1+v) = 2")
pr(f"  Since 7/4 < 2, the cell is ENTIRELY INSIDE Ω.")
pr(f"  The cell doesn't touch the D-odd boundary!")
pr(f"  It's classified as 'boundary' because the RECTANGLE overlaps the boundary.")
pr()

pr("  Rectangle [1/4,1/2]²:")
u_cross_bdy = mpf(1) / 3
pr(f"    D-odd curve enters rectangle at u = 1/3 (where v_max = 1/2 = v_hi)")
pr(f"    D-odd curve exits at u = 1/2 (where v_max = 1/3 > v_lo = 1/4)")
pr()

area_above = mpf(1) / 4 - 2 * log(mpf(9) / 8)
pr(f"  Area above D-odd boundary in rectangle:")
pr(f"    = ∫_{{1/3}}^{{1/2}} [1/2 - (1-u)/(1+u)] du")
pr(f"    = 1/4 - 2·ln(9/8)")
pr(f"    = {nstr(area_above, 20)}")
pr()

area_rect = mpf(1) / 16
area_in_omega = area_rect - area_above
pr(f"  Rectangle total area: 1/16 = {nstr(area_rect, 15)}")
pr(f"  Rectangle area in Ω: {nstr(area_in_omega, 15)} = -3/16 + 2·ln(9/8)")
pr(f"  Rectangle area above Ω: {nstr(area_above, 15)} = 1/4 - 2·ln(9/8)")
pr(f"  Cell area / Rectangle-in-Ω: {nstr(area_num / area_in_omega, 10)}")
pr()

# ── PART E: Why val = -1 (the carry overshoot) ───────────────────
pr("=" * 78)
pr("PART E: Why val = -1 — The Carry Overshoot")
pr("-" * 60)
pr()

pr("  At d₀=1, the cascade terminates when the carry at position 1 is 0.")
pr("  Process: check next bits (a₂, c₂, b₂).")
pr()
pr("  Carry value C_{d₀+1} = c₂ + (cross term) + a₂")
pr("    cross = Σ a_i · c_{d₀+1-i} for i=1..d₀ = a₁·c₁ = 0·0 = 0")
pr("    For a₂=1, c₂=1: Cv = 1 + 0 + 1 = 2")
pr()
pr("  val = 1 + b₂ - Cv = 1 + 0 - 2 = -1")
pr()
pr("  MEANING: When both next bits are 1 (a₂=c₂=1), the cross-product")
pr("  generates a carry of 2 (impossible in binary). This 'overshoot'")
pr("  represents products where the cascade value is NEGATIVE.")
pr("  The carry can't fit in one bit → the product's binary representation")
pr("  is disrupted, creating a DEFECT in the cascade.")
pr()
pr("  The other d=1 cells all have val ≥ +1. Only this cell has val < 0.")
pr("  This single negative-val cell makes σ₀₀ negative, flipping R's sign.")
pr()

# ── PART F: All d=1 areas + contribution to σ ─────────────────────
pr("=" * 78)
pr("PART F: All d=1 cell areas and σ contributions")
pr("-" * 60)
pr()

from mpmath import quad as mquad

total_sigma_00 = ZERO
total_area_00 = ZERO

pr(f"  {'Cell':>6s}  {'(a₂,c₂,b₂)':>12s}  {'val':>4s}  {'area':>18s}  {'σ₀₀ contrib':>18s}  {'bdy':>4s}")
for i, c in enumerate(cells):
    def make_area_integrand(cell):
        ulo_m = cell['ulo_m']
        uhi_m = cell['uhi_m']
        vlo_m = cell['vlo_m']
        vhi_m = cell['vhi_m']
        tlo_m = cell['tlo_m']
        thi_m = cell['thi_m']

        def integrand(u):
            v_upper_t = thi_m / (1 + u) - 1
            v_lower_t = tlo_m / (1 + u) - 1
            v_up = min(vhi_m, v_upper_t)
            v_dn = max(vlo_m, v_lower_t)
            return max(v_up - v_dn, ZERO)

        return integrand

    integrand = make_area_integrand(c)
    bps = [c['ulo_m']]
    for t in [c['tlo_m'], c['thi_m']]:
        for v in [c['vlo_m'], c['vhi_m']]:
            bp = t / (1 + v) - 1
            if c['ulo_m'] < bp < c['uhi_m']:
                bps.append(bp)
    bps.append(c['uhi_m'])
    bps = sorted(set(bps))

    ar = ZERO
    for j in range(len(bps) - 1):
        ar += mquad(integrand, [bps[j], bps[j + 1]])

    sigma_contrib = c['val'] * ar
    total_sigma_00 += sigma_contrib
    total_area_00 += ar

    tag = "YES" if c['is_bdy'] else "no"
    pr(f"  {i+1:6d}  ({c['an']},{c['cn']},{c['bn']}){' ':>5s}  {c['val']:>4d}  {nstr(ar, 12):>18s}  "
       f"{nstr(sigma_contrib, 12):>18s}  {tag:>4s}")

pr()
pr(f"  Total area (d=1): {nstr(total_area_00, 15)}")
pr(f"  Total σ₀₀(d=1):  {nstr(total_sigma_00, 15)}")
pr()

# ── PART G: Relationship to A* and π ─────────────────────────────
pr("=" * 78)
pr("PART G: Search for Connections to A* and π")
pr("-" * 60)
pr()

Astar = (2 + 3 * mpi) / (2 * (1 + mpi))
R0 = (HALF - 4 * log(TWO) + 2 * log(mpf(3))) / (1 - 3 * log(TWO) + log(mpf(3)))
N00 = 2 * log(mpf(9) / 8)
N10 = 2 * log(mpf(4) / 3) - HALF

pr(f"  A* = {nstr(Astar, 20)}")
pr(f"  R₀ = {nstr(R0, 20)}")
pr(f"  N₀₀ (schoolbook) = 2·ln(9/8) = {nstr(N00, 20)}")
pr(f"  N₁₀ (schoolbook) = 2·ln(4/3) - 1/2 = {nstr(N10, 20)}")
pr()

pr("  Critical cell area: A_crit = -3/16 + 7/4·ln(28/25)")
pr(f"    = {nstr(area_num, 20)}")
pr()

pr("  Ratios involving A_crit:")
pr(f"    A_crit / N₀₀ = {nstr(area_num / N00, 15)}")
pr(f"    A_crit / (N₀₀ + N₁₀) = {nstr(area_num / (N00 + N10), 15)}")
pr(f"    A_crit × π = {nstr(area_num * mpi, 15)}")
pr(f"    A_crit × 4 = {nstr(area_num * 4, 15)}")
pr(f"    A_crit / (2ln2-1) = {nstr(area_num / (2*log(TWO)-1), 15)}")
pr()

pr("  σ₀₀_bdy(1) = -A_crit")
pr(f"    -A_crit = {nstr(-area_num, 15)}")
pr(f"    -A_crit / N₀₀ = {nstr(-area_num / N00, 15)}")
pr(f"    -A_crit / σ₀₀_school = {nstr(-area_num / (1 - 3*log(TWO) + log(mpf(3))), 15)}")
pr()

pr("  Area above boundary = 1/4 - 2·ln(9/8)")
area_above_exact = mpf(1) / 4 - 2 * log(mpf(9) / 8)
pr(f"    = {nstr(area_above_exact, 20)}")
pr(f"    area_above / A_crit = {nstr(area_above_exact / area_num, 15)}")
pr(f"    area_above / N₀₀ = {nstr(area_above_exact / N00, 15)}")
pr(f"    area_above × π = {nstr(area_above_exact * mpi, 15)}")
pr()

pr("  Checking A_crit in terms of known constants:")
pr(f"    28·ln(28/25) = {nstr(28 * log(mpf(28)/25), 15)} (vs π = {nstr(mpi, 15)})")
pr(f"    A_crit + 3/16 = 7/4·ln(28/25) = {nstr(mpf(7)/4 * log(mpf(28)/25), 15)}")
pr(f"    4·A_crit + 3/4 = 7·ln(28/25) = {nstr(7 * log(mpf(28)/25), 15)}")
pr()

pr("  Angular area:")
pr(f"    ang_area = {nstr(ang_area, 20)}")
pr(f"    ang_area × π = {nstr(ang_area * mpi, 15)}")
pr(f"    A_crit / ang_area = Avg Jacobian = {nstr(area_num / ang_area, 15)}")
pr()

pr("  Jacobian at cell center (u=3/8, v=3/8):")
u_c = mpf(3) / 8
v_c = mpf(3) / 8
J_center = (1 + u_c ** 2) * (1 + v_c ** 2)
pr(f"    J = (1+u²)(1+v²) = {nstr(J_center, 15)}")
pr(f"    A_crit / ang_area / J_center = {nstr(area_num / ang_area / J_center, 15)}")
pr()

# ── PART H: The perturbation perspective ──────────────────────────
pr("=" * 78)
pr("PART H: Perturbation Structure")
pr("-" * 60)
pr()

pr("  In the Linear Mix: K_eff = (1-A)·K_Markov + (A/2)·I")
pr("  The Identity term (A/2)·I means 'staying in place' — the fraction")
pr("  of probability retained due to geometric constraints.")
pr()

pr("  The critical cell's val = -1 means: products in this region have")
pr("  a NEGATIVE cascade value. In the resolvent framework, this is a")
pr("  'sink' that absorbs probability from the positive-val cells.")
pr()

pr("  The effective weight of the critical cell on σ₀₀:")
sigma_00_total_approx = mpf(-0.00714577538250439)
weight_crit = -area_num / sigma_00_total_approx
pr(f"    σ₀₀_bdy(1) / σ₀₀_total = {nstr(weight_crit, 10)}")
pr(f"    This cell contributes {nstr(weight_crit * 100, 6)}% of total σ₀₀")
pr()

pr("  If we define the 'perturbation strength' as the ratio of the critical")
pr("  cell's contribution to the Markov baseline:")
sigma_00_school = 1 - 3 * log(TWO) + log(mpf(3))
pert_strength = -area_num / sigma_00_school
pr(f"    σ₀₀_bdy(1) / σ₀₀_school = {nstr(pert_strength, 15)}")
pr(f"    A*/2 = {nstr(Astar / 2, 15)}")
pr(f"    1/π = {nstr(1/mpi, 15)}")
pr(f"    Ratio (pert_strength / A*/2) = {nstr(pert_strength / (Astar/2), 15)}")
pr(f"    Ratio (pert_strength × π) = {nstr(pert_strength * mpi, 15)}")
pr()

pr("  Checking the 'weight' of the critical cell per unit of the D-odd domain:")
area_omega = 2 * log(TWO) - 1
area_crit_frac = area_num / area_omega
pr(f"    A_crit / Area(Ω) = {nstr(area_crit_frac, 15)}")
pr(f"    A_crit / Area(Ω) × π = {nstr(area_crit_frac * mpi, 15)}")
pr(f"    A_crit / Area(Ω) × 4 = {nstr(area_crit_frac * 4, 15)}")
pr()

# ── PART I: The d=1 "residual" after schoolbook ──────────────────
pr("=" * 78)
pr("PART I: d=1 residual — cascade vs schoolbook")
pr("-" * 60)
pr()

pr("  The schoolbook value at d=1:")
pr(f"    σ₀₀_school(d=1) = Area of {{u<1/2, v<1/2}} ∩ Ω  weighted by val_school")
pr(f"    N₀₀ = 2·ln(9/8) = {nstr(N00, 15)}  (total schoolbook for sector 00)")
pr()

pr("  The cascade value at d=1:")
pr(f"    σ₀₀_casc(d=1) = {nstr(total_sigma_00, 15)}")
pr(f"    σ₀₀_casc(d=1) - N₀₀ contribution at d=1 ... ")
pr()

pr("  The excess from deeper depths carries the π correction.")
pr("  At d=1, the critical cell creates a 'seed' perturbation of magnitude:")
seed = -area_num
pr(f"    seed = |σ₀₀_bdy(1)| = {nstr(seed, 15)}")
pr(f"    seed / |N₀₀| = {nstr(seed / abs(N00), 12)}")
pr()

pr("  If this seed propagates through the resolvent as:")
pr("    ΔR = seed × (resolvent amplification)")
pr(f"  and ΔR = -π - R₀ = {nstr(-mpi - R0, 15)},")
pr(f"  then the resolvent amplification = ΔR / seed = {nstr((-mpi - R0) / seed, 15)}")
pr()

resolvent_amp = (-mpi - R0) / seed
pr(f"  Resolvent amplification = {nstr(resolvent_amp, 15)}")
pr(f"    Is it 1/(1-A*)? = {nstr(1 / (1 - Astar), 15)}")
pr(f"    Is it -1/A*? = {nstr(-1 / Astar, 15)}")
pr(f"    Is it π? = {nstr(mpi, 15)}")
pr(f"    Ratio to π: {nstr(resolvent_amp / mpi, 12)}")
pr(f"    Ratio to 1/(1-A*): {nstr(resolvent_amp * (1 - Astar), 12)}")
pr()

# ── SUMMARY ──────────────────────────────────────────────────────
pr("=" * 78)
pr("SUMMARY")
pr("=" * 78)
pr()
pr("  THE CRITICAL CELL:")
pr("    Location: [1/4, 1/2] × [1/4, 1/2], t-strip [3/2, 7/4)")
pr("    Effective region: u ∈ [1/4, 2/5], v ∈ [1/4, (3-4u)/(4(1+u))]")
pr(f"    val = -1 (carry overshoot: Cv = 2)")
pr()
pr(f"    Flat area = -3/16 + 7/4·ln(28/25) = {nstr(area_num, 15)}")
pr(f"    Angular area = {nstr(ang_area, 15)}")
pr(f"    Avg Jacobian = {nstr(area_num / ang_area, 10)}")
pr()
pr(f"    σ₀₀_bdy(d=1) = {nstr(sigma_00_bdy_1, 15)}")
pr(f"    This is {nstr(weight_crit * 100, 4)}% of total σ₀₀")
pr()
pr(f"    Area above D-odd boundary in rectangle: 1/4 - 2·ln(9/8) = {nstr(area_above_exact, 10)}")
pr(f"    But the cell itself is ENTIRELY INSIDE Ω (t_hi = 7/4 < 2)")
pr()
pr(f"    Resolvent amplification = ΔR / seed = {nstr(resolvent_amp, 10)}")
pr(f"    This is the factor by which the d=1 perturbation is amplified")
pr(f"    through the infinite cascade to produce ΔR = -π - R₀.")
pr()
