#!/usr/bin/env python3
"""
E26: Separable 1D Decomposition — The Path to π

KEY IDENTITIES:
  1. Angular coordinates: u=tan(α), v=tan(β) ⟹ D-odd ⟺ α+β < π/4
  2. Separability: C_{d₀+1} = Σ a[i]·c[d₀+1-i], each term depends on one variable
  3. Bit mass: G_j(x) = ∫₀ˣ c_j(v)dv has closed form

This script:
  Part 1 — Angular coordinate proof and verification
  Part 2 — Full σ(d₀) computation via DFS (reference)
  Part 3 — Separable decomposition: σ(d₀) = Area + ∫b - Σ M_{i,j}
  Part 4 — Weight matrix M_{i,j}: constrained vs unconstrained
  Part 5 — Cumulative ratio analysis → convergence to -π
  Part 6 — Where π enters: the angular/hyperbolic mechanism
"""

import time
from fractions import Fraction
from mpmath import (mp, mpf, log, pi as mpi, nstr, quad, floor as mfloor,
                    atan, tan, cos)

mp.dps = 40

ZERO = mpf(0)
ONE = mpf(1)
TWO = mpf(2)
HALF = mpf(1) / 2
MAX_D = 12
AREA_THRESH = mpf(10) ** (-35)


def pr(s=""):
    print(s, flush=True)


def mpf_area(u_lo, u_hi, v_lo, v_hi, t_lo, t_hi):
    if u_hi <= u_lo or v_hi <= v_lo or t_hi <= t_lo:
        return ZERO
    bps_set = {u_lo, u_hi}
    for t in (t_lo, t_hi):
        for v in (v_lo, v_hi):
            bp = t / (1 + v) - 1
            if u_lo < bp < u_hi:
                bps_set.add(bp)
    bps = sorted(bps_set)
    total = ZERO
    for idx in range(len(bps) - 1):
        ua, ub = bps[idx], bps[idx + 1]
        if ub <= ua:
            continue
        u_mid = (ua + ub) / 2
        vfh = t_hi / (1 + u_mid) - 1
        vfl = t_lo / (1 + u_mid) - 1
        v_up = min(v_hi, vfh)
        v_dn = max(v_lo, vfl)
        if v_up <= v_dn:
            continue
        uh = (vfh < v_hi)
        lh = (vfl > v_lo)
        lr = log((1 + ub) / (1 + ua))
        du = ub - ua
        if uh and lh:
            total += (t_hi - t_lo) * lr
        elif uh:
            total += t_hi * lr - (1 + v_lo) * du
        elif lh:
            total += (1 + v_hi) * du - t_lo * lr
        else:
            total += (v_hi - v_lo) * du
    return total


# ═══════════════════════════════════════════════════════════════
pr("=" * 78)
pr("E26: Separable 1D Decomposition — The Path to pi")
pr("=" * 78)
pr()

# ── Part 1: Angular Coordinates ───────────────────────────────
pr("PART 1: Angular Coordinate Identity")
pr("-" * 50)
pr()
pr("  THEOREM: (1+tan a)(1+tan b) = 2  iff  a + b = pi/4")
pr("  Proof: Expand, collect, use tan addition formula. QED.")
pr()
alpha_s = atan(HALF)
pr(f"  Sector boundary: arctan(1/2) = {nstr(alpha_s, 15)}")
pr(f"  Triangle vertex: pi/4 = {nstr(mpi/4, 15)}")
pr(f"  Crossover: arctan(1/3) = pi/4 - arctan(1/2) = {nstr(atan(ONE/3), 15)}")
pr()

N00_exact = 2 * log(mpf(9) / 8)
N10_exact = 2 * log(mpf(4) / 3) - HALF
pr(f"  N_00 = 2*ln(9/8) = {nstr(N00_exact, 20)}")
pr(f"  N_10 = 2*ln(4/3) - 1/2 = {nstr(N10_exact, 20)}")
pr()

# ── Part 2: Full DFS computation ─────────────────────────────
pr("=" * 78)
pr("PART 2: Full DFS sigma(d0) for both sectors")
pr("-" * 50)
pr()


def compute_full(sector, max_depth):
    a1 = int(sector[0])
    c1 = int(sector[1])
    MN = max_depth + 4
    a = [1, a1] + [0] * MN
    c = [1, c1] + [0] * MN
    b = [1] + [0] * MN

    sigma = [ZERO] * (max_depth + 1)
    area = [ZERO] * (max_depth + 1)
    int_b = [ZERO] * (max_depth + 1)
    int_C = [ZERO] * (max_depth + 1)
    counts = [0] * (max_depth + 1)
    # Per (i,j) decomposition of int_C
    M_constrained = [[None] * (max_depth + 3) for _ in range(max_depth + 3)]
    for ii in range(max_depth + 3):
        for jj in range(max_depth + 3):
            M_constrained[ii][jj] = [ZERO] * (max_depth + 1)

    def C_val(k):
        return sum(a[i] * c[k - i] for i in range(k + 1))

    def process_d0(d0):
        saved_b = b[d0]
        b[d0] = 1
        cross = sum(a[i] * c[d0 + 1 - i] for i in range(1, d0 + 1))
        n = d0 + 1
        pow2n = 1 << n

        for an in range(2):
            a[d0 + 1] = an
            for cn in range(2):
                c[d0 + 1] = cn
                Cv = cn + cross + an
                for bn in range(2):
                    val = 1 + bn - Cv
                    if val == 0:
                        continue
                    b[d0 + 1] = bn
                    t_int = sum(b[j] << (n - j) for j in range(n + 1))
                    tlo = mpf(t_int) / mpf(pow2n)
                    thi = tlo + ONE / mpf(pow2n)
                    if thi > TWO:
                        thi = TWO
                    if tlo >= TWO or thi <= tlo:
                        continue
                    u_int = sum(a[j] << (n - j) for j in range(1, n + 1))
                    ulo = mpf(u_int) / mpf(pow2n)
                    uhi = ulo + ONE / mpf(pow2n)
                    v_int = sum(c[j] << (n - j) for j in range(1, n + 1))
                    vlo = mpf(v_int) / mpf(pow2n)
                    vhi = vlo + ONE / mpf(pow2n)
                    ar = mpf_area(ulo, uhi, vlo, vhi, tlo, thi)
                    if ar > AREA_THRESH:
                        sigma[d0] += val * ar
                        area[d0] += ar
                        int_b[d0] += bn * ar
                        int_C[d0] += Cv * ar
                        counts[d0] += 1
                        # Decompose C contribution by (i,j) pair
                        for i in range(d0 + 2):
                            j = d0 + 1 - i
                            if 0 <= j <= d0 + 1:
                                aij = a[i] * c[j]
                                if aij:
                                    M_constrained[i][j][d0] += ar

        a[d0 + 1] = 0
        c[d0 + 1] = 0
        b[d0 + 1] = 0
        b[d0] = saved_b

    def dfs(depth):
        Cd = C_val(depth)
        if Cd < 0 or Cd > 1:
            return
        b[depth] = Cd
        if Cd == 0 and depth <= max_depth:
            process_d0(depth)
        if depth < max_depth:
            for an in range(2):
                a[depth + 1] = an
                for cn in range(2):
                    c[depth + 1] = cn
                    dfs(depth + 1)
            a[depth + 1] = 0
            c[depth + 1] = 0

    t0 = time.time()
    dfs(1)
    elapsed = time.time() - t0

    max_d = 0
    for d in range(1, max_depth + 1):
        if counts[d] > 0:
            max_d = d

    pr(f"  Sector ({sector[0]},{sector[1]}): {elapsed:.1f}s, max d0={max_d}")
    return {
        'sigma': sigma, 'area': area, 'int_b': int_b,
        'int_C': int_C, 'counts': counts, 'M_constr': M_constrained,
        'max_d': max_d
    }


r00 = compute_full('00', MAX_D)
r10 = compute_full('10', MAX_D)
max_d = min(r00['max_d'], r10['max_d'])
pr()

pr(f"  d0  {'sigma_00':>22s}  {'sigma_10':>22s}  {'Area_00':>14s}  {'Area_10':>14s}")
for d in range(1, max_d + 1):
    pr(f"  {d:2d}  {nstr(r00['sigma'][d], 18):>22s}  "
       f"{nstr(r10['sigma'][d], 18):>22s}  "
       f"{nstr(r00['area'][d], 10):>14s}  "
       f"{nstr(r10['area'][d], 10):>14s}")
pr()

# Cumulative ratio
pr("  Cumulative ratio sigma_10/sigma_00:")
cum00 = ZERO
cum10 = ZERO
for d in range(1, max_d + 1):
    cum00 += r00['sigma'][d]
    cum10 += r10['sigma'][d]
    ratio = cum10 / cum00 if abs(cum00) > ZERO else ZERO
    delta = ratio + mpi
    pr(f"    d0=1..{d:2d}: cum00={nstr(cum00, 15):>20s}  "
       f"cum10={nstr(cum10, 15):>20s}  R={nstr(ratio, 12):>15s}  "
       f"R+pi={nstr(delta, 6):>10s}")
pr()

# ── Part 3: Separable Decomposition ──────────────────────────
pr("=" * 78)
pr("PART 3: Separable Decomposition of int_C")
pr("-" * 50)
pr()
pr("  C_{d0+1} = sum_{i+j=d0+1} a[i]*c[j]")
pr("  Each a[i]*c[j] integral is constrained by valid carry paths")
pr()

for d in range(1, min(max_d, 5) + 1):
    pr(f"  d0={d}: int_C_00 = {nstr(r00['int_C'][d], 15)}, "
       f"int_C_10 = {nstr(r10['int_C'][d], 15)}")
    pr(f"         sigma = Area + int_b - int_C:")
    pr(f"         00: {nstr(r00['area'][d],12)} + {nstr(r00['int_b'][d],12)} "
       f"- {nstr(r00['int_C'][d],12)} = {nstr(r00['sigma'][d],15)}")
    pr(f"         10: {nstr(r10['area'][d],12)} + {nstr(r10['int_b'][d],12)} "
       f"- {nstr(r10['int_C'][d],12)} = {nstr(r10['sigma'][d],15)}")
    pr()

    pr(f"    M_constrained[i][j] (anti-diagonal i+j={d+1}):")
    for i in range(d + 2):
        j = d + 1 - i
        if 0 <= j:
            m00 = r00['M_constr'][i][j][d] if i < len(r00['M_constr']) and j < len(r00['M_constr'][0]) else ZERO
            m10 = r10['M_constr'][i][j][d] if i < len(r10['M_constr']) and j < len(r10['M_constr'][0]) else ZERO
            if abs(m00) > AREA_THRESH or abs(m10) > AREA_THRESH:
                pr(f"      a[{i}]*c[{j}]: M_00={nstr(m00, 12):>16s}  "
                   f"M_10={nstr(m10, 12):>16s}  "
                   f"diff={nstr(m00 - m10, 10):>14s}")
    sum_M00 = sum(r00['M_constr'][i][d+1-i][d] for i in range(d+2) if d+1-i >= 0)
    sum_M10 = sum(r10['M_constr'][i][d+1-i][d] for i in range(d+2) if d+1-i >= 0)
    pr(f"    sum M_00 = {nstr(sum_M00, 15)}, int_C_00 = {nstr(r00['int_C'][d], 15)}")
    pr(f"    sum M_10 = {nstr(sum_M10, 15)}, int_C_10 = {nstr(r10['int_C'][d], 15)}")
    pr()

# ── Part 4: Unconstrained Weight Matrix ──────────────────────
pr("=" * 78)
pr("PART 4: Unconstrained M_{i,j} vs Constrained")
pr("-" * 50)
pr()


def G_j(x, j):
    """Bit mass function: integral of j-th bit from 0 to x."""
    if x <= 0:
        return ZERO
    if x >= 1:
        x = ONE - mpf(10)**(-20)
    scale = mpf(2)**j
    m = int(mfloor(x * scale))
    frac = x * scale - m
    n_odd = m // 2
    result = mpf(n_odd) / scale
    if m % 2 == 1:
        result += frac / scale
    return result


def bit_j(x, j):
    return int(mfloor(x * mpf(2)**j)) % 2


def compute_M_unconstrained(sector, max_ij=8):
    """Compute M_{i,j} = int a_i(u) * G_j(v_max(u)) du using 1D quadrature."""
    if sector == '00':
        u_lo, u_hi = ZERO, HALF
    else:
        u_lo, u_hi = HALF, ONE

    M = {}
    for i in range(max_ij + 1):
        for j in range(1, max_ij + 1):
            breakpoints = [u_lo, u_hi]
            for k in range(2**min(i+1, 8)):
                bp = mpf(k) / mpf(2**i)
                if u_lo < bp < u_hi:
                    breakpoints.append(bp)
            breakpoints = sorted(set(breakpoints))

            def integrand(u, ii=i, jj=j):
                ai = mpf(1) if ii == 0 else mpf(bit_j(u, ii))
                vm = min(HALF, (1 - u) / (1 + u))
                if vm <= 0:
                    return ZERO
                return ai * G_j(vm, jj)

            val = quad(integrand, breakpoints, error=True)[0]
            M[(i, j)] = val
    return M


pr("Computing unconstrained M matrices (with subdivision)...")
t0 = time.time()
M00_unc = compute_M_unconstrained('00', max_ij=MAX_D + 2)
M10_unc = compute_M_unconstrained('10', max_ij=MAX_D + 2)
pr(f"  Done in {time.time()-t0:.1f}s")
pr()

pr("Unconstrained vs Constrained for first depths:")
pr()
for d in range(1, min(max_d, 5) + 1):
    pr(f"  d0={d}, anti-diagonal i+j={d+1}:")
    unc_sum_00 = ZERO
    unc_sum_10 = ZERO
    for i in range(d + 2):
        j = d + 1 - i
        if j < 0:
            continue
        m00_u = M00_unc.get((i, j), ZERO)
        m10_u = M10_unc.get((i, j), ZERO)
        m00_c = r00['M_constr'][i][j][d]
        m10_c = r10['M_constr'][i][j][d]
        unc_sum_00 += m00_u
        unc_sum_10 += m10_u
        if abs(m00_u) > AREA_THRESH or abs(m10_u) > AREA_THRESH:
            pr(f"    a[{i}]c[{j}]:  U_00={nstr(m00_u,10):>14s}  C_00={nstr(m00_c,10):>14s}  "
               f"U_10={nstr(m10_u,10):>14s}  C_10={nstr(m10_c,10):>14s}")
    pr(f"    sum_unc: 00={nstr(unc_sum_00,12)}, 10={nstr(unc_sum_10,12)}")
    pr(f"    sum_con: 00={nstr(r00['int_C'][d],12)}, 10={nstr(r10['int_C'][d],12)}")
    pr()

# ── Part 5: The Ratio Anatomy ────────────────────────────────
pr("=" * 78)
pr("PART 5: Anatomy of the Ratio — What produces -pi?")
pr("-" * 50)
pr()

cum_area_00 = ZERO
cum_area_10 = ZERO
cum_intb_00 = ZERO
cum_intb_10 = ZERO
cum_intC_00 = ZERO
cum_intC_10 = ZERO

for d in range(1, max_d + 1):
    cum_area_00 += r00['area'][d]
    cum_area_10 += r10['area'][d]
    cum_intb_00 += r00['int_b'][d]
    cum_intb_10 += r10['int_b'][d]
    cum_intC_00 += r00['int_C'][d]
    cum_intC_10 += r10['int_C'][d]

sigma_00 = cum_area_00 + cum_intb_00 - cum_intC_00
sigma_10 = cum_area_10 + cum_intb_10 - cum_intC_10

pr(f"  Cumulative (d0=1..{max_d}):")
pr(f"    sigma_00 = {nstr(sigma_00, 20)}")
pr(f"    sigma_10 = {nstr(sigma_10, 20)}")
pr(f"    ratio    = {nstr(sigma_10/sigma_00, 15)}")
pr(f"    -pi      = {nstr(-mpi, 15)}")
pr()

pr("  Component breakdown:")
pr(f"    Area_00 = {nstr(cum_area_00, 18)},   Area_10 = {nstr(cum_area_10, 18)}")
pr(f"    intb_00 = {nstr(cum_intb_00, 18)},   intb_10 = {nstr(cum_intb_10, 18)}")
pr(f"    intC_00 = {nstr(cum_intC_00, 18)},   intC_10 = {nstr(cum_intC_10, 18)}")
pr()

pr("  Key ratios:")
ratio_area = cum_area_10 / cum_area_00
ratio_intb = cum_intb_10 / cum_intb_00
ratio_intC = cum_intC_10 / cum_intC_00 if abs(cum_intC_00) > ZERO else ZERO
pr(f"    Area_10/Area_00 = {nstr(ratio_area, 15)}")
pr(f"    intb_10/intb_00 = {nstr(ratio_intb, 15)}")
pr(f"    intC_10/intC_00 = {nstr(ratio_intC, 15)}")
pr()

pr("  The ratio R = sigma_10/sigma_00 = (A10+b10-C10)/(A00+b00-C00)")
pr("  For R = -pi, the C terms must create the right asymmetry.")
pr()

# ── Part 6: Angular Coordinate Analysis ──────────────────────
pr("=" * 78)
pr("PART 6: Angular Coordinate Analysis")
pr("-" * 50)
pr()

pr("  In angular coordinates alpha, beta (u=tan alpha, v=tan beta):")
pr("    D-odd region = TRIANGLE: alpha + beta < pi/4")
pr("    Sector (0,0): alpha in [0, arctan(1/2))")
pr("    Sector (1,0): alpha in [arctan(1/2), pi/4)")
pr()

pr("  The SECTOR AREAS in angular coordinates:")
pr("  (integrating sec^2(a)*sec^2(b) over the triangle)")
pr()

# Compute sector integrals in angular coordinates
alpha_s = atan(HALF)
pi4 = mpi / 4
alpha_cross = pi4 - alpha_s

pr(f"    alpha_s = arctan(1/2) = {nstr(alpha_s, 15)}")
pr(f"    alpha_cross = pi/4 - alpha_s = arctan(1/3) = {nstr(alpha_cross, 15)}")
pr()

# For sector (0,0): alpha in [0, alpha_s), beta in [0, min(alpha_s, pi/4-alpha))
# Region 1: alpha in [0, alpha_cross): beta up to alpha_s
# Region 2: alpha in [alpha_cross, alpha_s): beta up to pi/4-alpha
pr("  Sector (0,0) decomposes into:")
pr(f"    Region 1: alpha in [0, arctan(1/3)), beta in [0, arctan(1/2))")
pr(f"    Region 2: alpha in [arctan(1/3), arctan(1/2)), beta in [0, pi/4-alpha)")
pr()

pr("  Area in angular coords: integral of sec^2(a)*sec^2(b)")
pr("    = integral_a sec^2(a) * tan(beta_max) da")
pr()

# Symbolic computation of areas
area_00_r1 = quad(lambda a: (1/cos(a))**2 * tan(alpha_s), [0, alpha_cross])
area_00_r2 = quad(lambda a: (1/cos(a))**2 * tan(pi4 - a), [alpha_cross, alpha_s])
pr(f"    Region 1 area = {nstr(area_00_r1, 15)}")
pr(f"    Region 2 area = {nstr(area_00_r2, 15)}")
pr(f"    Total N_00 = {nstr(area_00_r1 + area_00_r2, 15)}")
pr(f"    Exact N_00 = {nstr(N00_exact, 15)}")
pr()

pr("  For sector (1,0): alpha in [alpha_s, pi/4), beta in [0, min(alpha_s, pi/4-alpha))")
pr(f"    Since pi/4-alpha < alpha_s for all alpha > alpha_cross,")
pr(f"    and alpha_s > alpha_cross, the entire sector has beta up to pi/4-alpha.")
pr()
area_10_ang = quad(lambda a: (1/cos(a))**2 * tan(pi4 - a), [alpha_s, pi4])
pr(f"    N_10 (angular) = {nstr(area_10_ang, 15)}")
pr(f"    Exact N_10 = {nstr(N10_exact, 15)}")
pr()

# ── Part 7: The Hyperbolic-Angular Mechanism ─────────────────
pr("=" * 78)
pr("PART 7: How pi Enters Through the Geometry")
pr("-" * 50)
pr()

pr("  KEY INSIGHT: The D-odd boundary alpha+beta = pi/4 is a STRAIGHT LINE")
pr("  in angular coordinates. The sectors are VERTICAL STRIPS.")
pr()
pr("  The beta-integral for ANY function f(beta):")
pr("    integral_0^{pi/4-alpha} f(beta) sec^2(beta) d(beta)")
pr("  has UPPER LIMIT pi/4-alpha that depends linearly on alpha.")
pr()
pr("  When f(beta) = 1: the integral is tan(pi/4-alpha) = (1-tan(a))/(1+tan(a))")
pr("  This is a RATIONAL function of tan(alpha) = u.")
pr()
pr("  When f(beta) = c_j(tan(beta)): the integral involves G_j(tan(pi/4-alpha))")
pr("  evaluated at a point that DEPENDS ON pi/4.")
pr("  The pi enters through this angular boundary!")
pr()

# Demonstrate the mechanism
pr("  DEMONSTRATION: Compute sigma_00(d0=1) in angular coordinates")
pr()

# For d0=1, sector (0,0): t in [3/2, 2)
# In angular coords: (1+u)(1+v) in [3/2, 2) means alpha+beta in [pi/4-arctan(?), pi/4)
# (1+tan a)(1+tan b) = 3/2 when... let s = a+b, then 2sec^2(s/2)... no, not simple.
# But: (1+u)(1+v) = t, with u=tan(a), v=tan(b).

# The product constraint t >= 3/2 means:
# In (u,v) space: v >= 3/(2(1+u)) - 1 = (1-2u)/(2(1+u))
# This is v_low(u) for the d0=1 region

pr("  For d0=1: carry threshold t >= 3/2")
pr("    v_low(u) = (1-2u)/(2(1+u)), v_high(u) = min(1/2, (1-u)/(1+u))")
pr()

# Compute sigma_00(d0=1) via angular 1D quadrature
def sigma_d01_integrand_00(alpha):
    u = tan(alpha)
    sec2a = (1/cos(alpha))**2

    v_lo = max(ZERO, (1 - 2*u) / (2*(1+u)))
    v_hi = min(HALF, (1-u)/(1+u))
    if v_hi <= v_lo:
        return ZERO

    # val = 1 + b2 - a2 - c2 (for sector 0,0 at d0=1)
    # Need to integrate this over v in [v_lo, v_hi]
    # b2 depends on t = (1+u)(1+v): b2=0 if t in [3/2, 7/4), b2=1 if t in [7/4, 2)
    # Boundary: (1+u)(1+v) = 7/4 => v = 7/(4(1+u)) - 1

    v_b2 = mpf(7)/(4*(1+u)) - 1  # boundary for b2

    # a2(u): second bit of u (with a1=0)
    a2_val = bit_j(u, 2) if u < HALF else 0

    # Integrate (1 - a2 - c2(v)) over [v_lo, min(v_b2, v_hi)] (b2=0 region)
    # plus (2 - a2 - c2(v)) over [max(v_b2, v_lo), v_hi] (b2=1 region)

    result = ZERO

    # b2=0 region
    vA = v_lo
    vB = min(v_hi, max(v_lo, v_b2))
    if vB > vA:
        area_piece = vB - vA
        c2_integral = G_j(vB, 2) - G_j(vA, 2)
        result += (1 - a2_val) * area_piece - c2_integral

    # b2=1 region
    vA2 = max(v_lo, v_b2)
    vB2 = v_hi
    if vB2 > vA2:
        area_piece2 = vB2 - vA2
        c2_integral2 = G_j(vB2, 2) - G_j(vA2, 2)
        result += (2 - a2_val) * area_piece2 - c2_integral2

    return result * sec2a


sigma_00_d1_angular = quad(sigma_d01_integrand_00, [0, alpha_s],
                           maxdegree=8)
pr(f"  sigma_00(d0=1) via angular 1D: {nstr(sigma_00_d1_angular, 18)}")
pr(f"  sigma_00(d0=1) via DFS:        {nstr(r00['sigma'][1], 18)}")
pr(f"  Difference: {nstr(abs(sigma_00_d1_angular - r00['sigma'][1]), 10)}")
pr()

# Now sector (1,0) — first contributing depth
first_d10 = 0
for d in range(1, max_d + 1):
    if abs(r10['sigma'][d]) > AREA_THRESH:
        first_d10 = d
        break
pr(f"  First nonzero depth for sector (1,0): d0={first_d10}")
pr(f"    sigma_10(d0={first_d10}) = {nstr(r10['sigma'][first_d10], 18)}")
pr()

# ── Part 8: The Separable Sum Structure ──────────────────────
pr("=" * 78)
pr("PART 8: Cumulative M_{i,j} — Which Terms Drive pi?")
pr("-" * 50)
pr()

pr("  For each depth d0, int_C decomposes as sum of M_constrained[i][j][d0]")
pr("  The ASYMMETRY between sectors is in specific (i,j) pairs.")
pr()

# Cumulate M_constrained across depths
cum_M00 = {}
cum_M10 = {}

for d in range(1, max_d + 1):
    for i in range(d + 2):
        j = d + 1 - i
        if j < 0:
            continue
        key = (i, j)
        if key not in cum_M00:
            cum_M00[key] = ZERO
            cum_M10[key] = ZERO
        cum_M00[key] += r00['M_constr'][i][j][d]
        cum_M10[key] += r10['M_constr'][i][j][d]

pr(f"  Cumulative M (d0=1..{max_d}):")
pr(f"  {'(i,j)':>8s}  {'cum_M_00':>18s}  {'cum_M_10':>18s}  {'diff':>18s}  {'ratio':>12s}")

sorted_keys = sorted(cum_M00.keys())
total_diff = ZERO
for key in sorted_keys:
    m00 = cum_M00[key]
    m10 = cum_M10[key]
    diff = m00 - m10
    total_diff += diff
    ratio_str = nstr(m10/m00, 8) if abs(m00) > AREA_THRESH else "n/a"
    if abs(m00) > AREA_THRESH or abs(m10) > AREA_THRESH:
        pr(f"  {str(key):>8s}  {nstr(m00, 14):>18s}  {nstr(m10, 14):>18s}  "
           f"{nstr(diff, 12):>18s}  {ratio_str:>12s}")
pr()
pr(f"  Total diff: {nstr(total_diff, 15)}")
pr(f"  intC_00 - intC_10 = {nstr(cum_intC_00 - cum_intC_10, 15)}")
pr()

# ── Part 9: Summary ──────────────────────────────────────────
pr("=" * 78)
pr("PART 9: Summary and Next Steps")
pr("-" * 50)
pr()

final_ratio = sigma_10 / sigma_00
pr(f"  R = sigma_10/sigma_00 = {nstr(final_ratio, 15)}")
pr(f"  -pi                   = {nstr(-mpi, 15)}")
pr(f"  R + pi                = {nstr(final_ratio + mpi, 10)}")
pr()

pr("  KEY FINDINGS:")
pr("  1. Angular coordinates transform D-odd to TRIANGLE alpha+beta < pi/4")
pr("     pi is structurally embedded in the integration boundary")
pr("  2. The separable decomposition C = sum a[i]*c[j] allows each term")
pr("     to be computed as a 1D integral, with G_j(v_max(u)) as kernel")
pr("  3. The constraint corrections (carry-free paths) modify the M matrix")
pr("     but preserve the overall structure")
pr("  4. The sector asymmetry arises from DIFFERENT integration domains")
pr("     (alpha < arctan(1/2) vs alpha > arctan(1/2)) interacting with")
pr("     the angular boundary at pi/4")
pr()
pr("  THE PATH TO PROOF:")
pr("  Prove that sum_d0 [sigma_10(d0) + pi*sigma_00(d0)] = 0")
pr("  Using the separable structure, this becomes:")
pr("    sum_d0 sum_{i+j=d0+1} [M10_ij + pi*M00_ij] * weight_ij = 0")
pr("  where weight_ij encodes the val function.")
pr("  The angular coordinate representation makes each M_ij an integral")
pr("  over [0, pi/4] with explicit dependence on pi in the limits.")
pr()
pr("=" * 78)
pr("E26 complete.")
pr("=" * 78)
