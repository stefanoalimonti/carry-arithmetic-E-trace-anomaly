#!/usr/bin/env python3
"""
E41: Stokes/Angular Decomposition — Where Does π Live?

Decomposes the cancellation series f(d) = σ₁₀(d) + π·σ₀₀(d) into:
  - Interior: cells fully inside D-odd domain Ω
  - Boundary: cells cut by the D-odd curve v = (1-u)/(1+u)
  - Exterior: hypothetical extension beyond Ω (boundary cells, unclipped)

Tests the hypothesis: π resides ENTIRELY in the boundary cells
along the hypotenuse α + β = π/4 in angular coordinates.

Key identity in angular coords:
  arctan((1-u)/(1+u)) = π/4 - arctan(u)
  → boundary cells inject π/4 linearly into the area computation

If the interior contribution f_int(d) sums independently to 0,
then π comes from the boundary alone, and the proof reduces to
a 1D contour integral along α + β = π/4.
"""

import sys
import time
from fractions import Fraction
from mpmath import (mp, mpf, log, pi as mpi, nstr, atan, sqrt)

mp.dps = 50

ZERO = mpf(0)
ONE = mpf(1)
TWO = mpf(2)
HALF = mpf(1) / 2
AREA_THRESH = mpf(10) ** (-45)
MAX_D = 12


def pr(s=""):
    print(s, flush=True)


def mpf_area(u_lo, u_hi, v_lo, v_hi, t_lo, t_hi):
    """Area of rectangle ∩ hyperbolic strip {t_lo ≤ (1+u)(1+v) < t_hi}."""
    if u_hi <= u_lo or v_hi <= v_lo or t_hi <= t_lo:
        return ZERO
    bps_set = {u_lo, u_hi}
    for t in (t_lo, t_hi):
        for v in (v_lo, v_hi):
            denom = 1 + v
            if denom > 0:
                bp = t / denom - 1
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


def area_above_boundary(ulo, uhi, vlo, vhi):
    """Area of rectangle [ulo,uhi]×[vlo,vhi] ABOVE the D-odd curve v=(1-u)/(1+u).
    Returns the area of {(u,v) in rect : (1+u)(1+v) > 2}."""
    v_max_lo = (ONE - ulo) / (ONE + ulo)
    v_max_hi = (ONE - uhi) / (ONE + uhi)

    if v_max_hi >= vhi:
        return ZERO

    if v_max_lo <= vlo:
        return (uhi - ulo) * (vhi - vlo)

    u_cross = (ONE - vhi) / (ONE + vhi)
    u_start = max(ulo, u_cross)

    ext = ZERO
    if u_cross > ulo:
        pass

    ext = (ONE + vhi) * (uhi - u_start) - TWO * log((ONE + uhi) / (ONE + u_start))
    return ext


def compute_decomposed(max_depth):
    """Continuum DFS with interior/boundary/exterior decomposition.
    Boundary = cells whose (u,v) rectangle is intersected by v=(1-u)/(1+u)."""
    results = {}
    for sector in ['00', '10']:
        a1 = int(sector[0])
        c1 = int(sector[1])
        MN = max_depth + 4
        a = [1, a1] + [0] * MN
        c = [1, c1] + [0] * MN
        b = [1] + [0] * MN

        D = max_depth + 1
        sigma = [ZERO] * D
        sigma_int = [ZERO] * D
        sigma_bdy = [ZERO] * D
        sigma_ext = [ZERO] * D
        area_tot = [ZERO] * D
        area_int = [ZERO] * D
        area_bdy = [ZERO] * D
        area_ext = [ZERO] * D
        cnt_tot = [0] * D
        cnt_int = [0] * D
        cnt_bdy = [0] * D

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

                        t_int_val = sum(b[j] << (n - j) for j in range(n + 1))
                        tlo_f = Fraction(t_int_val, pow2n)
                        thi_f = min(tlo_f + Fraction(1, pow2n), Fraction(2))

                        if tlo_f >= 2 or thi_f <= tlo_f:
                            continue

                        u_int_val = sum(a[j] << (n - j) for j in range(1, n + 1))
                        ulo_f = Fraction(u_int_val, pow2n)
                        uhi_f = ulo_f + Fraction(1, pow2n)
                        v_int_val = sum(c[j] << (n - j) for j in range(1, n + 1))
                        vlo_f = Fraction(v_int_val, pow2n)
                        vhi_f = vlo_f + Fraction(1, pow2n)

                        ulo = mpf(ulo_f.numerator) / mpf(ulo_f.denominator)
                        uhi = mpf(uhi_f.numerator) / mpf(uhi_f.denominator)
                        vlo = mpf(vlo_f.numerator) / mpf(vlo_f.denominator)
                        vhi = mpf(vhi_f.numerator) / mpf(vhi_f.denominator)
                        tlo = mpf(tlo_f.numerator) / mpf(tlo_f.denominator)
                        thi = mpf(thi_f.numerator) / mpf(thi_f.denominator)

                        ar = mpf_area(ulo, uhi, vlo, vhi, tlo, thi)
                        if ar > AREA_THRESH:
                            sigma[d0] += val * ar
                            area_tot[d0] += ar
                            cnt_tot[d0] += 1

                            v_max_at_uhi = (ONE - uhi) / (ONE + uhi)
                            is_boundary = (vhi > v_max_at_uhi + AREA_THRESH)

                            if is_boundary:
                                sigma_bdy[d0] += val * ar
                                area_bdy[d0] += ar
                                cnt_bdy[d0] += 1
                                ext = area_above_boundary(ulo, uhi, vlo, vhi)
                                if ext > ZERO:
                                    sigma_ext[d0] += val * ext
                                    area_ext[d0] += ext
                            else:
                                sigma_int[d0] += val * ar
                                area_int[d0] += ar
                                cnt_int[d0] += 1

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
        pr(f"  Sector ({sector[0]},{sector[1]}): {elapsed:.2f}s")

        results[sector] = {
            'sigma': sigma, 'sigma_int': sigma_int,
            'sigma_bdy': sigma_bdy, 'sigma_ext': sigma_ext,
            'area': area_tot, 'area_int': area_int,
            'area_bdy': area_bdy, 'area_ext': area_ext,
            'cnt': cnt_tot, 'cnt_int': cnt_int, 'cnt_bdy': cnt_bdy,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
pr("=" * 78)
pr("E41: Stokes/Angular Decomposition — Where Does π Live?")
pr("=" * 78)
pr()

R0_exact = (HALF - 4 * log(TWO) + 2 * log(mpf(3))) / (1 - 3 * log(TWO) + log(mpf(3)))
Astar = (2 + 3 * mpi) / (2 * (1 + mpi))

pr(f"  R₀ = {nstr(R0_exact, 20)}")
pr(f"  -π = {nstr(-mpi, 20)}")
pr(f"  ΔR = -π - R₀ = {nstr(-mpi - R0_exact, 20)}")
pr(f"  A* = {nstr(Astar, 20)}")
pr()

# ── PART A: DFS with decomposition ────────────────────────────
pr("=" * 78)
pr(f"PART A: Continuum DFS with boundary decomposition (MAX_D={MAX_D})")
pr("-" * 60)
pr()

results = compute_decomposed(MAX_D)
pr()

r00 = results['00']
r10 = results['10']

max_d = 0
for d in range(1, MAX_D + 1):
    if r00['cnt'][d] > 0:
        max_d = d

pr(f"  Max effective depth: {max_d}")
pr()

pr("  Per-depth cell counts:")
pr(f"  {'d':>3s}  {'total':>7s}  {'interior':>8s}  {'boundary':>8s}  {'%bdy':>6s}")
for d in range(1, max_d + 1):
    ct = r00['cnt'][d] + r10['cnt'][d]
    ci = r00['cnt_int'][d] + r10['cnt_int'][d]
    cb = r00['cnt_bdy'][d] + r10['cnt_bdy'][d]
    pct = cb / ct * 100 if ct > 0 else 0
    pr(f"  {d:3d}  {ct:7d}  {ci:8d}  {cb:8d}  {pct:5.1f}%")
pr()

# ── PART B: f(d) decomposition ────────────────────────────────
pr("=" * 78)
pr("PART B: Cancellation f(d) = σ₁₀(d) + π·σ₀₀(d) — Interior vs Boundary")
pr("-" * 60)
pr()

cum_f = ZERO
cum_f_int = ZERO
cum_f_bdy = ZERO

f_vals = []
f_int_vals = []
f_bdy_vals = []

pr(f"  {'d':>3s}  {'f(d)':>20s}  {'f_int(d)':>20s}  {'f_bdy(d)':>20s}  "
   f"{'f_int/f':>10s}  {'f_bdy/f':>10s}")
for d in range(1, max_d + 1):
    fd = r10['sigma'][d] + mpi * r00['sigma'][d]
    fd_int = r10['sigma_int'][d] + mpi * r00['sigma_int'][d]
    fd_bdy = r10['sigma_bdy'][d] + mpi * r00['sigma_bdy'][d]

    cum_f += fd
    cum_f_int += fd_int
    cum_f_bdy += fd_bdy

    f_vals.append(fd)
    f_int_vals.append(fd_int)
    f_bdy_vals.append(fd_bdy)

    ri = nstr(fd_int / fd, 6) if abs(fd) > AREA_THRESH else "—"
    rb = nstr(fd_bdy / fd, 6) if abs(fd) > AREA_THRESH else "—"

    pr(f"  {d:3d}  {nstr(fd, 14):>20s}  {nstr(fd_int, 14):>20s}  {nstr(fd_bdy, 14):>20s}  "
       f"{ri:>10s}  {rb:>10s}")

pr()
pr(f"  Cumulative Σf     = {nstr(cum_f, 20)}  (should → 0)")
pr(f"  Cumulative Σf_int = {nstr(cum_f_int, 20)}")
pr(f"  Cumulative Σf_bdy = {nstr(cum_f_bdy, 20)}")
pr()

if abs(cum_f) > AREA_THRESH:
    pr(f"  Σf_int / Σf = {nstr(cum_f_int / cum_f, 12)}")
    pr(f"  Σf_bdy / Σf = {nstr(cum_f_bdy / cum_f, 12)}")
pr()

# Ratio convergence per-depth
pr("  Per-depth f_int/f ratio (does it → 0 or → const?):")
pr(f"  {'d':>3s}  {'f_int/f':>14s}  {'f_bdy/f':>14s}  {'ratio_d/ratio_{d-1}':>20s}")
prev_ratio = None
for d in range(1, max_d + 1):
    fd = f_vals[d - 1]
    fi = f_int_vals[d - 1]
    if abs(fd) > AREA_THRESH:
        ratio = fi / fd
        trend = nstr(ratio / prev_ratio, 8) if prev_ratio and abs(prev_ratio) > AREA_THRESH else "—"
        prev_ratio = ratio
        pr(f"  {d:3d}  {nstr(ratio, 10):>14s}  {nstr(1 - ratio, 10):>14s}  {trend:>20s}")
pr()

# ── PART C: Sector ratios ────────────────────────────────────
pr("=" * 78)
pr("PART C: Cumulative sector ratios")
pr("-" * 60)
pr()

cum00 = sum(r00['sigma'][d] for d in range(1, max_d + 1))
cum10 = sum(r10['sigma'][d] for d in range(1, max_d + 1))
cum00_int = sum(r00['sigma_int'][d] for d in range(1, max_d + 1))
cum10_int = sum(r10['sigma_int'][d] for d in range(1, max_d + 1))
cum00_bdy = sum(r00['sigma_bdy'][d] for d in range(1, max_d + 1))
cum10_bdy = sum(r10['sigma_bdy'][d] for d in range(1, max_d + 1))
cum00_ext = sum(r00['sigma_ext'][d] for d in range(1, max_d + 1))
cum10_ext = sum(r10['sigma_ext'][d] for d in range(1, max_d + 1))

R_tot = cum10 / cum00
R_int = cum10_int / cum00_int if abs(cum00_int) > AREA_THRESH else ZERO
R_bdy = cum10_bdy / cum00_bdy if abs(cum00_bdy) > AREA_THRESH else ZERO
R_full = (cum10 + cum10_ext) / (cum00 + cum00_ext) if abs(cum00 + cum00_ext) > AREA_THRESH else ZERO

pr(f"  R_total     = {nstr(R_tot, 20)}   (→ -π)")
pr(f"  R_interior  = {nstr(R_int, 20)}")
pr(f"  R_boundary  = {nstr(R_bdy, 20)}")
pr(f"  R_full_sq   = {nstr(R_full, 20)}  (ext beyond Ω)")
pr()
pr(f"  R₀ (school) = {nstr(R0_exact, 20)}")
pr()
pr(f"  R_int + π   = {nstr(R_int + mpi, 15)}")
pr(f"  R_int - R₀  = {nstr(R_int - R0_exact, 15)}")
pr(f"  R_bdy + π   = {nstr(R_bdy + mpi, 15)}")
pr(f"  R_full + π  = {nstr(R_full + mpi, 15)}")
pr(f"  R_full - R₀ = {nstr(R_full - R0_exact, 15)}")
pr()

# Per-depth evolution
pr("  Per-depth cumulative R evolution:")
pr(f"  {'d':>3s}  {'R(≤d)':>16s}  {'R_int(≤d)':>16s}  {'R_bdy(≤d)':>16s}  {'R+π':>14s}")

c00 = ZERO
c10 = ZERO
c00i = ZERO
c10i = ZERO
c00b = ZERO
c10b = ZERO
for d in range(1, max_d + 1):
    c00 += r00['sigma'][d]
    c10 += r10['sigma'][d]
    c00i += r00['sigma_int'][d]
    c10i += r10['sigma_int'][d]
    c00b += r00['sigma_bdy'][d]
    c10b += r10['sigma_bdy'][d]

    Rd = c10 / c00 if abs(c00) > AREA_THRESH else ZERO
    Ri = c10i / c00i if abs(c00i) > AREA_THRESH else ZERO
    Rb = c10b / c00b if abs(c00b) > AREA_THRESH else ZERO

    pr(f"  {d:3d}  {nstr(Rd, 12):>16s}  {nstr(Ri, 12):>16s}  {nstr(Rb, 12):>16s}  "
       f"{nstr(Rd + mpi, 8):>14s}")
pr()

# ── PART D: Area decomposition ───────────────────────────────
pr("=" * 78)
pr("PART D: Area fractions on boundary")
pr("-" * 60)
pr()

pr(f"  {'d':>3s}  {'area_int':>16s}  {'area_bdy':>16s}  {'area_ext':>16s}  "
   f"{'%int':>6s}  {'%bdy':>6s}")
total_area_int = ZERO
total_area_bdy = ZERO
total_area_ext = ZERO
for d in range(1, max_d + 1):
    ai = r00['area_int'][d] + r10['area_int'][d]
    ab = r00['area_bdy'][d] + r10['area_bdy'][d]
    ae = r00['area_ext'][d] + r10['area_ext'][d]
    at = ai + ab
    total_area_int += ai
    total_area_bdy += ab
    total_area_ext += ae
    pi_ = nstr(ai / at * 100, 4) if at > AREA_THRESH else "—"
    pb_ = nstr(ab / at * 100, 4) if at > AREA_THRESH else "—"
    pr(f"  {d:3d}  {nstr(ai, 10):>16s}  {nstr(ab, 10):>16s}  {nstr(ae, 10):>16s}  "
       f"{pi_:>6s}%  {pb_:>6s}%")
pr()
total_at = total_area_int + total_area_bdy
pr(f"  Total area_int = {nstr(total_area_int, 15)}")
pr(f"  Total area_bdy = {nstr(total_area_bdy, 15)}")
pr(f"  Total area_ext = {nstr(total_area_ext, 15)}")
pr(f"  %interior = {nstr(total_area_int / total_at * 100, 6) if total_at > AREA_THRESH else 'N/A'}%")
pr(f"  %boundary = {nstr(total_area_bdy / total_at * 100, 6) if total_at > AREA_THRESH else 'N/A'}%")
pr()

# ── PART E: Angular coordinate analysis ──────────────────────
pr("=" * 78)
pr("PART E: Angular Coordinate Perspective")
pr("-" * 60)
pr()

pr("  In angular coordinates (α = arctan u, β = arctan v):")
pr("    D-odd domain = Triangle T: α + β < π/4")
pr()
pr("  KEY IDENTITY (boundary cells):")
pr("    arctan(v_max(u)) = arctan((1-u)/(1+u)) = π/4 - arctan(u)")
pr("    → Each boundary cell contributes π/4 LINEARLY to angular area")
pr()

alpha_half = atan(HALF)
pr(f"  arctan(1/2) = {nstr(alpha_half, 15)}")
pr(f"  π/4 - arctan(1/2) = arctan(1/3) = {nstr(mpi / 4 - alpha_half, 15)}")
pr()

pr("  Angular areas (in α,β flat measure):")
pr("    Full triangle T: (π/4)²/2 = π²/32")
pr(f"    = {nstr(mpi ** 2 / 32, 15)}")
pr()

pr("  Interior cell angular area = Δα · Δβ  (product of arctan diffs)")
pr("  Boundary cell angular area involves π/4 − α integral:")
pr("    ∫ (π/4 − α − β₀) dα = (π/4−β₀)·Δα − (α₂²−α₁²)/2")
pr("    → EXPLICIT π/4 factor in boundary contribution")
pr()

# Sector angular areas at d=1
pr("  Angular decomposition at d=1:")
for sector in ['00', '10']:
    a1 = int(sector[0])
    c1 = int(sector[1])
    ulo, uhi = mpf(a1) / 2, mpf(a1 + 1) / 2
    vlo, vhi = mpf(c1) / 2, mpf(c1 + 1) / 2

    alpha_lo, alpha_hi = atan(ulo), atan(uhi)
    beta_lo, beta_hi = atan(vlo), atan(vhi)

    v_max_hi = (1 - uhi) / (1 + uhi)
    is_bdy = (vhi > v_max_hi)

    pr(f"    Sector ({sector[0]},{sector[1]}): "
       f"α∈[{nstr(alpha_lo, 6)},{nstr(alpha_hi, 6)}] "
       f"β∈[{nstr(beta_lo, 6)},{nstr(beta_hi, 6)}]")
    pr(f"      α_hi + β_hi = {nstr(alpha_hi + beta_hi, 8)} vs π/4 = {nstr(mpi / 4, 8)}")
    pr(f"      Boundary: {is_bdy}")

    if is_bdy:
        alpha_cross = mpi / 4 - beta_hi
        pr(f"      α_cross (where boundary enters) = {nstr(alpha_cross, 10)}")

        if alpha_cross > alpha_lo:
            rect_part = (beta_hi - beta_lo) * (alpha_cross - alpha_lo)
            tri_alpha_a = alpha_cross
            tri_alpha_b = alpha_hi
        else:
            rect_part = ZERO
            tri_alpha_a = alpha_lo
            tri_alpha_b = alpha_hi

        tri_part = (mpi / 4 - beta_lo) * (tri_alpha_b - tri_alpha_a) - \
                   (tri_alpha_b ** 2 - tri_alpha_a ** 2) / 2
        ang_area_Omega = rect_part + tri_part
        ang_area_full = (beta_hi - beta_lo) * (alpha_hi - alpha_lo)
        ang_area_cut = ang_area_full - ang_area_Omega

        pr(f"      Angular area (in Ω): {nstr(ang_area_Omega, 12)}")
        pr(f"      Angular area (full rect): {nstr(ang_area_full, 12)}")
        pr(f"      Angular area (cut): {nstr(ang_area_cut, 12)}")
        pr(f"      π content: ∝ (π/4−β₀)·Δα − Δ(α²)/2")
    else:
        ang_area = (beta_hi - beta_lo) * (alpha_hi - alpha_lo)
        pr(f"      Angular area: {nstr(ang_area, 12)} (no π involvement)")
    pr()

# ── PART F: Weighted contribution analysis ────────────────────
pr("=" * 78)
pr("PART F: Boundary-weighted f(d) vs depth")
pr("-" * 60)
pr()

pr("  σ₁₀_bdy and σ₀₀_bdy contain the D-odd clipped cells.")
pr("  Their ratio R_bdy determines if boundary alone gives -π.")
pr()
pr(f"  {'d':>3s}  {'σ₀₀_bdy':>16s}  {'σ₁₀_bdy':>16s}  {'σ₀₀_int':>16s}  {'σ₁₀_int':>16s}")
for d in range(1, max_d + 1):
    pr(f"  {d:3d}  {nstr(r00['sigma_bdy'][d], 10):>16s}  {nstr(r10['sigma_bdy'][d], 10):>16s}  "
       f"{nstr(r00['sigma_int'][d], 10):>16s}  {nstr(r10['sigma_int'][d], 10):>16s}")
pr()

# ── PART G: 1D boundary integral ─────────────────────────────
pr("=" * 78)
pr("PART G: 1D Boundary Integral along v = (1-u)/(1+u)")
pr("-" * 60)
pr()

pr("  The Stokes approach: if ΔR = -π can be reduced to a")
pr("  contour integral along the D-odd boundary α + β = π/4,")
pr("  then the 1D integral should encode π directly.")
pr()

pr("  Computing the 'boundary density': for each depth d₀,")
pr("  the reward val along the curve v = (1-u)/(1+u).")
pr()

from mpmath import quad

def boundary_reward(u, d0_max, sector_a1, sector_c1):
    """Compute the reward value along the boundary v=(1-u)/(1+u)
    by enumerating carry patterns at each depth."""
    v = (1 - u) / (1 + u)
    if v < 0 or v >= 1 or u < 0 or u >= 1:
        return ZERO

    MN = d0_max + 4
    a_bits = [1, sector_a1] + [0] * MN
    c_bits = [1, sector_c1] + [0] * MN
    b_bits = [1] + [0] * MN

    pow2 = HALF
    u_rem = u - sector_a1 * HALF
    v_rem = v - sector_c1 * HALF

    total_val = ZERO

    for d0 in range(1, d0_max + 1):
        pow2 /= 2
        a_bit = 1 if u_rem >= pow2 else 0
        c_bit = 1 if v_rem >= pow2 else 0
        a_bits[d0 + 1] = a_bit
        c_bits[d0 + 1] = c_bit
        if a_bit:
            u_rem -= pow2
        if c_bit:
            v_rem -= pow2

        carry = sum(a_bits[i] * c_bits[d0 + 1 - i] for i in range(d0 + 2))
        if carry < 0 or carry > 1:
            break
        b_bits[d0] = carry

        if carry == 0:
            n = d0 + 1
            for bn in range(2):
                Cv_next = sum(a_bits[i] * c_bits[n - i] for i in range(1, n + 1))
                Cv_next += a_bit * c_bit
                val = 1 + bn - Cv_next
                if val != 0:
                    total_val += val * pow2
            break

    return total_val

pr("  Testing boundary integral numerically (low precision)...")
pr()

from mpmath import quadgl

for s_name, s_a1, s_c1 in [("00", 0, 0), ("10", 1, 0)]:
    u_lo = mpf(s_a1) / 2
    u_hi = mpf(s_a1 + 1) / 2

    pr(f"  Sector ({s_name}): u ∈ [{nstr(u_lo, 4)}, {nstr(u_hi, 4)}]")

    for d_max_test in [4, 6, 8]:
        def integrand(u):
            return boundary_reward(u, d_max_test, s_a1, s_c1)

        try:
            val = quad(integrand, [u_lo, u_hi], method='tanh-sinh')
            pr(f"    d_max={d_max_test}: ∫ boundary_reward du = {nstr(val, 15)}")
        except Exception as e:
            pr(f"    d_max={d_max_test}: integration failed ({e})")
    pr()

# ── SUMMARY ──────────────────────────────────────────────────
pr("=" * 78)
pr("SUMMARY AND CONCLUSIONS")
pr("=" * 78)
pr()

pr(f"  R(d≤{max_d}) = {nstr(R_tot, 16)}  (gap to -π: {nstr(R_tot + mpi, 8)})")
pr()

pr("  DECOMPOSITION:")
pr(f"    σ₀₀ = {nstr(cum00, 16)}  (int: {nstr(cum00_int / cum00 * 100, 4)}%, "
   f"bdy: {nstr(cum00_bdy / cum00 * 100, 4)}%)")
pr(f"    σ₁₀ = {nstr(cum10, 16)}  (int: {nstr(cum10_int / cum10 * 100, 4)}%, "
   f"bdy: {nstr(cum10_bdy / cum10 * 100, 4)}%)")
pr()

pr("  CANCELLATION f(d) = σ₁₀ + π·σ₀₀:")
pr(f"    Σf_int = {nstr(cum_f_int, 16)}")
pr(f"    Σf_bdy = {nstr(cum_f_bdy, 16)}")
pr(f"    Σf_tot = {nstr(cum_f, 16)}")
pr()

if abs(cum_f) > AREA_THRESH and abs(cum_f_bdy) > AREA_THRESH:
    pr(f"    FRACTION from interior: {nstr(cum_f_int / cum_f * 100, 6)}%")
    pr(f"    FRACTION from boundary: {nstr(cum_f_bdy / cum_f * 100, 6)}%")
    pr()

pr("  RATIOS:")
pr(f"    R_total    = {nstr(R_tot, 16)}")
pr(f"    R_interior = {nstr(R_int, 16)}")
pr(f"    R_boundary = {nstr(R_bdy, 16)}")
pr(f"    R_full_sq  = {nstr(R_full, 16)}")
pr(f"    R₀ (school)= {nstr(R0_exact, 16)}")
pr()

pr("  HYPOTHESIS TEST: 'π lives on the boundary'")
if abs(cum_f_int) < abs(cum_f) * mpf('0.01'):
    pr("    ✓ f_int is negligible → π is ENTIRELY in boundary cells")
elif abs(cum_f_int / cum_f - 1) < mpf('0.01'):
    pr("    ✗ f_int ≈ f → π is ENTIRELY in interior cells (unexpected)")
else:
    pr(f"    ~ π is SHARED between interior ({nstr(cum_f_int / cum_f * 100, 4)}%) "
       f"and boundary ({nstr(cum_f_bdy / cum_f * 100, 4)}%)")
    pr("    The simple 'boundary-only' hypothesis is not supported.")
    pr("    However, the ANGULAR COORDINATE analysis shows that boundary cells")
    pr("    inject π/4 explicitly via arctan((1-u)/(1+u)) = π/4 - arctan(u),")
    pr("    while interior cells involve only arctan(rational) differences.")
pr()
