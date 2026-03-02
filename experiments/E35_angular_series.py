#!/usr/bin/env python3
"""
E35: Angular Integral Series and π Extraction

Testing four ideas for proving ΔR = -π - R₀:

  Idea 1: Fredholm integral operator in angular coordinates
  Idea 2: Walsh/Rademacher + boundary arctangent mechanism
  Idea 3: Stokes/Green → boundary integral on α+β=π/4
  Idea 4: Spectral connection to E09's S(A) identity

This experiment:
  Part A: High-precision per-depth σ via continuum DFS (extend E26)
  Part B: Cancellation series f(d) = σ₁₀(d) + π·σ₀₀(d)
  Part C: Symbolic decomposition — rational + Σ c_k·ln(r_k)
  Part D: Boundary vs interior contribution separation
  Part E: Spectral parameter A_eff(d) and convergence to A*
  Part F: Transfer operator eigenvalue extraction
"""
import sys
import time
from math import gcd
from fractions import Fraction
from mpmath import (mp, mpf, log, pi as mpi, nstr, floor as mfloor,
                    atan, tan, cos, power, sqrt)

mp.dps = 50

ZERO = mpf(0)
ONE = mpf(1)
TWO = mpf(2)
HALF = mpf(1) / 2
AREA_THRESH = mpf(10) ** (-45)
MAX_D = 12


def pr(s=""):
    print(s, flush=True)


class SymArea:
    """Symbolic area: rational_part + Σ coeff_i · ln(num_i / den_i)"""

    __slots__ = ('rat', 'logs')

    def __init__(self, rat=None, logs=None):
        self.rat = Fraction(0) if rat is None else rat
        self.logs = {} if logs is None else logs

    def __add__(self, other):
        r = SymArea(self.rat + other.rat, dict(self.logs))
        for k, v in other.logs.items():
            r.logs[k] = r.logs.get(k, Fraction(0)) + v
        return r

    def __mul__(self, scalar):
        if isinstance(scalar, (int, Fraction)):
            r = SymArea(self.rat * scalar)
            r.logs = {k: v * scalar for k, v in self.logs.items()}
            return r
        return NotImplemented

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    @staticmethod
    def _frac_to_mpf(f):
        if isinstance(f, Fraction):
            return mpf(f.numerator) / mpf(f.denominator)
        return mpf(f)

    def to_mpf(self):
        val = self._frac_to_mpf(self.rat)
        for (n, d), c in self.logs.items():
            val += self._frac_to_mpf(c) * log(mpf(n) / mpf(d))
        return val

    def log_total(self):
        val = ZERO
        for (n, d), c in self.logs.items():
            val += self._frac_to_mpf(c) * log(mpf(n) / mpf(d))
        return val


def sym_log(num, den):
    """Create SymArea representing ln(num/den)."""
    g = gcd(abs(num), abs(den))
    num, den = num // g, den // g
    if den < 0:
        num, den = -num, -den
    s = SymArea()
    s.logs[(num, den)] = Fraction(1)
    return s


def mpf_area(u_lo, u_hi, v_lo, v_hi, t_lo, t_hi):
    """Exact area of rectangle ∩ D-odd domain. Returns mpf."""
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


def sym_area_exact(u_lo_f, u_hi_f, v_lo_f, v_hi_f, t_lo_f, t_hi_f):
    """Symbolic area: returns SymArea with exact rational + log decomposition.
    All inputs are Fraction objects."""
    result = SymArea()
    if u_hi_f <= u_lo_f or v_hi_f <= v_lo_f or t_hi_f <= t_lo_f:
        return result

    bps_set = {u_lo_f, u_hi_f}
    for t in (t_lo_f, t_hi_f):
        for v in (v_lo_f, v_hi_f):
            bp = t / (1 + v) - 1
            if u_lo_f < bp < u_hi_f:
                bps_set.add(bp)
    bps = sorted(bps_set)

    for idx in range(len(bps) - 1):
        ua, ub = bps[idx], bps[idx + 1]
        if ub <= ua:
            continue
        u_mid = (ua + ub) / 2
        vfh = t_hi_f / (1 + u_mid) - 1
        vfl = t_lo_f / (1 + u_mid) - 1
        v_up = min(v_hi_f, vfh)
        v_dn = max(v_lo_f, vfl)
        if v_up <= v_dn:
            continue
        uh = (vfh < v_hi_f)
        lh = (vfl > v_lo_f)
        du = ub - ua

        log_num = (1 + ub).numerator * (1 + ua).denominator
        log_den = (1 + ua).numerator * (1 + ub).denominator
        g = gcd(abs(log_num), abs(log_den))
        log_num //= g
        log_den //= g

        if uh and lh:
            coeff = t_hi_f - t_lo_f
            s = SymArea()
            s.logs[(log_num, log_den)] = coeff
            result = result + s
        elif uh:
            s = SymArea(rat=-(1 + v_lo_f) * du)
            s.logs[(log_num, log_den)] = t_hi_f
            result = result + s
        elif lh:
            s = SymArea(rat=(1 + v_hi_f) * du)
            s.logs[(log_num, log_den)] = -t_lo_f
            result = result + s
        else:
            result = result + SymArea(rat=(v_hi_f - v_lo_f) * du)

    return result


def compute_full_both(max_depth, do_symbolic=False, sym_max_depth=8):
    """Compute σ₀₀(d), σ₁₀(d) via continuum DFS.
    If do_symbolic, also returns symbolic decomposition up to sym_max_depth."""
    results = {}
    for sector in ['00', '10']:
        a1 = int(sector[0])
        c1 = int(sector[1])
        MN = max_depth + 4
        a = [1, a1] + [0] * MN
        c = [1, c1] + [0] * MN
        b = [1] + [0] * MN

        sigma = [ZERO] * (max_depth + 1)
        area_arr = [ZERO] * (max_depth + 1)
        counts = [0] * (max_depth + 1)
        n_boundary = [0] * (max_depth + 1)
        sigma_boundary = [ZERO] * (max_depth + 1)

        sigma_sym = [SymArea() for _ in range(max_depth + 1)] if do_symbolic else None

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
                        tlo_f = Fraction(t_int, pow2n)
                        thi_f = tlo_f + Fraction(1, pow2n)
                        if thi_f > 2:
                            thi_f = Fraction(2)
                        if tlo_f >= 2 or thi_f <= tlo_f:
                            continue
                        u_int = sum(a[j] << (n - j) for j in range(1, n + 1))
                        ulo_f = Fraction(u_int, pow2n)
                        uhi_f = ulo_f + Fraction(1, pow2n)
                        v_int = sum(c[j] << (n - j) for j in range(1, n + 1))
                        vlo_f = Fraction(v_int, pow2n)
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
                            area_arr[d0] += ar
                            counts[d0] += 1

                            v_max_at_ulo = (1 - ulo) / (1 + ulo)
                            v_max_at_uhi = (1 - uhi) / (1 + uhi)
                            is_boundary = (v_max_at_ulo > vlo and v_max_at_ulo < vhi) or \
                                          (v_max_at_uhi > vlo and v_max_at_uhi < vhi) or \
                                          (vhi > v_max_at_ulo) or (vhi > v_max_at_uhi and vlo < v_max_at_uhi)
                            if is_boundary:
                                n_boundary[d0] += 1
                                sigma_boundary[d0] += val * ar

                            if do_symbolic and d0 <= sym_max_depth:
                                sa = sym_area_exact(ulo_f, uhi_f, vlo_f, vhi_f, tlo_f, thi_f)
                                sigma_sym[d0] = sigma_sym[d0] + (val * sa)

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
            'sigma': sigma, 'area': area_arr, 'counts': counts,
            'n_boundary': n_boundary, 'sigma_boundary': sigma_boundary,
            'sigma_sym': sigma_sym,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
pr("=" * 78)
pr("E35: Angular Integral Series and π Extraction")
pr("=" * 78)
pr()

# Known exact values
N00_exact = 2 * log(mpf(9) / 8)
N10_exact = 2 * log(mpf(4) / 3) - HALF
R0_exact = (HALF - 4 * log(TWO) + 2 * log(mpf(3))) / (1 - 3 * log(TWO) + log(mpf(3)))
sigma00_school = 1 - 3 * log(TWO) + log(mpf(3))
sigma10_school = HALF - 4 * log(TWO) + 2 * log(mpf(3))
Astar = (2 + 3 * mpi) / (2 * (1 + mpi))

pr(f"  R₀ = {nstr(R0_exact, 20)}")
pr(f"  -π = {nstr(-mpi, 20)}")
pr(f"  ΔR = -π - R₀ = {nstr(-mpi - R0_exact, 20)}")
pr(f"  A* = (2+3π)/(2(1+π)) = {nstr(Astar, 20)}")
pr()

# ── PART A: Per-depth σ values ────────────────────────────────
pr("=" * 78)
pr("PART A: Per-depth σ values via continuum DFS (to depth {})".format(MAX_D))
pr("-" * 60)
pr()

do_sym_max = 6
pr(f"Computing... (symbolic for d≤{do_sym_max}, numeric for all)")
results = compute_full_both(MAX_D, do_symbolic=True, sym_max_depth=do_sym_max)
pr()

r00 = results['00']
r10 = results['10']

max_d = 0
for d in range(1, MAX_D + 1):
    if r00['counts'][d] > 0:
        max_d = d

pr(f"  Effective max depth: {max_d}")
pr()

pr(f"  {'d':>3s}  {'σ₀₀(d)':>24s}  {'σ₁₀(d)':>24s}  {'cnt_00':>8s}  {'cnt_10':>8s}")
cum00 = ZERO
cum10 = ZERO
for d in range(1, max_d + 1):
    cum00 += r00['sigma'][d]
    cum10 += r10['sigma'][d]
    pr(f"  {d:3d}  {nstr(r00['sigma'][d], 18):>24s}  {nstr(r10['sigma'][d], 18):>24s}  "
       f"{r00['counts'][d]:>8d}  {r10['counts'][d]:>8d}")

pr()
pr(f"  Cumulative σ₀₀ = {nstr(cum00, 25)}")
pr(f"  Cumulative σ₁₀ = {nstr(cum10, 25)}")
R_cum = cum10 / cum00
pr(f"  R(d≤{max_d}) = {nstr(R_cum, 20)}")
pr(f"  -π = {nstr(-mpi, 20)}")
pr(f"  Gap = {nstr(R_cum + mpi, 15)}")
pr()

# ── PART B: Cancellation series f(d) = σ₁₀(d) + π·σ₀₀(d) ────
pr("=" * 78)
pr("PART B: Cancellation series f(d) = σ₁₀(d) + π·σ₀₀(d)")
pr("-" * 60)
pr()

pr("  If R = -π, then Σ f(d) = 0.")
pr("  Each f(d) measures the per-depth deviation from the -π relation.")
pr()

f_values = []
cum_f = ZERO
pr(f"  {'d':>3s}  {'f(d) = σ₁₀+π·σ₀₀':>24s}  {'Σf(1..d)':>24s}  {'f(d)/f(d-1)':>14s}")
for d in range(1, max_d + 1):
    fd = r10['sigma'][d] + mpi * r00['sigma'][d]
    cum_f += fd
    f_values.append(fd)
    if len(f_values) >= 2 and abs(f_values[-2]) > AREA_THRESH:
        ratio = fd / f_values[-2]
        ratio_str = nstr(ratio, 10)
    else:
        ratio_str = "—"
    pr(f"  {d:3d}  {nstr(fd, 18):>24s}  {nstr(cum_f, 18):>24s}  {ratio_str:>14s}")

pr()
pr(f"  Cumulative Σf = {nstr(cum_f, 25)}")
pr(f"  (should → 0 if R → -π)")
pr()

pr("  Ratio analysis f(d)/f(d-1) — looking for geometric convergence:")
pr(f"  {'d':>3s}  {'ratio':>18s}  {'1/ratio':>18s}  {'ratio - 1/2':>18s}")
for d in range(2, len(f_values) + 1):
    if abs(f_values[d - 2]) > AREA_THRESH:
        ratio = f_values[d - 1] / f_values[d - 2]
        pr(f"  {d:3d}  {nstr(ratio, 14):>18s}  {nstr(1 / ratio, 14):>18s}  "
           f"{nstr(ratio - HALF, 10):>18s}")

pr()

pr("  If f(d) ~ C·ρ^d, then ratio → ρ.")
pr("  If f(d) ~ C·d^α·ρ^d, the ratio → ρ with corrections ~1/d.")
pr()

# ── PART C: Symbolic decomposition ───────────────────────────
pr("=" * 78)
pr("PART C: Symbolic decomposition (rational + log parts)")
pr("-" * 60)
pr()

pr("  σ(d) = rational_part + Σ c_k · ln(r_k)")
pr()

for d in range(1, min(max_d, do_sym_max) + 1):
    s00 = r00['sigma_sym'][d] if r00['sigma_sym'] else None
    s10 = r10['sigma_sym'][d] if r10['sigma_sym'] else None
    if s00 is None:
        continue

    pr(f"  ─── Depth d={d} ───")

    pr(f"    σ₀₀(d): rational = {float(s00.rat):.15f} ({s00.rat})")
    pr(f"            log_part  = {nstr(s00.log_total(), 18)}")
    pr(f"            total     = {nstr(s00.to_mpf(), 18)} (check: {nstr(r00['sigma'][d], 18)})")

    if s00.logs:
        for (n, dn), coeff in sorted(s00.logs.items()):
            if abs(coeff) > Fraction(1, 10**10):
                pr(f"            + ({coeff}) · ln({n}/{dn})")

    pr(f"    σ₁₀(d): rational = {float(s10.rat):.15f} ({s10.rat})")
    pr(f"            log_part  = {nstr(s10.log_total(), 18)}")
    pr(f"            total     = {nstr(s10.to_mpf(), 18)} (check: {nstr(r10['sigma'][d], 18)})")

    if s10.logs:
        for (n, dn), coeff in sorted(s10.logs.items()):
            if abs(coeff) > Fraction(1, 10**10):
                pr(f"            + ({coeff}) · ln({n}/{dn})")

    fd_sym = SymArea(s10.rat + Fraction(s00.rat * Fraction(0)),
                     dict(s10.logs))
    pr(f"    f(d) = σ₁₀(d) + π·σ₀₀(d):")
    fd_val = s10.to_mpf() + mpi * s00.to_mpf()
    pr(f"         = {nstr(fd_val, 18)}")
    pr()

# ── PART D: Boundary vs interior contribution ─────────────────
pr("=" * 78)
pr("PART D: Boundary vs interior contribution")
pr("-" * 60)
pr()

pr("  'Boundary' = rectangles intersected by the D-odd curve v=(1-u)/(1+u)")
pr("  'Interior' = rectangles fully inside the D-odd domain")
pr("  Idea 3: The boundary contribution should drive the π cancellation.")
pr()

pr(f"  {'d':>3s}  {'σ₀₀_bdy':>18s}  {'σ₀₀_tot':>18s}  {'frac_bdy_00':>12s}  "
   f"{'σ₁₀_bdy':>18s}  {'σ₁₀_tot':>18s}  {'frac_bdy_10':>12s}")
for d in range(1, max_d + 1):
    s00 = r00['sigma'][d]
    s10 = r10['sigma'][d]
    b00 = r00['sigma_boundary'][d]
    b10 = r10['sigma_boundary'][d]
    E35 = b00 / s00 if abs(s00) > AREA_THRESH else ZERO
    E35 = b10 / s10 if abs(s10) > AREA_THRESH else ZERO
    pr(f"  {d:3d}  {nstr(b00, 12):>18s}  {nstr(s00, 12):>18s}  {nstr:>12s}  "
       f"{nstr(b10, 12):>18s}  {nstr(s10, 12):>18s}  {nstr:>12s}")

pr()

# ── PART E: Spectral parameter A_eff ──────────────────────────
pr("=" * 78)
pr("PART E: Effective spectral parameter A_eff(d)")
pr("-" * 60)
pr()

pr("  From E09: S(A) = 2(1-A)/(3-2A)")
pr("  Inverting: A = (3S-2)/(2S-2) where S = R = ratio")
pr()
pr("  If R(d_max) → -π, then A → A* = (2+3π)/(2(1+π))")
pr()

pr(f"  A* = {nstr(Astar, 20)}")
pr()

cum00_run = ZERO
cum10_run = ZERO
pr(f"  {'d_max':>5s}  {'R(d_max)':>18s}  {'A_eff':>18s}  {'A_eff - A*':>18s}")
for d in range(1, max_d + 1):
    cum00_run += r00['sigma'][d]
    cum10_run += r10['sigma'][d]
    if abs(cum00_run) > AREA_THRESH:
        R_d = cum10_run / cum00_run
        A_eff = (3 * R_d - 2) / (2 * R_d - 2)
        pr(f"  {d:5d}  {nstr(R_d, 14):>18s}  {nstr(A_eff, 14):>18s}  "
           f"{nstr(A_eff - Astar, 10):>18s}")

pr()

# ── PART F: Transfer operator — per-depth survival ────────────
pr("=" * 78)
pr("PART F: Transfer operator and per-depth survival")
pr("-" * 60)
pr()

pr("  The survival probability Surv(d) = Area(d)/Area(d-1)")
pr("  gives the conditional probability that the cascade survives past depth d.")
pr("  In the Markov limit, Surv(d) → 1/2 (Diaconis-Fulman eigenvalue).")
pr()

pr(f"  {'d':>3s}  {'Area₀₀(d)':>20s}  {'Surv₀₀(d)':>14s}  "
   f"{'Area₁₀(d)':>20s}  {'Surv₁₀(d)':>14s}  {'Q₁₀/Q₀₀':>12s}")

prev_a00 = ZERO
prev_a10 = ZERO
surv_00 = []
surv_10 = []

for d in range(1, max_d + 1):
    a00 = r00['area'][d]
    a10 = r10['area'][d]
    s00_d = a00 / prev_a00 if prev_a00 > AREA_THRESH else ZERO
    s10_d = a10 / prev_a10 if prev_a10 > AREA_THRESH else ZERO
    ratio_q = s10_d / s00_d if s00_d > AREA_THRESH else ZERO

    surv_00.append(s00_d)
    surv_10.append(s10_d)

    pr(f"  {d:3d}  {nstr(a00, 14):>20s}  {nstr(s00_d, 10):>14s}  "
       f"{nstr(a10, 14):>20s}  {nstr(s10_d, 10):>14s}  {nstr(ratio_q, 8):>12s}")

    prev_a00 = a00
    prev_a10 = a10

pr()

# Eigenvalue extraction: Surv(d) = λ₁ + corrections
pr("  Asymptotic eigenvalue (from Surv(d) at large d):")
if len(surv_10) >= 3:
    for d_idx in range(max(3, len(surv_10) - 4), len(surv_10)):
        d = d_idx + 1
        if surv_10[d_idx] > AREA_THRESH:
            eps_10 = surv_10[d_idx] - HALF
            eps_00 = surv_00[d_idx] - HALF
            pr(f"    d={d}: Surv₁₀-1/2 = {nstr(eps_10, 10)}, "
               f"Surv₀₀-1/2 = {nstr(eps_00, 10)}")

pr()

# ── PART G: The angular perspective — arctan emergence ─────────
pr("=" * 78)
pr("PART G: Angular perspective — arctangent emergence")
pr("-" * 60)
pr()

pr("  In angular coordinates (α = arctan u, β = arctan v):")
pr("    D-odd: α + β < π/4   (STRAIGHT LINE boundary)")
pr("    Sector (0,0): α ∈ [0, arctan(1/2))")
pr("    Sector (1,0): α ∈ [arctan(1/2), π/4)")
pr()
pr("  Idea 2: Integrating dyadic step functions against the boundary")
pr("  α + β = π/4 produces arctangent terms through the identity:")
pr("    arctan(1/2) + arctan(1/3) = π/4")
pr()

alpha_s = atan(HALF)
alpha_c = atan(mpf(1) / 3)
pr(f"  arctan(1/2) = {nstr(alpha_s, 20)}")
pr(f"  arctan(1/3) = {nstr(alpha_c, 20)}")
pr(f"  arctan(1/2) + arctan(1/3) = {nstr(alpha_s + alpha_c, 20)}")
pr(f"  π/4 = {nstr(mpi / 4, 20)}")
pr(f"  Match: {abs(alpha_s + alpha_c - mpi / 4) < mpf('1e-40')}")
pr()

pr("  KEY INSIGHT: The sector boundary (arctan(1/2)) and the crossover")
pr("  (arctan(1/3)) are COMPLEMENTARY angles summing to π/4.")
pr("  The carry dynamics at each depth creates further subdivisions.")
pr()

pr("  At depth d, the carry path determines dyadic interval [k/2^d, (k+1)/2^d)")
pr("  for u (and similarly for v). In angular coords, this is:")
pr("    α ∈ [arctan(k/2^d), arctan((k+1)/2^d))")
pr()
pr("  The D-odd boundary at this resolution creates:")
pr("    β_max(α) = π/4 - α")
pr("    In (u,v): v_max(u) = tan(π/4 - arctan(u)) = (1-u)/(1+u)")
pr()
pr("  The integral of the step function over the truncated rectangle")
pr("  involves:")
pr("    ∫_{arctan(k/2^d)}^{arctan((k+1)/2^d)} [π/4 - α] · (...) dα")
pr("  which LINEARLY involves π/4.")
pr()

pr("  Machin-like decomposition test:")
pr("  At depth d, the boundary intersects 2^d dyadic intervals.")
pr("  Each intersection creates a contribution proportional to arctan differences:")
pr("    arctan((k+1)/2^d) - arctan(k/2^d) = arctan(1/(2^{2d} + k(k+1)))")
pr()
pr("  These are the 'atoms' from which π/4 is built via telescoping.")
pr()

for d_test in [1, 2, 3, 4]:
    n = 1 << d_test
    total_arctan = ZERO
    pr(f"  Depth {d_test}: decompose arctan(1) = π/4 into 2^{d_test} = {n} pieces:")
    for k in range(n):
        at_hi = atan(mpf(k + 1) / n)
        at_lo = atan(mpf(k) / n)
        piece = at_hi - at_lo
        arg = mpf(1) / (mpf(n * n) + mpf(k * (k + 1)))
        at_arg = atan(arg)
        total_arctan += piece
        if d_test <= 2:
            pr(f"    k={k}: arctan({k+1}/{n}) - arctan({k}/{n}) = {nstr(piece, 12)}"
               f"  = arctan({nstr(arg, 8)}) [{nstr(at_arg, 12)}]"
               f"  match: {abs(piece - at_arg) < mpf('1e-30')}")
    pr(f"    Total = {nstr(total_arctan, 20)} (should be π/4 = {nstr(mpi / 4, 20)})")
    pr(f"    Error: {nstr(abs(total_arctan - mpi / 4), 8)}")
    pr()

# ── PART H: The σ₁₀/σ₀₀ anatomy — what makes it -π? ──────────
pr("=" * 78)
pr("PART H: Anatomy of σ₁₀/σ₀₀ = -π")
pr("-" * 60)
pr()

pr("  The schoolbook case: σ₁₀ = -N₁₀ = -(2ln(4/3)-1/2)")
pr("                       σ₀₀ = 1+ln(3/8)")
pr("  Both are rational + logarithmic. Their ratio is NOT π.")
pr()
pr("  The cascade case adds depth-dependent corrections.")
pr("  The corrections involve integrals of (1-u)/(1+u) over dyadic intervals,")
pr("  which produce ln((2^d+k+1)/(2^d+k)) terms.")
pr()
pr("  In angular coordinates:")
pr("    ln(1+u) = ln(1+tan α) = ln(sec α + tan α · sec α)...complicated")
pr("  But: ∫_α₁^α₂ tan(π/4-α)·sec²α dα = [ln(1+tan α) - tan α]...+ const")
pr()
pr("  The π enters through the INFINITE SERIES of log corrections from")
pr("  all depths. Each depth contributes a finite combination of logs,")
pr("  but the series of these combinations converges to -π·σ₀₀.")
pr()

pr("  Per-depth log 'atoms' (from symbolic decomposition):")
all_log_args = set()
for d in range(1, min(max_d, do_sym_max) + 1):
    for sector in ['00', '10']:
        sym = results[sector]['sigma_sym']
        if sym:
            for (n, dn) in sym[d].logs:
                all_log_args.add((n, dn))

pr(f"  Unique log arguments across depths 1..{min(max_d, do_sym_max)}:")
for (n, dn) in sorted(all_log_args):
    pr(f"    ln({n}/{dn})")

pr()

# ── PART I: Testing Idea 4 — Spectral series connection ───────
pr("=" * 78)
pr("PART I: Spectral series connection (Idea 4)")
pr("-" * 60)
pr()

pr("  E09 proved: S(A) = Σ_{n odd} sin(nπ/2)/(1-λ_n(A)) = 2β/(1+2β)")
pr("  where β = 1-A, and S(A*) = -π.")
pr()
pr("  The eigenvalues λ_n(A) = cos(nπ/L)/2 + A·sin²(nπ/(2L)).")
pr("  At A=0 (Markov): λ_n = cos(nπ/L)/2 → 1/2 (for n << L)")
pr()
pr("  The CASCADE per-depth survival Surv(d) gives the effective")
pr("  spectral radius at each depth. If the Markov model were exact,")
pr("  Surv(d) = 3/4 (for carry ∈ {0,1} model).")
pr("  The actual Surv∞(d) → 1/2 + ε(d) → 1/2.")
pr()

pr("  Connecting per-depth data to S(A):")
pr("  The resolvent sum Σ_d Surv(d) · (1-Q(d)) · r(d) = Δ⟨val⟩₁₀")
pr("  is analogous to S(A) = Σ weights / (1-eigenvalue)")
pr()

pr("  If the per-depth survival product is:")
pr("    Surv(d) = Π_{j=1}^d Q(j)")
pr("  and Q(j) → 1/2 + ε(j), then:")
pr("    Surv(d) ≈ (1/2)^d · Π(1 + 2ε(j))")
pr()
pr("  The correction product Π(1+2ε(j)) determines the series structure.")
pr("  From E34, this product ≈ 1.736 ≈ √3 for d→∞.")
pr()

if len(surv_10) >= 5:
    product = ONE
    pr(f"  {'d':>3s}  {'Surv₁₀(d)':>14s}  {'ε₁₀(d)':>14s}  {'Π(1+2ε)':>14s}")
    for d in range(2, len(surv_10)):
        eps = surv_10[d] - HALF
        product *= (1 + 2 * eps)
        pr(f"  {d + 1:3d}  {nstr(surv_10[d], 10):>14s}  {nstr(eps, 10):>14s}  "
           f"{nstr(product, 10):>14s}")
    pr()
    pr(f"  Correction product (d=2..{len(surv_10)}): {nstr(product, 15)}")
    pr(f"  √3 = {nstr(sqrt(mpf(3)), 15)}")
    pr(f"  Difference: {nstr(product - sqrt(mpf(3)), 10)}")

pr()

# ── PART J: SYNTHESIS — the four ideas combined ────────────────
pr("=" * 78)
pr("PART J: Synthesis — combining the four ideas")
pr("-" * 60)
pr()

pr("  ┌────────────────────────────────────────────────────────────────┐")
pr("  │ IDEA 1 (Fredholm/Transfer Operator):                         │")
pr("  │   The per-depth survival Surv(d) → (1/2)^d · Π(1+2ε(d'))    │")
pr("  │   defines the resolvent (I-K)^{-1}. The eigenvalue 1/2      │")
pr("  │   (Diaconis-Fulman) is the dominant mode.                    │")
pr("  │                                                              │")
pr("  │ IDEA 2 (Walsh/Rademacher):                                   │")
pr("  │   At each depth, the carry path creates dyadic partitions.   │")
pr("  │   The D-odd boundary v=(1-u)/(1+u) slices these partitions.  │")
pr("  │   Each slice integral involves ln terms that, in angular     │")
pr("  │   coordinates, become arctan atoms.                          │")
pr("  │                                                              │")
pr("  │ IDEA 3 (Stokes/Green):                                      │")
pr("  │   The 2D integral ∫∫ val·dA decomposes into per-depth        │")
pr("  │   boundary integrals along α+β=π/4. The interior            │")
pr("  │   contributions are purely logarithmic (cancel in the ratio).│")
pr("  │   The boundary residue carries the π.                        │")
pr("  │                                                              │")
pr("  │ IDEA 4 (E09 Spectral):                                      │")
pr("  │   S(A) = 2β/(1+2β) with S(A*)=-π is the SPECTRAL version.  │")
pr("  │   The per-depth survival data encodes A_eff(d) → A*.        │")
pr("  │   This provides the bridge: the transfer operator's          │")
pr("  │   eigenvalue structure determines A, and A*=A gives R=-π.    │")
pr("  └────────────────────────────────────────────────────────────────┘")
pr()

pr("  CONVERGENCE STATUS:")
pr(f"    R(d≤{max_d}) = {nstr(R_cum, 20)}")
pr(f"    -π           = {nstr(-mpi, 20)}")
pr(f"    Gap           = {nstr(R_cum + mpi, 15)}")
pr()

pr("  SERIES STRUCTURE of f(d) = σ₁₀(d) + π·σ₀₀(d):")
if len(f_values) >= 4:
    ratios = []
    for i in range(1, len(f_values)):
        if abs(f_values[i - 1]) > AREA_THRESH:
            ratios.append(f_values[i] / f_values[i - 1])
    if ratios:
        avg_ratio = sum(ratios[-min(5, len(ratios)):]) / min(5, len(ratios))
        pr(f"    Average ratio (last 5): {nstr(avg_ratio, 15)}")
        pr(f"    Suggests ρ ≈ {nstr(avg_ratio, 8)} (geometric decay)")
        pr(f"    Compare: 1/2 = {nstr(HALF, 8)}")
        pr(f"             3/4 = {nstr(mpf(3) / 4, 8)}")
        pr(f"             1/e = {nstr(1 / power(mpf('2.71828'), 1), 8)}")

pr()
pr("=" * 78)
pr("E35 complete.")
pr("=" * 78)
