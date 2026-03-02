#!/usr/bin/env python3
"""
E25: High-precision deep σ(d₀) computation via combined DFS + mpmath

Computes σ₀₀(d₀) and σ₁₀(d₀) for d₀ = 1..18 using mpmath at 60-digit precision.
A single DFS pass processes ALL depth levels simultaneously (no restart per d₀).
Also tracks the Area / ∫b / ∫C decomposition for Strategy A analysis.

Output:
  Part A — per-depth σ values and pattern counts
  Part B — cumulative ratio σ₁₀/σ₀₀ → −π convergence
  Part C — Δ_d = σ₁₀(d) + π·σ₀₀(d) growth analysis
  Part D — decomposition σ = Area + ∫b − ∫C (Strategy A data)
  Part E — PSLQ on individual Δ_d
  Part F — cumulative Δ convergence to 0
  Part G — val distribution per depth

Usage:
  python3 E25_deep_precision.py [max_depth] [time_limit_seconds]
  Defaults: max_depth=18, time_limit=7200 (2h per sector, 4h total)
"""

import sys
import time

try:
    from mpmath import mp, mpf, log, pi as mpi, fsum, nstr, pslq
except ImportError:
    print("ERROR: mpmath not installed. Run: pip install mpmath")
    sys.exit(1)

MAX_DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 18
TIME_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 7200

mp.dps = 60

ZERO = mpf(0)
ONE = mpf(1)
TWO = mpf(2)
AREA_THRESH = mpf(10) ** (-50)


def pr(s=""):
    print(s, flush=True)


def mpf_area(u_lo, u_hi, v_lo, v_hi, t_lo, t_hi):
    """∫∫ dv du over {u∈[u_lo,u_hi], v∈[v_lo,v_hi], t_lo ≤ (1+u)(1+v) < t_hi}."""
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


def compute_all(sector, max_depth, time_limit):
    """Combined DFS: σ(d₀), Area(d₀), B(d₀), C(d₀), val histogram for all d₀."""
    a1 = int(sector[0])
    c1 = int(sector[1])

    MN = max_depth + 4
    a = [1, a1] + [0] * MN
    c = [1, c1] + [0] * MN
    b = [1] + [0] * MN

    D = max_depth + 1
    sigma = [ZERO] * D
    area  = [ZERO] * D
    int_b = [ZERO] * D
    int_C = [ZERO] * D
    counts = [0] * D
    val_hist = [{} for _ in range(D)]  # val_hist[d][val] = area

    t0 = time.time()
    nodes = [0]
    last_rpt = [t0]

    def C_val(k):
        s = 0
        for i in range(k + 1):
            s += a[i] * c[k - i]
        return s

    def process_d0(d0):
        saved_b = b[d0]
        b[d0] = 1
        cross = 0
        for i in range(1, d0 + 1):
            cross += a[i] * c[d0 + 1 - i]

        n = d0 + 1
        pow2n = 1 << n

        for an in range(2):
            a[d0 + 1] = an
            for cn in range(2):
                c[d0 + 1] = cn
                Cd = cn + cross + an
                for bn in range(2):
                    val = 1 + bn - Cd
                    if val == 0:
                        continue
                    b[d0 + 1] = bn

                    t_int = 0
                    for j in range(n + 1):
                        t_int += b[j] << (n - j)
                    tlo = mpf(t_int) / mpf(pow2n)
                    thi = tlo + ONE / mpf(pow2n)
                    if thi > TWO:
                        thi = TWO
                    if tlo >= TWO or thi <= tlo:
                        continue

                    u_int = 0
                    for j in range(1, n + 1):
                        u_int += a[j] << (n - j)
                    ulo = mpf(u_int) / mpf(pow2n)
                    uhi = ulo + ONE / mpf(pow2n)

                    v_int = 0
                    for j in range(1, n + 1):
                        v_int += c[j] << (n - j)
                    vlo = mpf(v_int) / mpf(pow2n)
                    vhi = vlo + ONE / mpf(pow2n)

                    ar = mpf_area(ulo, uhi, vlo, vhi, tlo, thi)
                    if ar > AREA_THRESH:
                        sigma[d0] += val * ar
                        area[d0]  += ar
                        int_b[d0] += bn * ar
                        int_C[d0] += Cd * ar
                        counts[d0] += 1
                        val_hist[d0][val] = val_hist[d0].get(val, ZERO) + ar

        a[d0 + 1] = 0
        c[d0 + 1] = 0
        b[d0 + 1] = 0
        b[d0] = saved_b

    def dfs(depth):
        nodes[0] += 1
        now = time.time()
        if now - t0 > time_limit:
            return
        if now - last_rpt[0] > 30:
            pr(f"    [{time.time()-t0:.0f}s] depth={depth}, nodes={nodes[0]:,d}")
            last_rpt[0] = now

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

    dfs(1)
    elapsed = time.time() - t0
    timed_out = elapsed >= time_limit * 0.99

    max_d_actual = 0
    for d in range(1, max_depth + 1):
        if counts[d] > 0:
            max_d_actual = d

    pr(f"  Sector ({sector[0]},{sector[1]}): {nodes[0]:,d} nodes, "
       f"{elapsed:.1f}s, max_d₀={max_d_actual}"
       f"{' (TIME LIMIT)' if timed_out else ''}")

    return {
        'sigma': sigma, 'area': area, 'int_b': int_b,
        'int_C': int_C, 'counts': counts, 'val_hist': val_hist,
        'max_d': max_d_actual
    }


# ═══════════════════════════════════════════════════════════════
pr("=" * 78)
pr(f"E25: High-Precision Deep σ(d₀)  —  mpmath {mp.dps}-digit")
pr(f"  max_depth = {MAX_DEPTH},  time_limit = {TIME_LIMIT}s per sector")
pr("=" * 78)
pr()

pr("--- Computing σ₀₀ [sector (0,0)] ---")
r00 = compute_all('00', MAX_DEPTH, TIME_LIMIT)
pr()
pr("--- Computing σ₁₀ [sector (1,0)] ---")
r10 = compute_all('10', MAX_DEPTH, TIME_LIMIT)
pr()

max_d = min(r00['max_d'], r10['max_d'])

# ═══════════════════════════════════════════════════════════════
pr("=" * 78)
pr("PART A: σ(d₀) per depth")
pr("=" * 78)
pr()
pr(f"{'d₀':>4s}  {'σ₀₀(d₀)':>32s}  {'σ₁₀(d₀)':>32s}  {'cnt00':>7s} {'cnt10':>7s}")
for d in range(1, max_d + 1):
    pr(f"{d:4d}  {nstr(r00['sigma'][d], 25):>32s}  "
       f"{nstr(r10['sigma'][d], 25):>32s}  "
       f"{r00['counts'][d]:7d} {r10['counts'][d]:7d}")

# ═══════════════════════════════════════════════════════════════
pr()
pr("=" * 78)
pr("PART B: Cumulative ratio σ₁₀/σ₀₀ → −π")
pr("=" * 78)
pr()

cum00 = ZERO
cum10 = ZERO
pr(f"{'d':>4s}  {'R = σ₁₀/σ₀₀':>36s}  {'|R + π|':>24s}  {'digits':>7s}")
for d in range(1, max_d + 1):
    cum00 += r00['sigma'][d]
    cum10 += r10['sigma'][d]
    if abs(cum00) > mpf(10) ** (-40):
        R = cum10 / cum00
        err = abs(R + mpi)
        if err > 0:
            digits = float(-log(err, 10))
        else:
            digits = mp.dps
        pr(f"{d:4d}  {nstr(R, 30):>36s}  {nstr(err, 16):>24s}  {digits:7.2f}")

pr()
pr(f"  σ₀₀  = {nstr(cum00, 40)}")
pr(f"  σ₁₀  = {nstr(cum10, 40)}")
pr(f"  R     = {nstr(cum10 / cum00, 40)}")
pr(f"  −π    = {nstr(-mpi, 40)}")

# ═══════════════════════════════════════════════════════════════
pr()
pr("=" * 78)
pr("PART C: Δ_d = σ₁₀(d) + π·σ₀₀(d)  growth analysis")
pr("=" * 78)
pr()

deltas = []
pr(f"{'d':>4s}  {'Δ_d':>28s}  {'|Δ|·2^d':>22s}  "
   f"{'Δ ratio':>12s}  {'scaled ratio':>14s}")
for d in range(1, max_d + 1):
    delta = r10['sigma'][d] + mpi * r00['sigma'][d]
    deltas.append(delta)
    scaled = abs(delta) * mpf(2) ** d
    if len(deltas) >= 2 and abs(deltas[-2]) > 0:
        dr = deltas[-1] / deltas[-2]
        prev_sc = abs(deltas[-2]) * mpf(2) ** (d - 1)
        sr = scaled / prev_sc if prev_sc > 0 else ZERO
        pr(f"{d:4d}  {nstr(delta, 18):>28s}  {nstr(scaled, 14):>22s}  "
           f"{nstr(dr, 8):>12s}  {nstr(sr, 8):>14s}")
    else:
        pr(f"{d:4d}  {nstr(delta, 18):>28s}  {nstr(scaled, 14):>22s}")

pr()
pr("  If |Δ_d| ~ C·d^α / 2^d  then  |Δ|·2^d ~ C·d^α")
pr("  α ≈ 0.5 (sub-geometric) was observed in E24f.")

# ═══════════════════════════════════════════════════════════════
pr()
pr("=" * 78)
pr("PART D: Decomposition  σ = Area + ∫b − ∫C  (Strategy A)")
pr("=" * 78)
pr()

for label, res in [("(0,0)", r00), ("(1,0)", r10)]:
    pr(f"  Sector {label}:")
    pr(f"  {'d':>4s}  {'Area':>20s}  {'∫b':>20s}  {'∫C':>20s}  {'σ':>20s}")
    for d in range(1, max_d + 1):
        pr(f"  {d:4d}  {nstr(res['area'][d], 14):>20s}  "
           f"{nstr(res['int_b'][d], 14):>20s}  "
           f"{nstr(res['int_C'][d], 14):>20s}  "
           f"{nstr(res['sigma'][d], 14):>20s}")
    pr()

cumA00 = fsum(r00['area'][1:max_d + 1])
cumB00 = fsum(r00['int_b'][1:max_d + 1])
cumC00 = fsum(r00['int_C'][1:max_d + 1])
cumA10 = fsum(r10['area'][1:max_d + 1])
cumB10 = fsum(r10['int_b'][1:max_d + 1])
cumC10 = fsum(r10['int_C'][1:max_d + 1])

N00 = 2 * log(mpf(9) / 8)
N10 = 2 * log(mpf(4) / 3) - mpf(1) / 2

pr(f"  Cumulative areas vs E19 exact:")
pr(f"    A₀₀ = {nstr(cumA00, 25)}   N₀₀ = {nstr(N00, 25)}")
pr(f"    A₁₀ = {nstr(cumA10, 25)}   N₁₀ = {nstr(N10, 25)}")
pr(f"    Area gap (0,0): {nstr(cumA00 - N00, 18)}")
pr(f"    Area gap (1,0): {nstr(cumA10 - N10, 18)}")
pr()

eta = N10 + mpi * N00
E00 = cum00 - N00
E10 = cum10 - N10
excess_check = E10 + mpi * E00

pr(f"  η = N₁₀ + π·N₀₀ = {nstr(eta, 30)}")
pr(f"  Need: E₁₀ + π·E₀₀ = −η = {nstr(-eta, 30)}")
pr(f"  Got:  E₁₀ + π·E₀₀ = {nstr(excess_check, 30)}")
pr(f"  Residual: {nstr(excess_check + eta, 18)}")
pr()
pr(f"  ⟨val⟩₀₀ = σ₀₀/A₀₀ = {nstr(cum00 / cumA00, 25)}")
pr(f"  ⟨val⟩₁₀ = σ₁₀/A₁₀ = {nstr(cum10 / cumA10, 25)}")
pr(f"  ⟨b⟩₀₀ = B₀₀/A₀₀ = {nstr(cumB00 / cumA00, 25)}")
pr(f"  ⟨b⟩₁₀ = B₁₀/A₁₀ = {nstr(cumB10 / cumA10, 25)}")
pr(f"  ⟨C⟩₀₀ = C₀₀/A₀₀ = {nstr(cumC00 / cumA00, 25)}")
pr(f"  ⟨C⟩₁₀ = C₁₀/A₁₀ = {nstr(cumC10 / cumA10, 25)}")

# ═══════════════════════════════════════════════════════════════
pr()
pr("=" * 78)
pr("PART E: PSLQ on Δ_d · 2^d")
pr("=" * 78)
pr()

for d in range(1, min(max_d + 1, 11)):
    x = deltas[d - 1] * mpf(2) ** d
    if abs(x) < mpf(10) ** (-40):
        continue
    basis = [x, ONE, mpi, log(mpf(2)), log(mpf(3)), log(mpf(5)),
             mpi * log(mpf(2)), mpi * log(mpf(3))]
    labels = ['Δ·2^d', '1', 'π', 'ln2', 'ln3', 'ln5', 'π·ln2', 'π·ln3']
    try:
        rel = pslq(basis, maxcoeff=10000)
        if rel is not None:
            terms = [f"{c}·{l}" for c, l in zip(rel, labels) if c != 0]
            pr(f"  d={d}: {' + '.join(terms)} = 0")
        else:
            pr(f"  d={d}: no relation found")
    except Exception as e:
        pr(f"  d={d}: PSLQ error: {e}")

# ═══════════════════════════════════════════════════════════════
pr()
pr("=" * 78)
pr("PART F: Cumulative Δ → 0")
pr("=" * 78)
pr()

cum_delta = ZERO
pr(f"{'d':>4s}  {'Σ Δ_d':>30s}  {'|Σ Δ|':>22s}")
for d in range(1, max_d + 1):
    cum_delta += deltas[d - 1]
    pr(f"{d:4d}  {nstr(cum_delta, 22):>30s}  {nstr(abs(cum_delta), 14):>22s}")

pr()
pr(f"  |Σ Δ| at d={max_d}: {nstr(abs(cum_delta), 20)}")
pr(f"  σ₁₀ + π·σ₀₀ = Σ Δ should → 0 if R = −π.")

# ═══════════════════════════════════════════════════════════════
pr()
pr("=" * 78)
pr("PART G: Val distribution per depth")
pr("=" * 78)
pr()

for label, res in [("(0,0)", r00), ("(1,0)", r10)]:
    pr(f"  Sector {label}:")
    for d in range(1, min(max_d + 1, 11)):
        vh = res['val_hist'][d]
        if not vh:
            continue
        parts = [f"val={v}: {nstr(a, 10)}" for v, a in sorted(vh.items())]
        total = sum(vh.values())
        fracs = [f"val={v}: {float(a/total):.4f}" for v, a in sorted(vh.items())
                 if total > 0]
        pr(f"    d={d}: {', '.join(fracs)}")
    pr()

# ═══════════════════════════════════════════════════════════════
pr()
pr("=" * 78)
pr("SUMMARY")
pr("=" * 78)
pr()
if max_d > 0:
    final_R = cum10 / cum00
    final_err = abs(final_R + mpi)
    final_digits = float(-log(final_err, 10)) if final_err > 0 else mp.dps
    pr(f"  Depths computed: d₀ = 1..{max_d}")
    pr(f"  σ₁₀/σ₀₀ = {nstr(final_R, 35)}")
    pr(f"  −π       = {nstr(-mpi, 35)}")
    pr(f"  |R + π|  = {nstr(final_err, 20)}")
    pr(f"  Matching digits: {final_digits:.1f}")
    pr(f"  |Σ Δ|    = {nstr(abs(cum_delta), 20)}")
pr()
pr("=" * 78)
pr("E25 complete.")
pr("=" * 78)
