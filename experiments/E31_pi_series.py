#!/usr/bin/env python3
"""
E31: The Pi-Generating Series — Per-Depth Decomposition of R

Strategy: decompose R_casc = σ₁₀/σ₀₀ into per-depth contributions from the
cascade stopping mechanism. For each boundary layer width M and sector:

  ⟨val_casc⟩ = Σ_d P(depth=d) · E[val | depth=d]

By extrapolating each per-depth contribution to M → ∞, we construct
the INFINITE SERIES whose ratio gives −π.

Based on E31 boundary layer enumeration. The cascade finds the SHALLOWEST
(closest to MSB) nonzero carry — the LAST update wins as loop scans
from deep to shallow.
"""
import numpy as np
import math
import time
import sys
from collections import defaultdict

L2 = math.log(2)
L3 = math.log(3)
N10_area = 2 * math.log(4 / 3) - 0.5
N00_area = 2 * math.log(9 / 8)
R0_EXACT = (0.5 - 4 * L2 + 2 * L3) / (1 - 3 * L2 + L3)


def boundary_layer_perdepth(sector, M, batch_size=None):
    """
    Enumerate boundary layer patterns of width M for given sector.
    Returns per-depth cascade data and total valid count.

    Cascade depth = BL position of the shallowest nonzero carry.
    Position 0 = MSB, position M = deep.
    """
    a, b = sector
    n_rand = M - 1
    total_patterns = 1 << (2 * n_rand)

    if batch_size is None:
        batch_size = min(total_patterns, 1 << 22)

    SENTINEL = M + 2
    depth_data = defaultdict(lambda: [0, 0])
    total_valid = 0

    for batch_start in range(0, total_patterns, batch_size):
        batch_end = min(batch_start + batch_size, total_patterns)
        n_pat = batch_end - batch_start
        pat = np.arange(batch_start, batch_end, dtype=np.int64)

        x_bits = {}
        y_bits = {}
        for d_idx in range(n_rand):
            d = d_idx + 2
            x_bits[d] = ((pat >> (2 * d_idx)) & 1).astype(np.int32)
            y_bits[d] = ((pat >> (2 * d_idx + 1)) & 1).astype(np.int32)

        ones_arr = np.ones(n_pat, dtype=np.int32)
        a_arr = np.full(n_pat, a, dtype=np.int32)
        b_arr = np.full(n_pat, b, dtype=np.int32)
        zero_arr = np.zeros(n_pat, dtype=np.int32)

        def get_x(k):
            if k == 0:
                return ones_arr
            if k == 1:
                return a_arr
            return x_bits.get(k, zero_arr)

        def get_y(k):
            if k == 0:
                return ones_arr
            if k == 1:
                return b_arr
            return y_bits.get(k, zero_arr)

        for c_start in [0, 1]:
            s = np.full(n_pat, c_start, dtype=np.int32)

            cascade_pos = np.full(n_pat, SENTINEL, dtype=np.int32)
            cascade_val = np.full(n_pat, c_start - 1, dtype=np.int64)
            carry_below = np.full(n_pat, c_start, dtype=np.int32)

            for m in range(M, 0, -1):
                conv = np.zeros(n_pat, dtype=np.int32)
                for k in range(min(m + 1, M + 1)):
                    dp = m - k
                    if dp < 0 or dp > M:
                        continue
                    conv = conv + get_x(k) * get_y(dp)

                s_new = (conv + s) >> 1

                nonzero = (s > 0)
                cascade_pos = np.where(nonzero, m + 1, cascade_pos)
                cascade_val = np.where(nonzero, carry_below.astype(np.int64) - 1, cascade_val)

                carry_below = s.copy()
                s = s_new

            valid = (s == 0)
            n_valid = int(np.sum(valid))
            if n_valid == 0:
                continue
            total_valid += n_valid

            v_pos = cascade_pos[valid]
            v_val = cascade_val[valid]

            for pos_val in range(2, M + 3):
                mask = (v_pos == pos_val)
                cnt = int(np.sum(mask))
                if cnt > 0:
                    depth_data[pos_val][0] += cnt
                    depth_data[pos_val][1] += int(np.sum(v_val[mask]))

            survived = (v_pos == SENTINEL)
            n_surv = int(np.sum(survived))
            if n_surv > 0:
                depth_data[SENTINEL][0] += n_surv
                depth_data[SENTINEL][1] += int(np.sum(v_val[survived]))

    return dict(depth_data), total_valid


def pr(s=""):
    print(s, flush=True)


def prf(s):
    sys.stdout.write(s)
    sys.stdout.flush()


pr("=" * 80)
pr("E31: The Pi-Generating Series — Per-Depth Decomposition")
pr("=" * 80)
pr()
pr(f"  N₁₀ = {N10_area:.12f}")
pr(f"  N₀₀ = {N00_area:.12f}")
pr(f"  R₀  = {R0_EXACT:.12f}")
pr(f"  -π  = {-math.pi:.12f}")
pr(f"  ΔR  = {-math.pi - R0_EXACT:+.12f}")
pr()

# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Per-depth data for each M
# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("PHASE 1: Per-depth cascade decomposition")
pr("=" * 80)
pr()

M_MAX = 15
M_LIST = list(range(4, M_MAX + 1))
all_data = {}

pr(f"  {'M':>3s}  {'⟨val⟩₁₀':>12s}  {'⟨val⟩₀₀':>12s}  {'R':>12s}  {'|R+π|':>12s}  {'time':>8s}")

for M in M_LIST:
    n_rand = M - 1
    total_pat = 1 << (2 * n_rand)

    if total_pat > 1 << 30:
        pr(f"  {M:3d}  SKIPPED (too large: 2^{2*n_rand})")
        continue

    t0 = time.time()
    for sec_name, sec_bits in [("(1,0)", (1, 0)), ("(0,0)", (0, 0))]:
        depth_data, n_valid = boundary_layer_perdepth(sec_bits, M)
        all_data[(sec_name, M)] = (depth_data, n_valid)
    dt = time.time() - t0

    d10, nv10 = all_data[("(1,0)", M)]
    d00, nv00 = all_data[("(0,0)", M)]

    avg10 = sum(v[1] for v in d10.values()) / nv10 if nv10 > 0 else 0
    avg00 = sum(v[1] for v in d00.values()) / nv00 if nv00 > 0 else 0

    R_M = (avg10 * N10_area) / (avg00 * N00_area) if abs(avg00) > 1e-15 else float('nan')

    pr(f"  {M:3d}  {avg10:+12.8f}  {avg00:+12.8f}  {R_M:+12.8f}  {abs(R_M + math.pi):12.6f}  {dt:7.1f}s")

pr()

# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Per-depth profiles — w(d) = P(d)·E[val|d]
# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("PHASE 2: Weighted per-depth contributions w(d) = P(d)·E[val|d]")
pr("=" * 80)
pr()

M_display = [M for M in M_LIST if ("(1,0)", M) in all_data]
M_best = max(M_display) if M_display else 0

for sec_name in ["(1,0)", "(0,0)"]:
    pr(f"  Sector {sec_name}  (M={M_best}):")
    pr(f"  {'pos':>4s}  {'count':>10s}  {'P(pos)':>12s}  {'E[val|pos]':>12s}  {'w(pos)':>14s}  {'Σw':>14s}")

    key = (sec_name, M_best)
    if key not in all_data:
        pr("    No data.")
        continue

    depth_data, n_valid = all_data[key]
    cumul = 0.0
    for pos in sorted(depth_data.keys()):
        cnt, sv = depth_data[pos]
        if cnt == 0:
            continue
        p_d = cnt / n_valid
        e_val = sv / cnt
        w_d = sv / n_valid
        cumul += w_d
        label = f"p={pos}" if pos <= M_best + 1 else "bulk"
        pr(f"  {label:>4s}  {cnt:10d}  {p_d:12.8f}  {e_val:+12.6f}  {w_d:+14.10f}  {cumul:+14.10f}")
    pr(f"  ⟨val⟩ = {cumul:+.10f}")
    pr()

# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Per-step survival Q(d) at best M
# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("PHASE 3: Per-step empirical survival Q (largest M)")
pr("=" * 80)
pr()

for sec_name in ["(1,0)", "(0,0)"]:
    pr(f"  Sector {sec_name}  (M={M_best}):")
    pr(f"  {'pos':>4s}  {'P(stop)':>12s}  {'surv':>12s}  {'Q':>12s}")

    key = (sec_name, M_best)
    if key not in all_data:
        continue

    depth_data, n_valid = all_data[key]
    surv = 1.0
    for pos in range(2, M_best + 3):
        cnt = depth_data.get(pos, [0, 0])[0]
        p_stop = cnt / n_valid if n_valid > 0 else 0
        surv_new = surv - p_stop
        Q = surv_new / surv if surv > 1e-10 else 0
        pr(f"  {pos:4d}  {p_stop:12.8f}  {surv_new:12.8f}  {Q:12.6f}")
        surv = surv_new
        if surv < 1e-10:
            break
    pr()

# ═══════════════════════════════════════════════════════════════════════
# Phase 4: R convergence + acceleration
# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("PHASE 4: R convergence & acceleration")
pr("=" * 80)
pr()

R_vals = []
pr(f"  {'M':>3s}  {'R':>12s}  {'|R+π|':>12s}  {'ratio':>10s}")
prev_err = None
for M in M_display:
    d10, nv10 = all_data[("(1,0)", M)]
    d00, nv00 = all_data[("(0,0)", M)]
    avg10 = sum(v[1] for v in d10.values()) / nv10
    avg00 = sum(v[1] for v in d00.values()) / nv00
    R_M = (avg10 * N10_area) / (avg00 * N00_area)
    R_vals.append(R_M)
    err = abs(R_M + math.pi)
    rat_str = ""
    if prev_err is not None and err > 0:
        rat_str = f"{prev_err / err:.3f}"
    prev_err = err
    pr(f"  {M:3d}  {R_M:+12.8f}  {err:12.8f}  {rat_str:>10s}")

pr()

if len(R_vals) >= 5:
    n = len(R_vals)
    e = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        e[i][1] = R_vals[i]
    for k in range(2, n + 1):
        for i in range(n - k + 1):
            diff = e[i + 1][k - 1] - e[i][k - 1]
            if abs(diff) < 1e-30:
                e[i][k] = 1e30
            else:
                e[i][k] = e[i + 1][k - 2] + 1.0 / diff
    pr("  Wynn epsilon acceleration:")
    for k in range(2, min(n + 1, 12), 2):
        w = e[0][k]
        if abs(w) < 100:
            pr(f"    Order {k}: R* = {w:+.10f}, |R*+π| = {abs(w + math.pi):.2e}")
    pr()

# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Per-depth convergence across M
# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("PHASE 5: Per-depth w(pos) convergence across M")
pr("=" * 80)
pr()

last_Ms = M_display[-7:]

for sec_name in ["(1,0)", "(0,0)"]:
    pr(f"  Sector {sec_name} — w(pos) = Σval(pos)/N_valid:")
    prf(f"  {'pos':>4s}")
    for M in last_Ms:
        prf(f"  {'M='+str(M):>14s}")
    pr()

    for pos in range(2, 16):
        prf(f"  {pos:4d}")
        for M in last_Ms:
            key = (sec_name, M)
            if key not in all_data:
                prf(f"  {'—':>14s}")
                continue
            depth_data, n_valid = all_data[key]
            if pos in depth_data and depth_data[pos][0] > 0:
                w = depth_data[pos][1] / n_valid
                prf(f"  {w:+14.10f}")
            else:
                prf(f"  {0:+14.10f}")
        pr()
    pr()

# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Ratio analysis w₁₀(d)/w₀₀(d)
# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("PHASE 6: Per-depth ratio and partial R reconstruction")
pr("=" * 80)
pr()

key10 = ("(1,0)", M_best)
key00 = ("(0,0)", M_best)

if key10 in all_data and key00 in all_data:
    d10_data, nv10 = all_data[key10]
    d00_data, nv00 = all_data[key00]

    pr(f"  Per-depth contributions at M={M_best}:")
    pr(f"  {'pos':>4s}  {'w₁₀':>14s}  {'w₀₀':>14s}  {'Σw₁₀':>12s}  {'Σw₀₀':>12s}  {'R_partial':>12s}")

    cum10 = 0.0
    cum00 = 0.0

    all_positions = sorted(set(list(d10_data.keys()) + list(d00_data.keys())))
    for pos in all_positions:
        c10, sv10 = d10_data.get(pos, [0, 0])
        c00, sv00 = d00_data.get(pos, [0, 0])

        w10 = sv10 / nv10 if nv10 > 0 else 0
        w00 = sv00 / nv00 if nv00 > 0 else 0

        cum10 += w10
        cum00 += w00

        R_part = (cum10 * N10_area) / (cum00 * N00_area) if abs(cum00) > 1e-15 else float('nan')

        label = f"p={pos}" if pos <= M_best + 1 else "bulk"
        if abs(w10) + abs(w00) > 1e-12:
            pr(f"  {label:>4s}  {w10:+14.10f}  {w00:+14.10f}  {cum10:+12.8f}  {cum00:+12.8f}  {R_part:+12.6f}")

    pr()
    pr(f"  Final R = {(cum10 * N10_area) / (cum00 * N00_area):+.10f}")
    pr(f"  -π      = {-math.pi:.10f}")
    pr(f"  gap     = {(cum10 * N10_area) / (cum00 * N00_area) + math.pi:+.10f}")
    pr()

# ═══════════════════════════════════════════════════════════════════════
# Phase 7: Q_emp convergence across M
# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("PHASE 7: Q_emp(pos) across M — survival probability per step")
pr("=" * 80)
pr()

for sec_name in ["(1,0)", "(0,0)"]:
    pr(f"  Sector {sec_name}:")
    prf(f"  {'pos':>4s}")
    for M in last_Ms:
        prf(f"  {'M='+str(M):>10s}")
    pr()

    for pos in range(2, 14):
        prf(f"  {pos:4d}")
        for M in last_Ms:
            key = (sec_name, M)
            if key not in all_data:
                prf(f"  {'—':>10s}")
                continue
            depth_data, n_valid = all_data[key]

            surv_d = n_valid
            for p2 in range(2, pos):
                surv_d -= depth_data.get(p2, [0, 0])[0]

            stop_here = depth_data.get(pos, [0, 0])[0]
            surv_after = surv_d - stop_here

            Q = surv_after / surv_d if surv_d > 0 else 0
            prf(f"  {Q:10.6f}")
        pr()
    pr()

# ═══════════════════════════════════════════════════════════════════════
# Phase 8: PSLQ / series identification
# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("PHASE 8: Series identity search")
pr("=" * 80)
pr()

try:
    import mpmath
    mpmath.mp.dps = 50

    if key10 in all_data and key00 in all_data:
        d10_data, nv10 = all_data[key10]
        d00_data, nv00 = all_data[key00]

        ratio_N = N00_area / N10_area

        pr("  Testing: Σ_d [w₁₀(d) + π·(N₀₀/N₁₀)·w₀₀(d)] should = 0 if R = −π")
        pr(f"  (ratio N₀₀/N₁₀ = {ratio_N:.8f})")
        pr()
        pr(f"  {'pos':>4s}  {'term':>16s}  {'cumul':>16s}")

        cumul = 0.0
        for pos in sorted(set(list(d10_data.keys()) + list(d00_data.keys()))):
            c10, sv10 = d10_data.get(pos, [0, 0])
            c00, sv00 = d00_data.get(pos, [0, 0])
            w10 = sv10 / nv10 if nv10 > 0 else 0
            w00 = sv00 / nv00 if nv00 > 0 else 0
            term = w10 + math.pi * ratio_N * w00
            cumul += term
            if abs(w10) + abs(w00) > 1e-12:
                label = f"p={pos}" if pos <= M_best + 1 else "bulk"
                pr(f"  {label:>4s}  {term:+16.10f}  {cumul:+16.10f}")

        pr(f"\n  Residual: {cumul:+.10f}  (→ 0 as M→∞ if R = −π)")
        pr()

        pr("  PSLQ on R values:")
        for i, M in enumerate(M_display):
            R_M = R_vals[i]
            vec = [mpmath.mpf(str(R_M)), mpmath.mpf(1), mpmath.pi, mpmath.log(2), mpmath.log(3)]
            try:
                rel = mpmath.pslq(vec)
                if rel and rel[0] != 0:
                    exact = -(rel[1] + rel[2] * float(mpmath.pi) + rel[3] * float(mpmath.log(2)) + rel[4] * float(mpmath.log(3))) / rel[0]
                    pr(f"    M={M:2d}: PSLQ relation: {rel}  → R = {exact:.10f}")
            except Exception:
                pass
        pr()

except ImportError:
    pr("  mpmath not available — skipping PSLQ analysis")

pr("=" * 80)
pr("E31 complete.")
pr("=" * 80)
