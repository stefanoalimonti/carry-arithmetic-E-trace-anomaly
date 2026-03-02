#!/usr/bin/env python3
"""
E23: Direct continuum calculation of ⟨val⟩_{ac}

Strategy: same approach that succeeded for N₁₀/N₀₀ in E23.
  p = 2^{K-1}(1+u),  q = 2^{K-1}(1+v),  u,v ∈ [0,1)
  t = (1+u)(1+v) ∈ [1,2)  ←  D-odd condition
  Sectors: a = ⌊2u⌋, c = ⌊2v⌋

Key formula (derived analytically):
  carries[j] = ⌊A_j / 2^j⌋   where  A_j = Σ_{k<j} conv_k · 2^k
  Equivalently:  carries[D-1-m] = ⌊2^m · ε_m⌋
  where ε_m = tail of t past position m in the "top convolution" expansion.

Plan:
  A. For fixed (u,v) on a grid, compute val(u,v; K) for increasing K.
     Check pointwise convergence → val(u,v) = lim_{K→∞} val(u,v; K).
  B. Integrate val(u,v) over sectors numerically.
  C. Extract ⟨val⟩₀₀, ⟨val⟩₁₀ and the ratio R.
  D. Analytical structure: express val(u,v) in closed form.
  E. Compare with exact enumeration.
"""

import math
import numpy as np
from collections import defaultdict

def pr(s=""):
    print(s)

# ---------- constants from prior experiments ----------
LN_RATIO = (2*math.log(4/3) - 0.5) / (2*math.log(9/8))

# ---------- core: compute val for given (u,v) at bit-width K ----------
def compute_val(u, v, K):
    """Compute val = c_{M-1} - 1 for the D-odd pair (p,q) where
    p = floor(2^{K-1}(1+u)), q = floor(2^{K-1}(1+v)).
    Returns (val, M) or (None, 0) if not D-odd or M=0."""
    D = 2*K - 1
    p = int(2**(K-1) * (1 + u))
    q = int(2**(K-1) * (1 + v))
    if p < (1 << (K-1)) or p >= (1 << K):
        return None, 0
    if q < (1 << (K-1)) or q >= (1 << K):
        return None, 0
    n = p * q
    if n.bit_length() != D:
        return None, 0

    carries = [0] * (D + 1)
    for j in range(D):
        conv_j = 0
        for i in range(max(0, j - K + 1), min(j, K - 1) + 1):
            conv_j += ((p >> i) & 1) * ((q >> (j - i)) & 1)
        carries[j + 1] = (conv_j + carries[j]) >> 1

    M = 0
    cm1 = 0
    for j in range(D, 0, -1):
        if carries[j] > 0:
            M = j
            cm1 = carries[j - 1]
            break
    if M == 0:
        return None, 0
    return cm1 - 1, M


# ---------- PART A: Pointwise convergence of val(u,v; K) ----------
pr("=" * 70)
pr("PART A: Pointwise convergence of val(u, v; K)")
pr("=" * 70)
pr()

test_points = [
    (0.1, 0.2, "00"),
    (0.3, 0.4, "00"),
    (0.1, 0.45, "00"),
    (0.45, 0.1, "00"),
    (0.6, 0.1, "10"),
    (0.7, 0.2, "10"),
    (0.8, 0.15, "10"),
    (0.9, 0.05, "10"),
    (0.55, 0.25, "10"),
]

K_range = list(range(4, 22))
pr(f"Testing {len(test_points)} points for K = {K_range[0]}..{K_range[-1]}")
pr()

for u, v, sec_label in test_points:
    t = (1 + u) * (1 + v)
    if t >= 2.0:
        pr(f"  (u={u}, v={v}) sec={sec_label}: t={t:.4f} >= 2, NOT D-odd, skip")
        continue
    pr(f"  (u={u:.2f}, v={v:.2f}) sec={sec_label}, t={t:.6f}:")
    vals_line = []
    for K in K_range:
        val, M = compute_val(u, v, K)
        D = 2 * K - 1
        if val is not None:
            vals_line.append(f"{val:+.0f}(M={M},D-1-M={D-1-M})")
        else:
            vals_line.append("--")
    for i in range(0, len(K_range), 6):
        chunk = K_range[i:i+6]
        chunk_vals = vals_line[i:i+6]
        pr(f"    K={chunk[0]:2d}-{chunk[-1]:2d}: " + "  ".join(chunk_vals))
    pr()


# ---------- PART B: val as a function of (u,v) — heatmap data ----------
pr("=" * 70)
pr("PART B: val(u,v) for large K — structure in (u,v) plane")
pr("=" * 70)
pr()

K_large = 20
N_grid = 200
pr(f"Computing val(u,v) on {N_grid}x{N_grid} grid at K={K_large}")

val_grid_00 = []
val_grid_10 = []

for iu in range(N_grid):
    u = (iu + 0.5) / N_grid
    for iv in range(N_grid):
        v = (iv + 0.5) / N_grid
        t = (1 + u) * (1 + v)
        if t >= 2.0:
            continue
        a = int(2 * u)
        c = int(2 * v)
        if a == 1 and c == 1:
            continue
        val, M = compute_val(u, v, K_large)
        if val is None:
            continue
        D = 2 * K_large - 1
        if a == 0 and c == 0:
            val_grid_00.append((u, v, val, M, D))
        elif a == 1 and c == 0:
            val_grid_10.append((u, v, val, M, D))

pr(f"  Sector (0,0): {len(val_grid_00)} points")
pr(f"  Sector (1,0): {len(val_grid_10)} points")

if val_grid_00:
    vals_00 = [x[2] for x in val_grid_00]
    pr(f"    ⟨val⟩₀₀ = {np.mean(vals_00):.6f},  std = {np.std(vals_00):.4f}")
    pr(f"    min = {min(vals_00)}, max = {max(vals_00)}")
    M_vals_00 = [x[3] for x in val_grid_00]
    D_00 = val_grid_00[0][4]
    from collections import Counter
    M_counts = Counter([D_00 - 1 - m for m in [D_00 - 1 - M for M in M_vals_00]])
    pr(f"    M distribution (distance from top): {dict(sorted(Counter([D_00-1-M for M in M_vals_00]).items()))}")

if val_grid_10:
    vals_10 = [x[2] for x in val_grid_10]
    pr(f"    ⟨val⟩₁₀ = {np.mean(vals_10):.6f},  std = {np.std(vals_10):.4f}")
    pr(f"    min = {min(vals_10)}, max = {max(vals_10)}")
    M_vals_10 = [x[3] for x in val_grid_10]
    D_10 = val_grid_10[0][4]
    pr(f"    M distribution (distance from top): {dict(sorted(Counter([D_10-1-M for M in M_vals_10]).items()))}")

if val_grid_00 and val_grid_10:
    mean_val_00 = np.mean(vals_00)
    mean_val_10 = np.mean(vals_10)
    if abs(mean_val_00) > 1e-15:
        val_ratio = mean_val_10 / mean_val_00
        R_est = val_ratio * LN_RATIO
        pr(f"\n  val ratio ⟨val⟩₁₀/⟨val⟩₀₀ = {val_ratio:.6f}")
        pr(f"  R_est = val_ratio × N₁₀/N₀₀ = {R_est:.6f}")
        pr(f"  Target: -π = {-math.pi:.6f}")
pr()


# ---------- PART C: val(u,v) analytical structure ----------
pr("=" * 70)
pr("PART C: Analytical structure — carries near the top from t = (1+u)(1+v)")
pr("=" * 70)
pr()
pr("Formula: carries[D-1-m] = floor(2^m * ε_m)")
pr("where ε_m = tail of product t past position m.")
pr()

K_test = 16
D_test = 2 * K_test - 1

pr(f"Testing the formula carries[D-1-m] = floor(2^m * ε_m) at K={K_test}")
pr()

test_uv = [(0.3, 0.4), (0.1, 0.2), (0.7, 0.1), (0.6, 0.15)]
for u, v in test_uv:
    t = (1 + u) * (1 + v)
    if t >= 2.0:
        continue
    p = int(2**(K_test-1) * (1 + u))
    q = int(2**(K_test-1) * (1 + v))
    n = p * q
    if n.bit_length() != D_test:
        continue

    carries_exact = [0] * (D_test + 1)
    for j in range(D_test):
        conv_j = 0
        for i in range(max(0, j - K_test + 1), min(j, K_test - 1) + 1):
            conv_j += ((p >> i) & 1) * ((q >> (j - i)) & 1)
        carries_exact[j + 1] = (conv_j + carries_exact[j]) >> 1

    pr(f"  (u={u}, v={v}), t={t:.10f}, p={p}, q={q}")

    p_bits = [(p >> i) & 1 for i in range(K_test)]
    q_bits = [(q >> i) & 1 for i in range(K_test)]

    a_seq = [1] + list(reversed(p_bits[:-1]))
    c_seq = [1] + list(reversed(q_bits[:-1]))

    for m in range(8):
        conv_top_sum = 0.0
        for k in range(m + 1):
            ck = sum(a_seq[i] * c_seq[k - i] for i in range(k + 1))
            conv_top_sum += ck * 2**(-k)
        eps_m = t - conv_top_sum
        carry_formula = int(math.floor(2**m * eps_m))
        carry_exact_val = carries_exact[D_test - 1 - m]
        match = "✓" if carry_formula == carry_exact_val else "✗"
        pr(f"    m={m}: ε_{m}={eps_m:.10f}, 2^m·ε={2**m*eps_m:.6f}, "
           f"⌊·⌋={carry_formula}, exact={carry_exact_val} {match}")
    pr()


# ---------- PART D: What determines val(u,v) in the continuous limit ----------
pr("=" * 70)
pr("PART D: val(u,v) structure — dependence on t = (1+u)(1+v)")
pr("=" * 70)
pr()
pr("Key question: does val depend only on t, or separately on (u,v)?")
pr()

K_check = 18
pairs_by_t = defaultdict(list)
N_samp = 500
rng = np.random.RandomState(42)
for _ in range(N_samp):
    u = rng.uniform(0, 0.5)
    v = rng.uniform(0, 0.5)
    t = (1 + u) * (1 + v)
    if t >= 2.0:
        continue
    val, M = compute_val(u, v, K_check)
    if val is not None:
        pairs_by_t['00'].append((t, val, M, u, v))

for _ in range(N_samp):
    u = rng.uniform(0.5, 1.0)
    v = rng.uniform(0, 0.5)
    t = (1 + u) * (1 + v)
    if t >= 2.0:
        continue
    val, M = compute_val(u, v, K_check)
    if val is not None:
        pairs_by_t['10'].append((t, val, M, u, v))

for sec in ['00', '10']:
    data = sorted(pairs_by_t[sec], key=lambda x: x[0])
    pr(f"Sector ({sec[0]},{sec[1]}): {len(data)} samples")
    n_bins = 10
    for i in range(n_bins):
        lo_idx = i * len(data) // n_bins
        hi_idx = (i + 1) * len(data) // n_bins
        chunk = data[lo_idx:hi_idx]
        if chunk:
            t_lo, t_hi = chunk[0][0], chunk[-1][0]
            vals_chunk = [x[1] for x in chunk]
            Ms_chunk = [x[2] for x in chunk]
            D_c = 2 * K_check - 1
            pr(f"  t ∈ [{t_lo:.4f}, {t_hi:.4f}]: ⟨val⟩={np.mean(vals_chunk):+.4f}, "
               f"std={np.std(vals_chunk):.3f}, ⟨M⟩={np.mean(Ms_chunk):.1f} (D-2={D_c-2}), "
               f"n={len(chunk)}")
    pr()


# ---------- PART E: Exact enumeration comparison (small K) ----------
pr("=" * 70)
pr("PART E: Grid-based ⟨val⟩ vs exact enumeration")
pr("=" * 70)
pr()

for K in [6, 8, 10]:
    D = 2 * K - 1
    lo, hi = 1 << (K - 1), 1 << K

    exact = defaultdict(lambda: {'count': 0, 'sum_val': 0})
    for p in range(lo, hi):
        for q in range(lo, hi):
            n = p * q
            if n.bit_length() != D:
                continue
            carries = [0] * (D + 1)
            for j in range(D):
                conv_j = 0
                for i in range(max(0, j - K + 1), min(j, K - 1) + 1):
                    conv_j += ((p >> i) & 1) * ((q >> (j - i)) & 1)
                carries[j + 1] = (conv_j + carries[j]) >> 1
            M = 0
            cm1 = 0
            for j in range(D, 0, -1):
                if carries[j] > 0:
                    M = j
                    cm1 = carries[j - 1]
                    break
            if M == 0:
                continue
            val = cm1 - 1
            a = (p >> (K - 2)) & 1
            c_sec = (q >> (K - 2)) & 1
            sec = f"{a}{c_sec}"
            exact[sec]['count'] += 1
            exact[sec]['sum_val'] += val

    pr(f"K={K}, D={D}")
    for sec in ['00', '10']:
        if exact[sec]['count'] > 0:
            mean_exact = exact[sec]['sum_val'] / exact[sec]['count']
            pr(f"  Sector ({sec[0]},{sec[1]}): N={exact[sec]['count']}, ⟨val⟩_exact = {mean_exact:.6f}")

    if exact['00']['count'] > 0 and exact['10']['count'] > 0:
        mean_00 = exact['00']['sum_val'] / exact['00']['count']
        mean_10 = exact['10']['sum_val'] / exact['10']['count']
        n_ratio = exact['10']['count'] / exact['00']['count']
        R = (mean_10 / mean_00) * n_ratio if abs(mean_00) > 1e-15 else float('inf')
        pr(f"  N₁₀/N₀₀ = {n_ratio:.6f} (target {LN_RATIO:.6f})")
        pr(f"  ⟨val⟩₁₀/⟨val⟩₀₀ = {mean_10/mean_00:.6f}" if abs(mean_00) > 1e-15 else "  ⟨val⟩₀₀ = 0!")
        pr(f"  R = {R:.6f} (target {-math.pi:.6f})")
    pr()


# ---------- PART F: The carry formula — express val analytically ----------
pr("=" * 70)
pr("PART F: Analytical val from carries formula")
pr("=" * 70)
pr()
pr("carries[D-1-m] = floor(2^m * ε_m)")
pr("where ε_m = t - Σ_{k=0}^{m} C_k * 2^{-k}")
pr("and C_k = Σ_{i=0}^{k} a_i * c_{k-i} = k-th digit-convolution of (1+u) and (1+v)")
pr()
pr("For the CONTINUOUS limit, t = (1+u)(1+v) exactly,")
pr("so ε_m = the tail of t past position m in its 'digit-convolution expansion'.")
pr()

pr("Key insight: the digit-convolution expansion of t is NOT the binary expansion!")
pr("It carries redundant digits (conv_k can be > 1).")
pr("The carry chain converts redundant digits → binary by propagating carries upward.")
pr()

pr("In the continuous limit (K → ∞), for positions near the top (small m),")
pr("the tail ε_m converges to a function of (u,v) that depends on O(m) bits.")
pr("But val depends on M (where carries first become nonzero from the top),")
pr("and c_{M-1}.")
pr()

pr("Let's compute val(u,v) for a fine grid and examine its functional form.")
pr()

K_fine = 20
N_fine = 400
sector_00_vals = []
sector_10_vals = []
sector_00_ts = []
sector_10_ts = []

for iu in range(N_fine):
    u = (iu + 0.5) / N_fine
    for iv in range(N_fine):
        v = (iv + 0.5) / N_fine
        t = (1 + u) * (1 + v)
        if t >= 2.0:
            continue
        a = int(2 * u)
        c = int(2 * v)
        if a == 1 and c == 1:
            continue
        val, M = compute_val(u, v, K_fine)
        if val is None:
            continue
        if a == 0 and c == 0:
            sector_00_vals.append(val)
            sector_00_ts.append(t)
        elif a == 1 and c == 0:
            sector_10_vals.append(val)
            sector_10_ts.append(t)

area_00 = len(sector_00_vals) / N_fine**2
area_10 = len(sector_10_vals) / N_fine**2
f00_theory = 2 * math.log(9/8)
f10_theory = 2 * math.log(4/3) - 0.5

pr(f"K={K_fine}, grid={N_fine}x{N_fine}")
pr(f"  Sector (0,0): {len(sector_00_vals)} points, area≈{area_00:.6f} (theory {f00_theory:.6f})")
pr(f"  Sector (1,0): {len(sector_10_vals)} points, area≈{area_10:.6f} (theory {f10_theory:.6f})")
pr()

mean_val_00 = np.mean(sector_00_vals)
mean_val_10 = np.mean(sector_10_vals)
pr(f"  ⟨val⟩₀₀ = {mean_val_00:.8f}")
pr(f"  ⟨val⟩₁₀ = {mean_val_10:.8f}")
if abs(mean_val_00) > 1e-15:
    val_rat = mean_val_10 / mean_val_00
    R_grid = val_rat * LN_RATIO
    pr(f"  ⟨val⟩₁₀/⟨val⟩₀₀ = {val_rat:.8f}")
    pr(f"  R = (⟨val⟩₁₀/⟨val⟩₀₀) × (N₁₀/N₀₀) = {R_grid:.8f}")
    pr(f"  Target: -π = {-math.pi:.8f}")
    pr(f"  |R + π| = {abs(R_grid + math.pi):.6f}")
pr()


# ---------- PART G: Dependence on individual bits ----------
pr("=" * 70)
pr("PART G: val(u,v) — dependence on leading bits of u,v")
pr("=" * 70)
pr()

pr("Sector (0,0): u,v ∈ [0, 1/2). Sub-divide by 2nd bits a₂=⌊4u⌋-2⌊2u⌋, c₂=⌊4v⌋-2⌊2v⌋.")
pr()

K_g = 20
sub_00 = defaultdict(list)
sub_10 = defaultdict(list)

for iu in range(N_fine):
    u = (iu + 0.5) / N_fine
    for iv in range(N_fine):
        v = (iv + 0.5) / N_fine
        t = (1 + u) * (1 + v)
        if t >= 2.0:
            continue
        a = int(2 * u)
        c_sec = int(2 * v)
        if a == 1 and c_sec == 1:
            continue
        val, M = compute_val(u, v, K_g)
        if val is None:
            continue
        a2 = int(4 * u) - 2 * int(2 * u)
        c2 = int(4 * v) - 2 * int(2 * v)
        key = (a2, c2)
        if a == 0 and c_sec == 0:
            sub_00[key].append(val)
        elif a == 1 and c_sec == 0:
            sub_10[key].append(val)

pr("Sector (0,0) by (a₂, c₂):")
for key in sorted(sub_00.keys()):
    vals = sub_00[key]
    pr(f"  (a₂={key[0]}, c₂={key[1]}): n={len(vals)}, ⟨val⟩={np.mean(vals):+.6f}, "
       f"std={np.std(vals):.4f}")

pr()
pr("Sector (1,0) by (a₂, c₂):")
for key in sorted(sub_10.keys()):
    vals = sub_10[key]
    pr(f"  (a₂={key[0]}, c₂={key[1]}): n={len(vals)}, ⟨val⟩={np.mean(vals):+.6f}, "
       f"std={np.std(vals):.4f}")
pr()


# ---------- PART H: val vs t — is val a function of t alone? ----------
pr("=" * 70)
pr("PART H: Is val(u,v) a function of t=(1+u)(1+v) alone within each sector?")
pr("=" * 70)
pr()

pr("For sector (0,0), compare pairs with same t but different (u,v):")
K_h = 16
from collections import defaultdict as dd

t_to_vals_00 = dd(list)
N_h = 300
for iu in range(N_h):
    u = (iu + 0.5) / N_h
    if u >= 0.5:
        continue
    for iv in range(N_h):
        v = (iv + 0.5) / N_h
        if v >= 0.5:
            continue
        t = (1 + u) * (1 + v)
        if t >= 2.0:
            continue
        val, M = compute_val(u, v, K_h)
        if val is None:
            continue
        t_bin = round(t, 4)
        t_to_vals_00[t_bin].append((val, u, v))

multi_t = {k: v for k, v in t_to_vals_00.items() if len(v) >= 3}
pr(f"  {len(multi_t)} t-bins with ≥ 3 points")

n_agree = 0
n_disagree = 0
for t_bin, entries in sorted(multi_t.items())[:20]:
    vals_at_t = [e[0] for e in entries]
    if len(set(vals_at_t)) == 1:
        n_agree += 1
    else:
        n_disagree += 1
        pr(f"  t≈{t_bin}: vals = {vals_at_t[:6]} — NOT a function of t alone!")

for t_bin, entries in sorted(multi_t.items()):
    vals_at_t = [e[0] for e in entries]
    if len(set(vals_at_t)) == 1:
        n_agree += 1
    else:
        n_disagree += 1

pr(f"\n  t-bins where val is constant: {n_agree}")
pr(f"  t-bins where val varies: {n_disagree}")
pr(f"  → val(u,v) {'IS' if n_disagree == 0 else 'is NOT'} a function of t alone")
pr()


# ---------- PART I: Convergence with K — grid integral ----------
pr("=" * 70)
pr("PART I: ⟨val⟩ convergence with K (grid integral)")
pr("=" * 70)
pr()

N_int = 200
pr(f"Grid {N_int}x{N_int}, K = 8..22")
pr()
pr(f"{'K':>4s} {'⟨val⟩₀₀':>12s} {'⟨val⟩₁₀':>12s} {'val_ratio':>12s} {'R_est':>12s} {'|R+π|':>10s}")

for K in range(8, 23):
    s00_sum = 0.0
    s00_cnt = 0
    s10_sum = 0.0
    s10_cnt = 0

    for iu in range(N_int):
        u = (iu + 0.5) / N_int
        for iv in range(N_int):
            v = (iv + 0.5) / N_int
            t = (1 + u) * (1 + v)
            if t >= 2.0:
                continue
            a = int(2 * u)
            c_sec = int(2 * v)
            if a == 1 and c_sec == 1:
                continue
            val, M = compute_val(u, v, K)
            if val is None:
                continue
            if a == 0 and c_sec == 0:
                s00_sum += val
                s00_cnt += 1
            elif a == 1 and c_sec == 0:
                s10_sum += val
                s10_cnt += 1

    if s00_cnt > 0 and s10_cnt > 0:
        m00 = s00_sum / s00_cnt
        m10 = s10_sum / s10_cnt
        if abs(m00) > 1e-15:
            vr = m10 / m00
            R = vr * LN_RATIO
            pr(f"{K:4d} {m00:12.6f} {m10:12.6f} {vr:12.6f} {R:12.6f} {abs(R+math.pi):10.6f}")
        else:
            pr(f"{K:4d} {m00:12.6f} {m10:12.6f} {'div/0':>12s}")
    else:
        pr(f"{K:4d} {'--':>12s} {'--':>12s}")

pr()
pr("=" * 70)
pr("PART J: Synthesis")
pr("=" * 70)
pr()
pr("If val(u,v) converges pointwise as K → ∞, then:")
pr("  ⟨val⟩_{ac} = ∫∫_{sector ac, D-odd} val(u,v) du dv / ∫∫_{sector ac, D-odd} du dv")
pr("and R = (⟨val⟩₁₀/⟨val⟩₀₀) × (N₁₀/N₀₀) = -π.")
pr()
pr("The convergence rate and functional form of val(u,v) will reveal")
pr("whether an analytical closed-form integral is possible.")
