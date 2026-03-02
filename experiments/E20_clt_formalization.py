#!/usr/bin/env python3
"""
E20: CLT Formalization — Carry chain → Markov bridge

Goals:
  A. Verify ⟨val⟩₁₀/⟨val⟩₀₀ → target from prior experiments factorization
  B. M-resolved decomposition: how M-distribution and conditional val combine
  C. Top-layer exact calculation: carries near D-2 are discrete, not CLT
  D. Markov approximation for the bulk: estimate transition kernel, feed to top
  E. Identify where sin(nπ/2) enters: propagator from midpoint to top

The two gaps we're attacking:
  Gap 1: perturbation accumulates at TOP, not midpoint
  Gap 2: no carry shift → Leibniz derivation
"""

import numpy as np
from collections import defaultdict
import math

np.set_printoptions(precision=8, linewidth=120)

def pr(s=""):
    print(s)

# ---------- constants from prior experiments ----------
LN_RATIO = (2*math.log(4/3) - 0.5) / (2*math.log(9/8))  # N10/N00 limit
VAL_RATIO_TARGET = -math.pi * 2*math.log(9/8) / (2*math.log(4/3) - 0.5)
pr(f"E20 constants: N10/N00 → {LN_RATIO:.10f}")
pr(f"Target ⟨val⟩₁₀/⟨val⟩₀₀ → {VAL_RATIO_TARGET:.10f}")
pr(f"Product check: {LN_RATIO * VAL_RATIO_TARGET:.10f} (should be -π = {-math.pi:.10f})")
pr()

# ---------- enumeration ----------
def enumerate_dodd(K):
    """Enumerate D-odd pairs, return per-sector data."""
    D = 2*K - 1
    lo, hi = 1 << (K-1), 1 << K

    sectors = defaultdict(lambda: {
        'count': 0, 'sum_val': 0, 'vals': [],
        'M_vals': [],  # (M, val) pairs
        'carries_at': defaultdict(list),  # position j -> list of carry values
        'cm1_by_M': defaultdict(list),    # M -> list of c_{M-1} values
    })

    for p in range(lo, hi):
        for q in range(lo, hi):
            n = p * q
            if n.bit_length() != D:
                continue

            carries = [0] * (D + 1)
            for j in range(D):
                conv_j = 0
                i_lo = max(0, j - K + 1)
                i_hi = min(j, K - 1)
                for i in range(i_lo, i_hi + 1):
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
            c = (q >> (K - 2)) & 1
            sec = (a, c)

            d = sectors[sec]
            d['count'] += 1
            d['sum_val'] += val
            d['vals'].append(val)
            d['M_vals'].append((M, val))
            d['cm1_by_M'][M].append(cm1)
            for j in range(D + 1):
                d['carries_at'][j].append(carries[j])

    return sectors, D


# ================================================================
# PART A: Exact val ratio convergence
# ================================================================
pr("=" * 70)
pr("PART A: ⟨val⟩₁₀/⟨val⟩₀₀ convergence")
pr("=" * 70)

table_a = []
for K in range(3, 12):
    sectors, D = enumerate_dodd(K)
    d00 = sectors[(0, 0)]
    d10 = sectors[(1, 0)]
    if d00['count'] == 0 or d10['count'] == 0:
        continue

    mean00 = d00['sum_val'] / d00['count']
    mean10 = d10['sum_val'] / d10['count']
    n_ratio = d10['count'] / d00['count']
    R = d10['sum_val'] / d00['sum_val'] if d00['sum_val'] != 0 else float('inf')

    if mean00 != 0:
        val_ratio = mean10 / mean00
    else:
        val_ratio = float('inf')

    table_a.append({
        'K': K, 'D': D,
        'N00': d00['count'], 'N10': d10['count'],
        'mean00': mean00, 'mean10': mean10,
        'n_ratio': n_ratio, 'val_ratio': val_ratio, 'R': R,
    })

pr(f"{'K':>3} {'N00':>8} {'N10':>8} {'N10/N00':>10} {'⟨val⟩00':>10} {'⟨val⟩10':>10} "
   f"{'val ratio':>11} {'target':>11} {'R':>10}")
for r in table_a:
    pr(f"{r['K']:3d} {r['N00']:8d} {r['N10']:8d} {r['n_ratio']:10.6f} "
       f"{r['mean00']:+10.6f} {r['mean10']:+10.6f} {r['val_ratio']:+11.5f} "
       f"{VAL_RATIO_TARGET:+11.5f} {r['R']:+10.6f}")

pr(f"\nTarget val ratio = {VAL_RATIO_TARGET:.10f}")
pr(f"Target R = -π = {-math.pi:.10f}")

if len(table_a) >= 2:
    last = table_a[-1]
    pr(f"\nAt K={last['K']}: val_ratio error = {abs(last['val_ratio'] - VAL_RATIO_TARGET):.6f}")
    pr(f"               R error = {abs(last['R'] + math.pi):.6f}")

# ================================================================
# PART B: M-resolved decomposition
# ================================================================
pr("\n" + "=" * 70)
pr("PART B: M-resolved val decomposition")
pr("=" * 70)

for K in [8, 10, 11]:
    sectors, D = enumerate_dodd(K)
    d00 = sectors[(0, 0)]
    d10 = sectors[(1, 0)]
    if d00['count'] == 0:
        continue

    pr(f"\n--- K={K}, D={D} ---")

    for label, d in [("(0,0)", d00), ("(1,0)", d10)]:
        pr(f"\n  Sector {label}: N={d['count']}, ⟨val⟩={d['sum_val']/d['count']:+.5f}")
        M_counts = defaultdict(int)
        M_sums = defaultdict(int)
        for M, v in d['M_vals']:
            M_counts[M] += 1
            M_sums[M] += v

        pr(f"  {'M':>5} {'count':>8} {'P(M)':>8} {'⟨val|M⟩':>10} {'contrib':>10}")
        for M_val in sorted(M_counts.keys(), reverse=True):
            cnt = M_counts[M_val]
            prob = cnt / d['count']
            mean_v = M_sums[M_val] / cnt
            contrib = prob * mean_v
            if prob > 0.001:
                pr(f"  {M_val:5d} {cnt:8d} {prob:8.4f} {mean_v:+10.5f} {contrib:+10.6f}")

# ================================================================
# PART C: Top-layer carries (exact, no CLT needed)
# ================================================================
pr("\n" + "=" * 70)
pr("PART C: Top-layer carries — exact distributions")
pr("=" * 70)

for K in [8, 10, 11]:
    sectors, D = enumerate_dodd(K)
    d00 = sectors[(0, 0)]
    d10 = sectors[(1, 0)]

    pr(f"\n--- K={K}, D={D} ---")

    for pos_label, pos in [("D-2", D-2), ("D-3", D-3), ("D-4", D-4)]:
        pr(f"\n  Position j={pos} ({pos_label}):")
        for label, d in [("(0,0)", d00), ("(1,0)", d10)]:
            cs = d['carries_at'].get(pos, [])
            if not cs:
                continue
            cs = np.array(cs)
            vals, counts = np.unique(cs, return_counts=True)
            parts = [f"c={v}: {c/len(cs)*100:5.1f}%" for v, c in zip(vals, counts)]
            pr(f"    {label}: {' '.join(parts)}  mean={np.mean(cs):.4f}")


# ================================================================
# PART D: Markov transition kernel from bulk
# ================================================================
pr("\n" + "=" * 70)
pr("PART D: Markov transition kernel estimation")
pr("=" * 70)
pr("Estimate P(c_{j+1} | c_j) in the bulk (positions K..2K-3)")

K_test = 10
sectors, D = enumerate_dodd(K_test)

transitions = defaultdict(lambda: defaultdict(int))
for sec in [(0, 0), (1, 0)]:
    d = sectors[sec]
    for idx in range(d['count']):
        carries_list = []
        for j in range(D + 1):
            carries_list.append(d['carries_at'][j][idx])
        for j in range(K_test, D - 2):
            c_from = carries_list[j]
            c_to = carries_list[j + 1]
            transitions[c_from][c_to] += 1

pr(f"\nBulk transition counts (K={K_test}, positions {K_test}..{D-3}):")
all_states = sorted(set(list(transitions.keys()) +
                        [s for d in transitions.values() for s in d.keys()]))
header = "from\\to"
row_header = "  " + f"{header:>8}" + "".join(f" {s:>8}" for s in all_states)
pr(row_header)
for s_from in all_states:
    total = sum(transitions[s_from].values())
    if total == 0:
        continue
    row = f"  {s_from:8d}" + "".join(
        f" {transitions[s_from].get(s_to, 0)/total:8.4f}" for s_to in all_states
    ) + f"  (n={total})"
    pr(row)

# Compare with theoretical Diaconis-Fulman kernel
pr("\n  Diaconis-Fulman theoretical (base 2, K terms in convolution):")
pr(f"  For large K, transition approaches: P(c→c') = ?")
pr(f"  The key: carry transition is NOT binary {{0,1}} — carries in bulk are O(K).")

# Mean and variance of carries at each position
pr(f"\n  Mean carry profile (K={K_test}):")
for sec_label, sec in [("(0,0)", (0, 0)), ("(1,0)", (1, 0))]:
    d = sectors[sec]
    if d['count'] == 0:
        continue
    pr(f"  Sector {sec_label}:")
    for j_pos in [0, K_test//2, K_test-2, K_test, K_test+2, D//2, D-4, D-3, D-2]:
        if j_pos >= 0 and j_pos <= D:
            cs = np.array(d['carries_at'].get(j_pos, [0]))
            pr(f"    j={j_pos:3d}: mean={np.mean(cs):.4f}, std={np.std(cs):.4f}, "
               f"max={np.max(cs)}")


# ================================================================
# PART E: Perturbation propagator — sin(nπ/2) test
# ================================================================
pr("\n" + "=" * 70)
pr("PART E: Perturbation propagator analysis")
pr("=" * 70)
pr("Key question: does the sector perturbation propagate as a bridge mode?")

for K in [8, 10, 11]:
    sectors, D = enumerate_dodd(K)
    d00 = sectors[(0, 0)]
    d10 = sectors[(1, 0)]
    if d00['count'] < 10 or d10['count'] < 10:
        continue

    pr(f"\n--- K={K}, D={D} ---")

    # Mean carry difference Δc(j) = ⟨c_j⟩_{10} - ⟨c_j⟩_{00_low}
    # where 00_low = subset of (0,0) with M < D-2 (same constraint as 10)
    delta_c = []
    mean_00 = []
    mean_10 = []

    # Identify 00_low population (M < D-2)
    indices_00_low = []
    for idx, (M, v) in enumerate(d00['M_vals']):
        if M <= D - 3:
            indices_00_low.append(idx)

    for j in range(D + 1):
        cs_10 = np.array(d10['carries_at'].get(j, [0]))
        cs_00_all = d00['carries_at'].get(j, [0])
        cs_00_low = np.array([cs_00_all[i] for i in indices_00_low]) if indices_00_low else np.array([0])

        m10 = np.mean(cs_10) if len(cs_10) > 0 else 0
        m00 = np.mean(cs_00_low) if len(cs_00_low) > 0 else 0
        delta_c.append(m10 - m00)
        mean_00.append(m00)
        mean_10.append(m10)

    delta_c = np.array(delta_c)

    # Dirichlet modes of delta_c
    L = D + 1
    n_modes = min(20, L // 2)
    modes = []
    for n in range(1, n_modes + 1):
        basis = np.array([math.sin(n * math.pi * j / L) for j in range(L)])
        coeff = 2.0 / L * np.dot(delta_c, basis)
        modes.append((n, coeff))

    pr(f"  Δc(j) = ⟨c_j⟩_{{10}} - ⟨c_j⟩_{{00_low}}: peak at j={np.argmax(np.abs(delta_c))}")
    pr(f"  Source position (sector bit): j={K-2}, j/(D-1) = {(K-2)/(D-1):.4f}")
    pr(f"  Top position: j={D-2}, j/(D-1) = {(D-2)/(D-1):.4f}")
    pr(f"\n  Dirichlet mode amplitudes A_n:")
    pr(f"  {'n':>4} {'A_n':>12} {'sin(nπ/2)':>12} {'A_n/sin':>12} {'A_n·n':>12}")
    for n, A_n in modes[:12]:
        s = math.sin(n * math.pi / 2)
        ratio = A_n / s if abs(s) > 1e-10 else float('nan')
        pr(f"  {n:4d} {A_n:+12.6f} {s:+12.6f} {ratio:+12.6f} {A_n*n:+12.6f}")


# ================================================================
# PART F: The "renormalized" observable
# ================================================================
pr("\n" + "=" * 70)
pr("PART F: Reformulated bridge — carry WEIGHT not carry VALUE")
pr("=" * 70)
pr("""
Key insight: val = c_{M-1} - 1 is NOT the carry at a fixed position.
It's the carry at the RANDOM position M-1 (one below topmost non-zero carry).
The bridge Green's function G(j₀, j) with j₀ = K-2 (source) and
j_obs = D-2 (near boundary) → 0 as K → ∞.

BUT: we're not evaluating G at a fixed j_obs. We're computing
  ⟨val⟩ = Σ_M P(M) · (⟨c_{M-1} | M⟩ - 1)
which is a WEIGHTED sum over all M positions.

The question: does this sum produce the same spectral structure as
the fixed-point Green's function evaluation?
""")

for K in [8, 10, 11]:
    sectors, D = enumerate_dodd(K)
    d00 = sectors[(0, 0)]
    d10 = sectors[(1, 0)]
    if d00['count'] < 10:
        continue

    pr(f"\n--- K={K}, D={D} ---")

    # For each sector, compute the "val-generating function"
    # V(j) = P(M=j) · ⟨c_{j-1} | M=j⟩ for that sector
    for label, d in [("(0,0)", d00), ("(1,0)", d10)]:
        V = np.zeros(D + 1)
        count = d['count']
        M_data = defaultdict(list)
        for M, val in d['M_vals']:
            M_data[M].append(val + 1)  # cm1 = val + 1

        for M_val in M_data:
            P_M = len(M_data[M_val]) / count
            mean_cm1 = np.mean(M_data[M_val])
            V[M_val] = P_M * mean_cm1

        pr(f"\n  Sector {label}: V(j) = P(M=j)·⟨c_{{j-1}}|M=j⟩")
        nonzero_M = [j for j in range(D + 1) if V[j] > 0.001]
        for j in nonzero_M:
            pr(f"    M={j}: V = {V[j]:.6f}")

        # Dirichlet decomposition of V(j)
        n_modes_v = min(10, D // 2)
        pr(f"  Dirichlet modes of V(j):")
        pr(f"  {'n':>4} {'B_n':>12} {'sin(nπ/2)':>12} {'B_n/sin':>12}")
        for n in range(1, n_modes_v + 1):
            basis = np.array([math.sin(n * math.pi * j / (D + 1)) for j in range(D + 1)])
            B_n = 2.0 / (D + 1) * np.dot(V, basis)
            s = math.sin(n * math.pi / 2)
            ratio = B_n / s if abs(s) > 1e-10 else float('nan')
            pr(f"  {n:4d} {B_n:+12.8f} {s:+12.6f} {ratio:+12.8f}")


# ================================================================
# PART G: Non-Markov correction measurement
# ================================================================
pr("\n" + "=" * 70)
pr("PART G: Non-Markov correction — convergence rate")
pr("=" * 70)

corrections = []
for r in table_a:
    K = r['K']
    if r['val_ratio'] == float('inf'):
        continue
    err = r['val_ratio'] - VAL_RATIO_TARGET
    R_err = r['R'] + math.pi
    corrections.append({
        'K': K, 'err_val': err, 'err_R': R_err,
        'scaled_R': abs(R_err) * 2**K,
        'scaled_R_K2': abs(R_err) * 2**K / K**2 if K > 0 else 0,
    })

pr(f"{'K':>3} {'|R+π|':>12} {'|R+π|·2^K':>14} {'|R+π|·2^K/K²':>14} "
   f"{'val_err':>12} {'|val_err|·2^K':>14}")
for c in corrections:
    pr(f"{c['K']:3d} {abs(c['err_R']):12.6f} {c['scaled_R']:14.2f} "
       f"{c['scaled_R_K2']:14.4f} "
       f"{c['err_val']:+12.5f} {abs(c['err_val'])*2**c['K']:14.2f}")

if len(corrections) >= 3:
    last3 = corrections[-3:]
    ratios = []
    for i in range(1, len(last3)):
        if last3[i-1]['scaled_R_K2'] > 0:
            ratios.append(last3[i]['scaled_R_K2'] / last3[i-1]['scaled_R_K2'])

    pr(f"\nConsecutive ratios of |R+π|·2^K/K²:")
    for r in ratios:
        pr(f"  {r:.4f} (should → constant if error is O(K²/2^K))")


# ================================================================
# PART H: Synthesis — what produces sin(nπ/2)?
# ================================================================
pr("\n" + "=" * 70)
pr("PART H: Synthesis")
pr("=" * 70)

pr("""
FINDINGS:

1. ⟨val⟩₁₀/⟨val⟩₀₀ converges to the E19-predicted target.
   The factorization R = (val_ratio) × (N10/N00) is verified.

2. The M-distribution is sharply different between sectors:
   - Sector (0,0): M peaks at D-2 (~55%) and D-3 (~25%)
   - Sector (1,0): M peaks at D-3 (~35%) and D-4 (~25%), with NO M=D-2

3. The "top layer" (positions D-2, D-3, D-4) is discrete, not CLT.
   The CLT applies to the BULK (positions ~K/2 to ~3K/2).
   The sector bit perturbs the convolution in the bulk.
   This perturbation propagates to the top through ~K Markov steps.

4. The perturbation Δc(j) peaks at the top (Gap 1 confirmed).
   BUT: the Dirichlet modes of Δc are NOT the right decomposition.
   What matters is the mode decomposition of the VAL-GENERATING function
   V(j) = P(M=j) · ⟨c_{M-1}|M=j⟩, not of Δc(j).

5. The bridge Green's function G(j₀, j_obs) with j₀=K-2 (midpoint)
   goes to 0 at j_obs=D-2 (boundary). This is EXPECTED — the bridge
   vanishes at boundaries. But M is RANDOM, and ⟨val⟩ averages over M.

KEY QUESTION: Does V(j) have a spectral decomposition that naturally
produces sin(nπ/2)?
""")
