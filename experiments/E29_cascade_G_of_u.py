#!/usr/bin/env python3
"""
E29: CASCADE vs SCHOOLBOOK val — exact G(u) comparison

CASCADE val (E01/E03): val = carries[M-1] - 1
  where M = top nonzero carry. This gives R → -π.

SCHOOLBOOK val (E3): val = 2·carry_into[D-2] - 1
  For sector (0,0): = 2·product_bit[D-2] - 1 (conv[D-2]=0)
  For sector (1,0): = -1 always (carries[D-2]=0 forced)
  This gives R → -3.93.

Key question: how do the G(u) = Σ_Y val(X,Y) profiles compare?
"""
import sys
import time
import math
import numpy as np

K = 12
if len(sys.argv) > 1:
    K = int(sys.argv[1])

D = 2 * K - 1


def pr(s=""):
    print(s, flush=True)


pr("=" * 78)
pr(f"E29: CASCADE vs SCHOOLBOOK val — K={K}, D={D}")
pr("=" * 78)
pr()

X_base = 1 << (K - 1)
half_range = 1 << (K - 2)
Y_range_arr = np.arange(X_base, X_base + half_range, dtype=np.int64)
P_min = 1 << (D - 1)
P_max = 1 << D

pr(f"  X_base = {X_base}, half_range = {half_range}")
pr(f"  D-odd: P ∈ [{P_min}, {P_max})")
pr()

N_STRIPS = min(100, half_range)
strip_size = max(1, half_range // N_STRIPS)


def compute_cascade_val_batch(X_int, Y_arr):
    """Compute cascade val for X * each Y in Y_arr.
    Returns (val_array, dodd_mask)."""
    P_arr = X_int * Y_arr
    d_odd = (P_arr >= P_min) & (P_arr < P_max)
    d_odd_idx = np.where(d_odd)[0]

    vals = np.zeros(len(Y_arr), dtype=np.int64)
    x_bits = [(int(X_int) >> i) & 1 for i in range(K)]

    for yi in d_odd_idx:
        Y = int(Y_arr[yi])
        y_bits = [(Y >> i) & 1 for i in range(K)]

        carries = [0] * (D + 1)
        for j in range(D):
            conv_j = 0
            i_lo = max(0, j - K + 1)
            i_hi = min(j, K - 1)
            for i in range(i_lo, i_hi + 1):
                conv_j += x_bits[i] * y_bits[j - i]
            carries[j + 1] = (conv_j + carries[j]) >> 1

        M = 0
        for j in range(D, 0, -1):
            if carries[j] > 0:
                M = j
                break
        if M == 0:
            continue

        vals[yi] = carries[M - 1] - 1

    return vals, d_odd


# ── Sector (0,0) ─────────────────────────────────────────────
pr("Computing sector (0,0)...")
t0 = time.time()

n_X = half_range
X_00_lo = X_base

G_casc_00 = np.zeros(n_X, dtype=np.int64)
G_school_00 = np.zeros(n_X, dtype=np.int64)
N_dodd_00 = np.zeros(n_X, dtype=np.int64)

for ix in range(n_X):
    X = X_00_lo + ix

    # Schoolbook: val = 2*product_bit[D-2] - 1 (vectorized, fast)
    P_arr = X * Y_range_arr
    d_odd = (P_arr >= P_min) & (P_arr < P_max)
    p_Dm2 = (P_arr >> (D - 2)) & 1
    school_vals = 2 * p_Dm2 - 1

    N_dodd_00[ix] = np.sum(d_odd)
    G_school_00[ix] = np.sum(school_vals[d_odd])

    # Cascade: slow per-pair loop
    casc_vals, _ = compute_cascade_val_batch(X, Y_range_arr)
    G_casc_00[ix] = np.sum(casc_vals[d_odd])

    if ix % max(1, n_X // 10) == 0:
        elapsed = time.time() - t0
        pr(f"  ix={ix}/{n_X}  ({elapsed:.1f}s)")

elapsed_00 = time.time() - t0
pr(f"  Done in {elapsed_00:.1f}s")

sigma_00_casc = int(np.sum(G_casc_00))
sigma_00_school = int(np.sum(G_school_00))
N_total_00 = int(np.sum(N_dodd_00))

pr(f"  σ₀₀_cascade  = {sigma_00_casc}")
pr(f"  σ₀₀_schoolbook = {sigma_00_school}")
pr(f"  N_00          = {N_total_00}")
pr()

# ── Sector (1,0) ─────────────────────────────────────────────
pr("Computing sector (1,0)...")
t0 = time.time()

X_10_lo = X_base + half_range

G_casc_10 = np.zeros(n_X, dtype=np.int64)
N_dodd_10 = np.zeros(n_X, dtype=np.int64)

for ix in range(n_X):
    X = X_10_lo + ix

    P_arr = X * Y_range_arr
    d_odd = (P_arr >= P_min) & (P_arr < P_max)
    N_dodd_10[ix] = np.sum(d_odd)

    # Cascade val
    casc_vals, _ = compute_cascade_val_batch(X, Y_range_arr)
    G_casc_10[ix] = np.sum(casc_vals[d_odd])

    if ix % max(1, n_X // 10) == 0:
        elapsed = time.time() - t0
        pr(f"  ix={ix}/{n_X}  ({elapsed:.1f}s)")

elapsed_10 = time.time() - t0
pr(f"  Done in {elapsed_10:.1f}s")

sigma_10_casc = int(np.sum(G_casc_10))
sigma_10_school = -int(np.sum(N_dodd_10))  # schoolbook: val=-1 always
N_total_10 = int(np.sum(N_dodd_10))

pr(f"  σ₁₀_cascade   = {sigma_10_casc}")
pr(f"  σ₁₀_schoolbook = {sigma_10_school}  (= -N₁₀)")
pr(f"  N_10           = {N_total_10}")
pr()

# ── R comparison ──────────────────────────────────────────────
R_casc = sigma_10_casc / sigma_00_casc if sigma_00_casc != 0 else 0
R_school = sigma_10_school / sigma_00_school if sigma_00_school != 0 else 0

pr("=" * 78)
pr("R = σ₁₀/σ₀₀ COMPARISON")
pr("=" * 78)
pr()
pr(f"  CASCADE:    R = {R_casc:.12f}  |R+π| = {abs(R_casc + math.pi):.6e}")
pr(f"  SCHOOLBOOK: R = {R_school:.12f}  |R+π| = {abs(R_school + math.pi):.6e}")
pr(f"  -π =          {-math.pi:.12f}")
pr()

# ── G(u) profiles ─────────────────────────────────────────────
pr("=" * 78)
pr("⟨val⟩(u) PROFILES")
pr("-" * 60)
pr()

# Bin into strips
def bin_profile(G_arr, N_arr, n_strips, strip_sz):
    Gs = np.zeros(n_strips)
    Ns = np.zeros(n_strips)
    for s in range(n_strips):
        lo = s * strip_sz
        hi = min(lo + strip_sz, len(G_arr))
        Gs[s] = np.sum(G_arr[lo:hi])
        Ns[s] = np.sum(N_arr[lo:hi])
    avg = np.where(Ns > 0, Gs / Ns, 0)
    return avg, Ns

avg_00_casc, N_00_strip = bin_profile(G_casc_00, N_dodd_00, N_STRIPS, strip_size)
avg_00_school, _ = bin_profile(G_school_00, N_dodd_00, N_STRIPS, strip_size)
avg_10_casc, N_10_strip = bin_profile(G_casc_10, N_dodd_10, N_STRIPS, strip_size)

pr("Sector (0,0):")
pr(f"  {'u_mid':>8s}  {'⟨val⟩_casc':>12s}  {'⟨val⟩_school':>12s}  {'Δ':>10s}")
step = max(1, N_STRIPS // 25)
for s in range(0, N_STRIPS, step):
    u_mid = (s + 0.5) / N_STRIPS * 0.5
    d = avg_00_casc[s] - avg_00_school[s]
    pr(f"  {u_mid:8.4f}  {avg_00_casc[s]:12.6f}  {avg_00_school[s]:12.6f}  {d:10.6f}")

pr()
pr("Sector (1,0):")
pr(f"  {'u_mid':>8s}  {'⟨val⟩_casc':>12s}  {'⟨val⟩_school':>12s}")
for s in range(0, N_STRIPS, step):
    u_mid = 0.5 + (s + 0.5) / N_STRIPS * 0.5
    pr(f"  {u_mid:8.4f}  {avg_10_casc[s]:12.6f}  {'  -1.000000':>12s}")

pr()

# ── Smoothness metrics ────────────────────────────────────────
pr("=" * 78)
pr("SMOOTHNESS (sector 0,0)")
pr("-" * 60)
pr()

dc = np.abs(np.diff(avg_00_casc))
ds = np.abs(np.diff(avg_00_school))

pr(f"  Cascade:    range=[{avg_00_casc.min():.6f}, {avg_00_casc.max():.6f}]  "
   f"std={avg_00_casc.std():.6f}  max|Δ|={dc.max():.6f}")
pr(f"  Schoolbook: range=[{avg_00_school.min():.6f}, {avg_00_school.max():.6f}]  "
   f"std={avg_00_school.std():.6f}  max|Δ|={ds.max():.6f}")
pr()

mono_s = all(avg_00_school[i] <= avg_00_school[i + 1]
             for i in range(len(avg_00_school) - 1))
mono_c = all(avg_00_casc[i] <= avg_00_casc[i + 1]
             for i in range(len(avg_00_casc) - 1))
pr(f"  Monotonic: cascade={mono_c}  schoolbook={mono_s}")
pr()

# ── Decomposition: cascade = schoolbook + correction ──────────
pr("=" * 78)
pr("DECOMPOSITION: cascade = schoolbook + correction")
pr("-" * 60)
pr()

delta_00 = avg_00_casc - avg_00_school
pr(f"  Correction Δ = ⟨val⟩_casc - ⟨val⟩_school for sector (0,0):")
pr(f"  {'u_mid':>8s}  {'Δ(u)':>12s}")
for s in range(0, N_STRIPS, step):
    u_mid = (s + 0.5) / N_STRIPS * 0.5
    pr(f"  {u_mid:8.4f}  {delta_00[s]:12.6f}")

pr()
pr(f"  Δ range: [{delta_00.min():.6f}, {delta_00.max():.6f}]")
pr(f"  Δ mean:  {delta_00.mean():.6f}")
pr(f"  Δ std:   {delta_00.std():.6f}")
dd = np.abs(np.diff(delta_00))
pr(f"  max|ΔΔ|: {dd.max():.6f}")
pr()

# ── Where does M fall? ────────────────────────────────────────
pr("=" * 78)
pr("WHERE IS M (top nonzero carry)?")
pr("-" * 60)
pr()

# Sample some pairs to see M distribution
M_hist = np.zeros(D + 1, dtype=np.int64)
n_sample = min(500, n_X)

for ix in range(0, n_X, max(1, n_X // n_sample)):
    X = X_00_lo + ix
    for yi in range(0, len(Y_range_arr), max(1, len(Y_range_arr) // 50)):
        Y = int(Y_range_arr[yi])
        P = X * Y
        if P < P_min or P >= P_max:
            continue

        x_bits = [(X >> i) & 1 for i in range(K)]
        y_bits = [(Y >> i) & 1 for i in range(K)]

        carries = [0] * (D + 1)
        for j in range(D):
            conv_j = 0
            i_lo = max(0, j - K + 1)
            i_hi = min(j, K - 1)
            for i in range(i_lo, i_hi + 1):
                conv_j += x_bits[i] * y_bits[j - i]
            carries[j + 1] = (conv_j + carries[j]) >> 1

        for j in range(D, 0, -1):
            if carries[j] > 0:
                M_hist[j] += 1
                break

pr(f"  M distribution (D-2={D-2}):")
for j in range(max(0, D - 6), D + 1):
    pct = 100 * M_hist[j] / max(1, M_hist.sum())
    bar = "#" * int(pct / 2)
    pr(f"    M={j:3d}: {M_hist[j]:8d} ({pct:5.1f}%)  {bar}")

pr()

# ── SUMMARY ───────────────────────────────────────────────────
pr("=" * 78)
pr("SUMMARY")
pr("=" * 78)
pr()
pr(f"  K = {K}, D = {D}")
pr(f"  N_00 = {N_total_00}, N_10 = {N_total_10}")
pr()
pr(f"  R_cascade    = {R_casc:.10f}   (→ -π = {-math.pi:.10f})")
pr(f"  R_schoolbook = {R_school:.10f}   (→ -3.93...)")
pr()
pr(f"  ΔR = R_casc - R_school = {R_casc - R_school:.10f}")
pr(f"  This ΔR is the 'carry chain correction' from the cascade structure.")
pr()
pr("  CONCLUSION: The two val definitions produce different R limits.")
pr("  Only the CASCADE val (top nonzero carry) gives R → -π.")
pr("  The SCHOOLBOOK val (carry at fixed position D-2) gives R → -3.93.")
pr()
pr("=" * 78)
