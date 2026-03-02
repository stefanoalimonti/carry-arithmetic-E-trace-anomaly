#!/usr/bin/env python3
"""
E28: Exact G(u) via Integer Enumeration — Ground Truth

For specific K, enumerate ALL (X,Y) pairs in each sector.
Compute val = 2·carry_{D-2} - 1 using the schoolbook carry chain.
G(X) = Σ_Y val(X,Y) for each X, giving the EXACT 1D profile.

Key insight (from carry chain analysis):
  - Sector (1,0): conv[D-2] = a₁ + c₁ = 1 + 0 = 1.
    carry[D-2] = p_{D-2} - conv[D-2] = p_{D-2} - 1.
    Since carry ≥ 0: p_{D-2} = 1 always, carry[D-2] = 0, val = -1.
    → G_{10}(X) = −N_{D-odd}(X)  (trivially smooth!)

  - Sector (0,0): conv[D-2] = 0, carry[D-2] = p_{D-2}.
    val = 2·p_{D-2} - 1 = ±1 depending on bit D-2 of product.
    → G_{00}(X) = 2·#{p_{D-2}=1 among D-odd} − N_{D-odd}(X)

WARNING: The schoolbook val (= 2·carry[D-2]−1) and the cascade val
(= carries[M-1]−1, M=top nonzero carry) give DIFFERENT R limits!
  - Schoolbook: R → −3.93 (NOT −π)
  - Cascade (E01/E03): R → −π
See above for the comparison. The schoolbook framework captures only
the "zeroth-order geometric" component R₀ ≈ −3.93. The full −π
requires the cascade correction from pairs where M ≠ D−2.
"""
import sys
import time
import math
import numpy as np

K = 15
if len(sys.argv) > 1:
    K = int(sys.argv[1])

D = 2 * K - 1


def pr(s=""):
    print(s, flush=True)


pr("=" * 78)
pr(f"E28: Exact G(u) via Integer Enumeration — K={K}, D={D}")
pr("=" * 78)
pr()

# Sector ranges
# X = 2^{K-1}·(1+u), MSB = x_{K-1} = 1
# Sector bit = x_{K-2}: 0 for (0,0), 1 for (1,0)
# Y: MSB = 1, sector bit y_{K-2} = 0 (c_1 = 0)

X_base = 1 << (K - 1)  # 2^{K-1}
half_range = 1 << (K - 2)  # 2^{K-2}

X_00 = np.arange(X_base, X_base + half_range, dtype=np.int64)
X_10 = np.arange(X_base + half_range, X_base + 2 * half_range, dtype=np.int64)
Y_range = np.arange(X_base, X_base + half_range, dtype=np.int64)

P_min = 1 << (D - 1)
P_max = 1 << D

pr(f"  X range (0,0): [{X_00[0]}, {X_00[-1]}]  ({len(X_00)} values)")
pr(f"  X range (1,0): [{X_10[0]}, {X_10[-1]}]  ({len(X_10)} values)")
pr(f"  Y range:       [{Y_range[0]}, {Y_range[-1]}]  ({len(Y_range)} values)")
pr(f"  D-odd: P ∈ [{P_min}, {P_max})")
pr()

# ── Sector (0,0) ─────────────────────────────────────────────
pr("Computing sector (0,0)...")
t0 = time.time()

G_00 = np.zeros(len(X_00), dtype=np.int64)
N_00 = np.zeros(len(X_00), dtype=np.int64)  # D-odd count
N_carry1_00 = np.zeros(len(X_00), dtype=np.int64)  # p_{D-2}=1 count

for ix, X in enumerate(X_00):
    P_arr = X * Y_range  # vectorized multiplication
    d_odd = (P_arr >= P_min) & (P_arr < P_max)
    p_Dm2 = (P_arr >> (D - 2)) & 1
    val_arr = 2 * p_Dm2 - 1

    N_00[ix] = np.sum(d_odd)
    N_carry1_00[ix] = np.sum(d_odd & (p_Dm2 == 1))
    G_00[ix] = np.sum(val_arr[d_odd])

sigma_00 = np.sum(G_00)
N_total_00 = np.sum(N_00)
elapsed_00 = time.time() - t0
pr(f"  Done in {elapsed_00:.1f}s")
pr(f"  σ₀₀ = Σ G_00 = {sigma_00}")
pr(f"  N_00 (D-odd) = {N_total_00}")
pr(f"  N_carry1_00 = {np.sum(N_carry1_00)}")
pr()

# ── Sector (1,0) ─────────────────────────────────────────────
pr("Computing sector (1,0)...")
t0 = time.time()

G_10 = np.zeros(len(X_10), dtype=np.int64)
N_10 = np.zeros(len(X_10), dtype=np.int64)

for ix, X in enumerate(X_10):
    P_arr = X * Y_range
    d_odd = (P_arr >= P_min) & (P_arr < P_max)
    N_10[ix] = np.sum(d_odd)
    # val = -1 for ALL D-odd pairs in (1,0) → G = -N
    G_10[ix] = -N_10[ix]

sigma_10 = np.sum(G_10)
N_total_10 = np.sum(N_10)
elapsed_10 = time.time() - t0
pr(f"  Done in {elapsed_10:.1f}s")
pr(f"  σ₁₀ = Σ G_10 = {sigma_10}  (= −N_10 = {-N_total_10})")
pr(f"  N_10 (D-odd) = {N_total_10}")
pr()

R = sigma_10 / sigma_00 if sigma_00 != 0 else 0
pr(f"  R = σ₁₀/σ₀₀ = {R:.12f}")
pr(f"  −π         = {-math.pi:.12f}")
pr(f"  |R+π|      = {abs(R + math.pi):.6e}")
pr()

# ── G(u) profiles ─────────────────────────────────────────────
pr("=" * 78)
pr("G(u) Profiles — exact for K=" + str(K))
pr("-" * 60)
pr()

# Convert X index to u
u_00 = (X_00.astype(float) / X_base - 1.0)
u_10 = (X_10.astype(float) / X_base - 1.0)

# Bin into strips for smoother visualization
N_STRIPS = min(200, half_range)
strip_size = max(1, half_range // N_STRIPS)

pr(f"  Binned into {N_STRIPS} strips (strip_size={strip_size} X-values)")
pr()

G_00_binned = np.zeros(N_STRIPS)
N_00_binned = np.zeros(N_STRIPS)
G_10_binned = np.zeros(N_STRIPS)
N_10_binned = np.zeros(N_STRIPS)

for s in range(N_STRIPS):
    lo = s * strip_size
    hi = min(lo + strip_size, half_range)
    G_00_binned[s] = np.sum(G_00[lo:hi])
    N_00_binned[s] = np.sum(N_00[lo:hi])
    G_10_binned[s] = np.sum(G_10[lo:hi])
    N_10_binned[s] = np.sum(N_10[lo:hi])

# ⟨val⟩ per strip
pr("  Sector (0,0): G(u) and ⟨val⟩(u)")
pr(f"  {'strip':>6s}  {'u_mid':>8s}  {'G':>12s}  {'N':>8s}  {'⟨val⟩':>10s}")
step = max(1, N_STRIPS // 30)
for s in range(0, N_STRIPS, step):
    u_mid = (s + 0.5) / N_STRIPS * 0.5  # u ∈ [0, 0.5)
    avg_v = G_00_binned[s] / N_00_binned[s] if N_00_binned[s] > 0 else 0
    pr(f"  {s:6d}  {u_mid:8.4f}  {G_00_binned[s]:12.0f}  "
       f"{N_00_binned[s]:8.0f}  {avg_v:10.6f}")

pr()
pr("  Sector (1,0): G(u) = −N(u), ⟨val⟩ = −1 everywhere")
pr(f"  {'strip':>6s}  {'u_mid':>8s}  {'G':>12s}  {'N':>8s}  {'⟨val⟩':>10s}")
for s in range(0, N_STRIPS, step):
    u_mid = 0.5 + (s + 0.5) / N_STRIPS * 0.5  # u ∈ [0.5, 1)
    avg_v = G_10_binned[s] / N_10_binned[s] if N_10_binned[s] > 0 else 0
    pr(f"  {s:6d}  {u_mid:8.4f}  {G_10_binned[s]:12.0f}  "
       f"{N_10_binned[s]:8.0f}  {avg_v:10.6f}")

pr()

# ── Smoothness of G_00(u) ────────────────────────────────────
pr("=" * 78)
pr("Smoothness Analysis of G₀₀(u)")
pr("-" * 60)
pr()

avg_val_00_strips = []
for s in range(N_STRIPS):
    if N_00_binned[s] > 0:
        avg_val_00_strips.append(G_00_binned[s] / N_00_binned[s])
    else:
        avg_val_00_strips.append(0.0)

avg_val_00_arr = np.array(avg_val_00_strips)
diffs = np.abs(np.diff(avg_val_00_arr))

pr(f"  ⟨val⟩₀₀ across {N_STRIPS} strips:")
pr(f"    Range: [{avg_val_00_arr.min():.6f}, {avg_val_00_arr.max():.6f}]")
pr(f"    Mean:  {avg_val_00_arr.mean():.6f}")
pr(f"    Std:   {avg_val_00_arr.std():.6f}")
pr(f"    Max |Δ⟨val⟩|:  {diffs.max():.6f}")
pr(f"    Mean |Δ⟨val⟩|: {diffs.mean():.6f}")
pr()

# Per-X (highest resolution) analysis
G_00_float = G_00.astype(float)
N_00_float = N_00.astype(float)
val_perX = np.where(N_00_float > 0, G_00_float / N_00_float, 0)

pr(f"  Per-X (finest resolution, {len(X_00)} points):")
pr(f"    ⟨val⟩₀₀ range: [{val_perX.min():.6f}, {val_perX.max():.6f}]")
pr(f"    ⟨val⟩₀₀ mean:  {val_perX[N_00 > 0].mean():.6f}")
pr(f"    ⟨val⟩₀₀ std:   {val_perX[N_00 > 0].std():.6f}")
pr()

# How many X values have val > 0 vs < 0?
n_pos = np.sum(G_00 > 0)
n_neg = np.sum(G_00 < 0)
n_zero = np.sum(G_00 == 0)
pr(f"    X with G>0: {n_pos} ({100*n_pos/len(G_00):.1f}%)")
pr(f"    X with G<0: {n_neg} ({100*n_neg/len(G_00):.1f}%)")
pr(f"    X with G=0: {n_zero} ({100*n_zero/len(G_00):.1f}%)")
pr()

# ── Is val_perX smooth or fractal? ───────────────────────────
pr("  Hölder exponent (per-X resolution):")
diffs_perX = np.abs(np.diff(val_perX))
h = 1.0 / len(val_perX)
valid_diffs = diffs_perX[diffs_perX > 1e-10]
if len(valid_diffs) > 0:
    log_diffs = np.log(valid_diffs) / np.log(h)
    pr(f"    Mean α = {log_diffs.mean():.4f}")
    pr(f"    α ≈ 1 → Lipschitz, α < 1 → Hölder (rough), α > 1 → smooth")
pr()

# ── Does ⟨val⟩₀₀(u) have structure? Fourier analysis ────────
pr("=" * 78)
pr("Fourier Analysis of ⟨val⟩₀₀(u)")
pr("-" * 60)
pr()

# FFT of the binned ⟨val⟩ profile
fft_vals = np.fft.rfft(avg_val_00_arr)
fft_power = np.abs(fft_vals) ** 2
freqs = np.arange(len(fft_power))

pr(f"  Top 15 Fourier components of ⟨val⟩₀₀:")
pr(f"  {'n':>6s}  {'|c_n|²':>14s}  {'|c_n|':>10s}  {'period':>10s}")
top_idx = np.argsort(fft_power)[::-1][:15]
for idx in top_idx:
    n = freqs[idx]
    power = fft_power[idx]
    amp = math.sqrt(power)
    period = N_STRIPS / n if n > 0 else float('inf')
    pr(f"  {n:6d}  {power:14.4e}  {amp:10.6f}  {period:10.2f}")

pr()

# ── K convergence ─────────────────────────────────────────────
pr("=" * 78)
pr("SUMMARY")
pr("=" * 78)
pr()
pr(f"  K = {K}, D = {D}")
pr(f"  N_00 (D-odd pairs) = {N_total_00}")
pr(f"  N_10 (D-odd pairs) = {N_total_10}")
pr(f"  N_10/N_00 = {N_total_10/N_total_00:.10f}")
pr(f"  (exact limit = {2*math.log(4/3)-0.5:.10f} / {2*math.log(9/8):.10f} "
   f"= {(2*math.log(4/3)-0.5)/(2*math.log(9/8)):.10f})")
pr()
pr(f"  σ₀₀ = {sigma_00}")
pr(f"  σ₁₀ = {sigma_10}")
pr(f"  R = σ₁₀/σ₀₀ = {R:.12f}")
pr(f"  −π = {-math.pi:.12f}")
pr(f"  |R+π| = {abs(R + math.pi):.6e}")
pr()
pr("  In the schoolbook framework:")
pr("    val = -1 for ALL sector (1,0) pairs")
pr("    val = 2·p_{D-2} - 1 for sector (0,0) pairs")
pr("    σ₀₀ = 2·(#carry=1) − N₀₀")
pr("    σ₁₀ = −N₁₀")
pr()
pr(f"    fraction carry=1 in (0,0): "
   f"{np.sum(N_carry1_00)/N_total_00:.10f}")
pr(f"    → σ₀₀/N₀₀ = ⟨val⟩₀₀ = "
   f"{sigma_00/N_total_00:.10f}")
pr()
pr("  KEY INSIGHT: In the schoolbook framework, π enters through")
pr("  the fraction of sector (0,0) pairs with carry_{D-2} = 1.")
pr("  This fraction determines σ₀₀, and R = −N₁₀/σ₀₀ → −π")
pr("  means σ₀₀ → N₁₀/π in the limit.")
pr()
pr("=" * 78)
