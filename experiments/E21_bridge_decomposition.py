#!/usr/bin/env python3
"""
E21: Bridge decomposition — Top layer + Bulk Markov

Strategy: decompose the carry chain into:
  1. BULK (positions 0 to j₀): carry chain is approximately Markov
  2. TOP LAYER (positions j₀ to D-2): few convolution terms, EXACT calculation

The sector perturbation modifies the bulk, which feeds into the top layer.
The top layer converts the incoming carry distribution into the val statistic.

Key questions:
  A. Can we reproduce the val ratio using a Markov bulk + exact top?
  B. Where does the Markov approximation break down?
  C. What is the incoming carry distribution at the junction point?
  D. Does the difference in incoming carries between sectors produce π?
"""

import numpy as np
from collections import defaultdict
import math

def pr(s=""):
    print(s)

TARGET_VR = -math.pi * 2*math.log(9/8) / (2*math.log(4/3) - 0.5)

# ---------- enumeration ----------
def enumerate_full(K):
    """Full enumeration returning per-pair carry chains."""
    D = 2*K - 1
    lo, hi = 1 << (K-1), 1 << K

    results = []
    for p in range(lo, hi):
        for q in range(lo, hi):
            n = p * q
            if n.bit_length() != D:
                continue

            carries = [0] * (D + 1)
            convs = [0] * D
            for j in range(D):
                conv_j = 0
                i_lo = max(0, j - K + 1)
                i_hi = min(j, K - 1)
                for i in range(i_lo, i_hi + 1):
                    conv_j += ((p >> i) & 1) * ((q >> (j - i)) & 1)
                convs[j] = conv_j
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

            results.append({
                'a': a, 'c': c_sec, 'M': M, 'val': val,
                'carries': carries, 'convs': convs,
            })

    return results, D


# ================================================================
# PART A: Junction point analysis
# ================================================================
pr("=" * 70)
pr("PART A: Incoming carry distribution at junction point j₀ = D-4")
pr("=" * 70)
pr("The top layer (D-4 to D-2) involves 4, 3, 2 terms in convolution.")
pr("The carry at j₀ = D-4 encodes the bulk state.")

for K in [8, 10, 11]:
    results, D = enumerate_full(K)
    j0 = D - 4

    pr(f"\n--- K={K}, D={D}, j₀={j0} ---")

    sec_data = defaultdict(lambda: {'carries_j0': [], 'vals': [], 'M_vals': []})
    for r in results:
        sec = (r['a'], r['c'])
        sec_data[sec]['carries_j0'].append(r['carries'][j0])
        sec_data[sec]['vals'].append(r['val'])
        sec_data[sec]['M_vals'].append((r['M'], r['val']))

    for label, sec in [("(0,0)", (0,0)), ("(1,0)", (1,0))]:
        d = sec_data[sec]
        cs = np.array(d['carries_j0'])
        vals = np.array(d['vals'])

        pr(f"\n  Sector {label}: N={len(cs)}")
        unique_c, counts_c = np.unique(cs, return_counts=True)
        pr(f"  P(c_{{j0}}) distribution:")
        for c_val, cnt in zip(unique_c, counts_c):
            mask = cs == c_val
            mean_v = np.mean(vals[mask])
            pr(f"    c={c_val}: P={cnt/len(cs):.4f}, ⟨val|c⟩={mean_v:+.5f}, "
               f"contrib={cnt/len(cs) * mean_v:+.6f}")
        pr(f"  ⟨c_j0⟩ = {np.mean(cs):.5f}, ⟨val⟩ = {np.mean(vals):+.6f}")


# ================================================================
# PART B: Top-layer transfer function
# ================================================================
pr("\n" + "=" * 70)
pr("PART B: Top-layer transfer function T(val | c_{j0}, sector)")
pr("=" * 70)
pr("Given incoming carry c at j₀ = D-4 and sector, what is ⟨val⟩?")

for K in [10, 11]:
    results, D = enumerate_full(K)
    j0 = D - 4

    pr(f"\n--- K={K}, D={D} ---")

    for label, sec in [("(0,0)", (0,0)), ("(1,0)", (1,0))]:
        pr(f"\n  Sector {label}:")
        pr(f"  {'c_j0':>5} {'count':>7} {'⟨val⟩':>9} {'⟨M⟩':>7} {'P(M=D-2)':>10}")

        pairs = [(r['carries'][j0], r['val'], r['M'])
                 for r in results if (r['a'], r['c']) == sec]
        if not pairs:
            continue

        cs_arr = np.array([x[0] for x in pairs])
        vals_arr = np.array([x[1] for x in pairs])
        Ms_arr = np.array([x[2] for x in pairs])

        for c_val in sorted(np.unique(cs_arr)):
            mask = cs_arr == c_val
            cnt = np.sum(mask)
            if cnt < 5:
                continue
            mean_v = np.mean(vals_arr[mask])
            mean_M = np.mean(Ms_arr[mask])
            p_top = np.mean(Ms_arr[mask] == D-2)
            pr(f"  {c_val:5d} {cnt:7d} {mean_v:+9.5f} {mean_M:7.2f} {p_top:10.4f}")


# ================================================================
# PART C: Sector perturbation at the junction
# ================================================================
pr("\n" + "=" * 70)
pr("PART C: How the sector bit shifts the incoming carry distribution")
pr("=" * 70)

for K in [8, 10, 11]:
    results, D = enumerate_full(K)
    j0 = D - 4

    pr(f"\n--- K={K}, D={D} ---")

    # Get carry distributions at several positions
    for j_pos in [D-4, D-5, D-6, K-1, K-2]:
        if j_pos < 0 or j_pos > D:
            continue
        cs_00 = [r['carries'][j_pos] for r in results if (r['a'], r['c']) == (0,0)]
        cs_10 = [r['carries'][j_pos] for r in results if (r['a'], r['c']) == (1,0)]
        if not cs_00 or not cs_10:
            continue

        mean_diff = np.mean(cs_10) - np.mean(cs_00)
        std_00 = np.std(cs_00)
        frac = j_pos / (D - 1)
        pr(f"  j={j_pos:3d} (j/L={frac:.3f}): "
           f"⟨c⟩₀₀={np.mean(cs_00):.4f}, ⟨c⟩₁₀={np.mean(cs_10):.4f}, "
           f"Δ={mean_diff:+.4f}, Δ/σ={mean_diff/std_00 if std_00>0 else 0:+.4f}")


# ================================================================
# PART D: The "bridge argument" reformulated
# ================================================================
pr("\n" + "=" * 70)
pr("PART D: Bridge argument — response function")
pr("=" * 70)
pr("""
The bridge Green's function argument:
  G(j₀, j) = Σ_n φ_n(j₀) · φ_n(j) / (1 - λ_n)
where φ_n(j) = sin(nπj/L) for Dirichlet bridge.

At j₀ = K-2 (source), j₀/L → 1/2, so φ_n(j₀) → sin(nπ/2) = χ₄(n).

But we're NOT evaluating G at a fixed j. We compute:
  ⟨val⟩ = Σ_M P(M) · (⟨c_{M-1}|M⟩ - 1) = ∫ T(j) · P_bulk(j) dj
where T is the top-layer transfer and P_bulk is the bridge distribution.

KEY TEST: compute the "response" at each position j (= junction point)
and check if the sector DIFFERENCE follows a bridge mode pattern.
""")

for K in [10, 11]:
    results, D = enumerate_full(K)

    pr(f"\n--- K={K}, D={D} ---")

    # For each junction point j₀, compute ⟨val⟩ by sector
    # conditioned on j₀ being "the interface"
    pr(f"  {'j₀':>4} {'j₀/L':>6} {'⟨val⟩₀₀':>10} {'⟨val⟩₁₀':>10} {'Δval':>10} "
       f"{'sin(πj/L)':>10}")

    L = D - 1
    for j0 in range(max(1, K-4), D-1):
        vals_00 = [r['val'] for r in results if (r['a'], r['c']) == (0,0)]
        vals_10 = [r['val'] for r in results if (r['a'], r['c']) == (1,0)]
        if not vals_00 or not vals_10:
            continue

        # val doesn't depend on j₀ directly (it's a property of the pair)
        # What depends on j₀ is the carry AT that position.
        # The carry at j₀ encodes "how much bulk info propagated"
        # Let me instead compute: for pairs with carries[j₀] = c,
        # what is ⟨val⟩ by sector?
        # Skip this — val is fixed per pair. What changes with j₀ is what
        # we condition on.
        pass

    # Instead: compute the EFFECTIVE perturbation as seen by the val
    # The val = c_{M-1} - 1 depends on the carry at M-1.
    # For the dominant M values, let's trace the carry perturbation
    # at specific positions.

    # The "integrated response" is just ⟨val⟩ itself.
    # But the question is: does the carry perturbation at j₀ = K-2
    # produce a val perturbation that's proportional to sin(nπ/2)?

    # Alternative approach: decompose the perturbation of ⟨val⟩ into
    # contributions from different "frequency bands" of the carry chain.

    # For each pair, the "carry difference" is the difference between
    # the actual carry chain and a reference. Let's use the sector-averaged
    # carry as reference.

    # Mean carry by sector
    mean_c_00 = np.zeros(D + 1)
    mean_c_10 = np.zeros(D + 1)
    n_00 = n_10 = 0
    for r in results:
        if (r['a'], r['c']) == (0, 0):
            for j in range(D + 1):
                mean_c_00[j] += r['carries'][j]
            n_00 += 1
        elif (r['a'], r['c']) == (1, 0):
            for j in range(D + 1):
                mean_c_10[j] += r['carries'][j]
            n_10 += 1
    mean_c_00 /= max(n_00, 1)
    mean_c_10 /= max(n_10, 1)

    delta_mean = mean_c_10 - mean_c_00

    # The response function R(j) = Δ⟨c_j⟩ tells us how the carry
    # perturbation propagates. This is the Green's function G(K-2, j)
    # convolved with the perturbation profile.
    #
    # For a delta-function perturbation at j₀, the response would be:
    #   G(j₀, j) = Σ_n sin(nπj₀/L) sin(nπj/L) / (1-λ_n)
    #
    # The sector bit changes conv at MULTIPLE positions [K-2, D-2],
    # so Δ⟨c_j⟩ is G convolved with a "step" from K-2 to D-2.

    pr(f"\n  Carry perturbation Δ⟨c_j⟩ = ⟨c_j⟩₁₀ - ⟨c_j⟩₀₀:")
    pr(f"  {'j':>4} {'j/L':>6} {'Δc':>10} {'⟨c⟩₀₀':>10} {'⟨c⟩₁₀':>10}")
    for j in range(0, D + 1, max(1, D // 20)):
        frac = j / L if L > 0 else 0
        pr(f"  {j:4d} {frac:6.3f} {delta_mean[j]:+10.5f} "
           f"{mean_c_00[j]:10.5f} {mean_c_10[j]:10.5f}")

    # The perturbation has a "ramp" shape from K-2 to D-2, NOT a delta function.
    # This is because the sector bit adds one extra term to conv at EACH position
    # from K-2 to D-2 (the extra term is q_{j-K+2}).
    #
    # In mode space, a ramp from K-2 to D-2 has modes:
    # A_n = (2/L) ∫_{K-2}^{D-2} f(j) sin(nπj/L) dj
    # For a constant amplitude ramp (Δconv = 1/2 at each position):
    # A_n ∝ [cos(nπ(K-2)/L) - cos(nπ(D-2)/L)] / (nπ/L)
    #      → [cos(nπ/2) - cos(nπ)] / (nπ/L)  as K→∞

    pr(f"\n  Predicted mode amplitudes from ramp perturbation (continuous limit):")
    pr(f"  A_n ∝ [cos(nπ/2) - cos(nπ)] / n")
    pr(f"  {'n':>4} {'cos(nπ/2)-cos(nπ)':>20} {'predicted':>12} {'χ₄(n)·smth':>12}")
    for n in range(1, 13):
        diff_cos = math.cos(n*math.pi/2) - math.cos(n*math.pi)
        s = math.sin(n*math.pi/2)
        pr(f"  {n:4d} {diff_cos:+20.6f} {diff_cos/n:+12.6f} {s:+12.6f}")


# ================================================================
# PART E: The ramp vs delta distinction
# ================================================================
pr("\n" + "=" * 70)
pr("PART E: Why the perturbation is a RAMP, not a DELTA")
pr("=" * 70)
pr("""
The sector bit a = p_{K-2} contributes to conv_j for j ∈ [K-2, D-2]:
  Δconv_j = a · q_{j-K+2}    (a single random bit)
  ⟨Δconv_j⟩ = a/2 = 1/2 for sector (1,0)

This is a CONSTANT perturbation of 1/2 at every position from K-2 to D-2.
In Fourier space, a constant perturbation over half the chain [L/2, L]
has mode amplitudes:

  Δv_n = (2/L) ∫_{L/2}^{L} (1/2) sin(nπx/L) dx
       = (1/L) [-cos(nπx/L)/(nπ/L)]_{L/2}^{L}
       = (1/(nπ)) [cos(nπ/2) - cos(nπ)]

For n = 1: [cos(π/2) - cos(π)]/(π) = [0-(-1)]/(π) = 1/π
For n = 2: [cos(π) - cos(2π)]/(2π) = [-1-1]/(2π) = -1/π
For n = 3: [cos(3π/2) - cos(3π)]/(3π) = [0+1]/(3π) = 1/(3π)
For n = 4: [cos(2π) - cos(4π)]/(4π) = [1-1]/(4π) = 0
For n = 5: [cos(5π/2) - cos(5π)]/(5π) = [0+1]/(5π) = 1/(5π)

Pattern: Δv_n = 1/(nπ) for n odd, -1/π for n=2, 0 for n=4,8,...

But in the BRIDGE, this perturbation is multiplied by the Green's function:
  ⟨Δval⟩ = Σ_n Δv_n · G_n
where G_n depends on the bridge response at the observation point.

The carry chain response to a perturbation at mode n is:
  c_j^{(n)} = Δv_n · sin(nπj/L) / (1 - λ_n)

And val = c_{M-1} - 1 averages this over the M distribution.
""")

# Verify the ramp Fourier coefficients against data
for K in [10, 11]:
    results, D = enumerate_full(K)
    L = D - 1

    # Compute the mean convolution perturbation
    delta_conv = np.zeros(D)
    n_00 = n_10 = 0
    for r in results:
        if (r['a'], r['c']) == (0, 0):
            for j in range(D):
                delta_conv[j] -= r['convs'][j]
            n_00 += 1
        elif (r['a'], r['c']) == (1, 0):
            for j in range(D):
                delta_conv[j] += r['convs'][j]
            n_10 += 1

    delta_conv_mean = np.zeros(D)
    for j in range(D):
        delta_conv_mean[j] = delta_conv[j] / n_10 + delta_conv[j] / n_00
    # Actually: Δconv = ⟨conv⟩₁₀ - ⟨conv⟩₀₀
    conv_00 = np.zeros(D)
    conv_10 = np.zeros(D)
    for r in results:
        if (r['a'], r['c']) == (0, 0):
            for j in range(D):
                conv_00[j] += r['convs'][j]
        elif (r['a'], r['c']) == (1, 0):
            for j in range(D):
                conv_10[j] += r['convs'][j]
    conv_00 /= max(n_00, 1)
    conv_10 /= max(n_10, 1)
    delta_conv_mean = conv_10 - conv_00

    pr(f"\n--- K={K}, D={D} ---")
    pr(f"  Convolution perturbation Δ⟨conv_j⟩ = ⟨conv_j⟩₁₀ - ⟨conv_j⟩₀₀:")
    for j in range(max(0, K-4), D):
        pr(f"    j={j:3d}: Δconv = {delta_conv_mean[j]:+.5f}")

    # Fourier analysis
    pr(f"\n  Dirichlet modes of Δconv:")
    pr(f"  {'n':>4} {'measured':>12} {'ramp theory':>12} {'ratio':>12}")
    for n in range(1, 10):
        basis = np.array([math.sin(n * math.pi * j / L) for j in range(D)])
        coeff = 2.0 / L * np.dot(delta_conv_mean, basis)
        theory = (math.cos(n*math.pi/2) - math.cos(n*math.pi)) / (n * math.pi)
        ratio = coeff / theory if abs(theory) > 1e-10 else float('nan')
        pr(f"  {n:4d} {coeff:+12.6f} {theory:+12.6f} {ratio:+12.4f}")


# ================================================================
# PART F: Response sum — does Σ Δv_n · G_n give -π?
# ================================================================
pr("\n" + "=" * 70)
pr("PART F: Spectral response sum")
pr("=" * 70)
pr("""
If the carry bridge has eigenvalues λ_n = 1 - (nπ/L)² · τ for some τ,
and the perturbation has mode amplitudes Δv_n = [cos(nπ/2)-cos(nπ)]/(nπ),
then the response sum is:

  Σ_n Δv_n / (1-λ_n) = Σ_n [cos(nπ/2)-cos(nπ)] / (nπ · (nπ/L)² · τ)
  ~ (L²/τ) · Σ_n [cos(nπ/2)-cos(nπ)] / (n³π³)

The partial sums of [cos(nπ/2)-cos(nπ)]/(n³) are:
  n=1: [0-(-1)]/1 = 1
  n=2: [-1-1]/8 = -1/4
  n=3: [0-(-1)]/27 = 1/27
  n=5: [0-(-1)]/125 = 1/125
  ...

For n odd: [cos(nπ/2)-cos(nπ)] = -cos(nπ) = -(-1)^n = (-1)^{n+1}
  (since cos(nπ/2) = 0 for n odd)
For n even: cos(nπ/2) = (-1)^{n/2}, cos(nπ) = 1
  So [cos(nπ/2) - 1] = (-1)^{n/2} - 1

This does NOT simply reduce to a Leibniz series.
""")

# Compute the actual sum for several forms
pr("  Partial sums of [cos(nπ/2)-cos(nπ)]/(n^p) for various p:")
for p in [1, 2, 3]:
    S = 0
    pr(f"\n  p = {p}:")
    for n in range(1, 51):
        term = (math.cos(n*math.pi/2) - math.cos(n*math.pi)) / n**p
        S += term
        if n <= 10 or n % 10 == 0:
            pr(f"    n={n:3d}: term={term:+.8f}, S={S:+.8f}")

    pr(f"    Limit: S = {S:+.10f}")
    pr(f"    π/4 = {math.pi/4:.10f}, π = {math.pi:.10f}")

# Now try: what if the Green's function suppression is 1/(n²) ?
# Then Σ Δv_n / n² = Σ [cos(nπ/2)-cos(nπ)] / (nπ · n²)
pr("\n  Sum Σ [cos(nπ/2)-cos(nπ)] / (n^p · π) for p=2,3:")
for p in [2, 3]:
    S = 0
    for n in range(1, 10001):
        term = (math.cos(n*math.pi/2) - math.cos(n*math.pi)) / (n**p * math.pi)
        S += term
    pr(f"    p={p}: Σ = {S:+.10f}")
    pr(f"    Compare: π/4={math.pi/4:.10f}, 1={1:.10f}, "
       f"π²/12={math.pi**2/12:.10f}")


# ================================================================
# PART G: The key identity
# ================================================================
pr("\n" + "=" * 70)
pr("PART G: Key identity search")
pr("=" * 70)
pr("""
We need Σ [cos(nπ/2)-cos(nπ)] · f(n) = -π for some weight f(n).

Splitting into odd and even n:
  Odd n: cos(nπ/2) = 0, cos(nπ) = -1 → term = +1
  Even n: cos(nπ) = 1, cos(nπ/2) = (-1)^{n/2} → term = (-1)^{n/2} - 1

Odd n contribute: Σ_{n odd} f(n)
Even n contribute: Σ_{n even} [(-1)^{n/2} - 1] · f(n)

For n=2: [-1-1]·f(2) = -2f(2)
For n=4: [1-1]·f(4) = 0
For n=6: [-1-1]·f(6) = -2f(6)
For n=8: [1-1]·f(8) = 0

So: S = Σ_{n odd} f(n) - 2·Σ_{m odd} f(2m)

If f(n) = 1/n (weight for simple response):
  S = Σ_{n odd} 1/n - 2·Σ_{m odd} 1/(2m)
    = Σ_{n odd} 1/n - Σ_{m odd} 1/m = 0  ← trivially zero!

If f(n) = 1/n² (bridge response with 1/(1-λ) ~ 1/(nπ/L)² ~ L²/n²):
  S = Σ_{n odd} 1/n² - 2·Σ_{m odd} 1/(2m)²
    = Σ_{n odd} 1/n² - (1/2)·Σ_{m odd} 1/m²
    = (1/2)·Σ_{n odd} 1/n² = (1/2)·(π²/8) = π²/16

That's π²/16, not -π. But π²/16 = π·(π/16), and the prefactor from
Δv_n = 1/(nπ) makes this: S/(nπ) total = (π²/16)/π = π/16.
Still not -π.

Let me check what weight f(n) makes the sum equal -π:
""")

# What weight f(n) works?
# Try f(n) = 1/(n · something) to get Leibniz
pr("  Testing: Σ_{n odd} sin(nπ/2)/n — this IS the Leibniz series!")
S = 0
for n in range(1, 10001, 2):
    S += math.sin(n * math.pi / 2) / n
pr(f"    = {S:.10f}")
pr(f"    π/4 = {math.pi/4:.10f}")

pr("\n  Testing: Σ [cos(nπ/2)-cos(nπ)] · sin(nπ/2) / n:")
S = 0
for n in range(1, 10001):
    term = (math.cos(n*math.pi/2) - math.cos(n*math.pi)) * math.sin(n*math.pi/2) / n
    S += term
pr(f"    = {S:.10f}")
pr(f"    π/4 = {math.pi/4:.10f}")

pr("\n  Key observation: sin(nπ/2) · [cos(nπ/2)-cos(nπ)] = ")
for n in range(1, 9):
    val = math.sin(n*math.pi/2) * (math.cos(n*math.pi/2) - math.cos(n*math.pi))
    pr(f"    n={n}: {val:+.6f}")

pr("\nFor n odd: sin(nπ/2)·1 = sin(nπ/2) = ±1")
pr("For n even: sin(nπ/2) = 0, so even terms vanish!")
pr("Therefore: Σ sin(nπ/2)·[cos(nπ/2)-cos(nπ)]/n = Σ_{n odd} sin(nπ/2)/n = π/4")

pr("\n  ★★★ SYNTHESIS ★★★")
pr(f"""
The carry chain response to the sector perturbation:

1. The sector bit creates a RAMP perturbation Δv_n = [cos(nπ/2)-cos(nπ)]/(nπ)
2. The bridge Green's function at source j₀ = L/2 gives sin(nπ/2)
3. The response at the observation point involves 1/(1-λ_n)
4. The product sin(nπ/2) · [cos(nπ/2)-cos(nπ)] = sin(nπ/2) for odd n, 0 for even n

So the sum becomes:
  Σ_n sin(nπ/2) · [cos(nπ/2)-cos(nπ)] / (nπ · (1-λ_n))
  = Σ_{{n odd}} sin(nπ/2) / (nπ · (1-λ_n))

In the Markov limit with λ_n = 1/2 for all n (Diaconis-Fulman gap):
  = (1/(π·1/2)) Σ_{{n odd}} sin(nπ/2)/n
  = (2/π) · (π/4) = 1/2

But we need -π, not 1/2. The discrepancy factor is -2π.
Something is missing — the normalization, the exact form of 1-λ_n,
or the observation functional.
""")
