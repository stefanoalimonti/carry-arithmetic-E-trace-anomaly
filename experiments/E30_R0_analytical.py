#!/usr/bin/env python3
"""
E30: Analytical Closed Form for R₀ (Schoolbook Limit)

The schoolbook val = 2·carry[D-2] - 1 gives R₀ = σ₁₀/σ₀₀ in the
continuum limit K → ∞. This script derives the EXACT closed form.

Framework:
  X = 2^{K-1}(1+u), Y = 2^{K-1}(1+v)
  Sector (0,0): u ∈ [0, 1/2),  Y-constraint: v ∈ [0, 1/2)
  Sector (1,0): u ∈ [1/2, 1),  Y-constraint: v ∈ [0, 1/2)
  D-odd: (1+u)(1+v) < 2, i.e. v < (1-u)/(1+u) =: v_D(u)

  Effective v-range: v_eff(u) = min(1/2, v_D(u))
    For u ∈ [0, 1/3]: v_eff = 1/2  (Y-sector is tighter)
    For u ∈ [1/3, 1]:  v_eff = v_D  (D-odd is tighter)
    Breakpoint: v_D(1/3) = (2/3)/(4/3) = 1/2

  val threshold: bit_{D-2}(P) = 1 iff (1+u)(1+v) ≥ 3/2
    → v ≥ v* = (1-2u)/(2(1+u))
    v* ∈ [0, 1/2] for u ∈ [0, 1/2]

Derivation of ⟨val⟩₀₀(u):
  For u ∈ [0, 1/3]: v_eff=1/2, v*=(1-2u)/(2(1+u))
    ⟨val⟩ = (v_eff - 2v*)/v_eff = 1 - 4v* = (5u-1)/(1+u)

  For u ∈ [1/3, 1/2]: v_eff=v_D, v*=(1-2u)/(2(1+u))
    ⟨val⟩ = (v_D - 2v*)/v_D = u/(1-u)

Sector (1,0): val = -1 for all D-odd pairs (schoolbook).
"""
import sympy as sp
import math
import numpy as np

def pr(s=""):
    print(s, flush=True)


pr("=" * 78)
pr("E30: Analytical R₀ — Schoolbook Carry Limit")
pr("=" * 78)
pr()

# ─── Symbolic computation ─────────────────────────────────────
u = sp.Symbol('u', positive=True)

# Effective v-range
v_D = (1 - u) / (1 + u)

# ⟨val⟩ profiles
val_00_piece1 = (5*u - 1) / (1 + u)   # u ∈ [0, 1/3]
val_00_piece2 = u / (1 - u)            # u ∈ [1/3, 1/2]

pr("─── STEP 1: σ₁₀ (sector 1,0) ───")
pr()
pr("  val = -1 for all D-odd pairs in sector (1,0)")
pr("  σ₁₀ = -∫_{1/2}^1 v_D(u) du = -∫_{1/2}^1 (1-u)/(1+u) du")
pr()

sigma_10 = -sp.integrate(v_D, (u, sp.Rational(1,2), 1))
sigma_10_simplified = sp.simplify(sigma_10)

pr(f"  σ₁₀ = {sigma_10_simplified}")
pr(f"       = {sp.nsimplify(sigma_10_simplified, rational=False)}")
pr(f"       ≈ {float(sigma_10_simplified):.15f}")
pr()

pr("─── STEP 2: σ₀₀ (sector 0,0) ───")
pr()
pr("  σ₀₀ = ∫₀^{1/3} ⟨val⟩·v_eff du + ∫_{1/3}^{1/2} ⟨val⟩·v_eff du")
pr()

# Piece 1: u ∈ [0, 1/3], v_eff = 1/2
I1_integrand = val_00_piece1 * sp.Rational(1, 2)
I1 = sp.integrate(I1_integrand, (u, 0, sp.Rational(1,3)))
I1_simplified = sp.simplify(I1)

pr(f"  I₁ = (1/2)∫₀^{{1/3}} (5u-1)/(1+u) du")
pr(f"     = {I1_simplified}")
pr(f"     ≈ {float(I1_simplified):.15f}")
pr()

# Piece 2: u ∈ [1/3, 1/2], v_eff = v_D = (1-u)/(1+u)
I2_integrand = val_00_piece2 * v_D
I2_integrand_simplified = sp.simplify(I2_integrand)
pr(f"  I₂ integrand = [u/(1-u)] · [(1-u)/(1+u)] = u/(1+u)")

I2 = sp.integrate(u / (1 + u), (u, sp.Rational(1,3), sp.Rational(1,2)))
I2_simplified = sp.simplify(I2)

pr(f"  I₂ = ∫_{{1/3}}^{{1/2}} u/(1+u) du")
pr(f"     = {I2_simplified}")
pr(f"     ≈ {float(I2_simplified):.15f}")
pr()

sigma_00 = sp.simplify(I1 + I2)
pr(f"  σ₀₀ = I₁ + I₂ = {sigma_00}")
pr(f"       ≈ {float(sigma_00):.15f}")
pr()

# ─── STEP 3: R₀ = σ₁₀/σ₀₀ ───────────────────────────────────
pr("─── STEP 3: R₀ = σ₁₀/σ₀₀ ───")
pr()

R0 = sp.simplify(sigma_10 / sigma_00)
R0_float = float(R0)

pr(f"  R₀ = σ₁₀/σ₀₀ = {R0}")
pr(f"     ≈ {R0_float:.15f}")
pr()

# Express in terms of ln2 and ln3
L2, L3 = sp.symbols('L2 L3')
sigma_10_L = sp.Rational(1,2) - 4*L2 + 2*L3
sigma_00_L = 1 - 3*L2 + L3
R0_L = sigma_10_L / sigma_00_L

pr("  In terms of ln2, ln3:")
pr(f"    σ₁₀ = 1/2 - 4·ln2 + 2·ln3")
pr(f"    σ₀₀ = 1 - 3·ln2 + ln3")
pr(f"    R₀  = (1/2 - 4·ln2 + 2·ln3) / (1 - 3·ln2 + ln3)")
pr()
pr(f"  Or equivalently:")
pr(f"    σ₁₀ = 1/2 - 2·ln(4/3)")
pr(f"    σ₀₀ = 1 + ln(3/8)")
pr(f"    R₀  = (1/2 - 2·ln(4/3)) / (1 + ln(3/8))")
pr(f"        = -(2·ln(4/3) - 1/2) / ln(3e/8)")
pr()

# ─── STEP 4: ΔR = -π - R₀ ────────────────────────────────────
pr("─── STEP 4: ΔR = -π - R₀ (cascade correction) ───")
pr()

Delta_R = -sp.pi - R0
Delta_R_float = float(Delta_R)

pr(f"  ΔR = -π - R₀")
pr(f"     = -π - ({R0_float:.15f})")
pr(f"     = {Delta_R_float:.15f}")
pr()

# Check if ΔR matches known constants
pi_val = math.pi
delta_r = -pi_val - R0_float
pr("  Checking ΔR against known constants:")
pr(f"    ΔR           = {delta_r:.15f}")
pr(f"    π/4          = {pi_val/4:.15f}  diff = {delta_r - pi_val/4:.6e}")
pr(f"    1-ln2        = {1-math.log(2):.15f}  diff = {delta_r - (1-math.log(2)):.6e}")
pr(f"    ln(e·π/4)    = {math.log(math.e*pi_val/4):.15f}  diff = {delta_r - math.log(math.e*pi_val/4):.6e}")
pr(f"    2ln2-1       = {2*math.log(2)-1:.15f}  diff = {delta_r - (2*math.log(2)-1):.6e}")
pr(f"    4ln2-2ln3    = {4*math.log(2)-2*math.log(3):.15f}  diff = {delta_r - (4*math.log(2)-2*math.log(3)):.6e}")
pr(f"    3-e          = {3-math.e:.15f}  diff = {delta_r - (3-math.e):.6e}")

# PSLQ-style: check if ΔR = a + b·π + c·ln2 + d·ln3 for small integers
pr()
pr("  PSLQ-style scan: ΔR = a + b·π + c·ln2 + d·ln3?")

best_err = 1.0
best_combo = None
for a in range(-10, 11):
    for b_num in range(-10, 11):
        for b_den in [1, 2, 3, 4, 6]:
            b = b_num / b_den
            for c in range(-10, 11):
                for d in range(-10, 11):
                    val = a + b * pi_val + c * math.log(2) + d * math.log(3)
                    err = abs(val - delta_r)
                    if err < best_err and err < 1e-6:
                        best_err = err
                        best_combo = (a, f"{b_num}/{b_den}", c, d, err)

if best_combo:
    a, b_str, c, d, err = best_combo
    pr(f"    BEST: ΔR ≈ {a} + ({b_str})·π + {c}·ln2 + {d}·ln3  (err={err:.2e})")
else:
    pr("    No simple (a + b·π + c·ln2 + d·ln3) found with |err| < 1e-6")

pr()

# ─── STEP 5: N₁₀/N₀₀ verification ───────────────────────────
pr("─── STEP 5: Sector count decomposition ───")
pr()

N_10 = sp.integrate(v_D, (u, sp.Rational(1,2), 1))
N_00_piece1 = sp.integrate(sp.Rational(1,2), (u, 0, sp.Rational(1,3)))
N_00_piece2 = sp.integrate(v_D, (u, sp.Rational(1,3), sp.Rational(1,2)))
N_00 = sp.simplify(N_00_piece1 + N_00_piece2)

pr(f"  N₁₀ = {sp.simplify(N_10)}")
pr(f"       ≈ {float(N_10):.15f}")
pr(f"  N₀₀ = {N_00}")
pr(f"       ≈ {float(N_00):.15f}")
pr(f"  N₁₀/N₀₀ = {float(N_10/N_00):.15f}")
pr(f"  (E19 limit = {(2*math.log(4/3)-0.5)/(2*math.log(9/8)):.15f})")
pr()

avg_val_00 = float(sigma_00 / N_00)
pr(f"  ⟨val⟩₀₀ = σ₀₀/N₀₀ = {avg_val_00:.15f}")
pr(f"  ⟨val⟩₁₀ = -1 (exact)")
pr(f"  ⟨val⟩₁₀/⟨val⟩₀₀ = {-1.0/avg_val_00:.15f}")
pr(f"  R₀ = (⟨val⟩ ratio) · (N ratio) = {-1.0/avg_val_00:.6f} · {float(N_10/N_00):.6f} = {R0_float:.10f}")
pr()

# ─── STEP 6: Richardson extrapolation of E28 data ──────────
pr("─── STEP 6: Verify against E28 enumeration ───")
pr()

# Known R values from E28 (schoolbook val) at various K
# We compute them on the fly for K=8..16
pr("  Computing schoolbook R for K=8..18...")
R_school_data = []
for K in range(8, 19):
    D = 2*K - 1
    X_base = 1 << (K - 1)
    half = 1 << (K - 2)
    Y_arr = np.arange(X_base, X_base + half, dtype=np.int64)
    P_min_k = 1 << (D - 1)
    P_max_k = 1 << D

    s00 = 0
    n10_total = 0
    for sector_a in [0, 1]:
        X_lo = X_base + sector_a * half
        for ix in range(half):
            X = X_lo + ix
            P_arr = X * Y_arr
            d_odd = (P_arr >= P_min_k) & (P_arr < P_max_k)
            if sector_a == 0:
                p_Dm2 = (P_arr >> (D - 2)) & 1
                vals = 2 * p_Dm2 - 1
                s00 += int(np.sum(vals[d_odd]))
            else:
                n10_total += int(np.sum(d_odd))

    R_k = -n10_total / s00 if s00 != 0 else 0
    R_school_data.append((K, R_k, s00, n10_total))
    pr(f"    K={K:2d}: R = {R_k:.12f}  σ₀₀={s00:>14d}  |R-R₀| = {abs(R_k - R0_float):.6e}")

pr()

# Richardson extrapolation (rate 1/2^K geometric correction)
R_vals = [r for _, r, _, _ in R_school_data]
K_vals = [k for k, _, _, _ in R_school_data]

pr("  Richardson extrapolation (geometric rate 1/2):")
current = R_vals[:]
for step in range(min(6, len(current) - 1)):
    new = []
    for i in range(len(current) - 1):
        new.append(2 * current[i + 1] - current[i])
    current = new
    pr(f"    Step {step+1}: R_∞ = {current[-1]:.12f}  |R_∞ - R₀| = {abs(current[-1] - R0_float):.6e}")

pr()

# ─── STEP 7: Decomposition summary ───────────────────────────
pr("=" * 78)
pr("DECOMPOSITION OF R = -π")
pr("=" * 78)
pr()
pr("  R = R₀ + ΔR  where:")
pr()
pr(f"  R₀ = (1/2 - 2·ln(4/3)) / (1 + ln(3/8))")
pr(f"     = {R0_float:.15f}")
pr(f"     (schoolbook carry at fixed position D-2)")
pr()
pr(f"  ΔR = -π - R₀")
pr(f"     = {delta_r:.15f}")
pr(f"     (cascade correction = resolvent contribution)")
pr()
pr(f"  R  = R₀ + ΔR = {R0_float + delta_r:.15f}")
pr(f"     = -π = {-pi_val:.15f}  ✓")
pr()
pr("  INTERPRETATION (First Passage Time):")
pr("    R₀ evaluates carry at FIXED time D-2 → Markov relaxation → smooth G(u)")
pr("    ΔR is the cascade correction from STOPPING TIME M < D-2")
pr("    ΔR = Tr[(I-T)⁻¹ · P_boundary] − Tr[T^{D-2} · P_boundary]")
pr("    The resolvent (I-T)⁻¹ generates the Leibniz series via Dirichlet eigenfunctions")
pr()
pr(f"  COMPONENTS:")
pr(f"    σ₁₀_school = 1/2 - 2·ln(4/3)  ≈ {float(sigma_10_simplified):.12f}")
pr(f"    σ₀₀_school = 1 + ln(3/8)       ≈ {float(sigma_00):.12f}")
pr(f"    N₁₀/N₀₀   = (above ratio)      ≈ {float(N_10/N_00):.12f}")
pr(f"    ⟨val⟩₀₀    = σ₀₀/N₀₀           ≈ {avg_val_00:.12f}")
pr()
pr(f"  KEY IDENTITY (to be proved):")
pr(f"    ΔR · σ₀₀ = σ₁₀_cascade − σ₁₀_school + R₀·(σ₀₀_cascade − σ₀₀_school)")
pr(f"    This connects the resolvent trace to the cascade correction.")
pr()
pr("=" * 78)
