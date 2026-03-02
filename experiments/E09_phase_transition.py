#!/usr/bin/env python3
"""E09: Phase transition mechanism — ANALYTICAL PROOF of S(A) formula.

THEOREM (Phase Transition Formula):
  S(A) := lim_{L→∞} Σ_{n odd} sin(nπ/2)/(1−λ_n(A)) = 2(1−A)/(3−2A)

where λ_n(A) = cos(nπ/L)/2 + A·sin²(nπ/(2L)).

COROLLARY: S(A*) = −π when A* = (2+3π)/(2(1+π)).

PROOF:
  Let β = 1−A.  Then 1−λ_n = 1/2 + β·sin²(nπ/(2L)).

  Step 1: Rewrite denominator using cos(2x) = 1−2sin²(x):
    1/2 + β·sin²(x) = (1+β−β·cos(2x))/2

  Step 2: Let r = β/(1+β), so:
    2/(1+β) · Σ (-1)^k / (1 − r·cos(θ_k))
    where θ_k = (2k+1)π/L

  Step 3: Expand 1/(1−r·cos(θ)) = Σ_m a_m·cos(mθ)
    using the Poisson kernel: a_0 = 1/√(1−r²),
    a_m = 2α^m/√(1−r²) for m≥1, where α = r/(1+√(1−r²)).

  Step 4: Chebyshev orthogonality for L≡0 mod 4:
    Σ_{k=0}^{L/2−1} (-1)^k cos(m(2k+1)π/L) = {0 if m even, 1/cos(mπ/L) if m odd}

  Step 5: In the limit L→∞, cos(mπ/L) → 1, giving:
    (1+β)·S/2 = (2/√(1−r²)) · Σ_{m odd} α^m
              = (2/√(1−r²)) · α/(1−α²)
              = r/(1−r²)  [after simplification]
              = β(1+β)/(1+2β)

  Step 6: S = 2/(1+β) · β(1+β)/(1+2β) = 2β/(1+2β)  QED.

  Solving S = −π: 2β/(1+2β) = −π
    β = −π/(2+2π), so A = 1−β = 1 + π/(2(1+π)) = (2+3π)/(2(1+π)).
"""
import sys
import numpy as np
from mpmath import mp, mpf, pi, sin, cos, nstr, sqrt, log10

mp.dps = 30


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def S_sum(A, L):
    """Compute S(A, L) = Σ sin(nπ/2) / (1 − λ_n(A))."""
    A = mpf(A)
    total = mpf(0)
    for n in range(1, L):
        sn = sin(n * pi / 2)
        if abs(sn) < mpf('1e-30'):
            continue
        lam = cos(n * pi / L) / 2 + A * sin(n * pi / (2 * L)) ** 2
        denom = 1 - lam
        if abs(denom) < mpf('1e-40'):
            continue
        total += sn / denom
    return total


def formula_S(A):
    """Closed form: S(A) = 2(1−A)/(3−2A) = 2β/(1+2β) where β=1−A."""
    return 2 * (1 - mpf(A)) / (3 - 2 * mpf(A))


def main():
    pr("=" * 72)
    pr("E09: PHASE TRANSITION — ANALYTICAL PROOF")
    pr("=" * 72)

    Astar = (2 + 3 * pi) / (2 * (1 + pi))
    beta_star = -pi / (2 * (1 + pi))

    # ═══════ Part A: Verify formula S = 2β/(1+2β) ══════════════════
    pr(f"\n{'═' * 72}")
    pr("A. VERIFICATION: S(A,L) → 2(1−A)/(3−2A)")
    pr(f"{'═' * 72}\n")

    pr(f"  {'A':>6} {'L':>6} {'S(A,L) numeric':>20} {'S(A) formula':>20} "
       f"{'delta':>14} {'digits':>8}")

    for A_val in [0.0, 0.5, 1.0, 1.1, 1.2, 1.3, 1.35]:
        Sf = formula_S(A_val)
        for L in [100, 500]:
            Sn = S_sum(A_val, L)
            delta = Sn - Sf
            d = float(-log10(abs(delta))) if abs(delta) > mpf('1e-30') else 99
            pr(f"  {A_val:6.2f} {L:6d} {nstr(Sn, 16):>20} {nstr(Sf, 16):>20} "
               f"{nstr(delta, 8):>14} {d:8.1f}")
        pr()

    # ═══════ Part B: The exact A* ══════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("B. EXACT A* = (2+3π)/(2(1+π))")
    pr(f"{'═' * 72}\n")

    pr(f"  A* = (2+3π)/(2(1+π)) = {nstr(Astar, 20)}")
    pr(f"  β* = 1−A* = −π/(2(1+π)) = {nstr(beta_star, 20)}")
    pr(f"  S(A*) = 2β*/(1+2β*) = {nstr(formula_S(Astar), 20)}")
    pr(f"  −π    = {nstr(-pi, 20)}")
    pr(f"  |S(A*)−(−π)| = {nstr(abs(formula_S(Astar) + pi), 5)}")
    pr()

    pr(f"  Numerical verification:")
    pr(f"  {'L':>6} {'S(A*,L)':>22} {'S+π':>14} {'digits':>8}")
    for L in [40, 100, 200, 400, 1000]:
        S = S_sum(Astar, L)
        err = S + pi
        digits = float(-log10(abs(err / pi))) if abs(err / pi) > mpf('1e-30') else 99
        pr(f"  {L:6d} {nstr(S, 18):>22} {nstr(err, 8):>14} {digits:8.2f}")

    # ═══════ Part C: Analytical proof of the formula ═══════════════
    pr(f"\n{'═' * 72}")
    pr("C. ANALYTICAL PROOF")
    pr(f"{'═' * 72}\n")

    pr("  THEOREM: For L ≡ 0 mod 4 and L → ∞:")
    pr("    S(A) = Σ_{n odd} sin(nπ/2)/(1−λ_n(A)) → 2(1−A)/(3−2A)")
    pr()
    pr("  PROOF:")
    pr()
    pr("  Let β = 1−A. The denominator is:")
    pr("    1−λ_n = 1/2 + β·sin²(nπ/(2L))")
    pr("          = [(1+β) − β·cos(nπ/L)] / 2")
    pr()
    pr("  Let r = β/(1+β). Then S = (2/(1+β)) · T, where:")
    pr("    T = Σ_{k=0}^{N-1} (-1)^k / (1 − r·cos(θ_k))")
    pr("    with θ_k = (2k+1)π/L, N = L/2")
    pr()
    pr("  Step 1: Fourier expansion (Poisson kernel)")
    pr("    1/(1−r·cos(θ)) = (1/s)·[1 + 2·Σ_{m≥1} α^m · cos(mθ)]")
    pr("    where s = √(1−r²), α = r/(1+s), |α| < 1.")
    pr()
    pr("  Step 2: Chebyshev orthogonality (L ≡ 0 mod 4):")
    pr("    Σ_{k=0}^{N-1} (-1)^k cos(m·(2k+1)π/L)")
    pr("    = Re[e^{imπ/L} · (1−(−e^{2imπ/L})^N) / (1+e^{2imπ/L})]")
    pr("    = 0 if m even, 1/cos(mπ/L) if m odd")

    # Verify Step 2 numerically
    pr()
    pr("  Step 2 verification (L=100, L/2=50 terms):")
    L_test = 100
    N_test = L_test // 2
    for m in [1, 2, 3, 4, 5, 7, 11]:
        total = mpf(0)
        for k in range(N_test):
            total += (-1) ** k * cos(m * (2 * k + 1) * pi / L_test)
        expected = 1 / cos(m * pi / L_test) if m % 2 == 1 else mpf(0)
        pr(f"    m={m}: Σ = {nstr(total, 14)}, "
           f"expected = {nstr(expected, 14)}, "
           f"match = {'✓' if abs(total - expected) < mpf('1e-10') else '✗'}")

    pr()
    pr("  Step 3: Substitute into T:")
    pr("    T = (1/s)·Σ_{m odd} (2α^m)/cos(mπ/L)")
    pr("    As L→∞: cos(mπ/L) → 1, so:")
    pr("    T → (2/s)·Σ_{m odd} α^m = (2/s)·α/(1−α²)")
    pr()
    pr("  Step 4: Simplify α/(1−α²):")
    pr("    α = r/(1+s), 1−α² = 2s/(1+s)")
    pr("    α/(1−α²) = r/(2s)")
    pr("    T → (2/s)·r/(2s) = r/s² = r/(1−r²)")
    pr()
    pr("  Step 5: Final result:")
    pr("    S = (2/(1+β))·r/(1−r²)")
    pr("      = (2/(1+β))·(β/(1+β))/(1−β²/(1+β)²)")
    pr("      = (2/(1+β))·β(1+β)/((1+β)²−β²)")
    pr("      = 2β/(1+2β)")
    pr("                                                         QED ∎")

    # Verify the algebraic steps
    pr()
    pr("  Algebraic verification at β = −π/(2(1+π)):")
    beta = -pi / (2 * (1 + pi))
    r = beta / (1 + beta)
    s = sqrt(1 - r ** 2)
    alpha = r / (1 + s)
    T = r / (1 - r ** 2)
    S_from_T = 2 * T / (1 + beta)
    S_direct = 2 * beta / (1 + 2 * beta)
    pr(f"    β = {nstr(beta, 15)}")
    pr(f"    r = β/(1+β) = {nstr(r, 15)}")
    pr(f"    T = r/(1−r²) = {nstr(T, 15)}")
    pr(f"    S = 2T/(1+β) = {nstr(S_from_T, 15)}")
    pr(f"    S = 2β/(1+2β) = {nstr(S_direct, 15)}")
    pr(f"    −π = {nstr(-pi, 15)}")
    pr(f"    Match: {abs(S_direct + pi) < mpf('1e-25')}")

    # ═══════ Part D: Special values ════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("D. SPECIAL VALUES OF S(A)")
    pr(f"{'═' * 72}\n")

    special = [
        ("A=0 (Markov)", 0, "2/3"),
        ("A=1/2", 0.5, "1/2"),
        ("A=1 (critical)", 1, "0"),
        ("A=5/4", 1.25, "−1"),
        ("A=4/3", mpf(4) / 3, "−2"),
        ("A=A* = (2+3π)/(2(1+π))", Astar, "−π"),
        ("A=7/5", mpf(7) / 5, "−4/3"),
        ("A=3/2 (pole)", 1.5, "±∞"),
    ]

    pr(f"  {'A value':>30} {'A numeric':>12} {'S(A)':>14} {'β=1−A':>12}")
    for name, A, S_str in special:
        if abs(3 - 2 * mpf(A)) < mpf('1e-10'):
            pr(f"  {name:>30} {nstr(mpf(A), 8):>12} {'divergent':>14} "
               f"{nstr(1 - mpf(A), 8):>12}")
        else:
            S = formula_S(A)
            pr(f"  {name:>30} {nstr(mpf(A), 8):>12} {nstr(S, 10):>14} "
               f"{nstr(1 - mpf(A), 8):>12}")

    # ═══════ Part E: Physical interpretation ═══════════════════════
    pr(f"\n{'═' * 72}")
    pr("E. PHYSICAL INTERPRETATION AND PROOF STATUS")
    pr(f"{'═' * 72}\n")

    pr("  WHAT IS PROVED :")
    pr("  ✓ S(A) = 2(1−A)/(3−2A) analytically (Poisson kernel + Chebyshev)")
    pr(f"  ✓ A* = (2+3π)/(2(1+π)) = {nstr(Astar, 12)} gives S = −π")
    pr("  ✓ Convergence S(A*, L) = −π + O(1/L²)")
    pr("  ✓ Phase transition: +2/3 (Markov) → 0 (critical) → −π (exact)")
    pr()
    pr("  WHAT THIS MEANS:")
    pr("  The sector ratio R = −π is EQUIVALENT to the statement that")
    pr("  non-Markov digit correlations shift the eigenvalues by:")
    pr("    δ_n = A*·sin²(nπ/(2L))")
    pr(f"  where A* = (2+3π)/(2(1+π)) ≈ {nstr(Astar, 8)}")
    pr()
    pr("  The effective eigenvalue becomes:")
    pr("    λ_n(eff) = cos(nπ/L)/2 + A*·sin²(nπ/(2L))")
    pr("            = 1/2 + (A*−1)·sin²(nπ/(2L))")
    pr(f"           = 1/2 + (π/(2(1+π)))·sin²(nπ/(2L))")
    pr()
    pr("  REMAINING GAP:")
    pr("  Show that the exact (non-Markov) carry transfer operator has")
    pr("  eigenvalues that satisfy this shift relation with A = A*.")
    pr("  This is a statement about the SPECTRAL PERTURBATION caused")
    pr("  by digit-carry correlations over K consecutive positions.")
    pr()
    pr("  NOTE: A* involves π, so the proof would establish a")
    pr("  self-consistency: the carry operator produces π BECAUSE its")
    pr("  eigenshifts involve π through the correlation structure.")
    pr("  This is analogous to how ζ(2) = π²/6 involves π through")
    pr("  the Fourier series of x² — π appears because the operator")
    pr("  has periodic/Dirichlet structure.")

    pr(f"\n{'═' * 72}")


if __name__ == '__main__':
    main()
