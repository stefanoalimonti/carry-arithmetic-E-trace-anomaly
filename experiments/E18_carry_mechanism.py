#!/usr/bin/env python3
"""E18: Base-b universality of the bare coupling and phase transition.

In base 2: Markov eigenvalue = 1/2, Möbius S(A) = 2(1-A)/(3-2A), A_bare → 3/2 (pole).
For base b: Markov eigenvalue = 1/b, the spectral sum changes.

QUESTIONS:
1. What is the Möbius formula S_b(A) for general base b?
2. Does the Fejér-like coupling produce A_bare → pole for all bases?
3. Is the sector ratio rational for b ≥ 3? (Known: R(b=5) = 5/4)
4. Where is the phase transition for each base?

SETUP:
- Base b: carry c ∈ {0, ..., b-1}, spectral gap = 1/b
- Convolution: conv_j = Σ x_i y_{j-i} with x_i, y_i ∈ {0,...,b-1}
- Markov eigenvalue: λ = (1/b)cos(nπ/L)
- Fluctuation: ε_n = 1/b - λ_n/b = (1/b)(1 - cos(nπ/L)/2) ... needs derivation.

Actually the carry chain in base b has spectral gap 1 - 1/b (eigenvalue 1/b
for the sub-leading eigenvalue). The exact form depends on the transition
probabilities.

For base b, the convolution conv_j of K b-ary digits is the sum of K products
x_i·y_{j-i} where each x_i,y_i ∈ {0,...,b-1}. The carry update is:
    c_{j+1} = floor((conv_j + c_j) / b)

The digit correlation profile for base b:
    ρ_b(d) = (K-d) · Var[x·y] / (K · Var[conv_center])

For base b with uniform digits on {0,...,b-1}:
    E[x] = (b-1)/2, Var[x] = (b²-1)/12
    E[xy] = ((b-1)/2)², Var[xy] = E[x²y²] - E[xy]²
"""
import sys
import time
import numpy as np
from numpy.linalg import solve

def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def digit_stats(b):
    """Statistics for uniform digit on {0,...,b-1}."""
    digits = np.arange(b, dtype=float)
    mean_x = np.mean(digits)
    var_x = np.var(digits)
    products = np.outer(digits, digits).flatten()
    mean_xy = np.mean(products)
    var_xy = np.var(products)
    return mean_x, var_x, mean_xy, var_xy


def correlation_profile_base(K, b):
    """ρ_b(d) for base b."""
    _, _, _, var_xy = digit_stats(b)
    center_var = K * var_xy
    rho = np.array([(K - d) * var_xy / center_var for d in range(K)])
    return rho


def markov_eigenvalues_base(L, b, K):
    """Markov eigenvalues for base-b carry chain with Dirichlet BCs.

    For base b, the bulk transfer matrix eigenvalue for the carry chain
    is 1/b. With Dirichlet BCs on length L, the eigenvalues are
    approximately:
        λ_n = (1/b) cos(nπ/(L+1))
    """
    return np.array([(1.0 / b) * np.cos(n * np.pi / (L + 1))
                     for n in range(1, L)])


def build_coupling_matrix_base(K, L, b):
    """Fejér coupling V_{nn'} for base b."""
    rho = correlation_profile_base(K, b)
    N = L - 1
    V = np.zeros((N, N))
    js = np.arange(1, L, dtype=np.float64)

    sin_table = np.zeros((N, L - 1))
    for n in range(1, L):
        sin_table[n - 1] = np.sin(n * np.pi * js / L)

    for d in range(min(K, L)):
        rho_d = rho[d] if d < K else 0.0
        if abs(rho_d) < 1e-30:
            continue
        sin_shifted = np.zeros((N, L - 1))
        jp = js + d
        mask = (jp >= 1) & (jp <= L - 1)
        for n in range(1, L):
            sin_shifted[n - 1, mask] = np.sin(n * np.pi * jp[mask] / L)
        V += rho_d * (sin_table @ sin_shifted.T)

    V *= 2.0 / L
    return V


def spectral_sum_base(L, b, V_scaled):
    """S = Σ_n sin(nπ/2) / (1 - λ_n - V_nn) using full resolvent."""
    N = L - 1
    D_diag = np.array([(1.0 / b) * np.cos(n * np.pi / (L + 1))
                        for n in range(1, L)])
    T = np.diag(D_diag) + V_scaled
    A_mat = np.eye(N) - T
    e = np.ones(N)
    v = np.array([np.sin(n * np.pi / 2) for n in range(1, L)])
    try:
        G_e = solve(A_mat, e)
        return v @ G_e
    except np.linalg.LinAlgError:
        return float('inf')


def spectral_sum_markov_base(L, b):
    """Markov sum for base b."""
    S = 0.0
    for n in range(1, L):
        vn = np.sin(n * np.pi / 2)
        if abs(vn) < 1e-10:
            continue
        lam_n = (1.0 / b) * np.cos(n * np.pi / (L + 1))
        S += vn / (1.0 - lam_n)
    return S


def mobius_inversion(S):
    """A_eff from S: if S = 2(1-A)/(3-2A), then A = (3S-2)/(2S-2)."""
    if abs(2 * S - 2) < 1e-12:
        return float('inf')
    return (3 * S - 2) / (2 * S - 2)


def main():
    t0 = time.time()
    pr("=" * 72)
    pr("E18: BASE-b UNIVERSALITY OF BARE COUPLING AND PHASE TRANSITION")
    pr("=" * 72)

    # ═══════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("A. DIGIT STATISTICS BY BASE")
    pr(f"{'═' * 72}\n")

    pr(f"  {'b':>3} {'E[x]':>10} {'Var[x]':>10} {'E[xy]':>10} "
       f"{'Var[xy]':>10} {'ρ(0)=1/Var':>12}")
    for b in [2, 3, 4, 5, 7, 10]:
        mx, vx, mxy, vxy = digit_stats(b)
        pr(f"  {b:3d} {mx:10.4f} {vx:10.4f} {mxy:10.4f} "
           f"{vxy:10.4f} {1.0/vxy if vxy > 0 else 0:12.4f}")

    # ═══════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("B. MARKOV SPECTRAL SUM BY BASE")
    pr(f"{'═' * 72}\n")

    pr("  Markov sum S_b(0, L) for various bases and bridge lengths:")
    pr(f"  {'b':>3} {'L=19':>12} {'L=39':>12} {'L=99':>12} {'L→∞':>12}")
    for b in [2, 3, 5, 7, 10]:
        s19 = spectral_sum_markov_base(19, b)
        s39 = spectral_sum_markov_base(39, b)
        s99 = spectral_sum_markov_base(99, b)
        s_inf = 2 * (1.0 - 0) / (3 - 2 * 0)  # = 2/3 for the base-2 formula
        pr(f"  {b:3d} {s19:+12.6f} {s39:+12.6f} {s99:+12.6f} {'?':>12}")

    # ═══════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("C. FULL COUPLING FOR BASE b: DOES A_bare → POLE?")
    pr(f"{'═' * 72}\n")

    for b in [2, 3, 5]:
        pr(f"  ── Base {b} (Markov eigenvalue = 1/{b}) ──")
        pr(f"  {'K':>4} {'L':>5} {'S_Markov':>12} {'S_full':>14} {'A_eff':>12}")

        for K in [6, 8, 10, 15, 20, 30]:
            L = 2 * K - 1
            V = build_coupling_matrix_base(K, L, b)
            S_m = spectral_sum_markov_base(L, b)
            S_f = spectral_sum_base(L, b, V)
            A_eff = mobius_inversion(S_f) if abs(S_f) < 1e15 else float('inf')

            if abs(S_f) > 1e10:
                pr(f"  {K:4d} {L:5d} {S_m:+12.6f} {'DIVERGES':>14} "
                   f"{A_eff:+12.6f}")
            else:
                pr(f"  {K:4d} {L:5d} {S_m:+12.6f} {S_f:+14.6f} "
                   f"{A_eff:+12.6f}")
        pr()

    # ═══════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("D. BASE-b MÖBIUS FORMULA: WHAT IS THE ANALOG?")
    pr(f"{'═' * 72}\n")

    pr("  For base 2: S(A) = 2(1-A)/(3-2A), pole at A = 3/2")
    pr("  For base b: the Markov eigenvalue is 1/b, so the spectral sum")
    pr("  S_b(β, L) = Σ sin(nπ/2) / (1 - (1/b)cos(nπ/L) - β·sin²(nπ/(2L)))")
    pr()
    pr("  Let's compute S_b(β, ∞) numerically for various β:")
    pr()

    for b in [2, 3, 5]:
        pr(f"  ── Base {b}: S_b(β) ──")
        pr(f"    {'β':>8} {'S_b(β, L=999)':>16}")
        L = 999
        for beta in [0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
            S = 0.0
            for n in range(1, L):
                vn = np.sin(n * np.pi / 2)
                if abs(vn) < 1e-10:
                    continue
                lam_n = (1.0 / b) * np.cos(n * np.pi / L)
                s2 = np.sin(n * np.pi / (2 * L)) ** 2
                denom = 1.0 - lam_n - beta * s2
                if abs(denom) > 1e-12:
                    S += vn / denom
            pr(f"    {beta:8.2f} {S:+16.8f}")

        pr()
        pr(f"    Closed form test: for β = 0, S_b = ?")
        S0 = 0.0
        L = 999
        for n in range(1, L):
            vn = np.sin(n * np.pi / 2)
            if abs(vn) < 1e-10:
                continue
            lam_n = (1.0 / b) * np.cos(n * np.pi / L)
            S0 += vn / (1.0 - lam_n)
        pr(f"    S_b(0, L={L}) = {S0:+.10f}")
        pr(f"    Candidate: 2/(b+1) = {2.0/(b+1):.10f}")
        pr(f"    Candidate: b/(2b-1) = {b/(2.0*b - 1):.10f}")
        pr()

    # ═══════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("E. KNOWN SECTOR RATIOS BY BASE")
    pr(f"{'═' * 72}\n")

    pr("  Base 2: R = -π (transcendental)")
    pr("  Base 3: R = ? (needs computation)")
    pr("  Base 5: R = 5/4 = 1.25 (rational, base-dependent)")
    pr()
    pr("  For rational R, the phase transition doesn't cross the critical point.")
    pr("  For transcendental R = -π (base 2), it does.")
    pr("  Question: is base 2 unique in producing a transcendental R?")

    # ═══════════════════════════════════════════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("F. SUMMARY")
    pr(f"{'═' * 72}\n")

    pr("  The base-b analysis reveals:")
    pr("  1. The Markov sum S_b(0) depends on b (different baseline)")
    pr("  2. The Fejér coupling strength depends on Var[xy] (base-dependent)")
    pr("  3. Whether A_bare hits the pole is base-dependent")
    pr("  4. For base 2: A_bare → 3/2 (pole), R = -π (transcendental)")
    pr("     For base b ≥ 3: to be determined")

    pr(f"\n  Total runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
