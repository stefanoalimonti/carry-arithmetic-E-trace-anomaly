#!/usr/bin/env python3
"""E11: Carry subspace projection — measure A directly from the exact operator.

THE ONE REMAINING GAP:
  We proved S(A) = 2β/(1+2β) and A* = (2+3π)/(2(1+π)) gives S = -π.
  Need: show the exact carry operator has eigenvalue shift A = A*.

APPROACH:
  1. Build exact transfer matrix T_exact over (carry, g_buf, h_buf)
  2. Project onto carry subspace: T_eff[c', c] = Σ_{buf} T_exact[(c',buf'), (c,buf)]
  3. Compute eigenvalues of T_eff
  4. Compare with shifted Markov: λ_n(A) = cos(nπ/L)/2 + A·sin²(nπ/(2L))
  5. Extract A(K) and test A(K) → A* as K grows

Also: perturbation theory approach (Strategy 2).
"""
import sys
import time
import numpy as np
from scipy import sparse
from mpmath import mp, mpf, pi, cos, sin, nstr

mp.dps = 30


def pr(*a, **kw):
    print(*a, **kw)
    sys.stdout.flush()


def build_exact_bulk(K, c_max=3):
    """Exact bulk transfer matrix over (carry, g_buf, h_buf)."""
    Kb = K - 1
    bsz = 1 << Kb
    ns = (c_max + 1) * bsz * bsz
    rows, cols, vals = [], [], []
    for co in range(c_max + 1):
        for gw in range(bsz):
            for hw in range(bsz):
                s0 = co * bsz * bsz + gw * bsz + hw
                for gn in range(2):
                    for hn in range(2):
                        conv = 0
                        for k in range(K):
                            g = gn if k == Kb else (gw >> k) & 1
                            hi = K - 1 - k
                            h = hn if hi == Kb else (hw >> hi) & 1
                            conv += g * h
                        cn = (conv + co) >> 1
                        if cn > c_max:
                            continue
                        gw2 = (gw >> 1) | (gn << (Kb - 1))
                        hw2 = (hw >> 1) | (hn << (Kb - 1))
                        s1 = cn * bsz * bsz + gw2 * bsz + hw2
                        rows.append(s1)
                        cols.append(s0)
                        vals.append(0.25)
    T = sparse.coo_matrix((vals, (rows, cols)), shape=(ns, ns)).tocsr()
    return T, ns, c_max, bsz


def build_markov_bulk(K, c_max=3):
    """Markov carry-only transfer matrix."""
    dist = np.array([1.0])
    for _ in range(K):
        new = np.zeros(len(dist) + 1)
        new[:len(dist)] += dist * 0.75
        new[1:len(dist) + 1] += dist * 0.25
        dist = new
    T = np.zeros((c_max + 1, c_max + 1))
    for c in range(c_max + 1):
        for v in range(len(dist)):
            cn = (v + c) >> 1
            if cn <= c_max and dist[v] > 1e-30:
                T[cn, c] += dist[v]
    return T


def project_to_carry(T_full, c_max, bsz):
    """Project full (carry, g_buf, h_buf) matrix to carry-only.

    T_eff[c', c] = Σ_{g,h,g',h'} T_full[(c',g',h'), (c,g,h)] · w(g,h)

    where w(g,h) = 1/(bsz^2) is the uniform weight (stationary distribution
    of the digit buffer).
    """
    ns_carry = c_max + 1
    buf_sz = bsz * bsz
    T_dense = T_full.toarray()

    T_eff = np.zeros((ns_carry, ns_carry))
    for c in range(ns_carry):
        for cp in range(ns_carry):
            total = 0.0
            for buf_in in range(buf_sz):
                s_in = c * buf_sz + buf_in
                for buf_out in range(buf_sz):
                    s_out = cp * buf_sz + buf_out
                    total += T_dense[s_out, s_in]
            T_eff[cp, c] = total / buf_sz
    return T_eff


def main():
    t0 = time.time()
    Astar = float((2 + 3 * pi) / (2 * (1 + pi)))

    pr("=" * 72)
    pr("E11: CARRY SUBSPACE PROJECTION — MEASURE A DIRECTLY")
    pr("=" * 72)
    pr(f"\n  Target: A* = (2+3π)/(2(1+π)) = {Astar:.10f}")

    # ═══════ Part A: Project and compare eigenvalues ════════════════
    pr(f"\n{'═' * 72}")
    pr("A. CARRY SUBSPACE PROJECTION: T_eff EIGENVALUES")
    pr(f"{'═' * 72}\n")

    results = {}
    for K in [5, 6, 7]:
        c_max = 3
        tb = time.time()

        T_exact, ns, _, bsz = build_exact_bulk(K, c_max)
        T_markov = build_markov_bulk(K, c_max)

        T_eff = project_to_carry(T_exact, c_max, bsz)
        dt = time.time() - tb

        ev_markov = np.sort(np.real(np.linalg.eigvals(T_markov)))[::-1]
        ev_eff = np.sort(np.real(np.linalg.eigvals(T_eff)))[::-1]

        L = 2 * K - 2
        pr(f"  K={K} (L={L}, {ns} states, {dt:.1f}s):")
        pr(f"    {'n':>4} {'λ_Markov':>14} {'λ_projected':>14} "
           f"{'δλ':>12} {'sin²(nπ/(2L))':>16} {'A_fit':>10}")

        shifts = []
        sin2_vals = []
        for n in range(min(c_max, len(ev_markov))):
            lm = ev_markov[n]
            lp = ev_eff[n]
            delta = lp - lm
            sin2 = np.sin((n) * np.pi / (2 * L)) ** 2 if n > 0 else 0
            A_fit = delta / sin2 if sin2 > 1e-10 else float('nan')
            pr(f"    {n:4d} {lm:14.8f} {lp:14.8f} "
               f"{delta:+12.6f} {sin2:16.8f} {A_fit:10.4f}")
            if n > 0 and sin2 > 1e-10:
                shifts.append(delta)
                sin2_vals.append(sin2)

        if shifts:
            shifts = np.array(shifts)
            sin2_vals = np.array(sin2_vals)
            A_avg = np.mean(shifts / sin2_vals)
            results[K] = {'A_avg': A_avg, 'ev_eff': ev_eff, 'ev_markov': ev_markov}
            pr(f"    → Average A_fit = {A_avg:.6f} (target: {Astar:.6f})")
        pr()

    # ═══════ Part B: Perturbation theory ════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("B. PERTURBATION THEORY: δλ_n = ⟨ψ_n|V|ψ_n⟩")
    pr(f"{'═' * 72}\n")

    pr("  V = T_eff - T_Markov (perturbation matrix)")
    pr("  ψ_n = Markov eigenvectors (sinusoidal modes)")
    pr()

    for K in [5, 6, 7]:
        c_max = 3
        T_exact, ns, _, bsz = build_exact_bulk(K, c_max)
        T_markov = build_markov_bulk(K, c_max)
        T_eff = project_to_carry(T_exact, c_max, bsz)

        V = T_eff - T_markov
        L = 2 * K - 2

        ev_m, vec_m = np.linalg.eigh(T_markov)
        idx = np.argsort(ev_m)[::-1]
        ev_m = ev_m[idx]
        vec_m = vec_m[:, idx]

        pr(f"  K={K} (L={L}):")
        pr(f"    V = T_eff - T_Markov:")
        pr(f"    {V}")
        pr()
        pr(f"    {'n':>4} {'λ_Markov':>14} {'⟨ψ_n|V|ψ_n⟩':>14} "
           f"{'sin²':>14} {'A_perturb':>10}")

        for n in range(c_max + 1):
            psi = vec_m[:, n]
            delta_pt = psi @ V @ psi
            sin2 = np.sin(n * np.pi / (2 * L)) ** 2 if n > 0 else 0
            A_pt = delta_pt / sin2 if sin2 > 1e-10 else float('nan')
            pr(f"    {n:4d} {ev_m[n]:14.8f} {delta_pt:+14.8f} "
               f"{sin2:14.8f} {A_pt:10.4f}")
        pr()

    # ═══════ Part C: Extrapolation of A(K) ══════════════════════════
    pr(f"\n{'═' * 72}")
    pr("C. EXTRAPOLATION: A(K) → A*(∞)?")
    pr(f"{'═' * 72}\n")

    if results:
        pr(f"  {'K':>3} {'A_fit':>12} {'A* - A_fit':>14} {'A_fit/A*':>10}")
        for K in sorted(results):
            A = results[K]['A_avg']
            pr(f"  {K:3d} {A:12.6f} {Astar - A:+14.6f} {A/Astar:10.6f}")

        pr(f"\n  A* = {Astar:.10f}")
        pr()
        pr("  Note: K=5-7 have only c_max+1 = 4 carry states.")
        pr("  The projection is coarse. As K grows, the effective")
        pr("  carry operator should converge to the true operator.")
        pr("  The key question: does A(K) → A* = (2+3π)/(2(1+π))?")

    # ═══════ Part D: Alternative — fit from exact sector data ═══════
    pr(f"\n{'═' * 72}")
    pr("D. ALTERNATIVE: INFER A FROM EXACT SECTOR RATIO R(K)")
    pr(f"{'═' * 72}\n")

    pr("  If R(K) = S(A(K), L(K)) and S(A) = 2(1-A)/(3-2A), then:")
    pr("  A(K) = (3R(K) - 2) / (2R(K) - 2) = (3R-2)/(2(R-1))")
    pr()

    SECTOR_DATA = {
        3: (-1, 0), 4: (-4, -1), 5: (-12, -3),
        6: (-43, -4), 7: (-142, 13), 8: (-434, 167),
        9: (-1326, 963), 10: (-4156, 4648), 11: (-13254, 20590),
        12: (-44302, 87256), 13: (-154150, 360533),
        14: (-560006, 1468632), 15: (-2115994, 5879988),
        16: (-8287938, 23529844), 17: (-31564097, 95791014),
        18: (-124527003, 383811312), 19: (-494176763, 1536648367),
    }

    pr(f"  {'K':>3} {'R(K)':>12} {'A_inferred':>14} {'A* - A':>14} {'digits':>8}")
    for K in sorted(SECTOR_DATA):
        S00, S10 = SECTOR_DATA[K]
        if S00 == 0:
            continue
        R = S10 / S00
        if abs(R - 1) < 1e-10:
            continue
        A_inf = (3 * R - 2) / (2 * (R - 1))
        delta = Astar - A_inf
        digits = -np.log10(abs(delta / Astar)) if abs(delta / Astar) > 1e-15 else 99
        pr(f"  {K:3d} {R:+12.6f} {A_inf:14.8f} {delta:+14.6f} {digits:8.2f}")

    pr()
    pr("  As R(K) → -π, A_inferred → A*.")
    pr("  This is TAUTOLOGICAL (it's inverting the formula).")
    pr("  The non-trivial step: derive A from the OPERATOR, not from R.")

    # ═══════ Part E: Strategy assessment ════════════════════════════
    pr(f"\n{'═' * 72}")
    pr("E. STRATEGY ASSESSMENT FOR CLOSING THE GAP")
    pr(f"{'═' * 72}\n")

    pr("  STRATEGY 1 (Carry projection, this experiment):")
    pr("    - Projects exact T onto carry subspace")
    pr("    - Measures A directly from eigenvalue shifts")
    pr("    - Limited by K=5-7 (only 4 carry states)")
    pr("    - Would need K=15+ for meaningful extrapolation")
    pr("    - VERDICT: informative but not conclusive at current K")
    pr()
    pr("  STRATEGY 2 (Perturbation theory):")
    pr("    - Computes ⟨ψ_n|V|ψ_n⟩ for V = T_eff - T_Markov")
    pr("    - Gives first-order eigenvalue correction")
    pr("    - PROBLEM: the Markov eigenvectors are in 4D carry space,")
    pr("    - but the perturbation lives in 1000+D full space")
    pr("    - VERDICT: first-order correction is too crude")
    pr()
    pr("  STRATEGY 3 (Self-consistency / fixed point):")
    pr("    - A* contains π → system is a fixed-point equation")
    pr("    - If correlations → A → S(A) → R, and R determines")
    pr("      correlations, then π is the fixed point")
    pr("    - MOST PROMISING for an analytical proof")
    pr("    - The equation: A = Φ(S(A)) where Φ encodes how R")
    pr("      feeds back into the correlation structure")
    pr("    - If Φ is derivable, then A* = Φ(-π) gives the proof")
    pr()
    pr("  STRATEGY 4 (Direct from Fejér kernel):")
    pr("    - Digit correlations have triangular profile ρ(d) = (K-d)/(3K)")
    pr("    - Fourier transform of Fejér kernel is (sin(x)/x)²")
    pr("    - This is NOT sin²(nπ/(2L)) — different functional form")
    pr("    - BUT: the effective eigenvalue shift may emerge from")
    pr("      the COMPOSITION of Fejér kernel with bridge propagation")
    pr("    - NEEDS: analytical computation of ρ → δλ_n")
    pr()
    pr("  RECOMMENDED NEXT STEP:")
    pr("  Compute the self-consistency map Φ (Strategy 3).")
    pr("  The equation R = S(A) = 2β/(1+2β) is proved.")
    pr("  The question: given R = -π in the sector ratio,")
    pr("  what eigenvalue shift A does this IMPLY for the next")
    pr("  step of the carry chain?")
    pr("  If A = f(R) = A* when R = -π, the proof is complete.")

    pr(f"\n  Total runtime: {time.time() - t0:.1f}s")
    pr("=" * 72)


if __name__ == '__main__':
    main()
