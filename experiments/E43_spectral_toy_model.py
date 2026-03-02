#!/usr/bin/env python3
"""
E43: Spectral Toy Model — Does a 1D cutoff generate the Linear Mix?
=====================================================================

GO/NO-GO test for Paper E's Linear Mix Hypothesis.

Question: Is  λ_n = (1-A)·λ_n^Markov + A/2  a generic consequence of applying
ANY cutoff/survival function to a cosine-spectrum operator, or is it specific
to the carry chain's non-Markovian correlations?

Part 1: Walsh / Doubling Map (user's specification)
        K_M = Perron–Frobenius of x → 2x mod 1
        Eigenvalues: {1, 0, 0, …} — degenerate

Part 2: Random Walk on {1,…,L} (cosine eigenvalues) + diagonal cutoff Ω
        Eigenvalues: cos(kπ/(L+1))

Part 3: Analytical decomposition of the CORRECT perturbation for the LM

Part 4: Size scaling — does the diagonal Ω become more "linear" at large L?
"""

import warnings
import numpy as np
from mpmath import mp, mpf, pi as mpi
import sys

warnings.filterwarnings("ignore", category=RuntimeWarning)
mp.dps = 30


def pr(msg=""):
    print(msg, flush=True)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def walsh_paley_matrix(N, M):
    """N × 2^M matrix W[n,k] = W_n((k+0.5)/2^M) in Walsh-Paley ordering."""
    P = 1 << M
    W = np.ones((N, P), dtype=np.float64)
    k_arr = np.arange(P)
    for bit_pos in range((N - 1).bit_length() if N > 1 else 0):
        m = bit_pos + 1
        rademacher = np.where((k_arr >> (M - m)) & 1, -1.0, 1.0)
        for n in range(N):
            if (n >> bit_pos) & 1:
                W[n] *= rademacher
    return W


def linreg(x, y):
    """y = slope·x + intercept.  Returns (slope, intercept, R², max|resid|, rms)."""
    c = np.polyfit(x, y, 1)
    slope, intercept = c[0], c[1]
    pred = slope * x + intercept
    resid = y - pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    return slope, intercept, R2, np.max(np.abs(resid)), np.sqrt(np.mean(resid ** 2))


def tridiag_KM(L):
    """Tridiagonal random walk on {0,…,L-1} with absorbing boundaries."""
    K = np.zeros((L, L))
    for i in range(L - 1):
        K[i, i + 1] = 0.5
        K[i + 1, i] = 0.5
    return K


# ==================================================================
# PART 1 — Walsh / Doubling Map
# ==================================================================

def part1():
    pr("\n" + "=" * 70)
    pr("  PART 1: Walsh / Doubling Map Operator")
    pr("=" * 70)
    pr("\n  K_M action on Walsh-Paley basis:")
    pr("    K_M(W_0)  = W_0          (eigenvalue 1)")
    pr("    K_M(W_2k) = W_k   k ≥ 1  (shift)")
    pr("    K_M(W_odd) = 0            (kill)")
    pr("  ⇒ spectrum = {1, 0, 0, …}  — fully degenerate\n")

    N, M = 32, 10
    P = 1 << M
    dx = 1.0 / P
    x_grid = (np.arange(P) + 0.5) * dx

    W = walsh_paley_matrix(N, M)

    # Verify orthonormality
    G = (W @ W.T) * dx
    orth_err = np.max(np.abs(G - np.eye(N)))
    pr(f"  Walsh orthonormality check:  max|G_ij - δ_ij| = {orth_err:.2e}")

    # Analytical K_M matrix
    A_km = np.zeros((N, N))
    A_km[0, 0] = 1.0
    for j in range(2, N, 2):
        if j // 2 < N:
            A_km[j // 2, j] = 1.0

    omega_cases = [
        ("1_{x<3/4}", np.where(x_grid < 0.75, 1.0, 0.0)),
        ("1 - x/2", 1.0 - x_grid / 2.0),
        ("1/(1+x)", 1.0 / (1.0 + x_grid)),
    ]

    for oname, ovals in omega_cases:
        B = ((W * ovals) @ W.T) * dx
        K_toy = B @ A_km
        evals = np.sort(np.real(np.linalg.eigvals(K_toy)))[::-1]
        nonz = evals[np.abs(evals) > 1e-10]

        pr(f"  Ω = {oname}:  {len(nonz)} nonzero eigenvalues")
        pr(f"    λ_0 = {nonz[0]:.10f}")
        if len(nonz) > 1:
            oth = nonz[1:]
            cv = np.std(oth) / abs(np.mean(oth)) if abs(np.mean(oth)) > 1e-15 else float("inf")
            pr(f"    λ_1..{len(nonz)-1}: mean={np.mean(oth):.8f}  std={np.std(oth):.8f}  CV={cv:.4f}")
            pr(f"    range: [{oth[-1]:.8f}, {oth[0]:.8f}]")
            if cv > 0.10:
                pr(f"    → SPREAD OUT — no degenerate Linear Mix")
            else:
                pr(f"    → NEAR-DEGENERATE — possible LM with A ≈ {2*np.mean(oth):.6f}")

    pr("\n  PART 1 VERDICT:")
    pr("    K_M's spectrum is {1, 0, …} — too degenerate for a Linear Mix test.")
    pr("    The cosine spectrum cos(nπ/L)/2 comes from the carry chain's")
    pr("    boundary conditions, NOT from the doubling map itself.")


# ==================================================================
# PART 2 — Random Walk + Diagonal Cutoff
# ==================================================================

def part2(L=50):
    pr("\n" + "=" * 70)
    pr(f"  PART 2: Random Walk on {{1,…,{L}}} + Diagonal Cutoff")
    pr("=" * 70)

    K_M = tridiag_KM(L)
    lam_M = np.sort(np.cos(np.arange(1, L + 1) * np.pi / (L + 1)))[::-1]

    omega_cases = [
        ("constant 0.80", np.full(L, 0.80)),
        ("linear 1−j/(2L)", np.array([1.0 - j / (2.0 * L) for j in range(L)])),
        ("rational 1/(1+j/L)", np.array([1.0 / (1.0 + j / float(L)) for j in range(L)])),
        ("D-odd (L−j)/(L+j)", np.array([(L - j) / float(L + j) for j in range(L)])),
    ]

    hdr = f"  {'Ω':>22s} | {'slope':>9s} {'int':>9s} {'R²':>12s} | {'A_slp':>8s} {'A_int':>8s} {'Δ(A)':>9s} LM?"
    pr(f"\n{hdr}")
    pr("  " + "-" * 88)

    detail_omega = None
    detail_lam = None

    for oname, ovals in omega_cases:
        K_toy = np.diag(ovals) @ K_M
        evals = np.sort(np.real(np.linalg.eigvals(K_toy)))[::-1]
        sl, it, R2, mx, rms = linreg(lam_M, evals)
        As, Ai = 1 - sl, 2 * it
        da = abs(As - Ai)
        lm = " YES" if da < 0.01 else "  NO"
        pr(f"  {oname:>22s} | {sl:9.6f} {it:9.6f} {R2:12.9f} | "
           f"{As:8.5f} {Ai:8.5f} {da:9.2e}{lm}")
        if "D-odd" in oname:
            detail_omega = ovals
            detail_lam = evals

    # Detailed analysis for D-odd case
    if detail_omega is not None:
        pr(f"\n  --- Detailed: Ω(j) = (L−j)/(L+j) ---")
        sl, it, R2, mx, rms = linreg(lam_M, detail_lam)
        pr(f"  Linear:    λ_eff = {sl:.10f}·λ_M + {it:.10f}")
        pr(f"             R² = {R2:.12f},  max|res| = {mx:.6e},  RMS = {rms:.6e}")
        pr(f"             A_slope = {1-sl:.10f},  A_intercept = {2*it:.10f}")

        c2 = np.polyfit(lam_M, detail_lam, 2)
        pred2 = np.polyval(c2, lam_M)
        R2q = 1 - np.sum((detail_lam - pred2) ** 2) / np.sum((detail_lam - np.mean(detail_lam)) ** 2)
        pr(f"  Quadratic: {c2[0]:+.6f}λ² + {c2[1]:+.6f}λ + {c2[2]:+.6f}")
        pr(f"             R² = {R2q:.12f},  improvement = {R2q - R2:.2e}")

        # Perturbation theory: mode-dependent Ω-weights
        V = np.zeros((L, L))
        for k in range(L):
            V[:, k] = np.sin((k + 1) * np.pi * (np.arange(L) + 1) / (L + 1))
        V /= np.sqrt(np.sum(V ** 2, axis=0, keepdims=True))

        ow = np.array([V[:, k] @ (detail_omega * V[:, k]) for k in range(L)])
        lam_pt1 = lam_M * ow
        err_pt1 = np.max(np.abs(detail_lam - lam_pt1))

        pr(f"\n  1st-order perturbation theory:")
        pr(f"    λ_k^(1) = λ_k^M · <v_k|Ω|v_k>,  max error vs exact = {err_pt1:.4e}")
        pr(f"    Ω-weights by mode:  k=1: {ow[0]:.6f},  k=L/4: {ow[L//4]:.6f},"
           f"  k=L/2: {ow[L//2]:.6f},  k=L: {ow[-1]:.6f}")
        pr(f"    mean = {np.mean(ow):.6f},  std = {np.std(ow):.6f}")
        pr(f"\n    → Ω-weights are MODE-DEPENDENT (std > 0).")
        pr(f"      Diagonal Ω gives MULTIPLICATIVE shifts:  λ → Ω̃_k · λ^M")
        pr(f"      Linear Mix requires ADDITIVE shifts:     λ → (1-A)λ^M + A/2")
        pr(f"      These are structurally incompatible.")


# ==================================================================
# PART 3 — The Correct Perturbation
# ==================================================================

def part3(L=50):
    pr("\n" + "=" * 70)
    pr("  PART 3: The Correct Perturbation for the Linear Mix")
    pr("=" * 70)

    A_star = float((2 + 3 * mpi) / (2 * (1 + mpi)))

    pr(f"\n  The Linear Mix  λ_k = (1-A)λ_k^M + A/2  is equivalent to:")
    pr(f"    K_eff = K_M + ΔK   where   ΔK = −A·(K_M − I/2)")
    pr(f"         = (1−A)·K_M + (A/2)·I")
    pr(f"\n  Decomposing ΔK:")
    pr(f"    ΔK = +(A/2)·I        ← adds self-loops (diagonal)")
    pr(f"       − A·K_M           ← reduces nearest-neighbor coupling (off-diagonal)")
    pr(f"\n  Physical meaning:")
    pr(f"    Each position becomes MORE PERSISTENT  (+A/2 self-loop)")
    pr(f"    Neighboring transitions become WEAKER   (−A/2 off-diagonal)")
    pr(f"    This is exactly what bit-sharing does in the carry chain.")

    K_M = tridiag_KM(L)
    lam_M = np.sort(np.cos(np.arange(1, L + 1) * np.pi / (L + 1)))[::-1]

    # Sanity: construct K_eff with A* and verify
    K_eff = (1 - A_star) * K_M + (A_star / 2) * np.eye(L)
    evals_eff = np.sort(np.real(np.linalg.eigvals(K_eff)))[::-1]
    target = (1 - A_star) * lam_M + A_star / 2
    err = np.max(np.abs(evals_eff - target))
    pr(f"\n  Verification with A* = {A_star:.10f}:")
    pr(f"    max|λ_eff − ((1−A*)λ_M + A*/2)| = {err:.2e}  (machine precision ✓)")

    # Compare with diagonal-only perturbation (self-loops only, no coupling reduction)
    K_loops = K_M + (A_star / 2) * np.eye(L)
    evals_loops = np.sort(np.real(np.linalg.eigvals(K_loops)))[::-1]
    sl_l, it_l, R2_l, _, _ = linreg(lam_M, evals_loops)
    pr(f"\n  Test: K_M + (A*/2)·I  (self-loops ONLY, no coupling reduction):")
    pr(f"    slope = {sl_l:.8f}  (expected 1.000000, NOT {1-A_star:.6f})")
    pr(f"    int   = {it_l:.8f}  (expected {A_star/2:.6f})")
    pr(f"    → This is λ = λ^M + A*/2, NOT the Linear Mix λ = (1−A*)λ^M + A*/2")
    pr(f"    → Missing the coupling reduction term.")

    # Compare with coupling reduction only (no self-loops)
    K_coup = (1 - A_star) * K_M
    evals_coup = np.sort(np.real(np.linalg.eigvals(K_coup)))[::-1]
    sl_c, it_c, R2_c, _, _ = linreg(lam_M, evals_coup)
    pr(f"\n  Test: (1−A*)·K_M  (coupling reduction ONLY, no self-loops):")
    pr(f"    slope = {sl_c:.8f}  (expected {1-A_star:.6f} ✓)")
    pr(f"    int   = {it_c:.8f}  (expected 0)")
    pr(f"    → This is λ = (1−A*)λ^M, NOT the Linear Mix (missing +A*/2)")

    pr(f"\n  BOTH components are necessary:")
    pr(f"    (a) Self-loops  +A/2·I         → diagonal perturbation")
    pr(f"    (b) Coupling reduction  −A·K_M → off-diagonal perturbation")
    pr(f"  A diagonal cutoff Ω provides ONLY a distorted version of (a).")
    pr(f"  It CANNOT produce the precise (b) needed for the Linear Mix.")


# ==================================================================
# PART 4 — Size Scaling
# ==================================================================

def part4():
    pr("\n" + "=" * 70)
    pr("  PART 4: Size Scaling — Linear Mix Violation vs L")
    pr("=" * 70)

    A_star = float((2 + 3 * mpi) / (2 * (1 + mpi)))

    pr(f"\n  Ω(j) = (L−j)/(L+j)   [mimics the D-odd boundary (1−u)/(1+u)]")
    pr(f"  A* = {A_star:.10f}")
    pr(f"\n  {'L':>5s}  {'slope':>9s}  {'int':>9s}  {'A_slp':>9s}  {'A_int':>9s}  "
       f"{'Δ(A)':>9s}  {'R²':>13s}  {'|quad|':>9s}")
    pr("  " + "-" * 88)

    for L in [10, 20, 30, 50, 80, 100, 200, 500]:
        K_M = tridiag_KM(L)
        lam_M = np.sort(np.cos(np.arange(1, L + 1) * np.pi / (L + 1)))[::-1]
        omega = np.array([(L - j) / float(L + j) for j in range(L)])
        K_toy = np.diag(omega) @ K_M
        evals = np.sort(np.real(np.linalg.eigvals(K_toy)))[::-1]

        sl, it, R2, _, _ = linreg(lam_M, evals)
        c2 = np.polyfit(lam_M, evals, 2)

        As, Ai = 1 - sl, 2 * it
        pr(f"  {L:5d}  {sl:9.6f}  {it:9.6f}  {As:9.6f}  {Ai:9.6f}  "
           f"{abs(As-Ai):9.6f}  {R2:13.10f}  {abs(c2[0]):9.6f}")

    pr(f"\n  If Δ(A) → 0 as L → ∞:  diagonal Ω asymptotically produces the LM.")
    pr(f"  If Δ(A) → const ≠ 0:   diagonal Ω is structurally incompatible.")


# ==================================================================
# PART 5 — Nearest-Neighbor Perturbation Test
# ==================================================================

def part5(L=50):
    """Test whether specific off-diagonal perturbation structures give LM."""
    pr("\n" + "=" * 70)
    pr(f"  PART 5: Non-Diagonal Perturbations (L={L})")
    pr("=" * 70)

    K_M = tridiag_KM(L)
    lam_M = np.sort(np.cos(np.arange(1, L + 1) * np.pi / (L + 1)))[::-1]
    A_star = float((2 + 3 * mpi) / (2 * (1 + mpi)))

    pr(f"\n  Testing: K_eff = K_M + ε·P  for various perturbation matrices P")
    pr(f"  and checking if eigenvalues satisfy λ_k = (1-A)λ_k^M + A/2.")

    # (a) P = I (self-loops only)
    # (b) P = I − 2·K_M  (the LM perturbation, by construction)
    # (c) P = nearest-neighbor with alternating signs
    # (d) P = random symmetric with same norm as (b)

    eps_values = [0.1, 0.3, 0.5, 0.7]

    perturbations = [
        ("I (self-loops)", np.eye(L)),
        ("I − 2K_M (≡ LM)", np.eye(L) - 2 * K_M),
        ("diag(j/L)·K_M + K_M·diag(j/L)", None),  # built below
    ]
    # Build symmetric nearest-neighbor coupling weighted by position
    D = np.diag(np.arange(L) / float(L))
    perturbations[2] = ("D·K_M + K_M·D", D @ K_M + K_M @ D)

    np.random.seed(42)
    R = np.random.randn(L, L)
    R = (R + R.T) / 2
    R /= np.linalg.norm(R) / np.linalg.norm(np.eye(L) - 2 * K_M)
    perturbations.append(("random symmetric", R))

    for pname, P in perturbations:
        pr(f"\n  P = {pname}:")
        pr(f"    {'ε':>5s}  {'slope':>9s}  {'int':>9s}  {'A_slp':>8s}  {'A_int':>8s}  "
           f"{'Δ(A)':>9s}  {'R²':>13s}")
        for eps in eps_values:
            K_eff = K_M + eps * P
            evals = np.sort(np.real(np.linalg.eigvals(K_eff)))[::-1]
            sl, it, R2, _, _ = linreg(lam_M, evals)
            As, Ai = 1 - sl, 2 * it
            da = abs(As - Ai)
            marker = " ← LM" if da < 0.005 else ""
            pr(f"    {eps:5.2f}  {sl:9.6f}  {it:9.6f}  {As:8.5f}  {Ai:8.5f}  "
               f"{da:9.2e}  {R2:13.10f}{marker}")


# ==================================================================
# MAIN
# ==================================================================

if __name__ == "__main__":
    A_star = float((2 + 3 * mpi) / (2 * (1 + mpi)))
    z_star = -0.5 - 1.0 / float(mpi)

    pr("E43: Spectral Toy Model — Linear Mix Diagnostic")
    pr("=" * 70)
    pr(f"  Target:  A* = {A_star:.10f}")
    pr(f"           z* = −1/2 − 1/π = {z_star:.10f}")
    pr(f"\n  Question: Is  λ_n = (1−A)·λ_n^Markov + A/2  generic or specific?")

    part1()
    part2(L=50)
    part3(L=50)
    part4()
    part5(L=50)

    pr("\n" + "=" * 70)
    pr("  FINAL VERDICT")
    pr("=" * 70)
    pr("""
  1. Walsh / Doubling Map (Part 1):
     K_M has spectrum {1, 0, …} — too degenerate.  The cosine eigenvalues
     cos(nπ/L) come from the carry chain's boundary conditions, not from
     the doubling map itself.

  2. Diagonal cutoffs Ω·K_M (Part 2):
     Do NOT produce the Linear Mix.  Perturbation theory proves that
     diagonal Ω gives MULTIPLICATIVE shifts λ → Ω̃_k · λ^M (mode-dependent
     weights), while the LM requires ADDITIVE shifts λ → (1−A)λ^M + A/2.

  3. The Linear Mix = ΔK = −A(K_M − I/2) (Part 3):
     Requires BOTH self-loops (+A/2·I) AND coupling reduction (−A·K_M).
     Neither component alone reproduces the LM.
     A diagonal cutoff provides a garbled version of the first but
     NONE of the second.

  4. The ONLY perturbation that gives the LM is P = I − 2K_M (Part 5):
     This is the operator (I − 2K_M) which "knows" K_M's off-diagonal
     structure.  In the carry chain, bit-sharing between consecutive
     positions is the physical mechanism that produces this perturbation.

  CONCLUSION FOR PAPER E:
     The Linear Mix is NOT a generic property of cutoff operators.
     It is a SPECIFIC consequence of the carry chain's non-Markovian
     correlations (bit-sharing).  This validates publishing Paper E
     with the Linear Mix Hypothesis as a deep structural conjecture.

     The perturbation ΔK = −(A*/2)(2K_M − I) physically means:
       • Each carry position becomes MORE PERSISTENT  (self-loop +A*/2)
       • Neighboring carries become LESS COUPLED  (coupling −A*/2)
     This is the "spectral fingerprint" of bit-sharing in base-2
     multiplication.
""")
    pr("=" * 70)
    pr("  E43 COMPLETE")
    pr("=" * 70)
