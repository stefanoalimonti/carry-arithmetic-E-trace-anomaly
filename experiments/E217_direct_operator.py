#!/usr/bin/env python3
"""
E217: Direct Measurement of K_eff — The Carry Transfer Operator

Instead of INFERRING K_eff from the output R(K), measure it DIRECTLY
from the carry transition statistics in D-odd products.

METHOD:
  For each K, cycle through all D-odd products (X·Y with D=2K-1 bits).
  At each position j (1 ≤ j ≤ D-2), record the carry transition:
      (carry[j], carry[j+1])  ∈ {(0,0), (0,1), (1,0), (1,1)}
  Build the empirical 2×2 transition matrix at each layer j.
  Average across all layers and products → K_empirical(K).

TESTS:
  1. Is K_empirical close to K_Markov = [[3/4, 1/4], [1/4, 3/4]]?
  2. Does K_empirical - K_Markov have the LMH structure?
     LMH predicts: ΔK = -A*(K_Markov - I/2)
                       = -A* · [[ 1/4, 1/4], [1/4, 1/4]] · (-1)  ... hmm
     Actually:  K_eff = (1-A)K_M + (A/2)I
                K_eff - K_M = -A(K_M - I/2)
     K_M - I/2 = [[1/4, 1/4], [1/4, 1/4]] ... no, wait.
     K_M = [[3/4, 1/4], [1/4, 3/4]] (for base-2 addition carry)
     I/2 = [[1/2, 0], [0, 1/2]]
     K_M - I/2 = [[1/4, 1/4], [1/4, 1/4]]
     So ΔK = K_eff - K_M = -A · [[1/4, 1/4], [1/4, 1/4]]
     K_eff = K_M - A·[[1/4, 1/4], [1/4, 1/4]]
           = [[3/4 - A/4, 1/4 - A/4], [1/4 - A/4, 3/4 - A/4]]
     With A* ≈ 1.379: K_eff ≈ [[0.405, -0.095], [-0.095, 0.405]]

  3. Per-layer variation: is K(j) approximately stationary (same at every j)?
  4. Boundary effects: do layers near j=0 or j=D-1 deviate?
  5. Extract A_eff from the measured ΔK and compare to A*.

IMPORTANT DISTINCTION:
  The carry chain for MULTIPLICATION is NOT the same as for addition.
  In multiplication, carry[j+1] = floor((conv[j] + carry[j]) / 2),
  where conv[j] = Σ a_i·b_{j-i} depends on ALL bits of X and Y.
  The "Markov" approximation treats carry[j+1] as depending only on
  carry[j], but in reality it depends on the digit pair (a,b) at
  position j, which introduces non-Markovian correlations.
  The D-odd conditioning (carry[D-1] = 0) further deforms the chain.
"""

import numpy as np
import math
import time

PI = math.pi


def pr(s=""):
    print(s, flush=True)


def measure_transitions(K):
    """
    For K-bit D-odd multiplication, measure carry transition counts
    at each layer j.  Returns:
      trans[j][c_in][c_out] = count of transitions carry[j]=c_in → carry[j+1]=c_out
      total[j] = total transitions at layer j
    Also returns per-sector transition counts.
    """
    D = 2 * K - 1
    X_lo, X_hi = 1 << (K - 1), 1 << K
    Y_lo = X_lo
    Y_hi = X_lo + (1 << (K - 2))
    D_lo, D_hi = 1 << (D - 1), 1 << D

    trans = np.zeros((D + 1, 2, 2), dtype=np.int64)
    layer_count = np.zeros(D + 1, dtype=np.int64)

    Y_arr = np.arange(Y_lo, Y_hi, dtype=np.int64)
    n_dodd = 0

    for X in range(X_lo, X_hi):
        P_arr = X * Y_arr
        dodd = (P_arr >= D_lo) & (P_arr < D_hi)
        n_valid = int(dodd.sum())
        if n_valid == 0:
            continue
        n_dodd += n_valid

        valid_Y = Y_arr[dodd]

        carries = np.zeros((D + 2, n_valid), dtype=np.int8)
        for j in range(D):
            conv_j = np.zeros(n_valid, dtype=np.int32)
            lo, hi = max(0, j - K + 1), min(j, K - 1)
            for i in range(lo, hi + 1):
                if (X >> i) & 1:
                    conv_j += ((valid_Y >> (j - i)) & 1).astype(np.int32)
            carries[j + 1] = ((conv_j + carries[j].astype(np.int32)) >> 1).astype(np.int8)

        for j in range(D - 1):
            c_in  = carries[j + 1]     # carry at position j+1
            c_out = carries[j + 2]     # carry at position j+2

            for ci in range(2):
                for co in range(2):
                    count = int(np.sum((c_in == ci) & (c_out == co)))
                    trans[j][ci][co] += count
            layer_count[j] += n_valid

    return trans, layer_count, n_dodd, D


def transition_matrix(counts):
    """Convert raw counts to row-stochastic transition matrix."""
    T = counts.astype(np.float64)
    row_sums = T.sum(axis=1)
    for i in range(T.shape[0]):
        if row_sums[i] > 0:
            T[i] /= row_sums[i]
    return T


def main():
    pr()
    pr("E217: DIRECT MEASUREMENT OF K_eff")
    pr("=" * 72)
    pr()

    K_MARKOV = np.array([[3/4, 1/4], [1/4, 3/4]])
    A_star = (2 + 3*PI) / (2*(1 + PI))
    K_LMH = (1 - A_star) * K_MARKOV + (A_star / 2) * np.eye(2)

    pr(f"  K_Markov (Holte, base-2 addition):")
    pr(f"    [[{K_MARKOV[0,0]:.4f}, {K_MARKOV[0,1]:.4f}],")
    pr(f"     [{K_MARKOV[1,0]:.4f}, {K_MARKOV[1,1]:.4f}]]")
    pr()
    pr(f"  K_LMH (predicted, A* = {A_star:.6f}):")
    pr(f"    [[{K_LMH[0,0]:.4f}, {K_LMH[0,1]:.4f}],")
    pr(f"     [{K_LMH[1,0]:.4f}, {K_LMH[1,1]:.4f}]]")
    pr()
    delta_LMH = K_LMH - K_MARKOV
    pr(f"  ΔK_LMH = K_LMH - K_Markov:")
    pr(f"    [[{delta_LMH[0,0]:+.4f}, {delta_LMH[0,1]:+.4f}],")
    pr(f"     [{delta_LMH[1,0]:+.4f}, {delta_LMH[1,1]:+.4f}]]")
    pr()
    pr(f"  LMH structure: ΔK should be proportional to")
    pr(f"    -(K_M - I/2) = [[{-(K_MARKOV[0,0]-0.5):+.4f}, {-(K_MARKOV[0,1]-0):+.4f}],")
    pr(f"                     [{-(K_MARKOV[1,0]-0):+.4f}, {-(K_MARKOV[1,1]-0.5):+.4f}]]")
    pr(f"  with proportionality constant A* = {A_star:.6f}")
    pr()

    # ─── Phase 1: Measure per-layer transitions ────────────────
    pr("PHASE 1: PER-LAYER TRANSITION MATRICES")
    pr("─" * 72)

    all_results = []
    for K in range(5, 16):
        t0 = time.time()
        trans, layer_count, n_dodd, D = measure_transitions(K)
        dt = time.time() - t0

        pr(f"\n  K={K:2d}  D={D:2d}  n_dodd={n_dodd:10d}  [{dt:.1f}s]")

        layer_matrices = []
        for j in range(D - 2):
            T_j = transition_matrix(trans[j])
            layer_matrices.append(T_j)

        interior_start = max(1, D // 4)
        interior_end   = min(D - 2, 3 * D // 4)
        if interior_end <= interior_start:
            interior_start, interior_end = 1, D - 2

        avg_counts = np.zeros((2, 2), dtype=np.int64)
        for j in range(interior_start, interior_end):
            avg_counts += trans[j]
        K_emp = transition_matrix(avg_counts)

        delta = K_emp - K_MARKOV

        pr(f"  K_emp (interior layers j={interior_start}..{interior_end-1}):")
        pr(f"    [[{K_emp[0,0]:.6f}, {K_emp[0,1]:.6f}],")
        pr(f"     [{K_emp[1,0]:.6f}, {K_emp[1,1]:.6f}]]")
        pr(f"  ΔK = K_emp - K_Markov:")
        pr(f"    [[{delta[0,0]:+.6f}, {delta[0,1]:+.6f}],")
        pr(f"     [{delta[1,0]:+.6f}, {delta[1,1]:+.6f}]]")

        target = -(K_MARKOV - 0.5 * np.eye(2))
        flat_delta = delta.flatten()
        flat_target = target.flatten()
        if np.dot(flat_target, flat_target) > 0:
            A_measured = np.dot(flat_delta, flat_target) / np.dot(flat_target, flat_target)
        else:
            A_measured = float('nan')

        residual = delta - A_measured * target
        res_norm = np.linalg.norm(residual)
        delta_norm = np.linalg.norm(delta)
        rel_res = res_norm / delta_norm if delta_norm > 1e-15 else float('inf')

        pr(f"  A_measured = {A_measured:.6f}  (A* = {A_star:.6f}, "
           f"gap = {abs(A_measured - A_star):.6f})")
        pr(f"  LMH fit residual: |ΔK - A·target| / |ΔK| = {rel_res:.6f}")

        eigenvalues = np.linalg.eigvals(K_emp)
        pr(f"  eigenvalues(K_emp) = [{eigenvalues[0]:.6f}, {eigenvalues[1]:.6f}]")
        pr(f"  eigenvalues(K_LMH) = [{np.linalg.eigvals(K_LMH)[0]:.6f}, "
           f"{np.linalg.eigvals(K_LMH)[1]:.6f}]")

        all_results.append({
            'K': K, 'D': D, 'n_dodd': n_dodd,
            'K_emp': K_emp, 'delta': delta,
            'A_measured': A_measured, 'rel_residual': rel_res,
            'trans': trans, 'layer_count': layer_count,
            'layer_matrices': layer_matrices,
        })

        if dt > 600:
            pr("  [time limit]")
            break

    # ─── Phase 2: A_measured convergence ───────────────────────
    pr()
    pr("PHASE 2: A_measured(K) CONVERGENCE")
    pr("─" * 72)
    pr()
    pr(f"  {'K':>3s}  {'A_meas':>10s}  {'A*':>10s}  {'gap':>10s}  "
       f"{'LMH fit':>10s}  {'ratio':>10s}")
    pr(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    prev_gap = None
    for r in all_results:
        gap = abs(r['A_measured'] - A_star)
        ratio_str = f"{gap/prev_gap:.4f}" if prev_gap and prev_gap > 1e-15 else "—"
        pr(f"  {r['K']:3d}  {r['A_measured']:10.6f}  {A_star:10.6f}  "
           f"{gap:10.6f}  {r['rel_residual']:10.6f}  {ratio_str:>10s}")
        prev_gap = gap

    # ─── Phase 3: Layer-by-layer variation ─────────────────────
    pr()
    pr("PHASE 3: LAYER-BY-LAYER VARIATION (largest K)")
    pr("─" * 72)
    pr()

    r = all_results[-1]
    K, D = r['K'], r['D']
    pr(f"  K={K}, D={D}: transition matrix diagonal K(j)[0,0] per layer")
    pr(f"  (K_Markov[0,0] = 0.75, K_LMH[0,0] = {K_LMH[0,0]:.4f})")
    pr()
    pr(f"  {'j':>4s}  {'K[0,0]':>8s}  {'K[0,1]':>8s}  {'K[1,0]':>8s}  "
       f"{'K[1,1]':>8s}  {'A_j':>8s}")
    pr(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    target = -(K_MARKOV - 0.5 * np.eye(2))
    flat_target = target.flatten()

    for j in range(min(D - 2, len(r['layer_matrices']))):
        T_j = r['layer_matrices'][j]
        dj = T_j - K_MARKOV
        A_j = (np.dot(dj.flatten(), flat_target) / np.dot(flat_target, flat_target)
               if np.dot(flat_target, flat_target) > 0 else float('nan'))
        pr(f"  {j:4d}  {T_j[0,0]:8.4f}  {T_j[0,1]:8.4f}  "
           f"{T_j[1,0]:8.4f}  {T_j[1,1]:8.4f}  {A_j:+8.4f}")

    # ─── Phase 4: Structure test ───────────────────────────────
    pr()
    pr("PHASE 4: LMH STRUCTURE TEST")
    pr("─" * 72)
    pr()
    pr("  ΔK = K_emp - K_Markov. LMH predicts ΔK = -A*(K_M - I/2).")
    pr("  The matrix K_M - I/2 = [[1/4, 1/4],[1/4, 1/4]] has rank 1.")
    pr("  So ΔK should also be rank-1 and proportional to [[1,1],[1,1]].")
    pr()

    for r in all_results:
        delta = r['delta']
        flat = delta.flatten()
        if abs(flat[0]) > 1e-15:
            ratios = flat / flat[0]
            is_rank1 = all(abs(ratios[i] - 1.0) < 0.2 for i in range(4))
            pr(f"  K={r['K']:2d}:  ΔK ratios = [{ratios[0]:.3f}, {ratios[1]:.3f}, "
               f"{ratios[2]:.3f}, {ratios[3]:.3f}]  "
               f"{'rank-1 ✓' if is_rank1 else 'NOT rank-1 ✗'}")
        else:
            pr(f"  K={r['K']:2d}:  ΔK[0,0] ≈ 0, cannot test rank-1")

    # ─── Verdict ───────────────────────────────────────────────
    pr()
    pr("=" * 72)
    pr("VERDICT")
    pr("=" * 72)
    pr()

    last = all_results[-1]
    pr(f"  A_measured(K={last['K']}) = {last['A_measured']:.6f}")
    pr(f"  A*                        = {A_star:.6f}")
    pr(f"  gap                       = {abs(last['A_measured'] - A_star):.6f}")
    pr(f"  LMH fit residual          = {last['rel_residual']:.6f}")
    pr()

    if last['rel_residual'] < 0.05:
        pr("  ═══ LMH STRUCTURE CONFIRMED MICROSCOPICALLY ═══")
        pr("  K_emp - K_Markov is proportional to -(K_M - I/2).")
        pr("  The Linear Mix Hypothesis holds at the operator level,")
        pr("  not just at the scalar level (R → -π).")
    elif last['rel_residual'] < 0.2:
        pr("  ─── LMH structure approximately holds ───")
        pr("  Residual is moderate. ΔK is close to rank-1 form.")
    else:
        pr("  ═══ LMH STRUCTURE NOT CONFIRMED ═══")
        pr("  K_emp - K_Markov does NOT have the predicted rank-1 form.")
        pr("  The macroscopic R → -π must come from a different mechanism.")

    pr()
    pr("  KEY QUESTION: Does A_measured converge to A* as K → ∞?")
    if len(all_results) >= 3:
        gaps = [abs(r['A_measured'] - A_star) for r in all_results[-3:]]
        if gaps[-1] < gaps[-2] < gaps[-3]:
            pr("  Last 3 gaps are monotonically decreasing — CONVERGENCE.")
        elif gaps[-1] < gaps[0]:
            pr("  Overall gap is decreasing — SLOW CONVERGENCE.")
        else:
            pr("  Gap is NOT clearly decreasing — CONVERGENCE UNCERTAIN.")
    pr()


if __name__ == "__main__":
    main()
