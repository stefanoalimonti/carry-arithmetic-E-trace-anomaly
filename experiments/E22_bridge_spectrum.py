#!/usr/bin/env python3
"""
E22: Exact spectrum of the carry chain Markov bridge

The carry chain transition at position j with n_j terms is:
  c_{j+1} = floor((conv_j + c_j) / 2)
  conv_j ~ Binomial(n_j, 1/4)
  P(c'|c) = P(conv=2c'-c) + P(conv=2c'-c+1)

Strategy:
  A. Build position-dependent transition matrices T_j
  B. Compute the full bridge transfer matrix M = T_{D-2}...T_0
  C. Extract val expectation from the Markov bridge model
  D. Compare with exact enumeration
  E. Diagonalize the BULK kernel (n=K terms) — eigenvalue spectrum
  F. Compute the spectral sum with exact eigenvalues to find -2π factor
"""

import numpy as np
from scipy.special import comb as binom_coeff
from collections import defaultdict
import math

def pr(s=""):
    print(s)

# ---------- Transition matrix construction ----------
def build_transition(n_terms, max_carry):
    """Build transition matrix T(c'|c) for carry chain with n_terms in convolution.
    conv ~ Binomial(n_terms, 1/4).
    c' = floor((conv + c) / 2).
    """
    T = np.zeros((max_carry + 1, max_carry + 1))
    for c in range(max_carry + 1):
        for v in range(n_terms + 1):
            p_v = binom_coeff(n_terms, v, exact=True) * (0.25)**v * (0.75)**(n_terms - v)
            c_new = (v + c) // 2
            if c_new <= max_carry:
                T[c_new, c] += p_v
    return T


def n_terms_at(j, K):
    """Number of terms in convolution at position j for K-bit multiplication."""
    D = 2*K - 1
    return min(j + 1, K, D - j)


# ---------- Exact enumeration ----------
def enumerate_dodd(K):
    D = 2*K - 1
    lo, hi = 1 << (K-1), 1 << K
    results = defaultdict(lambda: {'count': 0, 'sum_val': 0, 'vals': []})
    for p in range(lo, hi):
        for q in range(lo, hi):
            n = p * q
            if n.bit_length() != D:
                continue
            carries = [0] * (D + 1)
            for j in range(D):
                conv_j = 0
                for i in range(max(0, j-K+1), min(j, K-1)+1):
                    conv_j += ((p >> i) & 1) * ((q >> (j-i)) & 1)
                carries[j+1] = (conv_j + carries[j]) >> 1
            M = 0
            cm1 = 0
            for j in range(D, 0, -1):
                if carries[j] > 0:
                    M = j
                    cm1 = carries[j-1]
                    break
            if M == 0:
                continue
            val = cm1 - 1
            a = (p >> (K-2)) & 1
            c_sec = (q >> (K-2)) & 1
            sec = (a, c_sec)
            results[sec]['count'] += 1
            results[sec]['sum_val'] += val
            results[sec]['vals'].append(val)
    return results, D


# ================================================================
# PART A: Transition matrices at different positions
# ================================================================
pr("=" * 70)
pr("PART A: Carry chain transition matrices")
pr("=" * 70)

K = 10
D = 2*K - 1
max_c = K  # max carry in the bulk

pr(f"K={K}, D={D}, max_carry={max_c}")

# Build and show eigenvalues at key positions
for j in [0, K//2, K-2, K-1, K, 3*K//2, D-4, D-3, D-2]:
    if j >= D:
        continue
    n = n_terms_at(j, K)
    T = build_transition(n, max_c)
    evals = np.sort(np.real(np.linalg.eigvals(T)))[::-1]
    evals_sig = evals[evals > 1e-10]
    pr(f"  j={j:3d} (n={n:2d}): top eigenvalues = {evals_sig[:6]}")


# ================================================================
# PART B: Bulk kernel spectrum (n=K terms)
# ================================================================
pr("\n" + "=" * 70)
pr("PART B: Bulk kernel spectrum (n=K terms)")
pr("=" * 70)

for K_val in [6, 8, 10]:
    max_c = K_val
    T_bulk = build_transition(K_val, max_c)
    evals = np.sort(np.real(np.linalg.eigvals(T_bulk)))[::-1]
    evals_sig = evals[np.abs(evals) > 1e-12]

    pr(f"\n  K={K_val}: Bulk transition matrix ({max_c+1}x{max_c+1})")
    pr(f"  Eigenvalues: {evals_sig}")
    pr(f"  Spectral gap: 1 - λ₂ = {1 - evals_sig[1]:.8f}")
    pr(f"  1/2 = {0.5:.8f}")

    # Eigenvalue ratios
    pr(f"  Eigenvalue gaps 1-λ_n:")
    for i, ev in enumerate(evals_sig):
        gap = 1 - ev
        pr(f"    n={i}: λ={ev:+.8f}, 1-λ={gap:.8f}")


# ================================================================
# PART C: Full bridge transfer matrix
# ================================================================
pr("\n" + "=" * 70)
pr("PART C: Full Markov bridge — val expectation")
pr("=" * 70)
pr("Build the bridge M = T_{D-2}...T_0, conditioned on c_0=0, c_{D-1}=0")

for K_val in [6, 8, 10]:
    D_val = 2*K_val - 1
    max_c = K_val + 2  # safety margin

    pr(f"\n--- K={K_val}, D={D_val} ---")

    # Build forward propagation: P(c_j | c_0=0)
    # Start with e_0 = [1, 0, 0, ...]
    forward = [np.zeros(max_c + 1)]
    forward[0][0] = 1.0

    transition_mats = []
    for j in range(D_val):
        n = n_terms_at(j, K_val)
        T = build_transition(n, max_c)
        transition_mats.append(T)
        fwd_next = T @ forward[-1]
        forward.append(fwd_next)

    # Backward propagation: P(c_{D-1}=0 | c_j)
    # Start from e_0 at position D
    backward = [None] * (D_val + 1)
    backward[D_val] = np.zeros(max_c + 1)
    backward[D_val][0] = 1.0

    for j in range(D_val - 1, -1, -1):
        T = transition_mats[j]
        backward[j] = T.T @ backward[j + 1]

    # Bridge probability: Z = P(c_{D-1}=0 | c_0=0) = forward[D][0]
    # Actually Z = forward[-1][0] but let me use forward-backward properly
    # P(c_j=k | bridge) = forward[j][k] * backward[j][k] / Z
    Z = np.dot(forward[D_val], backward[D_val])
    pr(f"  Bridge partition Z = P(c_0=0, c_{{D-1}}=0) = {Z:.8e}")

    # Now compute ⟨val⟩ for each "sector" (Markov model)
    # In the Markov model, sector doesn't exist — the sector modifies
    # the transition matrices at positions K-2 to D-2.
    # For sector (0,0): standard transitions (conv has n_j terms with p_{K-2}=0)
    # For sector (1,0): conv at position j has an extra term +q_{j-K+2} for j >= K-2

    # In the Markov approximation, sector (1,0) adds 1 term to n_j for j ∈ [K-2, D-2]
    # (actually it's more subtle: the digit a=1 contributes to conv at those positions)
    # For the Markov model, this means n_j → n_j + 1 for those positions.
    # Wait, no — a=p_{K-2} is a SPECIFIC digit, not a random one. But in the Markov
    # approximation, we average over it. So the transition kernel already includes
    # the average over all digit configurations.
    #
    # The sector conditioning FIXES a=1, c=0. This changes the convolution distribution.
    # For position j ∈ [K-2, D-2], the term p_{K-2}·q_{j-K+2} = 1·q_{j-K+2} = q_{j-K+2}
    # which is Bernoulli(1/2). So conv_j gains an extra Bernoulli(1/2) term.
    # Also, the term p_{j-K+2}·q_{K-2} = p_{j-K+2}·0 = 0 (since c=q_{K-2}=0),
    # which REMOVES one Bernoulli(1/4) term.
    #
    # Net effect on conv at position j: +Bernoulli(1/2) - Bernoulli(1/4)
    # In the Markov approximation: the transition kernel changes.

    # Let me build sector-specific transition matrices.
    def build_transition_sector(n_terms_orig, max_carry, add_half=False, remove_quarter=False):
        """Build transition matrix with modified convolution.
        add_half: add a Bernoulli(1/2) term to conv
        remove_quarter: remove a Bernoulli(1/4) term from conv
        """
        n_eff = n_terms_orig
        if remove_quarter:
            n_eff -= 1
        T = np.zeros((max_carry + 1, max_carry + 1))
        for c in range(max_carry + 1):
            # Distribution of conv: Binomial(n_eff, 1/4) possibly + Bernoulli(1/2)
            for v_base in range(n_eff + 1):
                p_base = binom_coeff(n_eff, v_base, exact=True) * (0.25)**v_base * (0.75)**(n_eff - v_base)
                if add_half:
                    for extra in [0, 1]:
                        p_extra = 0.5
                        v = v_base + extra
                        p_v = p_base * p_extra
                        c_new = (v + c) // 2
                        if c_new <= max_carry:
                            T[c_new, c] += p_v
                else:
                    v = v_base
                    c_new = (v + c) // 2
                    if c_new <= max_carry:
                        T[c_new, c] += p_base
        return T

    # Build bridge for sector (0,0): standard transitions
    # In (0,0), a=0 and c=0, so the sector bits don't contribute.
    # Actually, in (0,0), p_{K-2}=0 removes one term (p_{K-2}·q_{j-K+2}=0)
    # and q_{K-2}=0 removes another term (p_{j-K+2}·q_{K-2}=0).
    # So in sector (0,0), the convolution at j ∈ [K-2, D-2] has n_j - 2 random terms
    # plus 2 zero terms = same as n_j-2 effective terms... 
    # No wait, the standard n_j already counts ALL terms including those involving
    # p_{K-2} and q_{K-2}. In the UNCONDITIONED model, these are random.
    # In sector (0,0), they're fixed to 0.
    # In sector (1,0), p_{K-2}=1 and q_{K-2}=0.
    #
    # So sector-specific transitions differ from the unconditioned model.
    # Let me handle this properly.
    
    # The unconditioned model has n_j Bernoulli(1/4) terms.
    # In sector (0,0): 2 of these terms are fixed to 0 → n_j-2 random terms
    # In sector (1,0): one term (p_{K-2}·q_{j-K+2}) is Bernoulli(1/2) instead of 
    #   Bernoulli(1/4) (because p_{K-2}=1 so it becomes q_{j-K+2} ~ Bernoulli(1/2))
    #   and one term (p_{j-K+2}·q_{K-2}) is 0 (since q_{K-2}=0) instead of Bernoulli(1/4)
    # Net: n_j-2 Bernoulli(1/4) terms + 1 Bernoulli(1/2) term
    # Wait, actually for j in [K-2, 2K-3], the sector bits affect TWO positions
    # in the convolution: i=K-2 and i such that j-i=K-2.
    # When i=K-2: term = p_{K-2}·q_{j-K+2}
    # When j-i=K-2 → i=j-K+2: term = p_{j-K+2}·q_{K-2}
    # These are different terms (unless j=2K-4, where both give i=K-2).

    # For sector (0,0): both terms are 0. So remove 2 terms from n_j.
    # For sector (1,0): first term becomes Bernoulli(1/2), second is 0. 
    # So remove 2 terms, add back 1 Bernoulli(1/2).

    # Positions affected: j ∈ [K-2, 2K-3] (= [K-2, D-2])
    # But at positions j < K-2 or j > D-2, sector bits don't appear in the convolution.

    # Let me build transition matrices for each sector.
    
    # Helper: transition for n_q Bernoulli(1/4) terms + n_h Bernoulli(1/2) terms
    def build_T_mixed(n_quarter, n_half, max_carry):
        """Transition matrix where conv = sum of n_quarter Bern(1/4) + n_half Bern(1/2)."""
        T = np.zeros((max_carry + 1, max_carry + 1))
        # Distribution of v = v_q + v_h where v_q ~ Binom(n_q, 1/4), v_h ~ Binom(n_h, 1/2)
        for c in range(max_carry + 1):
            for vq in range(n_quarter + 1):
                p_q = binom_coeff(n_quarter, vq, exact=True) * (0.25)**vq * (0.75)**(n_quarter-vq)
                for vh in range(n_half + 1):
                    p_h = binom_coeff(n_half, vh, exact=True) * (0.5)**n_half
                    v = vq + vh
                    p_v = p_q * p_h
                    c_new = (v + c) // 2
                    if c_new <= max_carry:
                        T[c_new, c] += p_v
        return T

    # Build sector-specific bridge
    for sec_label, sec_type in [("(0,0)", "00"), ("(1,0)", "10")]:
        fwd = np.zeros(max_c + 1)
        fwd[0] = 1.0

        # Also track val expectation using forward-backward
        fwd_list = [fwd.copy()]

        for j in range(D_val):
            n = n_terms_at(j, K_val)
            if K_val - 2 <= j <= D_val - 2:
                # Sector-affected position
                if sec_type == "00":
                    # Remove 2 Bernoulli(1/4) terms
                    T = build_T_mixed(max(0, n - 2), 0, max_c)
                else:  # "10"
                    # Remove 2 Bernoulli(1/4), add 1 Bernoulli(1/2)
                    T = build_T_mixed(max(0, n - 2), 1, max_c)
            else:
                T = build_transition(n, max_c)

            fwd = T @ fwd
            fwd_list.append(fwd.copy())

        Z_sec = fwd[0]
        mean_val_markov = float('nan')  # will compute below

        # For val computation, need to compute Σ_M P(M=m, c_{M-1}=c) · (c-1)
        # P(M=m) means c_m > 0 and c_{m+1}=...=c_{D-1}=0
        # Use forward-backward:
        # P(c_m=k, c_{m+1}=0,...,c_{D-1}=0, c_0=0) = fwd_list[m][k] · bwd_from_m[k]
        # where bwd_from_m = probability of reaching 0 at D-1 starting from k at m,
        # going through 0 at m+1, ..., 0 at D-1.

        # Actually, P(M=m) means c_m > 0 AND carries at m+1...D-1 are all 0.
        # But carries at m+1 depends on transition from m, etc.
        # P(c_{m+1}=0 | c_m=k) is not the same as all subsequent being 0.

        # Simpler: compute P(c_j > 0, c_{j+1}=0, ..., c_{D-1}=0 | c_0=0)
        # This requires the "absorption probability" from position j.

        # Let's compute iteratively from the top:
        # prob_zero_from[j] = P(c_j=0, c_{j+1}=0, ..., c_{D-1}=0 | c_j)
        # prob_zero_from[D-1][0] = 1, prob_zero_from[D-1][k>0] = 0

        prob_zero_from = [None] * (D_val + 1)
        prob_zero_from[D_val] = np.zeros(max_c + 1)
        prob_zero_from[D_val][0] = 1.0

        # Actually we need: probability that c_j=0, c_{j+1}=0, ..., c_{D-1}=0
        # given we're at carry k at position j.
        # This is: for c_j=0 and all subsequent=0, we need:
        # T_j(0|k) · T_{j+1}(0|0) · ... · T_{D-2}(0|0)
        # which equals T_j(0|k) · prod_{l=j+1}^{D-2} T_l(0|0)

        # Let's compute P(all zeros from j+1 to D-1 | c_j = k)
        # = Σ_{k'} T_j(k'|k) · P(all zeros from j+2 to D-1 | c_{j+1}=k')
        # with boundary: P(... | position D-1, c=0) = 1, P(... | position D-1, c>0) = 0

        # Start from position D-1
        # prob_allzero[j][k] = P(c_{j+1}=...=c_{D-1}=0 | c_j=k)
        prob_allzero = [None] * D_val
        prob_allzero_base = np.zeros(max_c + 1)
        prob_allzero_base[0] = 1.0  # at position D-1, "all zero from D" is trivially true if c_{D-1}=0

        # Hmm, let me redefine:
        # tail[j][k] = P(c_j = k AND c_{j+1}=...=c_{D-1}=0 | bridge starts from c_0=0)
        # = P(c_j = k | c_0=0) · P(c_{j+1}=0,...,c_{D-1}=0 | c_j = k)
        # = fwd_list[j][k] · prob_tail[j][k]
        
        # where prob_tail[j][k] = P(c_{j+1}=0,...,c_{D-1}=0 | c_j = k)

        # Compute prob_tail backward:
        # prob_tail[D-1][k] = 1 if k=0, 0 otherwise (c_{D-1}=0 is the condition)
        # Wait, prob_tail[D-1][k] should be P(no more conditions | c_{D-1}=k) = δ_{k,0}
        
        # Actually this is getting complicated. Let me use a different approach.
        # I'll just compute M and val directly from the forward distribution.
        
        # For each position M from D-1 down to 1:
        # P(M=m | bridge) = P(c_m > 0, c_{m+1}=...=c_{D-1}=0 | c_0=0, c_{D-1}=0)
        # And ⟨c_{M-1} | M=m, bridge⟩ requires the joint distribution (c_{m-1}, c_m).

        # Let me compute this using matrix products more carefully.
        # I'll track the 2D state (c_{j-1}, c_j) through the chain.

        # Actually, the simplest approach: just sample from the Markov bridge.
        # For K=6,8,10, sampling is fast.

        # Or: compute exactly using the forward distributions.
        # The bridge distribution at position j is:
        # P_bridge(c_j = k) = fwd_list[j][k] * bwd_list[j][k] / Z
        
        # where bwd_list[j][k] = P(c_{D-1}=0 | c_j=k) (backward from D-1)

        # Recompute backward properly for this sector
        bwd = [None] * (D_val + 1)
        bwd[D_val] = np.zeros(max_c + 1)
        bwd[D_val][0] = 1.0

        for j in range(D_val - 1, -1, -1):
            n = n_terms_at(j, K_val)
            if K_val - 2 <= j <= D_val - 2:
                if sec_type == "00":
                    T = build_T_mixed(max(0, n - 2), 0, max_c)
                else:
                    T = build_T_mixed(max(0, n - 2), 1, max_c)
            else:
                T = build_transition(n, max_c)
            bwd[j] = T.T @ bwd[j + 1]

        # Now: P_bridge(c_j=k) ∝ fwd_list[j][k] · bwd[j][k]
        # Z_check = sum(fwd_list[D_val] * bwd[D_val]) should = fwd_list[D_val][0]
        Z_check = np.sum(fwd_list[0] * bwd[0])
        
        # Compute M distribution and ⟨val|M⟩
        # P(M ≥ j) = P(c_j > 0 | bridge) = Σ_{k>0} fwd[j][k]*bwd[j][k] / Z_check
        # P(M = j) = P(c_j > 0, c_{j+1}=0 | bridge)
        
        # For joint (c_j, c_{j+1}):
        # P(c_j=k, c_{j+1}=k' | bridge) ∝ fwd[j][k] · T_j(k'|k) · bwd[j+1][k']
        
        # P(M=m, c_{m-1}=c) = P(c_{m-1}=c, c_m>0, c_{m+1}=0 | bridge)
        # This requires: P(c_{m-1}=c, c_m=k, c_{m+1}=0 | bridge) for k>0

        total_val = 0.0
        total_prob = 0.0

        for m in range(1, D_val):
            # P(c_{m-1}=c, c_m=k, c_{m+1}=0)
            # Need to split into consecutive transitions

            n_m_minus_1 = n_terms_at(m-1, K_val)
            n_m = n_terms_at(m, K_val)

            if K_val - 2 <= (m-1) <= D_val - 2:
                if sec_type == "00":
                    T_m1 = build_T_mixed(max(0, n_m_minus_1 - 2), 0, max_c)
                else:
                    T_m1 = build_T_mixed(max(0, n_m_minus_1 - 2), 1, max_c)
            else:
                T_m1 = build_transition(n_m_minus_1, max_c)

            if K_val - 2 <= m <= D_val - 2:
                if sec_type == "00":
                    T_m = build_T_mixed(max(0, n_m - 2), 0, max_c)
                else:
                    T_m = build_T_mixed(max(0, n_m - 2), 1, max_c)
            else:
                T_m = build_transition(n_m, max_c)

            for c_prev in range(max_c + 1):
                for k in range(1, max_c + 1):  # c_m > 0
                    # P(c_{m-1}=c_prev | c_0=0) = fwd_list[m-1][c_prev] (unnormalized by bridge)
                    # P(c_m=k | c_{m-1}=c_prev) = T_{m-1}(k | c_prev)
                    # P(c_{m+1}=0 | c_m=k) = T_m(0 | k)
                    # P(rest=0 after m+1) = bwd[m+1][0]... wait, bwd is already the backward
                    
                    # Actually bwd[m+1][0] = P(c_{D-1}=0 | c_{m+1}=0)
                    # But I need P(c_{m+1}=0, c_{m+2}=0, ..., c_{D-1}=0 | c_m=k)
                    # which is T_m(0|k) · P(c_{m+2}=0,...|c_{m+1}=0)
                    
                    # Hmm, this isn't right. M=m means c_m>0 and c_{m+1}=...=c_{D-1}=0.
                    # That's stronger than just c_{m+1}=0.
                    # It means the chain reaches 0 at m+1 and STAYS at 0 forever.

                    # For carries: once c=0, does it stay 0?
                    # c_{j+1} = floor((conv + 0)/2) = floor(conv/2)
                    # If conv=0, c_{j+1}=0. But conv can be >0 with some probability.
                    # So c=0 does NOT guarantee c stays 0!
                    
                    # This means computing P(M=m) is more complex — need to track
                    # the probability that the chain stays at 0 from m+1 to D-1.
                    pass

        # This is getting complicated. Let me use a simpler approach:
        # Sample from the Markov bridge using rejection sampling.
        pr(f"\n  Sector {sec_label}: Z = {Z_sec:.8e}")

    # Simpler approach: forward simulation with bridge conditioning
    pr(f"\n  Using forward simulation with bridge reweighting...")

    np.random.seed(42)
    N_samples = 200000

    for sec_label, sec_type in [("(0,0)", "00"), ("(1,0)", "10")]:
        vals_sampled = []
        weights = []

        for _ in range(N_samples):
            carries = [0] * (D_val + 1)
            for j in range(D_val):
                n = n_terms_at(j, K_val)
                if K_val - 2 <= j <= D_val - 2:
                    if sec_type == "00":
                        n_q = max(0, n - 2)
                        conv = np.random.binomial(n_q, 0.25)
                    else:
                        n_q = max(0, n - 2)
                        conv = np.random.binomial(n_q, 0.25) + np.random.binomial(1, 0.5)
                else:
                    conv = np.random.binomial(n, 0.25)
                carries[j + 1] = (conv + carries[j]) // 2

            # Bridge condition: c_{D-1} = 0
            if carries[D_val] != 0:
                continue

            # Find M and val
            M = 0
            cm1 = 0
            for jj in range(D_val, 0, -1):
                if carries[jj] > 0:
                    M = jj
                    cm1 = carries[jj - 1]
                    break
            if M == 0:
                continue

            val = cm1 - 1
            vals_sampled.append(val)

        if vals_sampled:
            mean_val = np.mean(vals_sampled)
            pr(f"  Sector {sec_label}: {len(vals_sampled)} bridge samples, "
               f"⟨val⟩_Markov = {mean_val:+.6f}")
        else:
            pr(f"  Sector {sec_label}: no valid bridge samples!")

    # Compare with exact
    exact, _ = enumerate_dodd(K_val)
    for sec in [(0,0), (1,0)]:
        d = exact[sec]
        if d['count'] > 0:
            pr(f"  Sector {sec}: ⟨val⟩_exact = {d['sum_val']/d['count']:+.6f}, "
               f"N = {d['count']}")


# ================================================================
# PART D: Eigenvalue spectrum of bulk kernel
# ================================================================
pr("\n" + "=" * 70)
pr("PART D: Detailed eigenvalue spectrum of bulk kernel")
pr("=" * 70)

for K_val in [8, 10, 12]:
    max_c = K_val + 1
    T_bulk = build_transition(K_val, max_c)

    evals, evecs = np.linalg.eig(T_bulk)
    idx = np.argsort(-np.real(evals))
    evals = np.real(evals[idx])
    evecs = np.real(evecs[:, idx])

    pr(f"\n--- K={K_val}: Bulk kernel ({max_c+1} states) ---")
    pr(f"  {'n':>3} {'λ_n':>12} {'1-λ_n':>12} {'(1-λ_n)/gap':>14} {'ratio to 1/2':>14}")
    gap = 1 - evals[1] if len(evals) > 1 else 1
    for i in range(min(len(evals), 10)):
        g = 1 - evals[i]
        ratio_gap = g / gap if gap > 1e-15 else float('nan')
        ratio_half = g / 0.5
        pr(f"  {i:3d} {evals[i]:+12.8f} {g:12.8f} {ratio_gap:14.4f} {ratio_half:14.4f}")

    # Stationary distribution
    stat = evecs[:, 0] / np.sum(evecs[:, 0])
    pr(f"  Stationary distribution: {stat[:8]}")

    # Check if gap ratios follow n² pattern
    pr(f"\n  Gap ratios: (1-λ_n)/(1-λ_1):")
    for i in range(1, min(len(evals), 8)):
        g_i = 1 - evals[i]
        g_1 = 1 - evals[1]
        ratio = g_i / g_1 if g_1 > 1e-15 else float('nan')
        n_sq = i * i
        pr(f"    n={i}: ratio = {ratio:.4f}, n² = {n_sq}, ratio/n² = {ratio/n_sq:.4f}")


# ================================================================
# PART E: Spectral sum with exact eigenvalues
# ================================================================
pr("\n" + "=" * 70)
pr("PART E: Spectral sum with exact bulk eigenvalues")
pr("=" * 70)
pr("Test: does the ramp × bridge sum give -π with exact eigenvalues?")

for K_val in [8, 10, 12]:
    max_c = K_val + 1
    T_bulk = build_transition(K_val, max_c)

    evals_all = np.sort(np.real(np.linalg.eigvals(T_bulk)))[::-1]
    evals_sig = evals_all[1:]  # exclude λ=1 (stationary)

    pr(f"\n--- K={K_val} ---")
    pr(f"  Non-trivial eigenvalues: {evals_sig[:8]}")

    # The bridge with L positions has eigenmodes.
    # For a Markov chain with transition T on states {0,...,N},
    # the bridge from state 0 to state 0 over L steps has:
    # P(bridge) = (T^L)[0,0]
    # The spectral decomposition uses the LEFT and RIGHT eigenvectors.

    # For mode n (eigenvector v_n with eigenvalue λ_n):
    # (T^L)[0,0] = Σ_n λ_n^L · v_n[0] · w_n[0]
    # where w_n are left eigenvectors.

    # The "response" to a perturbation δT at positions j₀..j₁:
    # This is more complex — it's a first-order perturbation of the matrix product.

    # Let me instead compute the bridge response numerically by
    # building the bridge with and without perturbation.

    # For simplicity, use L = 2K-2 (= D-1)
    L = 2*K_val - 2

    # Unperturbed bridge: T^L
    T_L = np.linalg.matrix_power(T_bulk, L)
    Z_0 = T_L[0, 0]

    # Perturbed bridge: T at positions K-2..D-2 has extra term
    # Build perturbed kernel
    T_pert = build_T_mixed(max(0, K_val - 2), 1, max_c)

    # Product: T^{K-2} · T_pert^{K} · T^{0}  (roughly)
    # Actually: T_0...T_{K-3} = T^{K-2}, T_{K-2}...T_{D-2} = T_pert^{K-1},
    # but transition matrices are position-independent in the bulk.
    # This is only approximate (ignores position-dependent n_j).

    M_left = np.linalg.matrix_power(T_bulk, K_val - 2)
    M_mid = np.linalg.matrix_power(T_pert, K_val - 1)
    M_product = M_mid @ M_left

    # Start from state 0
    fwd_pert = M_product[0, :]  # wrong order, should be column
    # T is P(c'|c), so column c → column c'
    # Starting from e_0 (column): T @ e_0 = T[:,0] gives distribution at j=1
    # After j steps: T^j @ e_0

    e0 = np.zeros(max_c + 1)
    e0[0] = 1.0

    dist_unpert = np.linalg.matrix_power(T_bulk, L) @ e0
    dist_pert = np.linalg.matrix_power(T_pert, K_val - 1) @ (np.linalg.matrix_power(T_bulk, K_val - 2) @ e0)
    # This only covers positions 0..D-2, not D-1. Need one more T for the last step.
    # Actually L = D-1 = 2K-2 steps. Split: K-2 unperturbed + K perturbed.
    # K-2 + K = 2K-2 ✓

    Z_unpert = dist_unpert[0]
    Z_pert = dist_pert[0]
    
    pr(f"  Z_unpert = {Z_unpert:.8e}")
    pr(f"  Z_pert = {Z_pert:.8e}")
    pr(f"  Z_pert/Z_unpert = {Z_pert/Z_unpert:.8f}")

    # The ratio R_Markov = ⟨val⟩_pert / ⟨val⟩_unpert
    # This requires computing ⟨val⟩ for each bridge, which needs the M distribution.
    # For now, just note the bridge probability ratio.


# ================================================================
# PART F: Key question — does 1-λ_n follow n² pattern?
# ================================================================
pr("\n" + "=" * 70)
pr("PART F: Does the gap structure resolve -2π?")
pr("=" * 70)
pr("""
For the abstract bridge (E09): S(A) = Σ_n sin(nπ/2)/(1-λ_n(A))
where 1-λ_n(A) = 1-λ_n(0) + A·sin²(nπ/(2L)).

In the Markov limit, 1-λ_n(0) = 1-cos(nπ/L) ≈ (nπ/L)²/2 for small n/L.
The bridge eigenvalue is 1-cos(nπ/L), NOT a constant 1/2.

The sum becomes:
  S = Σ_n sin(nπ/2) / (1-cos(nπ/L))
  → (L²/2) Σ_n sin(nπ/2) / (nπ)²·(1/2)
  → (2L²/π²) Σ_n sin(nπ/2) / n²

And Σ_{n odd} sin(nπ/2)/n² = Σ_{k=0}^∞ (-1)^k/(2k+1)² = G (Catalan's constant!)

But we need the ramp perturbation modes too:
  S_ramp = Σ_n [cos(nπ/2)-cos(nπ)]/(nπ) · sin(nπ/2) / (1-cos(nπ/L))

For n odd: [1/(nπ)] · sin(nπ/2) / [(nπ/L)²/2] = 2L²·sin(nπ/2)/(n³π³)
Sum: 2L²/π³ · Σ_{n odd} sin(nπ/2)/n³

Σ sin(nπ/2)/n³ for n odd = 1/1 - 1/27 + 1/125 - ... = π³/32
So S_ramp = 2L²/π³ · π³/32 = L²/16

Hmm, that gives L²/16, which grows with L. The missing piece is that
the perturbation AMPLITUDE is 1/2 per position, integrated over K positions,
and normalized by the bridge probability Z.
""")

# Compute the partial sums analytically
pr("  Exact partial sums:")
S1 = sum(math.sin(n*math.pi/2)/n for n in range(1, 10001, 2))
pr(f"  Σ sin(nπ/2)/n (n odd) = {S1:.10f}, π/4 = {math.pi/4:.10f}")

S2 = sum(math.sin(n*math.pi/2)/n**2 for n in range(1, 10001, 2))
pr(f"  Σ sin(nπ/2)/n² (n odd) = {S2:.10f}, G = {0.9159655941:.10f}")

S3 = sum(math.sin(n*math.pi/2)/n**3 for n in range(1, 10001, 2))
pr(f"  Σ sin(nπ/2)/n³ (n odd) = {S3:.10f}, π³/32 = {math.pi**3/32:.10f}")

# Now with bridge eigenvalues 1-cos(nπ/L):
for L_val in [13, 17, 19, 39]:
    S_bridge = 0
    S_ramp_bridge = 0
    for n in range(1, L_val):
        s = math.sin(n*math.pi/2)
        gap_n = 1 - math.cos(n*math.pi/L_val)
        ramp_n = (math.cos(n*math.pi/2) - math.cos(n*math.pi)) / (n*math.pi)

        S_bridge += s / gap_n if gap_n > 1e-15 else 0
        S_ramp_bridge += ramp_n * s / gap_n if gap_n > 1e-15 else 0

    pr(f"\n  L={L_val}:")
    pr(f"    Σ sin(nπ/2)/(1-cos(nπ/L)) = {S_bridge:.6f}")
    pr(f"    = {S_bridge:.6f}, 2L/3 = {2*L_val/3:.6f}")
    pr(f"    Σ ramp·sin/(1-cos) = {S_ramp_bridge:.6f}")
    pr(f"    Compare: -π = {-math.pi:.6f}, L²/16 = {L_val**2/16:.6f}")
    pr(f"    Ratio S_ramp/L = {S_ramp_bridge/L_val:.6f}")
    pr(f"    Ratio S_ramp/L² = {S_ramp_bridge/L_val**2:.8f}")

pr("\n  ★★★ KEY INSIGHT ★★★")
pr("""
The bridge eigenvalues are 1-cos(nπ/L) ≈ (nπ/L)²/2, NOT 1/2.
This changes the spectral sum dramatically:
- With 1-λ = 1/2: Σ sin(nπ/2)/n → π/4  (Leibniz)
- With 1-λ = (nπ/L)²/2: Σ sin(nπ/2)·L²/(n²π²) → L²·G/π² (Catalan!)

The ramp perturbation introduces an extra 1/n factor, converting
the sum from Catalan back toward Leibniz:
  Σ [cos(nπ/2)-cos(nπ)]·sin(nπ/2) / (n·π·(nπ/L)²/2)
  = (2L²/π³) Σ sin(nπ/2)/n³ (odd n)
  = (2L²/π³)(π³/32) = L²/16

This grows as L², which means the normalization involves 1/L².
The physical sector ratio R involves σ₁₀/σ₀₀, which are SUMS 
over all D-odd pairs. The bridge probability Z ~ 1/L provides
additional normalization.
""")
