#!/usr/bin/env python3
"""
E33: Exact Q(d,K) Fractions and Pattern Search

Focus: Extract exact rational Q(d,K) for sector (1,0), study
their structure as functions of K for fixed d, and search for
Wallis-type product formulas or hypergeometric connections.

Push computation to K=16 for deeper extrapolation.
"""
import numpy as np
import math
import sys
import time
from fractions import Fraction
from collections import defaultdict

PI = math.pi


def pr(s=""):
    print(s, flush=True)


def fast_enumerate(K):
    """Numpy-vectorized exact enumeration."""
    D = 2 * K - 1
    lo = 1 << (K - 1)
    hi = 1 << K

    data = {}
    for key in ['00', '10', '01', '11']:
        data[key] = {
            'n': 0, 'vc': 0,
            'depth_cnt': defaultdict(int),
            'depth_val': defaultdict(int),
            'depth_reward': defaultdict(int),
        }

    Y_all = np.arange(lo, hi, dtype=np.int64)
    y_bits = np.zeros((len(Y_all), K), dtype=np.int8)
    for i in range(K):
        y_bits[:, i] = (Y_all >> i) & 1
    b_bits = (Y_all >> (K - 2)) & 1

    for X in range(lo, hi):
        a = (X >> (K - 2)) & 1
        xb = [(X >> i) & 1 for i in range(K)]

        P = np.int64(X) * Y_all
        dodd = (P >> (D - 1) == 1) & (P >> D == 0)
        idx = np.where(dodd)[0]
        if len(idx) == 0:
            continue

        yb = y_bits[idx]
        b_sec = b_bits[idx]
        n = len(idx)

        c = np.zeros(n, dtype=np.int32)
        carries = np.zeros((D + 1, n), dtype=np.int32)

        for j in range(D):
            conv = np.zeros(n, dtype=np.int32)
            i_lo = max(0, j - K + 1)
            i_hi = min(j, K - 1)
            for i in range(i_lo, i_hi + 1):
                conv += xb[i] * yb[:, j - i].astype(np.int32)
            c = (conv + c) >> 1
            carries[j + 1] = c

        M_vals = np.full(n, -1, dtype=np.int32)
        for j in range(D - 1, 0, -1):
            update = (carries[j] > 0) & (M_vals == -1)
            M_vals[update] = j

        val_c = np.full(n, -1, dtype=np.int64)
        reward = np.zeros(n, dtype=np.int64)
        for i in range(n):
            m = M_vals[i]
            if m > 0:
                val_c[i] = carries[m - 1, i] - 1
                reward[i] = carries[m - 1, i]

        depth = np.where(M_vals >= 0, D - 2 - M_vals, -1)

        for b_val in [0, 1]:
            key = f"{a}{b_val}"
            mask = b_sec == b_val
            if not np.any(mask):
                continue

            vc_sec = val_c[mask]
            rw_sec = reward[mask]
            dp_sec = depth[mask]

            s = data[key]
            s['n'] += int(np.sum(mask))
            s['vc'] += int(np.sum(vc_sec))

            for d_val in np.unique(dp_sec):
                dm = dp_sec == d_val
                s['depth_cnt'][int(d_val)] += int(np.sum(dm))
                s['depth_val'][int(d_val)] += int(np.sum(vc_sec[dm]))
                s['depth_reward'][int(d_val)] += int(np.sum(rw_sec[dm]))

    return D, data


pr("=" * 80)
pr("E33: Exact Q(d,K) Fractions and Pattern Search")
pr("=" * 80)
pr()

K_MAX = 16
results = []

pr(f"  {'K':>3s}  {'D':>3s}  {'Σv₁₀':>14s}  {'Σv₀₀':>14s}  "
   f"{'R(K)':>14s}  {'gap':>12s}  {'time':>8s}")

for K in range(3, K_MAX + 1):
    t0 = time.time()
    D, data = fast_enumerate(K)
    dt = time.time() - t0

    s10, s00 = data['10'], data['00']
    R_frac = Fraction(s10['vc'], s00['vc']) if s00['vc'] != 0 else None
    R_float = float(R_frac) if R_frac else float('nan')
    gap = R_float + PI

    results.append({
        'K': K, 'D': D, 'data': data,
        'R_frac': R_frac, 'R_float': R_float, 'gap': gap, 'dt': dt,
    })

    pr(f"  {K:3d}  {D:3d}  {s10['vc']:14d}  {s00['vc']:14d}  "
       f"{R_float:+14.8f}  {gap:+12.8f}  {dt:7.1f}s")

pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 1: Exact Q(d,K) Fractions — Sector (1,0)")
pr("=" * 80)
pr()

Q_exact = {}
for res in results:
    K = res['K']
    s = res['data']['10']
    alive = s['n']
    for d in range(-1, res['D']):
        cnt = s['depth_cnt'].get(d, 0)
        alive_new = alive - cnt
        if alive > 0:
            Q_exact[(d, K)] = (Fraction(alive_new, alive), alive, alive_new)
        alive = alive_new

for d in range(1, 10):
    pr(f"  d={d}:")
    for K in range(max(5, d + 3), K_MAX + 1):
        entry = Q_exact.get((d, K))
        if entry is not None:
            q, alive, alive_new = entry
            if q.denominator > 1 and float(q) > 0 and float(q) < 1:
                pr(f"    K={K:2d}: Q = {q.numerator:>14d} / {q.denominator:<14d}  "
                   f"= {float(q):.12f}  (alive={alive})")
    pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 2: Numerator and Denominator Factorizations")
pr("=" * 80)
pr()

def prime_factors(n):
    if n <= 1:
        return {}
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


for d in range(1, 5):
    pr(f"  d={d}:")
    for K in range(5, K_MAX + 1):
        entry = Q_exact.get((d, K))
        if entry is not None:
            q, alive, alive_new = entry
            if q.denominator > 1 and float(q) > 0 and float(q) < 1:
                nf = prime_factors(abs(q.numerator))
                df = prime_factors(abs(q.denominator))
                nf_str = "·".join(f"{p}^{e}" if e > 1 else str(p)
                                  for p, e in sorted(nf.items()))
                df_str = "·".join(f"{p}^{e}" if e > 1 else str(p)
                                  for p, e in sorted(df.items()))
                pr(f"    K={K:2d}: {q.numerator:>14d} / {q.denominator:<14d}  "
                   f"num={nf_str:<30s} den={df_str}")
    pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 3: Q(d,K) - 1/2 Structure")
pr("=" * 80)
pr()

pr("  If Q∞(d) → 1/2, then Q(d,K) = 1/2 + δ(d,K).")
pr("  Studying δ(d,K) = Q(d,K) - 1/2 as exact fractions:")
pr()

for d in range(1, 8):
    pr(f"  d={d}:")
    for K in range(max(5, d + 3), K_MAX + 1):
        entry = Q_exact.get((d, K))
        if entry is not None:
            q = entry[0]
            if float(q) > 0 and float(q) < 1:
                delta = q - Fraction(1, 2)
                pr(f"    K={K:2d}: δ = {delta.numerator:>14d} / {delta.denominator:<14d}  "
                   f"= {float(delta):+.12f}  "
                   f"K·δ = {float(K * delta):+.6f}  "
                   f"2^K·δ = {float(2**K * delta):+.6f}")
    pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 4: Aitken Δ² Extrapolation for Q∞(d)")
pr("=" * 80)
pr()

pr("  Using last 3 K values with Aitken acceleration:")
pr(f"  {'d':>3s}  {'Q(d,K-2)':>10s}  {'Q(d,K-1)':>10s}  {'Q(d,K)':>10s}  "
   f"{'Aitken Q∞':>12s}  {'Q∞-1/2':>12s}")

Q_inf = {}
for d in range(0, 15):
    qs = []
    for K in range(5, K_MAX + 1):
        entry = Q_exact.get((d, K))
        if entry is not None:
            qs.append((K, float(entry[0])))

    if len(qs) >= 3:
        _, q0 = qs[-3]
        _, q1 = qs[-2]
        _, q2 = qs[-1]

        denom = q2 - 2 * q1 + q0
        if abs(denom) > 1e-15:
            aitken = q2 - (q2 - q1) ** 2 / denom
        else:
            aitken = q2

        Q_inf[d] = aitken
        excess = aitken - 0.5

        if 0 < aitken < 1.5:
            pr(f"  {d:3d}  {q0:10.6f}  {q1:10.6f}  {q2:10.6f}  "
               f"{aitken:12.8f}  {excess:+12.8f}")

pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 5: The Per-Depth Correction Series (sector 10)")
pr("=" * 80)
pr()

pr("  At K=K_MAX, the correction series Δ⟨val⟩₁₀ = Σ P·r:")
pr(f"  {'d':>3s}  {'P(d)·r(d)':>14s}  {'ratio d-1/d':>14s}  "
   f"{'cum':>14s}  {'1/(2^d)·t_d':>14s}")

res = results[-1]
s10 = res['data']['10']
N10 = s10['n']
cum = 0.0
prev_t = None
for d in sorted(s10['depth_cnt'].keys()):
    if d < 0:
        continue
    rwd = s10['depth_reward'].get(d, 0)
    term = rwd / N10
    cum += term

    ratio_s = ""
    if prev_t is not None and abs(term) > 1e-15:
        ratio_s = f"{prev_t / term:14.8f}"

    scaled = term * 2 ** d if abs(term) > 1e-15 else 0.0

    if abs(term) > 1e-14:
        pr(f"  {d:3d}  {term:+14.10f}  {ratio_s:>14s}  "
           f"{cum:+14.10f}  {scaled:+14.8f}")
    prev_t = term if abs(term) > 1e-15 else prev_t

pr()
pr(f"  Δ⟨val⟩₁₀(K={K_MAX}) = {cum:.12f}")
pr()

# Extrapolate Δ⟨val⟩₁₀ to K→∞
deltas = []
for i in range(len(results)):
    K = results[i]['K']
    s = results[i]['data']['10']
    if s['n'] > 0:
        total_rwd = sum(s['depth_reward'].get(d, 0) for d in s['depth_reward'] if d >= 0)
        deltas.append((K, total_rwd / s['n']))

if len(deltas) >= 4:
    pr("  Convergence of Δ⟨val⟩₁₀(K):")
    pr(f"  {'K':>3s}  {'Δ⟨val⟩₁₀':>14s}  {'delta':>12s}  {'ratio':>10s}")
    for i, (K, v) in enumerate(deltas):
        delta = ""
        ratio = ""
        if i > 0:
            d = v - deltas[i - 1][1]
            delta = f"{d:+.8f}"
            if i > 1:
                d_prev = deltas[i - 1][1] - deltas[i - 2][1]
                if abs(d) > 1e-12:
                    ratio = f"{d_prev / d:.4f}"
        pr(f"  {K:3d}  {v:+14.10f}  {delta:>12s}  {ratio:>10s}")
    pr()

    d1 = deltas[-1][1] - deltas[-2][1]
    d2 = deltas[-2][1] - deltas[-3][1]
    if abs(d1) > 1e-12:
        rho = d1 / d2
        limit = deltas[-1][1] + d1 * rho / (1 - rho)
        pr(f"  Extrapolated Δ⟨val⟩₁₀∞ ≈ {limit:.10f}  (ρ = {rho:.4f})")
    pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 6: Sector (0,0) — ⟨val⟩₀₀ Convergence")
pr("=" * 80)
pr()

vals00 = []
for res in results:
    K = res['K']
    s = res['data']['00']
    if s['n'] > 0:
        vals00.append((K, float(Fraction(s['vc'], s['n']))))

pr(f"  {'K':>3s}  {'⟨val⟩₀₀':>14s}  {'delta':>12s}  {'ratio':>10s}")
for i, (K, v) in enumerate(vals00):
    delta = ""
    ratio = ""
    if i > 0:
        d = v - vals00[i - 1][1]
        delta = f"{d:+.8f}"
        if i > 1:
            d_prev = vals00[i - 1][1] - vals00[i - 2][1]
            if abs(d) > 1e-12:
                ratio = f"{d_prev / d:.4f}"
    pr(f"  {K:3d}  {v:+14.10f}  {delta:>12s}  {ratio:>10s}")

if len(vals00) >= 4:
    d1 = vals00[-1][1] - vals00[-2][1]
    d2 = vals00[-2][1] - vals00[-3][1]
    if abs(d1) > 1e-12:
        rho = d1 / d2
        limit = vals00[-1][1] + d1 * rho / (1 - rho)
        pr(f"  Extrapolated ⟨val⟩₀₀∞ ≈ {limit:.10f}  (ρ = {rho:.4f})")
pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 7: N₁₀/N₀₀ Convergence and R∞ Estimate")
pr("=" * 80)
pr()

nratios = []
for res in results:
    n10 = res['data']['10']['n']
    n00 = res['data']['00']['n']
    if n00 > 0:
        nratios.append((res['K'], n10 / n00))

pr(f"  {'K':>3s}  {'N₁₀/N₀₀':>14s}  {'delta':>12s}")
for i, (K, nr) in enumerate(nratios):
    delta = ""
    if i > 0:
        delta = f"{nr - nratios[i - 1][1]:+.8f}"
    pr(f"  {K:3d}  {nr:14.10f}  {delta:>12s}")

L2 = math.log(2)
L3 = math.log(3)
N10_cont = 2 * math.log(4 / 3) - 0.5
N00_cont = 2 * math.log(9 / 8)
nr_cont = N10_cont / N00_cont
pr(f"  Continuum limit: N₁₀/N₀₀ = {nr_cont:.10f}")
pr()

# Estimate R∞ from extrapolated components
if len(deltas) >= 4 and len(vals00) >= 4:
    d1_v10 = deltas[-1][1] - deltas[-2][1]
    d2_v10 = deltas[-2][1] - deltas[-3][1]
    d1_v00 = vals00[-1][1] - vals00[-2][1]
    d2_v00 = vals00[-2][1] - vals00[-3][1]

    if abs(d1_v10) > 1e-12 and abs(d1_v00) > 1e-12:
        rho10 = d1_v10 / d2_v10
        rho00 = d1_v00 / d2_v00
        v10_inf = deltas[-1][1] + d1_v10 * rho10 / (1 - rho10) - 1
        v00_inf = vals00[-1][1] + d1_v00 * rho00 / (1 - rho00)

        R_est = nr_cont * v10_inf / v00_inf
        pr(f"  Estimated limits:")
        pr(f"    ⟨val⟩₁₀∞ = -1 + {deltas[-1][1] + d1_v10 * rho10 / (1 - rho10):.10f}"
           f" = {v10_inf:.10f}")
        pr(f"    ⟨val⟩₀₀∞ = {v00_inf:.10f}")
        pr(f"    R∞ = {nr_cont:.6f} · {v10_inf:.6f} / {v00_inf:.6f} = {R_est:.10f}")
        pr(f"    R∞ + π = {R_est + PI:.10f}")
        pr(f"    -π = {-PI:.10f}")
        pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 8: Gap Analysis with K=15,16 Data")
pr("=" * 80)
pr()

gaps = [(r['K'], r['gap']) for r in results if r['K'] >= 7]
pr(f"  {'K':>3s}  {'gap':>14s}  {'ratio':>10s}  {'ln(gap)':>12s}  "
   f"{'Δln':>10s}  {'Δ²ln':>10s}")

for i, (K, g) in enumerate(gaps):
    ratio_s = ""
    ln_g = math.log(g) if g > 0 else float('nan')
    dln = ""
    d2ln = ""

    if i > 0 and gaps[i - 1][1] > 0:
        ratio_s = f"{gaps[i - 1][1] / g:.4f}"
    if i > 0:
        dln = f"{ln_g - math.log(gaps[i - 1][1]):+.6f}" if gaps[i - 1][1] > 0 else ""
    if i > 1 and gaps[i - 2][1] > 0 and gaps[i - 1][1] > 0:
        ln_prev = math.log(gaps[i - 1][1])
        ln_pprev = math.log(gaps[i - 2][1])
        d2 = ln_g - 2 * ln_prev + ln_pprev
        d2ln = f"{d2:+.6f}"

    pr(f"  {K:3d}  {g:+14.8f}  {ratio_s:>10s}  {ln_g:12.6f}  "
       f"{dln:>10s}  {d2ln:>10s}")

pr()

# Gaussian model fit with new data
lngaps = [(K, math.log(g)) for K, g in gaps if g > 0]
if len(lngaps) >= 5:
    Ks = np.array([x[0] for x in lngaps])
    lgs = np.array([x[1] for x in lngaps])
    A = np.column_stack([np.ones_like(Ks), Ks, Ks ** 2])
    coeffs = np.linalg.lstsq(A, lgs, rcond=None)[0]
    a, b, c = coeffs

    pred = A @ coeffs
    rmse = np.sqrt(np.mean((lgs - pred) ** 2))

    pr(f"  Gaussian model: ln(gap) = {a:.6f} + {b:.6f}·K + {c:.6f}·K²")
    pr(f"  RMSE = {rmse:.8f}")
    pr()

    pr("  Gaussian model predictions:")
    for K_pred in [17, 18, 20, 25, 30]:
        g_pred = math.exp(a + b * K_pred + c * K_pred ** 2)
        pr(f"    K={K_pred}: gap ≈ {g_pred:.6e}")
    pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 9: The Scaled Series 2^d · P(d) · r(d) ")
pr("=" * 80)
pr()

pr("  If the series is ~ Σ (1/2)^d · f(d), the scaled terms f(d) = 2^d·P·r")
pr("  should converge to a function whose series gives π-related values.")
pr()

for res in results[-3:]:
    K = res['K']
    s10 = res['data']['10']
    N = s10['n']
    pr(f"  K={K}:")
    pr(f"  {'d':>3s}  {'2^d·P·r':>14s}  {'Δ':>12s}")

    prev_f = None
    for d in sorted(s10['depth_cnt'].keys()):
        if d < 1:
            continue
        rwd = s10['depth_reward'].get(d, 0)
        term = rwd / N
        scaled = term * 2 ** d

        delta = ""
        if prev_f is not None:
            delta = f"{scaled - prev_f:+.8f}"

        if abs(scaled) > 1e-10:
            pr(f"  {d:3d}  {scaled:+14.8f}  {delta:>12s}")
        prev_f = scaled
    pr()

# ═══════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SUMMARY")
pr("=" * 80)
pr()

R0 = (0.5 - 4 * L2 + 2 * L3) / (1 - 3 * L2 + L3)
pr(f"  R₀ = {R0:+.15f}")
pr(f"  -π = {-PI:+.15f}")
pr(f"  R({K_MAX}) = {results[-1]['R_float']:+.15f}")
pr(f"  gap({K_MAX}) = R({K_MAX})+π = {results[-1]['gap']:+.15f}")
pr()

if len(lngaps) >= 5:
    pr(f"  Gaussian decay: c = {-c:.6f} → gap ~ exp(-{-c:.4f}·K²)")
    pr(f"  Gap halves every ΔK ≈ {math.sqrt(math.log(2) / (-c)):.1f} steps")
pr()

pr("  Q∞(d) for sector (1,0):")
for d in range(1, 12):
    if d in Q_inf and 0 < Q_inf[d] < 1:
        pr(f"    d={d:2d}: Q∞ = {Q_inf[d]:.10f}  "
           f"(excess = {Q_inf[d] - 0.5:+.10f})")
pr()
pr("=" * 80)
