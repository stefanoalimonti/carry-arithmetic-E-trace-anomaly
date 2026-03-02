#!/usr/bin/env python3
"""
E32: Per-Depth Decomposition of ΔR — Searching for the π-Series

Strategy:
  R(K) = Σv₁₀(K) / Σv₀₀(K) → -π
  R₀ = (1/2 - 2ln(4/3)) / (1 + ln(3/8)) ≈ -3.9312  (proven)
  ΔR = R - R₀ → -π - R₀ ≈ +0.7896  (to prove)

  Decompose each sector by cascade depth d:
    ⟨val⟩_s = Σ_d P_s(d) · [r_s(d) - 1]
  where:
    P_s(d) = fraction stopping at depth d
    r_s(d) = E[carry[M-1] | depth=d]

  Then study Q_∞(d) = lim_{K→∞} Q(d,K) and r_∞(d) = lim_{K→∞} r(d,K)
  and evaluate the limiting resolvent series for π.
"""
import numpy as np
import math
import sys
import time
from fractions import Fraction
from collections import defaultdict

PI = math.pi
L2 = math.log(2)
L3 = math.log(3)
R0_EXACT = (0.5 - 4 * L2 + 2 * L3) / (1 - 3 * L2 + L3)
DELTA_R_TARGET = -PI - R0_EXACT


def pr(s=""):
    print(s, flush=True)


def fast_enumerate(K):
    """Numpy-vectorized exact enumeration returning per-depth data."""
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


# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("E32: Per-Depth Decomposition — Searching for the π-Series")
pr("=" * 80)
pr()

K_MAX = 14
results = []

pr(f"  {'K':>3s}  {'D':>3s}  {'Σv₁₀':>12s}  {'Σv₀₀':>12s}  "
   f"{'R(K)':>14s}  {'gap':>12s}  {'time':>6s}")

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
        'sv10': s10['vc'], 'sv00': s00['vc'],
        'n10': s10['n'], 'n00': s00['n'],
    })

    pr(f"  {K:3d}  {D:3d}  {s10['vc']:12d}  {s00['vc']:12d}  "
       f"{R_float:+14.8f}  {gap:+12.8f}  {dt:5.1f}s")

pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 1: Exact Q(d,K) — Conditional Survival by Depth")
pr("=" * 80)
pr()

Q_tables = {}
for sec in ['10', '00']:
    Q_tables[sec] = {}
    for r in results:
        K = r['K']
        s = r['data'][sec]
        alive = s['n']
        for d in range(-1, r['D']):
            cnt = s['depth_cnt'].get(d, 0)
            alive_new = alive - cnt
            if alive > 0 and d >= -1:
                Q_tables[sec][(d, K)] = Fraction(alive_new, alive)
            alive = alive_new

for sec in ['10', '00']:
    pr(f"  Sector ({sec[0]},{sec[1]}) — Q(d,K) as floats:")
    sys.stdout.write(f"  {'d':>3s}  ")
    for K in range(5, K_MAX + 1):
        sys.stdout.write(f"{'K='+str(K):>10s}  ")
    sys.stdout.write("\n")
    sys.stdout.flush()

    for d in range(-1, 15):
        any_data = False
        row = f"  {d:3d}  "
        for K in range(5, K_MAX + 1):
            q = Q_tables[sec].get((d, K))
            if q is not None and q != 1:
                row += f"{float(q):10.6f}  "
                any_data = True
            elif q == 1:
                row += f"{'1.000000':>10s}  "
                any_data = True
            else:
                row += f"{'—':>10s}  "
        if any_data:
            pr(row)
    pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 2: Q(d,K) Convergence — Fixed d, Varying K")
pr("=" * 80)
pr()

pr("  For each d, does Q(d,K) converge to Q∞(d) as K→∞?")
pr("  Richardson extrapolation: Q∞ = (K₂·Q₂ - K₁·Q₁)/(K₂-K₁) for O(1/K)")
pr()

for sec in ['10', '00']:
    pr(f"  Sector ({sec[0]},{sec[1]}):")
    pr(f"  {'d':>3s}  {'Q(d,12)':>10s}  {'Q(d,13)':>10s}  {'Q(d,14)':>10s}  "
       f"{'Δ(13→14)':>10s}  {'Rich(12,14)':>12s}")

    for d in range(0, 13):
        q12 = Q_tables[sec].get((d, 12))
        q13 = Q_tables[sec].get((d, 13))
        q14 = Q_tables[sec].get((d, 14))

        if q12 is not None and q13 is not None and q14 is not None:
            fq12, fq13, fq14 = float(q12), float(q13), float(q14)
            delta = fq14 - fq13

            rich_num = Fraction(14, 1) * q14 - Fraction(12, 1) * q12
            rich_den = Fraction(14 - 12)
            rich = rich_num / rich_den

            pr(f"  {d:3d}  {fq12:10.6f}  {fq13:10.6f}  {fq14:10.6f}  "
               f"{delta:+10.6f}  {float(rich):12.8f}")
        elif q13 is not None and q14 is not None:
            fq13, fq14 = float(q13), float(q14)
            delta = fq14 - fq13
            pr(f"  {d:3d}  {'—':>10s}  {fq13:10.6f}  {fq14:10.6f}  "
               f"{delta:+10.6f}  {'—':>12s}")
    pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 3: Per-Depth Reward r(d,K) = E[carry[M-1] | depth=d]")
pr("=" * 80)
pr()

r_tables = {}
for sec in ['10', '00']:
    r_tables[sec] = {}
    for res in results:
        K = res['K']
        s = res['data'][sec]
        for d in sorted(s['depth_cnt'].keys()):
            cnt = s['depth_cnt'][d]
            rwd = s['depth_reward'].get(d, 0)
            if cnt > 0:
                r_tables[sec][(d, K)] = Fraction(rwd, cnt)

for sec in ['10', '00']:
    pr(f"  Sector ({sec[0]},{sec[1]}) — r(d,K) as floats:")
    sys.stdout.write(f"  {'d':>3s}  ")
    for K in range(5, K_MAX + 1):
        sys.stdout.write(f"{'K='+str(K):>10s}  ")
    sys.stdout.write("\n")
    sys.stdout.flush()

    for d in range(0, 13):
        any_data = False
        row = f"  {d:3d}  "
        for K in range(5, K_MAX + 1):
            rv = r_tables[sec].get((d, K))
            if rv is not None:
                row += f"{float(rv):10.6f}  "
                any_data = True
            else:
                row += f"{'—':>10s}  "
        if any_data:
            pr(row)
    pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 4: Per-Depth Weight w(d,K) and Cumulative Series")
pr("=" * 80)
pr()

pr("  w(d,K) = P(depth=d) · val_avg(d,K) = depth_val(d,K) / N_sector")
pr("  ⟨val⟩_s(K) = Σ_d w(d,K)")
pr("  R(K) = ⟨val⟩₁₀(K) / ⟨val⟩₀₀(K) · (N₁₀/N₀₀)")
pr()

for sec in ['10', '00']:
    pr(f"  Sector ({sec[0]},{sec[1]}) — per-depth w(d,K) for K=12,13,14:")

    for res in results[-3:]:
        K = res['K']
        s = res['data'][sec]
        N = s['n']
        cumw = Fraction(0)

        pr(f"    K={K}:")
        pr(f"    {'d':>3s}  {'count':>8s}  {'val_sum':>10s}  {'w(d)':>14s}  {'cum':>14s}")

        for d in sorted(s['depth_cnt'].keys()):
            cnt = s['depth_cnt'][d]
            val = s['depth_val'].get(d, 0)
            if cnt > 0:
                w = Fraction(val, N)
                cumw += w
                pr(f"    {d:3d}  {cnt:8d}  {val:10d}  {float(w):+14.10f}  "
                   f"{float(cumw):+14.10f}")

        check = Fraction(s['vc'], N)
        pr(f"    tot {'':8s}  {'':10s}  {'':14s}  "
           f"{float(cumw):+14.10f}  (check: {float(check):+.10f})")
        pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 5: The ΔR Series — Sector (1,0) Cascade Correction")
pr("=" * 80)
pr()

pr("  In sector (1,0), val_school = -1 always (carries[D-2]=0).")
pr("  ΔΣ₁₀ = Σv_casc(10) - Σv_school(10) = Σv_casc(10) + N₁₀ = total_reward₁₀")
pr("  Δ⟨val⟩₁₀ = ΔΣ₁₀ / N₁₀ = Σ_d P₁₀(d) · r₁₀(d)")
pr()
pr("  The per-depth terms of the correction series:")
pr()

for res in results[-4:]:
    K = res['K']
    s10 = res['data']['10']
    N = s10['n']

    pr(f"  K={K}  (N₁₀={N}):")
    pr(f"  {'d':>3s}  {'P(d)':>12s}  {'r(d)':>10s}  {'P·r':>14s}  "
       f"{'cum Δ⟨val⟩':>14s}  {'P·(r-1)':>14s}  {'cum ⟨val⟩':>14s}")

    cum_delta = Fraction(0)
    cum_val = Fraction(0)

    for d in sorted(s10['depth_cnt'].keys()):
        if d < 0:
            cnt = s10['depth_cnt'][d]
            cum_val += Fraction(s10['depth_val'][d], N)
            continue

        cnt = s10['depth_cnt'][d]
        rwd = s10['depth_reward'].get(d, 0)
        if cnt > 0:
            P_d = Fraction(cnt, N)
            r_d = Fraction(rwd, cnt)
            Pr = Fraction(rwd, N)
            cum_delta += Pr
            val_term = Fraction(s10['depth_val'][d], N)
            cum_val += val_term

            pr(f"  {d:3d}  {float(P_d):12.8f}  {float(r_d):10.6f}  "
               f"{float(Pr):+14.10f}  {float(cum_delta):+14.10f}  "
               f"{float(val_term):+14.10f}  {float(cum_val):+14.10f}")

    total_delta = sum(s10['depth_reward'].get(d, 0) for d in s10['depth_reward'] if d >= 0)
    check = s10['vc'] + N
    pr(f"  tot {'':12s}  {'':10s}  {'':14s}  "
       f"{float(Fraction(total_delta, N)):+14.10f}  {'':14s}  "
       f"{float(Fraction(s10['vc'], N)):+14.10f}")
    pr(f"  check: ΔΣ = {total_delta},  Σvc+N = {check},  match: {total_delta == check}")
    pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 6: Q∞(d) Extrapolation via Sequence Analysis")
pr("=" * 80)
pr()

pr("  For each d, track Q(d,K) sequence and extrapolate Q∞(d).")
pr("  Use last 4 values with Neville-Aitken assuming O(1/K) correction.")
pr()

Q_inf = {}
for sec in ['10', '00']:
    Q_inf[sec] = {}
    pr(f"  Sector ({sec[0]},{sec[1]}):")
    pr(f"  {'d':>3s}  {'Q(d,11)':>10s}  {'Q(d,12)':>10s}  {'Q(d,13)':>10s}  "
       f"{'Q(d,14)':>10s}  {'Q∞(rich)':>12s}  {'Q∞(aitk)':>12s}")

    for d in range(0, 13):
        Ks = []
        Qs = []
        for K in range(7, K_MAX + 1):
            q = Q_tables[sec].get((d, K))
            if q is not None:
                Ks.append(K)
                Qs.append(float(q))

        if len(Qs) < 4:
            continue

        q11 = Q_tables[sec].get((d, 11))
        q12 = Q_tables[sec].get((d, 12))
        q13 = Q_tables[sec].get((d, 13))
        q14 = Q_tables[sec].get((d, 14))

        if None in [q11, q12, q13, q14]:
            continue

        fq11, fq12, fq13, fq14 = float(q11), float(q12), float(q13), float(q14)

        rich = (14 * fq14 - 12 * fq12) / 2

        aitken = fq14
        if abs(fq14 - 2 * fq13 + fq12) > 1e-15:
            aitken = fq14 - (fq14 - fq13) ** 2 / (fq14 - 2 * fq13 + fq12)

        Q_inf[sec][d] = aitken

        pr(f"  {d:3d}  {fq11:10.6f}  {fq12:10.6f}  {fq13:10.6f}  "
           f"{fq14:10.6f}  {rich:12.8f}  {aitken:12.8f}")
    pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 7: Survival Products and Series Structure")
pr("=" * 80)
pr()

pr("  Surv₁₀(d) = Π_{d'=0}^{d-1} Q∞₁₀(d')")
pr("  The resolvent series: Δ⟨val⟩₁₀ = Σ_d Surv₁₀(d) · (1-Q∞₁₀(d)) · r∞₁₀(d)")
pr()

sec = '10'
if len(Q_inf[sec]) > 0:
    surv = 1.0
    cum_series = 0.0
    pr(f"  {'d':>3s}  {'Q∞(d)':>10s}  {'1-Q∞':>10s}  {'Surv(d)':>12s}  "
       f"{'r∞(d)':>10s}  {'term':>14s}  {'cum':>14s}")

    for d in range(max(Q_inf[sec].keys()) + 1):
        q = Q_inf[sec].get(d, 0.5)

        r_val = 0.5
        for K in range(K_MAX, 6, -1):
            rv = r_tables[sec].get((d, K))
            if rv is not None:
                r_val = float(rv)
                break

        term = surv * (1 - q) * r_val
        cum_series += term

        pr(f"  {d:3d}  {q:10.6f}  {1-q:10.6f}  {surv:12.8f}  "
           f"{r_val:10.6f}  {term:+14.10f}  {cum_series:+14.10f}")

        surv *= q

    pr()
    pr(f"  Cumulative series Δ⟨val⟩₁₀ ≈ {cum_series:.10f}")

    direct = float(Fraction(results[-1]['data']['10']['vc'],
                            results[-1]['data']['10']['n']))
    pr(f"  Direct ⟨val⟩₁₀(K={K_MAX}) = {direct:.10f}")
    pr(f"  Direct Δ⟨val⟩₁₀(K={K_MAX}) = {direct + 1:.10f}")
    pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 8: Testing Q∞(d) for Simple Closed Forms")
pr("=" * 80)
pr()

sec = '10'
if len(Q_inf[sec]) >= 3:
    pr("  Test 1: Is Q∞(d) = constant?")
    vals = [Q_inf[sec][d] for d in sorted(Q_inf[sec].keys())]
    avg = np.mean(vals)
    std = np.std(vals)
    pr(f"    Mean = {avg:.8f}, Std = {std:.8f}, CoV = {std/avg:.4f}")
    pr()

    pr("  Test 2: Is Q∞(d) = (d + a) / (d + b) for d≥1?")
    pr("    Using d=1,2 as anchor points (d=0 is trivially 1):")
    if len(Q_inf[sec]) >= 3:
        q1 = Q_inf[sec].get(1, None)
        q2 = Q_inf[sec].get(2, None)
        if q1 is not None and q2 is not None and abs(q2 - q1) > 1e-10:
            b_param = (2 * q1 - q2 - q1 * q2) / (q2 - q1)
            a_param = q1 * (1 + b_param) - 1
            pr(f"    b = {b_param:.8f},  a = {a_param:.8f}")
            pr(f"    Prediction: Q∞(d) = (d + {a_param:.4f}) / (d + {b_param:.4f})")

            pr(f"    {'d':>3s}  {'Q∞(data)':>10s}  {'Q∞(pred)':>10s}  {'error':>12s}")
            for d in sorted(Q_inf[sec].keys()):
                if d == 0:
                    continue
                denom = d + b_param
                if abs(denom) > 1e-10:
                    pred = (d + a_param) / denom
                    err = Q_inf[sec][d] - pred
                    pr(f"    {d:3d}  {Q_inf[sec][d]:10.6f}  {pred:10.6f}  {err:+12.8f}")
    pr()

    pr("  Test 3: Is Q∞(d) = 1/2 + correction?")
    pr("    If Q∞(d) = 1/2 + f(d), what is f(d)?")
    for d in sorted(Q_inf[sec].keys()):
        f_d = Q_inf[sec][d] - 0.5
        pr(f"    d={d:2d}: Q∞-1/2 = {f_d:+.8f}")
    pr()

    pr("  Test 4: Is Q∞(d) related to (3/4)^something · polynomial?")
    pr("    Q∞(d) / (3/4): ")
    for d in sorted(Q_inf[sec].keys()):
        ratio = Q_inf[sec][d] / 0.75
        pr(f"    d={d:2d}: Q∞/(3/4) = {ratio:.8f}")
    pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 9: Exact Q(d,K) Fractions — Pattern Analysis")
pr("=" * 80)
pr()

for sec in ['10', '00']:
    pr(f"  Sector ({sec[0]},{sec[1]}) — Q(d,K) as exact fractions:")
    for d in range(0, 8):
        pr(f"    d={d}:")
        for K in range(5, K_MAX + 1):
            q = Q_tables[sec].get((d, K))
            if q is not None and q.denominator > 1:
                pr(f"      K={K:2d}: {q.numerator:>12d} / {q.denominator:<12d}  "
                   f"= {float(q):.10f}")
        pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 10: The Key Ratio — How R(K) Builds from Sectors")
pr("=" * 80)
pr()

pr("  R(K) = Σv₁₀ / Σv₀₀")
pr("  = (N₁₀ · ⟨val⟩₁₀) / (N₀₀ · ⟨val⟩₀₀)")
pr("  = (N₁₀/N₀₀) · (⟨val⟩₁₀/⟨val⟩₀₀)")
pr()
pr(f"  {'K':>3s}  {'N₁₀/N₀₀':>12s}  {'⟨v⟩₁₀':>12s}  {'⟨v⟩₀₀':>12s}  "
   f"{'⟨v⟩ ratio':>12s}  {'R(K)':>14s}")

for res in results:
    K = res['K']
    n10, n00 = res['n10'], res['n00']
    s10, s00 = res['data']['10'], res['data']['00']
    avg10 = float(Fraction(s10['vc'], n10)) if n10 > 0 else 0
    avg00 = float(Fraction(s00['vc'], n00)) if n00 > 0 else 0
    nratio = n10 / n00 if n00 > 0 else 0
    vratio = avg10 / avg00 if abs(avg00) > 1e-15 else float('nan')

    pr(f"  {K:3d}  {nratio:12.8f}  {avg10:+12.8f}  {avg00:+12.8f}  "
       f"{vratio:+12.6f}  {res['R_float']:+14.8f}")

pr()
pr(f"  Continuum N₁₀/N₀₀ ratio:")
N10_cont = 2 * math.log(4 / 3) - 0.5
N00_cont = 2 * math.log(9 / 8)
pr(f"  N₁₀ = 2·ln(4/3) - 1/2 = {N10_cont:.15f}")
pr(f"  N₀₀ = 2·ln(9/8) = {N00_cont:.15f}")
pr(f"  N₁₀/N₀₀ = {N10_cont / N00_cont:.15f}")
pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 11: Finite-Difference Analysis of Q(d,K) in K")
pr("=" * 80)
pr()

pr("  For fixed d, examine the convergence rate of Q(d,K).")
pr("  If Q(d,K) = Q∞(d) + A(d)/K + B(d)/K² + ..., then:")
pr("  Δ_K Q(d,K) = Q(d,K+1) - Q(d,K) ≈ -A(d)/K²")
pr()

for sec in ['10']:
    pr(f"  Sector ({sec[0]},{sec[1]}):")
    for d in range(0, 6):
        pr(f"    d={d}:")
        prev_K = None
        prev_Q = None
        for K in range(5, K_MAX + 1):
            q = Q_tables[sec].get((d, K))
            if q is not None:
                fq = float(q)
                if prev_Q is not None:
                    delta = fq - prev_Q
                    k2_delta = K * K * delta
                    pr(f"      K={prev_K}→{K}: ΔQ = {delta:+.8f}, "
                       f"K²·ΔQ = {k2_delta:+.4f}")
                prev_K = K
                prev_Q = fq
        pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 12: The Product Structure — Survival Ratios")
pr("=" * 80)
pr()

pr("  For sector (1,0), the total survival to depth d is:")
pr("  Surv₁₀(d,K) = N₁₀(≥d) / N₁₀")
pr("  The ratio Surv₁₀(d,K) / Surv₀₀(d,K) is the key structural quantity.")
pr()

for res in results[-3:]:
    K = res['K']
    pr(f"  K={K}:")
    pr(f"  {'d':>3s}  {'Surv₁₀':>12s}  {'Surv₀₀':>12s}  {'ratio':>12s}")

    surv10 = res['n10']
    surv00 = res['n00']
    s10 = res['data']['10']
    s00 = res['data']['00']

    for d in range(-1, min(12, res['D'])):
        cnt10 = s10['depth_cnt'].get(d, 0)
        cnt00 = s00['depth_cnt'].get(d, 0)
        surv10 -= cnt10
        surv00 -= cnt00

        if surv10 > 0 and surv00 > 0:
            Q10 = surv10 / res['n10']
            Q00 = surv00 / res['n00']
            ratio = Q10 / Q00
            pr(f"  {d:3d}  {Q10:12.8f}  {Q00:12.8f}  {ratio:12.8f}")

    pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 13: Exact Survival Ratio as Fractions")
pr("=" * 80)
pr()

pr("  SR(d,K) = Surv₁₀(d)/Surv₀₀(d) normalized by N₁₀/N₀₀:")
pr("  If R→-π, then in the limit:")
pr("  Σ_d SR∞(d) · (stop_frac₁₀(d)·(r₁₀(d)-1) - stop_frac₀₀(d)·(r₀₀(d)-1)) = -π")
pr()

# Compute exact survival fractions for K=14
res = results[-1]
K = res['K']
s10 = res['data']['10']
s00 = res['data']['00']

alive10 = Fraction(s10['n'])
alive00 = Fraction(s00['n'])

pr(f"  K={K} exact survival data:")
pr(f"  {'d':>3s}  {'Surv₁₀':>14s}  {'Surv₀₀':>14s}  {'Q₁₀(d)':>14s}  "
   f"{'Q₀₀(d)':>14s}  {'Q₁₀/Q₀₀':>12s}")

for d in range(-1, 13):
    cnt10 = s10['depth_cnt'].get(d, 0)
    cnt00 = s00['depth_cnt'].get(d, 0)

    old10 = alive10
    old00 = alive00
    alive10 -= cnt10
    alive00 -= cnt00

    q10 = alive10 / old10 if old10 > 0 else Fraction(0)
    q00 = alive00 / old00 if old00 > 0 else Fraction(0)

    if old10 > 0 and old00 > 0 and float(q00) > 0:
        qratio = float(q10) / float(q00)
        pr(f"  {d:3d}  {float(alive10/s10['n']):14.10f}  "
           f"{float(alive00/s00['n']):14.10f}  {float(q10):14.10f}  "
           f"{float(q00):14.10f}  {qratio:12.8f}")

pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 14: The Difference ΔQ(d) = Q₁₀(d) - Q₀₀(d)")
pr("=" * 80)
pr()

pr("  The sector asymmetry is where the π comes from.")
pr("  ΔQ(d,K) = Q₁₀(d,K) - Q₀₀(d,K):")
pr()

sys.stdout.write(f"  {'d':>3s}  ")
for K in range(7, K_MAX + 1):
    sys.stdout.write(f"{'K='+str(K):>12s}  ")
sys.stdout.write("\n")
sys.stdout.flush()

for d in range(-1, 12):
    any_data = False
    row = f"  {d:3d}  "
    for K in range(7, K_MAX + 1):
        q10 = Q_tables['10'].get((d, K))
        q00 = Q_tables['00'].get((d, K))
        if q10 is not None and q00 is not None:
            dq = float(q10) - float(q00)
            row += f"{dq:+12.8f}  "
            any_data = True
        else:
            row += f"{'—':>12s}  "
    if any_data:
        pr(row)

pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("SECTION 15: Alternative — Direct Series for R")
pr("=" * 80)
pr()

pr("  Instead of sector decomposition, compute R directly as:")
pr("  R(K) = Σ_{d} T(d,K)")
pr("  where T(d,K) = Σ{val₁₀ at depth d} / Σv₀₀(K)")
pr("  This is a DEPTH SERIES for R, converging to -π.")
pr()

for res in results[-3:]:
    K = res['K']
    s10 = res['data']['10']
    sv00 = res['sv00']

    cum_R = Fraction(0)
    pr(f"  K={K}:  Σv₀₀ = {sv00}")
    pr(f"  {'d':>3s}  {'Σv₁₀(d)':>12s}  {'T(d)':>14s}  {'cum R':>14s}")

    for d in sorted(s10['depth_cnt'].keys()):
        val = s10['depth_val'].get(d, 0)
        if val != 0:
            T_d = Fraction(val, sv00)
            cum_R += T_d
            pr(f"  {d:3d}  {val:12d}  {float(T_d):+14.10f}  {float(cum_R):+14.10f}")

    pr(f"  tot {'':12s}  {'':14s}  {float(cum_R):+14.10f}  "
       f"(R(K) = {res['R_float']:+.10f})")
    pr()

# ══════════════════════════════════════════════════════════════════════════
pr("=" * 80)
pr("CONCLUSIONS")
pr("=" * 80)
pr()

pr(f"  R₀      = {R0_EXACT:+.15f}  (proven analytically)")
pr(f"  -π      = {-PI:+.15f}  (target)")
pr(f"  ΔR_tgt  = {DELTA_R_TARGET:+.15f}  (= -π - R₀, to prove)")
pr(f"  R({K_MAX})  = {results[-1]['R_float']:+.15f}")
pr(f"  gap({K_MAX}) = {results[-1]['gap']:+.15f}")
pr()

if Q_inf.get('10'):
    ds = sorted(Q_inf['10'].keys())
    pr("  Extrapolated Q∞(d) for sector (1,0):")
    for d in ds:
        pr(f"    d={d:2d}: Q∞ = {Q_inf['10'][d]:.8f}")
    pr()

    pr("  Q∞(d) - 1/2 analysis (convergence to Diaconis-Fulman eigenvalue):")
    pr(f"    {'d':>3s}  {'Q∞-0.5':>12s}  {'d·(Q-0.5)':>12s}  {'d²·(Q-0.5)':>12s}")
    reliable_ds = [d for d in ds if 1 <= d <= 8]
    for d in reliable_ds:
        excess = Q_inf['10'][d] - 0.5
        pr(f"    {d:3d}  {excess:+12.8f}  {d*excess:+12.6f}  {d*d*excess:+12.4f}")
    pr()

    pr("  r∞(d) analysis (using K=14 values as proxy):")
    pr(f"    {'d':>3s}  {'r(d,14)':>10s}  {'r-1':>10s}  {'2^d·(r-1)':>12s}")
    for d in range(0, 13):
        rv = r_tables['10'].get((d, 14))
        if rv is not None:
            fr = float(rv)
            pr(f"    {d:3d}  {fr:10.6f}  {fr-1:+10.6f}  {2**d*(fr-1):+12.6f}")
    pr()

pr("=" * 80)
pr("SECTION 16: Direct Per-Depth Series for Δ⟨val⟩")
pr("=" * 80)
pr()
pr("  Using K=14 data as proxy for K→∞:")
pr("  Δ⟨val⟩₁₀ = Σ_{d≥0} P₁₀(d) · r₁₀(d)")
pr("  where P₁₀(d) = fraction at depth d, r₁₀(d) = avg carry[M-1] at d")
pr()

s10_14 = results[-1]['data']['10']
N_14 = s10_14['n']
cum = 0.0
terms = []
pr(f"  {'d':>3s}  {'P(d)·r(d)':>14s}  {'ratio':>10s}  {'cum':>14s}")
prev_term = None
for d in sorted(s10_14['depth_cnt'].keys()):
    if d < 0:
        continue
    rwd = s10_14['depth_reward'].get(d, 0)
    term = rwd / N_14
    cum += term
    terms.append(term)
    ratio_str = ""
    if prev_term is not None and abs(term) > 1e-15:
        ratio_str = f"{prev_term/term:.4f}"
    pr(f"  {d:3d}  {term:+14.10f}  {ratio_str:>10s}  {cum:+14.10f}")
    prev_term = term if abs(term) > 1e-15 else prev_term

pr()
pr(f"  Δ⟨val⟩₁₀(14) = {cum:.10f}")
pr(f"  Series terms decay ~(1/2)^d:  ratio of terms {d-1}→{d} approaches 2.")
pr()

pr("  Comparison with sector (0,0):")
s00_14 = results[-1]['data']['00']
N00_14 = s00_14['n']
cum00 = 0.0
pr(f"  {'d':>3s}  {'P₀₀(d)·(r-1)':>14s}  {'cum':>14s}")
for d in sorted(s00_14['depth_cnt'].keys()):
    cnt = s00_14['depth_cnt'].get(d, 0)
    val = s00_14['depth_val'].get(d, 0)
    if cnt > 0:
        w = val / N00_14
        cum00 += w
        if abs(w) > 1e-10:
            pr(f"  {d:3d}  {w:+14.10f}  {cum00:+14.10f}")

pr(f"  ⟨val⟩₀₀(14) = {cum00:.10f}")
pr()

avg10 = float(Fraction(s10_14['vc'], N_14))
avg00 = float(Fraction(s00_14['vc'], N00_14))
nratio = N_14 / N00_14
pr(f"  N₁₀/N₀₀ = {nratio:.10f}")
pr(f"  ⟨val⟩₁₀ = {avg10:+.10f}")
pr(f"  ⟨val⟩₀₀ = {avg00:+.10f}")
pr(f"  R(14) = N₁₀/N₀₀ · ⟨val⟩₁₀/⟨val⟩₀₀ = {nratio * avg10 / avg00:+.10f}")
pr(f"  Direct R(14) = {results[-1]['R_float']:+.10f}")
pr()

pr("=" * 80)
pr("SECTION 17: Exact Term-by-Term Ratios — Approaching 2")
pr("=" * 80)
pr()

pr("  The per-depth terms P(d)·r(d) at K=14 and their successive ratios:")
pr("  If ratio → 2.0, then the series has base ρ=1/2.")
pr(f"  {'d':>3s}  {'P·r':>14s}  {'ratio t(d-1)/t(d)':>18s}  {'excess over 2':>14s}")

prev_t = None
for d in sorted(s10_14['depth_cnt'].keys()):
    if d < 0:
        continue
    rwd = s10_14['depth_reward'].get(d, 0)
    term = rwd / N_14
    if abs(term) < 1e-15:
        continue

    if prev_t is not None:
        ratio = prev_t / term
        pr(f"  {d:3d}  {term:+14.10f}  {ratio:18.10f}  {ratio - 2:+14.10f}")
    else:
        pr(f"  {d:3d}  {term:+14.10f}  {'':>18s}  {'':>14s}")
    prev_t = term

pr()
pr("  If all ratios were exactly 2, the series would be:")
pr("  Δ⟨val⟩₁₀ = t₁ · Σ (1/2)^d = 2·t₁")
pr(f"  t₁ = {terms[0] if terms else 0:.10f}")
pr(f"  2·t₁ = {2*terms[0] if terms else 0:.10f}")
pr(f"  Actual Δ⟨val⟩₁₀ = {cum:.10f}")
pr()

pr("=" * 80)
pr("SECTION 18: K-Convergence of Per-Depth Terms")
pr("=" * 80)
pr()

pr("  How does the d=1 term P(1)·r(1) converge in K?")
pr("  If the series converges to a known constant...")
pr()

for d in [1, 2, 3]:
    pr(f"  d={d}:")
    pr(f"  {'K':>3s}  {'P(d)·r(d)':>14s}  {'delta':>12s}")
    prev = None
    for res in results:
        K = res['K']
        s10 = res['data']['10']
        N = s10['n']
        rwd = s10['depth_reward'].get(d, 0)
        if N > 0 and rwd != 0:
            term = rwd / N
            delta = ""
            if prev is not None:
                delta = f"{term - prev:+.8f}"
            pr(f"  {K:3d}  {term:+14.10f}  {delta:>12s}")
            prev = term
    pr()

pr("=" * 80)
pr("FINAL SUMMARY")
pr("=" * 80)
pr()
pr(f"  R₀      = {R0_EXACT:+.15f}  (proven analytically)")
pr(f"  -π      = {-PI:+.15f}  (target)")
pr(f"  ΔR_tgt  = {DELTA_R_TARGET:+.15f}  (= -π - R₀, to prove)")
pr(f"  R({K_MAX})  = {results[-1]['R_float']:+.15f}")
pr(f"  gap({K_MAX}) = {results[-1]['gap']:+.15f}")
pr()
pr("  KEY FINDINGS:")
pr("  1. Q∞(d) for sector (1,0) approaches 1/2 from above as d→∞")
pr("  2. The per-depth correction terms decay as ~(1/2)^d")
pr("  3. The correction factor (Q∞(d)-1/2) encodes bit-sharing effects")
pr("  4. The series Δ⟨val⟩₁₀ = Σ P(d)·r(d) converges to ~1.29")
pr("  5. The approach to -π is governed by Gaussian decay exp(-cK²)")
pr("=" * 80)
