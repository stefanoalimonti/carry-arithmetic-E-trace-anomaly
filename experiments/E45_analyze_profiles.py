#!/usr/bin/env python3
"""E45 post-processor: reads E45 output and computes sine-basis projections.

Usage: python3 E45_analyze_profiles.py E45_profiles.txt

Computes:
  - w_n^{00}, w_n^{10} for each K from the per-position cascade data
  - Fundamental mode ratio w_1^{10}/w_1^{00} (should → -π)
  - Resolvent reconstruction with LMH eigenvalues
"""
import sys
import re
import math
import numpy as np

PI = math.pi
A_STAR = (2 + 3*PI) / (2*(1 + PI))


def parse_e45_output(filename):
    """Parse E45 stdout into a list of dicts per K."""
    results = []
    current = None

    with open(filename) as f:
        for line in f:
            line = line.rstrip()

            m = re.match(r'>>> COMPLETED K=(\d+)', line)
            if m:
                current = {'K': int(m.group(1)), 'cas_00': {}, 'cas_10': {},
                           'stop_00': {}, 'stop_10': {}}
                results.append(current)
                continue

            if current is None:
                continue

            m = re.match(r'\s+D\s+=\s+(\d+)', line)
            if m:
                current['D'] = int(m.group(1))

            m = re.match(r'\s+S00\s+=\s+(-?\d+)', line)
            if m:
                current['S00'] = int(m.group(1))

            m = re.match(r'\s+S10\s+=\s+(-?\d+)', line)
            if m:
                current['S10'] = int(m.group(1))

            m = re.match(r'\s+R\s+=\s+([+-]?\d+\.\d+)', line)
            if m:
                current['R'] = float(m.group(1))

            m = re.match(r'\s+(\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$',
                         line)
            if m:
                d = int(m.group(1))
                current['cas_00'][d] = int(m.group(2))
                current['cas_10'][d] = int(m.group(3))
                current['stop_00'][d] = int(m.group(4))
                current['stop_10'][d] = int(m.group(5))

    return results


def project_sine(profile_dict, D):
    """Project a {d: value} dict onto sine basis phi_n(d) = sin(n*pi*d/D)."""
    vec = np.zeros(D - 1)
    for d, v in profile_dict.items():
        if 1 <= d <= D - 1:
            vec[d - 1] = v
    n_arr = np.arange(1, D, dtype=np.float64)
    k_arr = np.arange(1, D, dtype=np.float64)
    sin_mat = np.sin(np.outer(n_arr, k_arr) * PI / D)
    return sin_mat @ vec


def analyze(results):
    print("=" * 78)
    print("E45 PROFILE ANALYSIS: FUNDAMENTAL MODE DOMINANCE TEST")
    print("=" * 78)

    print(f"\n{'K':>3s}  {'D':>3s}  {'R(K)':>12s}  {'R+π':>10s}  "
          f"{'w1_10/w1_00':>14s}  {'gap(n=1)':>10s}  {'gap(R)':>10s}")
    print(f"{'─'*3}  {'─'*3}  {'─'*12}  {'─'*10}  {'─'*14}  {'─'*10}  {'─'*10}")

    for res in results:
        K = res['K']
        D = res.get('D', 2*K - 1)
        R = res.get('R', float('nan'))

        w00 = project_sine(res['cas_00'], D)
        w10 = project_sine(res['cas_10'], D)

        if abs(w00[0]) < 1e-15:
            ratio_n1 = float('nan')
        else:
            ratio_n1 = w10[0] / w00[0]

        gap_n1 = ratio_n1 + PI
        gap_R = R + PI

        print(f"{K:3d}  {D:3d}  {R:+12.6f}  {gap_R:+10.6f}  "
              f"{ratio_n1:+14.6f}  {gap_n1:+10.6f}  {gap_R:+10.6f}")

    print()
    print("─" * 78)
    print("RESOLVENT RECONSTRUCTION WITH REAL w_n AND LMH EIGENVALUES")
    print("─" * 78)
    print()
    print(f"{'K':>3s}  {'R(K)':>10s}  {'R_recon':>10s}  {'R_recon+π':>12s}  "
          f"{'R(K)+π':>10s}  {'improvement':>12s}")
    print(f"{'─'*3}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*12}")

    for res in results:
        K = res['K']
        D = res.get('D', 2*K - 1)
        R = res.get('R', float('nan'))

        w00 = project_sine(res['cas_00'], D)
        w10 = project_sine(res['cas_10'], D)

        G_n = np.zeros(D - 1)
        for n in range(1, D):
            G_n[n-1] = sum(math.sin(n * PI * d / D) for d in range(1, D))

        lam = np.array([(1-A_STAR)*math.cos(n*PI/D)/2 + A_STAR/2
                        for n in range(1, D)])
        denom = 1.0 - lam
        mask = np.abs(denom) > 1e-15
        weight = np.zeros(D - 1)
        weight[mask] = G_n[mask] / denom[mask]

        S00 = np.sum(w00 * weight)
        S10 = np.sum(w10 * weight)
        R_recon = S10 / S00 if abs(S00) > 1e-15 else float('nan')

        gap_recon = R_recon + PI
        gap_R = R + PI
        improvement = abs(gap_R) / abs(gap_recon) if abs(gap_recon) > 1e-15 else float('nan')

        print(f"{K:3d}  {R:+10.6f}  {R_recon:+10.6f}  {gap_recon:+12.6f}  "
              f"{gap_R:+10.6f}  {improvement:12.2f}x")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 E45_analyze_profiles.py <E45_output.txt>")
        sys.exit(1)

    results = parse_e45_output(sys.argv[1])
    if not results:
        print("No results found in file.")
        sys.exit(1)

    analyze(results)


if __name__ == '__main__':
    main()
