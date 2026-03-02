#!/usr/bin/env python3
"""E225: Fast count-ratio n00/n10 for D-odd cascade products.

Tests the discovery from E224b: the ratio of cascade counts
n_00/n_10 (number of cascade-stopping pairs per sector) appears
to converge to pi as K -> infinity.

This script counts ONLY, no val computation, for maximum speed.
Pushes to K=14 (or beyond if feasible).

Usage: python3 E225_count_ratio.py [K_max]
"""

import sys
import time
from math import pi


def count_sectors(K):
    """Count cascade events per sector for K-bit D-odd products."""
    D = 2 * K - 1
    lo = 1 << (K - 1)
    hi = 1 << K

    n00 = 0
    n10 = 0
    n01 = 0
    n_dodd = 0
    v00 = 0
    v10 = 0

    for p in range(lo, hi):
        a1 = (p >> (K - 2)) & 1
        for q in range(lo, hi):
            prod = p * q
            if prod.bit_length() != D:
                continue
            n_dodd += 1
            c1 = (q >> (K - 2)) & 1

            carries = [0] * (D + 1)
            for d in range(D):
                cv = 0
                for i in range(max(0, d - K + 1), min(d, K - 1) + 1):
                    cv += ((p >> i) & 1) * ((q >> (d - i)) & 1)
                carries[d + 1] = (cv + carries[d]) >> 1

            M = None
            for m in range(D, 0, -1):
                if carries[m] > 0:
                    M = m
                    break
            if M is None:
                continue

            val = carries[M - 1] - 1

            sec = a1 * 2 + c1
            if sec == 0:
                n00 += 1
                v00 += val
            elif sec == 2:
                n10 += 1
                v10 += val
            elif sec == 1:
                n01 += 1

    return n00, n10, n01, n_dodd, v00, v10


def main():
    K_max = int(sys.argv[1]) if len(sys.argv) > 1 else 13

    print("E225: SECTOR COUNT RATIO n00/n10 — TESTING n00/n10 → π")
    print("=" * 72)
    print(f"Target: n00/n10 → π = {pi:.10f}")
    print(f"K range: 5 .. {K_max}")
    print()

    print(f"  {'K':>3s}  {'n00':>10s}  {'n10':>10s}  {'n00/n10':>10s}  "
          f"{'gap':>10s}  {'R(K)':>10s}  {'R+pi':>10s}  {'time':>6s}")
    print("  " + "-" * 68)

    for K in range(5, K_max + 1):
        t0 = time.time()
        n00, n10, n01, n_dodd, v00, v10 = count_sectors(K)
        elapsed = time.time() - t0

        ratio = n00 / n10 if n10 > 0 else float('nan')
        gap = ratio - pi
        R = v10 / v00 if abs(v00) > 0 else float('nan')

        print(f"  {K:3d}  {n00:10d}  {n10:10d}  {ratio:10.6f}  "
              f"{gap:+10.6f}  {R:10.6f}  {R + pi:+10.6f}  {elapsed:5.1f}s")

    # Also include E45 K=21 data point
    print()
    print("  External data (E45):")
    n00_21 = 259003293792
    n10_21 = 82862658279
    r21 = n00_21 / n10_21
    print(f"  {21:3d}  {n00_21:10d}  {n10_21:10d}  {r21:10.6f}  "
          f"{r21 - pi:+10.6f}  {-3.1334:10.6f}  {-3.1334 + pi:+10.6f}   E45")


if __name__ == '__main__':
    main()
