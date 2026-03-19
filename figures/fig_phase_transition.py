#!/usr/bin/env python3
"""
fig_phase_transition.py — The S(A) Möbius curve: phase transition from +2/3 to -π.

Shows how the spectral sum S(A) = 2(1-A)/(3-2A) transforms continuously
from the Markov prediction +2/3 (A=0) through zero (A=1) to -π (A=A*≈1.379),
with a pole at A=3/2.

Output: fig_phase_transition.png
"""

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(11, 6.5))

# S(A) = 2(1-A)/(3-2A), pole at A = 3/2
A_left = np.linspace(-0.3, 1.48, 500)
A_right = np.linspace(1.52, 2.5, 200)

def S(A):
    return 2*(1-A)/(3-2*A)

# A* such that S(A*) = -π
A_star = (3*np.pi + 2) / (2*(np.pi + 1))  # ≈ 1.379

# Plot the curve
ax.plot(A_left, S(A_left), '-', color='#2255AA', linewidth=2.5, zorder=3)
ax.plot(A_right, S(A_right), '-', color='#2255AA', linewidth=2.5, zorder=3)

# Pole
ax.axvline(x=1.5, color='#CCCCCC', linestyle=':', linewidth=1.5, zorder=1)
ax.text(1.53, 8, 'pole\n$A=3/2$', fontsize=9, color='#999', ha='left')

# Key points
points = [
    (0, S(0), r"Markov: $S = +2/3$", '#2ECC40', 'right', (0.25, 1.8)),
    (1, S(1), r"Critical: $S = 0$", '#E8A838', 'right', (0.6, 1.2)),
    (A_star, -np.pi, r"$S = -\pi$, $A^* \approx 1.379$", '#CC3333', 'left', (0.5, -5.5)),
]

for (x, y, label, color, ha, xytext) in points:
    ax.plot(x, y, 'o', color=color, markersize=12, zorder=5, markeredgecolor='#333',
            markeredgewidth=1.5)
    ax.annotate(label, xy=(x, y), xytext=xytext,
                fontsize=11, fontweight='bold', color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.9))

# Horizontal reference lines
ax.axhline(y=2/3, color='#2ECC40', linestyle='--', alpha=0.4, linewidth=1)
ax.axhline(y=0, color='#999', linestyle='-', alpha=0.3, linewidth=1)
ax.axhline(y=-np.pi, color='#CC3333', linestyle='--', alpha=0.4, linewidth=1)

# Labels on right edge
ax.text(2.55, 2/3, '$+2/3$', fontsize=10, color='#2ECC40', va='center')
ax.text(2.55, -np.pi, '$-\\pi$', fontsize=10, color='#CC3333', va='center')

# Shading: Markov region
ax.axvspan(-0.3, 0, alpha=0.05, color='#2ECC40')
ax.text(-0.15, -8, 'sub-\ncritical', fontsize=9, color='#2ECC40', ha='center', alpha=0.7)

# Shading: supercritical
ax.axvspan(1, 1.5, alpha=0.05, color='#CC3333')
ax.text(1.25, -8, 'super-\ncritical', fontsize=9, color='#CC3333', ha='center', alpha=0.7)

ax.set_xlabel('Correlation parameter $A$', fontsize=13)
ax.set_ylabel('$S(A)$', fontsize=14)
ax.set_title(r'Phase Transition: $S(A) = \frac{2(1-A)}{3-2A}$',
             fontsize=16, fontweight='bold')

ax.set_xlim(-0.35, 2.6)
ax.set_ylim(-10, 10)
ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig("papers/carry-arithmetic-E-trace-anomaly/figures/fig_phase_transition.png",
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("OK: fig_phase_transition.png")
