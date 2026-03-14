# carry-arithmetic-E-trace-anomaly

**The Trace Anomaly of Binary Multiplication: A Spectral Phase Transition in the Carry Chain**

*Author: Stefano Alimonti* · [ORCID 0009-0009-1183-1698](https://orcid.org/0009-0009-1183-1698)

## Main Result

Numerically, the cascade sector ratio $R(K) = \sum \mathrm{val}_{10} / \sum \mathrm{val}_{00}$ over D-odd products converges toward $-\pi$ as $K \to \infty$ (Conjecture 1 of [P1]; 4.0-digit evidence via Richardson extrapolation). Under the Linear Mix Hypothesis (LMH), this is proved via the Shifted Resolvent Theorem: the carry transfer operator evaluated at $z = -1/2 - 1/\pi$ produces the spectral sum $S(A^*) = -\pi$ through the Dirichlet character $\chi_4(n) = \sin(n\pi/2)$, connecting to the Leibniz series $\pi/4 = 1 - 1/3 + 1/5 - \cdots$.

Key results:
- **Proposition 1** (unconditional): Schoolbook limit $R_0 = -3.9312\ldots$ (exact logarithmic form)
- **Theorem 1** (conditional on LMH): $R = -\pi$ via shifted resolvent at $z = -1/2 - 1/\pi$
- **Observation 2, Proposition 3, Observation 4**: Falsification of the main geometric and generic-operator alternatives for $\pi$

## Status

- **Proved:** Proposition 1, Observation 2, Proposition 3, Observation 4, and the associated structural falsifications.
- **Conditional:** Theorem 1 via LMH + resolvent universality.
- **Open target:** same closure pair as P1: scalar `C=-4` in `R(∞)=C·L(1,χ₄)` and analytic `1/2`-rate preservation proof for D-odd conditioning. The companion paper [L] now realizes the mixed stopping-time channel as a canonical weighted first-return resolvent but still finds no zero-transfer or simple completion mechanism.

Paper complete. The LMH (Conjecture 1) is the only open component. The carry-Dirichlet extension to L²(s,χ₄) is in [L].

## Repository Structure

This repository is the **experimental backbone** for three papers: Paper E (this paper), Paper P1, and Paper P2. All 58 experiments are housed here.

```
paper/trace_anomaly.md               The paper
experiments/
  E01-E08                            Sector ratio computation (P2 experiments)
  E09_phase_transition.py            Phase transition / Leibniz mechanism
  E10-E11                            Fejér kernel and carry projection
  E12-E15                            Model comparison and bridge connection (P1)
  E16-E19                            Dirichlet series and structural decomposition (P1)
  E20-E25                            CLT, bridge, continuum limit (P1)
  E26-E30                            Separable decomposition and R0 proof (P1)
  E31-E34                            Pi series, per-depth pi, Wallis product
  E35-E36                            Angular series and acceleration (Paper E)
  E37-E40                            Spectral regression, Weierstrass, Aeff
  E41-E43                            Falsification: boundary, critical cell, toy model
  E44_high_K_sectors.c               High-K sector enumeration (C, K=21-24+)
  E45_high_K_profiles.c              Per-position cascade profiles (C+OpenMP, K=19-21)
  E45_analyze_profiles.py            Modal analysis / Table 3 post-processing
  E215-E219                          FMD falsification, spectral cancellation discovery
  E220-E222                          Doob covariance, precision matrix, stopping dist.
  E223-E224b                         Boundary layer decomposition (KEY: §8.5)
  E225_count_ratio.py                n₀₀/n₁₀ count ratio falsification (§9.2)
```

## Reproduction

```bash
pip install numpy scipy sympy mpmath
python experiments/E09_phase_transition.py
# ... (see experiments/README.md for full list)
```

## Dependencies

- Python >= 3.8, NumPy, SciPy, SymPy, mpmath
- C compiler (with OpenMP support) for E08, E44, E45

## Companion Papers

| Label | Title | Repository |
|-------|-------|------------|
| [A] | Spectral Theory of Carries | [`carry-arithmetic-A-spectral-theory`](https://github.com/stefanoalimonti/carry-arithmetic-A-spectral-theory) |
| [B] | Carry Polynomials and the Euler Product | [`carry-arithmetic-B-zeta-approximation`](https://github.com/stefanoalimonti/carry-arithmetic-B-zeta-approximation) |
| [C] | Eigenvalue Statistics of Carry Companion Matrices | [`carry-arithmetic-C-matrix-statistics`](https://github.com/stefanoalimonti/carry-arithmetic-C-matrix-statistics) |
| [D] | The Carry-Zero Entropy Bound | [`carry-arithmetic-D-factorization-limits`](https://github.com/stefanoalimonti/carry-arithmetic-D-factorization-limits) |
| [F] | Exact Covariance Structure | [`carry-arithmetic-F-covariance-structure`](https://github.com/stefanoalimonti/carry-arithmetic-F-covariance-structure) |
| [G] | The Angular Uniqueness of Base 2 | [`carry-arithmetic-G-angular-uniqueness`](https://github.com/stefanoalimonti/carry-arithmetic-G-angular-uniqueness) |
| [H] | Carry Polynomials and the Partial Euler Product (Control) | [`carry-arithmetic-H-euler-control`](https://github.com/stefanoalimonti/carry-arithmetic-H-euler-control) |
| [P1] | Pi from Pure Arithmetic | [`carry-arithmetic-P1-pi-spectral`](https://github.com/stefanoalimonti/carry-arithmetic-P1-pi-spectral) |
| [P2] | The Sector Ratio in Binary Multiplication | [`carry-arithmetic-P2-sector-ratio`](https://github.com/stefanoalimonti/carry-arithmetic-P2-sector-ratio) |
| [L] | The Carry–Dirichlet Bridge | [`carry-arithmetic-L-dirichlet-bridge`](https://github.com/stefanoalimonti/carry-arithmetic-L-dirichlet-bridge) |

### Citation

```bibtex
@article{alimonti2026trace_anomaly,
  author  = {Alimonti, Stefano},
  title   = {The Trace Anomaly of Binary Multiplication},
  year    = {2026},
  note    = {Preprint},
  url     = {https://github.com/stefanoalimonti/carry-arithmetic-E-trace-anomaly}
}
```

## License

Paper: CC BY 4.0. Code: MIT License.
