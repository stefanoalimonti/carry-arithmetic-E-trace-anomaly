# Experiments — Paper E (+ P1, P2)

This directory contains all experiments referenced by Paper E, Paper P1, and Paper P2.

| Script | Description | Referenced in |
|--------|-------------|---------------|
| `E01_base_universality.py` | Base universality of sector ratio | P2 |
| `E02_zeta2_bernoulli.py` | ζ(2) and Bernoulli connection | P2 |
| `E03_K19_analysis.py` | K = 19 analysis | P2 |
| `E04_sector_analysis.py` | Sector analysis | P2 |
| `E05_bridge_greens_function.py` | Bridge Green's function | P2 |
| `E06_alpha3_analysis.py` | α₃ analysis | P2 |
| `E07_K20_analysis.py` | K = 20 analysis | P2 |
| `E08_K20_sectors.c` | K = 20 sector enumeration (C) | P2 |
| `E09_phase_transition.py` | Phase transition / Leibniz mechanism | P1; E §7 |
| `E10_fejer_kernel_shift.py` | Fejér kernel shift | P2 |
| `E11_carry_projection.py` | Carry projection | P2 |
| `E12_correct_model.py` | Correct spectral model | P1, P2 |
| `E13_mechanism_decomposition.py` | Mechanism decomposition | P1, P2 |
| `E14_three_approaches.py` | Three approaches comparison | P1, P2 |
| `E15_bridge_connection.py` | Bridge connection | P1 |
| `E16_dirichlet_series.py` | Dirichlet series approach | P1 |
| `E17_structural_decomp.py` | Structural decomposition | P1 |
| `E18_carry_mechanism.py` | Carry mechanism / boundary layer | P1, P2 |
| `E19_count_ratio.py` | Count ratio and carry weight isolation | P1, P2 |
| `E20_clt_formalization.py` | CLT formalization | P1 |
| `E21_bridge_decomposition.py` | Bridge decomposition / ramp perturbation | P1 |
| `E22_bridge_spectrum.py` | Bridge spectrum / Markov failure | P1 |
| `E23_continuum_val.py` | Continuum val formula | P1; E §2.4 |
| `E24_analytical_depth.py` | Analytical depth / exact symbolic forms | P1 |
| `E25_deep_precision.py` | Deep precision (mpmath 60-digit) | P1 |
| `E26_separable_1D.py` | Separable 1D decomposition | P1 |
| `E27_analytical_M0j.py` | Analytical M₀ⱼ matrix | P1 |
| `E28_exact_G_of_u.py` | Exact G(u) schoolbook | P1; E §3 |
| `E29_cascade_G_of_u.py` | Cascade G(u) | P1; E §4.1 |
| `E30_R0_analytical.py` | R₀ analytical proof | P1; E §3 (Prop. 1) |
| `E31_pi_series.py` | π series identity | P2 |
| `E32_per_depth_pi.py` | Per-depth cancellation series f(d) | E §4.2 |
| `E33_Q_fractions.py` | Q-fraction approach to resolvent | E §7 |
| `E34_wallis_product.py` | Wallis product connection | E §7 (supporting) |
| `E35_angular_series.py` | Continuum DFS cancellation series | E §4.2, App. B |
| `E36_series_acceleration.py` | Series acceleration (Richardson, Aitken) | E §4.2, App. C |
| `E37_spectral_regression.py` | Spectral regression / Linear Mix test | E §6.1, §6.2 |
| `E38_covariance_series.py` | Covariance series / resolvent point z* verification | E §7 Step 1 |
| `E39_weierstrass_gram.py` | Weierstrass integral and Gram matrix | E §7 Step 3 |
| `E40_Aeff_convergence.py` | A_eff(K) convergence to A* (K=3–17) | E §6.1, App. A.1, A.2, App. C |
| `E41_boundary_falsification.py` | Boundary hypothesis falsification | E §5.1 (Prop. 2) |
| `E42_critical_cell.py` | Critical cell analysis | E §5.2 (Prop. 3) |
| `E43_spectral_toy_model.py` | Spectral toy model failure | E §8.4 (Prop. 4) |
| `E44_high_K_sectors.c` | High-K sector enumeration (C, K=21–24+) | P1 §9, P2 §8 |
| `E45_high_K_profiles.c` | High-K per-position cascade profiles (C+OpenMP) | E §8.3, Tables 1–3 |
| `E45_analyze_profiles.py` | Post-processing: sine projections, modal analysis | E §8.3, Table 3 |
| `E45_K19_K20.txt` | E45 output data for K=19, K=20 | E §8.3 |
| `E45_K21.txt` | E45 output data for K=21 | E §8.3 |
| `E215_reward_projection.py` | Cascade reward projection onto sine basis | E §8 (FMD falsification) |
| `E216_parity_decay.py` | Parity-conditioned decay analysis | E §8 |
| `E217_direct_operator.py` | Direct operator measurement | E §8 |
| `E218_d_even_control.py` | D-even control experiment | E §8.1 |
| `E219_spectral_cancellation.py` | Spectral cancellation / modal discovery | E §8.2 |
| `E220_doob_spatial_covariance.py` | Doob spatial covariance / eigenvector analysis | E §8.5 |
| `E221_precision_matrix.py` | Precision matrix analysis | E §8.5 |
| `E221b_partial_correlations.py` | Precision matrix / partial correlations | E §8.5 |
| `E222_cascade_stopping_distribution.py` | Cascade stopping distribution + reflection | E §8.5 |
| `E223_boundary_layer.py` | Per-depth boundary layer decomposition (KEY) | E §8.5 (Table 4) |
| `E224_analytic_boundary.py` | Forward boundary enumeration | E §8.5 |
| `E224b_backward_tree.py` | Backward tree DP / structural exclusion proof (KEY) | E §8.5 |
| `E225_count_ratio.py` | Sector cascade count ratio n₀₀/n₁₀ | E §9.2 |
| `E153_alpha_one_sixth.py` | Convergence analysis: sector weight α(K) → 1/6 | P2 §7.1 |

## Requirements

Python >= 3.8, NumPy, SciPy, SymPy, mpmath. C compiler (with OpenMP) for E08, E44, E45.
