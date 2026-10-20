# Refined VDOS analysis

The current results use mass-weighted Hann-windowed Welch spectra, 50% overlap, a 2 ps target window, at least four overlapping segments and fourfold FFT padding. Reference and MLIP velocities share the same unwrapping and central-difference preprocessing. The displayed spectra are the spectra used for scoring.

## Coverage and model errors

| Cohort | Models/configurations | Systems | Pairs | Failure penalties | Legacy median | First corrected periodogram median | Refined median |
|---|---:|---:|---:|---:|---:|---:|---:|
| eager | 15 | 14 | 210 | 9 | 23.573% | 24.509% | 14.729% |
| accelerated | 19 | 14 | 266 | 11 | 23.409% | 24.817% | 15.806% |

Model/system coverage, physical source mappings and failure classification are unchanged. Failed pairs retain 100% error. Molecular crystals remain excluded; hydrogen and Pt/H2O remain in VDOS.

## Refined tier medians

| Cohort | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---:|---:|---:|---:|
| eager | 23.436% | 17.482% | 14.286% | 11.420% |
| accelerated | 23.506% | 17.322% | 14.200% | 16.946% |

## Stable-MD VDOS correlations

| Cohort | Force RMSE | F1 | kappa SRME |
|---|---:|---:|---:|
| eager | 0.726577 | -0.749314 | 0.201403 |
| accelerated | 0.722099 | -0.788266 | 0.405384 |

Figure 6 and SI7--9 use each model's stable-system subset. SI11/SI12 and combined Pareto figures use failure-penalized benchmark means.

## Combined Pareto frontier

- eager (6 configurations): orb-v3-direct-20-omat, orb-v2, orb-v3-conservative-inf-omat, MACE-MH-1-OMAT, UMA-S-P1, PET-OMAT-XL.
- accelerated (6 configurations): NequIP-OAM-L, orb-v2, GRACE-2L-OAM, UMA-S-P1 (turbo), UMA-S-P1 (compile), PET-OMAT-XL.

## ASE accelerated VDOS and Figure SI18

Native ASE accelerated VDOS has been recomputed with the same refined mass-weighted Welch settings: 16 configurations, 14 systems and 224 expected model/system pairs. There are 132 successful pairs and 92 missing, incomplete or failed pairs carrying 100% error. These results do not introduce eager substitutions into the native ASE cohort.

Figure SI18 uses refined VDOS for both runners and their existing runner-specific RDF and pressure errors. Shared configurations average timings over identical successful system workloads, using SI17's matching and validation; unshared configurations retain their cohort timings. Observable errors retain failure penalties. The metrics CSV records matched system names, counts and timing basis. Each runner has an independent Pareto frontier.

| Runner/cohort | Plotted configurations | Pareto configurations |
|---|---:|---:|
| mlip-trajs-ase-accelerated | 7 | 3 |
| mlip-trajs-torchsim-accelerated | 19 | 6 |

ASE configurations without a full set of RDF, pressure and timing inputs are excluded from the combined plot: GRACE-2L-MPtrj, GRACE-2L-OAM, MACE-MH-1-OMAT, MACE-MP-0, MACE-MPA-0, UMA-M-P1 (compile), UMA-S-P1 (compile).

ASE UMA-S-P1 (turbo) and UMA-M-P1 (turbo) are excluded from SI18 by request. Their TorchSim counterparts remain plotted. The exclusion applies only to SI18's plotted/exported comparison and its recalculated frontier, leaving native benchmark metrics and other figures unchanged.

The TorchSim cohort retains its documented eager EQ/eSEN and ASE-eager CHGNet substitutions. Missing native ASE metric inputs are not replaced with TorchSim values. The separate plotting command is `uv run --script paper_v2_configs/vdos/figure_SI_18.py`.


## Actual accelerated spectral resolution

Native spacing is 1/window duration; the Hann noise-equivalent bandwidth is wider. Neither quantity improves with zero padding.

| System | Window (fs) | Segments | Native spacing (cm^-1) | Hann bandwidth (cm^-1) |
|---|---:|---:|---:|---:|
| CsSnI3_500K_Ivor_VASP | 2000.0 | 13 | 16.678 | 25.017 |
| H_1050K_Rupp_QE | 142.4 | 4 | 234.244 | 351.367 |
| MAPbBr3_300K_Ivor_VASP | 2000.0 | 28 | 16.678 | 25.017 |
| Pt111w24H2O_380K_Heenen_VASP | 1536.0 | 4 | 21.716 | 32.575 |
| TiSe2_400K_Ivor_VASP | 2000.0 | 13 | 16.678 | 25.017 |
| bulkAg_600K_Kapil | 2000.0 | 23 | 16.678 | 25.017 |
| bulkAu_1500K_Kapil | 2000.0 | 19 | 16.678 | 25.017 |
| bulkCuAu_500K-Artrith_VASP | 440.0 | 4 | 75.810 | 113.715 |
| bulkCuZrAl_1500K_A.Wadowski-J.Schmidt_VASP | 1086.0 | 4 | 30.715 | 46.072 |
| bulkCu_1000K_Kapil | 2000.0 | 35 | 16.678 | 25.017 |
| bulkLiMgAlZnSn_600K_J_Schmidt_VASP | 1204.0 | 4 | 27.705 | 41.557 |
| bulkLiMgAlZnSn_900K_J_Schmidt_VASP | 871.0 | 4 | 38.297 | 57.445 |
| bulkMoS2_300K_NO-VdW_J.Kioseoglou_VASP | 2000.0 | 18 | 16.678 | 25.017 |
| bulkPt3Co_300K_J.Kioseoglou_VASP | 2000.0 | 19 | 16.678 | 25.017 |

## Interpretation and sensitivity

The default has a 2 ps target, not a universal 2 ps effective window. Hydrogen and CuAu require shorter windows and therefore have coarser resolution. Overlapping segments are not independent trajectories. Finite-length uncertainty remains, and close model differences should be interpreted with the sensitivity results.

Comparisons to the legacy and first corrected periodogram runs are in results/comparison/ and results/comparison_periodogram/. The mass weighting changes the observable for mixed-element systems; differences are not solely due to averaging.

Requested 1/2/4 ps windows, stronger short-record averaging, unweighted spectra and a denser FFT grid are assessed in results/sensitivity/.

The full mlip-trajs-torchsim-accelerated cohort was scored under all six settings: 19 model configurations, 14 systems and 266 pairs per setting, including 11 unchanged failure penalties. The diagnostic baseline reproduces the active per-system scores to 1e-9 percentage points.

Changes below refer to each model's failure-penalized benchmark mean; all scores are compared with the default.

| Setting | Median absolute change (pp) | Maximum absolute change (pp) | Spearman rank correlation | Largest rank change |
|---|---:|---:|---:|---:|
| Default: mass, 2 ps, minimum 4 segments | 0.000 | 0.000 | 1.0000 | 0 |
| Mass, 1 ps, minimum 4 segments | 0.227 | 0.818 | 0.9912 | 2 |
| Mass, 4 ps, minimum 4 segments | 0.984 | 1.804 | 0.9947 | 2 |
| Mass, 2 ps, minimum 8 segments | 0.257 | 0.720 | 0.9982 | 1 |
| Unweighted, 2 ps, minimum 4 segments | 1.002 | 2.366 | 0.9895 | 2 |
| Default with eightfold FFT padding | 0.002 | 0.006 | 1.0000 | 0 |

Across the tested mass-weighted window settings, model-mean rank correlations are at least 0.9912; the largest model-mean change is 1.804 pp. This supports stability of the overall accelerated VDOS ranking across these settings, rather than convergence of every system-specific error. The 2 ps, four-segment default is retained as a compromise between averaging and frequency resolution; settings were not selected to minimize benchmark errors.

| System | Median score range across mass-weighted window settings (pp) | Maximum range (pp) |
|---|---:|---:|
| CsSnI3_500K_Ivor_VASP | 1.319 | 5.903 |
| H_1050K_Rupp_QE | 6.213 | 8.422 |
| MAPbBr3_300K_Ivor_VASP | 1.829 | 7.342 |
| Pt111w24H2O_380K_Heenen_VASP | 0.385 | 1.371 |
| TiSe2_400K_Ivor_VASP | 4.773 | 12.868 |
| bulkAg_600K_Kapil | 2.650 | 4.578 |
| bulkAu_1500K_Kapil | 0.850 | 2.263 |
| bulkCuAu_500K-Artrith_VASP | 2.652 | 6.038 |
| bulkCuZrAl_1500K_A.Wadowski-J.Schmidt_VASP | 11.212 | 11.298 |
| bulkCu_1000K_Kapil | 1.575 | 5.238 |
| bulkLiMgAlZnSn_600K_J_Schmidt_VASP | 0.585 | 1.602 |
| bulkLiMgAlZnSn_900K_J_Schmidt_VASP | 0.308 | 0.698 |
| bulkMoS2_300K_NO-VdW_J.Kioseoglou_VASP | 9.721 | 19.700 |
| bulkPt3Co_300K_J.Kioseoglou_VASP | 1.783 | 2.922 |

The system ranges use successful pairs only and vary the 1/2/4 ps target and the minimum of four/eight segments. Short records may share identical effective windows for several requested targets. The unweighted probe changes the observable; the padding probe changes the integration grid, not physical resolution. These diagnostics measure method sensitivity, not confidence intervals or convergence with MD trajectory length.

All 6 settings retain the same 6 accelerated combined RPV Pareto configurations, with RDF, pressure and timings held fixed.

Stable-MD VDOS correlation sensitivity:

| Setting | Force RMSE r (n) | F1 r (n) | kappa SRME r (n) |
|---|---:|---:|---:|
| Mass, 1 ps, minimum 4 segments | 0.7057 (19) | -0.7817 (11) | 0.3849 (11) |
| Default: mass, 2 ps, minimum 4 segments | 0.7221 (19) | -0.7883 (11) | 0.4054 (11) |
| Default with eightfold FFT padding | 0.7220 (19) | -0.7882 (11) | 0.4056 (11) |
| Mass, 2 ps, minimum 8 segments | 0.6773 (19) | -0.7969 (11) | 0.3937 (11) |
| Mass, 4 ps, minimum 4 segments | 0.7355 (19) | -0.8160 (11) | 0.3833 (11) |
| Unweighted, 2 ps, minimum 4 segments | 0.7210 (19) | -0.7816 (11) | 0.3592 (11) |

The correlation directions and relative strengths remain qualitatively consistent under these choices. These are correlations across models with model-specific stable-system subsets, not significance tests or a guarantee of predictive accuracy for a particular system.

Lower refined errors do not demonstrate improved model accuracy: the spectral estimator, bandwidth and mass convention changed. Close rankings and membership of the combined Pareto frontier should be assessed under those choices before revising claims in the paper.

No MD integrations or model inference were rerun. The original vdos directory is retained for comparison. See README.md and methods_vdos.tex for methods.
