# Audited failed MD runs

[failed_md_runs.csv](failed_md_runs.csv) registers six TorchSim runs that reached
their requested final step but produced physically invalid trajectories. These
entries override completed timing records in the shared `md_success.py` check.
Each entry applies only to its exact source, system directory, and raw model name.

| Model | Source mode | System | Failure evidence |
| --- | --- | --- | --- |
| MACE-MH-OMAT | Eager | Hydrogen | Peak temperature 2.70 × 10⁸ K |
| MACE-MPA-0 | Eager | Hydrogen | Peak temperature 6.42 × 10⁸ K |
| eSEN-30M-OAM | Eager | Pt/water | Non-finite velocities from step 268; positions from step 269 |
| GRACE-MP | Accelerated | Hydrogen | Peak temperature 2.08 × 10¹⁹ K |
| MACE-MH-OMAT-compile | Accelerated | Hydrogen | Peak temperature 1.50 × 10⁹ K |
| MACE-MPA-0-compile | Accelerated | Hydrogen | Peak temperature 2.44 × 10⁶ K |

The audit examined every saved position and velocity frame, logged temperatures,
kinetic and potential energies, and cells for 424 completed noncrystal runs:
204 eager and 220 accelerated. Molecular crystals were excluded by default.
All six registered runs reached their requested final step.

Hydrogen's target temperature is 1050 K. Peak temperatures and the first step
exceeding ten times the target were computed from the saved velocities using
`T = sum(m_i * v_i²) / (3 * (N - 1) * k_B)`, which agrees with the simulator's
logged temperatures. This also checks steps between temperature log entries.
All five hydrogen runs remained severely overheated through the final logged
step. The Pt/water entry fails because of non-finite state values, rather than
its finite temperature peak.

Registered runs receive 100% RDF, VDOS, and pressure histogram error and remain
included in percentage metric averages. They are excluded from energy/force
RMSE and pressure MAE. Pt/water remains excluded from pressure analysis for all
models. Registered failed runs are also excluded from the timing averages used
by the timing analysis and Pareto plots.

The registry records audited failures; it does not introduce an automatic
temperature threshold for other runs. Modest temporary TiSe2 temperature
overshoots were not classified as failures. Normal temperatures and finite state
values alone do not establish correct structure or chemistry.

Keep entries when reproducing metrics for these trajectories. After replacing
a trajectory with a new, validated run, re-audit it before updating or removing
its registry entry. The raw trajectories and timing records are preserved.
