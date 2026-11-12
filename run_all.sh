#!/usr/bin/env bash
# Reproduces all results and figures. Run from this folder. Extra arguments (e.g. --quick) are passed to every script.
# Calibration (01) and the Hessian ensemble (04) are the slowest steps (minutes to hours depending on the machine).
set -e

# 1. Calibration on LR04 saving top-10 local minima (MIS 11 excluded by default)
python 01_calibrate.py --top-k 10 "$@"

# 2. Convergence of the solution as dt -> 0
python 02_convergence_dt.py "$@"

# 3. Insolation-inertia experiment using 'calibrated' and 'mis6-adaptable' parameter sets
python 03_tau_experiment.py --params calibrated mis6-adaptable --fixed-params "$@"

# 4. Local robustness analysis (Figure 5) on the calibrated solution (standard D >= 0 mode)
python 04_hessian_ensemble.py --params calibrated --enforce-d-positive "$@"

# 5. Paillard (1998) re-calibrated with the same objective
python 05_paillard_calibration.py "$@"

# 6. This model vs Paillard, last 1 Myr
python 06_comparison_1myr.py "$@"

# 7. Extension to 2 Ma, three Paillard variants
python 07_extension_2ma.py "$@"

# 8. Sea-level test (0-798 ka)
python 08_sealevel_comparison.py "$@"

# 9. Numbers quoted in the text
python 09_diagnostics.py "$@"
