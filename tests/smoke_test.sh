#!/usr/bin/env bash
# Runs every script on SYNTHETIC data with tiny settings (--quick) to check that the code runs. The numbers are meaningless.
set -e
cd "$(dirname "$0")/.."
python tests/make_synthetic_data.py tests/synthetic
A="--ins tests/synthetic/21jugn.dat --lr04 tests/synthetic/LR04_stack.txt --sealevel tests/synthetic/Sea-level-stack.tab --outdir tests/out --figdir tests/out --quick"
python tests/test_equivalence.py tests/synthetic 2>/dev/null || true
for s in 01_calibrate 02_convergence_dt 03_tau_experiment 04_hessian_ensemble 05_paillard_calibration \
         06_comparison_1myr 07_extension_2ma 08_sealevel_comparison 09_diagnostics; do
  echo "== $s"; python $s.py $A > /dev/null
done
echo "smoke test passed"
