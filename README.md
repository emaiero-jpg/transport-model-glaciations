# One-equation (1-ODE) model of glacial–interglacial cycles — code

Code accompanying *"Geometry-controlled meridional transport explains threshold deglaciations: a 1-ODE conceptual model of glacial–interglacial cycles"* (Maiero, Giorgi, Solidoro). It reproduces the calibration, the figures and the numbers of the paper.

## The model in one paragraph
The only state variable is the normalised latitude `x` of the southern margin of the Northern-Hemisphere ice sheet
(`x = 1`: fully deglaciated; `xH = 0.3`: northern boundary of the Hadley cell), forced by the normalised 65°N June-21 insolation `F`:

    dx/dt = d·F + u·[ S·F/g^N + c·F·g ] − B1/g^N − B0 ,        g = x − xH

* `c = C2` if `F ≥ F0`, else `C1` (albedo switch); `d = D` only at `x = 1` (interglacial inertia, `D ≥ 0`);
* `u = exp[U (x − x_min)^Mu]` while the margin retreats (`dx/dt > 0`), else 1; `x_min` is the value of `x` at the last
  glaciation → retreat reversal. The sign of `dx/dt` is evaluated at every integration step.
* Explicit Euler with step `dt = 0.1 kyr` (use `02_convergence_dt.py` to check convergence), `x` clipped to `[xH + 1e-6, 1]`,
  rate coefficients in kyr⁻¹, forcing taken from the native 1-kyr insolation series.
* 2-Ma extension: `xH` (0.54 → 0.30) and `U` (2/7 U → U) drift linearly from −2000 to −800 ka (hand-set, not fitted).

## Content
| File | Purpose |
|---|---|
| `v40lib.py` | Shared library: data I/O and normalisation, model (numba), objective J (Eq. 8), metrics, Paillard (1998) models |
| `01_calibrate.py` | Calibration on LR04, last 1 Myr (LHS + Nelder–Mead). MIS 6 always excluded; MIS 11 excluded (test interval) or included. Supports `--top-k N` to save top local minima |
| `02_convergence_dt.py` | Convergence as `dt → 0` (no re-calibration) |
| `03_tau_experiment.py` | Insolation low-pass filter (time constant `tau`): supports joint calibration (default) or fixed physical parameters via `--fixed-params` |
| `04_hessian_ensemble.py` | Hessian sensitivity, ellipsoid and ensemble around a solution (Figure 5). Supports optional `--enforce-d-positive` |
| `05_paillard_calibration.py` | Paillard (1998) ODE model re-calibrated with the same objective J |
| `06_comparison_1myr.py` | This model vs Paillard (published / recalibrated), last 1 Myr |
| `07_extension_2ma.py` | 2-Ma extension; three Paillard variants; sub-window metrics |
| `08_sealevel_comparison.py` | Sea-level test (0–798 ka) |
| `09_diagnostics.py` | Numbers quoted in the text (`F_trig`, threshold on `D`, `u`, ramp coefficients, counts of reversals/terminations) |
| `run_all.sh` | Runs everything end-to-end |
| `tests/` | Synthetic-data generator, smoke test and an equivalence test against the original notebook implementation |

## Installation and use
    pip install -r requirements.txt          # Python ≥ 3.9
    # put the input files in data/ (see data/README.md)
    python 01_calibrate.py --mis11 exclude   # or: ./run_all.sh

Every script has `--help`. Common options: `--ins --lr04 --sealevel` (input paths), `--mis11 exclude|include`, `--outdir`, `--figdir`.
`--quick` runs tiny settings for smoke tests (`tests/smoke_test.sh` runs everything on synthetic data; the numbers are meaningless).

### Advanced script usages
* **Calibration saving Top-K local minima:**
  ```bash
  python 01_calibrate.py --mis11 exclude --top-k 10
