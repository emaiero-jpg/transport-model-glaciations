# Input data (not redistributed here)

Place the following files in this folder (or pass their paths with `--ins`, `--lr04`, `--sealevel`):

| File | Content | Source |
|---|---|---|
| `21jugn.dat` | June-21 insolation at 65°N, two columns: time (kyr, negative = past, here -50000 to +20) and insolation (W m⁻²). Fortran `D` exponents are accepted. | Computed with the software of Laskar et al. (2004) (La2004 solution) |
| `LR04_stack.txt` | LR04 benthic δ¹⁸O stack: age (kyr BP, positive) and δ¹⁸O (‰); lines starting with `#` are ignored. | Lisiecki & Raymo (2005) |
| `Sea-level-stack.tab` | Spratt & Lisiecki sea-level stack v2 (PANGAEA tab file, 0-798 ka). Column index 5 (`SeaLev_longPC1`) is used by default (`--sl-col`). | Spratt & Lisiecki (2025), https://doi.org/10.1594/PANGAEA.979830 |

Add the exact versions/DOIs of the files you used before publishing.
