"""
02_convergence_dt.py -- convergence of the solution as the Euler step dt -> 0 (Table: dt convergence).
Parameters are NOT re-calibrated: they are used as given for every dt.

Usage:  python 02_convergence_dt.py [--params grant5 secont ... | file.npy]
Output: results/convergence_dt.csv  (and a LaTeX tabular on screen)
"""
import argparse

import numpy as np

import v40lib as V

DTS = [0.1, 0.01, 0.001, 0.0001]


def main():
    ap = V.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    ap.add_argument("--params", nargs="+", default=["calibrated"]) #"grant5", "secont"])
    args = ap.parse_args()
    V.prepare_dirs(args)
    if args.quick:
        global DTS
        DTS = [0.2, 0.1]
    inp = V.load_inputs(args.ins, args.lr04)
    V.print_normalisation(inp)
    win = V.lr04_window(inp, -1000.0, args.mis11)

    rows = []
    for spec in args.params:
        p = V.get_params(spec)
        print(f"\n=== {spec} ===\n{'dt':>6} {'J':>8} {'RMS_all':>8} {'r':>7} {'MIS6':>7} {'MIS11':>7} {'reversals':>10}")
        for dt in DTS:
            xs = V.simulate(p, inp, win, dt)
            e = V.evaluate(xs, win)
            _, _, sp, _ = V.simulate_fine(p, inp, win, dt)
            nrev = int(np.sum(np.diff(sp.astype(int)) == 1))
            print(f"{dt:6.3f} {e['J']:8.4f} {e['rms_all']:8.4f} {e['r']:7.4f} {e['rms_mis6']:7.4f} "
                  f"{e['rms_mis11']:7.4f} {nrev:10d}")
            rows.append((spec, dt, e["J"], e["rms_all"], e["r"], e["rms_mis6"], e["rms_mis11"], nrev))
    with open(f"{args.outdir}/convergence_dt.csv", "w") as fh:
        fh.write("params,dt_kyr,J,rms_all,r,rms_mis6,rms_mis11,n_reversals\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.5f},{r[3]:.5f},{r[4]:.5f},{r[5]:.5f},{r[6]:.5f},{r[7]}\n")
    print("\nLaTeX rows (dt & RMS & r per parameter set):")
    for dt in DTS:
        cells = [f"{r[3]:.4f} & {r[4]:.4f}" for r in rows if r[1] == dt]
        print(f"{dt:g} & " + " & ".join(cells) + r" \\")
    print("saved", f"{args.outdir}/convergence_dt.csv")


if __name__ == "__main__":
    main()
