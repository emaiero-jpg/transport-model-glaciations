"""
03_tau_experiment.py -- insolation inertia: the forcing F is replaced by a low-pass filtered Fs,
dFs/dt = (F - Fs)/tau  (exact exponential update, 100-kyr spin-up).

For each parameter set:
- By default: (i) tau scan with other parameters fixed, (ii) joint Nelder-Mead calibration of the
  10 parameters + tau starting from the scan minimum.
- If --fixed-params is set: tau scan with other parameters fixed, choosing the best-fit tau without
  joint Nelder-Mead re-calibration.

Produces the figure comparing tau = 0 (dashed) with the optimal/calibrated tau (solid).

Usage:  python 03_tau_experiment.py [--params calibrated mis6-adaptable] [--mis11 exclude|include] [--fixed-params]
Output: results/tau_<name>_mis11_<mode>.npy, results/tau_scan_<name>.csv, figures/tau_calibration.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import v40lib as V

COL = {
    "grant5": "#1f77b4",
    "calibrated": "#1f77b4",
    "secont": "#d62728",
    "mis6-adaptable": "#d62728",
    "mis6_adaptable": "#d62728",
}

LABEL = {
    "grant5": "Best-fit",
    "calibrated": "Best-fit",
    "secont": "Second best-fit",
    "mis6-adaptable": "MIS6-adaptable",
    "mis6_adaptable": "MIS6-adaptable",
}


def main():
    ap = V.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    ap.add_argument("--params", nargs="+", default=["calibrated", "mis6-adaptable"],
                    help="parameter sets to test (e.g. calibrated, grant5, secont, mis6-adaptable or file.npy)")
    ap.add_argument("--fixed-params", action="store_true",
                    help="keep initial 10 physical parameters fixed and select best tau without joint Nelder-Mead re-calibration")
    args = ap.parse_args()
    V.prepare_dirs(args)

    inp = V.load_inputs(args.ins, args.lr04)
    V.print_normalisation(inp)
    win = V.lr04_window(inp, -1000.0, args.mis11)
    obj = V.Objective(inp, win, with_tau=True)

    scan_taus = np.concatenate([np.linspace(0, 1, 11), np.linspace(1, 10, 19)[1:]])
    if args.quick:
        scan_taus = np.array([0.0, 1.0, 2.0])

    res = {}
    for spec in args.params:
        p10 = V.get_params(spec)[:10]
        base = np.append(p10, 0.0)

        print(f"\n=== {spec}: tau scan (other parameters fixed) ===")
        print(f"{'tau':>6} {'J':>8} {'RMS_all':>8} {'r':>7} {'MIS6':>7} {'MIS11':>7}")
        scan = []
        for tau in scan_taus:
            q = base.copy()
            q[10] = tau
            e = V.evaluate(V.simulate(q, inp, win), win)
            scan.append((tau, e["J"], e["rms_all"], e["r"], e["rms_mis6"], e["rms_mis11"]))
            print(f"{tau:6.2f} {e['J']:8.4f} {e['rms_all']:8.4f} {e['r']:7.4f} {e['rms_mis6']:7.4f} {e['rms_mis11']:7.4f}")

        scan = np.array(scan)
        param_tag = spec.replace("/", "_").replace(".", "_")
        np.savetxt(f"{args.outdir}/tau_scan_{param_tag}.csv", scan, delimiter=",",
                   header="tau,J,rms_all,r,rms_mis6,rms_mis11", comments="")

        best_scan_tau = scan[np.argmin(scan[:, 1]), 0]
        start = base.copy()
        start[10] = best_scan_tau

        if args.fixed_params:
            # Selezione diretta del miglior tau dal grid scan senza Nelder-Mead
            best = start
            bestJ = obj(best)
            print(f"\n[FIXED PARAMS MODE] J(tau=0)={scan[0, 1]:.5f}   J(scan minimum, tau={best_scan_tau:.2f})={bestJ:.5f}")
        else:
            # Ri-calibrazione congiunta Nelder-Mead di 10 parametri + tau
            best, bestJ = start, obj(start)
            for _ in range(1 if args.quick else 4):
                r = minimize(obj, best, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-9, fatol=1e-9))
                if r.fun < bestJ:
                    bestJ, best = r.fun, r.x.copy()
            print(f"\n[JOINT CALIBRATION MODE] J(tau=0)={obj(base):.5f}   J(scan minimum)={obj(start):.5f}   J(joint calibration)={bestJ:.5f}")

        for n, v in zip(V.NAMES + ["tau"], best):
            print(f"  {n:4s} = {v:.8f}")
        print(f"  F0 = {inp.F_to_W(best[8]):.2f} W m-2")
        if best[10] < 1e-3:
            print("  NOTE: tau -> 0: the filter is not needed by this solution")

        np.save(f"{args.outdir}/tau_{param_tag}_mis11_{args.mis11}.npy", best)
        res[spec] = (base, best)

    # ── Generation of Figure (stato stilistico di riferimento) ─────────────────
    tg = win.tg
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                             gridspec_kw={"height_ratios": [0.6, 0.9]})
    ax0, ax1 = axes

    # Insolation panel
    ax0.plot(tg, np.interp(tg, inp.t_ins, inp.F_raw), color="goldenrod", lw=0.8, label="Raw Insolation F(t)")
    for spec, (p0, p1) in res.items():
        m = int(round(V.GRID_DT / V.DT_CAL))
        nf = (len(tg) - 1) * m + 1
        Fs = inp.F_to_W(V.fine_forcing(inp, win.t_start, V.DT_CAL, nf, p1[10]))[::m]
        lab_sm = f"{LABEL.get(spec, spec)}, smoothed insolation"
        ax0.plot(tg, Fs, color=COL.get(spec, "k"), lw=1.0, alpha=0.7,
                 label=f"Fs  τ={p1[10]:.2f} kyr ({lab_sm})")
    ax0.axvspan(*V.MIS6, color="orange", alpha=0.12, label="MIS6")
    ax0.set_ylabel("W m⁻²", fontsize=9)
    ax0.set_title("Insolation F(t) and smoothed Fs(t)", fontsize=9)
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(True, alpha=0.25)

    # Ice margin panel
    ax1.plot(tg, win.xo, "k-", lw=1.8, alpha=0.6, label="Normalized δ¹⁸O", zorder=10)
    for spec, (p0, p1) in res.items():
        col = COL.get(spec, "k")
        base_lab = LABEL.get(spec, spec)
        variants = [
            (p0, f"{base_lab} (τ=0)", dict(color=col, lw=1.1, ls="--", alpha=0.80)),
            (p1, f"{base_lab}, smoothed insolation", dict(color=col, lw=1.5, ls="-", alpha=1.00)),
        ]
        for p, lab, sty in variants:
            xs = V.simulate(p, inp, win)
            m = V.evaluate(xs, win)
            ax1.plot(tg, xs,
                     label=f"{lab}  obj={m['J']:.4f}  r={m['r']:.3f}  RMS6={m['rms_mis6']:.4f}",
                     **sty)

    ax1.axvspan(*V.MIS6, color="orange", alpha=0.10)
    ax1.axvspan(*V.MIS11, color="lightblue", alpha=0.10)
    ax1.invert_yaxis()
    ax1.set_ylabel("Ice margin x (normalized latitude, reversed)", fontsize=9)
    ax1.set_xlabel("Time (kyr BP)", fontsize=9)
    ax1.legend(fontsize=7, loc="upper left", ncol=2)
    ax1.set_title("Ice margin", fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax0.margins(x=0)
    ax1.margins(x=0)

    plt.tight_layout()
    out = f"{args.figdir}/tau_calibration.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
