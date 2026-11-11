"""
03_tau_experiment.py -- insolation inertia: the forcing F is replaced by a low-pass filtered Fs,
dFs/dt = (F - Fs)/tau  (exact exponential update, 100-kyr spin-up).

For each parameter set: (i) tau scan with the other parameters fixed, (ii) joint Nelder-Mead
calibration of the 10 parameters + tau starting from the scan minimum (the calibrated model has 11 parameters).
Produces the figure comparing tau = 0 (dashed) with the calibrated tau (solid).

Usage:  python 03_tau_experiment.py [--params grant5 secont] [--mis11 exclude|include]
Output: results/tau_<name>_mis11_<mode>.npy, results/tau_scan_<name>.csv, figures/tau_calibration.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import v40lib as V

COL = {"grant5": "#1f77b4", "secont": "#d62728"}
LABEL = {"grant5": "Best-fit", "secont": "Second best-fit"}


def main():
    ap = V.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    ap.add_argument("--params", nargs="+", default=["calibrated","secont"]) #grant5", "secont"])
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
        base = np.append(V.get_params(spec)[:10], 0.0)
        print(f"\n=== {spec}: tau scan (other parameters fixed) ===")
        print(f"{'tau':>6} {'J':>8} {'RMS_all':>8} {'r':>7} {'MIS6':>7} {'MIS11':>7}")
        scan = []
        for tau in scan_taus:
            q = base.copy(); q[10] = tau
            e = V.evaluate(V.simulate(q, inp, win), win)
            scan.append((tau, e["J"], e["rms_all"], e["r"], e["rms_mis6"], e["rms_mis11"]))
            print(f"{tau:6.2f} {e['J']:8.4f} {e['rms_all']:8.4f} {e['r']:7.4f} {e['rms_mis6']:7.4f} {e['rms_mis11']:7.4f}")
        scan = np.array(scan)
        np.savetxt(f"{args.outdir}/tau_scan_{spec}.csv", scan, delimiter=",",
                   header="tau,J,rms_all,r,rms_mis6,rms_mis11", comments="")
        start = base.copy(); start[10] = scan[np.argmin(scan[:, 1]), 0]
        best, bestJ = start, obj(start)
        for _ in range(1 if args.quick else 4):
            r = minimize(obj, best, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-9, fatol=1e-9))
            if r.fun < bestJ:
                bestJ, best = r.fun, r.x.copy()
        print(f"\nJ(tau=0)={obj(base):.5f}   J(scan minimum)={obj(start):.5f}   J(joint calibration)={bestJ:.5f}")
        for n, v in zip(V.NAMES + ["tau"], best):
            print(f"  {n:4s} = {v:.8f}")
        print(f"  F0 = {inp.F_to_W(best[8]):.2f} W m-2")
        if best[10] < 1e-3:
            print("  NOTE: tau -> 0: the filter is not needed by this solution")
        np.save(f"{args.outdir}/tau_{spec}_mis11_{args.mis11}.npy", best)
        res[spec] = (base, best)

    # figure
    tg = win.tg
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [0.6, 0.9]})
    fig.suptitle("Insolation inertia tau: solid = calibrated tau, dashed = tau = 0", fontsize=10)
    ax0.plot(tg, np.interp(tg, inp.t_ins, inp.F_raw), color="goldenrod", lw=0.8, label="F (raw)")
    for spec, (p0, p1) in res.items():
        m = int(round(V.GRID_DT / V.DT_CAL))
        nf = (len(tg) - 1) * m + 1
        Fs = inp.F_to_W(V.fine_forcing(inp, win.t_start, V.DT_CAL, nf, p1[10]))[::m]
        ax0.plot(tg, Fs, color=COL.get(spec, "k"), lw=1.0, alpha=0.7, label=f"Fs, tau = {p1[10]:.2f} kyr ({spec})")
    ax0.axvspan(*V.MIS6, color="orange", alpha=0.12, label="MIS 6")
    ax0.set_ylabel("W m$^{-2}$"); ax0.set_title("Insolation F(t) and filtered Fs(t)", fontsize=9)
    ax0.legend(fontsize=8, loc="upper right"); ax0.grid(alpha=0.25)
    ax1.plot(tg, win.xo, "k-", lw=1.8, alpha=0.6, label=r"Normalised $\delta^{18}$O", zorder=10)
    for spec, (p0, p1) in res.items():
        for p, ls, lw, lab in ((p0, "--", 1.1, f"{LABEL.get(spec, spec)} (tau = 0)"),
                               (p1, "-", 1.5, f"{LABEL.get(spec, spec)}, filtered insolation (tau = {p1[10]:.2f} kyr)")):
            e = V.evaluate(V.simulate(p, inp, win), win)
            ax1.plot(tg, V.simulate(p, inp, win), color=COL.get(spec, "k"), ls=ls, lw=lw,
                     alpha=0.85, label=f"{lab}  RMS(MIS 6) = {e['rms_mis6']:.3f}")
    ax1.axvspan(*V.MIS6, color="orange", alpha=0.10); ax1.axvspan(*V.MIS11, color="lightblue", alpha=0.10)
    ax1.invert_yaxis(); ax1.set_xlabel("Time (kyr)")
    ax1.set_ylabel("Ice margin x (normalised latitude, reversed)"); ax1.set_title("Ice margin", fontsize=9)
    ax1.legend(fontsize=7, loc="upper left", ncol=2); ax1.grid(alpha=0.25)
    ax0.margins(x=0); ax1.margins(x=0)
    fig.tight_layout()
    out = f"{args.figdir}/tau_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
