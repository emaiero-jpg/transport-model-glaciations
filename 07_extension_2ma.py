"""
07_extension_2ma.py -- extension to 2 Ma. The 10 parameters are those calibrated on the last 1 Myr and are
NOT re-calibrated; only two quantities drift linearly before -800 ka (hand-set, not fitted): xH from 0.54 to 0.30
and U from 2/7 U to U (Eq. 6). Same d18O normalisation as for 1 Myr (full-stack percentiles).

Paillard (1998) is run in the variants of the paper:
  A. published parameters, trends in insolation (3 W m-2 Myr-1) and in v_max over the WHOLE 2-Myr domain;
  B. published parameters, the same total trends compressed between -2000 and -800 ka (comparable ramp window);
  C. recalibrated parameters (script 05, v_max = 1 at the end of the ramp), same ramp window as B.
Variants B and C are "Paillard-type" models: they are not the published 2-Ma configuration.

Metrics are reported on sub-windows (0-1 Ma, 1-2 Ma, pre-/post-MPT, MIS 6, MIS 11).
Usage:  python 07_extension_2ma.py [--params grant5 secont]
Output: results/extension_2ma.csv, figures/extension_2ma.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v40lib as V


def main():
    ap = V.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    ap.add_argument("--params", nargs="+", default=["calibrated"]) #grant5"])
    args = ap.parse_args()
    V.prepare_dirs(args)
    inp = V.load_inputs(args.ins, args.lr04)
    V.print_normalisation(inp)
    win = V.lr04_window(inp, -2000.0, args.mis11)
    pfile = f"{args.outdir}/paillard_m2_calibrated.npy"
    if not os.path.isfile(pfile):
        raise FileNotFoundError(f"{pfile} not found: run 05_paillard_calibration.py first")
    p_rec = np.load(pfile)
    print(f"x_obs at -2000 kyr = {win.xo[0]:.3f}  (xH starts at {V.RAMP_XH_START}; x is clipped above xH)")

    runs = {}
    for s in args.params:
        p = V.get_params(s)
        runs[f"This model ({s})"] = V.simulate(p, inp, win, ramp=True)
        U = p[1]
        print(f"ramp for {s}: U0 = {V.RAMP_U_FRAC*U:.4f}, W = {(1-V.RAMP_U_FRAC)*U/(V.RAMP_END-V.RAMP_START):.6f} kyr-1, "
              f"Q = {(V.XH_BASE-V.RAMP_XH_START)/(V.RAMP_END-V.RAMP_START):.6f} kyr-1")
    runs["Paillard A: published, trend over 2 Ma"] = V.paillard_m3(V.PAILLARD_M3_ORIG, inp, win, 0.0, V.PAILLARD_M3_VMAX_END)
    runs["Paillard B: published, ramp -2 to -0.8 Ma"] = V.paillard_m3(V.PAILLARD_M3_ORIG, inp, win, V.RAMP_END, V.PAILLARD_M3_VMAX_END)
    runs["Paillard C: recalibrated, ramp -2 to -0.8 Ma"] = V.paillard_m3(p_rec, inp, win, V.RAMP_END, 1.0)

    names = None
    table = {}
    for k, xs in runs.items():
        table[k] = V.subwindow_metrics(xs, win)
        names = names or list(table[k].keys())
    w = max(len(k) for k in runs)
    print(f"\nRMS / r on sub-windows")
    print(f"{'model':{w}s} " + " ".join(f"{n:>22s}" for n in names))
    for k, d in table.items():
        print(f"{k:{w}s} " + " ".join(f"{d[n][0]:9.4f} /{d[n][1]:7.4f}   " for n in names))
    with open(f"{args.outdir}/extension_2ma.csv", "w") as fh:
        fh.write("model,window,rms,r\n")
        for k, d in table.items():
            for n, (a, b) in d.items():
                fh.write(f"{k},{n},{a:.5f},{b:.5f}\n")

    tg = win.tg
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [0.5, 1.0]})
    ax0.plot(tg, np.interp(tg, inp.t_ins, inp.F_raw), color="goldenrod", lw=0.7)
    ax0.set_ylabel("W m$^{-2}$"); ax0.set_title("65N June-21 insolation"); ax0.grid(alpha=0.25)
    ax1.plot(tg, win.xo, color="royalblue", lw=1.4, alpha=0.6, label=r"Normalised $\delta^{18}$O", zorder=10)
    cols = ["#3cb44b", "#7fbf7f", "#f58231", "#e6a000", "crimson"]
    for (k, xs), c, ls in zip(runs.items(), cols, ["-", "-", "--", "-.", ":"]):
        e = table[k]["all"]
        ax1.plot(tg, xs, color=c, ls=ls, lw=1.2, label=f"{k} (RMS={e[0]:.3f}, r={e[1]:.3f})")
    ax1.axvline(V.RAMP_END, color="k", lw=0.6, ls=":")
    ax1.axvspan(*V.MIS6, color="orange", alpha=0.10); ax1.axvspan(*V.MIS11, color="lightblue", alpha=0.12)
    ax1.invert_yaxis(); ax1.set_xlabel("Time (kyr)"); ax1.set_ylabel("Ice margin x (reversed)")
    ax1.legend(fontsize=7, loc="lower left", ncol=2); ax1.grid(alpha=0.25)
    ax0.margins(x=0); ax1.margins(x=0)
    out = f"{args.figdir}/extension_2ma.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
