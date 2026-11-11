"""
06_comparison_1myr.py -- v40 model vs Paillard (1998) (published and recalibrated parameters) on LR04, last 1 Myr
(Figure 2 and the 1-Myr rows of the skill table). MIS 6 and (by default) MIS 11 are outside the objective.

Needs results/paillard_m2_calibrated.npy (run 05_paillard_calibration.py first).
Usage:  python 06_comparison_1myr.py [--params grant5 secont]
Output: results/comparison_1myr.csv, figures/comparison_1myr.png
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
    win = V.lr04_window(inp, -1000.0, args.mis11)
    pfile = f"{args.outdir}/paillard_m2_calibrated.npy"
    if not os.path.isfile(pfile):
        raise FileNotFoundError(f"{pfile} not found: run 05_paillard_calibration.py first")
    p_rec = np.load(pfile)

    runs = {f"This model ({s})": V.simulate(V.get_params(s), inp, win) for s in args.params}
    runs["Paillard (published)"] = V.paillard_m2(V.PAILLARD_M2_ORIG, inp, win)
    runs["Paillard (recalibrated)"] = V.paillard_m2(p_rec, inp, win)

    hdr = f"{'model':26s} {'J':>8} {'RMS_all':>8} {'r':>7} {'RMS_cal':>8} {'RMS_up':>7} {'RMS_low':>8} {'MIS6':>7} {'MIS11':>7}"
    print(hdr)
    rows = []
    for k, xs in runs.items():
        e = V.evaluate(xs, win)
        print(f"{k:26s} {e['J']:8.4f} {e['rms_all']:8.4f} {e['r']:7.4f} {e['rms_used']:8.4f} {e['ru']:7.4f} "
              f"{e['rl']:8.4f} {e['rms_mis6']:7.4f} {e['rms_mis11']:7.4f}")
        rows.append((k, e))
    with open(f"{args.outdir}/comparison_1myr.csv", "w") as fh:
        fh.write("model,J,rms_all,r,rms_calib,rms_upper,rms_lower,rms_mis6,rms_mis11\n")
        for k, e in rows:
            fh.write(f"{k},{e['J']:.5f},{e['rms_all']:.5f},{e['r']:.5f},{e['rms_used']:.5f},{e['ru']:.5f},"
                     f"{e['rl']:.5f},{e['rms_mis6']:.5f},{e['rms_mis11']:.5f}\n")

    p0 = V.get_params(args.params[0])
    tg = win.tg
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [0.6, 0.9]})
    ax0.plot(tg, np.interp(tg, inp.t_ins, inp.F_raw), color="goldenrod", lw=0.8)
    F0w = inp.F_to_W(p0[8])
    ax0.axhline(F0w, color="red", ls="--", lw=1.0, label=f"$F_0$ = {F0w:.0f} W m$^{{-2}}$")
    Fn = np.interp(tg, inp.t_ins, inp.Fn)
    ax0.fill_between(tg, 430, 570, where=Fn >= p0[8], color="salmon", alpha=0.15,
                     label=f"c = C2 ({100*np.mean(Fn >= p0[8]):.0f}% of the record)")
    ax0.set_ylabel("W m$^{-2}$"); ax0.set_title("65N June-21 insolation"); ax0.legend(fontsize=8); ax0.grid(alpha=0.25)
    ax1.plot(tg, win.xo, color="royalblue", lw=1.4, alpha=0.6, label=r"Normalised $\delta^{18}$O")
    style = {"Paillard (published)": ("#f58231", "--"), "Paillard (recalibrated)": ("crimson", ":")}
    for k, xs in runs.items():
        c, ls = style.get(k, ("#3cb44b", "-"))
        ax1.plot(tg, xs, color=c, ls=ls, lw=1.3, label=k)
    ax1.axvspan(*V.MIS6, color="orange", alpha=0.10); ax1.axvspan(*V.MIS11, color="lightblue", alpha=0.12)
    ax1.invert_yaxis(); ax1.set_xlabel("Time (kyr)"); ax1.set_ylabel("Ice margin x (reversed)")
    ax1.legend(fontsize=8, loc="lower left", ncol=2); ax1.grid(alpha=0.25)
    ax0.margins(x=0); ax1.margins(x=0)
    out = f"{args.figdir}/comparison_1myr.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
