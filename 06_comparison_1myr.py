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
    evals = {}
    for k, xs in runs.items():
        e = V.evaluate(xs, win)
        evals[k] = e
        print(f"{k:26s} {e['J']:8.4f} {e['rms_all']:8.4f} {e['r']:7.4f} {e['rms_used']:8.4f} {e['ru']:7.4f} "
              f"{e['rl']:8.4f} {e['rms_mis6']:7.4f} {e['rms_mis11']:7.4f}")
        rows.append((k, e))
    with open(f"{args.outdir}/comparison_1myr.csv", "w") as fh:
        fh.write("model,J,rms_all,r,rms_calib,rms_upper,rms_lower,rms_mis6,rms_mis11\n")
        for k, e in rows:
            fh.write(f"{k},{e['J']:.5f},{e['rms_all']:.5f},{e['r']:.5f},{e['rms_used']:.5f},{e['ru']:.5f},"
                     f"{e['rl']:.5f},{e['rms_mis6']:.5f},{e['rms_mis11']:.5f}\n")

    # ── Plotting con lo stile grafico di riferimento ───────────────────────────
    p0 = V.get_params(args.params[0])
    tg = win.tg
    FW1 = np.interp(tg, inp.t_ins, inp.F_raw)
    Fn = np.interp(tg, inp.t_ins, inp.Fn)
    F0w = inp.F_to_W(p0[8])

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [0.6, 0.9]})
    ax0, ax1 = axes

    # --- Pannello 0: Insolazione ---
    ax0.plot(tg, FW1, color="goldenrod", lw=0.8)
    ax0.axhline(F0w, color="red", ls="--", lw=1.0, label=f"F0={F0w:.0f} W/m²")
    ax0.fill_between(tg, FW1.min() - 5, FW1.max() + 5, where=Fn >= p0[8],
                     color="salmon", alpha=0.13, label=f"c=C2 ({100 * np.mean(Fn >= p0[8]):.0f}%)")
    ax0.set_ylim(FW1.min() - 5, FW1.max() + 5)
    ax0.set_ylabel("W m⁻²")
    ax0.legend(fontsize=7, loc="upper right")
    ax0.grid(True, alpha=0.25)
    ax0.set_title("Insolation (June 21, 65°N)", loc="left", fontsize=9)

    # --- Pannello 1: Margine del Ghiaccio x / Target ---
    ax1.plot(tg, win.xo, color="royalblue", lw=1.5, alpha=0.6, label=r"Normalized $\delta^{18}$O", zorder=5)

    # Paillard (Originale)
    xs_orig = runs["Paillard (published)"]
    e_orig = evals["Paillard (published)"]
    ax1.plot(tg, xs_orig, color="#f58231", ls="-.", lw=1.1,
             label=f"Original Paillard's model ($r={e_orig['r']:.3f}$, $RMS={e_orig['rms_all']:.4f}$)")

    # Paillard (Calibrato)
    xs_opt = runs["Paillard (recalibrated)"]
    e_opt = evals["Paillard (recalibrated)"]
    ax1.plot(tg, xs_opt, color="grey", alpha=0.6, ls=":", lw=1.6,
             label=f"Recalibrated Paillard's model ($r={e_opt['r']:.3f}$, $RMS={e_opt['rms_all']:.4f}$)")

    # Modello v40 (Transport model)
    for k, xs in runs.items():
        if k.startswith("This model"):
            e_v40 = evals[k]
            lbl = "Transport model" if len(args.params) == 1 else f"Transport model ({k.split('(')[1]}"
            ax1.plot(tg, xs, color="#3cb44b", lw=1.4,
                     label=f"{lbl} ($r={e_v40['r']:.3f}$, $RMS={e_v40['rms_all']:.4f}$)")

    # Zone escluse dal calcolo (MIS6 e MIS11) e soglie
    ax1.axvspan(*V.MIS6, color="orange", alpha=0.08, label="MIS6 excluded")
    ax1.axvspan(*V.MIS11, color="lightblue", alpha=0.10, label="MIS11 excluded")
    ax1.axhline(1.0, color="green", ls="-", lw=0.6, alpha=0.35)

    ax1.invert_yaxis()
    ax1.set_ylabel("Ice margin x (normalized latitude, reversed)")
    ax1.legend(fontsize=7, loc="lower left", ncol=2)
    ax1.set_title("Ice margin", loc="left", fontsize=9)
    ax1.grid(True, alpha=0.25)

    # Limiti asse X e ticks coerenti con lo stile di riferimento
    for ax in axes:
        ax.set_xlim(-1000, 0)

    axes[-1].set_xticks(range(-1000, 1, 100))
    axes[-1].set_xlabel("Time (kyr BP)", fontsize=10)

    out = f"{args.figdir}/comparison_1myr.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
