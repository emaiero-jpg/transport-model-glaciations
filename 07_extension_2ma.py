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

    # ── Plotting con lo stile grafico di riferimento (Figure B) ────────────────
    p0 = V.get_params(args.params[0])
    tg = win.tg
    t_BP = -tg
    T2 = 2000

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [0.6, 0.9]})
    ax0, ax1 = axes

    # --- Pannello 0: Insolazione ---
    FW2 = np.interp(tg, inp.t_ins, inp.F_raw)
    FW2_trend = FW2 + 3.0 * (tg / 1000.0)
    Fn = np.interp(tg, inp.t_ins, inp.Fn)
    F0_W = inp.F_to_W(p0[8])

    ax0.plot(t_BP, FW2, color="goldenrod", lw=0.7, alpha=0.7, label="Raw insolation")
    ax0.plot(t_BP, FW2_trend, color="tomato", lw=0.9, label=r"With $+3\ \mathrm{W\,m^{-2}\,Myr^{-1}}$ trend")
    ax0.axhline(F0_W, color="red", ls="--", lw=1.0, label=f"F0={F0_W:.0f} W/m²")
    ax0.fill_between(t_BP, FW2.min() - 5, FW2.max() + 5, where=Fn >= p0[8],
                     color="salmon", alpha=0.13, label=f"c=C2 ({100 * np.mean(Fn >= p0[8]):.0f}%)")
    ax0.set_ylim(FW2.min() - 8, FW2.max() + 5)
    ax0.set_ylabel("W m⁻²")
    ax0.legend(fontsize=7, loc="upper right", ncol=2)
    ax0.grid(True, alpha=0.25)
    ax0.set_title("Insolation (June 21, 65°N)", loc="left", fontsize=9)

    # --- Pannello 1: Curve di estensione glaciale ---
    ax1.plot(t_BP, win.xo, color="royalblue", lw=1.5, alpha=0.55, label=r"Normalized $\delta^{18}$O", zorder=5)

    m_porig = table["Paillard A: published, trend over 2 Ma"]["all"]
    ax1.plot(t_BP, runs["Paillard A: published, trend over 2 Ma"], color="#f58231", ls="-.", lw=1.2,
             label=f"Original Paillard's model for 2Ma ($r={m_porig[1]:.3f}$, $RMS={m_porig[0]:.4f}$)")

    m_pmpt = table["Paillard B: published, ramp -2 to -0.8 Ma"]["all"]
    ax1.plot(t_BP, runs["Paillard B: published, ramp -2 to -0.8 Ma"], color="tab:pink", ls="--", lw=1.2,
             label=f"Original Paillard's model, linear trends before MPT (-2Myr to -0.8Myr) ($r={m_pmpt[1]:.3f}$, $RMS={m_pmpt[0]:.4f}$)")

    m_pcal = table["Paillard C: recalibrated, ramp -2 to -0.8 Ma"]["all"]
    ax1.plot(t_BP, runs["Paillard C: recalibrated, ramp -2 to -0.8 Ma"], color="gray", alpha=0.6, ls=":", lw=1.3,
             label=f"Recalibrated Paillard's model, linear trends before MPT (-2Myr to -0.8Myr) ($r={m_pcal[1]:.3f}$, $RMS={m_pcal[0]:.4f}$)")

    for k, xs in runs.items():
        if k.startswith("This model"):
            m_v40 = table[k]["all"]
            lbl = "Transport model" if len(args.params) == 1 else f"Transport model ({k.split('(')[1]}"
            ax1.plot(t_BP, xs, color="#3cb44b", lw=1.6,
                     label=f"{lbl} ($r={m_v40[1]:.3f}$, $RMS={m_v40[0]:.4f}$)")

    ax1.axvspan(140, 200, color="orange", alpha=0.08, label="MIS6 excl.")
    ax1.axvspan(370, 430, color="lightblue", alpha=0.10, label="MIS11 excl.")
    #ax1.axhline(0.85, color="steelblue", ls=":", lw=0.7)
    #ax1.axhline(0.58, color="steelblue", ls=":", lw=0.7)
    ax1.axhline(1.0, color="green", ls="-", lw=0.6, alpha=0.35)

    ax1.invert_yaxis()
    ax1.set_ylabel("Ice margin x (normalized latitude, reversed)")
    ax1.legend(fontsize=7, loc="upper left", ncol=2)
    ax1.set_title("Ice margin x", loc="left", fontsize=9)
    ax1.grid(True, alpha=0.25)

    for ax in axes:
        ax.set_xlim(2000, 0)

    axes[-1].set_xticks(range(0, T2 + 1, 200))
    axes[-1].set_xlabel("Time (kyr BP)", fontsize=10)

    out = f"{args.figdir}/extension_2ma.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
