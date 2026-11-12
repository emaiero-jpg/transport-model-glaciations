"""
08_sealevel_comparison.py -- comparison with the Spratt & Lisiecki (2025) sea-level stack (0-798 ka), used as a robustness
test of the proxy: the models are NOT re-calibrated. The sea-level series is mapped to x with its own 1st/99th percentiles
(x = 1: highest sea level; sign inverted so that more ice = larger value) and then rescaled to the LR04 x-range.
Models are integrated from -1000 ka, started at x = 0.7300104334629611, and metrics are computed only where data exist (>-798 ka).
For a like-for-like comparison, the LR04 metrics are also computed on the same 798-kyr window.
NOTE: the sea-level stack is not independent of d18O (partly derived from it, same chronology).

Usage:  python 08_sealevel_comparison.py [--params grant5]
Output: results/sealevel_comparison.csv, figures/sealevel_comparison.png
"""
import argparse
import dataclasses
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v40lib as V

X0 = 0.7300104334629611


def main():
    ap = V.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    ap.add_argument("--params", default="calibrated") #grant5")
    ap.add_argument("--sl-col", type=int, default=5, help="column of the sea-level stack (5 = SeaLev_longPC1)")
    args = ap.parse_args()
    V.prepare_dirs(args)
    inp = V.load_inputs(args.ins, args.lr04)
    pfile = f"{args.outdir}/paillard_m2_calibrated.npy"
    if not os.path.isfile(pfile):
        raise FileNotFoundError(f"{pfile} not found: run 05_paillard_calibration.py first")
    p_rec = np.load(pfile)

    t_sl, d_sl = V.read_sealevel(args.sealevel, args.sl_col)
    lo, hi = float(np.percentile(d_sl, 1.0)), float(np.percentile(d_sl, 99.0))
    print(f"sea level: {t_sl.min():.0f} to {t_sl.max():.0f} kyr; mapping percentiles (-SL): {lo:.2f}, {hi:.2f}")
    wsl = V.make_window(-1000.0, t_sl, d_sl, lo, hi, args.mis11)
    wlr = V.lr04_window(inp, -1000.0, args.mis11)

    t_obs0 = wsl.tg[wsl.obs].min()
    same = {"LR04, same window as sea level": wlr.obs & (wlr.tg >= t_obs0)}
    # second scaling: the sea-level x is mapped linearly so that its 1st/99th percentiles coincide with the 99th/1st
    # percentiles of the LR04 x over the same window (same x-range as the proxy used for calibration)
    q_hi, q_lo = np.percentile(wlr.xo[same["LR04, same window as sea level"]], [99, 1])
    x_m = q_hi - (1.0 - wsl.xo) / 0.5 * (q_hi - q_lo)
    wsl_m = dataclasses.replace(wsl, xo=x_m, std=float(np.nanstd(x_m)))

    p = V.get_params(args.params)

    runs = {"This model": V.simulate(p, inp, wsl_m, x0=X0),
            "Paillard M1 (3 states)": V.paillard_m1(inp, wsl_m, x0=X0),
            "Paillard M2 (published)": V.paillard_m2(V.PAILLARD_M2_ORIG, inp, wsl_m, x0=X0),
            "Paillard M2 (recalibrated)": V.paillard_m2(p_rec, inp, wsl_m, x0=X0)}
    runs_lr = {"This model": V.simulate(p, inp, wlr, x0=X0),
               "Paillard M1 (3 states)": V.paillard_m1(inp, wlr, x0=X0),
               "Paillard M2 (published)": V.paillard_m2(V.PAILLARD_M2_ORIG, inp, wlr, x0=X0),
               "Paillard M2 (recalibrated)": V.paillard_m2(p_rec, inp, wlr, x0=X0)}

    print(f"LR04 x-range over the same window (p1, p99): {q_lo:.3f}, {q_hi:.3f}   (sea level, own scaling: 0.5, 1.0)")
    print(f"\n{'model':28s} {'proxy':40s} {'RMS':>8} {'r':>7}")
    rows = []
    metrics_m = {}
    for k in runs:
        s = V.subwindow_metrics(runs[k], wsl)["all"]
        sm = V.subwindow_metrics(runs[k], wsl_m)["all"]
        metrics_m[k] = sm
        l = V.subwindow_metrics(runs_lr[k], wlr, same)["LR04, same window as sea level"]
        for lab, (a, b) in (("sea level, own percentiles", s), ("sea level, LR04 x-range", sm),
                            ("LR04 (same window)", l)):
            print(f"{k:28s} {lab:40s} {a:8.4f} {b:7.4f}")
            rows.append((k, lab, a, b))
    with open(f"{args.outdir}/sealevel_comparison.csv", "w") as fh:
        fh.write("model,proxy,rms,r\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.5f},{r[3]:.5f}\n")

    # ── Plotting con lo stile grafico di riferimento ───────────────────────────
    tg = wsl_m.tg
    FW1 = np.interp(tg, inp.t_ins, inp.F_raw)
    Fa1 = np.interp(tg, inp.t_ins, inp.Fn)
    F0_W = inp.F_to_W(p[8])

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [0.6, 0.9]})
    ax0, ax1 = axes

    # Insolazione
    ax0.plot(tg, FW1, color="goldenrod", lw=0.8)
    ax0.axhline(F0_W, color="red", ls="--", lw=1.0, label=f"F0={F0_W:.0f} W/m²")
    ax0.fill_between(tg, FW1.min() - 5, FW1.max() + 5, where=Fa1 >= p[8],
                     color="salmon", alpha=0.13, label=f"c=C2 ({100 * np.mean(Fa1 >= p[8]):.0f}%)")
    ax0.set_ylim(FW1.min() - 5, FW1.max() + 5)
    ax0.set_ylabel("W m⁻²")
    ax0.legend(fontsize=7, loc="upper right")
    ax0.grid(True, alpha=0.25)
    ax0.set_title("Insolation (June 21, 65°N)", loc="left", fontsize=9)

    # Ice margin x / Sea Level
    ax1.plot(tg, wsl_m.xo, color="royalblue", lw=1.5, alpha=0.6,
             label="Normalized sea level (Spratt & Lisiecki 2025)", zorder=6)

    m_v40 = metrics_m["This model"]
    ax1.plot(tg, runs["This model"], color="#3cb44b", lw=1.4,
             label=f"Transport model (r={m_v40[1]:.3f}, RMS={m_v40[0]:.4f})")

    m_p2o = metrics_m["Paillard M2 (published)"]
    ax1.plot(tg, runs["Paillard M2 (published)"], color="#f58231", lw=1.1, ls="-.",
             label=f"Original Paillard's model (r={m_p2o[1]:.3f}, RMS={m_p2o[0]:.4f})")

    m_p2c = metrics_m["Paillard M2 (recalibrated)"]
    ax1.plot(tg, runs["Paillard M2 (recalibrated)"], color="gray", alpha=0.6, lw=1.2, ls=":",
             label=f"Recalibrated Paillard's model (r={m_p2c[1]:.3f}, RMS={m_p2c[0]:.4f})")

    ax1.axvline(t_obs0, color="gray", ls="--", lw=0.8, label=f"Start of observed data ({t_obs0:.0f} ka)")
    ax1.axvspan(*V.MIS6, color="orange", alpha=0.08, label="MIS6 excl.")
    ax1.axvspan(*V.MIS11, color="lightblue", alpha=0.10, label="MIS11 excl.")
    #ax1.axhline(0.85, color="steelblue", ls=":", lw=0.7)
    #ax1.axhline(0.58, color="steelblue", ls=":", lw=0.7)
    ax1.axhline(1.0, color="green", ls="-", lw=0.6, alpha=0.35)

    ax1.invert_yaxis()
    ax1.set_xlim(-1000, 0)
    ax1.set_xlabel("Time (kyr BP)", fontsize=11)
    ax1.set_ylabel("Ice margin x (normalized latitude, reversed)")
    ax1.set_title("Ice margin", loc="left", fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=7, loc="lower left", ncol=2)

    axes[-1].set_xticks(range(-1000, 1, 100))
    plt.tight_layout(rect=[0, 0, 1, 0.940])

    out = f"{args.figdir}/sealevel_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
