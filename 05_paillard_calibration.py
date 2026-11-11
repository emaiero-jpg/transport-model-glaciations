"""
05_paillard_calibration.py -- re-calibration of the Paillard (1998) ODE model (M2) on LR04, last 1 Myr,
with EXACTLY the same objective J (Eq. 8), window, normalisation, exclusions (MIS 6; MIS 11 as selected) and
plausibility filters as the model of this paper.

Free parameters: tau_i, tau_g, tau_G, v_r, tau_F (thresholds i0 = -0.75, i1 = 0, v_max = 1 are kept).
Multi-start: random seeds (including the published parameters) are screened, Nelder-Mead is run from the best ones.

Usage:  python 05_paillard_calibration.py [--mis11 exclude|include]
Output: results/paillard_m2_calibrated.npy, figures/paillard_m2_calibration.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import v40lib as V

BOUNDS = [(2.0, 30.0), (15.0, 90.0), (15.0, 90.0), (0.5, 1.5), (10.0, 50.0)]   # tau_i, tau_g, tau_G, v_r, tau_F


def main():
    ap = V.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    ap.add_argument("--n-seeds", type=int, default=1000)
    ap.add_argument("--n-top", type=int, default=20)
    args = ap.parse_args()
    V.prepare_dirs(args)
    if args.quick:
        args.n_seeds, args.n_top = 40, 2
    inp = V.load_inputs(args.ins, args.lr04)
    V.print_normalisation(inp)
    win = V.lr04_window(inp, -1000.0, args.mis11)

    def loss(par):
        par = np.asarray(par, float)
        if any(not (lo <= v <= hi) for v, (lo, hi) in zip(par, BOUNDS)):
            return 1e6
        xs = V.paillard_m2(par, inp, win)
        if not np.all(np.isfinite(xs)) or xs.max() - xs.min() < 0.18:
            return 1e6
        rc, ru, rl = V.terms(xs, win)
        return rc + V.AL * ru + V.BE * rl

    rng = np.random.default_rng(42)
    seeds = np.column_stack([rng.uniform(lo, hi, args.n_seeds) for lo, hi in BOUNDS])
    seeds[0] = V.PAILLARD_M2_ORIG
    scores = np.array([loss(s) for s in seeds])
    top = seeds[np.argsort(scores)[:args.n_top]]
    best, bestJ = None, 1e9
    for s in top:
        r = minimize(loss, s, method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-6, fatol=1e-7))
        if r.fun < bestJ:
            bestJ, best = r.fun, r.x.copy()
    xs0, xs1 = V.paillard_m2(V.PAILLARD_M2_ORIG, inp, win), V.paillard_m2(best, inp, win)
    e0, e1 = V.evaluate(xs0, win), V.evaluate(xs1, win)
    print(f"\n{'':22s} {'J':>8} {'RMS_all':>8} {'r':>7} {'MIS6':>7} {'MIS11':>7}")
    for lab, e in (("Paillard (published)", e0), ("Paillard (recalibrated)", e1)):
        print(f"{lab:22s} {e['J']:8.4f} {e['rms_all']:8.4f} {e['r']:7.4f} {e['rms_mis6']:7.4f} {e['rms_mis11']:7.4f}")
    print("recalibrated: tau_i, tau_g, tau_G, v_r, tau_F =", np.round(best, 3))
    np.save(f"{args.outdir}/paillard_m2_calibrated.npy", best)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axvspan(*V.MIS6, color="orange", alpha=0.12, label="MIS 6 excl.")
    ax.axvspan(*V.MIS11, color="lightblue", alpha=0.15, label=f"MIS 11 ({args.mis11}d)".replace("included", "included"))
    ax.plot(win.tg, win.xo, color="royalblue", lw=1.5, alpha=0.6, label=r"LR04 ($\delta^{18}$O, normalised)")
    ax.plot(win.tg, xs0, color="#f58231", ls="-.", lw=1.1, label=f"Paillard, published (r={e0['r']:.3f}, J={e0['J']:.3f})")
    ax.plot(win.tg, xs1, color="crimson", lw=1.6, label=f"Paillard, recalibrated (r={e1['r']:.3f}, J={e1['J']:.3f})")
    ax.invert_yaxis(); ax.set_xlabel("Time (kyr)"); ax.set_ylabel("Ice margin x (reversed)")
    ax.legend(ncol=2, fontsize=9, loc="lower left"); ax.grid(alpha=0.4, ls=":")
    out = f"{args.figdir}/paillard_m2_calibration.png"
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
