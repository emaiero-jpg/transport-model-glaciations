"""
09_diagnostics.py -- numbers quoted in the text, computed for a given parameter set (no fitting):
  * normalisation constants and the share of the record with c = C2;
  * trigger insolation F_trig(x) (Eq. 11, c = C1, u = 1) at selected x - xH, and the x - xH at which F_trig = F0;
  * weight of the moisture-transport term B1 g^-N relative to B0, and B1/S;
  * threshold on D for the interglacial-inertia term at x = 1;
  * amplifier u after a retreat of 1e-3, and the 2-Ma ramp coefficients (U0, W, Q);
  * transport factor g^-N at x = 1 for xH = 0.30 and xH = 0.54;
  * counts of reversals of dx/dt (glaciation -> retreat) and of committed terminations, split by the
    regime of c at the reversal (F < F0: transport alone / F >= F0: albedo switch), and short "chattering" events.

A reversal is a 'committed termination' if the subsequent retreat amplitude exceeds --amp (default 0.2 in x).
Usage:  python 09_diagnostics.py [--params grant5] [--dt 0.1] [--amp 0.2]
Output: results/diagnostics_<name>.txt, results/reversal_events_<name>.csv
"""
import argparse

import numpy as np

import v40lib as V


def main():
    ap = V.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    ap.add_argument("--params", default="calibrated") #grant5")
    ap.add_argument("--dt", type=float, default=V.DT_CAL)
    ap.add_argument("--amp", type=float, default=0.2)
    args = ap.parse_args()
    V.prepare_dirs(args)
    inp = V.load_inputs(args.ins, args.lr04)
    win = V.lr04_window(inp, -1000.0, args.mis11)
    p = V.get_params(args.params)[:10]
    D, U, S, C1, C2, B0, B1, N, F0, Mu = p
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    out(f"parameters ({args.params}): " + ", ".join(f"{n}={v:.8g}" for n, v in zip(V.NAMES, p)))
    out(f"F_min={inp.F_min:.2f}, F_max={inp.F_max:.2f} W m-2; d18O percentiles (full stack) = {inp.d18_lo:.4f}, {inp.d18_hi:.4f}")
    Fw = inp.F_to_W(F0)
    Fn_win = np.interp(win.tg, inp.t_ins, inp.Fn)
    out(f"F0 = {F0:.4f} (normalised) = {Fw:.2f} W m-2 ; c = C2 for {100*np.mean(Fn_win >= F0):.1f}% of the last 1 Myr")
    out(f"insolation range over the last 1 Myr: {inp.F_to_W(Fn_win.min()):.1f} - {inp.F_to_W(Fn_win.max()):.1f} W m-2")

    ftrig = lambda g: (B0 + B1 * g ** -N) / (S * g ** -N + C1 * g)
    out("\nTrigger insolation, c = C1, u = 1 (Eq. 11):")
    out(f"{'x-xH':>6} {'F_trig':>8} {'W m-2':>8} {'B1 g^-N/B0':>11}")
    for g in (0.20, 0.25, 0.30, 0.35):
        out(f"{g:6.2f} {ftrig(g):8.3f} {inp.F_to_W(ftrig(g)):8.1f} {B1*g**-N/B0:11.3f}")
    gs = np.linspace(0.12, 0.7, 5801)
    ok = (S * gs ** -N + C1 * gs) > 0
    cross = gs[ok][np.argmax(ftrig(gs[ok]) >= F0)] if np.any(ftrig(gs[ok]) >= F0) else float("nan")
    out(f"F_trig(c=C1) < F0 for x - xH < {cross:.3f}  (i.e. x < {V.XH_BASE + cross:.3f})")
    out(f"B1/S = {B1/S:.4f}")
    g1 = 1.0 - V.XH_BASE
    thr = (B1 * g1 ** -N + B0) / F0 - S * g1 ** -N - C1 * g1
    out(f"D needed to hold x = 1 for F just below F0 (c = C1): D > {thr:.4f}   (calibrated D = {D:.3g})")
    out(f"u after a retreat of 1e-3: {np.exp(U * 1e-3 ** Mu):.3f} ; after 0.05: {np.exp(U * 0.05 ** Mu):.3f} ; after 0.3: {np.exp(U * 0.3 ** Mu):.3f}")
    out(f"2-Ma ramp: U0 = {V.RAMP_U_FRAC*U:.4f}, W = {(1-V.RAMP_U_FRAC)*U/(V.RAMP_END-V.RAMP_START):.6f} kyr-1, "
        f"Q = {(V.XH_BASE-V.RAMP_XH_START)/(V.RAMP_END-V.RAMP_START):.6f} kyr-1, xH(-2 Ma) = {V.RAMP_XH_START}")
    out(f"transport factor (x-xH)^-N at x = 1: {(1-V.XH_BASE)**-N:.3f} (xH = 0.30) vs {(1-V.RAMP_XH_START)**-N:.3f} (xH = 0.54)")

    # reversal events
    t, x, sp, Ff = V.simulate_fine(p, inp, win, args.dt)
    d = np.diff(sp.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    rows = []
    for s in starts:
        e = ends[ends > s]
        e = e[0] if len(e) else len(x) - 1
        amp = x[s:e + 1].max() - x[s - 1]
        rows.append((t[s], x[s - 1], Ff[s], amp, (e - s) * args.dt))
    rows = np.array(rows) if rows else np.zeros((0, 5))
    n = len(rows)
    comm = rows[:, 3] >= args.amp if n else np.array([], bool)
    c1 = rows[:, 2] < F0 if n else np.array([], bool)
    chat = rows[:, 4] < 0.5 if n else np.array([], bool)
    out(f"\nReversals (glaciation -> retreat) over -1000..0 kyr at dt = {args.dt}: {n}")
    out(f"  committed terminations (amplitude >= {args.amp}): {int(comm.sum())}")
    out(f"    starting with F < F0 (c = C1, transport alone): {int((comm & c1).sum())}")
    out(f"    starting with F >= F0 (albedo switch active)  : {int((comm & ~c1).sum())}")
    out(f"  minor reversals (amplitude < {args.amp}): {int((~comm).sum())}")
    out(f"  very short retreats (< 0.5 kyr, possible numerical chattering): {int(chat.sum())}")
    np.savetxt(f"{args.outdir}/reversal_events_{args.params}.csv", rows, delimiter=",",
               header="t_kyr,x_at_reversal,F_norm,retreat_amplitude,retreat_duration_kyr", comments="")
    with open(f"{args.outdir}/diagnostics_{args.params}.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nsaved", f"{args.outdir}/diagnostics_{args.params}.txt")


if __name__ == "__main__":
    main()
