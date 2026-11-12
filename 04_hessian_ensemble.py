"""
04_hessian_ensemble.py -- local robustness analysis around a solution (Figure 5 of the paper).

1. Hessian H of J at the solution, by central finite differences (second differences on the diagonal,
   four-point mixed differences off the diagonal), relative step FD_FRAC*|p_j|, parameters in physical units;
   H is symmetrised and a small ridge (1e-8 * mean|diag H|) is added.
2. Eigendecomposition; the eigendirections are ordered by |lambda| (negative curvature, if any, is treated
   like positive curvature: H is only a local quadratic description of a piecewise-smooth function).
3. Along each eigendirection the semi-axis s_k is found by bisection: the largest step for which the
   maximum relative change of J, RMS and r stays within TOL (starting from the quadratic estimate).
4. The N_ENS members are sampled uniformly in the resulting ellipsoid and mapped to parameter space.
   By default, D >= 0 is allowed. If --enforce-d-positive is set, any clipped D <= 0 is rejected (J = 1e6).
5. Members are evaluated; the fraction within tolerance is reported. NOTE: this fraction is partly set by the
   construction of the axes and is not an independent measure of robustness.

Usage:  python 04_hessian_ensemble.py [--params calibrated] [--tol 0.20] [--n-ens 500] [--enforce-d-positive] [--histograms | --no-histograms]
Output: results/ellipsoid_<name>.npz, figures/ellipsoid_ensemble.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.qmc import LatinHypercube

import v40lib as V

FD_FRAC, BS_STEPS, BS_MAX_S, SEED = 0.01, 24, 5.0, 42

LOWER = np.array([0.0, 1e-12, 1e-12, -np.inf, -np.inf, 1e-12, 1e-12, 1e-12, 1e-6, 1e-12])
UPPER = np.array([np.inf] * 8 + [1 - 1e-6, np.inf])


def clip(p):
    return np.clip(p, LOWER, UPPER)


def main():
    ap = V.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    ap.add_argument("--params", default="calibrated", help="parameter set name, ranking selector (e.g. top:1) or .npy file")
    ap.add_argument("--tol", type=float, default=0.20, help="metric tolerance (fraction)")
    ap.add_argument("--n-ens", type=int, default=500, help="ensemble size")
    ap.add_argument("--enforce-d-positive", action="store_true",
                    help="reject members with D <= 0 after clipping (returns J = 1e6, matching model_v40_ellipsoid_robustness.py)")
    ap.add_argument("--histograms", action=argparse.BooleanOptionalAction, default=True,
                    help="include the row of metric distribution histograms (use --no-histograms to omit)")
    args = ap.parse_args()
    V.prepare_dirs(args)

    if args.quick:
        args.n_ens = 30

    inp = V.load_inputs(args.ins, args.lr04)
    V.print_normalisation(inp)
    win = V.lr04_window(inp, -1000.0, args.mis11)
    base_obj = V.Objective(inp, win, dt=0.01)

    def obj(p):
        if args.enforce_d_positive and p[0] <= 0.0:
            return 1e6
        return base_obj(p)

    P = V.get_params(args.params)[:10]
    nd, tol = 10, args.tol

    def metrics(p):
        e = V.evaluate(V.simulate(p, inp, win, dt=0.01), win)
        return e["J"], e["rms_all"], e["r"], e

    def change(p, ref):
        p = clip(p)
        if obj(p) >= 1e5:
            return np.inf
        J, rms, r, _ = metrics(p)
        return max(abs(a - b) / abs(b) for a, b in zip((J, rms, r), ref) if b != 0)

    J0, rms0, r0, e0 = metrics(P)
    ref = (J0, rms0, r0)
    print(f"Solution ({args.params}): J={J0:.5f}  RMS={rms0:.4f}  r={r0:.4f}")
    if args.enforce_d_positive:
        print("  NOTE: Strict D > 0 rejection enabled (D <= 0 -> J = 1e6).")

    # 1. Hessian
    eps = np.abs(P) * FD_FRAC
    eps[eps == 0] = FD_FRAC
    H = np.zeros((nd, nd))
    for i in range(nd):
        pp, pm = P.copy(), P.copy()
        pp[i] += eps[i]; pm[i] -= eps[i]
        H[i, i] = (obj(clip(pp)) - 2 * J0 + obj(clip(pm))) / eps[i] ** 2
        for j in range(i + 1, nd):
            f = []
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                q = P.copy(); q[i] += si * eps[i]; q[j] += sj * eps[j]
                f.append(obj(clip(q)))
            H[i, j] = H[j, i] = (f[0] - f[1] - f[2] + f[3]) / (4 * eps[i] * eps[j])
    H = 0.5 * (H + H.T)
    H += 1e-8 * np.abs(np.diag(H)).mean() * np.eye(nd)

    # 2. Eigendecomposition
    lam, vec = np.linalg.eigh(H)
    order = np.argsort(np.abs(lam))[::-1]
    lam, vec = lam[order], vec[:, order]
    s_quad = np.array([np.sqrt(2 * tol * J0 / l) if l > 1e-20 else BS_MAX_S for l in lam])

    # 3. Bisection of the semi-axes
    s_cal = np.empty(nd)
    for k in range(nd):
        v = vec[:, k]
        lo, hi = 0.0, min(s_quad[k] * 3, BS_MAX_S)
        if max(change(P + hi * v, ref), change(P - hi * v, ref)) < tol:
            s_cal[k] = hi                     # flat direction
            continue
        for _ in range(BS_STEPS):
            mid = 0.5 * (lo + hi)
            if max(change(P + mid * v, ref), change(P - mid * v, ref)) > tol:
                hi = mid
            else:
                lo = mid
        s_cal[k] = lo

    print(f"\n{'mode':>4} {'lambda':>12} {'s_quad':>10} {'s_cal':>10}  dominant")
    for k in range(nd):
        print(f"{k:4d} {lam[k]:12.4g} {s_quad[k]:10.5f} {s_cal[k]:10.5f}  {V.NAMES[np.argmax(np.abs(vec[:, k]))]}")

    # 4. Sampling of the ellipsoid
    rng = np.random.default_rng(SEED)
    u = LatinHypercube(d=nd, seed=SEED).random(args.n_ens)
    g = rng.standard_normal((args.n_ens, nd))
    ball = (u[:, 0] ** (1 / nd))[:, None] * g / np.linalg.norm(g, axis=1, keepdims=True)
    members = (P[:, None] + (vec * s_cal) @ ball.T).T

    # 5. Evaluation
    keys = ("J", "rms_all", "r")
    M = np.full((args.n_ens, 3), np.nan)
    traces = np.full((args.n_ens, len(win.tg)), np.nan)
    passed = np.zeros(args.n_ens, bool)
    for j in range(args.n_ens):
        p = clip(members[j])
        members[j] = p
        if obj(p) >= 1e5:
            continue
        J, rms, r, _ = metrics(p)
        M[j] = (J, rms, r)
        traces[j] = V.simulate(p, inp, win, dt=0.01)
        passed[j] = max(abs(a - b) / abs(b) for a, b in zip(M[j], ref)) <= tol

    param_tag = args.params.replace("/", "_").replace(".", "_")
    print(f"\nMembers within {100*tol:.0f}% tolerance: {passed.sum()}/{args.n_ens} ({100*passed.mean():.1f}%)")
    for i, k in enumerate(keys):
        ok = ~np.isnan(M[:, i])
        pc = (M[ok, i] - ref[i]) / abs(ref[i]) * 100
        print(f"  {k:8s} best={ref[i]:.5f}  mean%Δ={pc.mean():+.2f}  std%Δ={pc.std():.2f}")

    np.savez(f"{args.outdir}/ellipsoid_{param_tag}.npz", P=P, H=H, eigvals=lam, eigvecs=vec, s_quad=s_quad,
             s_cal=s_cal, members=members, metrics=M, traces=traces, passed=passed, tg=win.tg, xo=win.xo)

    # 6. Figure generation (stile grafico di riferimento)
    n_rows = 5 if args.histograms else 4
    fig_h = 20 if args.histograms else 16.5
    fig = plt.figure(figsize=(18, fig_h))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.42, wspace=0.32)

    # Traces (Riga 0)
    ax_tr = fig.add_subplot(gs[0, :])
    obj_arr = M[:, 0]
    valid_m = ~np.isnan(obj_arr)
    obj_v = obj_arr[valid_m]
    traces_v = traces[valid_m]
    rank = np.argsort(np.argsort(obj_v)) / max(len(obj_v) - 1, 1)
    cmap = plt.cm.RdYlGn_r

    pass_v = passed[valid_m]
    for j in range(len(traces_v)):
        col = cmap(rank[j]) if pass_v[j] else "#aaaaaa"
        lw = 0.6 if pass_v[j] else 0.3
        alp = 0.30 if pass_v[j] else 0.15
        ax_tr.plot(win.tg, traces_v[j], color=col, lw=lw, alpha=alp)

    ax_tr.plot(win.tg, win.xo, "b-", lw=1.8, alpha=0.85, label="LR04 observed", zorder=10)
    ax_tr.plot(win.tg, V.simulate(P, inp, win, dt=0.01), "k-", lw=1.6,
               label=f"Best-fit (obj={J0:.4f})", zorder=11)

    n_pass = int(passed.sum())
    if n_pass >= 10:
        pass_traces = traces[passed]
        ax_tr.fill_between(win.tg,
                           np.nanpercentile(pass_traces, 5, axis=0),
                           np.nanpercentile(pass_traces, 95, axis=0),
                           alpha=0.22, color="#3cb44b", label=f"5–95% (passed, N={n_pass})")

    ax_tr.axvspan(*V.MIS6, color="orange", alpha=0.08)
    ax_tr.axvspan(*V.MIS11, color="lightblue", alpha=0.10)
    ax_tr.invert_yaxis()
    ax_tr.set_ylabel("Ice margin x")
    ax_tr.set_xlabel("Time (kyr BP)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_tr, fraction=0.012, pad=0.01).set_label("obj rank", fontsize=8)
    ax_tr.legend(fontsize=8, loc="upper left", ncol=4)
    ax_tr.set_title("Ensemble traces — grey: outside tolerance, coloured: within", fontsize=9)
    ax_tr.grid(True, alpha=0.22)

    # Metric distributions (Riga 1 - opzionale)
    if args.histograms:
        for col, (m_idx, label) in enumerate(
                [(0, r"Objective $\mathcal{J}$"), (1, "RMS"), (2, r"Pearson $r$")]):
            ax = fig.add_subplot(gs[1, col])
            arr = M[:, m_idx]
            ref_val = ref[m_idx]
            valid = arr[~np.isnan(arr)]
            pct = (valid - ref_val) / abs(ref_val) * 100.0
            bin_edges = np.histogram_bin_edges(valid, bins=35)

            ax.hist(arr[passed & ~np.isnan(arr)], bins=bin_edges,
                    color="#3cb44b", alpha=0.80, label="passed", edgecolor="white", lw=0.3, zorder=3)
            ax.hist(arr[~passed & ~np.isnan(arr)], bins=bin_edges,
                    color="#e6194b", alpha=0.60, label="failed tol.", edgecolor="white", lw=0.3, zorder=2)
            ax.axvline(ref_val, color="k", lw=1.8, ls="-", label=f"best={ref_val:.4f}", zorder=4)
            ax.axvline(ref_val * (1 + tol), color="orange", lw=1.1, ls="--", label=f"±{tol * 100:.0f}%", zorder=4)
            ax.axvline(ref_val * (1 - tol), color="orange", lw=1.1, ls="--", zorder=4)
            ax.set_title(f"{label}\nall: mean%Δ={pct.mean():+.1f}%  std={pct.std():.1f}%", fontsize=9)
            ax.set_xlabel(label, fontsize=8)
            ax.set_ylabel("count", fontsize=8)
            ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(0.9, 1.0))
            ax.grid(True, alpha=0.25)

    # Proiezioni 2D nello spazio dei parametri
    scatter_start_row = 2 if args.histograms else 1
    pairs_grid = [
        [(0, 1, "D", "U"), (2, 4, "S", "C2"), (7, 8, "N", "F0n")],
        [(1, 9, "U", "Mu"), (3, 4, "C1", "C2"), (7, 2, "N", "S")],
        [(5, 6, "B0", "B1"), (0, 8, "D", "F0n"), (2, 6, "S", "B1")]
    ]

    ok_f = ~np.isnan(M[:, 0])
    for row_idx, pairs in enumerate(pairs_grid):
        for col_idx, (i, j, ln, rn) in enumerate(pairs):
            ax = fig.add_subplot(gs[scatter_start_row + row_idx, col_idx])
            ax.scatter(members[ok_f & ~passed, i], members[ok_f & ~passed, j],
                       s=6, color="#e6194b", alpha=0.4, label="outside tol.")
            ax.scatter(members[ok_f & passed, i], members[ok_f & passed, j],
                       s=6, color="#3cb44b", alpha=0.6, label="within tol.")
            ax.scatter([P[i]], [P[j]], marker="*", s=200,
                       color="k", zorder=10, label="best-fit")
            ax.set_xlabel(ln, fontsize=9)
            ax.set_ylabel(rn, fontsize=9)
            ax.set_title(f"Parameter space: {ln} vs {rn}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25)

    out = f"{args.figdir}/ellipsoid_ensemble.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out)
    print("\nDone.")


if __name__ == "__main__":
    main()
