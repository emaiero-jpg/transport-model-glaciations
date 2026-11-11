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
   By default, D is clipped to D >= 0. If --enforce-d-positive is set, D is constrained to be strictly positive (D >= 1e-6).
5. Members are evaluated; the fraction within tolerance is reported. NOTE: this fraction is partly set by the
   construction of the axes and is not an independent measure of robustness.

Usage:  python 04_hessian_ensemble.py [--params calibrated] [--tol 0.20] [--n-ens 500] [--enforce-d-positive]
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

# Limiti standard per clipping (D >= 0)
LOWER_DEFAULT = np.array([0.0, 1e-12, 1e-12, -np.inf, -np.inf, 1e-12, 1e-12, 1e-12, 1e-6, 1e-12])
# Limiti quando si impone D strettamente positivo (D >= 1e-6)
LOWER_STRICT = np.array([1e-6, 1e-12, 1e-12, -np.inf, -np.inf, 1e-12, 1e-12, 1e-12, 1e-6, 1e-12])
UPPER = np.array([np.inf] * 8 + [1 - 1e-6, np.inf])


def clip(p, lower_bounds):
    return np.clip(p, lower_bounds, UPPER)


def main():
    ap = V.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    ap.add_argument("--params", default="calibrated", help="parameter set name, ranking selector (e.g. top:1) or .npy file")
    ap.add_argument("--tol", type=float, default=0.20, help="metric tolerance (fraction)")
    ap.add_argument("--n-ens", type=int, default=500, help="ensemble size")
    ap.add_argument("--enforce-d-positive", action="store_true", help="strictly enforce D >= 1e-6 during clipping and evaluation")
    args = ap.parse_args()
    V.prepare_dirs(args)

    if args.quick:
        args.n_ens = 30

    #lower_bounds = LOWER_STRICT if args.enforce_d-positive else LOWER_DEFAULT
    lower_bounds = LOWER_STRICT if args.enforce_d_positive else LOWER_DEFAULT

    inp = V.load_inputs(args.ins, args.lr04)
    V.print_normalisation(inp)
    win = V.lr04_window(inp, -1000.0, args.mis11)
    obj = V.Objective(inp, win, dt=0.01)

    P = V.get_params(args.params)[:10]
    nd, tol = 10, args.tol

    def metrics(p):
        e = V.evaluate(V.simulate(p, inp, win, dt=0.01), win)
        return e["J"], e["rms_all"], e["r"], e

    def change(p, ref):
        p = clip(p, lower_bounds)
        if obj(p) >= 1e5:
            return np.inf
        J, rms, r, _ = metrics(p)
        return max(abs(a - b) / abs(b) for a, b in zip((J, rms, r), ref) if b != 0)

    J0, rms0, r0, e0 = metrics(P)
    ref = (J0, rms0, r0)
    print(f"Solution ({args.params}): J={J0:.5f}  RMS={rms0:.4f}  r={r0:.4f}")
    #if args.enforce-d-positive:
    #    print("  NOTE: Strict D > 0 constraint enabled.")
    if args.enforce_d_positive:
        print("  NOTE: Strict D > 0 constraint enabled.")
    
    # 1. Hessian
    eps = np.abs(P) * FD_FRAC
    eps[eps == 0] = FD_FRAC
    H = np.zeros((nd, nd))
    for i in range(nd):
        pp, pm = P.copy(), P.copy()
        pp[i] += eps[i]; pm[i] -= eps[i]
        H[i, i] = (obj(clip(pp, lower_bounds)) - 2 * J0 + obj(clip(pm, lower_bounds))) / eps[i] ** 2
        for j in range(i + 1, nd):
            f = []
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                q = P.copy(); q[i] += si * eps[i]; q[j] += sj * eps[j]
                f.append(obj(clip(q, lower_bounds)))
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
        p = clip(members[j], lower_bounds)
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

    # 6. Figure generation
    fig = plt.figure(figsize=(18, 17))
    gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.3, height_ratios=[1.2, 0.9, 0.9, 0.9])
    
    ax = fig.add_subplot(gs[0, :])
    okm = ~np.isnan(M[:, 0])
    rank = np.argsort(np.argsort(M[okm, 0])) / max(okm.sum() - 1, 1)
    for tr, pas, rk in zip(traces[okm], passed[okm], rank):
        ax.plot(win.tg, tr, color=plt.cm.RdYlGn_r(rk) if pas else "#aaaaaa", lw=0.5, alpha=0.3 if pas else 0.12)
    ax.plot(win.tg, win.xo, "b-", lw=1.6, label="LR04 (normalised)")
    ax.plot(win.tg, V.simulate(P, inp, win, dt=0.01), "k-", lw=1.5, label=f"solution J={J0:.4f}")
    ax.axvspan(*V.MIS6, color="orange", alpha=0.1)
    ax.axvspan(*V.MIS11, color="lightblue", alpha=0.1)
    ax.invert_yaxis()
    ax.set_xlabel("Time (kyr)")
    ax.set_ylabel("Ice margin x")
    ax.legend(fontsize=8)

    pairs = [(0, 1), (2, 4), (7, 8), (1, 9), (3, 4), (7, 2), (5, 6), (0, 8), (2, 6)]
    for n, (i, j) in enumerate(pairs):
        a = fig.add_subplot(gs[1 + n // 3, n % 3])
        a.scatter(members[okm & ~passed, i], members[okm & ~passed, j], s=5, c="#e6194b", alpha=0.4)
        a.scatter(members[passed, i], members[passed, j], s=5, c="#3cb44b", alpha=0.6)
        a.scatter([P[i]], [P[j]], marker="*", s=180, c="k", zorder=10)
        a.set_xlabel(V.NAMES[i])
        a.set_ylabel(V.NAMES[j])

    out = f"{args.figdir}/ellipsoid_ensemble.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
