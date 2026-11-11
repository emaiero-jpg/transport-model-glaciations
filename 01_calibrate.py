"""
01_calibrate.py -- calibration of the 10 parameters on LR04 over the last 1 Myr.

Strategy (as in the paper): Latin-hypercube sampling of the bounds, the best LHS points plus the two
reference solutions (and random perturbations of them) are used as starting points of Nelder-Mead
(two successive passes each). The best solution is saved.
Objective: J = RMS_calib + 3 RMS_upper + 3 RMS_lower, MIS 6 always excluded, MIS 11 excluded
(default, MIS 11 is then a test interval) or included (--mis11 include).

Usage:  python 01_calibrate.py [--mis11 exclude|include] [--jobs N] [--top-k N]
Output: results/calibration_mis11_<mode>.npy  and  .txt  (plus _topK.npy if top-k > 1)
"""
import argparse
import multiprocessing as mp

import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

import v40lib as V

_OBJ = None


def _init(ins, lr04, mode):
    global _OBJ
    inp = V.load_inputs(ins, lr04)
    _OBJ = V.Objective(inp, V.lr04_window(inp, -1000.0, mode))


def _eval(p):
    return _OBJ(p)


def _optimise(p0):
    best_p, best_J = np.array(p0, float), _OBJ(p0)
    for maxiter, tol in ((15000, 1e-8), (25000, 1e-10)):
        r = minimize(_OBJ, best_p, method="Nelder-Mead", options=dict(maxiter=maxiter, xatol=tol, fatol=tol))
        if r.fun < best_J:
            best_J, best_p = r.fun, r.x.copy()
    return best_J, best_p


def main():
    ap = V.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[1]))
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (0 = all cores)")
    ap.add_argument("--n-lhs", type=int, default=10000)
    ap.add_argument("--n-top", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=1, help="number of top local minima to print and save in output (default: 1)")
    args = ap.parse_args()
    V.prepare_dirs(args)

    if args.quick:
        args.n_lhs, args.n_top = 60, 3

    jobs = args.jobs or mp.cpu_count()
    inp = V.load_inputs(args.ins, args.lr04)
    V.print_normalisation(inp)

    lhs = V.BOUNDS[:, 0] + LatinHypercube(d=10, seed=42).random(args.n_lhs) * (V.BOUNDS[:, 1] - V.BOUNDS[:, 0])
    rng = np.random.default_rng(99)
    ref = [V.P_GRANT5, V.P_SECONT]
    n_pert = 2 if args.quick else 25

    with mp.Pool(jobs, initializer=_init, initargs=(args.ins, args.lr04, args.mis11)) as pool:
        print(f"1. evaluating {args.n_lhs} LHS points ...", flush=True)
        scores = np.array(pool.map(_eval, list(lhs), chunksize=50))
        starts = list(lhs[np.argsort(scores)[:args.n_top]]) + ref
        for base in ref:
            for _ in range(n_pert):
                starts.append(np.clip(base * (0.8 + 0.4 * rng.random(10)), V.BOUNDS[:, 0], V.BOUNDS[:, 1]))
        print(f"2. Nelder-Mead from {len(starts)} starting points ...", flush=True)
        results = sorted(pool.map(_optimise, starts), key=lambda r: r[0])

    J, p = results[0]
    print(f"\nBest J = {J:.5f}  (MIS 11 {'excluded' if args.mis11 == 'exclude' else 'included'})")
    for n, v in zip(V.NAMES, p):
        print(f"  {n:4s} = {v:.8f}")
    print(f"  F0   = {inp.F_to_W(p[8]):.2f} W m-2")
    if p[0] < 1e-6:
        print("  NOTE: D is at its lower bound (0)")

    base_path = f"{args.outdir}/calibration_mis11_{args.mis11}"

    # 1. Salvataggio sempre della soluzione migliore singola (.npy)
    np.save(base_path + ".npy", p)

    # 2. Gestione in base al parametro --top-k
    top_k_count = min(args.top_k, len(results))

    if top_k_count > 1:
        # Salvataggio della matrice 2D con i Top-K risultati
        top_p_matrix = np.array([res[1] for res in results[:top_k_count]])
        top_file = f"{base_path}_top{top_k_count}.npy"
        np.save(top_file, top_p_matrix)

        # Scrittura del file .txt con sezioni per Best Solution e Top-K
        with open(base_path + ".txt", "w") as fh:
            fh.write("=== BEST SOLUTION ===\n")
            fh.write(f"J = {J:.6f}\n" + "\n".join(f"{n} = {v:.8f}" for n, v in zip(V.NAMES, p)) + "\n\n")

            fh.write(f"=== TOP {top_k_count} LOCAL MINIMA ===\n")
            for rank, (J_val, p_val) in enumerate(results[:top_k_count], start=1):
                fh.write(f"\n--- RANK {rank} | J = {J_val:.6f} | F0 = {inp.F_to_W(p_val[8]):.2f} W m-2 ---\n")
                fh.write("\n".join(f"  {n:4s} = {v:.8f}" for n, v in zip(V.NAMES, p_val)) + "\n")

        print(f"saved {base_path}.npy, {top_file} and {base_path}.txt")
    else:
        # Scrittura standard del solo file .txt per il miglior punto
        with open(base_path + ".txt", "w") as fh:
            fh.write(f"J = {J:.6f}\n" + "\n".join(f"{n} = {v:.8f}" for n, v in zip(V.NAMES, p)) + "\n")

        print(f"saved {base_path}.npy and {base_path}.txt")


if __name__ == "__main__":
    main()
