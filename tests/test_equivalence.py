"""Check that v40lib.simulate reproduces the original notebook function sim_cont (extracted to /tmp/orig_sim.py)
on the same data, and that the 2-Ma ramp matches the explicit linear ramps. Usage: python tests/test_equivalence.py <data dir>"""
import sys, importlib.util
import numpy as np
sys.path.insert(0, ".")
import v40lib as V

d = sys.argv[1] if len(sys.argv) > 1 else "tests/synthetic"
inp = V.load_inputs(f"{d}/21jugn.dat", f"{d}/LR04_stack.txt")
win = V.lr04_window(inp, -1000.0)
spec = importlib.util.spec_from_file_location("orig", "/tmp/orig_sim.py")
orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig)

mask = (inp.t_ins >= -1000) & (inp.t_ins <= 0)
for name in ("grant5", "secont"):
    p = V.get_params(name)
    a = V.simulate(p, inp, win, dt=0.1)
    b = orig.sim_cont(p, inp.t_ins[mask], inp.Fn[mask], win.tg, float(win.xo[0]), 0.1)
    print(f"{name}: max |lib - original| = {np.abs(a-b).max():.2e}")
    assert np.abs(a - b).max() < 1e-9

# 2-Ma ramp: explicit linear ramps (elapsed-time form used in the notebook)
w2 = V.lr04_window(inp, -2000.0)
p = V.get_params("grant5")
x_lib = V.simulate(p, inp, w2, dt=0.1, ramp=True)
U = p[1]; Q = (0.3 - 0.54) / 1200.0; W = (1 - 2/7) * U / 1200.0
def explicit(p, Ff, dt, tg):
    D, U, S, C1, C2, B0, B1, N, F0, Mu = p
    x = float(w2.xo[0]); xm = x; sp = 0; k = 0; xs = [x]
    for i in range(1, len(tg)):
        for _ in range(int(round(2/dt))):
            t = -2000 + dt*k; Fi = Ff[k]; k += 1
            if t < -800:
                xh = 0.54 + Q*(t+2000); Ue = 2/7*U + W*(t+2000)
            else:
                xh = 0.3; Ue = U
            c = C2 if Fi >= F0 else C1
            dd = D if x >= 1 else 0.0
            g = max(x-xh, 1e-6)
            u = np.exp(min(Ue*max(x-xm, 0)**Mu, 50)) if sp == 1 else 1.0
            xn = min(max(x + dt*(dd*Fi + u*(S*Fi/g**N + c*Fi*g) - B1/g**N - B0), xh+1e-6), 1.0)
            spn = 1 if xn > x else 0
            if sp == 0 and spn == 1: xm = x
            x, sp = xn, spn
        xs.append(x)
    return np.array(xs)
m = int(round(2/0.1)); nf = (len(w2.tg)-1)*m + 1
B0, B1, C2 = p[5], p[6], p[4]
Ff = V.fine_forcing(inp, -2000.0, 0.1, nf, 0.0)
x_ex = explicit(p, Ff, 0.1, w2.tg)
print(f"2-Ma ramp: max |lib - explicit| = {np.abs(x_lib-x_ex).max():.2e}")
assert np.abs(x_lib - x_ex).max() < 1e-9
print("all equivalence tests passed")
