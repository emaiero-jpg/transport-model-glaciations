"""Create SYNTHETIC input files (same format as the real ones) for smoke-testing the scripts.
The numbers are meaningless: use only to check that the code runs.
Usage: python tests/make_synthetic_data.py <outdir>"""
import os, sys
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "tests/synthetic"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)

t = np.arange(-5000.0, 21.0, 1.0)
F = 490 + 45*np.sin(2*np.pi*t/23) + 20*np.sin(2*np.pi*t/41) + 10*np.sin(2*np.pi*t/100)
with open(f"{out}/21jugn.dat", "w") as f:
    for a, b in zip(t, F):
        f.write(f"{a:.1f} {b:.5f}\n")

age = np.arange(0.0, 5300.0, 2.0)
d = (3.9 + 0.55*np.sin(2*np.pi*age/100) + 0.25*np.sin(2*np.pi*age/41)
     + 0.1*np.sin(2*np.pi*age/23) + 0.03*rng.standard_normal(len(age)))
with open(f"{out}/LR04_stack.txt", "w") as f:
    f.write("# synthetic LR04-like stack\n")
    for a, b in zip(age, d):
        f.write(f"{a:.1f} {b:.4f}\n")

with open(f"{out}/Sea-level-stack.tab", "w") as f:
    f.write("/* synthetic sea-level stack\nDATA DESCRIPTION: ... */\n")
    f.write("Age\tc1\tc2\tc3\tc4\tSeaLev\n")
    for a in np.arange(0.0, 799.0, 1.0):
        sl = -60 - 50*np.sin(2*np.pi*a/100) - 15*np.sin(2*np.pi*a/41)
        f.write(f"{a}\t0\t0\t0\t0\t{sl:.3f}\n")
print("synthetic data written to", out)
