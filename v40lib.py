"""
v40lib.py -- shared code for the one-equation (1-ODE) model of Late Pleistocene
glacial-interglacial cycles driven by 65N June-21 insolation.

Contents
--------
1. Constants, reference parameter sets and parameter bounds.
2. Input/output: insolation (Laskar et al. 2004), LR04 benthic d18O stack,
   Spratt & Lisiecki sea-level stack; normalisation to the model variable x.
3. The model: explicit-Euler integration of

       dx/dt = d*F + u*[ S*F/g^N + c*F*g ] - B1/g^N - B0 ,      g = x - xH

   with   c = C2 if F >= F0 else C1,        d = D if x >= 1 else 0,
          u = exp[ U (x - x_min)^Mu ] while x is increasing (retreat), else 1,
   x_min = value of x at the last glaciation->retreat reversal.
   The sign of dx/dt is evaluated at every integration step (no sub-stepping).
   x is clipped to [xH + 1e-6, 1].
4. Optional extension to 2 Ma: U and xH ramp linearly in time between
   RAMP_START and RAMP_END (values at RAMP_START: xH = 0.54, U = 2/7 U*).
5. Optional first-order low-pass filtering of the forcing (time constant tau).
6. Objective function J (Eq. 8 of the paper) and skill metrics.
7. Paillard (1998) reference models (3-state model M1 and ODE model M2/M3).

Conventions
-----------
* Time is in kyr relative to the present (negative = past), x in [xH, 1]
  (x = 1: fully deglaciated; smaller x: larger ice sheet).
* F is the 65N June-21 insolation normalised with the minimum and maximum of the
  FULL insolation file (so the same constants hold for every window).
* d18O is mapped to x with the 1st and 99th percentiles of the FULL LR04 stack
  (same reference for the 1-Myr and 2-Myr windows):
       x_obs = clip( 1 - 0.5 * (d18O - p01) / (p99 - p01), 0, 1 )
* Rate coefficients D, S, C1, C2, B0, B1 are in kyr^-1.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from numba import njit

# --------------------------------------------------------------------------- #
# 1. Constants, parameter sets, bounds
# --------------------------------------------------------------------------- #
XH_BASE = 0.30                 # northern boundary of the Hadley cell
XU, XL = 0.85, 0.58            # x_obs thresholds for the RMS_upper / RMS_lower subsets
AL = BE = 3.0                  # weights of RMS_upper and RMS_lower in J
GRID_DT = 1.0                  # kyr, grid of the observations and of the outputs
DT_CAL = 0.01                   # kyr, default Euler step
SPINUP = 100.0                 # kyr of forcing history used to spin up the low-pass filter
MIS6 = (-200.0, -140.0)        # kyr, excluded from the objective
MIS11 = (-430.0, -370.0)       # kyr, excluded from the objective if mis11_mode == 'exclude'

# 2-Ma extension (hand-set, not calibrated)
RAMP_START, RAMP_END = -2000.0, -800.0
RAMP_XH_START = 0.54           # xH at RAMP_START (xH reaches 0.30 at RAMP_END)
RAMP_U_FRAC = 2.0 / 7.0        # U at RAMP_START as a fraction of U (U reaches U at RAMP_END)

NAMES = ["D", "U", "S", "C1", "C2", "B0", "B1", "N", "F0n", "Mu"]

# Reference solution (best fit) and alternative solutions
P_GRANT5 = np.array([0.00011489, 2.08371614, 0.00652181, -0.00550636, 0.00848905,
                     0.01681127, 0.00018610, 1.14124925, 0.67307166, 0.26638365])

#P_GRANT5 = np.array([
#0.00013427,
#1.91203752,
#0.00628070,
#-0.00541382,
#0.00799330,
#0.01599546,
#0.00023071,
#1.13519006,
#0.61920160, #F0   = 516.59 W m-2
#0.22704891,
#])

# Second best-fit solution da integrazione precedente
P_SECONT = np.array([0.01941115, 4.03765095, 0.00331562, -0.00521534, 0.04204843,
                     0.01085264, 0.00022074, 1.38201055, 0.76657666, 0.12681253])

# Parametri fisici prescritti per esperimento alternativo con tau (MIS6-adaptable)
P_MIS6_ADAPTABLE = np.array([0.01183075, 5.05632731, 0.00372492, -0.00455484, 0.06329507,
                             0.01068121, 0.00020768, 1.25756532, 0.73007911, 0.10471276])

NAMED_PARAMS = {
    "grant5": P_GRANT5,
    "secont": P_SECONT,
    "mis6-adaptable": P_MIS6_ADAPTABLE,
    "mis6_adaptable": P_MIS6_ADAPTABLE,
}

# Search bounds for the Latin-hypercube stage of the calibration (D>=0 and C1<C2
# are enforced in the objective; the signs of C1 and C2 are left free).
BOUNDS = np.array([[0.0, 0.20],      # D
                   [1.0, 15.0],      # U
                   [5e-4, 0.015],    # S
                   [-0.020, 0.010],  # C1
                   [-0.010, 0.060],  # C2
                   [0.002, 0.030],   # B0
                   [1e-5, 0.003],    # B1
                   [1.0, 5.0],       # N
                   [0.25, 0.85],     # F0n
                   [0.05, 5.0]])     # Mu
TAU_BOUNDS = (0.0, 10.0)             # kyr

# Paillard (1998) published parameters: [tau_i, tau_g, tau_G, v_r, tau_F]
PAILLARD_M2_ORIG = np.array([10.0, 50.0, 50.0, 1.0, 25.0])
PAILLARD_M3_ORIG = np.array([5.0, 75.0, 75.0, 1.0, 28.0])   # values used in the paper's comparison;
PAILLARD_M3_VMAX_END = 1.1                                  # verify against Paillard (1998), Fig. 4
PAILLARD_VMAX_START = 0.35
PAILLARD_TREND_TOTAL = 6.0     # W m-2, total insolation trend over the ramp (3 W m-2 Myr-1 over 2 Myr)


def get_params(spec: str) -> np.ndarray:
    """
    Return a parameter vector from a name ('grant5', 'secont', 'mis6-adaptable', 'calibrated'),
    a Top-K selector (e.g. 'calibrated:10' or 'top:10' for the 10th best), or a .npy file.
    """
    # Gestione delle selezioni Top-K (es. "calibrated:10", "top:5", "top:10")
    if spec.startswith("calibrated:") or spec.startswith("top:"):
        rank = int(spec.split(":")[1])
        top_file = "results/calibration_mis11_exclude_top10.npy"
        if os.path.isfile(top_file):
            top_matrix = np.load(top_file)
            if 1 <= rank <= len(top_matrix):
                return top_matrix[rank - 1].copy()
            raise ValueError(f"rank {rank} out of bounds (file contains {len(top_matrix)} sets)")
        print(f"WARNING: {top_file} not found. Falling back to 'grant5'.")
        return P_GRANT5.copy()

    # Gestione di "calibrated" standard (1° classificato)
    if spec in ("calibrated", "default"):
        default_cal_file = "results/calibration_mis11_exclude.npy"
        if os.path.isfile(default_cal_file):
            return np.load(default_cal_file)
        print(f"WARNING: {default_cal_file} not found. Falling back to 'grant5'.")
        return P_GRANT5.copy()

    if spec in NAMED_PARAMS:
        return NAMED_PARAMS[spec].copy()

    if os.path.isfile(spec):
        return np.load(spec)

    raise ValueError(f"unknown parameter set '{spec}' (use calibrated, top:N, grant5, secont, mis6-adaptable or a .npy file)")

# --------------------------------------------------------------------------- #
# 2. Input / output
# --------------------------------------------------------------------------- #
def _need(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"input file not found: {path}  (see data/README.md)")
    return path


def read_insolation(path: str):
    """Two columns: time (kyr, negative = past) and insolation (W m-2). Fortran 'D' exponents allowed."""
    rows = []
    with open(_need(path)) as fh:
        for line in fh:
            line = re.sub(r"(\d)([Dd])([+-]?\d)", r"\1E\3", line)
            try:
                t, f = map(float, line.split()[:2])
                rows.append((t, f))
            except ValueError:
                continue
    a = np.array(rows)
    return a[np.argsort(a[:, 0])]


def read_lr04(path: str):
    """LR04 stack: age (kyr BP, positive) and d18O. Returns time (kyr, negative = past) sorted ascending."""
    ti, d = [], []
    with open(_need(path)) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            pp = line.split()
            if len(pp) < 2:
                continue
            try:
                ti.append(float(pp[0]))
                d.append(float(pp[1]))
            except ValueError:
                continue
    t = -np.array(ti)
    d = np.array(d)
    i = np.argsort(t)
    return t[i], d[i]


def read_sealevel(path: str, col: int = 5):
    """
    Spratt & Lisiecki (2025) sea-level stack (PANGAEA .tab file, tab separated, header in /* ... */).
    Column `col` (default 5 = SeaLev_longPC1, 0-798 ka). Returns time (kyr, negative = past) and
    -sea level, so that larger values mean more ice, like d18O.
    """
    ti, sl, in_comment = [], [], False
    with open(_need(path)) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("/*"):
                in_comment = True
            if in_comment:
                if "*/" in s:
                    in_comment = False
                continue
            pp = line.rstrip("\n").split("\t")
            if len(pp) <= col:
                continue
            try:
                a, v = float(pp[0]), float(pp[col])
            except ValueError:
                continue
            if np.isnan(v):
                continue
            ti.append(a)
            sl.append(v)
    t = -np.array(ti)
    d = -np.array(sl)
    i = np.argsort(t)
    return t[i], d[i]


@dataclass
class Inputs:
    """Insolation and LR04 with the (global) normalisation constants."""
    t_ins: np.ndarray
    F_raw: np.ndarray     # W m-2
    Fn: np.ndarray        # normalised insolation
    F_min: float
    F_max: float
    t_lr: np.ndarray
    d18: np.ndarray
    d18_lo: float         # 1st percentile of the full LR04 stack (x = 1)
    d18_hi: float         # 99th percentile of the full LR04 stack (x = 0.5)

    def F_to_W(self, Fn):
        return np.asarray(Fn) * (self.F_max - self.F_min) + self.F_min


def load_inputs(ins_file: str, lr04_file: str) -> Inputs:
    ins = read_insolation(ins_file)
    t_ins, F_raw = ins[:, 0], ins[:, 1]
    F_min, F_max = float(F_raw.min()), float(F_raw.max())
    t_lr, d18 = read_lr04(lr04_file)
    return Inputs(t_ins, F_raw, (F_raw - F_min) / (F_max - F_min), F_min, F_max,
                  t_lr, d18, float(np.percentile(d18, 1.0)), float(np.percentile(d18, 99.0)))


def x_from_d18(d, lo, hi):
    """Un-clipped mapping of d18O (or -sea level) to the model variable (Eq. 7 of the paper)."""
    return 1.0 - (d - lo) / (hi - lo) * 0.5


@dataclass
class Window:
    """Time grid, observations and masks of an integration/evaluation window."""
    t_start: float
    tg: np.ndarray        # 2-kyr grid from t_start to 0
    xo: np.ndarray        # observed x on tg (NaN where no data)
    obs: np.ndarray       # data available
    mis6: np.ndarray
    mis11: np.ndarray
    use: np.ndarray       # points entering the objective J (all three terms)
    std: float            # standard deviation of xo (used by the plausibility filter)


def make_window(t_start, t_series, d_series, d_lo, d_hi, mis11_mode="exclude") -> Window:
    """
    Build a window from an observed series. MIS 6 is always excluded from J;
    MIS 11 is excluded if mis11_mode == 'exclude' and used otherwise ('include').
    The exclusion applies to RMS_calib, RMS_upper and RMS_lower alike.
    """
    if mis11_mode not in ("exclude", "include"):
        raise ValueError("mis11_mode must be 'exclude' or 'include'")
    tg = np.arange(float(t_start), 1.0, GRID_DT)
    inside = (t_series >= t_start) & (t_series <= 0)
    xo = np.clip(np.interp(tg, t_series[inside], x_from_d18(d_series[inside], d_lo, d_hi),
                           left=np.nan), 0.0, 1.0)
    obs = ~np.isnan(xo)
    mis6 = (tg >= MIS6[0]) & (tg <= MIS6[1])
    mis11 = (tg >= MIS11[0]) & (tg <= MIS11[1])
    excl = mis6 | (mis11 if mis11_mode == "exclude" else False)
    return Window(float(t_start), tg, xo, obs, mis6, mis11, obs & ~excl, float(np.nanstd(xo)))


def lr04_window(inp: Inputs, t_start: float, mis11_mode="exclude") -> Window:
    return make_window(t_start, inp.t_lr, inp.d18, inp.d18_lo, inp.d18_hi, mis11_mode)


# --------------------------------------------------------------------------- #
# 3. Forcing and model
# --------------------------------------------------------------------------- #
@njit(cache=True)
def lowpass(F, dt, tau):
    """dFs/dt = (F - Fs)/tau, exact exponential update (stable for any tau/dt)."""
    out = np.empty_like(F)
    fs = F[0]
    if tau <= 0.0:
        for k in range(len(F)):
            out[k] = F[k]
        return out
    a = 1.0 - np.exp(-dt / tau)
    for k in range(len(F)):
        fs += a * (F[k] - fs)
        out[k] = fs
    return out


_RAW_FORCING = {}


def fine_forcing(inp: Inputs, t_start: float, dt: float, n_fine: int, tau: float = 0.0):
    """Normalised insolation on the integration grid t_start + dt*k (k = 0..n_fine-1)."""
    n_sp = int(round(SPINUP / dt))
    key = (id(inp), float(t_start), float(dt), int(n_fine))
    raw = _RAW_FORCING.get(key)
    if raw is None:
        t_f = (t_start - SPINUP) + dt * np.arange(n_sp + n_fine)
        raw = np.interp(t_f, inp.t_ins, inp.Fn)
        _RAW_FORCING[key] = raw
    F = lowpass(raw, dt, float(tau)) if tau > 0 else raw
    return F[n_sp:]


@njit(cache=True)
def _integrate(p, Ff, m, nout, x0, dt, t0, ramp_end, xh_start, u_frac):
    """
    Explicit Euler, m steps of length dt between consecutive outputs.
    Returns x and the retreat flag (1 if x increased in the last step) at the outputs.
    """
    D = p[0]; U = p[1]; S = p[2]; C1 = p[3]; C2 = p[4]
    B0 = p[5]; B1 = p[6]; N = p[7]; F0 = p[8]; Mu = p[9]
    xs = np.empty(nout)
    sps = np.zeros(nout, np.int8)
    xs[0] = x0
    x = x0
    xm = x0          # x_min
    sp = 0           # 1 = retreat (x increasing), 0 = glaciation
    k = 0
    for i in range(1, nout):
        for _ in range(m):
            t = t0 + dt * k
            Fi = Ff[k]
            k += 1
            if t < ramp_end:                           # linear ramps of xH and U (2-Ma extension)
                frac = (t - t0) / (ramp_end - t0)
                xh = xh_start + (XH_BASE - xh_start) * frac
                Ue = U * (u_frac + (1.0 - u_frac) * frac)
            else:
                xh = XH_BASE
                Ue = U
            c = C2 if Fi >= F0 else C1
            d = D if x >= 1.0 else 0.0
            g = max(x - xh, 1e-6)
            u = np.exp(min(Ue * max(x - xm, 0.0) ** Mu, 50.0)) if sp == 1 else 1.0
            xn = x + dt * (d * Fi + u * (S * Fi / g ** N + c * Fi * g) - B1 / g ** N - B0)
            xn = min(max(xn, xh + 1e-6), 1.0)
            spn = 1 if xn > x else 0
            if sp == 0 and spn == 1:
                xm = x                                  # reversal: x_min = last glaciation value
            x = xn
            sp = spn
        xs[i] = x
        sps[i] = sp
    return xs, sps


def _split(p, tau):
    p = np.asarray(p, float)
    if tau is None:
        tau = float(p[10]) if p.size > 10 else 0.0
    return p[:10], float(tau)


def _ramp_args(ramp: bool, win: Window):
    if ramp and win.t_start < RAMP_END:
        return RAMP_END, RAMP_XH_START, RAMP_U_FRAC
    return win.t_start, XH_BASE, 1.0          # ramp disabled (t < t0 is never true)


def simulate(p, inp: Inputs, win: Window, dt: float = DT_CAL, tau=None, ramp=False, x0=None):
    """Integrate the model; returns x on win.tg (2-kyr grid)."""
    p10, tau = _split(p, tau)
    m = int(round(GRID_DT / dt))
    if abs(m * dt - GRID_DT) > 1e-9:
        raise ValueError("dt must divide the 2-kyr grid")
    n_fine = (len(win.tg) - 1) * m + 1
    Ff = fine_forcing(inp, win.t_start, dt, n_fine, tau)
    x0 = float(win.xo[0] if x0 is None else x0)
    re_, xh0, uf = _ramp_args(ramp, win)
    return _integrate(p10, Ff, m, len(win.tg), x0, dt, win.t_start, re_, xh0, uf)[0]


def simulate_fine(p, inp: Inputs, win: Window, dt: float = DT_CAL, tau=None, ramp=False, x0=None):
    """Same as simulate() but returns the full fine-step trajectory: (t, x, retreat flag, F)."""
    p10, tau = _split(p, tau)
    m = int(round(GRID_DT / dt))
    n_fine = (len(win.tg) - 1) * m + 1
    Ff = fine_forcing(inp, win.t_start, dt, n_fine, tau)
    x0 = float(win.xo[0] if x0 is None else x0)
    re_, xh0, uf = _ramp_args(ramp, win)
    xs, sps = _integrate(p10, Ff, 1, n_fine, x0, dt, win.t_start, re_, xh0, uf)
    return win.t_start + dt * np.arange(n_fine), xs, sps, Ff


# --------------------------------------------------------------------------- #
# 4. Objective and metrics
# --------------------------------------------------------------------------- #
def _rms(xs, xo, mask):
    return float(np.sqrt(np.mean((xs[mask] - xo[mask]) ** 2))) if mask.any() else float("nan")


def terms(xs, win: Window):
    """RMS_calib, RMS_upper, RMS_lower on the points used for calibration (win.use).
    Each term is normalised by the number of points of its own subset."""
    xoc = win.xo[win.use]
    xc = xs[win.use]
    mu, ml = xoc > XU, xoc < XL
    rms = np.sqrt(np.mean((xc - xoc) ** 2))
    ru = np.sqrt(np.mean((xc[mu] - xoc[mu]) ** 2)) if mu.any() else 0.0
    rl = np.sqrt(np.mean((xc[ml] - xoc[ml]) ** 2)) if ml.any() else 0.0
    return float(rms), float(ru), float(rl)


def evaluate(xs, win: Window) -> dict:
    """Objective and skill metrics. rms_all and r use all points with data (including MIS 6/11)."""
    rc, ru, rl = terms(xs, win)
    o = win.obs
    return dict(J=rc + AL * ru + BE * rl, rms_used=rc, ru=ru, rl=rl,
                rms_all=_rms(xs, win.xo, o),
                r=float(np.corrcoef(xs[o], win.xo[o])[0, 1]),
                rms_mis6=_rms(xs, win.xo, win.mis6 & o),
                rms_mis11=_rms(xs, win.xo, win.mis11 & o))


def subwindow_metrics(xs, win: Window, extra: dict | None = None) -> dict:
    """RMS and r of xs on named sub-windows (all / 0-1 Ma / 1-2 Ma / pre- and post-MPT / MIS 6 / MIS 11)."""
    tg, o = win.tg, win.obs
    masks = {"all": o, "0-1 Ma": o & (tg >= -1000), "1-2 Ma": o & (tg < -1000),
             "post-MPT (>-800 ka)": o & (tg >= RAMP_END), "pre-MPT (<-800 ka)": o & (tg < RAMP_END),
             "J set (excl. MIS 6, and MIS 11 if excluded)": win.use, "MIS 6": o & win.mis6, "MIS 11": o & win.mis11}
    if extra:
        masks.update(extra)
    out = {}
    for k, mk in masks.items():
        if mk.sum() > 2:
            r = (float(np.corrcoef(xs[mk], win.xo[mk])[0, 1])
                 if xs[mk].std() > 0 and win.xo[mk].std() > 0 else float("nan"))
            out[k] = (_rms(xs, win.xo, mk), r)
    return out


class Objective:
    """Callable J(p) used by the optimisers. Returns 1e6 for invalid or implausible solutions."""

    def __init__(self, inp: Inputs, win: Window, dt: float = DT_CAL, with_tau=False, ramp=False):
        self.inp, self.win, self.dt, self.with_tau, self.ramp = inp, win, dt, with_tau, ramp

    def valid(self, p) -> bool:
        D, U, S, C1, C2, B0, B1, N, F0, Mu = p[:10]
        if D < 0 or C1 >= C2 or min(U, S, B0, B1, N, Mu) <= 0 or not (0.0 < F0 < 1.0):
            return False
        return not (self.with_tau and p[10] < 0)

    def __call__(self, p) -> float:
        p = np.asarray(p, float)
        if not self.valid(p):
            return 1e6
        xs = simulate(p, self.inp, self.win, self.dt, ramp=self.ramp)
        if not np.all(np.isfinite(xs)):
            return 1e6
        if xs.std() < 0.45 * self.win.std or xs.max() - xs.min() < 0.18 or xs.min() > 0.72:
            return 1e6        # plausibility filters: collapsed, flat or never-glaciated solutions
        rc, ru, rl = terms(xs, self.win)
        return rc + AL * ru + BE * rl


# --------------------------------------------------------------------------- #
# 5. Paillard (1998) reference models
# --------------------------------------------------------------------------- #
def _trunc(x, a=1.0):
    return 0.5 * (x + np.sqrt(4.0 * a ** 2 + x ** 2))


def paillard_forcing(inp: Inputs, t_start: float, ramp_end=None):
    """
    1-kyr forcing for the Paillard models on t_start..+10 kyr.
    Fz: insolation standardised to zero mean, unit variance (thresholds are applied to Fz);
    Ft: smoothed truncation of Fz (a = 1), re-standardised (enters the ODE).
    If ramp_end is given, a linear trend of PAILLARD_TREND_TOTAL W m-2 is added between t_start and
    ramp_end (ramp_end = 0: trend over the whole domain, as in Paillard 1998; ramp_end = -800: our ramp window).
    """
    t = np.arange(float(t_start), 11.0, 1.0)
    F = np.interp(t, inp.t_ins, inp.F_raw)
    if ramp_end is not None:
        F = F + PAILLARD_TREND_TOTAL * np.clip((t - t_start) / (ramp_end - t_start), 0.0, 1.0)
    Fz = (F - F.mean()) / F.std()
    Ft = _trunc(Fz)
    Ft = (Ft - Ft.mean()) / Ft.std()
    return t, Fz, Ft


def vmax_curve(t, t_start, ramp_end, vmax_end):
    frac = np.clip((t - t_start) / (ramp_end - t_start), 0.0, 1.0)
    return PAILLARD_VMAX_START + (vmax_end - PAILLARD_VMAX_START) * frac


@njit(cache=True)
def _paillard_core(Fz, Ft, vmax, par, v0, state0, clip_hi):
    """dv/dt = (v_R - v)/tau_R - F/tau_F ; states 0=i, 1=g, 2=G; i->g: Fz < -0.75; g->G: v > vmax; G->i: Fz > 0."""
    tau_i, tau_g, tau_G, vr, tau_F = par[0], par[1], par[2], par[3], par[4]
    n = len(Fz)
    v = np.empty(n)
    v[0] = v0
    s = state0
    for k in range(1, n):
        vm = vmax[k]
        vref = 0.0 if s == 0 else vm * vr
        tr = tau_i if s == 0 else (tau_g if s == 1 else tau_G)
        vn = v[k - 1] + (vref - v[k - 1]) / tr - Ft[k - 1] / tau_F
        hi = clip_hi if clip_hi > 0 else vm + 0.3
        vn = min(max(vn, -0.3), hi)
        if s == 0 and Fz[k] < -0.75:
            s = 1
        elif s == 1 and vn > vm:
            s = 2
        elif s == 2 and Fz[k] > 0.0:
            s = 0
        v[k] = vn
    return v


def _v_to_x(v):
    return 1.0 - np.clip(v, 0.0, 1.0) * 0.5


def _initial_state(x0):
    v0 = (1.0 - x0) / 0.5
    return v0, (0 if v0 < 0.2 else (2 if v0 >= 1.0 else 1))


def paillard_m2(par, inp: Inputs, win: Window, x0=None):
    """Paillard (1998) ODE model with constant parameters (v_max = 1), started from the observed x."""
    t, Fz, Ft = paillard_forcing(inp, win.t_start)
    v0, s0 = _initial_state(float(win.xo[0] if x0 is None else x0))
    v = _paillard_core(Fz, Ft, np.ones_like(t), np.asarray(par, float), v0, s0, 1.8)
    return np.interp(win.tg, t, _v_to_x(v))


def paillard_m3(par, inp: Inputs, win: Window, ramp_end: float, vmax_end: float):
    """
    Paillard model with trends in insolation and v_max (2-Ma runs). ramp_end = 0 reproduces a trend over
    the whole domain (Paillard 1998); ramp_end = -800 compresses the same trends between -2000 and -800 ka.
    Started from the interglacial state with v = 0.
    """
    t, Fz, Ft = paillard_forcing(inp, win.t_start, ramp_end)
    vmax = vmax_curve(t, win.t_start, ramp_end, vmax_end)
    v0, s0 = _initial_state(float(win.xo[0]))
    v = _paillard_core(Fz, Ft, vmax, np.asarray(par, float), v0, s0, -1.0)
    return np.interp(win.tg, t, _v_to_x(v))


def paillard_m1(inp: Inputs, win: Window, x0=None):
    """Discrete 3-state model of Paillard (1998), x = 1 / 0.75 / 0.5 for states i / g / G."""
    t, Fz, _ = paillard_forcing(inp, win.t_start)
    v0, state = _initial_state(float(win.xo[0] if x0 is None else x0))
    g_dur, last_peak = 0.0, -np.inf
    out = np.empty(len(t))
    out[0] = state
    for k in range(1, len(t)):
        Fc, Fp = Fz[k], Fz[k - 1]
        if state == 1:
            g_dur += 1.0
            if k >= 2 and Fz[k - 2] < Fp > Fc:
                last_peak = Fp
        if state == 0 and Fc < -0.75:
            state, g_dur, last_peak = 1, 0.0, -np.inf
        elif state == 1 and g_dur >= 33.0 and last_peak < 1.0 and Fc < 0.0:
            state = 2
        elif state == 2 and Fc > 0.0:
            state = 0
        out[k] = state
    x = np.array([{0: 1.0, 1: 0.75, 2: 0.5}[int(s)] for s in out])
    return np.interp(win.tg, t, x)


# --------------------------------------------------------------------------- #
# 6. Command-line helpers shared by the scripts
# --------------------------------------------------------------------------- #
def add_common_args(ap):
    ap.add_argument("--ins", default="data/21jugn.dat", help="insolation file")
    ap.add_argument("--lr04", default="data/LR04_stack.txt", help="LR04 stack")
    ap.add_argument("--sealevel", default="data/Sea-level-stack.tab", help="sea-level stack (PANGAEA .tab)")
    ap.add_argument("--mis11", default="exclude", choices=["exclude", "include"],
                    help="exclude MIS 11 from the objective (test interval) or include it")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--quick", action="store_true", help="tiny settings, for smoke tests only")
    return ap


def prepare_dirs(args):
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)


def print_normalisation(inp: Inputs):
    print(f"Insolation normalisation (full file, {inp.t_ins[0]:.0f} to {inp.t_ins[-1]:.0f} kyr): "
          f"F_min={inp.F_min:.2f}  F_max={inp.F_max:.2f} W m-2")
    print(f"d18O reference (full LR04 stack): 1st pct={inp.d18_lo:.4f}  99th pct={inp.d18_hi:.4f}  [permil]")
