"""
Run meep's own harmonic-inversion algorithm -- the thing behind mp.Harminv,
which is what found those 3 clean resonances in ring.py -- directly on
FDTDX's saved ring-down data. Not reimplementing it, reusing the real one.

Built from meep's actual source (mp.py_do_harminv, called inside
Simulation._analyze_harminv in python/simulation.py), not from memory. My
sandbox can't install meep at all (conda-forge is blocked there), so this
got verified together, same as ring.py was -- two real bugs found and fixed
by reading the actual SWIG source (python/meep.i) after it crashed:

  1. rel_err_thresh needs meep's own sentinel mp.inf (a plain 1e20, per
     meep.i), not true IEEE infinity -- harmless on its own, see #2.
  2. The actual crash (std::bad_array_new_length): py_do_harminv's C++ side
     does `Py_ssize_t n = PyList_Size(vals)`, which requires `vals` to be a
     genuine Python list. A numpy array fails that silently (PyList_Size
     returns -1, the C++ wrapper never checks), and `new complex<double>[-1]`
     is exactly what throws. Fix: pass Ex.tolist(), not Ex.
"""

import numpy as np
import meep as mp

C = 299792458.0  # m/s. Real physical units throughout -- NOT meep's normalized
# a/c=1 convention. py_do_harminv is being called directly here, bypassing
# Simulation/fields (which is what normally handles that conversion), so as
# long as dt and the frequency bounds are consistent with each other, the
# actual unit system is arbitrary. Using real seconds/Hz means the results
# come out directly interpretable, no unnormalizing needed afterward.

d = np.load("ringdown_data.npz")
Ex = np.asarray(d["Ex"], dtype=np.float64)
dt = float(d["dt"])
center_wavelength = float(d["center_wavelength"])
spectral_width_hz = float(d["spectral_width_hz"])

f_center = C / center_wavelength
fmin = f_center - spectral_width_hz
fmax = f_center + spectral_width_hz
print(f"searching {fmin / 1e12:.1f}-{fmax / 1e12:.1f} THz  ({C / fmax * 1e9:.0f}-{C / fmin * 1e9:.0f} nm)")

bands = mp.py_do_harminv(
    Ex.tolist(),  # must be a plain Python list -- see module docstring, bug #2
    dt,
    fmin,
    fmax,
    100,  # maxbands
    1.1,  # spectral_density -- meep's own default
    50.0,  # Q_thresh -- meep's own default (modes below this are dropped as noise)
    mp.inf,  # rel_err_thresh -- meep's own sentinel (1e20, finite). Passing real
             # float("inf") here was the bug: it poisons harminv's internal
             # band-count arithmetic with inf/NaN, which becomes garbage once cast
             # to an integer array length -- hence std::bad_array_new_length.
    0.01,  # err_thresh
    -1.0,  # rel_amp_thresh
    -1.0,  # amp_thresh
)

print(f"\n{'wavelength (nm)':>16}  {'freq (THz)':>11}  {'Q':>10}  {'|amp|':>10}  {'err':>10}")
for freq, amp, err in sorted(bands, key=lambda b: -b[0].real):
    Q = freq.real / (-2 * freq.imag) if freq.imag != 0 else float("inf")
    wavelength_nm = C / freq.real * 1e9 if freq.real != 0 else float("inf")
    print(f"{wavelength_nm:16.1f}  {freq.real / 1e12:11.2f}  {Q:10.1f}  {abs(amp):10.3e}  {err:10.2e}")

if not bands:
    print("(no modes found above Q_thresh=50 -- means either nothing here rings that")
    print(" cleanly, or fmin/fmax missed it. Worth knowing either way.)")
