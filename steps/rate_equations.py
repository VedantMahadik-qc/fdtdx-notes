"""
Carrier-photon rate-equation model -- the "fast cross-check layer" from the
architecture doc. Standard single-mode semiconductor laser rate equations
(Coldren, Corzine & Masanovic, "Diode Lasers and Photonic Integrated
Circuits" -- this is the textbook Ch. 5 two-variable model: carrier density
N and photon density S, coupled through gain). Solved with
scipy.integrate.solve_ivp -- this runs in milliseconds, versus the
hours a full Meep saturable-gain run takes, so it's the quick sanity check
to run before (and alongside) the expensive simulation, and the fast way to
explore drive-current/design tradeoffs before committing to one for Meep.

No fdtdx/meep dependency -- pure scipy/numpy/matplotlib, runs anywhere. I
ran this myself (no cross-machine handoff needed for this one) and checked
it against two independent internal-consistency tests before calling it
done -- see the bottom of the script.

PARAMETERS BELOW ARE PLACEHOLDERS pending Srinjoy's material pick, with two
exceptions: the wavelength (706nm is confirmed) and the geometry/loss
numbers that get COMBINED into tau_p using real physics rather than picked
directly -- the relationship is already correct even though the specific
inputs will change. Two things to swap in once they exist:
  1. Gamma and tau_p: once FDTDX has a real 706nm cavity, replace the
     facet-loss estimate below with tau_p = Q / (2*pi*c/wavelength), Q
     read off the harminv table, and Gamma from the FDTDX mode overlap.
  2. a_gain, N_tr, tau_n, alpha_i: real values for whichever material
     Srinjoy picks off the materials table (from the manufacturer/paper
     the material comes from, or refractiveindex.info + literature).
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- physical constants -----------------------------------------------------
q = 1.602176634e-19   # C
c = 299792458.0       # m/s
h = 6.62607015e-34    # J s

# ---- device geometry (PLACEHOLDER -- generic edge-emitter scale) -----------
L = 300e-6      # cavity length, m
w = 2e-6        # stripe width, m
d = 80e-9       # active region thickness, m (a few-QW stack)
V = L * w * d   # active volume, m^3

# ---- optical / cavity parameters --------------------------------------------
wavelength = 706e-9      # m -- Srinjoy's confirmed target (this one is real)
n_g = 3.6                # group index (PLACEHOLDER, typical III-V)
v_g = c / n_g
Gamma = 0.10             # confinement factor (PLACEHOLDER -- from FDTDX mode
                         # overlap once real geometry exists)
alpha_i = 1000.0         # internal loss, 1/m == 10 /cm (PLACEHOLDER, typical)
R1 = R2 = 0.30           # facet power reflectivity (PLACEHOLDER, cleaved III-V/air)
alpha_m = (1 / L) * np.log(1 / np.sqrt(R1 * R2))
alpha_total = alpha_i + alpha_m
tau_p = 1 / (v_g * alpha_total)   # photon lifetime, s -- DERIVED from the
                                  # geometry/loss numbers above, not picked
                                  # directly. Replace with tau_p = Q/omega
                                  # once FDTDX+harminv give a real Q.

# ---- gain / recombination parameters (PLACEHOLDER, generic III-V QW) -------
a_gain = 2.5e-20       # differential gain, m^2
N_tr = 1.5e24          # transparency carrier density, m^-3  (1.5e18 cm^-3)
tau_n = 2.0e-9         # carrier lifetime, s
beta_sp = 1.0e-4       # spontaneous emission factor coupled into the lasing mode

photon_energy = h * c / wavelength

# ---- derived threshold quantities -------------------------------------------
# Threshold condition: modal gain = total loss -> Gamma*a*(N_th-N_tr) = 1/(v_g*tau_p)
N_th = N_tr + 1.0 / (Gamma * a_gain * v_g * tau_p)
I_th = q * V * N_th / tau_n


def rate_equations(t, y, I_func):
    N, S = y
    I = I_func(t)
    gain = a_gain * (N - N_tr)
    dN = I / (q * V) - N / tau_n - v_g * gain * S
    dS = Gamma * v_g * gain * S - S / tau_p + Gamma * beta_sp * N / tau_n
    return [dN, dS]


def output_power(S):
    # S is photon density averaged over the MODE volume (V/Gamma), not the
    # active volume V -- that's what makes the coupled ODEs above dimensionally
    # consistent (Gamma multiplies the gain term in dS/dt but not in dN/dt).
    # So total photon number in the cavity is S * (V/Gamma), not S * V --
    # missing the /Gamma here was bug #1, caught by the relaxation-oscillation
    # cross-check below coming out ~2.6x off instead of ~1x.
    photon_number = S * (V / Gamma)
    return photon_energy * v_g * alpha_m * photon_number


# The system is stiff: tau_n (~ns) and tau_p (~ps) differ by ~1000x, so an
# explicit RK solver either crawls or blows up. Radau (implicit) handles it.
SOLVE_KW = dict(method="Radau", rtol=1e-8, atol=1.0)

# =============================================================================
# 1) Turn-on transient at a fixed drive level -- shows delay + relaxation
#    oscillations settling to steady state, the classic diode-laser signature.
# =============================================================================
I_drive = 2.0 * I_th


def I_step(t):
    return I_drive if t >= 0 else 0.0


t_span = (0.0, 8e-9)
t_eval = np.linspace(*t_span, 4000)
sol = solve_ivp(rate_equations, t_span, y0=[0.0, 0.0], args=(I_step,), t_eval=t_eval, **SOLVE_KW)
assert sol.success, f"transient solve failed: {sol.message}"

N_t, S_t = sol.y
P_t = output_power(S_t)

# Relaxation-oscillation frequency read directly off the transient (spacing
# between the first two power peaks).
peak_idx = np.where((P_t[1:-1] > P_t[:-2]) & (P_t[1:-1] > P_t[2:]))[0] + 1
if len(peak_idx) >= 2:
    f_r_numeric = 1.0 / (t_eval[peak_idx[1]] - t_eval[peak_idx[0]])
else:
    f_r_numeric = float("nan")

# Independent check: the standard small-signal relaxation-oscillation formula,
# evaluated at the steady-state photon density this same run settles to.
# omega_r^2 = v_g*a*S_ss/tau_p -- NO Gamma here: linearizing the ODEs above
# around threshold, the Gamma in dS/dt's gain term and the 1/(Gamma*tau_p)
# that v_g*a*(N-N_tr) equals at threshold cancel exactly. Including Gamma
# anyway was bug #2 -- it's what actually caused the first failed check
# (bug #1, the output-power scaling, doesn't touch S itself, so it couldn't
# have caused a mismatch here).
S_ss_numeric = S_t[-200:].mean()
f_r_analytic = (1.0 / (2 * np.pi)) * np.sqrt(v_g * a_gain * S_ss_numeric / tau_p)

# =============================================================================
# 2) L-I curve -- steady-state output power vs drive current, swept.
# =============================================================================
I_vals = np.linspace(0.0, 4.0 * I_th, 60)
P_vals = np.empty_like(I_vals)
for i, I_cw in enumerate(I_vals):
    sol_cw = solve_ivp(
        rate_equations, (0.0, 20e-9), y0=[0.0, 0.0], args=(lambda t, I=I_cw: I,), **SOLVE_KW
    )
    assert sol_cw.success, f"L-I solve failed at I={I_cw:.3e} A: {sol_cw.message}"
    P_vals[i] = output_power(sol_cw.y[1, -1])

# Slope efficiency above threshold, from a straight-line fit to the top half
# of the above-threshold points -- should be roughly constant (linear L-I).
above_th = I_vals > 1.5 * I_th
slope_efficiency = np.polyfit(I_vals[above_th], P_vals[above_th], 1)[0]  # W/A

# =============================================================================
# Verification -- don't just trust this, check it.
# =============================================================================
print("=== device / derived parameters ===")
print(f"active volume V              = {V:.3e} m^3")
print(f"alpha_m (mirror loss)        = {alpha_m:.1f} /m  ({alpha_m/100:.1f} /cm)")
print(f"alpha_total                  = {alpha_total:.1f} /m  ({alpha_total/100:.1f} /cm)")
print(f"photon lifetime tau_p        = {tau_p*1e12:.3f} ps")
print(f"threshold carrier density N_th = {N_th:.3e} m^-3  (N_th/N_tr = {N_th/N_tr:.2f})")
print(f"threshold current I_th        = {I_th*1e3:.3f} mA")
print()
print("=== turn-on transient @ I = 2 x I_th ===")
print(f"steady-state P_out            = {output_power(S_ss_numeric)*1e3:.3f} mW")
print(f"relaxation osc. freq, numeric  = {f_r_numeric/1e9:.2f} GHz  (from transient peak spacing)")
print(f"relaxation osc. freq, analytic = {f_r_analytic/1e9:.2f} GHz  (small-signal formula)")
ratio = f_r_numeric / f_r_analytic
print(f"ratio (should be close to 1)   = {ratio:.2f}")
print()
print("=== L-I curve ===")
# slope_efficiency here is W/A (Watts out per Amp in) -- which is ALREADY
# numerically equal to mW/mA (the 1e-3 scale factors on top and bottom
# cancel). Multiplying by 1e3 here was bug #3: it looks like a units
# conversion but it's actually inventing a spurious extra factor of 1000.
print(f"slope efficiency (above threshold) = {slope_efficiency:.4f} mW/mA  (== W/A, no scaling needed)")
print(f"P_out just below I_th   = {P_vals[np.searchsorted(I_vals, 0.9*I_th)]*1e6:.3f} uW")
print(f"P_out at 2x I_th        = {P_vals[np.searchsorted(I_vals, 2.0*I_th)]*1e3:.3f} mW")

print()
print("=== sanity checks ===")
checks_passed = True
if not (np.all(np.isfinite(N_t)) and np.all(np.isfinite(S_t)) and np.all(np.isfinite(P_vals))):
    print("FAIL: NaN/Inf somewhere in N(t), S(t), or the L-I curve")
    checks_passed = False
else:
    print("PASS: no NaN/Inf anywhere")

below_th_power = P_vals[np.searchsorted(I_vals, 0.9 * I_th)]
above_th_power = P_vals[np.searchsorted(I_vals, 2.0 * I_th)]
if above_th_power > 50 * below_th_power:
    print(f"PASS: clear threshold behavior (P jumps {above_th_power/below_th_power:.0f}x from 0.9 I_th to 2 I_th)")
else:
    print(f"FAIL: no clear threshold kink (only {above_th_power/below_th_power:.1f}x)")
    checks_passed = False

if 0.5 < ratio < 2.0:
    print(f"PASS: numeric and analytic relaxation-oscillation frequency agree within 2x ({ratio:.2f})")
else:
    print(f"FAIL: numeric/analytic RO frequency disagree badly ({ratio:.2f})")
    checks_passed = False

print()
print("ALL CHECKS PASSED" if checks_passed else "SOME CHECKS FAILED -- do not trust the plot yet")

# =============================================================================
# plots
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].plot(t_eval * 1e9, P_t * 1e3)
axes[0].set_xlabel("time (ns)")
axes[0].set_ylabel("output power (mW)")
axes[0].set_title("turn-on transient, I = 2 x I_th")

axes[1].plot(I_vals * 1e3, P_vals * 1e3, "o-", ms=3)
axes[1].axvline(I_th * 1e3, color="gray", ls="--", lw=1, label=f"I_th = {I_th*1e3:.2f} mA")
axes[1].set_xlabel("drive current (mA)")
axes[1].set_ylabel("steady-state output power (mW)")
axes[1].set_title("L-I curve")
axes[1].legend()

fig.tight_layout()
fig.savefig("rate_equation_results.png", dpi=140)
print("\nwrote rate_equation_results.png")
