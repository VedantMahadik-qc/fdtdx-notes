# Carrier-photon rate equations: threshold and turn-on dynamics

*Part of the [fdtdx-notes](.) walkthrough. Covers `rate_equations.py`.*

This tutorial is about the actual physics inside a semiconductor laser's rate equations — what
carriers and photons are doing to each other, why the light-current curve has such a sharp kink at
threshold, why a laser "rings" when you switch it on, and a bug in the model's own verification
check that's worth understanding in detail because catching it required doing the physics properly
rather than trusting a plausible-looking formula.

## The physical picture

A laser diode has two populations constantly trading energy: electrons (carriers), injected by the
drive current, and photons, bouncing around inside the optical cavity. A carrier can disappear three
ways: non-radiatively (turns into heat, no photon produced), via ordinary spontaneous emission (a
photon emitted in a random direction and phase, mostly missing the lasing mode entirely), or via
**stimulated emission** — an existing photon in the cavity mode "tickles" an excited electron into
dropping down and emitting a second photon that is identical to the first: same frequency, phase,
and direction. That stimulated process is the actual mechanism of laser gain, and it's what couples
the two populations: more photons around means more stimulated recombination, which produces more
photons still. That positive feedback loop is the whole reason lasing is possible.

This gives two coupled ODEs — carrier density `N` and photon density `S`:

```python
def rate_equations(t, y, I_func):
    N, S = y
    I = I_func(t)
    gain = a_gain * (N - N_tr)
    dN = I / (q * V) - N / tau_n - v_g * gain * S
    dS = Gamma * v_g * gain * S - S / tau_p + Gamma * beta_sp * N / tau_n
    return [dN, dS]
```

`dN/dt`: carriers in from the pump current, out via spontaneous decay (`N/tau_n`), out via
stimulated depletion (`v_g * gain * S` — proportional to how many photons are already there to
stimulate emission). `dS/dt`: photons in via stimulated gain (the same term, scaled by the
confinement factor `Gamma`, since only a fraction of the mode overlaps the gain region — see
[tutorial 1](01_power_and_units.md) for why that factor matters), out via cavity loss (`S/tau_p`),
plus a small contribution from spontaneous emission that happens to land in the lasing mode anyway.

## Why the L-I curve kinks so sharply at threshold

Below threshold, there simply aren't enough photons yet for stimulated emission to matter much, so
carriers pumped in mostly just pile up as `N` rises with drive current. Once `N` reaches the
threshold density `N_th` — the point where modal gain exactly equals cavity loss — something
qualitatively different happens: the photon population that's now present becomes strong enough that
any further increase in carrier density gets converted almost immediately into stimulated photons
instead of raising `N` further. The carrier density **clamps** at `N_th` for essentially all drive
currents above threshold, and every additional electron pumped in becomes a photon instead. That
clamping is the entire reason real laser L-I curves have a sharp kink at threshold and go
essentially linear immediately afterward — it isn't an approximation or a modeling choice, it falls
directly out of these two equations.

## Relaxation oscillations: why a laser rings when you turn it on

Switch the current on suddenly and the carrier and photon populations don't settle to their new
steady state smoothly — they overshoot and oscillate a few times first before damping out, visible
directly in the simulated turn-on transient as a few damped power spikes. This is conceptually the
same phenomenon as a spring-mass system released from a stretched position: it overshoots
equilibrium, oscillates, and gradually settles, because there's an energy-exchange mechanism (here,
between carriers and photons, mediated by stimulated emission) with some inertia in it.

## The bug: does the confinement factor belong in the relaxation-oscillation formula?

There's an independent analytic formula for the relaxation-oscillation frequency, and comparing it
against the frequency actually seen in the simulated transient is a strong self-consistency check —
if a numeric simulation and an independent analytic formula for the same physical quantity disagree
badly, something is wrong with one of them. The first version of that formula included the
confinement factor `Gamma` inside the square root. It shouldn't be there, and working out why is a
genuinely instructive derivation.

Linearize the two rate equations around the threshold operating point (`N = N_th + n`,
`S = S_ss + s`, for small perturbations `n`, `s`) and keep only first-order terms. Two things happen:

1. In the `dS/dt` equation, the loss term `-s/tau_p` and the linearized gain term both involve `s`,
   and at exactly threshold the threshold condition (`Gamma * v_g * a * (N_th - N_tr) = 1/tau_p`)
   makes those two `s`-dependent contributions cancel exactly — the net damping of `s` from the gain
   term and the cavity-loss term cancel, leaving the oscillation's restoring force to come entirely
   from the cross-coupling with `n`.
2. Combining the two linearized equations into a single second-order equation for `s` produces a
   term `Gamma * v_g * a * S_ss` (from `dS/dt`'s own dependence on `n`) multiplying a term
   `1 / (Gamma * tau_p)` (which is exactly what the threshold condition says `v_g * a * (N_th - N_tr)`
   equals). Those two factors of `Gamma` — one direct, one inverted — cancel exactly, leaving:

```
omega_r^2 = v_g * a_gain * S_ss / tau_p        # no Gamma anywhere
```

```python
# wrong — Gamma doesn't belong here, it already cancelled in the derivation
f_r_analytic = (1/(2*np.pi)) * np.sqrt(Gamma * v_g * a_gain * S_ss / tau_p)

# right
f_r_analytic = (1/(2*np.pi)) * np.sqrt(v_g * a_gain * S_ss / tau_p)
```

Including the spurious `Gamma` threw the analytic prediction off against the transient's own
measured oscillation period by a factor of 2.62 instead of the expected ~1 — which is what flagged
that something upstream needed fixing (and, tracing further back, is what actually surfaced the
separate missing-`1/Gamma` bug in the output-power formula covered in
[tutorial 1](01_power_and_units.md) — that one didn't affect this frequency check directly, but
fixing both together is what got every cross-check passing at once).

## Verifying it properly

`rate_equations.py` checks itself two ways rather than trusting either the simulation or the formula
alone: it reads the relaxation-oscillation frequency directly off the simulated transient (the time
spacing between the first two power peaks after turn-on), and separately evaluates the analytic
formula above at the same run's steady-state photon density, then compares the two. After both
fixes, the transient gives 2.52 GHz and the analytic formula gives 3.04 GHz — agreement to within
about 20%, which is the right amount of agreement to expect, since the analytic formula is a
small-signal approximation and the transient here was deliberately driven at twice threshold,
somewhat outside the regime where that approximation is tightest.

The final, physically sensible numbers this model settles on: threshold current ≈13.48 mA, slope
efficiency ≈1.41 mW/mA — both realistic for a real edge-emitting diode laser, which is itself a
useful sanity check that the model behaves like an actual laser rather than merely satisfying its
own equations.

## The general lesson

A model can integrate cleanly, produce a smooth-looking curve, and still be built on a formula that
looks plausible but doesn't survive an actual derivation. The fix here wasn't "try removing terms
until the numbers match" — it was linearizing the real equations around the real operating point and
showing algebraically where a factor cancels. Whenever a "generally true" formula gets adapted to a
specific model, it's worth rederiving it in that model's own notation rather than assuming an
intuitive-looking version transfers over unchanged.
