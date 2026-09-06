# Power and units in semiconductor laser rate equations

*Part of the [fdtdx-notes](.) walkthrough. Covers `rate_equations.py`.*

This one is about two bugs that had nothing to do with the physics being wrong and everything to
do with bookkeeping — which turns out to be the more dangerous kind, because the code runs fine,
produces a plot, and the numbers just happen to be nonsense. Both were caught by cross-checking the
output against independent physical expectations rather than trusting a clean run.

## The setup

`rate_equations.py` solves the standard two-variable semiconductor laser rate equations: carrier
density `N` and photon density `S`, coupled through gain. The two quantities that ultimately matter
for comparing against a real device are the threshold current and the output power as a function of
drive current (the L-I curve). Getting those two numbers right turned out to hinge entirely on
getting two unit conversions right — not on the differential equations themselves, which were
correct from the start.

## Bug 1: which volume is the photon density actually "per"?

The rate equations couple a carrier density `N` (carriers per unit *active* volume `V` — the
electrically pumped region) to a photon density `S`. The natural question once you want an actual
output power in watts is: photon density times what volume gives you a total photon count?

The tempting answer is `V`, the same active volume used for the carrier density. It's wrong. The
optical mode inside a laser cavity is very rarely perfectly confined to just the active region — in
general only a fraction `Γ` (the confinement factor) of the mode's energy actually overlaps the
gain region. The photon density in these equations is naturally defined per *mode* volume, `V/Γ`,
not per active volume `V`. So the total photon number is:

```python
# wrong — silently assumes the photon density lives in the active volume
photon_number = S * V

# right — the photon density lives in the (larger) mode volume V/Gamma
photon_number = S * (V / Gamma)
```

Missing the `/Gamma` doesn't look wrong by inspection — `S * V` is a perfectly dimensionally sound
expression, it's just answering the wrong question. It made the predicted output power come out
about 80x too high.

**How it was actually caught:** not by staring at the code, but by an independent physical
cross-check (see [tutorial 3](03_carrier_photon_threshold.md) for the full derivation) — comparing
the relaxation-oscillation frequency read directly off the simulated turn-on transient against an
independent analytic formula for the same quantity. The two disagreed by a factor of ~2.6 instead of
matching, which was the signal something upstream was off. Chasing that down is what surfaced the
missing `/Gamma`.

**The general lesson:** when a variable is described as a "density," always ask *density per what
volume, exactly* — and don't assume every density-like quantity in the same set of equations shares
the same reference volume. Confinement factors are one of the more common places this goes wrong in
real laser physics, not just in this code.

## Bug 2: mW/mA already equals W/A

The second bug was purely a units-conversion instinct gone wrong. Slope efficiency (output power
per unit drive current, above threshold) was being computed correctly in SI units — watts out per
amp in — and then printed like this:

```python
# wrong — this "conversion" invents a spurious factor of 1000
print(f"slope efficiency = {slope_efficiency * 1e3:.4f} mW/mA")
```

The reasoning behind the `*1e3` was "converting W/A to mW/mA needs a scale factor," which sounds
right and is completely wrong. Milliwatts are `1e-3` watts and milliamps are `1e-3` amps, so
mW/mA = `(1e-3 W)/(1e-3 A)` = `W/A` exactly — the two `1e-3` factors cancel. **A quantity expressed
as W/A is already numerically identical to the same quantity expressed as mW/mA.** No conversion
factor belongs there at all:

```python
# right — no scale factor needed, W/A and mW/mA are the same number
print(f"slope efficiency = {slope_efficiency:.4f} mW/mA")
```

Before the fix, the script printed a slope efficiency of 140.58 mW/mA — a number that should have
been caught by inspection alone. Real edge-emitting diode lasers have slope efficiencies of roughly
0.1-1 mW/mA; anything two orders of magnitude above that is a sign to stop and check units, not a
sign of an unusually good laser.

**The general lesson:** "per-mA" and "per-A" quantities, and "m-something per m-something" ratios in
general, are common places where a reflexive unit conversion introduces an error instead of fixing
one. Before multiplying by a conversion factor, write out the actual units on both sides and check
whether they were already equal.

## Why both bugs mattered together

Neither of these was a sign the underlying physics was wrong — the two coupled ODEs for `dN/dt` and
`dS/dt` were correct from the start (see [tutorial 3](03_carrier_photon_threshold.md) for that
derivation). Both bugs were in the *reporting* layer: converting an internal state variable into a
physically meaningful, comparable number. That's exactly the layer that's easiest to get wrong
silently, because the simulation still runs, still produces a plausible-looking curve, and the only
way to catch it is to ask "does this final number make physical sense" rather than "did the code
crash."

## The checklist this suggests

Two questions are worth asking of every derived output in a simulation like this, before trusting
the number:

1. For every density-like quantity, what volume is it actually defined per — and is that the same
   volume used everywhere else it's combined with?
2. For every unit conversion, what are the units on both sides *before* applying the conversion
   factor — and does the factor actually change anything, or do the scales already cancel?

Both bugs here passed a "does the code run" test and failed a "does the number make physical sense"
test — which is why the verification checks built into `rate_equations.py` (a threshold-jump check
and the relaxation-oscillation cross-check) matter as much as the model itself.
