# Passive cavity characterization via dipole ring-down and harmonic inversion

*Part of the [fdtdx-notes](.) walkthrough. Covers `step5_dipole_ringdown.py` (FDTDX) and
`harminv_analyze.py` (Meep).*

This tutorial covers how to answer a specific question for any passive (no gain, no pump) optical
cavity: what wavelengths does it naturally resonate at, and how good — how high-Q — is each
resonance? This is stage one of the broader laser-simulation pipeline: before any gain medium or
electrical pumping enters the picture, you want to know the cold-cavity behavior of the bare
structure, since that's fast and cheap to compute and tells you a lot about whether a design is
worth pursuing further.

## Why not just solve for the modes directly?

For simple, highly symmetric structures you can. But in general, finding a cavity's resonances
directly is hard, and there's a much more practical approach borrowed straight from experimental
spectroscopy: **ring-down**. Strike a bell once, and it rings at its own natural frequencies,
decaying at a rate set by how lossy it is. Do the electromagnetic equivalent — excite the structure
with a short pulse and then just watch how the fields evolve afterward — and the same information
falls out: the frequencies present in the ringing are the resonances, and how fast each one decays
tells you its Q factor (a high-Q resonance rings for a long time before its energy leaks away).

## Step 1: exciting the cavity (FDTDX)

The excitation needs to be broadband — you want to test many candidate wavelengths in a single run,
not guess one frequency at a time. A point dipole source driven by a Gaussian pulse in time does
this: a Gaussian pulse in time corresponds to a broad spread of frequencies in the frequency domain,
so one such pulse effectively excites every resonance in range simultaneously.

```python
# step5_dipole_ringdown.py — the excitation
source = fdtdx.PointDipoleSource(
    position=..., 
    profile=fdtdx.GaussianPulseProfile(...),
)
detector = fdtdx.FieldDetector(...)  # records the raw time-domain field afterward
```

**This only works on a genuinely finite structure.** Earlier steps in this project used an
infinite/PML-extended slab, which supports lossless guided modes — modes that, by construction,
never leak energy and so never decay on any practical timescale. There's nothing to "ring down" in a
system that doesn't lose energy. Step 5 deliberately switches to a finite geometry that actually
leaks light out, which is also the physically relevant case — every real cavity is finite and every
real cavity loses some light.

## Step 2: extracting resonances (Meep's harminv, not a hand-rolled FFT)

Once you have the raw decaying time-domain signal from the detector, you need to turn it into a list
of (frequency, Q) pairs. The obvious first instinct is a Fourier transform and peak-finding — this
was tried and rejected. A plain FFT struggles badly when several resonances sit close together in
frequency and "beat" against each other, which is exactly what a compact multi-mode cavity produces.
A hand-rolled FFT-peak/decay-fit approach disagreed with the more careful method by 80-200%
depending on the excitation bandwidth used — not a rounding error, a fundamentally unreliable
answer.

Instead, `harminv_analyze.py` calls Meep's own harmonic-inversion routine, `mp.py_do_harminv`,
directly on the recorded field data. Harmonic inversion is a signal-processing method purpose-built
for this exact situation: given a signal that's a sum of several exponentially-decaying sinusoids
(precisely what a ringing multi-mode cavity produces), it extracts each component's frequency, decay
rate, and amplitude far more precisely than peak-finding on a spectrum — especially for closely
spaced or weak resonances. The general lesson: when a mature, purpose-built tool exists for your
exact problem, use it rather than refining your own approximation further.

## Two real bugs — found by reading Meep's actual source, not by guessing

Getting `harminv_analyze.py` working required cloning Meep's own repository and reading
`python/meep.i` (its SWIG interface layer between Python and the underlying C++) directly, because
neither failure mode was explained anywhere in the documentation.

**Bug 1 — Meep's "infinity" isn't IEEE infinity.** A relative-error threshold argument expects
Meep's own sentinel value `mp.inf`, which is a plain finite number, `1e20` — not Python's real
`float("inf")`. Certain internal computations misbehave with a genuine infinite value, so Meep
defines its own large-but-finite stand-in instead.

```python
# wrong — looks correct, isn't what Meep's API actually expects
rel_err_thresh = float("inf")

# right — Meep's own sentinel
rel_err_thresh = mp.inf   # == 1.0e20
```

**Bug 2 — the actual crash.** This one produced a fairly opaque
`std::bad_array_new_length` error with no obvious connection to its real cause:

```python
# wrong — a numpy array, not a Python list
mp.py_do_harminv(Ex, ...)

# right
mp.py_do_harminv(Ex.tolist(), ...)
```

`py_do_harminv`'s C++ layer requires its data argument to be a genuine Python `list`. Passing a
numpy array doesn't raise a clear type error — the C++ wrapper's internal `PyList_Size` call
silently returns `-1` for a non-list object, and that `-1` then gets used to allocate an array of
"negative length" deep inside the C++ code, which is what actually throws. Nothing in the Python-level
error message points at "wrong argument type" — the only way to find this was reading the actual
SWIG-wrapped C++ source and seeing what `py_do_harminv` does with its argument before it ever
reaches the real harmonic-inversion algorithm.

**The general lesson:** when a crash message doesn't obviously connect to anything in your own code,
and the library is open source, cloning it and reading the relevant source directly is often faster
— and more reliable — than guessing from the error text or searching for others who hit the same
message.

## The result

On a first proof-of-pipeline geometry — a plain 16×16×16-cell silicon cube, chosen only to validate
the toolchain, not as a real cavity design — the pipeline found seven resonances between
1200-2440nm with Q factors ranging roughly 50-1700. The specific numbers don't matter yet; what
matters is that dipole excitation (FDTDX), ring-down recording, and harminv extraction (Meep) all
work correctly together end to end, on real hardware, which means the same pipeline can now be
pointed at an actual candidate cavity geometry with confidence that a wrong answer would be a
geometry problem, not a toolchain problem.
