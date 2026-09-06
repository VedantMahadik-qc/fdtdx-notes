"""
Step 5 -- a real (if simple) resonator: dipole excitation, ring-down, save for analysis.

New idea: instead of a plane-wave source aimed at a target, a POINT DIPOLE excites
the structure broadband and we watch what's left oscillating after it turns off.
Whatever survives longest is a resonance of the structure.

fdtdx has no built-in equivalent of Meep's Harminv (the harmonic-inversion
algorithm that pulls out resonant frequencies + Q directly from a ring-down
signal). A hand-rolled FFT-peak/decay-fit stand-in works fine for a single,
well-isolated mode, but this cube supports several closely-spaced Mie-type
resonances and they beat together -- a plain FFT or single-exponential fit
can't tell them apart (verified: they disagreed by 2-3x depending on how the
excitation bandwidth was chosen). So this script just runs FDTDX and saves the
raw signal; harminv_analyze.py (WSL2 side) does the actual analysis, using
meep's own py_do_harminv directly on this data -- the real algorithm, not an
approximation of it, reused rather than reimplemented.

Geometry is a finite dielectric cube, not the infinite slab from steps 2-3:
an infinite slab supports lossless guided modes (trapped by total internal
reflection, only escaping once they reach the lateral PML, which takes far
longer than we want to simulate) -- finite removes that escape hatch.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx
from fdtdx.constants import c

WAVELENGTH = 1.55e-6
SPACING = WAVELENGTH / 20
N = 60

CENTER_WAVELENGTH = 1.6e-6
SPECTRAL_WIDTH_HZ = 0.35 * c / CENTER_WAVELENGTH  # broadband -- harminv (not us) sorts the individual modes out
RUN_TIME = 800e-15  # long enough for the resonances to ring down after the pulse ends


def build():
    ON_MAIN = hasattr(fdtdx, "UniformGrid")
    common = dict(time=RUN_TIME, dtype=jnp.float32, courant_factor=0.99, gradient_config=None)
    config = (
        fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=SPACING), **common)
        if ON_MAIN
        else fdtdx.SimulationConfig(resolution=SPACING, **common)
    )

    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(N, N, N))
    objects.append(volume)

    b_objs, b_cons = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=8, boundary_type="pml"), volume
    )
    objects.extend(b_objs.values())
    constraints.extend(b_cons)

    # Finite in all three axes (unlike steps 2-3's slab, which deliberately extended
    # into the PML to model an infinite waveguide -- see module docstring).
    slab = fdtdx.UniformMaterialObject(
        name="slab",
        partial_grid_shape=(16, 16, 16),
        material=fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_silicon),
    )
    constraints.append(slab.place_at_center(volume, axes=(0, 1, 2)))
    objects.append(slab)

    dipole = fdtdx.PointDipoleSource(
        name="dipole",
        partial_grid_shape=(1, 1, 1),
        polarization=0,  # Ex
        wave_character=fdtdx.WaveCharacter(wavelength=CENTER_WAVELENGTH),
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_wave=fdtdx.WaveCharacter(wavelength=CENTER_WAVELENGTH),
            spectral_width=fdtdx.WaveCharacter(frequency=SPECTRAL_WIDTH_HZ),
        ),
    )
    constraints.extend(
        [
            dipole.place_at_center(volume, axes=(0, 1)),
            dipole.place_relative_to(slab, axes=(2,), own_positions=(0,), other_positions=(-0.3,)),
        ]
    )
    objects.append(dipole)

    probe = fdtdx.FieldDetector(
        name="probe",
        partial_grid_shape=(1, 1, 1),
        components=("Ex",),
    )
    constraints.extend(
        [
            probe.place_at_center(volume, axes=(0, 1)),
            probe.place_relative_to(slab, axes=(2,), own_positions=(0,), other_positions=(-0.3,)),
        ]
    )
    objects.append(probe)

    key = jax.random.PRNGKey(42)
    obj_c, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays = fdtdx.extend_material_to_pml(objects=obj_c, arrays=arrays)
    arrays, obj_c, _ = fdtdx.apply_params(arrays, obj_c, params, key)
    return obj_c, arrays, config, key


objects, arrays, config, key = build()
print(f"grid {objects.volume.grid_shape}, {config.time_steps_total} time steps, dt={config.time_step_duration:.3e} s")

run = jax.jit(fdtdx.run_fdtd, static_argnames=["show_progress"])
final_step, out = run(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
jax.block_until_ready(out)

Ex = np.asarray(out.detector_states["probe"]["fields"]).squeeze()  # (num_time_steps,)
dt = float(config.time_step_duration)
print(f"probe recorded {Ex.shape[0]} samples, dt={dt:.4e} s")

np.savez("ringdown_data.npz", Ex=Ex, dt=dt, center_wavelength=CENTER_WAVELENGTH, spectral_width_hz=SPECTRAL_WIDTH_HZ)
print("wrote ringdown_data.npz -- hand this to harminv_analyze.py")
