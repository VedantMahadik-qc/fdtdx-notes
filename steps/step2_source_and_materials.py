
import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

WAVELENGTH = 1.55e-6
SPACING = WAVELENGTH / 20
N = 60

ON_MAIN = hasattr(fdtdx, "UniformGrid")
common = dict(time=50e-15, dtype=jnp.float32, courant_factor=0.99, gradient_config=None)
config = (
    fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=SPACING), **common)
    if ON_MAIN
    else fdtdx.SimulationConfig(resolution=SPACING, **common)
)

objects, constraints = [], []

volume = fdtdx.SimulationVolume(partial_grid_shape=(N, N, N))
objects.append(volume)

boundary_objects, boundary_constraints = fdtdx.boundary_objects_from_config(
    fdtdx.BoundaryConfig.from_uniform_bound(thickness=8, boundary_type="pml"), volume
)
objects.extend(boundary_objects.values())
constraints.extend(boundary_constraints)

# a silicon slab through the middle
# The slab spans the full x and y extent and is 12 cells thick in z, centred.
#
# GOTCHA: running the slab through the x/y PML is INTENTIONAL here — it models a
# slab that continues to infinity, and the PML has to contain the same material
# or the waveguide mode reflects off the boundary. fdtdx warns
#   "PML region at axis=0 direction='-' has non-default inv_permittivities ..."
# which looks alarming but is expected: extend_material_to_pml() (called below)
# is what resolves it. Only worry if you did NOT mean the material to reach the
# boundary, in which case shrink the object instead of silencing the warning.
slab = fdtdx.UniformMaterialObject(
    name="slab",
    partial_grid_shape=(None, None, 12),
    material=fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_silicon),
)
constraints.extend(
    [
        slab.same_size(volume, axes=(0, 1)),
        slab.place_at_center(volume, axes=(0, 1, 2)),
    ]
)
objects.append(slab)

# a plane source below the slab 
source = fdtdx.GaussianPlaneSource(
    name="source",
    partial_grid_shape=(None, None, 1),          # one cell thick in z
    fixed_E_polarization_vector=(1, 0, 0),       # E along x
    wave_character=fdtdx.WaveCharacter(wavelength=WAVELENGTH),
    radius=N * SPACING / 5,                      # beam waist
    std=1 / 3,
    direction="+",                               # propagate toward +z
)
constraints.extend(
    [
        source.same_size(volume, axes=(0, 1)),
        source.place_at_center(volume, axes=(0, 1)),
        source.place_relative_to(volume, axes=(2,), own_positions=(0,), other_positions=(-0.6,)),
    ]
)
objects.append(source)

# solve placement 
key = jax.random.PRNGKey(42)
obj_container, arrays, params, config, _ = fdtdx.place_objects(
    object_list=objects, config=config, constraints=constraints, key=key
)

# GOTCHA: two more calls are needed before the arrays are simulation-ready.
#   extend_material_to_pml — copies the adjacent material into the PML so the
#                             absorber matches its neighbour (else it reflects)
#   apply_params — resolves any designable parameters into the arrays.
#                             Required even with no design region present.
arrays = fdtdx.extend_material_to_pml(objects=obj_container, arrays=arrays)
arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)

# Result 
print(f"grid                {obj_container.volume.grid_shape}")
for name in ("slab", "source"):
    obj = obj_container[name]
    print(f"{name:8s} z-slice    {obj.grid_slice[2]}")

# inv_permittivity is 1/eps_r, so silicon (eps_r ~ 12.25) shows up as ~0.082
# and vacuum as 1.0. Print a profile down the z axis through the centre.
eps_inv = np.asarray(arrays.inv_permittivities[0, N // 2, N // 2, :])
print(f"\n1/eps_r down the z axis at (x,y) = centre:")
print(f"  vacuum value    {eps_inv.max():.3f}")
print(f"  slab value      {eps_inv.min():.3f}   (eps_r = {1 / eps_inv.min():.2f})")
print(f"  slab occupies z {np.flatnonzero(eps_inv < 0.5).min()}..{np.flatnonzero(eps_inv < 0.5).max()}")

print("\n  z:  " + "".join("#" if v < 0.5 else "." for v in eps_inv))
print("      ('#' = silicon, '.' = vacuum; source sits at z="
      f"{obj_container['source'].grid_slice[2].start})")