
import jax
import jax.numpy as jnp
 
import fdtdx
 
# 1. the simulation config 
# `grid` sets the spatial discretisation. `time` is physical duration, not steps;
# fdtdx derives the step count from the Courant condition.
WAVELENGTH = 1.55e-6                    # 1550 nm, standard telecom wavelength
SPACING = WAVELENGTH / 20               # 20 cells per wavelength -> ~77.5 nm
 
# GOTCHA: the PyPI release and the GitHub main branch have DIFFERENT APIs and
# BOTH report version 0.6.2, so nothing warns you.
#   PyPI  (pip install fdtdx) -> SimulationConfig(resolution=<float>)
#   main  (pip install git+…) -> SimulationConfig(grid=UniformGrid(spacing=<float>))
# This repo's examples and the readthedocs "latest" docs assume main.
# Detect which one is installed so the script runs either way.
ON_MAIN = hasattr(fdtdx, "UniformGrid")
print(f"fdtdx API          {'main (grid=)' if ON_MAIN else 'PyPI 0.6.2 (resolution=)'}")
 
common = dict(time=50e-15, dtype=jnp.float32, courant_factor=0.99, gradient_config=None)
if ON_MAIN:
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=SPACING), **common)
else:
    config = fdtdx.SimulationConfig(resolution=SPACING, **common)
 
print(f"spacing            {SPACING * 1e9:.1f} nm")
print(f"time step          {config.time_step_duration:.3e} s")
print(f"total time steps   {config.time_steps_total}")
 
# 2. the objects
objects = []
constraints = []
 
# The simulation volume is the root object. Everything else is positioned
# relative to it. 60x60x60 cells is deliberately tiny so this runs in seconds.
volume = fdtdx.SimulationVolume(partial_grid_shape=(60, 60, 60))
objects.append(volume)
 
# Boundaries. `from_uniform_bound` builds the same boundary on all six faces.
# PML = perfectly matched layer, an absorbing boundary that fakes open space.
# This helper returns BOTH the objects and the constraints that place them,
# which is why we extend two different lists.
boundary_objects, boundary_constraints = fdtdx.boundary_objects_from_config(
    fdtdx.BoundaryConfig.from_uniform_bound(thickness=8, boundary_type="pml"),
    volume,
)
objects.extend(boundary_objects.values())
constraints.extend(boundary_constraints)
 
print(f"\nobjects before placement: {len(objects)}")
print(f"constraints:              {len(constraints)}")
 
# 3. solve the placement
# place_objects returns five things. The two that matter now:
#   obj_container — the objects, now with concrete grid positions
#   arrays — the actual field/material arrays living on the grid
key = jax.random.PRNGKey(42)
obj_container, arrays, params, config, _ = fdtdx.place_objects(
    object_list=objects,
    config=config,
    constraints=constraints,
    key=key,
)
 
# 4. look at what came back 
print(f"\ngrid shape         {obj_container.volume.grid_shape}")
print(f"E field shape      {arrays.fields.E.shape}   <- (3, Nx, Ny, Nz): one per component")
print(f"H field shape      {arrays.fields.H.shape}")
print(f"permittivity shape {arrays.inv_permittivities.shape}")
 
print(f"\nplaced objects:")
for obj in obj_container.objects:
    print(f"  {type(obj).__name__:28s} {obj.name:20s} grid_slice={obj.grid_slice}")