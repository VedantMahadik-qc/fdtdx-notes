
import jax

jax.config.update("jax_enable_x64", True)   # must happen before any array work

import numpy as np                           # noqa: E402
import jax.numpy as jnp                      # noqa: E402

import fdtdx                                 # noqa: E402

WAVELENGTH = 1.55e-6
SPACING = WAVELENGTH / 20
N = 40
PML = 6
SCALE = 1e22          # lifts the loss off ~1e-23 so it is easy to read


def build(gradient_config):
    ON_MAIN = hasattr(fdtdx, "UniformGrid")
    common = dict(time=20e-15, dtype=jnp.float64, courant_factor=0.99, gradient_config=gradient_config)
    config = (
        fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=SPACING), **common)
        if ON_MAIN
        else fdtdx.SimulationConfig(resolution=SPACING, **common)
    )

    objects, constraints = [], []
    vol = fdtdx.SimulationVolume(partial_grid_shape=(N, N, N))
    objects.append(vol)

    b_objs, b_cons = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig.from_uniform_bound(thickness=PML, boundary_type="pml"), vol
    )
    objects.extend(b_objs.values())
    constraints.extend(b_cons)

    slab = fdtdx.UniformMaterialObject(
        name="slab",
        partial_grid_shape=(None, None, 8),
        material=fdtdx.Material(permittivity=fdtdx.constants.relative_permittivity_silicon),
    )
    constraints += [slab.same_size(vol, axes=(0, 1)), slab.place_at_center(vol, axes=(0, 1, 2))]
    objects.append(slab)

    src = fdtdx.GaussianPlaneSource(
        name="source",
        partial_grid_shape=(None, None, 1),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=fdtdx.WaveCharacter(wavelength=WAVELENGTH),
        radius=N * SPACING / 5,
        std=1 / 3,
        direction="+",
    )
    constraints += [
        src.same_size(vol, axes=(0, 1)),
        src.place_at_center(vol, axes=(0, 1)),
        src.place_relative_to(vol, axes=(2,), own_positions=(0,), other_positions=(-0.6,)),
    ]
    objects.append(src)

    det = fdtdx.EnergyDetector(
        name="transmitted", partial_grid_shape=(None, None, 1), as_slices=False, reduce_volume=True
    )
    constraints += [
        det.same_size(vol, axes=(0, 1)),
        det.place_at_center(vol, axes=(0, 1)),
        det.place_relative_to(vol, axes=(2,), own_positions=(0,), other_positions=(0.6,)),
    ]
    objects.append(det)

    key = jax.random.PRNGKey(42)
    oc, arr, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arr = fdtdx.extend_material_to_pml(objects=oc, arrays=arr)
    arr, oc, _ = fdtdx.apply_params(arr, oc, params, key)
    return oc, arr, config, key


# voxels to probe. The first three sit inside the slab near the beam axis, where
# the field is strong and the sensitivity is genuinely non-zero. The last is a
# far PML corner where there is essentially no field, so the true gradient
# should be ~0 — a good control.
PROBES = [(0, 20, 20, 18), (0, 23, 20, 20), (0, 20, 23, 23), (0, 37, 34, 16)]


def analyse(label, gradient_config):
    oc, arr, cfg, key = build(gradient_config)

    def loss(inv_perm):
        a = arr.aset("inv_permittivities", inv_perm)
        _, out = fdtdx.run_fdtd(arrays=a, objects=oc, config=cfg, key=key, show_progress=False)
        return SCALE * jnp.sum(out.detector_states["transmitted"]["energy"])

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    jloss = jax.jit(loss)

    val, grad = value_and_grad(arr.inv_permittivities)
    grad = np.asarray(grad)

    inside = np.zeros_like(grad, dtype=bool)
    inside[:, PML : N - PML, PML : N - PML, PML : N - PML] = True

    print(f"\n=== {label} ===   loss = {float(val):.8f}")
    print(f"  max |grad| inside PML     {np.abs(grad[~inside]).max():.4e}   <- should be tiny: no field there")
    print(f"  max |grad| in interior    {np.abs(grad[inside]).max():.4e}")

    base = np.asarray(arr.inv_permittivities)
    h = 1e-4
    print(f"  {'voxel':>14s} {'autodiff':>15s} {'finite diff':>15s} {'rel err':>9s}")
    for idx in PROBES:
        up, dn = base.copy(), base.copy()
        up[idx] += h
        dn[idx] -= h
        num = float(jloss(jnp.asarray(up)) - jloss(jnp.asarray(dn))) / (2 * h)
        ana = float(grad[idx])
        rel = abs(num - ana) / max(abs(num), abs(ana), 1e-30)
        tag = "  <- PML control" if not inside[idx] else ""
        print(f"  {str(idx[1:]):>14s} {ana:+15.5e} {num:+15.5e} {rel:8.1%}{tag}")
    return grad


g_rev = analyse("reversible", fdtdx.GradientConfig(method="reversible", recorder=fdtdx.Recorder(modules=[])))
g_chk = analyse("checkpointed", fdtdx.GradientConfig(method="checkpointed", num_checkpoints=8))

d = np.abs(g_rev - g_chk)
worst = np.unravel_index(int(d.argmax()), d.shape)
print(f"\nlargest disagreement between the two methods: {d.max():.4e} at voxel {worst[1:]}")