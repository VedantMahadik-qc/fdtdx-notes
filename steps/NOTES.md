## Gotchas

**PyPI vs GitHub main have different APIs and both report v0.6.2.**
PyPI: `SimulationConfig(resolution=<float>)`. main: `SimulationConfig(grid=UniformGrid(spacing=...))`.
Install from git: `pip install git+https://github.com/ymahlau/fdtdx.git`
