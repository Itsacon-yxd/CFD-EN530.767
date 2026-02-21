from burgers_solver import burgers_solver, get_init, get_vorticity
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


plt.rcParams.update({'font.size': 14})

delta_t = 1e-3
delta_x = 5e-3
delta_y = 5e-3
x_lim = 4
y_lim = 1
mu = 1e-2
num_steps = 3000
gridx, gridy = jnp.meshgrid(jnp.arange(0,x_lim+delta_x/2,delta_x), jnp.arange(0,y_lim+delta_y/2,delta_y), indexing='ij')
grid = jnp.stack([gridx, gridy], axis=-1)

init_solution = get_init(grid)
init_solution = init_solution.at[0,:,:].set(jnp.array([1.0, 0.0]))
init_solution = init_solution.at[-1,:,:].set(init_solution[-2,:,:])
init_solution = init_solution.at[:,0,:].set(jnp.array([1.0, 0.0]))
init_solution = init_solution.at[:,-1,:].set(jnp.array([1.0, 0.0]))

init_vorticity = get_vorticity(init_solution, delta_x, delta_y)

solver = burgers_solver(delta_x, delta_y, delta_t, mu)
solution = solver.solve(init_solution, num_steps, show_progress=True, bdry_type='neumann')
vorticity = get_vorticity(solution, delta_x, delta_y)

u_norm = Normalize(vmin=init_solution[...,0].min(), vmax=init_solution[...,0].max())
v_norm = Normalize(vmin=init_solution[...,1].min(), vmax=init_solution[...,1].max())
vort_norm = Normalize(vmin=init_vorticity.min(), vmax=init_vorticity.max())

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), layout='constrained')
im = axes[0].imshow(solution[...,0].T, origin='lower', extent=(0,x_lim,0,y_lim), cmap='plasma')
axes[0].set_title("u")
plt.colorbar(im)

im = axes[1].imshow(solution[...,1].T, origin='lower', extent=(0,x_lim,0,y_lim), cmap='plasma')
axes[1].set_title("v")
plt.colorbar(im)

im = axes[2].imshow(vorticity.T, origin='lower', extent=(0,x_lim,0,y_lim), cmap='plasma')
axes[2].set_title(r"$\nabla \times\mathbf{u}$")
plt.colorbar(im)

plt.savefig("prelim_tests/neumann_solution.png", dpi = 600)
plt.show()