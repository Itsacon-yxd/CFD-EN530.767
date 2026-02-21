from functools import partial

from burgers_solver import burgers_solver, get_init, get_vorticity
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import numpy as np
import jax
import jax.extend
from jax_tqdm import scan_tqdm

plt.rcParams['font.size'] = 16

delta_t = 1e-4
delta_x = 5e-3
delta_y = 5e-3
mu = 1e-3
gridx, gridy = jnp.meshgrid(jnp.arange(0,2+delta_x/2,delta_x), jnp.arange(0,1+delta_y/2,delta_y), indexing='ij')
grid = jnp.stack([gridx, gridy], axis=-1)

bdry_cond = jnp.zeros_like(grid)
bdry_cond = bdry_cond.at[0,:,:].set(jnp.array([1.0, 0.0]))
bdry_cond = bdry_cond.at[-1,:,:].set(jnp.array([1.0, 0.0]))
bdry_cond = bdry_cond.at[:,0,:].set(jnp.array([1.0, 0.0]))
bdry_cond = bdry_cond.at[:,-1,:].set(jnp.array([1.0, 0.0]))

init_solution = bdry_cond.at[1:-1,1:-1].set(get_init(grid[1:-1,1:-1]))
init_vortex = get_vorticity(init_solution, delta_x, delta_y)

fps = 50
frame_num = 101
num_steps = 1 / fps / delta_t

solver = burgers_solver(delta_x, delta_y, delta_t, mu)

def time_stepping(solution, xs, num_steps):
    solution = solver.solve(solution,  num_steps, show_progress=False)
    vortex = get_vorticity(solution, delta_x, delta_y)[:,:,None]
    return solution, jnp.concat([solution, vortex], axis=-1)

time_stepping = jax.jit(
    partial(time_stepping, num_steps=num_steps)
)
time_stepping = scan_tqdm(n=frame_num-1, print_rate=1)(time_stepping)

print(f"JAX backend platform: {jax.extend.backend.get_default_device()}")
print("Num steps per frame:", num_steps)

carry, ys = jax.lax.scan(time_stepping, init_solution, jnp.arange(frame_num-1))
solution = ys[..., :2]
solution = jnp.concat([init_solution[None, ...], solution], axis=0)
vortex = ys[..., 2]
vortex = jnp.concat([init_vortex[None, ...], vortex], axis=0)

# Convert JAX arrays to numpy for matplotlib
solution_np = np.array(solution)
vortex_np = np.array(vortex)

# Option 1: Create a video with both velocity magnitude and vorticity side by side
fig, axes = plt.subplots(3, 1, figsize=(8, 12), layout='constrained')

# Set up normalization for consistent colorbar across frames
u_norm = Normalize(vmin=solution_np[...,0].min(), vmax=solution_np[...,0].max())
v_norm = Normalize(vmin=solution_np[...,1].min(), vmax=solution_np[...,1].max())
vort_norm = Normalize(vmin=vortex_np.min(), vmax=vortex_np.max())

# Initial plots
im1 = axes[0].imshow(solution_np[0,...,0].T, origin='lower', cmap='plasma', norm=u_norm,
                      extent=[0, 2, 0, 1], aspect='equal')
axes[0].set_title('u')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(solution_np[0,...,1].T, origin='lower', cmap='plasma', norm=v_norm,
                      extent=[0, 2, 0, 1], aspect='equal')
axes[1].set_title('v')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(vortex_np[0].T, origin='lower', cmap='plasma', norm=vort_norm,
                      extent=[0, 2, 0, 1], aspect='equal')
axes[2].set_title(r'$\nabla \times \mathbf{u}$')
axes[2].set_xlabel('x')
axes[2].set_ylabel('y')
plt.colorbar(im3, ax=axes[2])

time_text = fig.suptitle(f'Time: {0:.3f} s')

def animate(frame):
    im1.set_array(solution_np[frame,...,0].T)
    im2.set_array(solution_np[frame,...,1].T)
    im3.set_array(vortex_np[frame].T)
    time_text.set_text(f'Time: {frame / fps:.3f} s')
    return im1, im2, im3, time_text

ani = animation.FuncAnimation(fig, animate, frames=frame_num, interval=1000/fps, blit=False)

# Save as MP4 (requires ffmpeg)
ani.save('burgers_evolution.mp4', writer='ffmpeg', fps=fps, dpi=200)

# Or save as GIF (requires pillow)
# ani.save('burgers_evolution.gif', writer='pillow', fps=fps, dpi=100)

plt.close()
print("Video saved!")