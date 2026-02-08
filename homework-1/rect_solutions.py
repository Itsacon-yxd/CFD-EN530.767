import jax
from wave_solver import wave_eqn_solver, get_exact
import jax.numpy as jnp
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

delta_x = 2 * jnp.pi / 20
delta_t = 1e-3
solver = wave_eqn_solver(delta_x=delta_x, delta_t=delta_t)
num_steps = 10000

def init_func(x):
    return jnp.where((x >= jnp.pi/2) & (x <= jnp.pi), 1.0, 0.0)

init = init_func(solver.x)
spectra = jnp.fft.fft(init)
plt.plot(jnp.abs(spectra))  # Plot only the positive frequencies
plt.vlines(10, label='Nyquist Frequency', ymin=0, ymax=jnp.max(jnp.abs(spectra)), colors='r', linestyles='dashed')
plt.title("Spectrum of Rectangular Wave Initial Condition")
plt.xlabel("Wavenumber")
plt.ylabel("Magnitude")
plt.grid()
plt.legend()
plt.savefig('rect_spectrum.png', dpi=400, bbox_inches='tight')
plt.close()

scheme_names = [
    "1st Order Upwind",
    "2nd Order Upwind",
    "3rd Order Upwind",
    "3rd Order Upwind Biased"
]

# Create a figure with subplots: rows = m values, cols = schemes
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
axes = axes.flatten()

# Exact solution at final time
exact = get_exact(
    init_func,  # capture m in lambda
    solver.x,
    num_steps * delta_t
)

for i, ax in enumerate(axes):
    
    # Solve using the scheme
    solution = solver.solve(init=init, num_steps=num_steps, scheme_num=i+1)
    
    # Plot
    ax.plot(solver.x, exact, 'b-', label='Exact', linewidth=2)
    ax.plot(solver.x, solution, 'r--', label='Numerical', linewidth=2)
    
    # Calculate L2 error
    error = jnp.sqrt(jnp.mean((solution - exact) ** 2))
    
    ax.set_title(f'{scheme_names[i]}\n, L2 err={error:.4f}', fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)
    
    
    ax.legend(loc='upper right', fontsize=8)

plt.suptitle(f'Wave Equation: Comparison of Schemes (Δx={delta_x:.4f}, Δt={delta_t}, steps={num_steps})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rect_comparison.png', dpi=400, bbox_inches='tight')
plt.show()