import jax
from wave_solver import wave_eqn_solver, get_exact
import jax.numpy as jnp
import matplotlib.pyplot as plt

delta_x = 2 * jnp.pi / 20
delta_t = 1e-3
solver = wave_eqn_solver(delta_x=delta_x, delta_t=delta_t)
num_steps = 200

m_ls = [2, 4, 6, 8]
scheme_names = [
    "1st Order Upwind",
    "2nd Order Upwind",
    "3rd Order Upwind",
    "3rd Order Upwind Biased"
]

# Create a figure with subplots: rows = m values, cols = schemes
fig, axes = plt.subplots(len(m_ls), 4, figsize=(16, 12))

for i, m in enumerate(m_ls):
    # Initial condition
    init = jnp.sin(m * solver.x)
    
    # Exact solution at final time
    exact = get_exact(
        lambda x, m=m: jnp.sin(m * x),  # capture m in lambda
        solver.x,
        num_steps * delta_t
    )
    
    for j in range(1,5):
        ax = axes[i, j-1]
        
        # Solve using the scheme
        solution = solver.solve(init=init, num_steps=num_steps, scheme_num=j)
        
        # Plot
        ax.plot(solver.x, exact, 'b-', label='Exact', linewidth=2)
        ax.plot(solver.x, solution, 'r--', label='Numerical', linewidth=2)
        
        # Calculate L2 error
        error = jnp.sqrt(jnp.mean((solution - exact) ** 2))
        
        ax.set_title(f'{scheme_names[j-1]}\nm={m}, L2 err={error:.4f}', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        
        
        ax.legend(loc='upper right', fontsize=8)

plt.suptitle(f'Wave Equation: Comparison of Schemes (Δx={delta_x:.4f}, Δt={delta_t}, steps={num_steps})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sine_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Also create an error summary plot
# Error summary plot: 4 subplots (one per m), log scale on y-axis
# Line plot version: 4 subplots (one per m), log scale
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
axes2 = axes2.flatten()

markers = ['o', 's', '^', 'd']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, m in enumerate(m_ls):
    ax = axes2[idx]
    
    init = jnp.sin(m * solver.x)
    exact = get_exact(lambda x, m=m: jnp.sin(m * x), solver.x, num_steps * delta_t)
    
    errors = []
    for scheme_num in [1, 2, 3, 4]:
        solution = solver.solve(init=init, num_steps=num_steps, scheme_num=scheme_num)
        amplitude_ratio = jnp.sqrt(jnp.mean(solution ** 2)) / jnp.sqrt(jnp.mean(exact ** 2))
        errors.append(float(amplitude_ratio))  # Store the error as a float for plotting
    
    # Line plot with markers
    x_pos = [1, 2, 3, 4]
    ax.semilogy(x_pos, errors, 'o-', markersize=12, linewidth=2.5, 
                color=colors[idx], markerfacecolor='white', 
                markeredgecolor=colors[idx], markeredgewidth=2,
                label=f'm={m}')
    
    # Add error value labels for each point
    for i, err in enumerate(errors):
        ax.annotate(f'{err:.2e}', 
                    xy=(x_pos[i], err), 
                    xytext=(8, 8),
                    textcoords='offset points', 
                    fontsize=9,
                    fontweight='bold',
                    color=colors[idx])
    
    ax.set_xlabel('Scheme', fontsize=11)
    ax.set_ylabel('L2 Norm Ratio (log scale)', fontsize=11)
    ax.set_title(f'Wavenumber m = {m}', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['1st Order', '2nd Order', '3rd Order', '3rd Biased'], 
                       fontsize=10, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=10)

plt.suptitle(f'L2 Norm Ratio vs Scheme (Δx={delta_x:.4f}, Δt={delta_t}, steps={num_steps})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sine_ratio_line.png', dpi=150, bbox_inches='tight')
plt.show()