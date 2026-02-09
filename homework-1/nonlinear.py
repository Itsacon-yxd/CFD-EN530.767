import numpy as np
import matplotlib.pyplot as plt

def solver(init, delta_t, num_steps, func):
    solution = init.copy()  # Make a copy to avoid modifying original
    for i in range(num_steps):
        solution = solution - delta_t * func(solution) * solution
    return solution

# Setup
delta_x = 2 * np.pi / 20
delta_t = 0.1
grid = np.arange(0, 2 * np.pi + delta_x / 2, delta_x)
init = np.sin(grid) + 0.5 * np.sin(4 * grid)

func_ls = [lambda x: x, lambda x, m=3: np.sin(m * x)]
func_names = ['g(x) = x', 'g(x) = sin(3x)']
steps = np.arange(1, 11)

num_funcs = len(func_ls)
num_steps_to_plot = len(steps)


fig, axes = plt.subplots(num_funcs, num_steps_to_plot, figsize=(20, 5))

# Plot results
for row, (func, func_name) in enumerate(zip(func_ls, func_names)):
    for col, step in enumerate(steps):
        ax = axes[row, col]
        
        # Compute solution
        solution = solver(init, delta_t, step, func)
        
        ax.plot(np.abs(np.fft.rfft(solution)), 'r-', linewidth=1.5)
        ax.set_yscale('log')
        ax.axvline(x=1, color='k', linestyle='--', linewidth=1)
        ax.axvline(x=4, color='k', linestyle='--', linewidth=1)
        
        ax.grid(True, alpha=0.3)
        
        # Set column titles (time stamps) on top row
        if row == 0:
            ax.set_title(f't = {step * delta_t:.1f}', fontsize=10)
        
        # Set row labels on leftmost column
        if col == 0:
            ax.set_ylabel(func_name, fontsize=10)
        else:
            ax.set_ylabel('Amplitude', fontsize=10)
        
        # Only show x-axis labels on bottom row
        if row < num_funcs - 1:
            ax.set_xlabel('Frequency', fontsize=8)
            ax.set_xticklabels([])
        

plt.suptitle('Evolution of Numerical Solutions', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('nonlinear.png', dpi=150, bbox_inches='tight')
plt.show()