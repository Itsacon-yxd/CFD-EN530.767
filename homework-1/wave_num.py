import jax.numpy as jnp
import matplotlib.pyplot as plt

def wave_curve(x, scheme_num):
    if scheme_num == 1:
        return jnp.sin(x), jnp.cos(x)-1
    elif scheme_num == 2:
        return 2*jnp.sin(x) - 0.5*jnp.sin(2*x), 2*jnp.cos(x) - 0.5*jnp.cos(2*x) - 1.5
    elif scheme_num == 3:
        return 3*jnp.sin(x) - 1.5*jnp.sin(2*x) + (1/3)*jnp.sin(3*x), 3*jnp.cos(x) - 1.5*jnp.cos(2*x) + (1/3)*jnp.cos(3*x) - 11/6
    elif scheme_num == 4:
        return 4/3 * jnp.sin(x) - 1/6 * jnp.sin(2*x), 2/3 * jnp.cos(x) - 1/6 * jnp.cos(2*x) - 1/2
    raise ValueError("Invalid scheme number. Must be 1, 2, 3, or 4.")

delta_x = jnp.pi/10
m_ls = [2,4,6,8]

scheme_names = [
    "1st Order Upwind",
    "2nd Order Upwind",
    "3rd Order Upwind",
    "3rd Order Upwind Biased"
]
fig, ax = plt.subplots(figsize=(16, 12))
x = jnp.linspace(0, jnp.pi, 100)
colors = ['blue', 'orange', 'green', 'red']
ax.plot(x, x, label='Exact wave curve', c='black')
ax.plot(x, jnp.zeros_like(x), c='black', linestyle='dashed')
for scheme_num in range(1, 5):
    re, im = wave_curve(x, scheme_num)
    ax.plot(x, re, label=scheme_names[scheme_num-1], c=colors[scheme_num-1])
    ax.plot(x, im, c=colors[scheme_num-1], linestyle='dashed')
ax.vlines([m * delta_x for m in m_ls], ymin=-6, ymax=4, colors='magenta', 
          label=r'$k\Delta x$ when $k \in \{2,4,6,8\}$', linestyles='dotted', linewidth=2)
ax.set_xlabel(r"$k\Delta x$", fontsize=30)
ax.text(-0.05, 0.75, r'$\mathrm{Re}(k\'\Delta x)$', 
        transform=ax.transAxes,
        va='center', ha='center', rotation='vertical', fontsize=25)

# Negative half label
ax.text(-0.05, 0.25, r'$\mathrm{Im}(k\'\Delta x)$', 
        transform=ax.transAxes,
        va='center', ha='center', rotation='vertical', fontsize=25)
ax.legend()
ax.grid()
plt.title("Modified wavenumber curves for different schemes", fontsize=30)
plt.savefig("wave_num.png", dpi=300)
plt.close()


fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
axes2 = axes2.flatten()

markers = ['o', 's', '^', 'd']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, m in enumerate(m_ls):
    ax = axes2[idx]
    
    errors = []
    for scheme_num in [1, 2, 3, 4]:
        re, im = wave_curve(m*delta_x, scheme_num)
        
        dissipation = im/delta_x
        errors.append(dissipation)
    # Line plot with markers
    x_pos = [1, 2, 3, 4]
    ax.plot(x_pos, errors, 'o-', markersize=12, linewidth=2.5, 
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
    ax.set_ylabel('Dissipation (log scale)', fontsize=11)
    ax.set_title(f'Wavenumber m = {m}', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['1st Order', '2nd Order', '3rd Order', '3rd Biased'], 
                       fontsize=10, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=10)

plt.suptitle(f'Dissipation vs Scheme (Δx={delta_x:.4f})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dissipation.png', dpi=400, bbox_inches='tight')
plt.show()