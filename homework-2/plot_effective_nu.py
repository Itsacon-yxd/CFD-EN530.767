from linear_burgers import linear_solver, get_exact
import matplotlib.pyplot as plt
import jax.numpy as jnp

delta_x_list = [1/20, 1/50, 1/100]
delta_t = 1e-3

fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained", sharey=True)
for delta_x, ax in zip(delta_x_list, axes.flatten()):
    x_coord = jnp.arange(0, 1+delta_x/2, delta_x)
    init = jnp.zeros_like(x_coord).at[-1].set(1)
    solver = linear_solver(delta_t = delta_t, delta_x = delta_x)

    ax.plot(x_coord, get_exact(x_coord), label="Exact", color="black")
    solution = solver.solve(init, num_steps=10_000_000, scheme_num=1)
    ax.plot(x_coord, solution, label='1st Upwind', color='b', linestyle="--")
    ax.plot(x_coord, get_exact(x_coord, Pec_num = 1 / (1/50 + delta_x/2)), label=r"Exact with $\nu_{eff}$", color="magenta", linestyle=":")
    ax.set_title(r"$\Delta x = $" + f"{delta_x}"+ r' $\nu_{eff} = $' + f"{1/50 + delta_x/2:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)") if ax == axes.flatten()[0] else None
    ax.grid()
    ax.legend()
plt.suptitle(r"Burgers' Equation Solution at $t=10000$, with $\Delta t = 10^{-3}$")
plt.savefig("effective_nu_comparison.png", dpi = 600)
plt.show()