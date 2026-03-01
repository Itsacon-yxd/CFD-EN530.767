from linear_burgers import linear_solver
import matplotlib.pyplot as plt
import jax.numpy as jnp
from tqdm import tqdm

delta_x = 1/100
delta_t = 1e-3

total_num_steps = 10_000_000
sub_steps = 1000

x_coord = jnp.arange(0, 1+delta_x/2, delta_x)
init = jnp.zeros_like(x_coord).at[-1].set(1)
solver = linear_solver(delta_t = delta_t, delta_x = delta_x)

solution = init
residual = []
for i in tqdm(range(total_num_steps // sub_steps)):
    prev = jnp.copy(solution)
    solution = solver.solve(solution, num_steps=sub_steps, scheme_num=0)
    diff = prev-solution
    residual.append(jnp.sqrt(jnp.mean(diff**2)))

residual = jnp.array(residual)
plt.plot(residual)
plt.xlabel("Time Steps (x1000)")
plt.ylabel(r"Absolute change in $L^2$ norm")
plt.title("Convergence to Steady State")
plt.grid()
plt.savefig("steady_state_check.png", dpi=600)
plt.close()