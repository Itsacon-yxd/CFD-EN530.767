import jax.numpy as jnp
import jax
from jax import jit
from functools import partial
import matplotlib.pyplot as plt

class wave_eqn_solver(object):

    def __init__(self, delta_x, delta_t):
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.x = jnp.arange(0, 2*jnp.pi+self.delta_x/2, self.delta_x)
    
    @partial(jit, static_argnums=(0,))
    def update_pixel_1st(self, solution, _):

        res = solution - self.delta_t/self.delta_x * (solution - jnp.roll(solution, shift=1))

        return res, None
    
    @partial(jit, static_argnums=(0,))
    def update_pixel_2nd(self, solution, _):

        res = solution - self.delta_t / (2*self.delta_x) * (3*solution - 4*jnp.roll(solution, shift=1) + jnp.roll(solution, shift=2))

        return res, None
    
    @partial(jit, static_argnums=(0,))
    def update_pixel_3rd(self, solution, _):

        res = solution - self.delta_t / (6*self.delta_x) * (11*solution - 18*jnp.roll(solution, shift=1) + 9*jnp.roll(solution, shift=2) - 2*jnp.roll(solution, shift=3))

        return res, None
    
    @partial(jit, static_argnums=(0,))
    def update_pixel_3rd_biased(self, solution, _):

        res = solution - self.delta_t / (6*self.delta_x) * (2*jnp.roll(solution, shift=-1) + 3*solution - 6*jnp.roll(solution, shift=1) + jnp.roll(solution, shift=2))

        return res, None

    def solve(
            self, 
            init:jnp.array,
            num_steps: int, 
            scheme_num: int = 1
            ):

        solution = init

        if scheme_num == 1:
            solution, _ = jax.lax.scan(self.update_pixel_1st, init=solution, xs=jnp.arange(0, num_steps))
        elif scheme_num == 2:
            solution, _ = jax.lax.scan(self.update_pixel_2nd, init=solution, xs=jnp.arange(0, num_steps))
        elif scheme_num == 3:
            solution, _ = jax.lax.scan(self.update_pixel_3rd, init=solution, xs=jnp.arange(0, num_steps))
        elif scheme_num == 4:
            solution, _ = jax.lax.scan(self.update_pixel_3rd_biased, init=solution, xs=jnp.arange(0, num_steps))

        return solution

def get_exact(f, x, t):
    return f((x - t) % (2*jnp.pi))

if __name__ == "__main__":
    delta_x = 2*jnp.pi/20
    delta_t = 1e-3
    num_steps = 100
    m = 2
    solver = wave_eqn_solver(
        delta_x = delta_x, 
        delta_t = delta_t
        )
    init = jnp.sin(m*solver.x)
    init_jmp = jnp.zeros_like(solver.x)
    for i in range(len(solver.x)):
        if jnp.pi/2 <= solver.x[i] <= jnp.pi:
            init_jmp = init_jmp.at[i].set(1.0)
    solution = solver.solve(init=init_jmp, num_steps=num_steps, scheme_num=1)
    exact = get_exact(lambda x: jnp.where((jnp.pi/2 <= x) & (x <= jnp.pi), 1.0, 0.0), solver.x , num_steps*delta_t)
    plt.plot(solver.x, solution, label="Numerical Solution", c='orange')
    plt.plot(solver.x, exact, label="Exact Solution", c='blue')
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Wave Equation Solution after {} steps".format(num_steps))
    plt.grid()
    plt.legend()
    plt.show()